"""MODEL-116 matched tokenizer-vocabulary / transformer-geometry experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.checkpoint import hash_json, sha256_file
from twelve_six.milestone100_first_learned import (
    EXPECTED_CORPUS_ID,
    MIXTURE,
    _build_corpus,
    _rows,
    _state_hash,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.packing import TextRecord, iter_packed_examples
from twelve_six.tokenization.experiments import (
    CorpusFileIdentity,
    HFTokenizerAdapter,
    TokenizerTrainingManifest,
    train_hf_tokenizer,
)
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.vocabulary import rebalance_d_ff_for_vocabulary

SCHEMA = "12-6.model116-tokenizer-geometry.v1"
AUTHORITY = "LOCAL_FREE_EXPERIMENTAL_NO_CANONICAL_FREEZE"
REPOSITORY = "Oleksii-debug/12-6-ai."
TOKENIZERS_VERSION = "0.23.1"
VOCABULARIES = (320, 384, 437)
SCALES = (100_000, 500_000, 1_000_000)
SEEDS = (1337, 7331)
SEQ = 128
BATCH = 4
DEFAULT_OPTIMIZED_TOKENS = 16_384
VALIDATION_SOURCE_BYTES_PER_STRATUM = 4_096
PARAMETER_TOLERANCE = 0.005
LR = 3e-4


class Model116Error(RuntimeError):
    """Fail-closed experiment error."""


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _require_head(repo: Path, source_sha: str) -> None:
    if len(source_sha) != 40 or any(c not in "0123456789abcdef" for c in source_sha):
        raise Model116Error("source_sha must be lowercase full 40-hex")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != source_sha:
        raise Model116Error(f"exact-head mismatch: {actual} != {source_sha}")


def _physical_rows(
    corpus: Path,
    manifest: dict[str, Any],
    *,
    split: str,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for shard in sorted(manifest["shards"], key=lambda item: str(item["path"])):
        path = corpus / str(shard["path"])
        if sha256_file(path) != str(shard["sha256"]):
            raise Model116Error(f"shard hash mismatch: {shard['path']}")
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("split") == split:
                    rows.append(row)
    return tuple(rows)


def _stream_identity(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    by_stratum = {name: {"documents": 0, "utf8_bytes": 0} for name in ("uk", "en", "code")}
    total_bytes = 0
    for row in rows:
        record_id = str(row["record_id"])
        stratum = str(row["stratum"])
        text = str(row["text"])
        raw = text.encode("utf-8")
        digest.update(record_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(stratum.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
        total_bytes += len(raw)
        by_stratum[stratum]["documents"] += 1
        by_stratum[stratum]["utf8_bytes"] += len(raw)
    return {
        "schema": "12-6.model116-tokenizer-training-selection.v1",
        "split": "train",
        "selection_order": "physical_shard_path_lexicographic_then_record_order",
        "documents": len(rows),
        "utf8_bytes": total_bytes,
        "by_stratum": by_stratum,
        "record_stream_sha256": digest.hexdigest(),
    }


def _train_tokenizers(
    *,
    corpus: Path,
    manifest: dict[str, Any],
    output: Path,
) -> tuple[dict[int, HFTokenizerAdapter], dict[str, Any]]:
    rows = _physical_rows(corpus, manifest, split="train")
    if len(rows) != int(manifest["by_split"]["train"]["documents"]):
        raise Model116Error("tokenizer train-document count does not match DATA-25 manifest")
    selection = _stream_identity(rows)
    if selection["utf8_bytes"] != int(manifest["by_split"]["train"]["bytes"]):
        raise Model116Error("tokenizer train-byte count does not match DATA-25 manifest")
    selection["source_corpus_identity_sha256"] = manifest["corpus_identity_sha256"]
    selection_sha = _canonical_sha(selection)
    selection["identity_sha256"] = selection_sha
    _write_json(output / "tokenizer-training-selection.json", selection)

    corpus_files = tuple(
        CorpusFileIdentity(
            path=str(item["path"]),
            sha256=str(item["sha256"]),
            byte_count=int(item["size_bytes"]),
        )
        for item in sorted(manifest["shards"], key=lambda item: str(item["path"]))
    )
    texts = tuple(str(row["text"]) for row in rows)
    adapters: dict[int, HFTokenizerAdapter] = {}
    evidence: dict[str, Any] = {}
    for requested_vocab in VOCABULARIES:
        plan = TokenizerTrainingManifest(
            experiment_id=f"MODEL-116-DATA25-BPE-v{requested_vocab}",
            algorithm="bpe",
            tokenizers_version=TOKENIZERS_VERSION,
            dataset_id="DATA-25-V0.1:split=train",
            dataset_manifest_sha256=selection_sha,
            corpus_files=corpus_files,
            vocab_size=requested_vocab,
            min_frequency=2,
        )
        first = train_hf_tokenizer(plan, texts)
        second = train_hf_tokenizer(plan, texts)
        if first.artifact_identity != second.artifact_identity:
            raise Model116Error(f"BPE-{requested_vocab} tokenizer identity is not repeatable")
        actual = first.vocab_size
        if actual != requested_vocab:
            raise Model116Error(
                f"BPE-{requested_vocab} saturated to {actual}; selected matched set unavailable"
            )
        runtime_json = first._tokenizer.to_str()
        runtime_sha = hashlib.sha256(runtime_json.encode("utf-8")).hexdigest()
        if runtime_sha != first.artifact_identity.tokenizer_json_sha256:
            raise Model116Error("tokenizer runtime JSON hash mismatch")
        tok_path = output / "tokenizers" / f"bpe-{requested_vocab}.tokenizer.json"
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tok_path.write_text(runtime_json, encoding="utf-8")
        _write_json(
            output / "tokenizers" / f"bpe-{requested_vocab}.training-manifest.json",
            plan.to_dict(),
        )
        adapters[requested_vocab] = first
        evidence[str(requested_vocab)] = {
            "requested_vocab_size": requested_vocab,
            "actual_vocab_size": actual,
            "version": first.identity.version,
            "config_sha256": first.identity.config_sha256,
            "vocab_sha256": first.identity.vocab_sha256,
            "training_manifest_sha256": plan.sha256,
            "tokenizer_json_sha256": runtime_sha,
            "repeatability": "PASS",
            "strict_round_trip_training_stream": True,
            "canonical": False,
        }
    return adapters, {
        "training_selection": selection,
        "candidates": evidence,
        "runtime": {"library": "tokenizers", "version": TOKENIZERS_VERSION},
        "source_incumbent": "TOK-37 exact repeatable HF ByteLevel-BPE harness",
    }


def _template_specs(repo: Path) -> dict[int, ModelSpec]:
    s1 = load_stage_config(repo / "configs/stages/s1_100k.json").model
    s2 = load_stage_config(repo / "configs/stages/s2_1m.json").model
    mid = ModelSpec(
        schema_version=1,
        vocab_size=320,
        max_seq_len=256,
        d_model=96,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        head_dim=24,
        d_ff=256,
        rope_rotary_dim=24,
    )
    return {
        100_000: replace(s1, vocab_size=320),
        500_000: mid,
        1_000_000: replace(s2, vocab_size=320),
    }


def _solve_under_cap(spec: ModelSpec, *, cap: int, vocab_size: int) -> ModelSpec:
    solved = rebalance_d_ff_for_vocabulary(
        spec,
        target_parameters=cap,
        vocab_size=vocab_size,
        d_ff_alignment=1,
    ).model
    if solved.parameter_count() > cap:
        if solved.d_ff <= 1:
            raise Model116Error("cannot lower d_ff while respecting parameter cap")
        solved = replace(solved, d_ff=solved.d_ff - 1)
    if solved.parameter_count() > cap:
        raise Model116Error("parameter solver failed monotone no-extra-capacity constraint")
    return solved


def solve_geometries(repo: Path) -> dict[int, dict[int, ModelSpec]]:
    """Solve fixed-budget geometries while never giving a larger vocabulary more parameters."""
    result: dict[int, dict[int, ModelSpec]] = {}
    for target, template in _template_specs(repo).items():
        cap = target
        scale_rows: dict[int, ModelSpec] = {}
        previous_total: int | None = None
        for vocab in VOCABULARIES:
            spec = _solve_under_cap(template, cap=cap, vocab_size=vocab)
            total = spec.parameter_count()
            if previous_total is not None and total > previous_total:
                raise Model116Error("larger vocabulary received extra total parameter capacity")
            relative_delta = (target - total) / target
            if not 0.0 <= relative_delta <= PARAMETER_TOLERANCE:
                raise Model116Error(
                    f"{target=} {vocab=} misses strict tolerance: {relative_delta:.6%}"
                )
            scale_rows[vocab] = spec
            cap = total
            previous_total = total
        result[target] = scale_rows
    return result


def _geometry_evidence(spec: ModelSpec, target: int) -> dict[str, Any]:
    breakdown = spec.parameter_breakdown()
    total = breakdown["total"]
    attention_total = spec.n_layers * breakdown["attention_per_layer"]
    mlp_total = spec.n_layers * breakdown["mlp_per_layer"]
    return {
        "target_parameters": target,
        "model_spec": spec.to_dict(),
        "model_identity_sha256": spec.identity_sha256(),
        "parameter_count": total,
        "target_delta": total - target,
        "target_relative_delta": (total - target) / target,
        "d_ff": spec.d_ff,
        "embedding_parameters": breakdown["token_embedding"],
        "embedding_fraction": breakdown["token_embedding"] / total,
        "blocks_parameters": breakdown["blocks_total"],
        "block_fraction": breakdown["blocks_total"] / total,
        "attention_projection_parameters": attention_total,
        "attention_projection_fraction": attention_total / total,
        "mlp_parameters": mlp_total,
        "mlp_fraction": mlp_total / total,
    }


def _mixed_train_records(rows: Sequence[dict[str, Any]]) -> Iterator[TextRecord]:
    groups: dict[str, list[dict[str, Any]]] = {"uk": [], "en": [], "code": []}
    for row in rows:
        groups[str(row["stratum"])].append(row)
    positions = {name: 0 for name in groups}
    while True:
        emitted = False
        for stratum in MIXTURE:
            pos = positions[stratum]
            group = groups[stratum]
            if pos >= len(group):
                continue
            row = group[pos]
            positions[stratum] = pos + 1
            emitted = True
            yield TextRecord(str(row["record_id"]), str(row["text"]), "train")
        if not emitted:
            return


def _mask_labels_to_remaining(labels: torch.Tensor, remaining: int) -> torch.Tensor:
    if remaining <= 0:
        raise ValueError("remaining must be positive")
    masked = labels.clone()
    shifted = masked[:, 1:]
    valid_positions = shifted.ne(-100).nonzero(as_tuple=False)
    if valid_positions.shape[0] <= remaining:
        return masked
    for batch_index, shifted_index in valid_positions[remaining:].tolist():
        masked[batch_index, shifted_index + 1] = -100
    return masked


def _next_batch(
    iterator: Iterator[Any],
    *,
    remaining: int,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    examples = []
    for _ in range(BATCH):
        try:
            examples.append(next(iterator))
        except StopIteration as exc:
            raise Model116Error("training corpus exhausted before optimized-token budget") from exc
    labels = torch.tensor([item.labels for item in examples], dtype=torch.long)
    labels = _mask_labels_to_remaining(labels, remaining)
    batch = {
        "input_ids": torch.tensor([item.input_ids for item in examples], dtype=torch.long),
        "labels": labels,
    }
    record_ids = tuple(dict.fromkeys(r for item in examples for r in item.record_ids))
    return batch, record_ids


def _trainer_config(seed: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=LR,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=10_000,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _validation_rows(
    corpus: Path,
    manifest: dict[str, Any],
    *,
    source_byte_floor: int,
) -> dict[str, tuple[dict[str, Any], ...]]:
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for stratum in ("uk", "en", "code"):
        selected: list[dict[str, Any]] = []
        source_bytes = 0
        for row in _rows(corpus, manifest, "validation", stratum):
            selected.append(row)
            source_bytes += len(str(row["text"]).encode("utf-8"))
            if source_bytes >= source_byte_floor:
                break
        if source_bytes < source_byte_floor:
            raise Model116Error(f"insufficient validation bytes for {stratum}")
        result[stratum] = tuple(selected)
    return result


@torch.no_grad()
def _document_nll(
    model: TwelveSixDecoder,
    tokenizer: HFTokenizerAdapter,
    text: str,
) -> tuple[float, int, int]:
    ids = tokenizer.encode(text)
    if tokenizer.decode(ids, skip_special_tokens=False) != text:
        raise Model116Error("held-out tokenizer round trip failed")
    if len(ids) < 2:
        return 0.0, 0, 0
    total_nll = 0.0
    target_tokens = 0
    for start in range(0, len(ids) - 1, SEQ - 1):
        window = ids[start : start + SEQ]
        if len(window) < 2:
            break
        input_ids = torch.tensor([window], dtype=torch.long)
        logits = model(input_ids).logits[:, :-1, :].contiguous()
        targets = torch.tensor([window[1:]], dtype=torch.long)
        nll = F.cross_entropy(
            logits.reshape(-1, model.spec.vocab_size),
            targets.reshape(-1),
            reduction="sum",
        )
        total_nll += float(nll.item())
        target_tokens += len(window) - 1
    modeled_suffix = tokenizer.decode(ids[1:], skip_special_tokens=False)
    modeled_bytes = len(modeled_suffix.encode("utf-8"))
    if modeled_bytes <= 0:
        raise Model116Error("held-out document has no modeled source bytes")
    return total_nll, target_tokens, modeled_bytes


@torch.no_grad()
def _evaluate(
    model: TwelveSixDecoder,
    tokenizer: HFTokenizerAdapter,
    rows: dict[str, tuple[dict[str, Any], ...]],
) -> dict[str, Any]:
    before = _state_hash(model)
    was_training = model.training
    model.eval()
    by_stratum: dict[str, Any] = {}
    total_nll = 0.0
    total_bytes = 0
    total_tokens = 0
    try:
        for stratum in ("uk", "en", "code"):
            nll_sum = 0.0
            modeled_bytes = 0
            target_tokens = 0
            source_bytes = 0
            for row in rows[stratum]:
                text = str(row["text"])
                nll, tokens, suffix_bytes = _document_nll(model, tokenizer, text)
                nll_sum += nll
                target_tokens += tokens
                modeled_bytes += suffix_bytes
                source_bytes += len(text.encode("utf-8"))
            bpb = nll_sum / math.log(2.0) / modeled_bytes
            by_stratum[stratum] = {
                "bits_per_byte": bpb,
                "nll_sum": nll_sum,
                "modeled_utf8_bytes": modeled_bytes,
                "source_utf8_bytes": source_bytes,
                "target_tokens": target_tokens,
                "documents": len(rows[stratum]),
            }
            total_nll += nll_sum
            total_bytes += modeled_bytes
            total_tokens += target_tokens
    finally:
        model.train(was_training)
    after = _state_hash(model)
    if before != after:
        raise Model116Error("evaluation mutated model state")
    macro_bpb = sum(by_stratum[s]["bits_per_byte"] for s in ("uk", "en", "code")) / 3.0
    return {
        "aggregate_bits_per_byte": total_nll / math.log(2.0) / total_bytes,
        "macro_bits_per_byte": macro_bpb,
        "nll_sum": total_nll,
        "modeled_utf8_bytes": total_bytes,
        "target_tokens": total_tokens,
        "by_stratum": by_stratum,
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "non_mutation_passed": True,
        "bpb_denominator": "UTF-8 bytes decoded by all scored target tokens; first token per document is unscored and excluded",
    }


def _run_one(
    *,
    spec: ModelSpec,
    tokenizer: HFTokenizerAdapter,
    train_rows: Sequence[dict[str, Any]],
    validation_rows: dict[str, tuple[dict[str, Any], ...]],
    optimized_tokens: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, InitSpec())
    initial_state = _state_hash(model)
    initial_eval = _evaluate(model, tokenizer, validation_rows)
    config = _trainer_config(seed)
    trainer = Trainer(model, config, device="cpu")
    iterator = iter_packed_examples(
        _mixed_train_records(train_rows),
        tokenizer,
        expected_split="train",
        sequence_length=SEQ,
        cross_document=False,
    )
    record_bytes = {
        str(row["record_id"]): len(str(row["text"]).encode("utf-8")) for row in train_rows
    }
    source_ids_seen: set[str] = set()
    update_losses: list[float] = []
    start = time.perf_counter()
    while trainer.tokens_seen < optimized_tokens:
        remaining = optimized_tokens - trainer.tokens_seen
        batch, record_ids = _next_batch(iterator, remaining=remaining)
        metrics = trainer.train_microbatch(batch)
        source_ids_seen.update(record_ids)
        if metrics.update_loss is not None:
            update_losses.append(metrics.update_loss)
    elapsed = time.perf_counter() - start
    if trainer.tokens_seen != optimized_tokens:
        raise Model116Error("optimized-token budget was not exact")
    if not update_losses:
        raise Model116Error("no optimizer update was executed")
    final_eval = _evaluate(model, tokenizer, validation_rows)
    final_state = _state_hash(model)
    first_n = update_losses[: min(5, len(update_losses))]
    last_n = update_losses[-min(5, len(update_losses)) :]
    initial_macro = float(initial_eval["macro_bits_per_byte"])
    final_macro = float(final_eval["macro_bits_per_byte"])
    return {
        "seed": seed,
        "random_initialization": {
            "pretrained_weights_loaded": False,
            "init_spec": InitSpec().to_dict(),
            "initial_model_state_sha256": initial_state,
        },
        "optimized_tokens": trainer.tokens_seen,
        "optimizer_steps": trainer.optimizer_step,
        "training_loss": {
            "first_update": update_losses[0],
            "last_update": update_losses[-1],
            "first5_mean": sum(first_n) / len(first_n),
            "last5_mean": sum(last_n) / len(last_n),
            "decreased_first5_to_last5": (sum(last_n) / len(last_n))
            < (sum(first_n) / len(first_n)),
        },
        "throughput": {
            "elapsed_seconds": elapsed,
            "optimized_tokens_per_second": optimized_tokens / elapsed,
            "unique_source_documents_touched": len(source_ids_seen),
            "unique_source_utf8_bytes_touched": sum(record_bytes[r] for r in source_ids_seen),
        },
        "evaluation": {
            "initial": initial_eval,
            "final": final_eval,
            "macro_bpb_improvement": initial_macro - final_macro,
            "macro_bpb_relative_improvement": (
                (initial_macro - final_macro) / initial_macro if initial_macro else 0.0
            ),
            "validation_improvement_per_million_parameters": (
                (initial_macro - final_macro) * 1_000_000 / spec.parameter_count()
            ),
        },
        "final_model_state_sha256": final_state,
    }


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(int(row["scale_target"]), {}).setdefault(
            int(row["vocab_size"]), []
        ).append(row)
    scales: dict[str, Any] = {}
    for scale, by_vocab in sorted(grouped.items()):
        candidates: list[dict[str, Any]] = []
        seed_winners: dict[int, tuple[float, int]] = {}
        for vocab, members in sorted(by_vocab.items()):
            final_macros = [m["run"]["evaluation"]["final"]["macro_bits_per_byte"] for m in members]
            initial_macros = [m["run"]["evaluation"]["initial"]["macro_bits_per_byte"] for m in members]
            throughput = [m["run"]["throughput"]["optimized_tokens_per_second"] for m in members]
            candidate = {
                "vocab_size": vocab,
                "parameter_count": members[0]["geometry"]["parameter_count"],
                "d_ff": members[0]["geometry"]["d_ff"],
                "embedding_fraction": members[0]["geometry"]["embedding_fraction"],
                "block_fraction": members[0]["geometry"]["block_fraction"],
                "attention_projection_fraction": members[0]["geometry"]["attention_projection_fraction"],
                "mean_initial_macro_bpb": sum(initial_macros) / len(initial_macros),
                "mean_final_macro_bpb": sum(final_macros) / len(final_macros),
                "mean_macro_bpb_improvement": sum(i - f for i, f in zip(initial_macros, final_macros)) / len(members),
                "mean_optimized_tokens_per_second": sum(throughput) / len(throughput),
                "mean_validation_improvement_per_million_parameters": sum(
                    m["run"]["evaluation"]["validation_improvement_per_million_parameters"] for m in members
                ) / len(members),
            }
            candidates.append(candidate)
            for member in members:
                seed = int(member["run"]["seed"])
                value = float(member["run"]["evaluation"]["final"]["macro_bits_per_byte"])
                if seed not in seed_winners or value < seed_winners[seed][0]:
                    seed_winners[seed] = (value, vocab)
        ranked = sorted(candidates, key=lambda item: item["mean_final_macro_bpb"])
        preferred = int(ranked[0]["vocab_size"])
        runner_gap = ranked[1]["mean_final_macro_bpb"] - ranked[0]["mean_final_macro_bpb"] if len(ranked) > 1 else 0.0
        unanimous = bool(seed_winners) and all(v == preferred for _, v in seed_winners.values())
        adjacent: list[dict[str, Any]] = []
        ordered = sorted(candidates, key=lambda item: item["vocab_size"])
        for small, large in zip(ordered, ordered[1:]):
            delta = large["mean_final_macro_bpb"] - small["mean_final_macro_bpb"]
            adjacent.append(
                {
                    "small_vocab": small["vocab_size"],
                    "large_vocab": large["vocab_size"],
                    "parameter_delta_large_minus_small": large["parameter_count"] - small["parameter_count"],
                    "embedding_fraction_delta": large["embedding_fraction"] - small["embedding_fraction"],
                    "d_ff_delta": large["d_ff"] - small["d_ff"],
                    "mean_final_macro_bpb_delta_large_minus_small": delta,
                    "larger_tokenizer_justified_at_this_budget": bool(
                        delta < -0.005 and large["parameter_count"] <= small["parameter_count"]
                    ),
                }
            )
        scales[str(scale)] = {
            "ranked_candidates": ranked,
            "seed_winners": {str(k): v for k, (_, v) in sorted(seed_winners.items())},
            "observed_preferred_vocab": preferred,
            "runner_up_mean_macro_bpb_gap": runner_gap,
            "seed_unanimity": unanimous,
            "scale_dependent_recommendation": (
                f"PROVISIONAL_PREFERRED_VOCAB_{preferred}"
                if unanimous and runner_gap > 0.005
                else "NO_ROBUST_FREEZE_RETEST"
            ),
            "canonical_freeze": False,
            "adjacent_tokenizer_tax_tests": adjacent,
        }
    return {
        "scales": scales,
        "global_freeze": False,
        "reason": (
            "Matched LOCAL_FREE evidence is intentionally exploratory: two seeds, bounded "
            "optimized-token budget, and DATA-25 has zero external training-eligible sources."
        ),
    }


def _machine_manifest(source_sha: str) -> dict[str, Any]:
    return {
        "source_sha": source_sha,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "cuda_available": torch.cuda.is_available(),
        "training_device": "cpu",
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "local_free_only": True,
        "paid_compute": False,
    }


def run(*, repo: Path, source_sha: str, output: Path, optimized_tokens: int) -> dict[str, Any]:
    if optimized_tokens <= 0:
        raise Model116Error("optimized_tokens must be positive")
    _require_head(repo, source_sha)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    manifest = _build_corpus(repo, output)
    if manifest["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise Model116Error("DATA-25 identity drift")
    corpus = output / "corpus-a"
    train_rows = _physical_rows(corpus, manifest, split="train")
    validation_rows = _validation_rows(
        corpus,
        manifest,
        source_byte_floor=VALIDATION_SOURCE_BYTES_PER_STRATUM,
    )
    tokenizers, tokenizer_evidence = _train_tokenizers(corpus=corpus, manifest=manifest, output=output)
    geometries = solve_geometries(repo)
    matrix_rows: list[dict[str, Any]] = []
    for scale in SCALES:
        for vocab in VOCABULARIES:
            spec = geometries[scale][vocab]
            geometry = _geometry_evidence(spec, scale)
            for seed in SEEDS:
                result = _run_one(
                    spec=spec,
                    tokenizer=tokenizers[vocab],
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                    optimized_tokens=optimized_tokens,
                    seed=seed,
                )
                row = {"scale_target": scale, "vocab_size": vocab, "geometry": geometry, "run": result}
                matrix_rows.append(row)
                _write_json(output / "rows" / f"scale-{scale}-vocab-{vocab}-seed-{seed}.json", row)
    summary = _summarize(matrix_rows)
    machine = _machine_manifest(source_sha)
    _write_json(output / "machine-manifest.json", machine)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": REPOSITORY,
            "git_sha": source_sha,
            "selective_parentage": {
                "milestone100": "b9bc147e0a08181b91798c2515cac7a79c66791c",
                "tok37": "25cf4798202c41dda4b5413052f5efc6ebbbbf2a",
                "model37": "f1a23e15d7d7c8d0d41221a5053f690c0b23d1bc",
            },
        },
        "corpus": {
            "identity_sha256": manifest["corpus_identity_sha256"],
            "train": manifest["by_split"]["train"],
            "validation": manifest["by_split"]["validation"],
            "truth_boundary": manifest["truth_boundary"],
            "same_corpus_all_rows": True,
        },
        "tokenizers": tokenizer_evidence,
        "experiment_control": {
            "vocabularies": list(VOCABULARIES),
            "scale_targets": list(SCALES),
            "seeds": list(SEEDS),
            "optimized_tokens_per_row": optimized_tokens,
            "optimizer": asdict(_trainer_config(SEEDS[0])),
            "sequence_length": SEQ,
            "batch_size": BATCH,
            "training_record_schedule": "DATA25 train rows per stratum consumed by incumbent MILESTONE-100 MIXTURE pattern",
            "heldout_source_byte_floor_per_stratum": VALIDATION_SOURCE_BYTES_PER_STRATUM,
            "parameter_tolerance": PARAMETER_TOLERANCE,
            "larger_vocab_extra_capacity_allowed": False,
            "foreign_pretrained_weights": False,
            "instruction_tuning": False,
            "paid_compute": False,
        },
        "matrix": matrix_rows,
        "recommendation": summary,
        "machine_manifest": machine,
    }
    report["report_sha256"] = hash_json(report)
    _write_json(output / "report.json", report)
    return report


def validate(path: Path, *, expected_source_sha: str | None = None) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA:
        raise Model116Error("report schema mismatch")
    if expected_source_sha and report["source"]["git_sha"] != expected_source_sha:
        raise Model116Error("report source SHA mismatch")
    stored = report.pop("report_sha256")
    actual = hash_json(report)
    report["report_sha256"] = stored
    if stored != actual:
        raise Model116Error("report hash mismatch")
    if report["corpus"]["identity_sha256"] != EXPECTED_CORPUS_ID:
        raise Model116Error("report corpus identity mismatch")
    if len(report["matrix"]) != len(SCALES) * len(VOCABULARIES) * len(SEEDS):
        raise Model116Error("matrix row count mismatch")
    for scale in SCALES:
        members = [r for r in report["matrix"] if int(r["scale_target"]) == scale]
        by_vocab: dict[int, int] = {}
        for row in members:
            vocab = int(row["vocab_size"])
            total = int(row["geometry"]["parameter_count"])
            by_vocab.setdefault(vocab, total)
            if by_vocab[vocab] != total:
                raise Model116Error("geometry changed across seed family")
            if abs(total - scale) / scale > PARAMETER_TOLERANCE:
                raise Model116Error("parameter tolerance violated")
            if row["run"]["optimized_tokens"] != report["experiment_control"]["optimized_tokens_per_row"]:
                raise Model116Error("optimized-token budget mismatch")
            if not row["run"]["evaluation"]["initial"]["non_mutation_passed"]:
                raise Model116Error("initial evaluation mutation gate failed")
            if not row["run"]["evaluation"]["final"]["non_mutation_passed"]:
                raise Model116Error("final evaluation mutation gate failed")
        totals = [by_vocab[v] for v in VOCABULARIES]
        if totals != sorted(totals, reverse=True):
            raise Model116Error("larger vocabulary received extra parameter capacity")
    if report["recommendation"]["global_freeze"]:
        raise Model116Error("MODEL-116 is not authorized to canonical-freeze a tokenizer")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repo-root", type=Path, default=Path("."))
    run_parser.add_argument("--source-sha", required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--optimized-tokens", type=int, default=DEFAULT_OPTIMIZED_TOKENS)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--expected-source-sha")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        report = run(
            repo=args.repo_root.resolve(),
            source_sha=args.source_sha,
            output=args.output_dir.resolve(),
            optimized_tokens=args.optimized_tokens,
        )
        print(json.dumps({"report_sha256": report["report_sha256"], "recommendation": report["recommendation"]}, ensure_ascii=False, sort_keys=True))
        return 0
    validate(args.report, expected_source_sha=args.expected_source_sha)
    print("MODEL-116 report validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
