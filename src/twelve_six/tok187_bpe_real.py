"""TOK-187 real-corpus HF ByteLevel-BPE selection experiment.

This worker composes the incumbent tokenizer implementation, DATA-183 corpus
candidate, RESEARCH-140 small-repeat policy, and matched RESEARCH-41 geometry.
It is deliberately selection-validation only and cannot freeze a tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.checkpoint import hash_json, sha256_file
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.packing import TextRecord, iter_packed_examples
from twelve_six.research_decision import (
    Decision,
    DecisionConfig,
    MetricDirection,
    MetricPurpose,
    Pair,
    analyze_paired_runs,
)
from twelve_six.scaling_experiment import controlled_specs
from twelve_six.tokenization.experiments import (
    CorpusFileIdentity,
    HFTokenizerAdapter,
    TokenizerTrainingManifest,
    train_hf_tokenizer,
)
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.vocabulary import rebalance_d_ff_for_vocabulary

SCHEMA = "12-6.tok187-bpe-real-selection.v1"
AUTHORITY = "LOCAL_FREE_SELECTION_VALIDATION_NOT_TOKENIZER_FREEZE"
TOKENIZERS_VERSION = "0.23.1"
REQUESTED_GRID = (320, 384, 437, 512)
MODEL_SEEDS = (1337, 7331, 18701)
SEQ = 128
BATCH = 4
OPTIMIZED_TOKENS = 16_384
LR = 3e-4
PARAMETER_TOLERANCE = 0.005
THROUGHPUT_REPEATS = 5
BPB_MATERIALITY = 0.01
PURPOSE_PROFILE = Path("requirements/profiles/linux-x86_64-tokenizer-experiment/profile.json")
EXPECTED_PURPOSE_PROFILE_ID = "linux-x86_64-tokenizer-experiment"
EXPECTED_PURPOSE_PROFILE_SHA256 = "e368fa4c9fb2fc924482de32d5057837959111e958649663813cb46dddf6b5e4"
EXPECTED_TOKENIZER_OVERLAY_SHA256 = "11f27613ee7c15585796af39accde71b1e7c2791c24ff98d74c395262ee68544"
REQUIRED_BOOTSTRAP_CAPABILITIES = ("runtime", "tokenizer", "tests", "lint")
SCALE_LABELS = ("500K", "1M")
TARGET_COUNTS = (467_808, 1_037_696)


class Tok187Error(RuntimeError):
    """Fail-closed TOK-187 evidence error."""


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
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, encoding="utf-8"
    ).strip()
    if actual != source_sha:
        raise Tok187Error(f"exact-head mismatch: {actual} != {source_sha}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Tok187Error(f"{path} must contain a JSON object")
    return value


def _validate_environment(repo: Path, environment_manifest: Path) -> dict[str, Any]:
    if platform.python_version() != "3.11.16":
        raise Tok187Error("TOK-187 requires CPython 3.11.16")
    actual_tokenizers = importlib.metadata.version("tokenizers")
    if actual_tokenizers != TOKENIZERS_VERSION:
        raise Tok187Error(
            f"tokenizers runtime {actual_tokenizers} != required {TOKENIZERS_VERSION}"
        )

    profile = _read_json(repo / PURPOSE_PROFILE)
    if profile.get("profile_id") != EXPECTED_PURPOSE_PROFILE_ID:
        raise Tok187Error("tokenizer purpose profile id drift")
    if profile.get("profile_sha256") != EXPECTED_PURPOSE_PROFILE_SHA256:
        raise Tok187Error("tokenizer purpose semantic identity drift")
    if profile.get("direct_requirements") != [f"tokenizers=={TOKENIZERS_VERSION}"]:
        raise Tok187Error("tokenizer purpose direct requirement drift")
    overlay_path = repo / "requirements/profiles/linux-x86_64-tokenizer-experiment/overlay.lock.txt"
    overlay_sha = sha256_file(overlay_path)
    if overlay_sha != EXPECTED_TOKENIZER_OVERLAY_SHA256:
        raise Tok187Error("tokenizer purpose overlay SHA-256 drift")

    manifest = _read_json(environment_manifest)
    if manifest.get("schema") != "12-6.execution-environment-manifest.v1":
        raise Tok187Error("universal bootstrap manifest schema mismatch")
    plan = manifest.get("plan")
    if not isinstance(plan, dict):
        raise Tok187Error("universal bootstrap plan missing")
    capabilities = tuple(plan.get("capabilities", ()))
    if capabilities != REQUIRED_BOOTSTRAP_CAPABILITIES:
        raise Tok187Error(
            f"bootstrap capabilities drift: {capabilities!r} != {REQUIRED_BOOTSTRAP_CAPABILITIES!r}"
        )
    roles = {str(item["role"]): item for item in plan.get("locks", [])}
    if "tokenizer_overlay" not in roles or "dev" not in roles or "cpu_runtime" not in roles:
        raise Tok187Error("bootstrap omitted tokenizer/dev/cpu lock roles")
    if roles["tokenizer_overlay"].get("sha256") != EXPECTED_TOKENIZER_OVERLAY_SHA256:
        raise Tok187Error("bootstrap tokenizer overlay identity drift")
    if manifest.get("preflight", {}).get("status") != "PASS":
        raise Tok187Error("universal bootstrap preflight did not pass")
    return {
        "bootstrap_manifest_identity_sha256": manifest["identity_sha256"],
        "bootstrap_plan_identity_sha256": plan["identity_sha256"],
        "capabilities": list(capabilities),
        "lock_roles": {
            role: {"path": row["path"], "sha256": row["sha256"]}
            for role, row in sorted(roles.items())
        },
        "purpose_profile_id": profile["profile_id"],
        "purpose_profile_semantic_sha256": profile["profile_sha256"],
        "purpose_profile_file_sha256": sha256_file(repo / PURPOSE_PROFILE),
        "tokenizer_overlay_sha256": overlay_sha,
        "python": platform.python_version(),
        "tokenizers": actual_tokenizers,
    }


def _validate_data183(
    report_path: Path,
    manifest_path: Path,
    source_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _read_json(report_path)
    expected = str(report.get("report_sha256", ""))
    core = dict(report)
    core.pop("report_sha256", None)
    if hash_json(core) != expected:
        raise Tok187Error("DATA-183 report self-hash mismatch")
    if report.get("schema") != "12-6.corpus-v0.2-real-candidate.v1":
        raise Tok187Error("wrong DATA-183 report schema")
    if report.get("source_sha") != source_sha:
        raise Tok187Error("DATA-183 report is not bound to current exact head")
    truth = report.get("truth_boundary", {})
    if truth.get("external_real_ua_present") is not True:
        raise Tok187Error("genuine external UA stratum is absent")
    if truth.get("external_real_en_present") is not True:
        raise Tok187Error("genuine external EN stratum is absent")
    if truth.get("local_free_only") is not True or report.get("authority") != (
        "LOCAL_FREE_CANDIDATE_NOT_CORPUS_FREEZE_OR_REPRESENTATIVENESS_PROMOTION"
    ):
        raise Tok187Error("DATA-183 LOCAL_FREE authority weakened")
    manifest = _read_json(manifest_path)
    manifest_identity = manifest.get("corpus_identity_sha256")
    if manifest_identity is not None and manifest_identity != report["corpus_identity_sha256"]:
        raise Tok187Error("DATA-183 physical manifest corpus identity mismatch")
    return report, manifest


def _physical_rows(
    build_root: Path,
    manifest: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[CorpusFileIdentity, ...]]:
    physical = manifest.get("physical")
    if not isinstance(physical, Mapping):
        raise Tok187Error("DATA-183 build manifest has no physical section")
    shards = physical.get("shards")
    if not isinstance(shards, list) or not shards:
        raise Tok187Error("DATA-183 physical shard list is empty")

    rows: list[dict[str, Any]] = []
    corpus_files: list[CorpusFileIdentity] = []
    for shard in sorted(shards, key=lambda item: str(item["path"])):
        rel = str(shard["path"])
        path = build_root / rel
        expected_sha = str(shard["sha256"])
        if sha256_file(path) != expected_sha:
            raise Tok187Error(f"physical shard hash mismatch: {rel}")
        corpus_files.append(
            CorpusFileIdentity(
                path=Path(rel).as_posix(),
                sha256=expected_sha,
                byte_count=path.stat().st_size,
            )
        )
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise Tok187Error(f"non-object row in {rel}")
            rows.append(row)
    if not rows:
        raise Tok187Error("physical corpus contains no rows")
    ids = [str(row["record_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise Tok187Error("duplicate record ids in physical corpus")
    return tuple(rows), tuple(corpus_files)


def _selection_manifest(
    rows: Sequence[dict[str, Any]],
    *,
    purpose: str,
    corpus_identity_sha256: str,
) -> dict[str, Any]:
    records = []
    by_stratum: dict[str, dict[str, int]] = defaultdict(
        lambda: {"documents": 0, "utf8_bytes": 0}
    )
    for row in rows:
        text = str(row["text"])
        raw = text.encode("utf-8")
        stratum = str(row["stratum"])
        by_stratum[stratum]["documents"] += 1
        by_stratum[stratum]["utf8_bytes"] += len(raw)
        records.append(
            {
                "record_id": str(row["record_id"]),
                "split": str(row["split"]),
                "stratum": stratum,
                "source_id": str(row.get("source_id", "")),
                "origin": str(row.get("origin", "")),
                "utf8_bytes": len(raw),
                "text_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    core: dict[str, Any] = {
        "schema": "12-6.tok187-record-selection.v1",
        "purpose": purpose,
        "source_corpus_identity_sha256": corpus_identity_sha256,
        "record_order": "physical_shard_path_lexicographic_then_record_order",
        "records": records,
        "documents": len(records),
        "utf8_bytes": sum(item["utf8_bytes"] for item in records),
        "by_stratum": {key: dict(value) for key, value in sorted(by_stratum.items())},
    }
    core["identity_sha256"] = _canonical_sha(core)
    return core


def _partition_rows(
    rows: Sequence[dict[str, Any]],
    corpus_identity_sha256: str,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any], dict[str, Any]]:
    train = tuple(row for row in rows if str(row.get("split")) == "train")
    validation = tuple(row for row in rows if str(row.get("split")) == "validation")
    if not train or not validation:
        raise Tok187Error("train or selection-validation split is empty")
    for name, selected in (("train", train), ("selection_validation", validation)):
        strata = {str(row["stratum"]) for row in selected}
        if not {"uk", "en", "code"}.issubset(strata):
            raise Tok187Error(f"{name} does not contain UA/EN/code: {sorted(strata)}")
    train_ids = {str(row["record_id"]) for row in train}
    validation_ids = {str(row["record_id"]) for row in validation}
    if train_ids & validation_ids:
        raise Tok187Error("record identity overlap between train and selection-validation")
    train_manifest = _selection_manifest(
        train, purpose="tokenizer_and_model_training", corpus_identity_sha256=corpus_identity_sha256
    )
    validation_manifest = _selection_manifest(
        validation,
        purpose="selection_validation_only_not_final_test",
        corpus_identity_sha256=corpus_identity_sha256,
    )
    return train, validation, train_manifest, validation_manifest


def _training_plan(
    corpus_files: tuple[CorpusFileIdentity, ...],
    train_manifest: Mapping[str, Any],
    corpus_identity_sha256: str,
    requested_vocab: int,
) -> TokenizerTrainingManifest:
    return TokenizerTrainingManifest(
        experiment_id=f"TOK-187-DATA183-BPE-v{requested_vocab}",
        algorithm="bpe",
        tokenizers_version=TOKENIZERS_VERSION,
        dataset_id=f"DATA183:{corpus_identity_sha256}:train",
        dataset_manifest_sha256=str(train_manifest["identity_sha256"]),
        corpus_files=corpus_files,
        vocab_size=requested_vocab,
        min_frequency=2,
    )


def _all_stream_mechanics(
    tokenizer: HFTokenizerAdapter,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    unknowns = 0
    total_tokens = 0
    for row in rows:
        text = str(row["text"])
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != text:
            raise Tok187Error(f"strict tokenizer roundtrip failed: {row['record_id']}")
        count = sum(token_id == tokenizer.unk_id for token_id in ids)
        if count:
            raise Tok187Error(
                f"unintended <unk> tokens in {row['record_id']}: {count}"
            )
        unknowns += count
        total_tokens += len(ids)
    return {
        "documents": len(rows),
        "tokens": total_tokens,
        "strict_round_trip_all": True,
        "unknown_tokens": unknowns,
    }


def _fertility(
    tokenizer: HFTokenizerAdapter,
    validation_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    key_map = {"uk": "ua", "en": "en", "code": "code"}
    for stratum, display in key_map.items():
        selected = [row for row in validation_rows if str(row["stratum"]) == stratum]
        codepoints = 0
        utf8_bytes = 0
        tokens = 0
        documents = 0
        for row in selected:
            text = str(row["text"])
            ids = tokenizer.encode(text)
            codepoints += len(text)
            utf8_bytes += len(text.encode("utf-8"))
            tokens += len(ids)
            documents += 1
        result[display] = {
            "source_stratum": stratum,
            "documents": documents,
            "codepoints": codepoints,
            "utf8_bytes": utf8_bytes,
            "tokens": tokens,
            "tokens_per_codepoint": tokens / codepoints if codepoints else 0.0,
            "tokens_per_utf8_byte": tokens / utf8_bytes if utf8_bytes else 0.0,
        }
    return result


def _tokenizer_throughput(
    tokenizer: HFTokenizerAdapter,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    texts = tuple(str(row["text"]) for row in rows)
    total_bytes = sum(len(text.encode("utf-8")) for text in texts)
    if total_bytes <= 0:
        raise Tok187Error("throughput payload is empty")

    reference_tokens = sum(len(tokenizer.encode(text)) for text in texts)
    durations = []
    for _ in range(THROUGHPUT_REPEATS):
        start = time.perf_counter()
        tokens = sum(len(tokenizer.encode(text)) for text in texts)
        elapsed = time.perf_counter() - start
        if tokens != reference_tokens or elapsed <= 0:
            raise Tok187Error("tokenizer throughput measurement drift")
        durations.append(elapsed)
    median_seconds = statistics.median(durations)
    return {
        "diagnostic_only": True,
        "repeats": THROUGHPUT_REPEATS,
        "payload_documents": len(texts),
        "payload_utf8_bytes": total_bytes,
        "payload_tokens": reference_tokens,
        "median_seconds": median_seconds,
        "median_mib_per_second": total_bytes / (1024.0 * 1024.0) / median_seconds,
        "median_tokens_per_second": reference_tokens / median_seconds,
        "all_seconds": durations,
    }


def _train_tokenizers(
    *,
    corpus_files: tuple[CorpusFileIdentity, ...],
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    train_manifest: Mapping[str, Any],
    corpus_identity_sha256: str,
    output: Path,
) -> tuple[dict[int, HFTokenizerAdapter], dict[str, Any]]:
    texts = tuple(str(row["text"]) for row in train_rows)
    all_rows = tuple(train_rows) + tuple(validation_rows)
    adapters_by_actual: dict[int, HFTokenizerAdapter] = {}
    evidence: dict[str, Any] = {}

    for requested in REQUESTED_GRID:
        plan = _training_plan(
            corpus_files, train_manifest, corpus_identity_sha256, requested
        )
        first = train_hf_tokenizer(plan, texts)
        second = train_hf_tokenizer(plan, texts)
        first_bytes = first._tokenizer.to_str().encode("utf-8")
        second_bytes = second._tokenizer.to_str().encode("utf-8")
        if first_bytes != second_bytes:
            raise Tok187Error(f"BPE-{requested}: tokenizer artifacts are not byte-identical")
        if first.artifact_identity != second.artifact_identity:
            raise Tok187Error(f"BPE-{requested}: tokenizer artifact identities differ")

        mechanics = _all_stream_mechanics(first, all_rows)
        actual = first.vocab_size
        artifact_path = output / "tokenizers" / f"bpe-r{requested}-a{actual}.tokenizer.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(first_bytes)
        artifact_sha = hashlib.sha256(first_bytes).hexdigest()
        if artifact_sha != first.artifact_identity.tokenizer_json_sha256:
            raise Tok187Error("serialized tokenizer artifact hash mismatch")
        _write_json(
            output / "tokenizers" / f"bpe-r{requested}.training-manifest.json",
            plan.to_dict(),
        )

        existing = adapters_by_actual.get(actual)
        alias_of = None
        if existing is not None:
            if existing.artifact_identity.vocab_sha256 != first.artifact_identity.vocab_sha256:
                raise Tok187Error(
                    f"actual vocab {actual} has multiple incompatible vocabularies"
                )
            alias_of = existing.artifact_identity.tokenizer_json_sha256
        else:
            adapters_by_actual[actual] = first

        evidence[str(requested)] = {
            "requested_vocab_size": requested,
            "actual_vocab_size": actual,
            "artifact": {
                **asdict(first.artifact_identity),
                "special_tokens": dict(first.artifact_identity.special_tokens),
                "config_sha256": first.artifact_identity.config_sha256,
            },
            "two_independent_deterministic_trainings": True,
            "byte_identical_tokenizer_artifact": True,
            "serialized_tokenizer_sha256": artifact_sha,
            "alias_of_tokenizer_json_sha256": alias_of,
            "mechanics": mechanics,
            "selection_validation_fertility": _fertility(first, validation_rows),
            "throughput": _tokenizer_throughput(first, all_rows),
        }
    return adapters_by_actual, evidence


def _anchors() -> dict[str, ModelSpec]:
    family = controlled_specs()
    candidates = {
        "500K": family[2],
        "1M": family[3],
    }
    counts = tuple(candidates[label].parameter_count() for label in SCALE_LABELS)
    if counts != TARGET_COUNTS:
        raise Tok187Error(f"RESEARCH-41 geometry anchors drifted: {counts!r}")
    return candidates


def _solve_not_above(template: ModelSpec, cap: int, vocab_size: int) -> ModelSpec:
    solved = rebalance_d_ff_for_vocabulary(
        template,
        target_parameters=cap,
        vocab_size=vocab_size,
        d_ff_alignment=1,
    ).model
    if solved.parameter_count() > cap:
        if solved.d_ff <= 1:
            raise Tok187Error("cannot lower d_ff under matched parameter cap")
        solved = replace(solved, d_ff=solved.d_ff - 1)
    if solved.parameter_count() > cap:
        raise Tok187Error("larger vocabulary received extra total capacity")
    return solved


def _solve_geometries(actual_vocabs: Sequence[int]) -> dict[str, dict[int, ModelSpec]]:
    unique = sorted(set(int(value) for value in actual_vocabs))
    result: dict[str, dict[int, ModelSpec]] = {}
    for label, template in _anchors().items():
        target = template.parameter_count()
        previous = target
        scale: dict[int, ModelSpec] = {}
        for vocab in unique:
            spec = _solve_not_above(template, previous, vocab)
            count = spec.parameter_count()
            if abs(count - target) / target > PARAMETER_TOLERANCE:
                raise Tok187Error(
                    f"{label} vocab {vocab} misses parameter tolerance: "
                    f"{abs(count-target)/target:.6%}"
                )
            if count > previous:
                raise Tok187Error("monotone parameter cap violated")
            scale[vocab] = spec
            previous = count
        result[label] = scale
    return result


def _embedding_tax(
    geometries: Mapping[str, Mapping[int, ModelSpec]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for label, by_vocab in geometries.items():
        rows: dict[str, Any] = {}
        for vocab, spec in sorted(by_vocab.items()):
            breakdown = spec.parameter_breakdown()
            embedding = breakdown["token_embedding"]
            byte_embedding = 256 * spec.d_model
            rows[str(vocab)] = {
                "actual_vocab_size": vocab,
                "model_identity_sha256": spec.identity_sha256(),
                "parameter_count": spec.parameter_count(),
                "target_anchor_parameters": _anchors()[label].parameter_count(),
                "d_model": spec.d_model,
                "d_ff": spec.d_ff,
                "tied_lm_head": spec.tie_word_embeddings,
                "embedding_parameters": embedding,
                "embedding_fraction": embedding / spec.parameter_count(),
                "incremental_embedding_parameters_vs_byte_vocab": embedding - byte_embedding,
                "incremental_embedding_fraction_vs_byte_vocab": (
                    (embedding - byte_embedding) / spec.parameter_count()
                ),
            }
        result[label] = rows
    return result


def _mixed_records(rows: Sequence[dict[str, Any]]) -> Iterator[TextRecord]:
    groups: dict[str, list[dict[str, Any]]] = {"uk": [], "en": [], "code": []}
    for row in rows:
        stratum = str(row["stratum"])
        if stratum in groups:
            groups[stratum].append(row)
    positions = {name: 0 for name in groups}
    while True:
        emitted = False
        for stratum in ("uk", "en", "code"):
            pos = positions[stratum]
            if pos >= len(groups[stratum]):
                continue
            row = groups[stratum][pos]
            positions[stratum] = pos + 1
            emitted = True
            yield TextRecord(str(row["record_id"]), str(row["text"]), "train")
        if not emitted:
            return


def _mask_labels(labels: torch.Tensor, remaining: int) -> torch.Tensor:
    masked = labels.clone()
    positions = masked[:, 1:].ne(-100).nonzero(as_tuple=False)
    if positions.shape[0] <= remaining:
        return masked
    for batch_index, shifted_index in positions[remaining:].tolist():
        masked[batch_index, shifted_index + 1] = -100
    return masked


def _next_batch(
    iterator: Iterator[Any],
    remaining: int,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    examples = []
    for _ in range(BATCH):
        try:
            examples.append(next(iterator))
        except StopIteration as exc:
            raise Tok187Error(
                "real training corpus exhausted before model-probe optimized-token budget"
            ) from exc
    labels = torch.tensor([item.labels for item in examples], dtype=torch.long)
    labels = _mask_labels(labels, remaining)
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


def _state_hash(model: TwelveSixDecoder) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _document_nll(
    model: TwelveSixDecoder,
    tokenizer: HFTokenizerAdapter,
    text: str,
) -> tuple[float, int, int]:
    ids = tokenizer.encode(text)
    if tokenizer.decode(ids, skip_special_tokens=False) != text:
        raise Tok187Error("selection-validation tokenizer roundtrip failed")
    if any(token_id == tokenizer.unk_id for token_id in ids):
        raise Tok187Error("selection-validation tokenizer emitted <unk>")
    if len(ids) < 2:
        return 0.0, 0, 0

    nll_sum = 0.0
    target_tokens = 0
    modeled_bytes = 0
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
        nll_sum += float(nll.item())
        target_tokens += targets.numel()
        modeled_bytes += len(
            tokenizer.decode(window[1:], skip_special_tokens=False).encode("utf-8")
        )
    if modeled_bytes <= 0:
        raise Tok187Error("selection-validation document has no modeled bytes")
    return nll_sum, target_tokens, modeled_bytes


@torch.no_grad()
def _evaluate(
    model: TwelveSixDecoder,
    tokenizer: HFTokenizerAdapter,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    before = _state_hash(model)
    was_training = model.training
    model.eval()
    groups: dict[str, list[dict[str, Any]]] = {"uk": [], "en": [], "code": []}
    for row in rows:
        stratum = str(row["stratum"])
        if stratum in groups:
            groups[stratum].append(row)
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
            for row in groups[stratum]:
                text = str(row["text"])
                nll, targets, byte_count = _document_nll(model, tokenizer, text)
                nll_sum += nll
                modeled_bytes += byte_count
                target_tokens += targets
                source_bytes += len(text.encode("utf-8"))
            if modeled_bytes <= 0:
                raise Tok187Error(f"no modeled selection-validation bytes for {stratum}")
            by_stratum[stratum] = {
                "bits_per_byte": nll_sum / math.log(2.0) / modeled_bytes,
                "nll_sum": nll_sum,
                "modeled_utf8_bytes": modeled_bytes,
                "source_utf8_bytes": source_bytes,
                "target_tokens": target_tokens,
                "documents": len(groups[stratum]),
            }
            total_nll += nll_sum
            total_bytes += modeled_bytes
            total_tokens += target_tokens
    finally:
        model.train(was_training)
    after = _state_hash(model)
    if before != after:
        raise Tok187Error("evaluation mutated model state")
    return {
        "aggregate_bits_per_byte": total_nll / math.log(2.0) / total_bytes,
        "macro_bits_per_byte": sum(
            by_stratum[name]["bits_per_byte"] for name in ("uk", "en", "code")
        )
        / 3.0,
        "nll_sum": total_nll,
        "modeled_utf8_bytes": total_bytes,
        "target_tokens": total_tokens,
        "by_stratum": by_stratum,
        "model_state_sha256_before": before,
        "model_state_sha256_after": after,
        "non_mutation_passed": True,
        "purpose": "selection_validation",
        "final_test_used": False,
        "bpb_denominator": (
            "UTF-8 bytes decoded from scored target-token suffixes; first token in each "
            "window is context-only and not scored"
        ),
    }


def _run_model_probe(
    *,
    spec: ModelSpec,
    tokenizer: HFTokenizerAdapter,
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, InitSpec())
    initial_state = _state_hash(model)
    initial_eval = _evaluate(model, tokenizer, validation_rows)
    trainer = Trainer(model, _trainer_config(seed), device="cpu")
    iterator = iter_packed_examples(
        _mixed_records(train_rows),
        tokenizer,
        expected_split="train",
        sequence_length=SEQ,
        cross_document=False,
    )
    record_bytes = {
        str(row["record_id"]): len(str(row["text"]).encode("utf-8"))
        for row in train_rows
    }
    touched: set[str] = set()
    losses: list[float] = []
    start = time.perf_counter()
    while trainer.tokens_seen < OPTIMIZED_TOKENS:
        batch, record_ids = _next_batch(iterator, OPTIMIZED_TOKENS - trainer.tokens_seen)
        metrics = trainer.train_microbatch(batch)
        touched.update(record_ids)
        if metrics.update_loss is not None:
            losses.append(float(metrics.update_loss))
    elapsed = time.perf_counter() - start
    if trainer.tokens_seen != OPTIMIZED_TOKENS:
        raise Tok187Error("model probe optimized-token ledger drift")
    if not losses:
        raise Tok187Error("model probe executed no optimizer update")
    final_eval = _evaluate(model, tokenizer, validation_rows)
    return {
        "seed": seed,
        "paired_seed": True,
        "random_initialization": {
            "pretrained_weights_loaded": False,
            "init_spec": InitSpec().to_dict(),
            "initial_model_state_sha256": initial_state,
        },
        "model": spec.to_dict(),
        "model_identity_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "optimized_tokens": trainer.tokens_seen,
        "optimizer_steps": trainer.optimizer_step,
        "training_loss": {
            "first_update": losses[0],
            "last_update": losses[-1],
            "decreased_first_to_last": losses[-1] < losses[0],
        },
        "training_throughput": {
            "elapsed_seconds": elapsed,
            "optimized_tokens_per_second": OPTIMIZED_TOKENS / elapsed,
            "unique_source_documents_touched": len(touched),
            "unique_source_utf8_bytes_touched": sum(record_bytes[r] for r in touched),
        },
        "evaluation": {"initial": initial_eval, "final": final_eval},
        "final_model_state_sha256": _state_hash(model),
    }


def _scale_summary(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    selected = [row for row in rows if row["scale"] == label]
    by_vocab: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_vocab[int(row["actual_vocab_size"])].append(row)

    candidates = []
    for vocab, members in sorted(by_vocab.items()):
        if {int(m["seed"]) for m in members} != set(MODEL_SEEDS):
            raise Tok187Error(f"{label} vocab {vocab} is missing paired seeds")
        aggregate = [
            float(m["run"]["evaluation"]["final"]["aggregate_bits_per_byte"])
            for m in members
        ]
        macro = [
            float(m["run"]["evaluation"]["final"]["macro_bits_per_byte"])
            for m in members
        ]
        by_stratum = {}
        for stratum in ("uk", "en", "code"):
            values = [
                float(m["run"]["evaluation"]["final"]["by_stratum"][stratum]["bits_per_byte"])
                for m in members
            ]
            by_stratum[stratum] = {
                "mean_bpb": statistics.mean(values),
                "values_by_seed": {
                    str(int(m["seed"])): float(
                        m["run"]["evaluation"]["final"]["by_stratum"][stratum][
                            "bits_per_byte"
                        ]
                    )
                    for m in sorted(members, key=lambda item: int(item["seed"]))
                },
            }
        candidates.append(
            {
                "actual_vocab_size": vocab,
                "parameter_count": members[0]["run"]["parameter_count"],
                "d_ff": members[0]["run"]["model"]["d_ff"],
                "mean_final_aggregate_bpb": statistics.mean(aggregate),
                "median_final_aggregate_bpb": statistics.median(aggregate),
                "mean_final_macro_bpb": statistics.mean(macro),
                "aggregate_bpb_by_seed": {
                    str(int(m["seed"])): float(
                        m["run"]["evaluation"]["final"]["aggregate_bits_per_byte"]
                    )
                    for m in sorted(members, key=lambda item: int(item["seed"]))
                },
                "by_stratum": by_stratum,
            }
        )

    ranked = sorted(
        candidates,
        key=lambda item: (
            item["mean_final_aggregate_bpb"],
            item["mean_final_macro_bpb"],
            item["actual_vocab_size"],
        ),
    )
    for rank, candidate in enumerate(ranked, 1):
        candidate["rank_primary_held_out_bpb"] = rank

    paired = None
    if len(ranked) >= 2:
        top, runner = ranked[0], ranked[1]
        pairs = [
            Pair(
                run_id=str(seed),
                baseline=float(runner["aggregate_bpb_by_seed"][str(seed)]),
                candidate=float(top["aggregate_bpb_by_seed"][str(seed)]),
            )
            for seed in MODEL_SEEDS
        ]
        decision = analyze_paired_runs(
            pairs,
            candidate=f"bpe-{top['actual_vocab_size']}",
            baseline=f"bpe-{runner['actual_vocab_size']}",
            config=DecisionConfig(
                materiality=BPB_MATERIALITY,
                metric_name="selection_validation_aggregate_bpb",
                metric_purpose=MetricPurpose.SELECTION_VALIDATION,
                direction=MetricDirection.LOWER_IS_BETTER,
                min_repeats=3,
            ),
        )
        paired = decision.to_dict()

    return {
        "scale": label,
        "primary_metric": "mean final selection-validation aggregate bits-per-byte",
        "secondary_metric": "mean final selection-validation macro bits-per-byte",
        "ranked_candidates": ranked,
        "paired_top_vs_runner_up": paired,
        "three_paired_seeds_complete": all(
            len(by_vocab[vocab]) == len(MODEL_SEEDS) for vocab in by_vocab
        ),
    }


def _promotion_status(
    data183_report: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    blockers = []
    truth = data183_report["truth_boundary"]
    if truth.get("external_real_code_present") is not True:
        blockers.append("EXTERNAL_REAL_CODE_UNAVAILABLE")
    if data183_report["representativeness"].get("full_v0_2_claim") is not True:
        blockers.append("FULL_V0_2_REPRESENTATIVENESS_NOT_ESTABLISHED")
    for label in SCALE_LABELS:
        summary = summaries[label]
        if summary.get("three_paired_seeds_complete") is not True:
            blockers.append(f"{label}_FEWER_THAN_THREE_PAIRED_SEEDS")
        decision = summary.get("paired_top_vs_runner_up")
        if not decision or decision.get("decision") != Decision.CLEAR_WIN.value:
            blockers.append(f"{label}_NO_CLEAR_PAIRED_BPB_WIN")
    winners = [
        int(summaries[label]["ranked_candidates"][0]["actual_vocab_size"])
        for label in SCALE_LABELS
    ]
    if len(set(winners)) != 1:
        blockers.append("SCALE_DEPENDENT_BPB_WINNER")
    return {
        "tokenizer_promoted": False,
        "tokenizer_frozen": False,
        "proposed_promotion": None,
        "promotion_allowed": False,
        "blockers": sorted(set(blockers)),
        "reason": (
            "TOK-187 records selection-validation evidence only. Current DATA-183 is "
            "external-real UA/EN plus project-authored code and is not a full V0.2 "
            "representative corpus; final-test material is excluded."
        ),
    }


def run(
    *,
    repo: Path,
    source_sha: str,
    data183_report_path: Path,
    build_root: Path,
    manifest_path: Path,
    environment_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    _require_head(repo, source_sha)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    environment = _validate_environment(repo, environment_manifest)
    data183_report, manifest = _validate_data183(
        data183_report_path, manifest_path, source_sha
    )
    rows, corpus_files = _physical_rows(build_root, manifest)
    train_rows, validation_rows, train_selection, validation_selection = _partition_rows(
        rows, str(data183_report["corpus_identity_sha256"])
    )
    _write_json(output / "tokenizer-training-selection.json", train_selection)
    _write_json(output / "selection-validation.json", validation_selection)

    tokenizers, tokenizer_evidence = _train_tokenizers(
        corpus_files=corpus_files,
        train_rows=train_rows,
        validation_rows=validation_rows,
        train_manifest=train_selection,
        corpus_identity_sha256=str(data183_report["corpus_identity_sha256"]),
        output=output,
    )
    geometries = _solve_geometries(tuple(tokenizers))
    embedding_tax = _embedding_tax(geometries)

    matrix: list[dict[str, Any]] = []
    for label in SCALE_LABELS:
        for actual_vocab, tokenizer in sorted(tokenizers.items()):
            spec = geometries[label][actual_vocab]
            for seed in MODEL_SEEDS:
                probe = _run_model_probe(
                    spec=spec,
                    tokenizer=tokenizer,
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                    seed=seed,
                )
                row = {
                    "scale": label,
                    "actual_vocab_size": actual_vocab,
                    "seed": seed,
                    "run": probe,
                }
                matrix.append(row)
                _write_json(
                    output
                    / "rows"
                    / f"scale-{label}-vocab-{actual_vocab}-seed-{seed}.json",
                    row,
                )

    summaries = {label: _scale_summary(matrix, label) for label in SCALE_LABELS}
    promotion = _promotion_status(data183_report, summaries)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
            "local_free_only": True,
            "paid_compute": False,
        },
        "corpus": {
            "data183_report_sha256": data183_report["report_sha256"],
            "corpus_identity_sha256": data183_report["corpus_identity_sha256"],
            "status": data183_report["status"],
            "truth_boundary": data183_report["truth_boundary"],
            "representativeness": data183_report["representativeness"],
        },
        "environment": environment,
        "protocol": {
            "implementation": "existing HF Tokenizers train_hf_tokenizer only",
            "algorithm": "ByteLevel BPE",
            "requested_grid": list(REQUESTED_GRID),
            "grid_rationale": (
                "MODEL-116 promising 320/384/437 actual-vocabulary neighborhood plus "
                "512 upper bracket"
            ),
            "tokenizer_trainings_per_candidate": 2,
            "byte_identical_artifact_required": True,
            "strict_roundtrip_required": True,
            "zero_unknowns_required": True,
            "model_scales": {
                label: _anchors()[label].parameter_count() for label in SCALE_LABELS
            },
            "paired_model_seeds": list(MODEL_SEEDS),
            "optimized_tokens_per_model_probe": OPTIMIZED_TOKENS,
            "selection_metric_primary": "held-out aggregate bits-per-byte",
            "selection_metric_secondary": "macro UA/EN/code bits-per-byte",
            "paired_bpb_materiality": BPB_MATERIALITY,
            "selection_validation_identity_sha256": validation_selection[
                "identity_sha256"
            ],
            "final_test_used": False,
        },
        "tokenizer_training_selection": train_selection,
        "selection_validation": validation_selection,
        "tokenizers": tokenizer_evidence,
        "embedding_tax": embedding_tax,
        "model_probe_summary": summaries,
        "promotion": promotion,
        "truth_boundary": {
            "mechanics_only_data": False,
            "external_real_ua": True,
            "external_real_en": True,
            "external_real_code": bool(
                data183_report["truth_boundary"]["external_real_code_present"]
            ),
            "project_authored_code": bool(
                data183_report["truth_boundary"]["project_authored_code_present"]
            ),
            "selection_validation_only": True,
            "final_test_touched": False,
            "tokenizer_freeze": False,
            "local_free_only": True,
        },
        "machine": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "tokenizers": importlib.metadata.version("tokenizers"),
            "logical_cpu_count": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
    }
    report["report_sha256"] = hash_json(report)
    _write_json(output / "tok187-bpe-real-selection.json", report)
    return report


def validate(path: Path, expected_source_sha: str | None = None) -> dict[str, Any]:
    report = _read_json(path)
    expected = str(report.get("report_sha256", ""))
    core = dict(report)
    core.pop("report_sha256", None)
    if hash_json(core) != expected:
        raise Tok187Error("TOK-187 report self-hash mismatch")
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise Tok187Error("TOK-187 schema/authority mismatch")
    if expected_source_sha and report["source"]["git_sha"] != expected_source_sha:
        raise Tok187Error("TOK-187 source SHA mismatch")
    if report["protocol"]["final_test_used"] is not False:
        raise Tok187Error("final-test material entered tokenizer selection")
    if report["promotion"]["tokenizer_promoted"] is not False:
        raise Tok187Error("TOK-187 unexpectedly promoted a tokenizer")
    if report["truth_boundary"]["local_free_only"] is not True:
        raise Tok187Error("LOCAL_FREE truth boundary weakened")
    for candidate in report["tokenizers"].values():
        if candidate["byte_identical_tokenizer_artifact"] is not True:
            raise Tok187Error("non-repeatable tokenizer artifact accepted")
        if candidate["mechanics"]["strict_round_trip_all"] is not True:
            raise Tok187Error("roundtrip failure accepted")
        if candidate["mechanics"]["unknown_tokens"] != 0:
            raise Tok187Error("unknown-token failure accepted")
    for label in SCALE_LABELS:
        if report["model_probe_summary"][label]["three_paired_seeds_complete"] is not True:
            raise Tok187Error(f"{label} missing three paired seeds")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repo-root", type=Path, default=Path("."))
    run_parser.add_argument("--source-sha", required=True)
    run_parser.add_argument("--data183-report", type=Path, required=True)
    run_parser.add_argument("--build-root", type=Path, required=True)
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--environment-manifest", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--expected-source-sha")
    args = parser.parse_args(argv)

    if args.cmd == "run":
        report = run(
            repo=args.repo_root,
            source_sha=args.source_sha,
            data183_report_path=args.data183_report,
            build_root=args.build_root,
            manifest_path=args.manifest,
            environment_manifest=args.environment_manifest,
            output=args.output_dir,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "corpus_identity_sha256": report["corpus"][
                        "corpus_identity_sha256"
                    ],
                    "selection_validation_identity_sha256": report["protocol"][
                        "selection_validation_identity_sha256"
                    ],
                    "observed_winners": {
                        label: report["model_probe_summary"][label][
                            "ranked_candidates"
                        ][0]["actual_vocab_size"]
                        for label in SCALE_LABELS
                    },
                    "promotion": report["promotion"],
                    "report_sha256": report["report_sha256"],
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
    else:
        report = validate(args.report, args.expected_source_sha)
        print(
            json.dumps(
                {
                    "validation": "PASS",
                    "report_sha256": report["report_sha256"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
