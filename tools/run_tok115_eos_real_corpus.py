#!/usr/bin/env python3
"""TOK-115 matched EOS/EOD boundary experiment on the DATA-25 corpus.

This is experiment orchestration only. Model math, tokenizers, packing, Trainer,
observability, checkpointing, evaluation primitives, and generation remain owned
by their incumbent modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from twelve_six.data.corpus_v01 import build_corpus
from twelve_six.inference import GenerationConfig, generate
from twelve_six.integration.s0_runtime import S0TorchInferenceBackend
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.packing import TextRecord, batch_examples, collate_rows, iter_packed_examples
from twelve_six.tokenization import (
    ByteTokenizer,
    EXPERIMENTAL_EOS_ID,
    ExperimentalByteEosTokenizer,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.observability import TrainingObserver
from twelve_six.training.trainer import Trainer


SCHEMA = "12-6.tok115-eos-real-corpus.v1"
CORPUS_CONFIG = Path("configs/data/corpus_v01.json")
STATIC_CORPUS_MANIFEST = Path("data/corpus/v0.1/manifest.json")
BUILT_CORPUS_DIR = Path("data/build/corpus_v01")
RUNTIME_LOCK = Path("requirements/locks/linux-x86_64/runtime.lock.txt")
EXPECTED_CORPUS_ID = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
SEED = 1337
SEQ_LEN = 128
BATCH_SIZE = 4
MATCHED_SOURCE_BYTES = 65_536
SUBSTANTIAL_SOURCE_BYTES = 262_144
HELDOUT_PANEL_BYTES = 65_536
BOUNDARY_DOCS = 48
MIXTURE = {"uk": 0.45, "en": 0.35, "code": 0.20}
MIXTURE_ORDER = {"uk": 0, "en": 1, "code": 2}

SCALES = {
    "268k": dict(d_model=72, n_layers=4, n_heads=6, n_kv_heads=6, head_dim=12, d_ff=192),
    "1m": dict(d_model=128, n_layers=5, n_heads=8, n_kv_heads=8, head_dim=16, d_ff=352),
}
CANDIDATES = {
    "A": dict(add_eos=False, cross_document=False, label="no_eos_document_isolated"),
    "B": dict(add_eos=True, cross_document=False, label="eos_document_isolated"),
    "C": dict(add_eos=True, cross_document=True, label="eos_multi_document"),
}


@dataclass(frozen=True)
class CorpusRow:
    record_id: str
    text: str
    split: str
    stratum: str
    byte_tokens: int
    content_sha256: str


@dataclass
class PreparedPlan:
    batches: list[dict[str, torch.Tensor]]
    stats: dict[str, Any]
    records: list[CorpusRow]


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _read_rows(corpus_dir: Path) -> list[CorpusRow]:
    rows: list[CorpusRow] = []
    for shard in sorted((corpus_dir / "shards").glob("part-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            rows.append(
                CorpusRow(
                    record_id=str(raw["record_id"]),
                    text=str(raw["text"]),
                    split=str(raw["split"]),
                    stratum=str(raw["stratum"]),
                    byte_tokens=int(raw["byte_tokens"]),
                    content_sha256=str(raw["content_sha256"]),
                )
            )
    if not rows:
        raise RuntimeError("DATA-25 corpus build emitted no rows")
    return rows


def ensure_corpus() -> tuple[dict[str, Any], list[CorpusRow]]:
    static = _json(STATIC_CORPUS_MANIFEST)
    if static["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise RuntimeError("static DATA-25 corpus identity drifted")
    external = _json(Path("data/external/external_sources.json"))
    if external.get("sources") != []:
        raise RuntimeError("TOK-115 truth boundary expects DATA-25 external source registry to be empty")
    shutil.rmtree(BUILT_CORPUS_DIR, ignore_errors=True)
    built = build_corpus(CORPUS_CONFIG, output_dir=BUILT_CORPUS_DIR)
    if not isinstance(built, Mapping):
        built = _json(BUILT_CORPUS_DIR / "manifest.json")
    built = dict(built)
    if built["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise RuntimeError(
            f"rebuilt corpus identity mismatch: {built['corpus_identity_sha256']} != {EXPECTED_CORPUS_ID}"
        )
    rows = _read_rows(BUILT_CORPUS_DIR)
    return built, rows


def _weighted_order(rows: Iterable[CorpusRow], split: str) -> list[CorpusRow]:
    groups: dict[str, list[CorpusRow]] = {key: [] for key in MIXTURE}
    for row in rows:
        if row.split == split and row.stratum in groups:
            groups[row.stratum].append(row)
    for key in groups:
        groups[key].sort(key=lambda r: (r.record_id, r.content_sha256))

    indexes = {key: 0 for key in MIXTURE}
    used = {key: 0 for key in MIXTURE}
    ordered: list[CorpusRow] = []
    while True:
        available = [key for key in MIXTURE if indexes[key] < len(groups[key])]
        if not available:
            break
        key = min(
            available,
            key=lambda item: (
                used[item] / MIXTURE[item],
                MIXTURE_ORDER[item],
            ),
        )
        row = groups[key][indexes[key]]
        indexes[key] += 1
        used[key] += row.byte_tokens
        ordered.append(row)
    return ordered


def select_source_slice(rows: Sequence[CorpusRow], target_bytes: int) -> list[CorpusRow]:
    selected: list[CorpusRow] = []
    total = 0
    for row in rows:
        selected.append(row)
        total += row.byte_tokens
        if total >= target_bytes:
            break
    if total < target_bytes:
        raise RuntimeError(f"insufficient source bytes: {total} < {target_bytes}")
    return selected


def source_slice_identity(rows: Sequence[CorpusRow]) -> str:
    return hash_json(
        [
            {
                "record_id": row.record_id,
                "content_sha256": row.content_sha256,
                "byte_tokens": row.byte_tokens,
                "stratum": row.stratum,
                "split": row.split,
            }
            for row in rows
        ]
    )


def source_slice_stats(rows: Sequence[CorpusRow]) -> dict[str, Any]:
    by_stratum = {key: {"documents": 0, "source_bytes": 0} for key in MIXTURE}
    for row in rows:
        by_stratum[row.stratum]["documents"] += 1
        by_stratum[row.stratum]["source_bytes"] += row.byte_tokens
    total = sum(item["source_bytes"] for item in by_stratum.values())
    return {
        "documents": len(rows),
        "source_bytes": total,
        "identity_sha256": source_slice_identity(rows),
        "by_stratum": by_stratum,
        "observed_byte_mix_percent": {
            key: 100.0 * value["source_bytes"] / total for key, value in by_stratum.items()
        },
    }


def tokenizer_for(candidate: str):
    return ByteTokenizer() if candidate == "A" else ExperimentalByteEosTokenizer()


def model_spec(scale: str, candidate: str) -> ModelSpec:
    vocab_size = 256 if candidate == "A" else 257
    dims = SCALES[scale]
    return ModelSpec(
        schema_version=1,
        vocab_size=vocab_size,
        max_seq_len=256,
        rope_rotary_dim=dims["head_dim"],
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10_000.0,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
        **dims,
    )


def build_model(scale: str, candidate: str) -> tuple[TwelveSixDecoder, ModelSpec, InitSpec]:
    spec = model_spec(scale, candidate)
    init = InitSpec()
    _seed_all(SEED)
    model = TwelveSixDecoder(spec, init_spec=init)
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if actual != spec.parameter_count():
        raise RuntimeError(f"parameter formula/runtime mismatch: {spec.parameter_count()} vs {actual}")
    return model, spec, init


def _as_text_records(records: Sequence[CorpusRow]) -> list[TextRecord]:
    return [TextRecord(record_id=row.record_id, text=row.text, split=row.split) for row in records]


def prepare_plan(candidate: str, records: Sequence[CorpusRow]) -> PreparedPlan:
    cfg = CANDIDATES[candidate]
    tokenizer = tokenizer_for(candidate)
    examples = list(
        iter_packed_examples(
            _as_text_records(records),
            tokenizer,
            expected_split=records[0].split,
            sequence_length=SEQ_LEN,
            fill_token_id=0,
            ignore_index=-100,
            add_bos=False,
            add_eos=cfg["add_eos"],
            cross_document=cfg["cross_document"],
        )
    )
    if not examples:
        raise RuntimeError("packer emitted no examples")

    batches: list[dict[str, torch.Tensor]] = []
    content_targets = 0
    eos_targets = 0
    valid_targets = 0
    allocated_positions = 0
    for group in batch_examples(examples, batch_size=BATCH_SIZE, drop_last=False):
        collated = collate_rows(group, target_mode="target_ids")
        input_ids = torch.tensor(collated["input_ids"], dtype=torch.long)
        target_ids = torch.tensor(collated["target_ids"], dtype=torch.long)
        loss_mask = torch.tensor(collated["loss_mask"], dtype=torch.bool)
        valid = loss_mask & target_ids.ne(-100)
        valid_values = target_ids.masked_select(valid)
        content_targets += int(((valid_values >= 0) & (valid_values <= 255)).sum().item())
        eos_targets += int((valid_values == EXPERIMENTAL_EOS_ID).sum().item())
        valid_targets += int(valid.sum().item())
        allocated_positions += int(target_ids.numel())
        batches.append(
            {
                "input_ids": input_ids,
                "target_ids": target_ids,
                "loss_mask": loss_mask,
            }
        )

    if candidate == "A" and eos_targets != 0:
        raise RuntimeError("candidate A unexpectedly contains EOS targets")
    if candidate in {"B", "C"} and eos_targets <= 0:
        raise RuntimeError(f"candidate {candidate} failed to expose EOS targets")

    stats = {
        "packing_version": "s0-byte-pack-v1",
        "candidate": candidate,
        "policy": cfg["label"],
        "sequence_length": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "examples": len(examples),
        "optimizer_steps": len(batches),
        "content_targets": content_targets,
        "eos_targets": eos_targets,
        "valid_targets": valid_targets,
        "allocated_positions": allocated_positions,
        "packing_efficiency_valid_targets": valid_targets / allocated_positions,
        "packing_efficiency_content_targets": content_targets / allocated_positions,
        "padding_positions": allocated_positions - valid_targets,
        "eos_overhead_vs_content": eos_targets / content_targets,
        "cross_document": cfg["cross_document"],
        "add_eos": cfg["add_eos"],
        "direct_unmarked_content_to_content_cross_document_transitions": 0,
        "source_slice": source_slice_stats(records),
    }
    return PreparedPlan(batches=batches, stats=stats, records=list(records))


def _content_eval_examples(records: Sequence[CorpusRow]):
    # Common held-out target universe: byte-only, intra-document, no EOS, no cross-document pairs.
    return list(
        iter_packed_examples(
            _as_text_records(records),
            ByteTokenizer(),
            expected_split=records[0].split,
            sequence_length=SEQ_LEN,
            fill_token_id=0,
            ignore_index=-100,
            add_bos=False,
            add_eos=False,
            cross_document=False,
        )
    )


@torch.no_grad()
def evaluate_content_bpb(model: TwelveSixDecoder, records: Sequence[CorpusRow]) -> dict[str, Any]:
    before_sha = _tensor_state_sha256(model)
    was_training = model.training
    model.eval()
    total_nll = 0.0
    count = 0
    examples = _content_eval_examples(records)
    for group in batch_examples(examples, batch_size=8, drop_last=False):
        collated = collate_rows(group, target_mode="target_ids")
        x = torch.tensor(collated["input_ids"], dtype=torch.long)
        y = torch.tensor(collated["target_ids"], dtype=torch.long)
        mask = torch.tensor(collated["loss_mask"], dtype=torch.bool) & y.ne(-100)
        logits = model(x).logits
        nll = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(y)
        total_nll += float(nll.masked_select(mask).sum().item())
        count += int(mask.sum().item())
    model.train(was_training)
    after_sha = _tensor_state_sha256(model)
    if before_sha != after_sha:
        raise RuntimeError("evaluation mutated model parameters")
    if count <= 0:
        raise RuntimeError("held-out panel emitted no content targets")
    return {
        "content_targets": count,
        "nll_nats": total_nll,
        "bits_per_byte": total_nll / (count * math.log(2.0)),
        "model_sha256_before": before_sha,
        "model_sha256_after": after_sha,
        "non_mutating": True,
        "training_mode_restored": model.training == was_training,
    }


def _next_log_probs(model: TwelveSixDecoder, context: Sequence[int]) -> torch.Tensor:
    if not context:
        raise ValueError("context cannot be empty")
    with torch.no_grad():
        was_training = model.training
        model.eval()
        x = torch.tensor([list(context)], dtype=torch.long)
        logits = model(x).logits[0, -1].float()
        model.train(was_training)
        return torch.log_softmax(logits, dim=-1)


def boundary_metrics(model: TwelveSixDecoder, candidate: str, records: Sequence[CorpusRow]) -> dict[str, Any]:
    if candidate == "A":
        return {
            "status": "NOT_APPLICABLE_NO_EOS",
            "document_start_prediction": None,
            "document_end_prediction": None,
            "cross_document_attention_influence": None,
        }
    tokenizer = tokenizer_for(candidate)
    docs = [row for row in records if row.text.encode("utf-8")][:BOUNDARY_DOCS]
    if len(docs) < 2:
        raise RuntimeError("boundary panel requires at least two documents")
    eos_id = tokenizer.eos_id
    assert eos_id == EXPERIMENTAL_EOS_ID

    end_bits: list[float] = []
    end_top1 = 0
    start_bits: list[float] = []
    start_top1 = 0
    contam_delta_bits: list[float] = []
    contam_abs_logp_bits: list[float] = []
    contam_top1_disagree = 0

    for row in docs:
        byte_ids = tokenizer.encode(row.text)
        tail = byte_ids[-min(len(byte_ids), 64):]
        lp = _next_log_probs(model, tail)
        end_bits.append(float(-lp[eos_id].item() / math.log(2.0)))
        end_top1 += int(int(torch.argmax(lp).item()) == eos_id)

        first = byte_ids[0]
        start_lp = _next_log_probs(model, [eos_id])
        start_bits.append(float(-start_lp[first].item() / math.log(2.0)))
        start_top1 += int(int(torch.argmax(start_lp).item()) == first)

    for prev, nxt in zip(docs[:-1], docs[1:]):
        prev_bytes = tokenizer.encode(prev.text)
        next_bytes = tokenizer.encode(nxt.text)
        target = next_bytes[0]
        eos_only = _next_log_probs(model, [eos_id])
        with_prev = _next_log_probs(model, prev_bytes[-63:] + [eos_id])
        nll_eos = float(-eos_only[target].item() / math.log(2.0))
        nll_prev = float(-with_prev[target].item() / math.log(2.0))
        contam_delta_bits.append(nll_prev - nll_eos)
        contam_abs_logp_bits.append(
            abs(float((with_prev[target] - eos_only[target]).item() / math.log(2.0)))
        )
        contam_top1_disagree += int(
            int(torch.argmax(eos_only).item()) != int(torch.argmax(with_prev).item())
        )

    return {
        "status": "MEASURED",
        "documents": len(docs),
        "uniform_eos_baseline_bits": math.log2(model.spec.vocab_size),
        "document_end_prediction": {
            "eos_bits_per_boundary": mean(end_bits),
            "eos_top1_accuracy": end_top1 / len(docs),
        },
        "document_start_prediction": {
            "first_byte_bits_after_eos": mean(start_bits),
            "first_byte_top1_accuracy_after_eos": start_top1 / len(docs),
        },
        "cross_document_attention_influence": {
            "definition": "change in next-document first-byte prediction when prior-document tail is visible before EOS versus EOS-only context; EOS is a semantic marker, not an attention reset",
            "mean_signed_bpb_delta_with_prior_context": mean(contam_delta_bits),
            "mean_absolute_target_logprob_delta_bits": mean(contam_abs_logp_bits),
            "top1_disagreement_rate": contam_top1_disagree / len(contam_delta_bits),
            "pairs": len(contam_delta_bits),
        },
    }


def generation_snapshots(model: TwelveSixDecoder, candidate: str) -> dict[str, Any]:
    tokenizer = tokenizer_for(candidate)
    backend = S0TorchInferenceBackend(model, tokenizer)
    backend.eos_token_id = tokenizer.eos_id
    prompts = [
        "Ukraine is",
        "Україна — це",
        "def add(a, b):",
        "The model predicts",
        "Київ",
        "for item in items:",
    ]
    outputs = []
    eos_terminations = 0
    for prompt in prompts:
        result = generate(
            backend,
            prompt,
            GenerationConfig(max_new_tokens=48, sample=False, seed=SEED),
        )
        eos_terminations += int(result.stop_reason == "eos")
        outputs.append(
            {
                "prompt": prompt,
                "generated_token_ids": list(result.generated_token_ids),
                "text": result.text,
                "stop_reason": result.stop_reason,
            }
        )
    return {
        "raw_greedy": outputs,
        "eos_terminations": eos_terminations,
        "prompts": len(prompts),
        "eos_termination_rate": eos_terminations / len(prompts),
    }


def trainer_config(max_steps: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=max_steps,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def packing_identity(candidate: str, plan: PreparedPlan) -> dict[str, Any]:
    payload = {
        "incumbent_packing_version": plan.stats["packing_version"],
        "candidate": candidate,
        "add_eos": CANDIDATES[candidate]["add_eos"],
        "cross_document": CANDIDATES[candidate]["cross_document"],
        "sequence_length": SEQ_LEN,
        "source_slice_identity_sha256": plan.stats["source_slice"]["identity_sha256"],
    }
    return {"payload": payload, "sha256": hash_json(payload)}


def run_identity(source_sha: str, scale: str, candidate: str, plan: PreparedPlan, phase: str) -> dict[str, Any]:
    spec = model_spec(scale, candidate)
    tok = tokenizer_for(candidate)
    packing = packing_identity(candidate, plan)
    return {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "phase": phase,
        "scale": scale,
        "candidate": candidate,
        "model_spec_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "tokenizer_version": tok.identity.version,
        "tokenizer_config_sha256": tok.identity.config_sha256,
        "tokenizer_vocab_sha256": tok.identity.vocab_sha256,
        "corpus_identity_sha256": EXPECTED_CORPUS_ID,
        "source_slice_identity_sha256": plan.stats["source_slice"]["identity_sha256"],
        "packing_sha256": packing["sha256"],
        "seed": SEED,
        "precision": "fp32",
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
    }


def checkpoint_identity(
    source_sha: str,
    model: TwelveSixDecoder,
    trainer: Trainer,
    candidate: str,
    plan: PreparedPlan,
    run_id: Mapping[str, Any],
) -> CheckpointIdentity:
    tok = tokenizer_for(candidate)
    init = InitSpec()
    packing = packing_identity(candidate, plan)
    cfg = trainer.config
    runtime_hash = sha256_file(RUNTIME_LOCK) if RUNTIME_LOCK.exists() else None
    training_config = {
        "init_spec_sha256": init.identity_sha256(),
        "data": {
            "corpus_identity_sha256": EXPECTED_CORPUS_ID,
            "split_identity": plan.stats["source_slice"]["identity_sha256"],
            "packing_sha256": packing["sha256"],
            "packing_version": plan.stats["packing_version"],
            "tokenizer_version": tok.identity.version,
            "eos_id": tok.eos_id,
        },
        "training": {
            "context_length": model.spec.max_seq_len,
            "sequence_length": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "betas": list(cfg.betas),
            "eps": cfg.eps,
            "gradient_clip_norm": cfg.gradient_clip_norm,
            "max_steps": cfg.max_steps,
            "precision": cfg.precision,
        },
    }
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=model.spec.to_dict(),
        parameter_count=model.spec.parameter_count(),
        tokenizer_hash=tok.identity.config_sha256,
        tokenizer_vocab_hash=tok.identity.vocab_sha256,
        dataset_manifest_hash=EXPECTED_CORPUS_ID,
        run_manifest_hash=hash_json(dict(run_id)),
        training_config=training_config,
        seed=SEED,
        precision="fp32",
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": cfg.learning_rate,
            "betas": list(cfg.betas),
            "eps": cfg.eps,
            "weight_decay": cfg.weight_decay,
        },
        scheduler=None,
        environment_lock_hash=runtime_hash,
    )


def save_ckpt(
    path: Path,
    source_sha: str,
    model: TwelveSixDecoder,
    trainer: Trainer,
    candidate: str,
    plan: PreparedPlan,
    run_id: Mapping[str, Any],
) -> dict[str, Any]:
    shutil.rmtree(path, ignore_errors=True)
    identity = checkpoint_identity(source_sha, model, trainer, candidate, plan, run_id)
    save_trainer_checkpoint(path, model=model, trainer=trainer, identity=identity)
    manifest = verify_checkpoint(path)
    return {
        "path": str(path),
        "checkpoint_id": manifest["checkpoint_id"],
        "step": manifest["identity"]["step"],
        "tokens_seen": manifest["identity"]["tokens_seen"],
        "parameter_count": manifest["identity"]["parameter_count"],
        "model_spec_sha256": manifest["identity"]["model_spec_hash"],
        "tokenizer_config_sha256": manifest["identity"]["tokenizer_hash"],
        "tokenizer_vocab_sha256": manifest["identity"]["tokenizer_vocab_hash"],
        "manifest_sha256": sha256_file(path / "manifest.json"),
        "manifest_checksum_sha256": sha256_file(path / "MANIFEST.sha256"),
    }


def load_ckpt(
    path: Path,
    source_sha: str,
    model: TwelveSixDecoder,
    trainer: Trainer,
    candidate: str,
    plan: PreparedPlan,
    run_id: Mapping[str, Any],
):
    tok = tokenizer_for(candidate)
    packing = packing_identity(candidate, plan)
    runtime_hash = sha256_file(RUNTIME_LOCK) if RUNTIME_LOCK.exists() else None
    return load_trainer_checkpoint(
        path,
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=source_sha,
        expected_model_spec_hash=model.spec.identity_sha256(),
        expected_init_spec_hash=InitSpec().identity_sha256(),
        expected_tokenizer_hash=tok.identity.config_sha256,
        expected_tokenizer_vocab_hash=tok.identity.vocab_sha256,
        expected_dataset_manifest_hash=EXPECTED_CORPUS_ID,
        expected_split_identity=plan.stats["source_slice"]["identity_sha256"],
        expected_packing_hash=packing["sha256"],
        expected_packing_version=plan.stats["packing_version"],
        expected_run_manifest_hash=hash_json(dict(run_id)),
        expected_environment_lock_hash=runtime_hash,
        expected_seed=SEED,
    )


def train_range(
    model: TwelveSixDecoder,
    trainer: Trainer,
    plan: PreparedPlan,
    observer: TrainingObserver,
    start_step: int,
    end_step: int,
) -> None:
    if trainer.optimizer_step != start_step:
        raise RuntimeError(f"trainer step {trainer.optimizer_step} != requested start {start_step}")
    for index in range(start_step, end_step):
        observer.train_microbatch(
            trainer,
            plan.batches[index],
            data_wait_seconds=0.0,
        )
    if trainer.optimizer_step != end_step:
        raise RuntimeError(f"trainer ended at {trainer.optimizer_step}, expected {end_step}")


def _run_one_matched(
    out: Path,
    source_sha: str,
    scale: str,
    candidate: str,
    train_records: Sequence[CorpusRow],
    train_panel: Sequence[CorpusRow],
    val_panel: Sequence[CorpusRow],
) -> tuple[dict[str, Any], TwelveSixDecoder, Trainer, PreparedPlan, dict[str, Any]]:
    plan = prepare_plan(candidate, train_records)
    model, spec, init = build_model(scale, candidate)
    config = trainer_config(len(plan.batches))
    trainer = Trainer(model, config, device="cpu")
    rid = run_identity(source_sha, scale, candidate, plan, "matched")
    observer = TrainingObserver(rid, device="cpu", max_step_samples=2048)
    init_state = _tensor_state_sha256(model)
    initial_eval = evaluate_content_bpb(model, val_panel)
    initial_train_eval = evaluate_content_bpb(model, train_panel)
    generation_before = generation_snapshots(model, candidate)
    train_range(model, trainer, plan, observer, 0, len(plan.batches))
    final_eval = evaluate_content_bpb(model, val_panel)
    final_train_eval = evaluate_content_bpb(model, train_panel)
    boundaries = boundary_metrics(model, candidate, val_panel)
    generation_after = generation_snapshots(model, candidate)
    telemetry = observer.summary()
    result = {
        "scale": scale,
        "candidate": candidate,
        "label": CANDIDATES[candidate]["label"],
        "random_initialization": {
            "seed": SEED,
            "init_spec": init.to_dict(),
            "init_spec_sha256": init.identity_sha256(),
            "initial_model_state_sha256": init_state,
            "foreign_pretrained_weights": False,
        },
        "model": {
            "spec": spec.to_dict(),
            "model_spec_sha256": spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
        },
        "tokenizer": {
            "version": tokenizer_for(candidate).identity.version,
            "config_sha256": tokenizer_for(candidate).identity.config_sha256,
            "vocab_sha256": tokenizer_for(candidate).identity.vocab_sha256,
            "vocab_size": tokenizer_for(candidate).vocab_size,
            "eos_id": tokenizer_for(candidate).eos_id,
            "bos_id": tokenizer_for(candidate).bos_id,
            "pad_id": tokenizer_for(candidate).pad_id,
        },
        "packing": plan.stats,
        "train_panel": {
            "initial_bpb": initial_train_eval["bits_per_byte"],
            "final_bpb": final_train_eval["bits_per_byte"],
            "decrease": initial_train_eval["bits_per_byte"] - final_train_eval["bits_per_byte"],
            "non_mutating": initial_train_eval["non_mutating"] and final_train_eval["non_mutating"],
        },
        "heldout": {
            "initial_bpb": initial_eval["bits_per_byte"],
            "final_bpb": final_eval["bits_per_byte"],
            "decrease": initial_eval["bits_per_byte"] - final_eval["bits_per_byte"],
            "relative_improvement": 1.0 - final_eval["bits_per_byte"] / initial_eval["bits_per_byte"],
            "content_targets": final_eval["content_targets"],
            "non_mutating": initial_eval["non_mutating"] and final_eval["non_mutating"],
        },
        "boundary_metrics": boundaries,
        "generation_before": generation_before,
        "generation_after": generation_after,
        "training": {
            "optimizer_steps": trainer.optimizer_step,
            "tokens_seen_including_eos": trainer.tokens_seen,
            "observer": telemetry,
            "source_content_tokens_per_second": (
                plan.stats["content_targets"] / telemetry["timing"]["training_observed_seconds"]
            ),
        },
    }
    if result["heldout"]["decrease"] <= 0:
        raise RuntimeError(f"{scale}/{candidate} held-out BPB failed to improve")
    if result["train_panel"]["decrease"] <= 0:
        raise RuntimeError(f"{scale}/{candidate} train-panel BPB failed to improve")
    return result, model, trainer, plan, rid


def _incompatibility_probe(
    source_sha: str,
    checkpoint_a: Path,
    checkpoint_b: Path,
    plan_a: PreparedPlan,
    plan_b: PreparedPlan,
    rid_a: Mapping[str, Any],
    rid_b: Mapping[str, Any],
) -> dict[str, Any]:
    # rid_a/rid_b are retained in the function contract so callers prove these
    # checkpoints came from the matched experiment identities.
    if not rid_a or not rid_b:
        raise RuntimeError("matched run identities are required for incompatibility proof")
    probes = []
    for source_path, target_candidate, target_plan in (
        (checkpoint_a, "B", plan_b),
        (checkpoint_b, "A", plan_a),
    ):
        model, _, _ = build_model("268k", target_candidate)
        trainer = Trainer(model, trainer_config(len(target_plan.batches)), device="cpu")
        before = _tensor_state_sha256(model)
        rejected = False
        error = None
        target_tok = tokenizer_for(target_candidate)
        try:
            # Deliberately bind only the target tokenizer identities here so the
            # failure is attributable to the special-token vocabulary contract,
            # not to the model-shape or run-manifest mismatch that follows it.
            load_trainer_checkpoint(
                source_path,
                model=model,
                trainer=trainer,
                strict_model=True,
                restore_rng=False,
                expected_git_sha=source_sha,
                expected_tokenizer_hash=target_tok.identity.config_sha256,
                expected_tokenizer_vocab_hash=target_tok.identity.vocab_sha256,
                expected_dataset_manifest_hash=EXPECTED_CORPUS_ID,
            )
        except CheckpointCompatibilityError as exc:
            rejected = True
            error = str(exc)
        after = _tensor_state_sha256(model)
        if not rejected or before != after:
            raise RuntimeError(
                f"incompatible checkpoint load failed closed={rejected}, nonmutation={before == after}"
            )
        probes.append(
            {
                "source_checkpoint": str(source_path),
                "target_candidate": target_candidate,
                "rejected_before_model_mutation": rejected,
                "target_model_state_unchanged": before == after,
                "error": error,
            }
        )
    return {
        "status": "PASS",
        "probes": probes,
        "claim": "vocab-256 and vocab-257 checkpoint/tokenizer identities reject each other before model mutation",
    }


def parameter_tax() -> dict[str, Any]:
    output = {}
    for scale in SCALES:
        a = model_spec(scale, "A").parameter_count()
        b = model_spec(scale, "B").parameter_count()
        output[scale] = {
            "vocab_256_parameters": a,
            "vocab_257_parameters": b,
            "extra_parameters_for_eos": b - a,
            "expected_extra_parameters_equal_d_model": SCALES[scale]["d_model"],
            "relative_tax": (b - a) / a,
        }
        if b - a != SCALES[scale]["d_model"]:
            raise RuntimeError("EOS parameter tax is not exactly one tied embedding row")
    return output


def choose_candidate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    one_m = [item for item in results if item["scale"] == "1m"]
    if len(one_m) != 3:
        raise RuntimeError("selection requires A/B/C at 1m")
    ranked = sorted(one_m, key=lambda item: (item["heldout"]["final_bpb"], item["candidate"]))
    winner = ranked[0]
    a = next(item for item in one_m if item["candidate"] == "A")
    eos_ranked = sorted(
        [item for item in one_m if item["candidate"] in {"B", "C"}],
        key=lambda item: item["heldout"]["final_bpb"],
    )
    eos_best = eos_ranked[0]
    eos_boundary = eos_best["boundary_metrics"]["document_end_prediction"]
    uniform = eos_best["boundary_metrics"]["uniform_eos_baseline_bits"]
    within_one_percent = eos_best["heldout"]["final_bpb"] <= a["heldout"]["final_bpb"] * 1.01
    learned_end = eos_boundary["eos_bits_per_boundary"] < uniform
    provisional = within_one_percent and learned_end
    return {
        "artifact_candidate": winner["candidate"],
        "artifact_reason": "lowest matched ~1M common-heldout content BPB",
        "ranking_1m": [
            {
                "candidate": item["candidate"],
                "heldout_final_bpb": item["heldout"]["final_bpb"],
                "packing_efficiency": item["packing"]["packing_efficiency_valid_targets"],
            }
            for item in ranked
        ],
        "eos_boundary_decision_pre_substantial": (
            "PROVISIONAL_EOS_BOUNDARY_CONTRACT" if provisional else "DO_NOT_PROMOTE_EOS_YET"
        ),
        "eos_best_candidate": eos_best["candidate"],
        "eos_vs_A_bpb_ratio": eos_best["heldout"]["final_bpb"] / a["heldout"]["final_bpb"],
        "eos_end_prediction_better_than_uniform": learned_end,
        "rule": "EOS is provisionally acceptable only if the best EOS candidate is within 1% of A on common held-out content BPB and predicts EOS at document ends better than the uniform-vocabulary baseline. Final decision is rechecked after the substantial run.",
    }


def phase_matched(out: Path, source_sha: str) -> None:
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    built_manifest, rows = ensure_corpus()
    ordered_train = _weighted_order(rows, "train")
    ordered_val = _weighted_order(rows, "validation")
    train_records = select_source_slice(ordered_train, MATCHED_SOURCE_BYTES)
    train_panel = select_source_slice(ordered_train, min(HELDOUT_PANEL_BYTES, MATCHED_SOURCE_BYTES))
    val_panel = select_source_slice(ordered_val, HELDOUT_PANEL_BYTES)

    results = []
    retained: dict[str, tuple[TwelveSixDecoder, Trainer, PreparedPlan, dict[str, Any]]] = {}
    for scale in ("268k", "1m"):
        for candidate in ("A", "B", "C"):
            print(f"TOK115 matched start {scale}/{candidate}", flush=True)
            result, model, trainer, plan, rid = _run_one_matched(
                out, source_sha, scale, candidate, train_records, train_panel, val_panel
            )
            results.append(result)
            if scale == "268k" and candidate in {"A", "B"}:
                retained[candidate] = (model, trainer, plan, rid)
            print(
                json.dumps(
                    {
                        "scale": scale,
                        "candidate": candidate,
                        "heldout_initial_bpb": result["heldout"]["initial_bpb"],
                        "heldout_final_bpb": result["heldout"]["final_bpb"],
                        "train_final_bpb": result["train_panel"]["final_bpb"],
                        "packing_efficiency": result["packing"]["packing_efficiency_valid_targets"],
                        "tokens_per_second": result["training"]["observer"]["throughput"]["train_tokens_per_second"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    checkpoint_a = out / "matched-checkpoints" / "268k-A"
    checkpoint_b = out / "matched-checkpoints" / "268k-B"
    meta_a = save_ckpt(checkpoint_a, source_sha, *retained["A"][:2], "A", retained["A"][2], retained["A"][3])
    meta_b = save_ckpt(checkpoint_b, source_sha, *retained["B"][:2], "B", retained["B"][2], retained["B"][3])
    incompat = _incompatibility_probe(
        source_sha,
        checkpoint_a,
        checkpoint_b,
        retained["A"][2],
        retained["B"][2],
        retained["A"][3],
        retained["B"][3],
    )
    selection = choose_candidate(results)
    payload = {
        "schema": SCHEMA,
        "worker_id": "TOK-115-EOS-REAL-CORPUS",
        "source_sha": source_sha,
        "corpus": {
            "identity_sha256": built_manifest["corpus_identity_sha256"],
            "version": built_manifest["corpus_version"],
            "train": built_manifest["by_split"]["train"],
            "validation": built_manifest["by_split"]["validation"],
            "truth_boundary": built_manifest["truth_boundary"],
            "external_training_eligible_sources": built_manifest["external_training_eligible_sources"],
        },
        "matched_source_slice": source_slice_stats(train_records),
        "heldout_panel": source_slice_stats(val_panel),
        "parameter_tax": parameter_tax(),
        "results": results,
        "checkpoint_identity_rejection": incompat,
        "retained_matched_checkpoints": {"A": meta_a, "B": meta_b},
        "selection": selection,
        "constraints": {
            "foreign_pretrained_weights": False,
            "instruction_tuning": False,
            "paid_compute": False,
            "bos": False,
            "pad_semantic": False,
            "chat_or_system_semantics": False,
        },
    }
    _write_json(out / "matched.json", payload)
    _write_json(out / "selection.json", selection)
    print(json.dumps(selection, indent=2, sort_keys=True), flush=True)


def _substantial_context(out: Path):
    matched = _json(out / "matched.json")
    selection = _json(out / "selection.json")
    candidate = selection["artifact_candidate"]
    built_manifest, rows = ensure_corpus()
    ordered_train = _weighted_order(rows, "train")
    ordered_val = _weighted_order(rows, "validation")
    train_records = select_source_slice(ordered_train, SUBSTANTIAL_SOURCE_BYTES)
    train_panel = select_source_slice(ordered_train, HELDOUT_PANEL_BYTES)
    val_panel = select_source_slice(ordered_val, HELDOUT_PANEL_BYTES)
    plan = prepare_plan(candidate, train_records)
    return matched, selection, candidate, built_manifest, plan, train_panel, val_panel


def phase_start(out: Path, source_sha: str) -> None:
    _, _, candidate, built_manifest, plan, train_panel, val_panel = _substantial_context(out)
    model, spec, init = build_model("1m", candidate)
    config = trainer_config(len(plan.batches))
    trainer = Trainer(model, config, device="cpu")
    rid = run_identity(source_sha, "1m", candidate, plan, "substantial")
    observer = TrainingObserver(rid, device="cpu", max_step_samples=4096)

    initial_state = _tensor_state_sha256(model)
    initial_heldout = evaluate_content_bpb(model, val_panel)
    initial_train = evaluate_content_bpb(model, train_panel)
    generation_before = generation_snapshots(model, candidate)

    quarter = max(1, len(plan.batches) // 4)
    midpoint = max(quarter + 1, len(plan.batches) // 2)
    if midpoint >= len(plan.batches):
        midpoint = len(plan.batches) - 1
    train_range(model, trainer, plan, observer, 0, quarter)
    q_meta = save_ckpt(out / "checkpoints" / "quarter", source_sha, model, trainer, candidate, plan, rid)
    train_range(model, trainer, plan, observer, quarter, midpoint)
    m_meta = save_ckpt(out / "checkpoints" / "midpoint", source_sha, model, trainer, candidate, plan, rid)

    state = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "process": {"pid": os.getpid(), "phase": "start"},
        "candidate": candidate,
        "scale": "1m",
        "model": {
            "spec": spec.to_dict(),
            "model_spec_sha256": spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "init_spec": init.to_dict(),
            "init_spec_sha256": init.identity_sha256(),
            "initial_state_sha256": initial_state,
        },
        "packing": plan.stats,
        "initial_heldout": initial_heldout,
        "initial_train_panel": initial_train,
        "generation_before": generation_before,
        "quarter_step": quarter,
        "midpoint_step": midpoint,
        "total_steps": len(plan.batches),
        "quarter_checkpoint": q_meta,
        "midpoint_checkpoint": m_meta,
        "pre_resume_telemetry": observer.summary(),
        "corpus_identity_sha256": built_manifest["corpus_identity_sha256"],
        "run_identity": rid,
        "run_identity_sha256": hash_json(rid),
    }
    _write_json(out / "substantial-start.json", state)
    print(json.dumps({
        "candidate": candidate,
        "parameter_count": spec.parameter_count(),
        "quarter_step": quarter,
        "midpoint_step": midpoint,
        "total_steps": len(plan.batches),
        "initial_heldout_bpb": initial_heldout["bits_per_byte"],
    }, indent=2, sort_keys=True), flush=True)


def machine_manifest(source_sha: str) -> dict[str, Any]:
    return {
        "schema": "12-6.machine-manifest.v1",
        "source_sha": source_sha,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "github_actions": os.environ.get("GITHUB_ACTIONS") == "true",
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "paid_compute": False,
        "runtime_lock_sha256": sha256_file(RUNTIME_LOCK) if RUNTIME_LOCK.exists() else None,
    }


def _final_eos_decision(matched: Mapping[str, Any], final_candidate: str, final_boundaries: Mapping[str, Any]) -> dict[str, Any]:
    pre = matched["selection"]
    if final_candidate == "A":
        return {
            "decision": "DO_NOT_PROMOTE_EOS_YET",
            "reason": "candidate A won the matched ~1M common-heldout BPB comparison; final retained learned artifact is byte-only document-isolated",
            "scope": "DATA-25 project-authored UK/EN/code corpus only",
        }
    end = final_boundaries["document_end_prediction"]
    learned_end = end["eos_bits_per_boundary"] < final_boundaries["uniform_eos_baseline_bits"]
    decision = "PROVISIONAL_EOS_BOUNDARY_CONTRACT" if learned_end else "DO_NOT_PROMOTE_EOS_YET"
    return {
        "decision": decision,
        "reason": (
            "matched selection chose an EOS candidate and the substantial final model predicts EOS at true document ends better than uniform"
            if learned_end
            else "matched selection chose EOS but substantial final document-end prediction did not beat the uniform-vocabulary baseline"
        ),
        "selected_eos_packing": CANDIDATES[final_candidate]["label"],
        "scope": "DATA-25 project-authored UK/EN/code corpus only; not an external-corpus or universal architecture claim",
        "no_attention_reset_claim": True,
        "matched_predecision": pre["eos_boundary_decision_pre_substantial"],
    }


def phase_resume(out: Path, source_sha: str) -> None:
    matched, _, candidate, built_manifest, plan, train_panel, val_panel = _substantial_context(out)
    start = _json(out / "substantial-start.json")
    if start["source_sha"] != source_sha or start["candidate"] != candidate:
        raise RuntimeError("substantial start identity mismatch")
    if start["process"]["pid"] == os.getpid():
        raise RuntimeError("resume must execute in a fresh Python process")

    model, spec, init = build_model("1m", candidate)
    trainer = Trainer(model, trainer_config(len(plan.batches)), device="cpu")
    rid = start["run_identity"]
    fresh_preload_state = _tensor_state_sha256(model)
    load_result = load_ckpt(
        out / "checkpoints" / "midpoint",
        source_sha,
        model,
        trainer,
        candidate,
        plan,
        rid,
    )
    midpoint = int(start["midpoint_step"])
    if trainer.optimizer_step != midpoint:
        raise RuntimeError("fresh resume restored wrong optimizer step")
    restored_state = _tensor_state_sha256(model)

    observer = TrainingObserver(rid, device="cpu", max_step_samples=4096)
    train_range(model, trainer, plan, observer, midpoint, len(plan.batches))
    in_memory_final_sha = _tensor_state_sha256(model)
    final_meta = save_ckpt(out / "checkpoints" / "final", source_sha, model, trainer, candidate, plan, rid)

    # Fresh exact final reload before authoritative final evaluation and generation.
    final_model, _, _ = build_model("1m", candidate)
    final_trainer = Trainer(final_model, trainer_config(len(plan.batches)), device="cpu")
    final_preload_sha = _tensor_state_sha256(final_model)
    load_ckpt(
        out / "checkpoints" / "final",
        source_sha,
        final_model,
        final_trainer,
        candidate,
        plan,
        rid,
    )
    final_reload_sha = _tensor_state_sha256(final_model)
    if final_reload_sha != in_memory_final_sha:
        raise RuntimeError("retained final checkpoint reload does not reproduce final model state")
    if final_trainer.optimizer_step != len(plan.batches):
        raise RuntimeError("final checkpoint restored wrong optimizer step")

    final_heldout = evaluate_content_bpb(final_model, val_panel)
    final_train = evaluate_content_bpb(final_model, train_panel)
    boundaries = boundary_metrics(final_model, candidate, val_panel)
    generation_after = generation_snapshots(final_model, candidate)

    if final_heldout["bits_per_byte"] >= start["initial_heldout"]["bits_per_byte"]:
        raise RuntimeError("substantial final held-out BPB did not improve from random initialization")
    if final_train["bits_per_byte"] >= start["initial_train_panel"]["bits_per_byte"]:
        raise RuntimeError("substantial final train-panel BPB did not improve from random initialization")

    decision = _final_eos_decision(matched, candidate, boundaries)
    manifest = machine_manifest(source_sha)
    _write_json(out / "machine-manifest.json", manifest)
    reproduction = (
        "PYTHONPATH=src python tools/run_tok115_eos_real_corpus.py --phase matched "
        f"--output-dir {out} --source-sha {source_sha} && "
        "PYTHONPATH=src python tools/run_tok115_eos_real_corpus.py --phase start "
        f"--output-dir {out} --source-sha {source_sha} && "
        "PYTHONPATH=src python tools/run_tok115_eos_real_corpus.py --phase resume "
        f"--output-dir {out} --source-sha {source_sha}"
    )
    final = {
        "schema": SCHEMA,
        "worker_ids": [
            "MILESTONE-100-FIRST-LEARNED-BASE",
            "TOK-115-EOS-REAL-CORPUS",
        ],
        "source_sha": source_sha,
        "exact_experimental_branch": "tok115/eos-real-corpus-20260826",
        "lineage": {
            "product_exact_green_parent": "fb9c6d9b73ce436d637077892d73edf136fcaeac",
            "data25_parent": "8af17afa7baf3d75c2328caf8b08af2400a95e09",
            "tok39_parent": "03ca7997cee01099a195d42fb264a3e35af0b751",
        },
        "selected_candidate": candidate,
        "selected_label": CANDIDATES[candidate]["label"],
        "model": {
            "spec": spec.to_dict(),
            "model_spec_sha256": spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "init_spec": init.to_dict(),
            "init_spec_sha256": init.identity_sha256(),
            "random_initialization": True,
            "foreign_pretrained_weights": False,
        },
        "tokenizer": {
            "version": tokenizer_for(candidate).identity.version,
            "config_sha256": tokenizer_for(candidate).identity.config_sha256,
            "vocab_sha256": tokenizer_for(candidate).identity.vocab_sha256,
            "vocab_size": tokenizer_for(candidate).vocab_size,
            "eos_id": tokenizer_for(candidate).eos_id,
            "bos_id": tokenizer_for(candidate).bos_id,
            "pad_id": tokenizer_for(candidate).pad_id,
            "chat_or_system_semantics": False,
        },
        "corpus": {
            "identity_sha256": built_manifest["corpus_identity_sha256"],
            "train": built_manifest["by_split"]["train"],
            "validation": built_manifest["by_split"]["validation"],
            "truth_boundary": built_manifest["truth_boundary"],
            "external_training_eligible_sources": built_manifest["external_training_eligible_sources"],
            "substantial_source_slice": plan.stats["source_slice"],
        },
        "packing": plan.stats,
        "learning_proof": {
            "initial_train_panel_bpb": start["initial_train_panel"]["bits_per_byte"],
            "final_train_panel_bpb": final_train["bits_per_byte"],
            "train_panel_bpb_decrease": start["initial_train_panel"]["bits_per_byte"] - final_train["bits_per_byte"],
            "initial_heldout_bpb": start["initial_heldout"]["bits_per_byte"],
            "final_heldout_bpb": final_heldout["bits_per_byte"],
            "heldout_bpb_decrease": start["initial_heldout"]["bits_per_byte"] - final_heldout["bits_per_byte"],
            "heldout_relative_improvement": 1.0 - final_heldout["bits_per_byte"] / start["initial_heldout"]["bits_per_byte"],
            "evaluation_non_mutating": start["initial_heldout"]["non_mutating"] and final_heldout["non_mutating"],
        },
        "training": {
            "quarter_step": start["quarter_step"],
            "midpoint_step": midpoint,
            "final_step": final_trainer.optimizer_step,
            "final_tokens_seen_including_eos": final_trainer.tokens_seen,
            "pre_resume_observer": start["pre_resume_telemetry"],
            "post_resume_observer": observer.summary(),
        },
        "fresh_process_resume": {
            "start_pid": start["process"]["pid"],
            "resume_pid": os.getpid(),
            "different_process": start["process"]["pid"] != os.getpid(),
            "fresh_random_state_before_midpoint_load_sha256": fresh_preload_state,
            "restored_midpoint_model_state_sha256": restored_state,
            "restored_optimizer_step": midpoint,
            "verified_manifest_checkpoint_id": load_result.manifest["checkpoint_id"],
        },
        "checkpoints": {
            "quarter": start["quarter_checkpoint"],
            "midpoint": start["midpoint_checkpoint"],
            "final": final_meta,
            "final_fresh_reload": {
                "fresh_preload_state_sha256": final_preload_sha,
                "loaded_state_sha256": final_reload_sha,
                "matches_in_memory_final_state": final_reload_sha == in_memory_final_sha,
                "optimizer_step": final_trainer.optimizer_step,
            },
        },
        "generation_before": start["generation_before"],
        "generation_after": generation_after,
        "boundary_metrics": boundaries,
        "matched_experiment": matched,
        "eos_decision": decision,
        "machine_manifest": manifest,
        "reproduction_command": reproduction,
        "truth_boundary": {
            "local_free_only": True,
            "paid_compute": False,
            "instruction_tuning": False,
            "broad_intelligence_claim": False,
            "representative_claim": "Representative only across the DATA-25 intended project-authored UK/EN/code modalities for this local small-model boundary experiment; not representative of external real-world corpora.",
        },
    }
    _write_json(out / "final-report.json", final)
    print(
        json.dumps(
            {
                "selected_candidate": candidate,
                "parameter_count": spec.parameter_count(),
                "initial_heldout_bpb": final["learning_proof"]["initial_heldout_bpb"],
                "final_heldout_bpb": final["learning_proof"]["final_heldout_bpb"],
                "heldout_bpb_decrease": final["learning_proof"]["heldout_bpb_decrease"],
                "initial_train_panel_bpb": final["learning_proof"]["initial_train_panel_bpb"],
                "final_train_panel_bpb": final["learning_proof"]["final_train_panel_bpb"],
                "fresh_process_resume": final["fresh_process_resume"]["different_process"],
                "final_checkpoint_id": final_meta["checkpoint_id"],
                "eos_decision": decision["decision"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("matched", "start", "resume"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    source_sha = args.source_sha.strip()
    if source_sha != _git_sha():
        raise RuntimeError(f"source SHA mismatch: {source_sha} != {_git_sha()}")
    if len(source_sha) != 40:
        raise RuntimeError("source SHA must be exact 40-hex Git identity")
    if args.phase == "matched":
        phase_matched(args.output_dir, source_sha)
    elif args.phase == "start":
        phase_start(args.output_dir, source_sha)
    else:
        phase_resume(args.output_dir, source_sha)


if __name__ == "__main__":
    main()
