"""Strict fixed-token model-efficiency experiment for the RESEARCH41 control family.

This is a narrow successor to ``twelve_six.scaling_experiment``.  It preserves
that experiment's model family, tokenizer, corpus split, initialization and AdamW
recipe while repairing one scientific-control defect: requested token checkpoints
must be reached exactly, not crossed by a whole shifted-loss microbatch.

Only valid aligned causal targets count as optimized tokens.  Evaluation never
mutates the trainer ledger.  Mid-run resume uses the incumbent D05 checkpoint
adapter and is intentionally executed by a separate CLI process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch

from .checkpoint.core import CheckpointIdentity, hash_json, sha256_file
from .checkpoint.trainer_adapter import load_trainer_checkpoint, save_trainer_checkpoint
from .model import InitSpec, TwelveSixDecoder
from .scaling_experiment import (
    _byte_stream,
    _read_jsonl,
    _validation_loss,
    controlled_specs,
)
from .tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from .training import Trainer, TrainerConfig

SCHEMA = "12-6.fixed-token-efficiency.v1"
MODEL_SCHEMA = "12-6.fixed-token-efficiency-model.v1"
PARTIAL_SCHEMA = "12-6.fixed-token-efficiency-partial.v1"
AUTHORITY = "LOCAL_FREE_FIXED_TOKEN_GENERALIZATION_EVIDENCE_NOT_PROMOTION"
DEFAULT_TOKEN_BUDGETS = (4_096, 16_384, 65_536)
DEFAULT_RESUME_TOKENS = 16_384
PACKING_VERSION = "research06-aligned-byte-causal-pairs-v1"
PACKING_DEFINITION = {
    "version": PACKING_VERSION,
    "stream": "research41 train byte stream",
    "pair": "input byte at causal offset t predicts byte at t+1",
    "layout": "row-major [batch, time]",
    "partial_step": "tail targets masked to hit requested optimized-token budget exactly",
    "evaluation_tokens_optimized": 0,
}
PACKING_SHA256 = hash_json(PACKING_DEFINITION)
COMPUTE_PROXY = "6 * trainable_parameters * optimized_causal_loss_tokens"
_EXPECTED_COUNTS = (95_568, 267_912, 467_808, 1_037_696)


def _canonical_hash(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("report_sha256", None)
    return hash_json(body)


def _write_hashed_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body["report_sha256"] = _canonical_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return body


def _read_hashed_json(path: Path, *, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != schema:
        raise RuntimeError(f"{path} schema mismatch: {payload.get('schema')!r}")
    recorded = payload.get("report_sha256")
    if not isinstance(recorded, str) or recorded != _canonical_hash(payload):
        raise RuntimeError(f"{path} self-hash mismatch")
    return payload


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _assert_exact_source(repo_root: Path, source_sha: str) -> None:
    if len(source_sha) != 40 or any(ch not in "0123456789abcdef" for ch in source_sha):
        raise ValueError("source_sha must be exact lowercase 40-hex")
    observed = _git_head(repo_root)
    if observed != source_sha:
        raise RuntimeError(f"exact-checkout mismatch: expected {source_sha}, observed {observed}")


def _trainer_config(*, final_tokens: int, batch_size: int, sequence_length: int, seed: int) -> TrainerConfig:
    capacity = batch_size * sequence_length
    return TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=math.ceil(final_tokens / capacity),
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _load_control_data(repo_root: Path) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_records = _read_jsonl(train_path)
    validation_records = _read_jsonl(validation_path)
    train_ids = {str(item["id"]) for item in train_records}
    validation_ids = {str(item["id"]) for item in validation_records}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise RuntimeError(f"train/validation record overlap: {overlap!r}")
    train_sha = sha256_file(train_path)
    validation_sha = sha256_file(validation_path)
    split_identity = hash_json(
        {
            "train_jsonl_sha256": train_sha,
            "validation_jsonl_sha256": validation_sha,
            "train_ids": sorted(train_ids),
            "validation_ids": sorted(validation_ids),
        }
    )
    return {
        "tokenizer": tokenizer,
        "train_records": train_records,
        "validation_records": validation_records,
        "train_stream": _byte_stream(train_records, tokenizer),
        "manifest_path": manifest_path,
        "dataset_manifest_hash": sha256_file(manifest_path),
        "train_jsonl_sha256": train_sha,
        "validation_jsonl_sha256": validation_sha,
        "split_identity": split_identity,
        "overlap": overlap,
    }


def _make_pair_batch(
    stream: bytes,
    *,
    causal_offset: int,
    batch_size: int,
    sequence_length: int,
    valid_pairs: int,
) -> dict[str, torch.Tensor]:
    """Build aligned next-byte pairs with an exact valid-target count.

    Invalid tail positions are after every valid position in row-major order, so
    causal attention at a valid position can never depend on an invalid filler.
    """
    if not stream:
        raise ValueError("training stream must be non-empty")
    capacity = batch_size * sequence_length
    if not 0 < valid_pairs <= capacity:
        raise ValueError(f"valid_pairs must be in [1,{capacity}]")
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    mask: list[list[int]] = []
    for row in range(batch_size):
        input_row: list[int] = []
        target_row: list[int] = []
        mask_row: list[int] = []
        for column in range(sequence_length):
            flat = row * sequence_length + column
            position = causal_offset + flat
            input_row.append(stream[position % len(stream)])
            target_row.append(stream[(position + 1) % len(stream)])
            mask_row.append(1 if flat < valid_pairs else 0)
        inputs.append(input_row)
        targets.append(target_row)
        mask.append(mask_row)
    return {
        "input_ids": torch.tensor(inputs, dtype=torch.long),
        "target_ids": torch.tensor(targets, dtype=torch.long),
        "loss_mask": torch.tensor(mask, dtype=torch.bool),
    }


def _parameter_snapshot(model: TwelveSixDecoder) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def _relative_update_ratio(model: TwelveSixDecoder, before: list[torch.Tensor]) -> float:
    parameters = list(model.parameters())
    if len(parameters) != len(before):
        raise RuntimeError("parameter topology drifted during update-ratio measurement")
    delta_sq = 0.0
    weight_sq = 0.0
    for parameter, old in zip(parameters, before, strict=True):
        current = parameter.detach()
        delta = current - old
        delta_sq += float(torch.sum(delta.double() * delta.double()).item())
        weight_sq += float(torch.sum(old.double() * old.double()).item())
    return math.sqrt(delta_sq) / max(math.sqrt(weight_sq), 1e-30)


def _parameter_bytes(model: TwelveSixDecoder) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def _tensor_tree_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_tree_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_bytes(item) for item in value)
    return 0


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def _control_bundle(
    *,
    repo_root: Path,
    source_sha: str,
    model_index: int,
    final_tokens: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
) -> dict[str, Any]:
    specs = controlled_specs()
    counts = tuple(spec.parameter_count() for spec in specs)
    if counts != _EXPECTED_COUNTS:
        raise RuntimeError(f"RESEARCH41 control-family drift: {counts!r}")
    if not 0 <= model_index < len(specs):
        raise ValueError(f"model_index must be in [0,{len(specs)-1}]")
    data = _load_control_data(repo_root)
    spec = specs[model_index]
    init_spec = InitSpec()
    trainer_config = _trainer_config(
        final_tokens=final_tokens,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed,
    )
    training_config = {
        "recipe": "research41-fixed-control-adamw-v1",
        "trainer": asdict(trainer_config),
        "init_spec_sha256": init_spec.identity_sha256(),
        "data": {
            "split_identity": data["split_identity"],
            "packing_sha256": PACKING_SHA256,
            "packing_version": PACKING_VERSION,
        },
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "token_budgets": list(DEFAULT_TOKEN_BUDGETS),
        "final_tokens": final_tokens,
    }
    run_manifest = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "model_identity_sha256": spec.identity_sha256(),
        "init_identity_sha256": init_spec.identity_sha256(),
        "tokenizer_hash": BYTE_TOKENIZER_HASH,
        "tokenizer_vocab_hash": BYTE_VOCAB_HASH,
        "dataset_manifest_hash": data["dataset_manifest_hash"],
        "training_config": training_config,
    }
    controls_hash = hash_json(run_manifest)
    return {
        "spec": spec,
        "init_spec": init_spec,
        "trainer_config": trainer_config,
        "training_config": training_config,
        "run_manifest_hash": controls_hash,
        "controls_hash": controls_hash,
        "data": data,
    }


def _checkpoint_identity(
    *,
    source_sha: str,
    bundle: dict[str, Any],
    trainer: Trainer,
) -> CheckpointIdentity:
    spec = bundle["spec"]
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        dataset_manifest_hash=bundle["data"]["dataset_manifest_hash"],
        run_manifest_hash=bundle["run_manifest_hash"],
        training_config=bundle["training_config"],
        seed=bundle["trainer_config"].seed,
        precision=bundle["trainer_config"].precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": bundle["trainer_config"].learning_rate,
            "betas": list(bundle["trainer_config"].betas),
            "eps": bundle["trainer_config"].eps,
            "weight_decay": bundle["trainer_config"].weight_decay,
        },
        scheduler=None,
    )


def _evaluate_checked(
    *,
    model: TwelveSixDecoder,
    trainer: Trainer,
    validation_records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    expected_validation_tokens: int | None = None,
) -> tuple[float, int, float]:
    before_tokens = trainer.tokens_seen
    before_step = trainer.optimizer_step
    started = time.perf_counter()
    loss, validation_tokens = _validation_loss(model, validation_records, tokenizer)
    elapsed = time.perf_counter() - started
    if trainer.tokens_seen != before_tokens or trainer.optimizer_step != before_step:
        raise RuntimeError("evaluation mutated optimized-token or optimizer-step accounting")
    if expected_validation_tokens is not None and validation_tokens != expected_validation_tokens:
        raise RuntimeError(
            f"validation target count drift: {validation_tokens} != {expected_validation_tokens}"
        )
    return loss, validation_tokens, elapsed


def _assert_token_transition(*, before: int, metrics_tokens: int, after: int, requested: int) -> None:
    if metrics_tokens != requested:
        raise RuntimeError(
            f"Trainer valid-causal-token count drift: metrics={metrics_tokens}, requested={requested}"
        )
    if after - before != requested:
        raise RuntimeError(
            f"optimized-token ledger drift: before={before}, after={after}, requested={requested}"
        )


def _train_until(
    *,
    model: TwelveSixDecoder,
    trainer: Trainer,
    bundle: dict[str, Any],
    stop_tokens: int,
    token_budgets: tuple[int, ...],
    state: dict[str, Any],
    batch_size: int,
    sequence_length: int,
) -> None:
    if stop_tokens <= trainer.tokens_seen:
        raise ValueError("stop_tokens must exceed current optimized-token count")
    capacity = batch_size * sequence_length
    stream = bundle["data"]["train_stream"]
    tokenizer = bundle["data"]["tokenizer"]
    validation_records = bundle["data"]["validation_records"]
    validation_tokens = int(state["validation_tokens"])

    while trainer.tokens_seen < stop_tokens:
        before_tokens = trainer.tokens_seen
        valid_pairs = min(capacity, stop_tokens - before_tokens)
        after_planned = before_tokens + valid_pairs
        sample_update = (
            trainer.optimizer_step == 0
            or (trainer.optimizer_step + 1) % 32 == 0
            or after_planned in token_budgets
            or after_planned == stop_tokens
        )
        snapshot = _parameter_snapshot(model) if sample_update else None
        batch = _make_pair_batch(
            stream,
            causal_offset=before_tokens,
            batch_size=batch_size,
            sequence_length=sequence_length,
            valid_pairs=valid_pairs,
        )
        train_started = time.perf_counter()
        metrics = trainer.train_microbatch(batch)
        train_elapsed = time.perf_counter() - train_started
        state["training_wall_seconds"] += train_elapsed
        state["grad_norms"].append(float(metrics.grad_norm))
        if metrics.grad_norm is not None and bundle["trainer_config"].gradient_clip_norm is not None:
            if metrics.grad_norm > bundle["trainer_config"].gradient_clip_norm:
                state["clip_count"] += 1
        if snapshot is not None:
            state["update_ratios"].append(
                {
                    "optimizer_step": trainer.optimizer_step,
                    "optimized_tokens": trainer.tokens_seen,
                    "relative_l2_update": _relative_update_ratio(model, snapshot),
                }
            )
        _assert_token_transition(
            before=before_tokens,
            metrics_tokens=metrics.tokens,
            after=trainer.tokens_seen,
            requested=valid_pairs,
        )
        if trainer.tokens_seen > stop_tokens:
            raise RuntimeError("optimized-token budget overshoot")

        if trainer.tokens_seen in token_budgets:
            loss, checked_tokens, eval_seconds = _evaluate_checked(
                model=model,
                trainer=trainer,
                validation_records=validation_records,
                tokenizer=tokenizer,
                expected_validation_tokens=validation_tokens,
            )
            point = {
                "parameters": bundle["spec"].parameter_count(),
                "requested_token_budget": trainer.tokens_seen,
                "optimized_tokens": trainer.tokens_seen,
                "optimizer_steps": trainer.optimizer_step,
                "validation_loss": loss,
                "validation_bpb": loss / math.log(2.0),
                "validation_loss_tokens": checked_tokens,
                "evaluation_optimized_tokens": 0,
                "compute_proxy": 6 * bundle["spec"].parameter_count() * trainer.tokens_seen,
                "last_train_loss": float(metrics.loss),
                "last_grad_norm": float(metrics.grad_norm),
                "evaluation_wall_seconds": eval_seconds,
            }
            if point["optimized_tokens"] != point["requested_token_budget"]:
                raise RuntimeError("checkpoint token budget is not exact")
            state["checkpoints"].append(point)


def _base_state(*, initial_loss: float, validation_tokens: int) -> dict[str, Any]:
    return {
        "initial_validation_loss": initial_loss,
        "initial_validation_bpb": initial_loss / math.log(2.0),
        "validation_tokens": validation_tokens,
        "training_wall_seconds": 0.0,
        "wall_seconds_end_to_end": 0.0,
        "grad_norms": [],
        "clip_count": 0,
        "update_ratios": [],
        "checkpoints": [],
    }


def start_model(
    *,
    repo_root: Path,
    source_sha: str,
    model_index: int,
    stop_tokens: int,
    final_tokens: int,
    partial_path: Path,
    checkpoint_dir: Path,
    batch_size: int = 4,
    sequence_length: int = 64,
    seed: int = 1337,
    torch_threads: int = 2,
) -> dict[str, Any]:
    _assert_exact_source(repo_root, source_sha)
    if tuple(sorted(DEFAULT_TOKEN_BUDGETS)) != DEFAULT_TOKEN_BUDGETS:
        raise RuntimeError("default token budgets are not strictly increasing")
    if stop_tokens not in DEFAULT_TOKEN_BUDGETS or stop_tokens >= final_tokens:
        raise ValueError("stop_tokens must be a non-final common token budget")
    if final_tokens != DEFAULT_TOKEN_BUDGETS[-1]:
        raise ValueError("final_tokens must equal the fixed experiment final budget")
    if checkpoint_dir.exists():
        raise FileExistsError(f"checkpoint destination already exists: {checkpoint_dir}")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    bundle = _control_bundle(
        repo_root=repo_root,
        source_sha=source_sha,
        model_index=model_index,
        final_tokens=final_tokens,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed,
    )
    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(bundle["spec"], bundle["init_spec"])
    trainer = Trainer(model, bundle["trainer_config"], device="cpu")
    phase_started = time.perf_counter()
    initial_loss, validation_tokens, initial_eval_seconds = _evaluate_checked(
        model=model,
        trainer=trainer,
        validation_records=bundle["data"]["validation_records"],
        tokenizer=bundle["data"]["tokenizer"],
    )
    state = _base_state(initial_loss=initial_loss, validation_tokens=validation_tokens)
    state["initial_evaluation_wall_seconds"] = initial_eval_seconds
    _train_until(
        model=model,
        trainer=trainer,
        bundle=bundle,
        stop_tokens=stop_tokens,
        token_budgets=DEFAULT_TOKEN_BUDGETS,
        state=state,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    if trainer.tokens_seen != stop_tokens:
        raise RuntimeError("start phase did not stop on exact optimized-token boundary")
    trainer.assert_checkpoint_safe()
    save_started = time.perf_counter()
    manifest = save_trainer_checkpoint(
        checkpoint_dir,
        model=model,
        trainer=trainer,
        identity=_checkpoint_identity(source_sha=source_sha, bundle=bundle, trainer=trainer),
    )
    save_seconds = time.perf_counter() - save_started
    state["wall_seconds_end_to_end"] = time.perf_counter() - phase_started
    payload = {
        "schema": PARTIAL_SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "model_index": model_index,
        "model_identity_sha256": bundle["spec"].identity_sha256(),
        "parameters": bundle["spec"].parameter_count(),
        "controls_hash": bundle["controls_hash"],
        "stop_tokens": stop_tokens,
        "final_tokens": final_tokens,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "seed": seed,
        "start_pid": os.getpid(),
        "state": state,
        "checkpoint": {
            "directory": str(checkpoint_dir),
            "checkpoint_id": manifest["checkpoint_id"],
            "save_seconds": save_seconds,
            "bytes": _directory_bytes(checkpoint_dir),
        },
        "memory": {
            "parameter_bytes": _parameter_bytes(model),
            "optimizer_tensor_bytes": _tensor_tree_bytes(trainer.optimizer.state_dict()),
            "process_peak_rss_bytes": _peak_rss_bytes(),
        },
    }
    return _write_hashed_json(partial_path, payload)


def resume_model(
    *,
    repo_root: Path,
    source_sha: str,
    partial_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
    torch_threads: int = 2,
) -> dict[str, Any]:
    _assert_exact_source(repo_root, source_sha)
    partial = _read_hashed_json(partial_path, schema=PARTIAL_SCHEMA)
    if partial["source_sha"] != source_sha:
        raise RuntimeError("partial evidence source SHA mismatch")
    if int(partial["start_pid"]) == os.getpid():
        raise RuntimeError("resume must execute in a fresh process")
    model_index = int(partial["model_index"])
    final_tokens = int(partial["final_tokens"])
    batch_size = int(partial["batch_size"])
    sequence_length = int(partial["sequence_length"])
    seed = int(partial["seed"])
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    bundle = _control_bundle(
        repo_root=repo_root,
        source_sha=source_sha,
        model_index=model_index,
        final_tokens=final_tokens,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed,
    )
    if bundle["controls_hash"] != partial["controls_hash"]:
        raise RuntimeError("fixed scientific controls drifted across fresh-process resume")

    model = TwelveSixDecoder(bundle["spec"], bundle["init_spec"])
    trainer = Trainer(model, bundle["trainer_config"], device="cpu")
    phase_started = time.perf_counter()
    load_started = time.perf_counter()
    load_result = load_trainer_checkpoint(
        checkpoint_dir,
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=source_sha,
        expected_model_spec_hash=bundle["spec"].identity_sha256(),
        expected_init_spec_hash=bundle["init_spec"].identity_sha256(),
        expected_tokenizer_hash=BYTE_TOKENIZER_HASH,
        expected_tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        expected_dataset_manifest_hash=bundle["data"]["dataset_manifest_hash"],
        expected_split_identity=bundle["data"]["split_identity"],
        expected_packing_hash=PACKING_SHA256,
        expected_packing_version=PACKING_VERSION,
        expected_run_manifest_hash=bundle["run_manifest_hash"],
        expected_training_config_hash=hash_json(bundle["training_config"]),
        expected_seed=seed,
    )
    load_seconds = time.perf_counter() - load_started
    stop_tokens = int(partial["stop_tokens"])
    if trainer.tokens_seen != stop_tokens:
        raise RuntimeError(
            f"resume token counter mismatch: {trainer.tokens_seen} != {stop_tokens}"
        )
    if int(load_result.manifest["identity"]["tokens_seen"]) != stop_tokens:
        raise RuntimeError("verified checkpoint identity token count mismatch")

    state = dict(partial["state"])
    state["grad_norms"] = list(state["grad_norms"])
    state["update_ratios"] = list(state["update_ratios"])
    state["checkpoints"] = list(state["checkpoints"])
    resume_loss, checked_tokens, resume_eval_seconds = _evaluate_checked(
        model=model,
        trainer=trainer,
        validation_records=bundle["data"]["validation_records"],
        tokenizer=bundle["data"]["tokenizer"],
        expected_validation_tokens=int(state["validation_tokens"]),
    )
    checkpoint_point = next(
        point for point in state["checkpoints"] if int(point["optimized_tokens"]) == stop_tokens
    )
    reload_abs_diff = abs(resume_loss - float(checkpoint_point["validation_loss"]))
    if reload_abs_diff > 1e-12:
        raise RuntimeError(
            f"fresh reload validation drift: abs_diff={reload_abs_diff:.3e}"
        )

    prior_end_to_end = float(state["wall_seconds_end_to_end"])
    _train_until(
        model=model,
        trainer=trainer,
        bundle=bundle,
        stop_tokens=final_tokens,
        token_budgets=DEFAULT_TOKEN_BUDGETS,
        state=state,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )
    if trainer.tokens_seen != final_tokens:
        raise RuntimeError("resume phase did not end on exact final token budget")
    state["wall_seconds_end_to_end"] = prior_end_to_end + (time.perf_counter() - phase_started)

    budgets_observed = [int(point["optimized_tokens"]) for point in state["checkpoints"]]
    if budgets_observed != list(DEFAULT_TOKEN_BUDGETS):
        raise RuntimeError(
            f"exact token checkpoints drifted: {budgets_observed!r} != {DEFAULT_TOKEN_BUDGETS!r}"
        )
    final_point = state["checkpoints"][-1]
    grad_norms = [float(value) for value in state["grad_norms"]]
    update_values = [float(item["relative_l2_update"]) for item in state["update_ratios"]]
    optimizer_steps = trainer.optimizer_step
    clip_count = int(state["clip_count"])
    initial_loss = float(state["initial_validation_loss"])
    final_loss = float(final_point["validation_loss"])
    improvement = initial_loss - final_loss
    memory = {
        "parameter_bytes": _parameter_bytes(model),
        "optimizer_tensor_bytes": _tensor_tree_bytes(trainer.optimizer.state_dict()),
        "model_plus_optimizer_tensor_bytes": (
            _parameter_bytes(model) + _tensor_tree_bytes(trainer.optimizer.state_dict())
        ),
        "process_peak_rss_bytes": max(
            int(partial["memory"]["process_peak_rss_bytes"]), _peak_rss_bytes()
        ),
    }
    report = {
        "schema": MODEL_SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "model_index": model_index,
        "parameters": bundle["spec"].parameter_count(),
        "model_spec": bundle["spec"].to_dict(),
        "model_identity_sha256": bundle["spec"].identity_sha256(),
        "init_identity_sha256": bundle["init_spec"].identity_sha256(),
        "controls_hash": bundle["controls_hash"],
        "tokenizer": {
            "id": BYTE_TOKENIZER_VERSION,
            "config_sha256": BYTE_TOKENIZER_HASH,
            "vocab_sha256": BYTE_VOCAB_HASH,
            "vocab_size": bundle["data"]["tokenizer"].vocab_size,
        },
        "data": {
            "dataset_manifest_hash": bundle["data"]["dataset_manifest_hash"],
            "train_jsonl_sha256": bundle["data"]["train_jsonl_sha256"],
            "validation_jsonl_sha256": bundle["data"]["validation_jsonl_sha256"],
            "split_identity": bundle["data"]["split_identity"],
            "train_validation_record_overlap": bundle["data"]["overlap"],
            "repeated_project_fixture": True,
        },
        "packing": {
            **PACKING_DEFINITION,
            "sha256": PACKING_SHA256,
        },
        "optimizer_recipe": bundle["training_config"],
        "token_budgets": list(DEFAULT_TOKEN_BUDGETS),
        "initial_validation_loss": initial_loss,
        "initial_validation_bpb": float(state["initial_validation_bpb"]),
        "checkpoints": state["checkpoints"],
        "final": {
            "optimized_tokens": trainer.tokens_seen,
            "optimizer_steps": optimizer_steps,
            "validation_loss": final_loss,
            "validation_bpb": float(final_point["validation_bpb"]),
            "validation_improvement": improvement,
            "compute_proxy": int(final_point["compute_proxy"]),
            "training_wall_seconds": float(state["training_wall_seconds"]),
            "end_to_end_wall_seconds": float(state["wall_seconds_end_to_end"]),
            "optimized_tokens_per_training_second": (
                trainer.tokens_seen / float(state["training_wall_seconds"])
            ),
        },
        "gradient_statistics": {
            "samples": len(grad_norms),
            "mean_preclip_global_norm": _mean(grad_norms),
            "max_preclip_global_norm": max(grad_norms) if grad_norms else None,
            "clip_threshold": bundle["trainer_config"].gradient_clip_norm,
            "clip_count": clip_count,
            "clip_frequency": clip_count / optimizer_steps if optimizer_steps else 0.0,
        },
        "update_statistics": {
            "samples": len(update_values),
            "mean_relative_l2_update": _mean(update_values),
            "max_relative_l2_update": max(update_values) if update_values else None,
            "sample_points": state["update_ratios"],
        },
        "memory": memory,
        "resume": {
            "required": True,
            "fresh_process": True,
            "start_pid": int(partial["start_pid"]),
            "resume_pid": os.getpid(),
            "resume_tokens": stop_tokens,
            "checkpoint_id": partial["checkpoint"]["checkpoint_id"],
            "checkpoint_bytes": int(partial["checkpoint"]["bytes"]),
            "checkpoint_save_seconds": float(partial["checkpoint"]["save_seconds"]),
            "checkpoint_load_seconds": load_seconds,
            "reload_validation_loss": resume_loss,
            "reload_validation_bpb": resume_loss / math.log(2.0),
            "reload_validation_abs_diff": reload_abs_diff,
            "resume_evaluation_wall_seconds": resume_eval_seconds,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": torch_threads,
            "paid_compute": False,
        },
        "truth_boundary": {
            "generalization_split": "held-out project S0 validation fixture",
            "bpb_valid_because_tokenizer_is_raw_utf8_bytes": True,
            "representative_scale_corpus": False,
            "stage_freeze": False,
            "promotion_authority": False,
            "quality_capability_claim": False,
            "paid_compute": False,
        },
    }
    return _write_hashed_json(output_path, report)


def _rank(records: list[dict[str, Any]], key: str, *, reverse: bool) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: float(item[key]), reverse=reverse)
    return [
        {
            "rank": rank,
            "parameters": int(item["parameters"]),
            "value": float(item[key]),
        }
        for rank, item in enumerate(ordered, 1)
    ]


def aggregate_models(
    *,
    input_paths: list[Path],
    output_path: Path,
    expected_source_sha: str,
) -> dict[str, Any]:
    if len(input_paths) != 4:
        raise ValueError("exactly four model reports are required")
    reports = [_read_hashed_json(path, schema=MODEL_SCHEMA) for path in input_paths]
    if {report["source_sha"] for report in reports} != {expected_source_sha}:
        raise RuntimeError("model reports do not share the expected source SHA")
    counts = tuple(sorted(int(report["parameters"]) for report in reports))
    if counts != _EXPECTED_COUNTS:
        raise RuntimeError(f"model parameter matrix drift: {counts!r}")
    if len({report["controls_hash"] for report in reports}) != 4:
        # controls hash intentionally includes model identity, so compare the actual fixed controls below.
        pass
    control_projections = {
        hash_json(
            {
                "tokenizer": report["tokenizer"],
                "data": report["data"],
                "packing": report["packing"],
                "optimizer_recipe": report["optimizer_recipe"],
                "token_budgets": report["token_budgets"],
                "init_identity_sha256": report["init_identity_sha256"],
            }
        )
        for report in reports
    }
    if len(control_projections) != 1:
        raise RuntimeError("scientific controls differ across model candidates")

    rows: list[dict[str, Any]] = []
    for report in reports:
        final = report["final"]
        improvement = float(final["validation_improvement"])
        parameters = int(report["parameters"])
        compute = int(final["compute_proxy"])
        wall = float(final["end_to_end_wall_seconds"])
        if int(final["optimized_tokens"]) != DEFAULT_TOKEN_BUDGETS[-1]:
            raise RuntimeError("candidate did not receive the exact final optimized-token budget")
        if report["resume"]["fresh_process"] is not True:
            raise RuntimeError("candidate lacks required fresh-process resume evidence")
        rows.append(
            {
                "parameters": parameters,
                "initial_validation_loss": float(report["initial_validation_loss"]),
                "final_validation_loss": float(final["validation_loss"]),
                "final_validation_bpb": float(final["validation_bpb"]),
                "validation_improvement": improvement,
                "validation_improvement_per_parameter": improvement / parameters,
                "validation_improvement_per_compute": improvement / compute,
                "validation_improvement_per_wall_second": improvement / wall,
                "compute_proxy": compute,
                "training_wall_seconds": float(final["training_wall_seconds"]),
                "end_to_end_wall_seconds": wall,
                "optimizer_tensor_bytes": int(report["memory"]["optimizer_tensor_bytes"]),
                "model_plus_optimizer_tensor_bytes": int(
                    report["memory"]["model_plus_optimizer_tensor_bytes"]
                ),
                "process_peak_rss_bytes": int(report["memory"]["process_peak_rss_bytes"]),
                "mean_grad_norm": report["gradient_statistics"]["mean_preclip_global_norm"],
                "clip_frequency": float(report["gradient_statistics"]["clip_frequency"]),
                "mean_relative_l2_update": report["update_statistics"]["mean_relative_l2_update"],
                "checkpoint_bytes": int(report["resume"]["checkpoint_bytes"]),
                "checkpoint_save_seconds": float(report["resume"]["checkpoint_save_seconds"]),
                "checkpoint_load_seconds": float(report["resume"]["checkpoint_load_seconds"]),
            }
        )

    rankings = {
        "best_validation": _rank(rows, "final_validation_loss", reverse=False),
        "validation_improvement_per_parameter": _rank(
            rows, "validation_improvement_per_parameter", reverse=True
        ),
        "validation_improvement_per_compute": _rank(
            rows, "validation_improvement_per_compute", reverse=True
        ),
        "validation_improvement_per_wall_second": _rank(
            rows, "validation_improvement_per_wall_second", reverse=True
        ),
    }
    rank_sums: dict[int, int] = {count: 0 for count in counts}
    for ranking in rankings.values():
        for item in ranking:
            rank_sums[int(item["parameters"])] += int(item["rank"])
    recommended = min(rank_sums, key=lambda count: (rank_sums[count], count))

    per_budget: dict[str, list[dict[str, Any]]] = {}
    for budget in DEFAULT_TOKEN_BUDGETS:
        budget_rows = []
        for report in reports:
            point = next(
                checkpoint
                for checkpoint in report["checkpoints"]
                if int(checkpoint["optimized_tokens"]) == budget
            )
            budget_rows.append(
                {
                    "parameters": int(report["parameters"]),
                    "validation_loss": float(point["validation_loss"]),
                    "validation_bpb": float(point["validation_bpb"]),
                }
            )
        per_budget[str(budget)] = sorted(
            budget_rows, key=lambda item: item["validation_loss"]
        )

    aggregate = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": expected_source_sha,
            "incumbent": "RESEARCH41 PR #162 fixed-control family",
        },
        "scientific_question": (
            "Which ~100K-1M 12-6 candidate generalizes best when every candidate "
            "receives exactly the same count of valid optimized causal loss tokens?"
        ),
        "fixed_controls": {
            "parameter_counts": list(counts),
            "token_budgets": list(DEFAULT_TOKEN_BUDGETS),
            "tokenizer_id": BYTE_TOKENIZER_VERSION,
            "packing_version": PACKING_VERSION,
            "packing_sha256": PACKING_SHA256,
            "evaluation_tokens_optimized": 0,
            "compute_proxy_definition": COMPUTE_PROXY,
            "control_projection_sha256": next(iter(control_projections)),
        },
        "matrix": sorted(rows, key=lambda item: item["parameters"]),
        "validation_by_budget": per_budget,
        "rankings": rankings,
        "recommendation": {
            "parameters": recommended,
            "rank_sum": rank_sums[recommended],
            "rank_sums": {str(key): value for key, value in sorted(rank_sums.items())},
            "selection_rule": (
                "lowest sum of ordinal ranks across the four requested final-budget criteria; "
                "ties favor fewer parameters"
            ),
            "scope": "next primary small-model research vehicle under this controlled fixture only",
        },
        "truth_boundary": {
            "train_loss_used_as_generalization_rank": False,
            "held_out_validation_used": True,
            "project_fixture_recycled": True,
            "representative_scale_corpus": False,
            "pretrained_weights": False,
            "sft": False,
            "paid_compute": False,
            "quality_overclaim": False,
        },
    }
    return _write_hashed_json(output_path, aggregate)


def validate_evidence(path: Path, *, expected_source_sha: str | None = None) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in {SCHEMA, MODEL_SCHEMA, PARTIAL_SCHEMA}:
        raise RuntimeError(f"unsupported evidence schema: {schema!r}")
    recorded = payload.get("report_sha256")
    if recorded != _canonical_hash(payload):
        raise RuntimeError("evidence self-hash mismatch")
    source = payload.get("source", {}).get("git_sha") if schema == SCHEMA else payload.get("source_sha")
    if expected_source_sha is not None and source != expected_source_sha:
        raise RuntimeError(f"source mismatch: {source!r} != {expected_source_sha!r}")
    if schema == SCHEMA:
        matrix = payload.get("matrix")
        if not isinstance(matrix, list) or len(matrix) != 4:
            raise RuntimeError("aggregate evidence must contain four candidates")
        if tuple(sorted(int(row["parameters"]) for row in matrix)) != _EXPECTED_COUNTS:
            raise RuntimeError("aggregate parameter matrix drift")
        if payload["fixed_controls"]["token_budgets"] != list(DEFAULT_TOKEN_BUDGETS):
            raise RuntimeError("aggregate token-budget drift")
    elif schema == MODEL_SCHEMA:
        checkpoints = payload.get("checkpoints")
        observed = [int(point["optimized_tokens"]) for point in checkpoints]
        requested = [int(point["requested_token_budget"]) for point in checkpoints]
        if observed != list(DEFAULT_TOKEN_BUDGETS) or requested != list(DEFAULT_TOKEN_BUDGETS):
            raise RuntimeError("model report does not prove exact common token budgets")
        if any(int(point["evaluation_optimized_tokens"]) != 0 for point in checkpoints):
            raise RuntimeError("evaluation tokens were counted as optimized")
        if payload["resume"]["fresh_process"] is not True:
            raise RuntimeError("fresh-process resume evidence missing")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="train one candidate to the forced-resume budget")
    start.add_argument("--repo-root", type=Path, required=True)
    start.add_argument("--source-sha", required=True)
    start.add_argument("--model-index", type=int, required=True)
    start.add_argument("--stop-tokens", type=int, default=DEFAULT_RESUME_TOKENS)
    start.add_argument("--final-tokens", type=int, default=DEFAULT_TOKEN_BUDGETS[-1])
    start.add_argument("--partial", type=Path, required=True)
    start.add_argument("--checkpoint-dir", type=Path, required=True)
    start.add_argument("--torch-threads", type=int, default=2)

    resume = sub.add_parser("resume", help="fresh-process D05 resume and finish one candidate")
    resume.add_argument("--repo-root", type=Path, required=True)
    resume.add_argument("--source-sha", required=True)
    resume.add_argument("--partial", type=Path, required=True)
    resume.add_argument("--checkpoint-dir", type=Path, required=True)
    resume.add_argument("--output", type=Path, required=True)
    resume.add_argument("--torch-threads", type=int, default=2)

    aggregate = sub.add_parser("aggregate", help="rank four executed candidate reports")
    aggregate.add_argument("--inputs", type=Path, nargs=4, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--expected-source-sha", required=True)

    validate = sub.add_parser("validate", help="validate retained evidence")
    validate.add_argument("path", type=Path)
    validate.add_argument("--expected-source-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "start":
        start_model(
            repo_root=args.repo_root,
            source_sha=args.source_sha,
            model_index=args.model_index,
            stop_tokens=args.stop_tokens,
            final_tokens=args.final_tokens,
            partial_path=args.partial,
            checkpoint_dir=args.checkpoint_dir,
            torch_threads=args.torch_threads,
        )
    elif args.command == "resume":
        resume_model(
            repo_root=args.repo_root,
            source_sha=args.source_sha,
            partial_path=args.partial,
            checkpoint_dir=args.checkpoint_dir,
            output_path=args.output,
            torch_threads=args.torch_threads,
        )
    elif args.command == "aggregate":
        aggregate_models(
            input_paths=args.inputs,
            output_path=args.output,
            expected_source_sha=args.expected_source_sha,
        )
    elif args.command == "validate":
        validate_evidence(args.path, expected_source_sha=args.expected_source_sha)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
