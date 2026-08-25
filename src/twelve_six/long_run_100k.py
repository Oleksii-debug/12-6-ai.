"""Long-run LOCAL_FREE evidence for the ~100K RESEARCH41 Base candidate.

TRAIN-41 deliberately changes training duration and observation cadence, not the
model/tokenizer/data/optimizer semantics.  It reuses the strict aligned causal-pair
accounting introduced by RESEARCH06 and the incumbent D05 checkpoint adapter.

The current repository does not yet contain a representative approved scale corpus
or a frozen next-stage tokenizer.  This experiment therefore measures what the
95,568-parameter from-scratch Base can optimize/memorize/generalize on the exact
project-owned S0 fixture; it must not be presented as representative language-model
quality evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .checkpoint.core import hash_json, sha256_file
from .checkpoint.trainer_adapter import load_trainer_checkpoint, save_trainer_checkpoint
from .fixed_token_efficiency import (
    PACKING_DEFINITION,
    PACKING_SHA256,
    PACKING_VERSION,
    _assert_exact_source,
    _assert_token_transition,
    _checkpoint_identity,
    _directory_bytes,
    _evaluate_checked,
    _load_control_data,
    _make_pair_batch,
    _parameter_bytes,
    _parameter_snapshot,
    _peak_rss_bytes,
    _read_hashed_json,
    _relative_update_ratio,
    _tensor_tree_bytes,
    _write_hashed_json,
)
from .inference.contracts import GenerationConfig
from .inference.generation import generate
from .integration.s0_runtime import S0TorchInferenceBackend
from .model import InitSpec, TwelveSixDecoder
from .scaling_experiment import controlled_specs
from .tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
)
from .training import Trainer, TrainerConfig

SCHEMA = "12-6.train41-long-100k.v1"
PARTIAL_SCHEMA = "12-6.train41-long-100k-partial.v1"
AUTHORITY = "LOCAL_FREE_LONG_100K_FIXTURE_EVIDENCE_NOT_PROMOTION_OR_CAPABILITY"
MODEL_INDEX = 0
EXPECTED_PARAMETERS = 95_568
BATCH_SIZE = 4
SEQUENCE_LENGTH = 64
CAPACITY = BATCH_SIZE * SEQUENCE_LENGTH
SEED = 1337
FINAL_TOKENS = 2_097_152
RESUME_TOKENS = 1_048_576
EVALUATION_BUDGETS = (
    1_024,
    2_048,
    4_096,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
    1_572_864,
    2_097_152,
)
GENERATION_BUDGETS = (0, 16_384, 65_536, 262_144, 1_048_576, 1_572_864, 2_097_152)
RETAINED_CHECKPOINT_BUDGETS = (65_536, 262_144, 1_048_576, 1_572_864, 2_097_152)
GENERATION_PROMPTS = ("The ", "Україна ", "def ")
GENERATION_NEW_TOKENS = 32
HIGH_GRAD_NORM = 100.0
HIGH_GRAD_STREAK_LIMIT = 8
VALIDATION_DIVERGENCE_DELTA_NATS = 2.0
NO_IMPROVEMENT_MIN_TOKENS = 1_572_864
NO_IMPROVEMENT_WINDOW = 4
NO_IMPROVEMENT_DELTA_NATS = 1e-4


def _mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def _trainer_config() -> TrainerConfig:
    if FINAL_TOKENS % CAPACITY:
        raise RuntimeError("final token budget must align to the strict training capacity")
    return TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=FINAL_TOKENS // CAPACITY,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _build_bundle(*, repo_root: Path, source_sha: str) -> dict[str, Any]:
    specs = controlled_specs()
    spec = specs[MODEL_INDEX]
    if spec.parameter_count() != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"controlled ~100K ModelSpec drift: {spec.parameter_count()} != {EXPECTED_PARAMETERS}"
        )
    if spec.vocab_size != 256:
        raise RuntimeError("TRAIN-41 requires the exact byte-vocabulary RESEARCH41 candidate")
    data = _load_control_data(repo_root)
    init_spec = InitSpec()
    trainer_config = _trainer_config()
    training_config = {
        "experiment": SCHEMA,
        "recipe": "research41-fixed-control-adamw-v1",
        "trainer": asdict(trainer_config),
        "init_spec_sha256": init_spec.identity_sha256(),
        "data": {
            "split_identity": data["split_identity"],
            "packing_sha256": PACKING_SHA256,
            "packing_version": PACKING_VERSION,
        },
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "evaluation_budgets": list(EVALUATION_BUDGETS),
        "generation_budgets": list(GENERATION_BUDGETS),
        "retained_checkpoint_budgets": list(RETAINED_CHECKPOINT_BUDGETS),
        "resume_tokens": RESUME_TOKENS,
        "final_tokens": FINAL_TOKENS,
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
    return {
        "spec": spec,
        "init_spec": init_spec,
        "trainer_config": trainer_config,
        "training_config": training_config,
        "run_manifest_hash": hash_json(run_manifest),
        "controls_hash": hash_json(run_manifest),
        "data": data,
    }


def _should_sample_step(step: int, optimized_tokens: int) -> bool:
    """Dense early telemetry with progressively wider late spacing."""
    if step <= 64:
        return True
    if step <= 512:
        return step % 8 == 0
    if step <= 2_048:
        return step % 32 == 0
    return step % 128 == 0 or optimized_tokens in EVALUATION_BUDGETS


def _generation_snapshot(model: TwelveSixDecoder, tokenizer: Any, optimized_tokens: int) -> dict[str, Any]:
    backend = S0TorchInferenceBackend(model, tokenizer)
    outputs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for prompt in GENERATION_PROMPTS:
        result = generate(
            backend,
            prompt,
            GenerationConfig(max_new_tokens=GENERATION_NEW_TOKENS, sample=False, seed=0),
        )
        outputs.append(
            {
                "prompt": prompt,
                "prompt_token_ids": list(result.prompt_token_ids),
                "generated_token_ids": list(result.generated_token_ids),
                "text": result.text,
                "stop_reason": result.stop_reason,
            }
        )
    return {
        "optimized_tokens": optimized_tokens,
        "greedy": True,
        "max_new_tokens": GENERATION_NEW_TOKENS,
        "outputs": outputs,
        "wall_seconds": time.perf_counter() - started,
    }


def _new_state(
    *,
    initial_validation_loss: float,
    validation_tokens: int,
    initial_eval_seconds: float,
    initial_generation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "initial_validation_loss": initial_validation_loss,
        "initial_validation_bpb": initial_validation_loss / math.log(2.0),
        "validation_tokens": validation_tokens,
        "evaluation_points": [
            {
                "optimized_tokens": 0,
                "optimizer_steps": 0,
                "train_loss_since_previous_eval": None,
                "train_bpb_since_previous_eval": None,
                "validation_loss": initial_validation_loss,
                "validation_bpb": initial_validation_loss / math.log(2.0),
                "evaluation_wall_seconds": initial_eval_seconds,
            }
        ],
        "generation_snapshots": [initial_generation],
        "step_telemetry": [],
        "update_ratios": [],
        "grad_norm_sum": 0.0,
        "grad_norm_count": 0,
        "grad_norm_max": 0.0,
        "clip_count": 0,
        "training_wall_seconds": 0.0,
        "evaluation_wall_seconds": initial_eval_seconds,
        "generation_wall_seconds": float(initial_generation["wall_seconds"]),
        "checkpoint_save_seconds": 0.0,
        "checkpoint_load_seconds": 0.0,
        "checkpoint_records": [],
        "interval_loss_token_sum": 0.0,
        "interval_tokens": 0,
        "high_grad_streak": 0,
        "stop": None,
        "end_to_end_wall_seconds": 0.0,
    }


def _checkpoint_record_at(state: dict[str, Any], tokens: int) -> dict[str, Any] | None:
    return next(
        (
            record
            for record in state["checkpoint_records"]
            if int(record["optimized_tokens"]) == tokens
        ),
        None,
    )


def _save_checkpoint(
    *,
    checkpoint_root: Path,
    source_sha: str,
    bundle: dict[str, Any],
    model: TwelveSixDecoder,
    trainer: Trainer,
    state: dict[str, Any],
    terminal_kind: str | None = None,
) -> dict[str, Any]:
    existing = _checkpoint_record_at(state, trainer.tokens_seen)
    if existing is not None:
        return existing
    trainer.assert_checkpoint_safe()
    destination = checkpoint_root / f"tokens-{trainer.tokens_seen}"
    if destination.exists():
        raise FileExistsError(f"checkpoint destination already exists: {destination}")
    started = time.perf_counter()
    manifest = save_trainer_checkpoint(
        destination,
        model=model,
        trainer=trainer,
        identity=_checkpoint_identity(
            source_sha=source_sha,
            bundle=bundle,
            trainer=trainer,
        ),
    )
    elapsed = time.perf_counter() - started
    record = {
        "optimized_tokens": trainer.tokens_seen,
        "optimizer_steps": trainer.optimizer_step,
        "directory": str(destination),
        "checkpoint_id": manifest["checkpoint_id"],
        "bytes": _directory_bytes(destination),
        "save_seconds": elapsed,
        "terminal_kind": terminal_kind,
    }
    state["checkpoint_records"].append(record)
    state["checkpoint_save_seconds"] += elapsed
    return record


def _no_improvement(points: list[dict[str, Any]], optimized_tokens: int) -> bool:
    if optimized_tokens < NO_IMPROVEMENT_MIN_TOKENS:
        return False
    trained = [point for point in points if int(point["optimized_tokens"]) > 0]
    if len(trained) < NO_IMPROVEMENT_WINDOW:
        return False
    window = trained[-NO_IMPROVEMENT_WINDOW:]
    first = float(window[0]["validation_loss"])
    later_best = min(float(point["validation_loss"]) for point in window[1:])
    return first - later_best < NO_IMPROVEMENT_DELTA_NATS


def _parameters_finite(model: TwelveSixDecoder) -> bool:
    return all(torch.isfinite(parameter.detach()).all().item() for parameter in model.parameters())


def _evaluate_point(
    *,
    model: TwelveSixDecoder,
    trainer: Trainer,
    bundle: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    validation_loss, validation_tokens, eval_seconds = _evaluate_checked(
        model=model,
        trainer=trainer,
        validation_records=bundle["data"]["validation_records"],
        tokenizer=bundle["data"]["tokenizer"],
        expected_validation_tokens=int(state["validation_tokens"]),
    )
    interval_tokens = int(state["interval_tokens"])
    if interval_tokens <= 0:
        raise RuntimeError("scheduled evaluation has no training tokens since the previous evaluation")
    train_loss = float(state["interval_loss_token_sum"]) / interval_tokens
    point = {
        "optimized_tokens": trainer.tokens_seen,
        "optimizer_steps": trainer.optimizer_step,
        "train_loss_since_previous_eval": train_loss,
        "train_bpb_since_previous_eval": train_loss / math.log(2.0),
        "validation_loss": validation_loss,
        "validation_bpb": validation_loss / math.log(2.0),
        "validation_loss_tokens": validation_tokens,
        "evaluation_optimized_tokens": 0,
        "evaluation_wall_seconds": eval_seconds,
    }
    state["evaluation_points"].append(point)
    state["evaluation_wall_seconds"] += eval_seconds
    state["interval_loss_token_sum"] = 0.0
    state["interval_tokens"] = 0
    return point


def _run_phase(
    *,
    source_sha: str,
    bundle: dict[str, Any],
    model: TwelveSixDecoder,
    trainer: Trainer,
    state: dict[str, Any],
    checkpoint_root: Path,
    stop_tokens: int,
    permit_no_improvement_stop: bool,
) -> None:
    stream = bundle["data"]["train_stream"]
    tokenizer = bundle["data"]["tokenizer"]
    initial_validation_loss = float(state["initial_validation_loss"])

    while trainer.tokens_seen < stop_tokens:
        before_tokens = trainer.tokens_seen
        valid_pairs = min(CAPACITY, stop_tokens - before_tokens)
        planned_tokens = before_tokens + valid_pairs
        planned_step = trainer.optimizer_step + 1
        sample = _should_sample_step(planned_step, planned_tokens)
        snapshot = _parameter_snapshot(model) if sample else None
        batch = _make_pair_batch(
            stream,
            causal_offset=before_tokens,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
            valid_pairs=valid_pairs,
        )
        started = time.perf_counter()
        metrics = trainer.train_microbatch(batch)
        step_seconds = time.perf_counter() - started
        state["training_wall_seconds"] += step_seconds
        state["interval_loss_token_sum"] += float(metrics.loss) * metrics.tokens
        state["interval_tokens"] += metrics.tokens

        _assert_token_transition(
            before=before_tokens,
            metrics_tokens=metrics.tokens,
            after=trainer.tokens_seen,
            requested=valid_pairs,
        )
        if metrics.grad_norm is None:
            raise RuntimeError("gradient norm is required for every TRAIN-41 optimizer step")
        grad_norm = float(metrics.grad_norm)
        state["grad_norm_sum"] += grad_norm
        state["grad_norm_count"] += 1
        state["grad_norm_max"] = max(float(state["grad_norm_max"]), grad_norm)
        clip_active = grad_norm > float(bundle["trainer_config"].gradient_clip_norm)
        if clip_active:
            state["clip_count"] += 1
        if grad_norm > HIGH_GRAD_NORM:
            state["high_grad_streak"] += 1
        else:
            state["high_grad_streak"] = 0

        update_ratio = None
        if snapshot is not None:
            update_ratio = _relative_update_ratio(model, snapshot)
            state["update_ratios"].append(
                {
                    "optimizer_step": trainer.optimizer_step,
                    "optimized_tokens": trainer.tokens_seen,
                    "relative_l2_update": update_ratio,
                }
            )
        if sample:
            state["step_telemetry"].append(
                {
                    "optimizer_step": trainer.optimizer_step,
                    "optimized_tokens": trainer.tokens_seen,
                    "train_loss": float(metrics.loss),
                    "train_bpb": float(metrics.loss) / math.log(2.0),
                    "learning_rate": float(metrics.learning_rate),
                    "grad_norm_preclip": grad_norm,
                    "clip_active": clip_active,
                    "relative_l2_update": update_ratio,
                    "step_wall_seconds": step_seconds,
                    "optimized_tokens_per_second": metrics.tokens / step_seconds,
                }
            )

        divergence_reason = None
        if int(state["high_grad_streak"]) >= HIGH_GRAD_STREAK_LIMIT:
            divergence_reason = (
                f"preclip gradient norm exceeded {HIGH_GRAD_NORM} for "
                f"{HIGH_GRAD_STREAK_LIMIT} consecutive committed updates"
            )

        evaluation_point = None
        if trainer.tokens_seen in EVALUATION_BUDGETS:
            if not _parameters_finite(model):
                raise RuntimeError("non-finite model parameter detected at evaluation boundary")
            evaluation_point = _evaluate_point(
                model=model,
                trainer=trainer,
                bundle=bundle,
                state=state,
            )
            if (
                float(evaluation_point["validation_loss"])
                > initial_validation_loss + VALIDATION_DIVERGENCE_DELTA_NATS
            ):
                divergence_reason = (
                    "held-out validation loss exceeded initial loss by more than "
                    f"{VALIDATION_DIVERGENCE_DELTA_NATS} nats"
                )
            if trainer.tokens_seen in GENERATION_BUDGETS:
                generation = _generation_snapshot(model, tokenizer, trainer.tokens_seen)
                state["generation_snapshots"].append(generation)
                state["generation_wall_seconds"] += float(generation["wall_seconds"])

        if trainer.tokens_seen in RETAINED_CHECKPOINT_BUDGETS:
            _save_checkpoint(
                checkpoint_root=checkpoint_root,
                source_sha=source_sha,
                bundle=bundle,
                model=model,
                trainer=trainer,
                state=state,
            )

        if divergence_reason is not None:
            state["stop"] = {
                "kind": "DIVERGENCE",
                "optimized_tokens": trainer.tokens_seen,
                "reason": divergence_reason,
            }
            _save_checkpoint(
                checkpoint_root=checkpoint_root,
                source_sha=source_sha,
                bundle=bundle,
                model=model,
                trainer=trainer,
                state=state,
                terminal_kind="DIVERGENCE",
            )
            break

        if (
            permit_no_improvement_stop
            and evaluation_point is not None
            and _no_improvement(state["evaluation_points"], trainer.tokens_seen)
        ):
            state["stop"] = {
                "kind": "NO_IMPROVEMENT",
                "optimized_tokens": trainer.tokens_seen,
                "reason": (
                    f"best validation improvement across the latest {NO_IMPROVEMENT_WINDOW} "
                    f"scheduled evaluations was < {NO_IMPROVEMENT_DELTA_NATS} nats"
                ),
            }
            _save_checkpoint(
                checkpoint_root=checkpoint_root,
                source_sha=source_sha,
                bundle=bundle,
                model=model,
                trainer=trainer,
                state=state,
                terminal_kind="NO_IMPROVEMENT",
            )
            break


def _selection_evidence(repo_root: Path) -> dict[str, Any]:
    multilingual = repo_root / "configs/data/multilingual_uk_en_code_v1.experimental.json"
    payload = json.loads(multilingual.read_text(encoding="utf-8"))
    bpe = payload["tokenizer_incumbent_evidence"]["bytelevel_bpe"]
    return {
        "corpus": {
            "selected": "s0-tiny-controlled-v1",
            "reason": (
                "only current project-owned train/held-out fixture with an existing exact training+evaluation "
                "identity; no representative external source is approved"
            ),
            "representative": False,
            "train_text_utf8_bytes": 1_920,
            "validation_text_utf8_bytes": 406,
            "data10_mechanics_corpus_bytes": int(payload["local_mechanics_corpus"]["bytes"]),
            "external_sources_training_approved": int(
                payload["source_admission"]["external_sources_training_approved_at_recipe_creation"]
            ),
        },
        "tokenizer": {
            "selected": BYTE_TOKENIZER_VERSION,
            "config_sha256": BYTE_TOKENIZER_HASH,
            "vocab_sha256": BYTE_VOCAB_HASH,
            "reason": (
                "canonical exact-artifact byte tokenizer keeps the incumbent 95,568-parameter geometry; "
                "the observed BPE artifact is promising but not frozen on representative data"
            ),
            "observed_bpe_actual_vocab_size": int(bpe["actual_vocab_size"]),
            "observed_bpe_repeatable_identity": bool(bpe["repeatable_artifact_identity"]),
            "recipe_freeze_decision": payload["tokenizer_incumbent_evidence"]["freeze_decision"],
        },
        "optimizer": {
            "selected": "research41-fixed-control-adamw-v1",
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "clip_norm": 1.0,
            "scheduler": "constant",
            "reason": (
                "preserve the executed fixed-control incumbent; TRAIN-33 optimizer matrix is not used as "
                "authority until its experiment actually executes"
            ),
        },
        "data10_recipe_sha256": sha256_file(multilingual),
    }


def start(
    *,
    repo_root: Path,
    source_sha: str,
    partial_path: Path,
    checkpoint_root: Path,
    torch_threads: int,
) -> dict[str, Any]:
    _assert_exact_source(repo_root, source_sha)
    if checkpoint_root.exists():
        raise FileExistsError(f"checkpoint root already exists: {checkpoint_root}")
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    bundle = _build_bundle(repo_root=repo_root, source_sha=source_sha)
    model = TwelveSixDecoder(bundle["spec"], bundle["init_spec"])
    trainer = Trainer(model, bundle["trainer_config"], device="cpu")

    phase_started = time.perf_counter()
    initial_loss, validation_tokens, initial_eval_seconds = _evaluate_checked(
        model=model,
        trainer=trainer,
        validation_records=bundle["data"]["validation_records"],
        tokenizer=bundle["data"]["tokenizer"],
    )
    initial_generation = _generation_snapshot(model, bundle["data"]["tokenizer"], 0)
    state = _new_state(
        initial_validation_loss=initial_loss,
        validation_tokens=validation_tokens,
        initial_eval_seconds=initial_eval_seconds,
        initial_generation=initial_generation,
    )
    _run_phase(
        source_sha=source_sha,
        bundle=bundle,
        model=model,
        trainer=trainer,
        state=state,
        checkpoint_root=checkpoint_root,
        stop_tokens=RESUME_TOKENS,
        permit_no_improvement_stop=False,
    )
    if state["stop"] is not None:
        raise RuntimeError(f"TRAIN-41 diverged before mandatory resume boundary: {state['stop']}")
    if trainer.tokens_seen != RESUME_TOKENS:
        raise RuntimeError("start phase failed to reach exact resume token boundary")
    resume_checkpoint = _checkpoint_record_at(state, RESUME_TOKENS)
    if resume_checkpoint is None:
        raise RuntimeError("mandatory resume checkpoint was not retained")
    state["end_to_end_wall_seconds"] = time.perf_counter() - phase_started

    payload = {
        "schema": PARTIAL_SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "start_pid": os.getpid(),
        "controls_hash": bundle["controls_hash"],
        "model_identity_sha256": bundle["spec"].identity_sha256(),
        "parameters": bundle["spec"].parameter_count(),
        "resume_tokens": RESUME_TOKENS,
        "final_tokens": FINAL_TOKENS,
        "resume_checkpoint": resume_checkpoint,
        "state": state,
    }
    return _write_hashed_json(partial_path, payload)


def resume(
    *,
    repo_root: Path,
    source_sha: str,
    partial_path: Path,
    checkpoint_root: Path,
    output_path: Path,
    torch_threads: int,
) -> dict[str, Any]:
    _assert_exact_source(repo_root, source_sha)
    partial = _read_hashed_json(partial_path, schema=PARTIAL_SCHEMA)
    if partial["source_sha"] != source_sha:
        raise RuntimeError("partial source SHA mismatch")
    if int(partial["start_pid"]) == os.getpid():
        raise RuntimeError("mandatory resume must run in a fresh process")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    bundle = _build_bundle(repo_root=repo_root, source_sha=source_sha)
    if bundle["controls_hash"] != partial["controls_hash"]:
        raise RuntimeError("scientific controls drifted across fresh-process resume")

    model = TwelveSixDecoder(bundle["spec"], bundle["init_spec"])
    trainer = Trainer(model, bundle["trainer_config"], device="cpu")
    resume_checkpoint = partial["resume_checkpoint"]
    checkpoint_dir = Path(resume_checkpoint["directory"])
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
        expected_seed=SEED,
    )
    load_seconds = time.perf_counter() - load_started
    if trainer.tokens_seen != RESUME_TOKENS:
        raise RuntimeError("restored Trainer token ledger does not match mandatory resume boundary")
    if int(load_result.manifest["identity"]["tokens_seen"]) != RESUME_TOKENS:
        raise RuntimeError("verified checkpoint identity token ledger mismatch")

    state = dict(partial["state"])
    for key in (
        "evaluation_points",
        "generation_snapshots",
        "step_telemetry",
        "update_ratios",
        "checkpoint_records",
    ):
        state[key] = list(state[key])
    state["checkpoint_load_seconds"] = float(state["checkpoint_load_seconds"]) + load_seconds

    reload_loss, checked_tokens, reload_eval_seconds = _evaluate_checked(
        model=model,
        trainer=trainer,
        validation_records=bundle["data"]["validation_records"],
        tokenizer=bundle["data"]["tokenizer"],
        expected_validation_tokens=int(state["validation_tokens"]),
    )
    resume_point = next(
        point
        for point in state["evaluation_points"]
        if int(point["optimized_tokens"]) == RESUME_TOKENS
    )
    reload_abs_diff = abs(reload_loss - float(resume_point["validation_loss"]))
    if reload_abs_diff > 1e-12:
        raise RuntimeError(f"fresh-process reload validation drift: {reload_abs_diff:.3e}")
    reload_generation = _generation_snapshot(model, bundle["data"]["tokenizer"], RESUME_TOKENS)
    original_generation = next(
        item
        for item in state["generation_snapshots"]
        if int(item["optimized_tokens"]) == RESUME_TOKENS
    )
    if [item["generated_token_ids"] for item in reload_generation["outputs"]] != [
        item["generated_token_ids"] for item in original_generation["outputs"]
    ]:
        raise RuntimeError("fresh-process generation snapshot drift at resume boundary")
    state["evaluation_wall_seconds"] += reload_eval_seconds
    state["generation_wall_seconds"] += float(reload_generation["wall_seconds"])

    prior_end_to_end = float(state["end_to_end_wall_seconds"])
    _run_phase(
        source_sha=source_sha,
        bundle=bundle,
        model=model,
        trainer=trainer,
        state=state,
        checkpoint_root=checkpoint_root,
        stop_tokens=FINAL_TOKENS,
        permit_no_improvement_stop=True,
    )
    state["end_to_end_wall_seconds"] = prior_end_to_end + (time.perf_counter() - phase_started)
    if state["stop"] is None:
        if trainer.tokens_seen != FINAL_TOKENS:
            raise RuntimeError("TRAIN-41 ended without final budget or defined early-stop reason")
        state["stop"] = {
            "kind": "FINAL_BUDGET_REACHED",
            "optimized_tokens": trainer.tokens_seen,
            "reason": "completed the planned long-run token budget",
        }
    if _checkpoint_record_at(state, trainer.tokens_seen) is None:
        _save_checkpoint(
            checkpoint_root=checkpoint_root,
            source_sha=source_sha,
            bundle=bundle,
            model=model,
            trainer=trainer,
            state=state,
            terminal_kind=str(state["stop"]["kind"]),
        )

    optimizer_steps = trainer.optimizer_step
    grad_count = int(state["grad_norm_count"])
    update_values = [float(item["relative_l2_update"]) for item in state["update_ratios"]]
    lr_values = [float(item["learning_rate"]) for item in state["step_telemetry"]]
    training_wall = float(state["training_wall_seconds"])
    end_to_end = float(state["end_to_end_wall_seconds"])
    save_seconds = float(state["checkpoint_save_seconds"])
    load_total = float(state["checkpoint_load_seconds"])
    final_eval = state["evaluation_points"][-1]

    report = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
            "incumbent": "RESEARCH06 PR #183 strict fixed-token 95,568-parameter family",
        },
        "selection": _selection_evidence(repo_root),
        "model": {
            "parameters": bundle["spec"].parameter_count(),
            "model_spec": bundle["spec"].to_dict(),
            "model_identity_sha256": bundle["spec"].identity_sha256(),
            "init_identity_sha256": bundle["init_spec"].identity_sha256(),
        },
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
            "train_stream_bytes": len(bundle["data"]["train_stream"]),
            "approximate_training_stream_reuses": trainer.tokens_seen / len(bundle["data"]["train_stream"]),
            "representative_scale_corpus": False,
        },
        "packing": {**PACKING_DEFINITION, "sha256": PACKING_SHA256},
        "optimizer_recipe": bundle["training_config"],
        "planned": {
            "final_optimized_tokens": FINAL_TOKENS,
            "tokens_per_parameter": FINAL_TOKENS / EXPECTED_PARAMETERS,
            "resume_tokens": RESUME_TOKENS,
            "evaluation_budgets": list(EVALUATION_BUDGETS),
            "retained_checkpoint_budgets": list(RETAINED_CHECKPOINT_BUDGETS),
            "generation_budgets": list(GENERATION_BUDGETS),
        },
        "observed": {
            "optimized_tokens": trainer.tokens_seen,
            "tokens_per_parameter": trainer.tokens_seen / EXPECTED_PARAMETERS,
            "optimizer_steps": optimizer_steps,
            "stop": state["stop"],
            "initial_validation_loss": float(state["initial_validation_loss"]),
            "initial_validation_bpb": float(state["initial_validation_bpb"]),
            "terminal_validation_loss": float(final_eval["validation_loss"]),
            "terminal_validation_bpb": float(final_eval["validation_bpb"]),
            "validation_improvement_nats": (
                float(state["initial_validation_loss"]) - float(final_eval["validation_loss"])
            ),
        },
        "evaluation_curve": state["evaluation_points"],
        "step_telemetry": state["step_telemetry"],
        "gradient": {
            "optimizer_step_samples": grad_count,
            "mean_preclip_global_norm": (
                float(state["grad_norm_sum"]) / grad_count if grad_count else None
            ),
            "max_preclip_global_norm": float(state["grad_norm_max"]),
            "clip_threshold": bundle["trainer_config"].gradient_clip_norm,
            "clip_count": int(state["clip_count"]),
            "clip_rate": int(state["clip_count"]) / optimizer_steps if optimizer_steps else 0.0,
        },
        "learning_rate": {
            "samples": len(lr_values),
            "min": min(lr_values) if lr_values else None,
            "max": max(lr_values) if lr_values else None,
            "terminal": lr_values[-1] if lr_values else None,
        },
        "update_ratio": {
            "samples": len(update_values),
            "mean_relative_l2_update": _mean(update_values),
            "max_relative_l2_update": max(update_values) if update_values else None,
            "points": state["update_ratios"],
        },
        "throughput": {
            "training_wall_seconds": training_wall,
            "end_to_end_wall_seconds": end_to_end,
            "optimized_tokens_per_training_second": trainer.tokens_seen / training_wall,
            "optimized_tokens_per_end_to_end_second": trainer.tokens_seen / end_to_end,
        },
        "checkpoint_overhead": {
            "retained": state["checkpoint_records"],
            "save_seconds_total": save_seconds,
            "load_seconds_total": load_total,
            "save_fraction_of_end_to_end": save_seconds / end_to_end,
            "save_plus_load_fraction_of_end_to_end": (save_seconds + load_total) / end_to_end,
        },
        "generation_snapshots": state["generation_snapshots"],
        "resume": {
            "required": True,
            "fresh_process": True,
            "start_pid": int(partial["start_pid"]),
            "resume_pid": os.getpid(),
            "optimized_tokens": RESUME_TOKENS,
            "checkpoint_id": resume_checkpoint["checkpoint_id"],
            "checkpoint_bytes": int(resume_checkpoint["bytes"]),
            "reload_validation_loss": reload_loss,
            "reload_validation_abs_diff": reload_abs_diff,
            "reload_validation_tokens": checked_tokens,
            "reload_evaluation_wall_seconds": reload_eval_seconds,
            "reload_generation_token_exact": True,
        },
        "memory": {
            "parameter_bytes": _parameter_bytes(model),
            "optimizer_tensor_bytes": _tensor_tree_bytes(trainer.optimizer.state_dict()),
            "process_peak_rss_bytes": _peak_rss_bytes(),
        },
        "stop_policy": {
            "high_grad_norm": HIGH_GRAD_NORM,
            "high_grad_consecutive_updates": HIGH_GRAD_STREAK_LIMIT,
            "validation_divergence_delta_nats": VALIDATION_DIVERGENCE_DELTA_NATS,
            "no_improvement_min_tokens": NO_IMPROVEMENT_MIN_TOKENS,
            "no_improvement_window_evaluations": NO_IMPROVEMENT_WINDOW,
            "no_improvement_delta_nats": NO_IMPROVEMENT_DELTA_NATS,
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
            "from_scratch_random_init": True,
            "base_next_token_training_only": True,
            "held_out_validation": True,
            "validation_tokens_optimized": 0,
            "fixture_recycled_heavily": True,
            "representative_scale_corpus": False,
            "production_tokenizer_frozen": False,
            "optimizer_retuned_for_this_run": False,
            "quality_or_capability_claim": False,
            "stage_or_canonical_config_changed": False,
            "paid_compute": False,
        },
    }
    return _write_hashed_json(output_path, report)


def validate(path: Path, *, expected_source_sha: str) -> dict[str, Any]:
    report = _read_hashed_json(path, schema=SCHEMA)
    if report["source"]["git_sha"] != expected_source_sha:
        raise RuntimeError("TRAIN-41 report source SHA mismatch")
    if int(report["model"]["parameters"]) != EXPECTED_PARAMETERS:
        raise RuntimeError("TRAIN-41 parameter count drift")
    if report["truth_boundary"]["paid_compute"] is not False:
        raise RuntimeError("TRAIN-41 evidence must be LOCAL_FREE")
    if report["truth_boundary"]["representative_scale_corpus"] is not False:
        raise RuntimeError("TRAIN-41 must not overclaim the controlled fixture")
    if report["resume"]["fresh_process"] is not True:
        raise RuntimeError("TRAIN-41 lacks mandatory fresh-process resume proof")
    if float(report["resume"]["reload_validation_abs_diff"]) > 1e-12:
        raise RuntimeError("TRAIN-41 resume validation drift exceeds contract")
    stop_kind = report["observed"]["stop"]["kind"]
    if stop_kind not in {"FINAL_BUDGET_REACHED", "NO_IMPROVEMENT", "DIVERGENCE"}:
        raise RuntimeError(f"invalid TRAIN-41 terminal reason: {stop_kind!r}")
    optimized_tokens = int(report["observed"]["optimized_tokens"])
    if stop_kind == "FINAL_BUDGET_REACHED" and optimized_tokens != FINAL_TOKENS:
        raise RuntimeError("final-budget completion did not hit the exact token target")
    if stop_kind == "NO_IMPROVEMENT" and optimized_tokens < NO_IMPROVEMENT_MIN_TOKENS:
        raise RuntimeError("no-improvement stop occurred before its permitted boundary")
    retained_tokens = {int(item["optimized_tokens"]) for item in report["checkpoint_overhead"]["retained"]}
    if RESUME_TOKENS not in retained_tokens or optimized_tokens not in retained_tokens:
        raise RuntimeError("mandatory resume/terminal checkpoint not retained")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    start_parser = sub.add_parser("start")
    start_parser.add_argument("--repo-root", type=Path, default=Path("."))
    start_parser.add_argument("--source-sha", required=True)
    start_parser.add_argument("--partial", type=Path, required=True)
    start_parser.add_argument("--checkpoint-root", type=Path, required=True)
    start_parser.add_argument("--torch-threads", type=int, default=2)

    resume_parser = sub.add_parser("resume")
    resume_parser.add_argument("--repo-root", type=Path, default=Path("."))
    resume_parser.add_argument("--source-sha", required=True)
    resume_parser.add_argument("--partial", type=Path, required=True)
    resume_parser.add_argument("--checkpoint-root", type=Path, required=True)
    resume_parser.add_argument("--output", type=Path, required=True)
    resume_parser.add_argument("--torch-threads", type=int, default=2)

    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--expected-source-sha", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "start":
        start(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            partial_path=args.partial,
            checkpoint_root=args.checkpoint_root,
            torch_threads=args.torch_threads,
        )
        return 0
    if args.command == "resume":
        resume(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            partial_path=args.partial,
            checkpoint_root=args.checkpoint_root,
            output_path=args.output,
            torch_threads=args.torch_threads,
        )
        return 0
    validate(args.path, expected_source_sha=args.expected_source_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
