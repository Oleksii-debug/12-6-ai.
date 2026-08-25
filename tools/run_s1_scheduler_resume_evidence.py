#!/usr/bin/env python3
"""Execute deterministic S1 scheduler interruption/resume replay evidence for TRAIN-51."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from twelve_six.checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.resilience import FailureClass, RecoveryPolicy, RecoveryStore
from twelve_six.training.s0_repeatability import _state_hash

_SCHEMA_VERSION = "12-6.train51-scheduler-resume-evidence.v1"
_AUTHORITY = "LOCAL_FREE_CPU_FP32_EXACT_REPLAY"
_EXACT_GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _git_head(repo_root: Path) -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("TRAIN-51 evidence requires an exact Git checkout") from exc
    if _EXACT_GIT_SHA.fullmatch(value) is None:
        raise ValueError("git HEAD is not an exact lowercase Git object id")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_text_rows(path: Path) -> list[str]:
    rows: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, Mapping):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        text = value.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number} requires non-empty text")
        rows.append(text)
    if not rows:
        raise ValueError(f"{path} contains no training rows")
    return rows


def _trainer_config(*, seed: int, max_steps: int, warmup_steps: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-3,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        scheduler="cosine",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _expected_applied_lr(config: TrainerConfig, update_number: int) -> float:
    """Independent 1-based oracle for the LR consumed by optimizer update N."""

    if update_number <= 0 or update_number > config.max_steps:
        raise ValueError("update_number must be in [1, max_steps]")
    scheduler_step_before_update = update_number - 1
    if config.warmup_steps and scheduler_step_before_update < config.warmup_steps:
        factor = (scheduler_step_before_update + 1) / config.warmup_steps
    else:
        denominator = max(config.max_steps - config.warmup_steps, 1)
        progress = min(
            max((scheduler_step_before_update - config.warmup_steps) / denominator, 0.0),
            1.0,
        )
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.learning_rate * factor


def _make_batch_trace(
    train_path: Path,
    *,
    max_steps: int,
    sequence_length: int,
) -> list[dict[str, torch.Tensor]]:
    rows = _load_text_rows(train_path)
    trace: list[dict[str, torch.Tensor]] = []
    for step in range(max_steps):
        raw = rows[step % len(rows)].encode("utf-8")[:sequence_length]
        if len(raw) < 2:
            raise ValueError("controlled training row must encode to at least two bytes")
        ids = torch.tensor([list(raw)], dtype=torch.long)
        trace.append({"input_ids": ids, "labels": ids})
    return trace


def _runtime_context(
    repo_root: Path,
    *,
    source_sha: str,
    seed: int,
    max_steps: int,
    warmup_steps: int,
    sequence_length: int,
) -> dict[str, Any]:
    stage = load_stage_config(repo_root / "configs/stages/s1_100k.json")
    if stage.expected_parameters < 100_000 or stage.expected_parameters > 1_000_000:
        raise RuntimeError("TRAIN-51 requires a ~100K-1M model")
    if sequence_length > stage.model.max_seq_len:
        raise ValueError("sequence_length exceeds S1 max_seq_len")
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    environment_lock_path = repo_root / "requirements/locks/index.json"
    config = _trainer_config(seed=seed, max_steps=max_steps, warmup_steps=warmup_steps)
    model_spec_sha256 = hash_json(stage.model.to_dict())
    init_spec_sha256 = hash_json(stage.init.to_dict())
    dataset_manifest_sha256 = sha256_file(repo_root / "data/s0/packaged/manifest.json")
    train_sha256 = sha256_file(train_path)
    environment_lock_sha256 = sha256_file(environment_lock_path)
    tokenizer_contract_sha256 = hash_json(
        {"kind": "controlled-byte-input", "model_vocab_size": stage.model.vocab_size}
    )
    tokenizer_vocab_sha256 = hash_json(
        {"valid_input_ids": [0, 255], "model_vocab_size": stage.model.vocab_size}
    )
    run_manifest = {
        "schema_version": "12-6.train51-run.v1",
        "run_id": f"s1-train51-scheduler-resume-{source_sha[:12]}",
        "stage": "S1",
        "run_kind": "scheduler_resume_correctness",
        "authorization": {"class": "LOCAL_FREE", "paid_compute_authorized": False},
        "candidate": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
            "modelspec_sha256": model_spec_sha256,
            "initspec_sha256": init_spec_sha256,
            "parameter_count": stage.expected_parameters,
        },
        "data": {
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "train_sha256": train_sha256,
            "controlled_input_contract_sha256": tokenizer_contract_sha256,
            "sequence_length": sequence_length,
        },
        "training": {
            "seed": seed,
            "device": "cpu",
            "precision": "fp32",
            "optimizer": {
                "name": "AdamW",
                "lr": config.learning_rate,
                "betas": list(config.betas),
                "eps": config.eps,
                "weight_decay": config.weight_decay,
            },
            "scheduler": {
                "name": "cosine",
                "warmup_steps": warmup_steps,
                "planned_horizon_steps": max_steps,
            },
            "gradient_accumulation_steps": 1,
            "target_steps": max_steps,
        },
        "environment": {"lock_sha256": environment_lock_sha256},
        "recovery": {
            "topology": {
                "backend": "single-process-cpu",
                "world_size": 1,
                "rank_count": 1,
                "resume_policy": "exact_topology",
            }
        },
    }
    return {
        "stage": stage,
        "config": config,
        "train_path": train_path,
        "run_manifest": run_manifest,
        "run_manifest_sha256": hash_json(run_manifest),
        "model_spec_sha256": model_spec_sha256,
        "init_spec_sha256": init_spec_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "environment_lock_sha256": environment_lock_sha256,
        "tokenizer_contract_sha256": tokenizer_contract_sha256,
        "tokenizer_vocab_sha256": tokenizer_vocab_sha256,
        "batch_trace": _make_batch_trace(
            train_path, max_steps=max_steps, sequence_length=sequence_length
        ),
    }


def _fresh_stack(context: Mapping[str, Any]) -> tuple[TwelveSixDecoder, Trainer]:
    config = context["config"]
    stage = context["stage"]
    _seed_all(config.seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(model, config, device="cpu")
    return model, trainer


def _checkpoint_identity(context: Mapping[str, Any], trainer: Trainer) -> CheckpointIdentity:
    config = context["config"]
    stage = context["stage"]
    run_manifest = context["run_manifest"]
    training_config = {
        "run_id": run_manifest["run_id"],
        "stage": "S1",
        "init_spec_sha256": context["init_spec_sha256"],
        "trainer": asdict(config),
        "data_trace_sha256": context["run_manifest"]["data"]["train_sha256"],
    }
    return CheckpointIdentity(
        git_sha=run_manifest["candidate"]["git_sha"],
        model_spec=stage.model.to_dict(),
        parameter_count=stage.expected_parameters,
        tokenizer_hash=context["tokenizer_contract_sha256"],
        tokenizer_vocab_hash=context["tokenizer_vocab_sha256"],
        dataset_manifest_hash=context["dataset_manifest_sha256"],
        run_manifest_hash=context["run_manifest_sha256"],
        training_config=training_config,
        seed=config.seed,
        precision="fp32",
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "lr": config.learning_rate,
            "betas": list(config.betas),
            "eps": config.eps,
            "weight_decay": config.weight_decay,
        },
        scheduler={
            "name": "LambdaLR-cosine",
            "warmup_steps": config.warmup_steps,
            "planned_horizon_steps": config.max_steps,
        },
        environment_lock_hash=context["environment_lock_sha256"],
    )


def _save_checkpoint(
    path: Path,
    *,
    context: Mapping[str, Any],
    model: TwelveSixDecoder,
    trainer: Trainer,
) -> dict[str, Any]:
    return save_trainer_checkpoint(
        path,
        model=model,
        trainer=trainer,
        identity=_checkpoint_identity(context, trainer),
    )


def _load_checkpoint(
    path: Path,
    *,
    context: Mapping[str, Any],
    model: TwelveSixDecoder,
    trainer: Trainer,
) -> None:
    load_trainer_checkpoint(
        path,
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=context["run_manifest"]["candidate"]["git_sha"],
        expected_model_spec_hash=context["model_spec_sha256"],
        expected_init_spec_hash=context["init_spec_sha256"],
        expected_tokenizer_hash=context["tokenizer_contract_sha256"],
        expected_tokenizer_vocab_hash=context["tokenizer_vocab_sha256"],
        expected_dataset_manifest_hash=context["dataset_manifest_sha256"],
        expected_run_manifest_hash=context["run_manifest_sha256"],
        expected_environment_lock_hash=context["environment_lock_sha256"],
        expected_seed=context["config"].seed,
    )


def _run_steps(
    trainer: Trainer,
    batches: Sequence[Mapping[str, torch.Tensor]],
    *,
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    config = trainer.config
    trace: list[dict[str, Any]] = []
    for batch_index in range(start, end):
        metrics = trainer.train_microbatch(batches[batch_index])
        if not metrics.optimizer_stepped:
            raise RuntimeError("TRAIN-51 controlled run requires one optimizer step per batch")
        expected_step = batch_index + 1
        if metrics.optimizer_step != expected_step:
            raise RuntimeError(
                f"optimizer step drift: got {metrics.optimizer_step}, expected {expected_step}"
            )
        expected_lr = _expected_applied_lr(config, expected_step)
        if not math.isclose(metrics.learning_rate, expected_lr, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError(
                f"LR application off-by-one at update {expected_step}: "
                f"applied={metrics.learning_rate:.17g}, expected={expected_lr:.17g}"
            )
        if trainer.scheduler is None:
            raise RuntimeError("cosine TRAIN-51 run requires scheduler state")
        if trainer.scheduler.last_epoch != metrics.optimizer_step:
            raise RuntimeError(
                "scheduler phase/counter mismatch after committed update: "
                f"last_epoch={trainer.scheduler.last_epoch}, "
                f"optimizer_step={metrics.optimizer_step}"
            )
        trace.append(
            {
                "optimizer_step": metrics.optimizer_step,
                "tokens_seen": trainer.tokens_seen,
                "applied_lr": metrics.learning_rate,
                "oracle_lr": expected_lr,
                "scheduler_last_epoch_after": trainer.scheduler.last_epoch,
                "next_lr_after": float(trainer.optimizer.param_groups[0]["lr"]),
            }
        )
    return trace


def _fingerprints(model: TwelveSixDecoder, trainer: Trainer) -> dict[str, str]:
    return {
        "model_sha256": _state_hash(model.state_dict()),
        "trainer_sha256": _state_hash(asdict(trainer.state_dict())),
    }


def _assert_trace_exact(
    control: Sequence[Mapping[str, Any]],
    replay: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    if len(control) != len(replay):
        raise RuntimeError(f"{label} trace length differs from control")
    fields = (
        "optimizer_step",
        "tokens_seen",
        "applied_lr",
        "oracle_lr",
        "scheduler_last_epoch_after",
        "next_lr_after",
    )
    for index, (expected, actual) in enumerate(zip(control, replay, strict=True), 1):
        for field in fields:
            if actual[field] != expected[field]:
                raise RuntimeError(
                    f"{label} differs at trace row {index} field {field}: "
                    f"{actual[field]!r} != {expected[field]!r}"
                )


def _run_interrupted_replay(
    *,
    context: Mapping[str, Any],
    workspace: Path,
    split_step: int,
    control_trace: Sequence[Mapping[str, Any]],
    control_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    model, trainer = _fresh_stack(context)
    prefix = _run_steps(trainer, context["batch_trace"], start=0, end=split_step)
    checkpoint = workspace / f"split-{split_step:03d}" / "checkpoint"
    _save_checkpoint(checkpoint, context=context, model=model, trainer=trainer)
    checkpoint_next_lr = float(trainer.optimizer.param_groups[0]["lr"])
    checkpoint_scheduler_epoch = trainer.scheduler.last_epoch if trainer.scheduler else None
    checkpoint_tokens = trainer.tokens_seen

    del trainer
    del model
    gc.collect()

    model, trainer = _fresh_stack(context)
    _load_checkpoint(checkpoint, context=context, model=model, trainer=trainer)
    if trainer.optimizer_step != split_step:
        raise RuntimeError("restored optimizer_step differs from checkpoint split")
    if trainer.tokens_seen != checkpoint_tokens:
        raise RuntimeError("restored tokens_seen differs from checkpoint")
    if trainer.scheduler is None or trainer.scheduler.last_epoch != split_step:
        raise RuntimeError("restored scheduler phase differs from optimizer_step")
    if float(trainer.optimizer.param_groups[0]["lr"]) != checkpoint_next_lr:
        raise RuntimeError("restored next-update LR differs from checkpoint")
    if split_step < trainer.config.max_steps:
        expected_next = control_trace[split_step]["applied_lr"]
        if float(trainer.optimizer.param_groups[0]["lr"]) != expected_next:
            raise RuntimeError("first resumed update would consume the wrong LR")

    suffix = _run_steps(
        trainer,
        context["batch_trace"],
        start=split_step,
        end=trainer.config.max_steps,
    )
    replay = [*prefix, *suffix]
    _assert_trace_exact(control_trace, replay, label=f"split-{split_step}")
    fingerprints = _fingerprints(model, trainer)
    if fingerprints != control_fingerprints:
        raise RuntimeError(f"split-{split_step} final state differs from control")
    return {
        "split_after_optimizer_step": split_step,
        "checkpoint_tokens_seen": checkpoint_tokens,
        "checkpoint_scheduler_last_epoch": checkpoint_scheduler_epoch,
        "checkpoint_next_update_lr": checkpoint_next_lr,
        "first_resumed_update_lr": None if not suffix else suffix[0]["applied_lr"],
        "exact_lr_sequence_match": True,
        "exact_counter_sequence_match": True,
        "exact_final_model_match": True,
        "exact_final_trainer_match": True,
    }


def _flip_checkpoint_byte(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    if not payload:
        raise RuntimeError(f"cannot corrupt empty checkpoint payload: {path}")
    payload[len(payload) // 2] ^= 0x01
    path.write_bytes(payload)


def _run_corrupt_latest_fallback(
    *,
    context: Mapping[str, Any],
    workspace: Path,
    control_trace: Sequence[Mapping[str, Any]],
    control_fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    config = context["config"]
    if config.max_steps < 48:
        raise ValueError("corrupt-latest scenario requires max_steps >= 48")
    policy = RecoveryPolicy(
        checkpoint_every_steps=1,
        retain_last=3,
        max_restarts=2,
        max_preemptions=0,
        target_checkpoint_overhead_fraction=0.2,
        max_recovery_window_seconds=30.0,
        require_exact_topology_resume=True,
    )
    store = RecoveryStore(
        workspace / "corrupt-latest",
        run_manifest=context["run_manifest"],
        policy=policy,
    )
    store.begin_attempt()
    model, trainer = _fresh_stack(context)
    prefix_46 = _run_steps(trainer, context["batch_trace"], start=0, end=46)
    older = store.commit_checkpoint(
        trainer,
        lambda path: _save_checkpoint(path, context=context, model=model, trainer=trainer),
    )
    if older.optimizer_step != 46:
        raise RuntimeError("expected older verified checkpoint at step 46")
    discarded_47 = _run_steps(trainer, context["batch_trace"], start=46, end=47)
    newer = store.commit_checkpoint(
        trainer,
        lambda path: _save_checkpoint(path, context=context, model=model, trainer=trainer),
    )
    if newer.optimizer_step != 47:
        raise RuntimeError("expected newer checkpoint at step 47")

    corrupt_file = store.checkpoints_dir / newer.directory / "weights.safetensors"
    _flip_checkpoint_byte(corrupt_file)
    store.record_failure(
        FailureClass.PROCESS_LOSS,
        optimizer_step=47,
        detail_code="train51-injected-loss-after-corrupt-newest",
    )
    fallback = store.last_known_good()
    if fallback is None or fallback.optimizer_step != 46:
        raise RuntimeError("newer corruption did not select previous verified step-46 LKG")
    reconciled = store.open()
    if newer.directory not in reconciled["invalid_checkpoint_directories"]:
        raise RuntimeError("corrupt newer checkpoint was not recorded invalid")

    del trainer
    del model
    gc.collect()

    model, trainer = _fresh_stack(context)
    _load_checkpoint(
        store.checkpoints_dir / fallback.directory,
        context=context,
        model=model,
        trainer=trainer,
    )
    if trainer.optimizer_step != 46 or trainer.scheduler is None:
        raise RuntimeError("fallback did not restore exact step-46 scheduler state")
    if trainer.scheduler.last_epoch != 46:
        raise RuntimeError("fallback scheduler phase is not step 46")
    store.begin_attempt()
    replayed_suffix = _run_steps(
        trainer, context["batch_trace"], start=46, end=config.max_steps
    )
    store.mark_completed()
    replay = [*prefix_46, *replayed_suffix]
    _assert_trace_exact(control_trace, replay, label="corrupt-latest-fallback")
    fingerprints = _fingerprints(model, trainer)
    if fingerprints != control_fingerprints:
        raise RuntimeError("corrupt-latest fallback final state differs from control")
    return {
        "older_verified_checkpoint_step": older.optimizer_step,
        "newer_corrupted_checkpoint_step": newer.optimizer_step,
        "selected_fallback_step": fallback.optimizer_step,
        "discarded_update_47_lr": discarded_47[0]["applied_lr"],
        "replayed_update_47_lr": replayed_suffix[0]["applied_lr"],
        "invalid_checkpoint_directory": newer.directory,
        "exact_lr_sequence_match_after_rollback": True,
        "exact_counter_sequence_match_after_rollback": True,
        "exact_final_model_match": True,
        "exact_final_trainer_match": True,
        "final_recovery_phase": store.open()["phase"],
    }


def _parent_run(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    if args.max_steps != 48 or args.warmup_steps != 6:
        raise ValueError("TRAIN-51 exact boundary evidence is fixed at 48 steps / 6 warmup steps")
    workspace = Path(args.workspace).resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    context = _runtime_context(
        repo_root,
        source_sha=args.source_sha,
        seed=args.seed,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        sequence_length=args.sequence_length,
    )

    control_model, control_trainer = _fresh_stack(context)
    control_trace = _run_steps(
        control_trainer, context["batch_trace"], start=0, end=args.max_steps
    )
    control_fingerprints = _fingerprints(control_model, control_trainer)
    if control_trainer.scheduler is None or control_trainer.scheduler.last_epoch != args.max_steps:
        raise RuntimeError("control scheduler did not finish at planned horizon")
    if float(control_trainer.optimizer.param_groups[0]["lr"]) != 0.0:
        raise RuntimeError("cosine scheduler did not reach zero after final committed update")

    split_steps = (5, 6, 26, 27, 46, 47)
    replays = [
        _run_interrupted_replay(
            context=context,
            workspace=workspace,
            split_step=split,
            control_trace=control_trace,
            control_fingerprints=control_fingerprints,
        )
        for split in split_steps
    ]
    corrupt_fallback = _run_corrupt_latest_fallback(
        context=context,
        workspace=workspace,
        control_trace=control_trace,
        control_fingerprints=control_fingerprints,
    )

    key_updates = (1, 5, 6, 7, 27, 28, 46, 47, 48)
    boundary_records = [control_trace[index - 1] for index in key_updates]
    evidence = {
        "schema_version": _SCHEMA_VERSION,
        "authority": _AUTHORITY,
        "identity": {
            "repository": "Oleksii-debug/12-6-ai.",
            "source_sha": args.source_sha,
            "stage": "S1",
            "parameter_count": context["stage"].expected_parameters,
            "precision": "fp32",
            "seed": args.seed,
            "max_steps": args.max_steps,
            "warmup_steps": args.warmup_steps,
            "scheduler": "cosine",
            "sequence_length": args.sequence_length,
            "dataset_manifest_sha256": context["dataset_manifest_sha256"],
            "run_manifest_sha256": context["run_manifest_sha256"],
        },
        "scheduler_semantics": {
            "applied_update_rule": "update_N_consumes_lambda_step_N_minus_1",
            "scheduler_step_order": "optimizer_step_then_scheduler_step",
            "checkpoint_boundary": "after_both_optimizer_and_scheduler_committed",
            "planned_horizon_steps": args.max_steps,
            "post_final_next_lr": float(control_trainer.optimizer.param_groups[0]["lr"]),
            "off_by_one_lr_application_detected": False,
        },
        "control": {
            "optimizer_step": control_trainer.optimizer_step,
            "tokens_seen": control_trainer.tokens_seen,
            "scheduler_last_epoch": control_trainer.scheduler.last_epoch,
            **control_fingerprints,
            "lr_sequence": [row["applied_lr"] for row in control_trace],
            "token_counter_sequence": [row["tokens_seen"] for row in control_trace],
            "boundary_records": boundary_records,
        },
        "interrupted_replays": replays,
        "corrupt_latest_fallback": corrupt_fallback,
        "claims": {
            "exact_optimizer_step_replay": True,
            "exact_token_counter_replay": True,
            "exact_applied_lr_replay": True,
            "warmup_boundary_resume_exact": True,
            "cosine_midpoint_resume_exact": True,
            "near_final_resume_exact": True,
            "corrupt_newer_checkpoint_falls_back_to_previous_verified_lkg": True,
            "fresh_model_and_trainer_used_after_interruption": True,
            "new_checkpoint_framework_created": False,
            "trainer_scheduler_semantics_changed": False,
            "paid_compute_executed": False,
        },
    }
    evidence["evidence_sha256"] = hash_json(evidence)
    return evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", default="s1-train51-scheduler-resume-evidence.json")
    parser.add_argument("--workspace", default=".train51-scheduler-resume-work")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--warmup-steps", type=int, default=6)
    parser.add_argument("--sequence-length", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if _EXACT_GIT_SHA.fullmatch(args.source_sha) is None:
        raise ValueError("--source-sha must be a full lowercase 40- or 64-hex Git SHA")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.max_steps <= 0 or args.warmup_steps < 0 or args.warmup_steps > args.max_steps:
        raise ValueError("invalid scheduler step counts")
    if args.sequence_length < 2:
        raise ValueError("sequence_length must be >= 2")
    repo_root = Path(__file__).resolve().parents[1]
    if _git_head(repo_root) != args.source_sha:
        raise ValueError("--source-sha does not equal checkout HEAD")
    evidence = _parent_run(args, repo_root)
    _write_json(Path(args.output), evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
