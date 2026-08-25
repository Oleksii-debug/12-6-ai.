#!/usr/bin/env python3
"""Execute LOCAL_FREE S0 process-failure/recovery evidence for TRAIN-39."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from twelve_six.checkpoint import (
    bind_checkpoint_identity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    sha256_file,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import PACKING_CONFIG_HASH, PACKING_VERSION
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training.resilience import (
    FailureClass,
    RecoveryPolicy,
    RecoveryStore,
    RunPhase,
    StopLatch,
    install_preemption_handlers,
    recommend_checkpoint_interval,
    run_resilient_training,
)
from twelve_six.training.s0_repeatability import _state_hash

_SCHEMA_VERSION = "12-6.train39-resilience-evidence.v1"
_AUTHORITY = "LOCAL_FREE_CPU_FAILURE_INJECTION_NOT_DISTRIBUTED_RECOVERY"
_EXACT_GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_HARD_EXIT_CODE = 86
_PREEMPT_EXIT_CODE = 75


class TimedTrainer(Trainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer_step_seconds: list[float] = []

    def train_microbatch(self, batch: Mapping[str, torch.Tensor]):
        started = time.perf_counter()
        metrics = super().train_microbatch(batch)
        elapsed = time.perf_counter() - started
        if metrics.optimizer_stepped:
            self.optimizer_step_seconds.append(elapsed)
        return metrics


class TimedRecoveryStore(RecoveryStore):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.checkpoint_seconds: list[float] = []

    def commit_checkpoint(self, trainer: Any, save):
        started = time.perf_counter()
        record = super().commit_checkpoint(trainer, save)
        self.checkpoint_seconds.append(time.perf_counter() - started)
        return record


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no records")
    return rows


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
        raise RuntimeError("TRAIN-39 evidence requires an exact Git checkout") from exc
    if _EXACT_GIT_SHA.fullmatch(value) is None:
        raise ValueError("git HEAD is not an exact lowercase Git object id")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _trainer_config(*, seed: int, max_steps: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-2,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        max_steps=max_steps,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _make_batches(
    train_path: Path,
    tokenizer: ByteTokenizer,
    *,
    max_seq_len: int,
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for row in _load_jsonl(train_path):
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("every S0 training row must contain non-empty text")
        token_ids = tokenizer.encode(text)[:max_seq_len]
        if len(token_ids) < 2:
            raise ValueError("every S0 training row must contain at least two tokens")
        ids = torch.tensor([token_ids], dtype=torch.long)
        batches.append({"input_ids": ids, "labels": ids})
    return batches


def _runtime_context(
    repo_root: Path,
    *,
    source_sha: str,
    seed: int,
    max_steps: int,
    checkpoint_every_steps: int,
) -> dict[str, Any]:
    stage = load_stage_config(repo_root / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    dataset_manifest_path = repo_root / "data/s0/packaged/manifest.json"
    environment_lock_path = repo_root / "requirements/locks/index.json"
    config = _trainer_config(seed=seed, max_steps=max_steps)
    policy = RecoveryPolicy(
        checkpoint_every_steps=checkpoint_every_steps,
        retain_last=3,
        max_restarts=4,
        max_preemptions=4,
        target_checkpoint_overhead_fraction=0.05,
        max_recovery_window_seconds=30.0,
        require_exact_topology_resume=True,
    )
    train_sha256 = sha256_file(train_path)
    dataset_manifest_sha256 = sha256_file(dataset_manifest_path)
    environment_lock_sha256 = sha256_file(environment_lock_path)
    run_manifest = {
        "schema_version": "12-6.train39-run.v1",
        "run_id": f"s0-train39-resilience-{source_sha[:12]}",
        "stage": "S0",
        "run_kind": "local_failure_recovery_evidence",
        "authorization": {
            "class": "LOCAL_FREE",
            "paid_compute_authorized": False,
        },
        "candidate": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
            "modelspec_sha256": hash_json(stage.model.to_dict()),
            "initspec_sha256": hash_json(stage.init.to_dict()),
            "parameter_count": stage.expected_parameters,
        },
        "data": {
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "tokenizer_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "tokenizer_version": tokenizer.identity.version,
            "split_identity": f"train:{train_sha256}",
            "packing_sha256": PACKING_CONFIG_HASH,
            "packing_version": PACKING_VERSION,
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
            "scheduler": {"name": config.scheduler},
            "context_length": stage.model.max_seq_len,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "target_steps": max_steps,
            "checkpoint_interval_steps": checkpoint_every_steps,
        },
        "environment": {"lock_sha256": environment_lock_sha256},
        "recovery": {
            "topology": {
                "backend": "single-process-cpu",
                "world_size": 1,
                "rank_count": 1,
                "resume_policy": "exact_topology",
            },
            "policy": asdict(policy),
            "distributed_rank_loss_recovery": "NOT_IMPLEMENTED_ABORT_ALL_RANKS_AND_RESTART",
        },
    }
    return {
        "stage": stage,
        "tokenizer": tokenizer,
        "train_path": train_path,
        "config": config,
        "policy": policy,
        "run_manifest": run_manifest,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "train_sha256": train_sha256,
        "environment_lock_sha256": environment_lock_sha256,
    }


def _fresh_stack(context: Mapping[str, Any], workspace: Path):
    stage = context["stage"]
    tokenizer = context["tokenizer"]
    config = context["config"]
    policy = context["policy"]
    run_manifest = context["run_manifest"]
    _seed_all(config.seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = TimedTrainer(model, config, device="cpu")
    store = TimedRecoveryStore(workspace, run_manifest=run_manifest, policy=policy)
    batches = _make_batches(
        context["train_path"], tokenizer, max_seq_len=stage.model.max_seq_len
    )
    last_good = store.last_known_good()
    if last_good is not None:
        load_trainer_checkpoint(
            store.checkpoints_dir / last_good.directory,
            model=model,
            trainer=trainer,
            strict_model=True,
            restore_rng=True,
            expected_git_sha=run_manifest["candidate"]["git_sha"],
            expected_model_spec_hash=run_manifest["candidate"]["modelspec_sha256"],
            expected_init_spec_hash=run_manifest["candidate"]["initspec_sha256"],
            expected_tokenizer_hash=tokenizer.identity.config_sha256,
            expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
            expected_dataset_manifest_hash=context["dataset_manifest_sha256"],
            expected_split_identity=f"train:{context['train_sha256']}",
            expected_packing_hash=PACKING_CONFIG_HASH,
            expected_packing_version=PACKING_VERSION,
            expected_run_manifest_hash=hash_json(run_manifest),
            expected_environment_lock_hash=context["environment_lock_sha256"],
            expected_seed=config.seed,
        )
        if trainer.optimizer_step != last_good.optimizer_step:
            raise RuntimeError("loaded trainer step does not equal selected last-known-good")
    return model, trainer, store, batches


def _save_callback(context: Mapping[str, Any], model: Any, trainer: Any):
    stage = context["stage"]
    tokenizer = context["tokenizer"]
    run_manifest = context["run_manifest"]

    def save(path: Path):
        identity = bind_checkpoint_identity(
            run_manifest=run_manifest,
            model_spec=stage.model.to_dict(),
            init_spec=stage.init.to_dict(),
            tokenizer_identity=tokenizer.identity.to_dict(),
            packing_identity={
                "version": PACKING_VERSION,
                "config_sha256": PACKING_CONFIG_HASH,
            },
            step=trainer.optimizer_step,
            tokens_seen=trainer.tokens_seen,
            environment_lock_hash=context["environment_lock_sha256"],
        )
        return save_trainer_checkpoint(
            path,
            model=model,
            trainer=trainer,
            identity=identity,
        )

    return save


def _child_run(args: argparse.Namespace, repo_root: Path) -> int:
    context = _runtime_context(
        repo_root,
        source_sha=args.source_sha,
        seed=args.seed,
        max_steps=args.max_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
    workspace = Path(args.child_workspace).resolve()
    model, trainer, store, batches = _fresh_stack(context, workspace)
    save = _save_callback(context, model, trainer)
    latch = StopLatch()

    def inject(metrics: Any) -> None:
        if not metrics.optimizer_stepped or metrics.optimizer_step != args.fail_step:
            return
        if args.child_mode == "hard-crash":
            os._exit(_HARD_EXIT_CODE)
        if args.child_mode == "preempt":
            os.kill(os.getpid(), signal.SIGTERM)

    started = time.perf_counter()
    with install_preemption_handlers(latch):
        result = run_resilient_training(
            trainer,
            batches,
            store,
            save_checkpoint=save,
            stop_latch=latch,
            after_metrics=inject,
        )
    wall_seconds = time.perf_counter() - started
    state = store.open()
    last_good = store.last_known_good()
    payload = {
        "status": result.status,
        "optimizer_step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
        "attempt": state["attempt"],
        "last_known_good": None if last_good is None else asdict(last_good),
        "model_sha256": _state_hash(model.state_dict()),
        "trainer_sha256": _state_hash(asdict(trainer.state_dict())),
        "optimizer_step_seconds": trainer.optimizer_step_seconds,
        "checkpoint_seconds": store.checkpoint_seconds,
        "wall_seconds": wall_seconds,
        "stop_reason": result.stop_reason,
    }
    _write_json(Path(args.child_output), payload)
    return _PREEMPT_EXIT_CODE if result.status == RunPhase.PREEMPTED.value else 0


def _run_child(
    repo_root: Path,
    *,
    args: argparse.Namespace,
    workspace: Path,
    mode: str,
    fail_step: int,
    output: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source-sha",
        args.source_sha,
        "--seed",
        str(args.seed),
        "--max-steps",
        str(args.max_steps),
        "--checkpoint-every-steps",
        str(args.checkpoint_every_steps),
        "--child-mode",
        mode,
        "--child-workspace",
        str(workspace),
        "--child-output",
        str(output),
        "--fail-step",
        str(fail_step),
    ]
    return subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def _require_exit(result: subprocess.CompletedProcess[str], expected: int, label: str) -> None:
    if result.returncode != expected:
        detail = (result.stdout + "\n" + result.stderr)[-3000:]
        raise RuntimeError(
            f"{label} returned {result.returncode}, expected {expected}:\n{detail}"
        )


def _child_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _store_for_parent(
    repo_root: Path, args: argparse.Namespace, workspace: Path
) -> RecoveryStore:
    context = _runtime_context(
        repo_root,
        source_sha=args.source_sha,
        seed=args.seed,
        max_steps=args.max_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
    )
    return RecoveryStore(
        workspace,
        run_manifest=context["run_manifest"],
        policy=context["policy"],
    )


def _assert_matches_baseline(
    baseline: Mapping[str, Any], recovered: Mapping[str, Any], label: str
) -> None:
    for field in ("model_sha256", "trainer_sha256", "optimizer_step", "tokens_seen"):
        if recovered.get(field) != baseline.get(field):
            raise RuntimeError(f"{label} final {field} differs from uninterrupted baseline")


def _flip_checkpoint_byte(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    if not payload:
        raise RuntimeError(f"cannot corrupt empty checkpoint payload: {path}")
    offset = len(payload) // 2
    payload[offset] ^= 0x01
    path.write_bytes(payload)


def _parent_run(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    if args.max_steps < 8:
        raise ValueError("TRAIN-39 evidence requires max_steps >= 8")
    if args.checkpoint_every_steps != 2:
        raise ValueError("TRAIN-39 controlled failure evidence currently requires cadence=2")

    workspace = Path(args.workspace).resolve()
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    baseline_workspace = workspace / "baseline"
    baseline_output = workspace / "baseline.json"
    baseline_run = _run_child(
        repo_root,
        args=args,
        workspace=baseline_workspace,
        mode="normal",
        fail_step=-1,
        output=baseline_output,
    )
    _require_exit(baseline_run, 0, "uninterrupted baseline")
    baseline = _child_json(baseline_output)

    hard_workspace = workspace / "hard-process-loss"
    hard_first = _run_child(
        repo_root,
        args=args,
        workspace=hard_workspace,
        mode="hard-crash",
        fail_step=5,
        output=workspace / "hard-first.json",
    )
    _require_exit(hard_first, _HARD_EXIT_CODE, "hard process-loss injection")
    hard_store = _store_for_parent(repo_root, args, hard_workspace)
    hard_before = hard_store.last_known_good()
    if hard_before is None or hard_before.optimizer_step != 4:
        raise RuntimeError("hard process loss did not leave step-4 last-known-good")
    hard_store.record_failure(
        FailureClass.PROCESS_LOSS,
        optimizer_step=5,
        detail_code="injected-hard-process-exit",
    )
    hard_output = workspace / "hard-recovered.json"
    hard_second = _run_child(
        repo_root,
        args=args,
        workspace=hard_workspace,
        mode="normal",
        fail_step=-1,
        output=hard_output,
    )
    _require_exit(hard_second, 0, "hard process-loss recovery")
    hard_recovered = _child_json(hard_output)
    _assert_matches_baseline(baseline, hard_recovered, "hard process-loss recovery")

    corrupt_workspace = workspace / "corrupt-latest"
    corrupt_first = _run_child(
        repo_root,
        args=args,
        workspace=corrupt_workspace,
        mode="hard-crash",
        fail_step=7,
        output=workspace / "corrupt-first.json",
    )
    _require_exit(corrupt_first, _HARD_EXIT_CODE, "corrupt-fallback setup crash")
    corrupt_store = _store_for_parent(repo_root, args, corrupt_workspace)
    latest_before_corruption = corrupt_store.last_known_good()
    if latest_before_corruption is None or latest_before_corruption.optimizer_step != 6:
        raise RuntimeError("corruption scenario did not produce step-6 latest checkpoint")
    corrupt_path = (
        corrupt_store.checkpoints_dir
        / latest_before_corruption.directory
        / "weights.safetensors"
    )
    _flip_checkpoint_byte(corrupt_path)
    corrupt_store.record_failure(
        FailureClass.PROCESS_LOSS,
        optimizer_step=7,
        detail_code="injected-process-loss-before-corruption-scan",
    )
    fallback = corrupt_store.last_known_good()
    if fallback is None or fallback.optimizer_step != 4:
        raise RuntimeError("corrupt latest checkpoint did not fall back to verified step 4")
    corrupt_state = corrupt_store.open()
    if latest_before_corruption.directory not in corrupt_state["invalid_checkpoint_directories"]:
        raise RuntimeError("corrupt checkpoint was not recorded as invalid")
    corrupt_output = workspace / "corrupt-recovered.json"
    corrupt_second = _run_child(
        repo_root,
        args=args,
        workspace=corrupt_workspace,
        mode="normal",
        fail_step=-1,
        output=corrupt_output,
    )
    _require_exit(corrupt_second, 0, "corrupt-checkpoint fallback recovery")
    corrupt_recovered = _child_json(corrupt_output)
    _assert_matches_baseline(
        baseline, corrupt_recovered, "corrupt-checkpoint fallback recovery"
    )

    preempt_workspace = workspace / "sigterm-preemption"
    preempt_output = workspace / "preempted.json"
    preempt_first = _run_child(
        repo_root,
        args=args,
        workspace=preempt_workspace,
        mode="preempt",
        fail_step=5,
        output=preempt_output,
    )
    _require_exit(preempt_first, _PREEMPT_EXIT_CODE, "SIGTERM preemption injection")
    preempted = _child_json(preempt_output)
    if preempted.get("status") != RunPhase.PREEMPTED.value:
        raise RuntimeError("SIGTERM injection did not produce PREEMPTED state")
    preempt_store = _store_for_parent(repo_root, args, preempt_workspace)
    preempt_lkg = preempt_store.last_known_good()
    if preempt_lkg is None or preempt_lkg.optimizer_step != 5:
        raise RuntimeError("SIGTERM safe stop did not checkpoint committed step 5")
    preempt_recovered_output = workspace / "preempt-recovered.json"
    preempt_second = _run_child(
        repo_root,
        args=args,
        workspace=preempt_workspace,
        mode="normal",
        fail_step=-1,
        output=preempt_recovered_output,
    )
    _require_exit(preempt_second, 0, "SIGTERM preemption recovery")
    preempt_recovered = _child_json(preempt_recovered_output)
    _assert_matches_baseline(baseline, preempt_recovered, "SIGTERM preemption recovery")

    step_times = [float(value) for value in baseline["optimizer_step_seconds"]]
    checkpoint_times = [float(value) for value in baseline["checkpoint_seconds"]]
    if not step_times or not checkpoint_times:
        raise RuntimeError("baseline timing evidence is empty")
    if not all(math.isfinite(value) and value > 0.0 for value in step_times + checkpoint_times):
        raise RuntimeError("baseline timing evidence contains non-positive/non-finite values")
    median_step = statistics.median(step_times)
    median_checkpoint = statistics.median(checkpoint_times)
    cadence = recommend_checkpoint_interval(
        optimizer_step_seconds=median_step,
        checkpoint_seconds=median_checkpoint,
        target_overhead_fraction=0.05,
        max_recovery_window_seconds=30.0,
    )
    measured_overhead = sum(checkpoint_times) / (sum(checkpoint_times) + sum(step_times))

    hard_state = hard_store.open()
    corrupt_final_state = corrupt_store.open()
    preempt_final_state = preempt_store.open()
    evidence = {
        "schema_version": _SCHEMA_VERSION,
        "authority": _AUTHORITY,
        "identity": {
            "repository": "Oleksii-debug/12-6-ai.",
            "source_sha": args.source_sha,
            "stage": "S0",
            "max_steps": args.max_steps,
            "seed": args.seed,
            "failure_test_checkpoint_every_steps": args.checkpoint_every_steps,
        },
        "baseline": {
            "model_sha256": baseline["model_sha256"],
            "trainer_sha256": baseline["trainer_sha256"],
            "optimizer_step": baseline["optimizer_step"],
            "tokens_seen": baseline["tokens_seen"],
        },
        "failure_injection": {
            "hard_process_loss": {
                "injected_after_optimizer_step": 5,
                "last_known_good_before_restart": hard_before.optimizer_step,
                "recovered_attempt": hard_recovered["attempt"],
                "exact_final_state_match": True,
                "final_phase": hard_state["phase"],
            },
            "corrupt_latest_checkpoint": {
                "injected_after_optimizer_step": 7,
                "corrupted_checkpoint_step": latest_before_corruption.optimizer_step,
                "fallback_optimizer_step": fallback.optimizer_step,
                "invalid_checkpoint_directory": latest_before_corruption.directory,
                "exact_final_state_match": True,
                "final_phase": corrupt_final_state["phase"],
            },
            "sigterm_preemption": {
                "signal": "SIGTERM",
                "requested_after_optimizer_step": 5,
                "safe_checkpoint_step": preempt_lkg.optimizer_step,
                "recovered_attempt": preempt_recovered["attempt"],
                "exact_final_state_match": True,
                "final_phase": preempt_final_state["phase"],
            },
        },
        "checkpoint_timing": {
            "observed_optimizer_step_seconds": step_times,
            "observed_checkpoint_commit_seconds": checkpoint_times,
            "median_optimizer_step_seconds": median_step,
            "median_checkpoint_commit_seconds": median_checkpoint,
            "measured_stress_cadence_overhead_fraction": measured_overhead,
            "stress_test_interval_steps": args.checkpoint_every_steps,
            "cadence_recommendation": asdict(cadence),
            "scope": "S0_LOCAL_CPU_ONLY_REMEASURE_ON_TARGET_CLOUD_STORAGE",
        },
        "claims": {
            "fresh_trainer_required_after_failed_attempt": True,
            "verified_checkpoint_is_resume_authority": True,
            "duplicate_persisted_optimizer_step_avoidance": True,
            "corrupt_latest_falls_back_to_older_verified_checkpoint": True,
            "sigterm_safe_boundary_checkpoint": True,
            "automatic_distributed_rank_replacement": False,
            "multi_node_elastic_recovery": False,
            "paid_compute_executed": False,
        },
    }
    evidence["evidence_sha256"] = hash_json(evidence)
    return evidence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", default="s0-train39-resilience-evidence.json")
    parser.add_argument("--workspace", default=".train39-resilience-work")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--checkpoint-every-steps", type=int, default=2)
    parser.add_argument("--child-mode", choices=("normal", "hard-crash", "preempt"))
    parser.add_argument("--child-workspace")
    parser.add_argument("--child-output")
    parser.add_argument("--fail-step", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if _EXACT_GIT_SHA.fullmatch(args.source_sha) is None:
        raise ValueError("--source-sha must be a full lowercase 40- or 64-hex Git SHA")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.max_steps <= 0 or args.checkpoint_every_steps <= 0:
        raise ValueError("step counts must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    if _git_head(repo_root) != args.source_sha:
        raise ValueError("--source-sha does not equal checkout HEAD")

    if args.child_mode is not None:
        if not args.child_workspace or not args.child_output:
            raise ValueError("child mode requires --child-workspace and --child-output")
        return _child_run(args, repo_root)

    evidence = _parent_run(args, repo_root)
    _write_json(Path(args.output), evidence)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
