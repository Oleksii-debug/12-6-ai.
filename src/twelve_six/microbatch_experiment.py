"""LOCAL_FREE microbatch geometry experiment on the fixed-control ~1M model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .model import InitSpec, TwelveSixDecoder
from .scaling_experiment import (
    _byte_stream,
    _make_batch,
    _read_jsonl,
    _validation_loss,
    controlled_specs,
)
from .tokenization import ByteTokenizer
from .training import Trainer, TrainerConfig

SCHEMA = "12-6.train47-microbatch-experiment.v1"
AUTHORITY = "LOCAL_FREE_CPU_MICROBATCH_EVIDENCE_PROVISIONAL"
LEARNING_RATE = 3e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.0
CLIP_NORM = 1.0
SEED = 1337
SEQUENCE_LENGTH = 64
EFFECTIVE_BATCH_SIZE = 8
MICROBATCH_SIZES = (8, 4, 2, 1)
EXECUTION_UPDATES = 48
SCHEDULER_HORIZON_STEPS = 512
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _rss_bytes() -> int:
    status = Path("/proc/self/status")
    if not status.exists():
        return 0
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def _model_snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def _update_ratio(before: list[torch.Tensor], model: torch.nn.Module) -> float:
    delta_sq = 0.0
    weight_sq = 0.0
    for old, parameter in zip(before, model.parameters(), strict=True):
        current = parameter.detach().float()
        previous = old.float()
        delta = current - previous
        delta_sq += float(torch.sum(delta * delta).item())
        weight_sq += float(torch.sum(previous * previous).item())
    return math.sqrt(delta_sq) / math.sqrt(weight_sq) if weight_sq > 0.0 else 0.0


def _model_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().float().reshape(-1).cpu() for parameter in model.parameters()])


def _optimizer_tensors(trainer: Trainer) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    state = trainer.optimizer.state_dict()["state"]
    for state_id in sorted(state):
        entry = state[state_id]
        for key in sorted(entry):
            value = entry[key]
            if isinstance(value, torch.Tensor):
                tensors.append(value.detach().float().reshape(-1).cpu())
    return tensors


def _max_tensor_diff(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    if len(left) != len(right):
        return math.inf
    result = 0.0
    for first, second in zip(left, right, strict=True):
        if first.shape != second.shape:
            return math.inf
        if first.numel():
            result = max(result, float(torch.max(torch.abs(first - second)).item()))
    return result


def _config(accumulation_steps: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        eps=1e-8,
        max_steps=SCHEDULER_HORIZON_STEPS,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=accumulation_steps,
        gradient_clip_norm=CLIP_NORM,
        precision="fp32",
        seed=SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _load_fixture(repo_root: Path) -> tuple[ByteTokenizer, list[dict[str, Any]], bytes]:
    tokenizer = ByteTokenizer()
    train = _read_jsonl(repo_root / "data/s0/packaged/train.jsonl")
    validation = _read_jsonl(repo_root / "data/s0/packaged/validation.jsonl")
    train_ids = {str(record["id"]) for record in train}
    validation_ids = {str(record["id"]) for record in validation}
    if train_ids & validation_ids:
        raise RuntimeError("train/validation record overlap")
    return tokenizer, validation, _byte_stream(train, tokenizer)


def _effective_batch_digest(train_stream: bytes) -> str:
    digest = hashlib.sha256()
    for update in range(EXECUTION_UPDATES):
        batch = _make_batch(
            train_stream,
            step=update,
            batch_size=EFFECTIVE_BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
        )
        digest.update(batch.numpy().tobytes())
    return digest.hexdigest()


def _run_candidate(
    *,
    microbatch_size: int,
    train_stream: bytes,
    validation_records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
) -> dict[str, Any]:
    if EFFECTIVE_BATCH_SIZE % microbatch_size:
        raise ValueError("microbatch size must divide effective batch size")
    accumulation_steps = EFFECTIVE_BATCH_SIZE // microbatch_size
    spec = controlled_specs()[-1]
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, InitSpec())
    trainer = Trainer(model, _config(accumulation_steps), device="cpu")
    initial_validation, validation_tokens = _validation_loss(
        model, validation_records, tokenizer
    )
    trace: list[dict[str, Any]] = []
    rss_samples = [_rss_bytes()]
    for update in range(EXECUTION_UPDATES):
        effective_batch = _make_batch(
            train_stream,
            step=update,
            batch_size=EFFECTIVE_BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
        )
        before = _model_snapshot(model)
        started = time.perf_counter()
        final_metrics = None
        for offset in range(0, EFFECTIVE_BATCH_SIZE, microbatch_size):
            final_metrics = trainer.train_microbatch(
                {"input_ids": effective_batch[offset : offset + microbatch_size]}
            )
        wall = time.perf_counter() - started
        if final_metrics is None or not final_metrics.optimizer_stepped:
            raise RuntimeError("effective optimizer update did not commit")
        rss_samples.append(_rss_bytes())
        trace.append(
            {
                "optimizer_step": trainer.optimizer_step,
                "optimized_tokens": trainer.tokens_seen,
                "update_loss": final_metrics.update_loss,
                "gradient_norm": final_metrics.grad_norm,
                "update_to_weight_ratio": _update_ratio(before, model),
                "step_wall_seconds": wall,
            }
        )
    final_validation, checked_tokens = _validation_loss(model, validation_records, tokenizer)
    if checked_tokens != validation_tokens:
        raise RuntimeError("validation token count drift")
    measured_wall = sum(float(item["step_wall_seconds"]) for item in trace)
    return {
        "parameters": spec.parameter_count(),
        "model_identity_sha256": spec.identity_sha256(),
        "microbatch_size": microbatch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "effective_predicted_tokens_per_update": EFFECTIVE_BATCH_SIZE
        * (SEQUENCE_LENGTH - 1),
        "optimizer_steps": trainer.optimizer_step,
        "optimized_tokens": trainer.tokens_seen,
        "initial_validation_loss": initial_validation,
        "final_validation_loss": final_validation,
        "mean_gradient_norm": sum(float(item["gradient_norm"]) for item in trace)
        / len(trace),
        "max_gradient_norm": max(float(item["gradient_norm"]) for item in trace),
        "mean_update_to_weight_ratio": sum(
            float(item["update_to_weight_ratio"]) for item in trace
        )
        / len(trace),
        "mean_step_wall_seconds": measured_wall / len(trace),
        "tokens_per_second": trainer.tokens_seen / measured_wall,
        "rss_max_sampled_bytes": max(rss_samples),
        "rss_end_bytes": _rss_bytes(),
        "rss_scope": "same_process_current_RSS_samples_not_fresh_process_peak",
        "trace": trace,
        "_final_model_vector": _model_vector(model),
        "_optimizer_tensors": _optimizer_tensors(trainer),
    }


def _summarize(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = next(run for run in runs if int(run["microbatch_size"]) == EFFECTIVE_BATCH_SIZE)
    baseline_vector = baseline["_final_model_vector"]
    baseline_optimizer = baseline["_optimizer_tensors"]
    public_runs: list[dict[str, Any]] = []
    for run in runs:
        vector = run["_final_model_vector"]
        difference = vector - baseline_vector
        public = {
            key: value
            for key, value in run.items()
            if key not in {"_final_model_vector", "_optimizer_tensors"}
        }
        public["equivalence_vs_microbatch_8"] = {
            "same_effective_batch_order": True,
            "final_parameter_l2_diff": float(torch.linalg.vector_norm(difference).item()),
            "final_parameter_max_abs_diff": float(torch.max(torch.abs(difference)).item()),
            "optimizer_state_max_abs_diff": _max_tensor_diff(
                run["_optimizer_tensors"], baseline_optimizer
            ),
            "final_validation_loss_abs_diff": abs(
                float(run["final_validation_loss"])
                - float(baseline["final_validation_loss"])
            ),
        }
        public_runs.append(public)
    eligible = [
        run
        for run in public_runs
        if float(run["equivalence_vs_microbatch_8"]["final_parameter_max_abs_diff"])
        <= 5e-5
        and float(run["equivalence_vs_microbatch_8"]["final_validation_loss_abs_diff"])
        <= 1e-4
    ]
    fastest = (
        max(eligible, key=lambda run: float(run["tokens_per_second"]))
        if eligible
        else baseline
    )
    decision = {
        "local_free_microbatch_size": int(fastest["microbatch_size"]),
        "local_free_gradient_accumulation_steps": int(
            fastest["gradient_accumulation_steps"]
        ),
        "selection_rule": (
            "fastest observed CPU candidate within final-parameter <=5e-5 max-abs and "
            "validation <=1e-4 absolute-difference tolerances versus microbatch 8"
        ),
        "target_gpu_hypothesis": (
            "later measure the largest microbatch that fits target-GPU VRAM while preserving "
            "the same effective tokens/update; do not transfer CPU throughput ranking to GPU"
        ),
        "target_gpu_status": "NOT_TESTED",
    }
    return public_runs, decision


def run_experiment(repo_root: Path, source_sha: str, output: Path) -> dict[str, Any]:
    if not _HEX40.fullmatch(source_sha):
        raise ValueError("source_sha must be a lowercase 40-hex Git SHA")
    observed = _git_head(repo_root)
    if observed != source_sha:
        raise RuntimeError(f"exact source checkout mismatch: {observed} != {source_sha}")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    tokenizer, validation_records, train_stream = _load_fixture(repo_root)
    batch_trace_sha256 = _effective_batch_digest(train_stream)
    raw_runs = [
        _run_candidate(
            microbatch_size=size,
            train_stream=train_stream,
            validation_records=validation_records,
            tokenizer=tokenizer,
        )
        for size in MICROBATCH_SIZES
    ]
    runs, decision = _summarize(raw_runs)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "paid_compute": False,
        },
        "controls": {
            "learning_rate": LEARNING_RATE,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": CLIP_NORM,
            "seed": SEED,
            "precision": "fp32",
            "scheduler": "constant",
            "scheduler_horizon_steps": SCHEDULER_HORIZON_STEPS,
            "execution_updates": EXECUTION_UPDATES,
            "sequence_length": SEQUENCE_LENGTH,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "microbatch_sizes": list(MICROBATCH_SIZES),
            "effective_batch_trace_sha256": batch_trace_sha256,
            "same_effective_batch_order": True,
        },
        "runs": runs,
        "decision": decision,
        "truth_boundary": {
            "cpu_only": True,
            "gpu_behavior_claimed": False,
            "gpu_executed": False,
            "rss_is_fresh_process_peak": False,
            "tiny_recycled_fixture": True,
            "stage_freeze": False,
            "paid_compute": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def validate_report(report: dict[str, Any], expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("schema/authority mismatch")
    source_sha = report.get("source", {}).get("git_sha")
    if not isinstance(source_sha, str) or not _HEX40.fullmatch(source_sha):
        raise ValueError("invalid source SHA")
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ValueError("source SHA mismatch")
    controls = report["controls"]
    if controls.get("learning_rate") != LEARNING_RATE:
        raise ValueError("learning-rate drift")
    if tuple(controls.get("betas", [])) != BETAS or controls.get("weight_decay") != 0.0:
        raise ValueError("optimizer identity drift")
    if controls.get("same_effective_batch_order") is not True:
        raise ValueError("data-order proof missing")
    if controls.get("scheduler_horizon_steps") <= controls.get("execution_updates"):
        raise ValueError("scheduler horizon collapsed to experiment length")
    runs = report.get("runs")
    if not isinstance(runs, list) or [run["microbatch_size"] for run in runs] != list(
        MICROBATCH_SIZES
    ):
        raise ValueError("microbatch grid drift")
    for run in runs:
        if run["microbatch_size"] * run["gradient_accumulation_steps"] != EFFECTIVE_BATCH_SIZE:
            raise ValueError("effective batch drift")
        if run["equivalence_vs_microbatch_8"].get("same_effective_batch_order") is not True:
            raise ValueError("run data-order proof missing")
    truth = report["truth_boundary"]
    if truth.get("cpu_only") is not True or truth.get("gpu_behavior_claimed") is not False:
        raise ValueError("CPU/GPU truth boundary weakened")
    supplied_hash = report["report_sha256"]
    unsigned = dict(report)
    unsigned.pop("report_sha256")
    if supplied_hash != _canonical_hash(unsigned):
        raise ValueError("report self-hash mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repo-root", type=Path, default=Path("."))
    run_parser.add_argument("--source-sha", required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--expected-source-sha")
    args = parser.parse_args(argv)
    if args.command == "run":
        report = run_experiment(args.repo_root.resolve(), args.source_sha, args.output)
        validate_report(report, expected_source_sha=args.source_sha)
        print(json.dumps({"decision": report["decision"], "report_sha256": report["report_sha256"]}))
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("report must be a JSON object")
    validate_report(report, expected_source_sha=args.expected_source_sha)
    print(f"{SCHEMA}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
