"""LOCAL_FREE warmup and microbatch experiments for the fixed-control 100K-1M family."""

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
from typing import Any, Literal

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

SCHEMA = "12-6.train43-train47-experiment.v1"
AUTHORITY = "LOCAL_FREE_CPU_EXPERIMENTAL_EVIDENCE"
BASE_LR = 3e-4
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.0
CLIP_NORM = 1.0
SCHEDULER_HORIZON_STEPS = 512
WARMUP_STEPS = (0, 8, 32)
WARMUP_UPDATES = 128
MICROBATCH_UPDATES = 48
EFFECTIVE_BATCH_SIZE = 8
SEQUENCE_LENGTH = 64
EVAL_STEPS = (1, 2, 4, 8, 16, 32, 64, 96, 128)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
SchedulerName = Literal["constant", "cosine"]


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _rss_bytes() -> int:
    status = Path("/proc/self/status")
    if not status.exists():
        return 0
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def _parameter_snapshot(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def _update_ratio(before: list[torch.Tensor], model: torch.nn.Module) -> float:
    delta_sq = 0.0
    weight_sq = 0.0
    params = [p.detach() for p in model.parameters() if p.requires_grad]
    for old, new in zip(before, params, strict=True):
        delta = new.float() - old.float()
        delta_sq += float(torch.sum(delta * delta).item())
        weight_sq += float(torch.sum(old.float() * old.float()).item())
    if weight_sq <= 0.0:
        return math.inf if delta_sq > 0.0 else 0.0
    return math.sqrt(delta_sq) / math.sqrt(weight_sq)


def _final_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().float().reshape(-1).cpu() for p in model.parameters()])


def _optimizer_tensors(trainer: Trainer) -> list[torch.Tensor]:
    state = trainer.optimizer.state_dict()["state"]
    tensors: list[torch.Tensor] = []
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
    maximum = 0.0
    for a, b in zip(left, right, strict=True):
        if a.shape != b.shape:
            return math.inf
        if a.numel():
            maximum = max(maximum, float(torch.max(torch.abs(a - b)).item()))
    return maximum


def _trainer_config(
    *, warmup_steps: int, accumulation: int, scheduler: SchedulerName
) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=BASE_LR,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        eps=1e-8,
        max_steps=SCHEDULER_HORIZON_STEPS,
        warmup_steps=warmup_steps,
        scheduler=scheduler,
        gradient_accumulation_steps=accumulation,
        gradient_clip_norm=CLIP_NORM,
        precision="fp32",
        seed=1337,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _load_fixture(
    repo_root: Path,
) -> tuple[ByteTokenizer, list[dict[str, Any]], list[dict[str, Any]], bytes]:
    tokenizer = ByteTokenizer()
    train = _read_jsonl(repo_root / "data/s0/packaged/train.jsonl")
    validation = _read_jsonl(repo_root / "data/s0/packaged/validation.jsonl")
    train_ids = {str(r["id"]) for r in train}
    validation_ids = {str(r["id"]) for r in validation}
    if train_ids & validation_ids:
        raise RuntimeError("train/validation overlap")
    return tokenizer, train, validation, _byte_stream(train, tokenizer)


def _warmup_run(
    *,
    spec_index: int,
    warmup_steps: int,
    train_stream: bytes,
    validation_records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
) -> dict[str, Any]:
    specs = controlled_specs()
    spec = (specs[0], specs[-1])[spec_index]
    random.seed(1337)
    torch.manual_seed(1337)
    model = TwelveSixDecoder(spec, InitSpec())
    config = _trainer_config(
        warmup_steps=warmup_steps,
        accumulation=1,
        scheduler="cosine",
    )
    trainer = Trainer(model, config, device="cpu")
    initial_val, validation_tokens = _validation_loss(model, validation_records, tokenizer)
    trace: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    first_recovery_tokens: int | None = None
    for update in range(WARMUP_UPDATES):
        batch = _make_batch(
            train_stream,
            step=update,
            batch_size=EFFECTIVE_BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
        )
        before = _parameter_snapshot(model)
        started = time.perf_counter()
        metrics = trainer.train_microbatch({"input_ids": batch})
        wall = time.perf_counter() - started
        grad_norm = float(metrics.grad_norm or 0.0)
        point = {
            "optimizer_step": metrics.optimizer_step,
            "optimized_tokens": trainer.tokens_seen,
            "update_loss": metrics.update_loss,
            "grad_norm": metrics.grad_norm,
            "learning_rate": metrics.learning_rate,
            "clipped": grad_norm > CLIP_NORM,
            "update_ratio": _update_ratio(before, model),
            "step_wall_seconds": wall,
        }
        trace.append(point)
        if metrics.optimizer_step in EVAL_STEPS:
            val_loss, checked = _validation_loss(model, validation_records, tokenizer)
            if checked != validation_tokens:
                raise RuntimeError("validation token count drift")
            validations.append(
                {
                    "optimizer_step": metrics.optimizer_step,
                    "optimized_tokens": trainer.tokens_seen,
                    "validation_loss": val_loss,
                }
            )
            if first_recovery_tokens is None and val_loss <= initial_val:
                first_recovery_tokens = trainer.tokens_seen
    elapsed = time.perf_counter() - total_started
    early = trace[:16]
    early_vals = [v for v in validations if int(v["optimizer_step"]) <= 16]
    first_loss = float(early[0]["update_loss"])
    final_val, _ = _validation_loss(model, validation_records, tokenizer)
    return {
        "parameters": spec.parameter_count(),
        "model_identity_sha256": spec.identity_sha256(),
        "warmup_steps": warmup_steps,
        "warmup_fraction_of_scheduler_horizon": warmup_steps / SCHEDULER_HORIZON_STEPS,
        "scheduler_horizon_steps": SCHEDULER_HORIZON_STEPS,
        "experiment_updates": WARMUP_UPDATES,
        "initial_validation_loss": initial_val,
        "final_validation_loss": final_val,
        "tokens_to_recover_initial_validation": first_recovery_tokens,
        "early_max_update_loss": max(float(p["update_loss"]) for p in early),
        "early_update_loss_spike_above_first": max(
            0.0, max(float(p["update_loss"]) for p in early) - first_loss
        ),
        "early_max_grad_norm": max(float(p["grad_norm"]) for p in early),
        "early_clip_frequency": sum(bool(p["clipped"]) for p in early) / len(early),
        "overall_clip_frequency": sum(bool(p["clipped"]) for p in trace) / len(trace),
        "early_max_update_ratio": max(float(p["update_ratio"]) for p in early),
        "early_validation_spike_above_initial": max(
            [0.0] + [float(v["validation_loss"]) - initial_val for v in early_vals]
        ),
        "mean_step_wall_seconds": sum(float(p["step_wall_seconds"]) for p in trace) / len(trace),
        "tokens_per_second": trainer.tokens_seen / elapsed,
        "rss_current_bytes": _rss_bytes(),
        "trace": trace,
        "validations": validations,
    }


def _warmup_decision(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_warmup: dict[int, list[dict[str, Any]]] = {}
    for run in runs:
        by_warmup.setdefault(int(run["warmup_steps"]), []).append(run)
    baseline = by_warmup[0]
    candidates: list[dict[str, Any]] = []
    for warmup in WARMUP_STEPS[1:]:
        current = by_warmup[warmup]
        stable_final = all(
            float(c["final_validation_loss"]) <= float(b["final_validation_loss"]) * 1.01
            for c, b in zip(current, baseline, strict=True)
        )
        mean_grad_ratio = sum(float(c["early_max_grad_norm"]) for c in current) / sum(
            float(b["early_max_grad_norm"]) for b in baseline
        )
        clip_reduction = sum(float(b["early_clip_frequency"]) for b in baseline) / len(
            baseline
        ) - sum(float(c["early_clip_frequency"]) for c in current) / len(current)
        loss_spike_reduction = sum(
            float(b["early_update_loss_spike_above_first"]) for b in baseline
        ) / len(baseline) - sum(
            float(c["early_update_loss_spike_above_first"]) for c in current
        ) / len(current)
        material = (
            mean_grad_ratio <= 0.90
            or clip_reduction >= 0.05
            or loss_spike_reduction >= 0.05
        )
        candidates.append(
            {
                "warmup_steps": warmup,
                "stable_final_validation": stable_final,
                "mean_early_max_grad_ratio_vs_no_warmup": mean_grad_ratio,
                "mean_early_clip_frequency_reduction": clip_reduction,
                "mean_early_update_loss_spike_reduction": loss_spike_reduction,
                "material_stability_improvement": material,
            }
        )
    accepted = [
        c
        for c in candidates
        if c["stable_final_validation"] and c["material_stability_improvement"]
    ]
    if accepted:
        chosen = min(accepted, key=lambda x: int(x["warmup_steps"]))
        rule = (
            f"use_{chosen['warmup_steps']}_warmup_steps_at_lr_3e-4_"
            "for_observed_100k_to_1m_local_family"
        )
    else:
        chosen = {"warmup_steps": 0}
        rule = "no_warmup_required_at_lr_3e-4_for_observed_100k_to_1m_local_family"
    return {"rule": rule, "chosen": chosen, "candidate_assessments": candidates}


def _microbatch_run(
    *,
    microbatch_size: int,
    train_stream: bytes,
    validation_records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
) -> dict[str, Any]:
    if EFFECTIVE_BATCH_SIZE % microbatch_size:
        raise ValueError("microbatch must divide effective batch")
    rss_before = _rss_bytes()
    accumulation = EFFECTIVE_BATCH_SIZE // microbatch_size
    spec = controlled_specs()[-1]
    random.seed(1337)
    torch.manual_seed(1337)
    model = TwelveSixDecoder(spec, InitSpec())
    trainer = Trainer(
        model,
        _trainer_config(warmup_steps=0, accumulation=accumulation, scheduler="constant"),
        device="cpu",
    )
    initial_val, _ = _validation_loss(model, validation_records, tokenizer)
    trace: list[dict[str, Any]] = []
    rss_samples = [_rss_bytes()]
    start = time.perf_counter()
    for update in range(MICROBATCH_UPDATES):
        effective = _make_batch(
            train_stream,
            step=update,
            batch_size=EFFECTIVE_BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
        )
        before = _parameter_snapshot(model)
        update_started = time.perf_counter()
        final_metrics = None
        for offset in range(0, EFFECTIVE_BATCH_SIZE, microbatch_size):
            final_metrics = trainer.train_microbatch(
                {"input_ids": effective[offset : offset + microbatch_size]}
            )
        if final_metrics is None or not final_metrics.optimizer_stepped:
            raise RuntimeError("effective update did not commit")
        wall = time.perf_counter() - update_started
        rss_samples.append(_rss_bytes())
        trace.append(
            {
                "optimizer_step": trainer.optimizer_step,
                "optimized_tokens": trainer.tokens_seen,
                "update_loss": final_metrics.update_loss,
                "grad_norm": final_metrics.grad_norm,
                "update_ratio": _update_ratio(before, model),
                "step_wall_seconds": wall,
            }
        )
    elapsed = time.perf_counter() - start
    final_val, _ = _validation_loss(model, validation_records, tokenizer)
    return {
        "parameters": spec.parameter_count(),
        "microbatch_size": microbatch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "effective_tokens_per_update": EFFECTIVE_BATCH_SIZE * (SEQUENCE_LENGTH - 1),
        "optimizer_steps": trainer.optimizer_step,
        "optimized_tokens": trainer.tokens_seen,
        "initial_validation_loss": initial_val,
        "final_validation_loss": final_val,
        "mean_grad_norm": sum(float(p["grad_norm"]) for p in trace) / len(trace),
        "mean_update_ratio": sum(float(p["update_ratio"]) for p in trace) / len(trace),
        "mean_step_wall_seconds": sum(float(p["step_wall_seconds"]) for p in trace) / len(trace),
        "tokens_per_second": trainer.tokens_seen / elapsed,
        "rss_before_run_bytes": rss_before,
        "rss_max_sampled_bytes": max(rss_samples),
        "rss_end_bytes": _rss_bytes(),
        "rss_measurement_scope": "same_process_current_RSS_samples_not_fresh_process_peak",
        "final_model_vector": _final_vector(model),
        "optimizer_tensors": _optimizer_tensors(trainer),
        "trace": trace,
    }


def _microbatch_summary(
    runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = next(
        run for run in runs if int(run["microbatch_size"]) == EFFECTIVE_BATCH_SIZE
    )
    baseline_vec = baseline["final_model_vector"]
    baseline_opt = baseline["optimizer_tensors"]
    summarized: list[dict[str, Any]] = []
    for run in runs:
        vector = run["final_model_vector"]
        diff = vector - baseline_vec
        item = {
            k: v
            for k, v in run.items()
            if k not in {"final_model_vector", "optimizer_tensors"}
        }
        item["equivalence_vs_fullbatch"] = {
            "same_effective_batch_order": True,
            "final_parameter_l2_diff": float(torch.linalg.vector_norm(diff).item()),
            "final_parameter_max_abs_diff": float(torch.max(torch.abs(diff)).item()),
            "optimizer_state_max_abs_diff": _max_tensor_diff(
                run["optimizer_tensors"], baseline_opt
            ),
            "final_validation_loss_abs_diff": abs(
                float(run["final_validation_loss"])
                - float(baseline["final_validation_loss"])
            ),
        }
        summarized.append(item)
    eligible = [
        r
        for r in summarized
        if float(r["equivalence_vs_fullbatch"]["final_parameter_max_abs_diff"])
        <= 5e-5
        and float(r["equivalence_vs_fullbatch"]["final_validation_loss_abs_diff"])
        <= 1e-4
    ]
    best = (
        max(eligible, key=lambda r: float(r["tokens_per_second"]))
        if eligible
        else summarized[0]
    )
    recommendation = {
        "local_free_microbatch_size": int(best["microbatch_size"]),
        "local_free_accumulation_steps": int(best["gradient_accumulation_steps"]),
        "basis": (
            "fastest observed CPU configuration within explicit numerical-equivalence "
            "tolerances on an identical effective-batch trace"
        ),
        "target_gpu_hypothesis": (
            "test the largest microbatch that fits measured VRAM while holding effective "
            "tokens/update fixed; larger GPU microbatches may improve kernel efficiency, "
            "but no GPU behavior is inferred from this CPU run"
        ),
        "gpu_evidence_status": "NOT_TESTED",
    }
    return summarized, recommendation


def run_experiment(
    *, repo_root: Path, output_path: Path, expected_source_sha: str | None = None
) -> dict[str, Any]:
    observed_source_sha = _git_head(repo_root)
    if not _HEX40.fullmatch(observed_source_sha):
        raise RuntimeError("observed Git HEAD is not a lowercase 40-hex SHA")
    if expected_source_sha is not None and observed_source_sha != expected_source_sha:
        raise RuntimeError(
            f"exact-checkout mismatch: expected {expected_source_sha}, observed {observed_source_sha}"
        )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    tokenizer, _train_records, validation_records, train_stream = _load_fixture(repo_root)
    warmup_runs = [
        _warmup_run(
            spec_index=scale_index,
            warmup_steps=warmup,
            train_stream=train_stream,
            validation_records=validation_records,
            tokenizer=tokenizer,
        )
        for scale_index in (0, 1)
        for warmup in WARMUP_STEPS
    ]
    micro_raw = [
        _microbatch_run(
            microbatch_size=size,
            train_stream=train_stream,
            validation_records=validation_records,
            tokenizer=tokenizer,
        )
        for size in (8, 4, 2, 1)
    ]
    micro_runs, micro_recommendation = _microbatch_summary(micro_raw)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": observed_source_sha,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "device": "cpu",
            "paid_compute": False,
        },
        "controls": {
            "learning_rate": BASE_LR,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": CLIP_NORM,
            "precision": "fp32",
            "seed": 1337,
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "scheduler_horizon_steps": SCHEDULER_HORIZON_STEPS,
            "warmup_experiment_updates": WARMUP_UPDATES,
            "microbatch_experiment_updates": MICROBATCH_UPDATES,
        },
        "warmup": {
            "scales": [95_568, 1_037_696],
            "candidates_steps": list(WARMUP_STEPS),
            "candidates_scheduler_fractions": [
                step / SCHEDULER_HORIZON_STEPS for step in WARMUP_STEPS
            ],
            "runs": warmup_runs,
            "decision": _warmup_decision(warmup_runs),
        },
        "microbatch": {
            "scale_parameters": 1_037_696,
            "same_effective_batch_trace": True,
            "runs": micro_runs,
            "recommendation": micro_recommendation,
        },
        "truth_boundary": {
            "cpu_only": True,
            "gpu_behavior_claimed": False,
            "stage_freeze": False,
            "paid_compute": False,
            "tiny_recycled_fixture": True,
            "rss_is_fresh_process_peak": False,
        },
    }
    report["report_sha256"] = _hash_payload(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def validate_report(
    report: dict[str, Any], *, expected_source_sha: str | None = None
) -> None:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("schema/authority mismatch")
    source_sha = report.get("source", {}).get("git_sha")
    if not isinstance(source_sha, str) or not _HEX40.fullmatch(source_sha):
        raise ValueError("invalid source Git SHA")
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ValueError("report source SHA mismatch")
    controls = report["controls"]
    if controls["learning_rate"] != BASE_LR or tuple(controls["betas"]) != BETAS:
        raise ValueError("optimizer identity drift")
    if controls["weight_decay"] != WEIGHT_DECAY:
        raise ValueError("weight decay drift")
    if controls["scheduler_horizon_steps"] <= controls["warmup_experiment_updates"]:
        raise ValueError("scheduler horizon collapsed to experiment length")
    if report["warmup"]["scales"] != [95_568, 1_037_696]:
        raise ValueError("warmup scale drift")
    runs = report["microbatch"]["runs"]
    if [int(r["microbatch_size"]) for r in runs] != [8, 4, 2, 1]:
        raise ValueError("microbatch grid drift")
    for run in runs:
        if (
            int(run["microbatch_size"])
            * int(run["gradient_accumulation_steps"])
            != EFFECTIVE_BATCH_SIZE
        ):
            raise ValueError("effective batch mismatch")
        if run["equivalence_vs_fullbatch"].get("same_effective_batch_order") is not True:
            raise ValueError("batch-order equivalence was not proven")
    truth = report["truth_boundary"]
    if truth.get("cpu_only") is not True or truth.get("gpu_behavior_claimed") is not False:
        raise ValueError("CPU/GPU truth boundary weakened")
    supplied = report["report_sha256"]
    unsigned = dict(report)
    unsigned.pop("report_sha256")
    if supplied != _hash_payload(unsigned):
        raise ValueError("report hash mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-sha")
    args = parser.parse_args(argv)
    report = run_experiment(
        repo_root=args.repo_root.resolve(),
        output_path=args.output,
        expected_source_sha=args.expected_source_sha,
    )
    validate_report(report, expected_source_sha=args.expected_source_sha)
    print(
        json.dumps(
            {
                "warmup_rule": report["warmup"]["decision"]["rule"],
                "microbatch_recommendation": report["microbatch"]["recommendation"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
