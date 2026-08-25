"""Focused LOCAL_FREE warmup and AdamW beta2 experiments for the 12-6 family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

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

SCHEMA = "12-6.optimizer-dynamics-experiment.v1"
AUTHORITY = "LOCAL_FREE_OPTIMIZER_DYNAMICS_EVIDENCE_PROVISIONAL"
LEARNING_RATE = 3e-4
BETA1 = 0.9
EPS = 1e-8
WEIGHT_DECAY = 0.0
CLIP_NORM = 1.0
SEED = 1337
BATCH_SIZE = 4
SEQUENCE_LENGTH = 64
TOKENS_PER_STEP = BATCH_SIZE * (SEQUENCE_LENGTH - 1)
WARMUP_STEPS = (0, 8, 32)
WARMUP_EXECUTION_STEPS = 96
SCHEDULE_HORIZON_STEPS = 2048
BETA2_VALUES = (0.95, 0.98, 0.99)
BETA2_EXECUTION_STEPS = 128
EARLY_WINDOW = 16
WARMUP_SPEC_INDICES = (0, 3)  # 95,568 and 1,037,696 parameters
BETA2_SPEC_INDICES = (0, 2)  # 95,568 and 467,808 parameters


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _parameter_l2(model: TwelveSixDecoder) -> float:
    total = 0.0
    for parameter in model.parameters():
        value = parameter.detach().float()
        total += float(torch.sum(value * value).item())
    return math.sqrt(total)


def _update_metrics(
    model: TwelveSixDecoder,
    before: dict[str, torch.Tensor],
    parameter_l2_before: float,
) -> tuple[float, float]:
    squared = 0.0
    for name, parameter in model.named_parameters():
        delta = parameter.detach().float() - before[name].float()
        squared += float(torch.sum(delta * delta).item())
    update_l2 = math.sqrt(squared)
    ratio = update_l2 / parameter_l2_before if parameter_l2_before else math.inf
    return update_l2, ratio


def _second_moment_metrics(trainer: Trainer, beta2: float, step: int) -> dict[str, float]:
    total = 0.0
    count = 0
    maximum = 0.0
    for state in trainer.optimizer.state.values():
        if not isinstance(state, dict):
            continue
        value = state.get("exp_avg_sq")
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().float()
        total += float(torch.sum(tensor).item())
        count += tensor.numel()
        maximum = max(maximum, float(torch.max(tensor).item()))
    if count == 0:
        raise RuntimeError("AdamW second-moment state was not materialized")
    raw_mean = total / count
    correction = 1.0 - beta2**step
    corrected_mean = raw_mean / correction if correction > 0 else math.inf
    return {
        "exp_avg_sq_mean": raw_mean,
        "exp_avg_sq_max": maximum,
        "bias_correction2": correction,
        "bias_corrected_second_moment_mean": corrected_mean,
        "bias_corrected_second_moment_rms": math.sqrt(corrected_mean),
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _first_recovery_tokens(
    progression: list[dict[str, Any]],
    initial_validation_loss: float,
) -> int | None:
    target = initial_validation_loss * 0.99
    for item in progression:
        if float(item["loss"]) <= target:
            return int(item["optimized_tokens"])
    return None


def _loss_spike(progression: list[dict[str, Any]]) -> dict[str, float]:
    early = progression[:EARLY_WINDOW]
    losses = [float(item["loss"]) for item in early]
    if not losses:
        return {"above_first": math.inf, "max_step_jump": math.inf}
    jumps = [losses[index] - losses[index - 1] for index in range(1, len(losses))]
    return {
        "above_first": max(losses) - losses[0],
        "max_step_jump": max(jumps, default=0.0),
    }


def _config(
    *,
    max_steps: int,
    warmup_steps: int,
    scheduler: str,
    beta2: float,
) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(BETA1, beta2),
        eps=EPS,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        scheduler=scheduler,  # type: ignore[arg-type]
        gradient_accumulation_steps=1,
        gradient_clip_norm=CLIP_NORM,
        precision="fp32",
        seed=SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _evaluation_steps(execution_steps: int) -> set[int]:
    candidates = {4, 8, 16, 32, 64, 96, execution_steps}
    return {step for step in candidates if 0 < step <= execution_steps}


def _run_one(
    *,
    spec_index: int,
    train_stream: bytes,
    validation_records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    execution_steps: int,
    config: TrainerConfig,
    beta2: float,
    capture_second_moment: bool,
) -> dict[str, Any]:
    spec = controlled_specs()[spec_index]
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, InitSpec())
    trainer = Trainer(model, config, device="cpu")
    initial_validation_loss, validation_tokens = _validation_loss(
        model, validation_records, tokenizer
    )
    progression: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = [
        {
            "optimizer_step": 0,
            "optimized_tokens": 0,
            "validation_loss": initial_validation_loss,
        }
    ]
    eval_steps = _evaluation_steps(execution_steps)
    for step in range(execution_steps):
        batch = _make_batch(
            train_stream,
            step=step,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
        )
        before = _snapshot(model)
        weight_l2 = _parameter_l2(model)
        metrics = trainer.train_microbatch({"input_ids": batch})
        update_l2, update_ratio = _update_metrics(model, before, weight_l2)
        item: dict[str, Any] = {
            "optimizer_step": metrics.optimizer_step,
            "optimized_tokens": trainer.tokens_seen,
            "loss": metrics.loss,
            "learning_rate": metrics.learning_rate,
            "gradient_norm": metrics.grad_norm,
            "clip_would_activate": (
                metrics.grad_norm is not None and metrics.grad_norm > CLIP_NORM
            ),
            "update_l2": update_l2,
            "update_to_weight_ratio": update_ratio,
        }
        if capture_second_moment:
            item["second_moment"] = _second_moment_metrics(
                trainer, beta2, metrics.optimizer_step
            )
        progression.append(item)
        if metrics.optimizer_step in eval_steps:
            validation_loss, _ = _validation_loss(model, validation_records, tokenizer)
            evaluations.append(
                {
                    "optimizer_step": metrics.optimizer_step,
                    "optimized_tokens": trainer.tokens_seen,
                    "validation_loss": validation_loss,
                }
            )

    grad_norms = [float(item["gradient_norm"]) for item in progression]
    update_ratios = [float(item["update_to_weight_ratio"]) for item in progression]
    early_grads = grad_norms[:EARLY_WINDOW]
    early_updates = update_ratios[:EARLY_WINDOW]
    spike = _loss_spike(progression)
    summary: dict[str, Any] = {
        "parameters": spec.parameter_count(),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": float(evaluations[-1]["validation_loss"]),
        "validation_tokens": validation_tokens,
        "training_loss_first": float(progression[0]["loss"]),
        "training_loss_last": float(progression[-1]["loss"]),
        "early_loss_spike_above_first": spike["above_first"],
        "early_loss_max_step_jump": spike["max_step_jump"],
        "gradient_norm_max_early": max(early_grads),
        "gradient_norm_median": statistics.median(grad_norms),
        "clip_frequency_early": sum(
            bool(item["clip_would_activate"]) for item in progression[:EARLY_WINDOW]
        )
        / len(progression[:EARLY_WINDOW]),
        "clip_frequency_all": sum(
            bool(item["clip_would_activate"]) for item in progression
        )
        / len(progression),
        "update_to_weight_ratio_max_early": max(early_updates),
        "update_to_weight_ratio_p95": _percentile(update_ratios, 0.95),
        "tokens_to_1pct_better_than_initial_validation": _first_recovery_tokens(
            progression, initial_validation_loss
        ),
        "finite": all(
            math.isfinite(float(item["loss"]))
            and math.isfinite(float(item["gradient_norm"]))
            and math.isfinite(float(item["update_to_weight_ratio"]))
            for item in progression
        ),
    }
    if capture_second_moment:
        first = progression[0]["second_moment"]
        last = progression[-1]["second_moment"]
        summary["second_moment"] = {
            "first_bias_corrected_rms": first["bias_corrected_second_moment_rms"],
            "final_bias_corrected_rms": last["bias_corrected_second_moment_rms"],
            "final_raw_mean": last["exp_avg_sq_mean"],
            "final_bias_correction2": last["bias_correction2"],
        }
    return {
        "model_identity_sha256": spec.identity_sha256(),
        "model_spec": asdict(spec),
        "config": asdict(config),
        "summary": summary,
        "evaluations": evaluations,
        "progression": progression,
    }


def _same_trace_digest(train_stream: bytes, execution_steps: int) -> str:
    digest = hashlib.sha256()
    for step in range(execution_steps):
        batch = _make_batch(
            train_stream,
            step=step,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
        )
        digest.update(batch.numpy().tobytes())
    return digest.hexdigest()


def run(repo_root: Path, source_sha: str, output: Path) -> dict[str, Any]:
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("exact source checkout mismatch")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    tokenizer = ByteTokenizer()
    train_records = _read_jsonl(repo_root / "data/s0/packaged/train.jsonl")
    validation_records = _read_jsonl(repo_root / "data/s0/packaged/validation.jsonl")
    train_stream = _byte_stream(train_records, tokenizer)

    warmup_runs: dict[str, Any] = {}
    for spec_index in WARMUP_SPEC_INDICES:
        parameters = controlled_specs()[spec_index].parameter_count()
        scale_runs: dict[str, Any] = {}
        for warmup_steps in WARMUP_STEPS:
            config = _config(
                max_steps=SCHEDULE_HORIZON_STEPS,
                warmup_steps=warmup_steps,
                scheduler="cosine",
                beta2=0.95,
            )
            scale_runs[str(warmup_steps)] = _run_one(
                spec_index=spec_index,
                train_stream=train_stream,
                validation_records=validation_records,
                tokenizer=tokenizer,
                execution_steps=WARMUP_EXECUTION_STEPS,
                config=config,
                beta2=0.95,
                capture_second_moment=False,
            )
        warmup_runs[str(parameters)] = scale_runs

    beta2_runs: dict[str, Any] = {}
    for spec_index in BETA2_SPEC_INDICES:
        parameters = controlled_specs()[spec_index].parameter_count()
        scale_runs = {}
        for beta2 in BETA2_VALUES:
            config = _config(
                max_steps=BETA2_EXECUTION_STEPS,
                warmup_steps=0,
                scheduler="constant",
                beta2=beta2,
            )
            scale_runs[str(beta2)] = _run_one(
                spec_index=spec_index,
                train_stream=train_stream,
                validation_records=validation_records,
                tokenizer=tokenizer,
                execution_steps=BETA2_EXECUTION_STEPS,
                config=config,
                beta2=beta2,
                capture_second_moment=True,
            )
        beta2_runs[str(parameters)] = scale_runs

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "controls": {
            "learning_rate": LEARNING_RATE,
            "beta1": BETA1,
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": CLIP_NORM,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "tokens_per_step": TOKENS_PER_STEP,
            "seed": SEED,
            "warmup_campaign": {
                "beta2_fixed": 0.95,
                "scheduler": "cosine",
                "schedule_horizon_steps": SCHEDULE_HORIZON_STEPS,
                "execution_steps": WARMUP_EXECUTION_STEPS,
                "warmup_steps": list(WARMUP_STEPS),
                "parameter_scales": [
                    controlled_specs()[index].parameter_count()
                    for index in WARMUP_SPEC_INDICES
                ],
                "batch_trace_sha256": _same_trace_digest(
                    train_stream, WARMUP_EXECUTION_STEPS
                ),
            },
            "beta2_campaign": {
                "beta2_values": list(BETA2_VALUES),
                "scheduler": "constant",
                "warmup_steps": 0,
                "execution_steps": BETA2_EXECUTION_STEPS,
                "parameter_scales": [
                    controlled_specs()[index].parameter_count()
                    for index in BETA2_SPEC_INDICES
                ],
                "batch_trace_sha256": _same_trace_digest(
                    train_stream, BETA2_EXECUTION_STEPS
                ),
                "nominal_second_moment_time_constants_steps": {
                    str(value): 1.0 / (1.0 - value) for value in BETA2_VALUES
                },
            },
        },
        "definitions": {
            "early_window_steps": EARLY_WINDOW,
            "loss_spike_above_first": "max(loss first 16) - loss(step 1)",
            "clip_frequency": "fraction of steps with pre-clip gradient norm > 1.0",
            "update_to_weight_ratio": "L2(parameter delta) / L2(parameters before update)",
            "recovery_tokens": "first training point with loss <= 0.99 * random-init held-out validation loss",
            "second_moment": "AdamW exp_avg_sq aggregated across trainable parameters; bias corrected with 1-beta2^step",
        },
        "warmup_runs": warmup_runs,
        "beta2_runs": beta2_runs,
        "scope_warning": (
            "This is controlled repeated-fixture LOCAL_FREE optimization evidence. "
            "It does not freeze a production schedule and does not establish 10M+ transfer."
        ),
    }
    report["report_sha256"] = _canonical_hash(report)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate(path: Path, expected_source_sha: str) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("wrong optimizer dynamics evidence contract")
    if report.get("source_sha") != expected_source_sha:
        raise ValueError("source SHA mismatch")
    supplied = report.get("report_sha256")
    without_hash = dict(report)
    without_hash.pop("report_sha256", None)
    if supplied != _canonical_hash(without_hash):
        raise ValueError("report self-hash mismatch")
    controls = report["controls"]
    warmup = controls["warmup_campaign"]
    if warmup["schedule_horizon_steps"] <= warmup["execution_steps"]:
        raise ValueError("scheduler horizon must exceed shortened warmup execution")
    for scale_runs in report["warmup_runs"].values():
        if set(scale_runs) != {"0", "8", "32"}:
            raise ValueError("warmup matrix incomplete")
        configs = [run["config"] for run in scale_runs.values()]
        invariants = {
            (
                cfg["learning_rate"],
                tuple(cfg["betas"]),
                cfg["eps"],
                cfg["weight_decay"],
                cfg["scheduler"],
                cfg["max_steps"],
            )
            for cfg in configs
        }
        if len(invariants) != 1:
            raise ValueError("warmup campaign changed a forbidden control")
    for scale_runs in report["beta2_runs"].values():
        if set(scale_runs) != {"0.95", "0.98", "0.99"}:
            raise ValueError("beta2 matrix incomplete")
        configs = [run["config"] for run in scale_runs.values()]
        invariants = {
            (
                cfg["learning_rate"],
                cfg["betas"][0],
                cfg["eps"],
                cfg["weight_decay"],
                cfg["scheduler"],
                cfg["warmup_steps"],
                cfg["max_steps"],
            )
            for cfg in configs
        }
        if len(invariants) != 1:
            raise ValueError("beta2 campaign changed a forbidden control")
        if any("second_moment" not in run["summary"] for run in scale_runs.values()):
            raise ValueError("beta2 second-moment evidence missing")


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_sha": report["source_sha"],
        "report_sha256": report["report_sha256"],
        "warmup": {
            scale: {candidate: run["summary"] for candidate, run in runs.items()}
            for scale, runs in report["warmup_runs"].items()
        },
        "beta2": {
            scale: {candidate: run["summary"] for candidate, run in runs.items()}
            for scale, runs in report["beta2_runs"].items()
        },
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repo-root", type=Path, required=True)
    run_parser.add_argument("--source-sha", required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--expected-source-sha", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "run":
        report = run(args.repo_root.resolve(), args.source_sha, args.output)
        print(json.dumps(_summary(report), indent=2, sort_keys=True))
        return 0
    validate(args.path, args.expected_source_sha)
    print("optimizer_dynamics_evidence=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
