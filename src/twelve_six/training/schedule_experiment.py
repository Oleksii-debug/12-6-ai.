"""TRAIN-50 controlled constant-with-warmup vs cosine-with-warmup experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from twelve_six.model import InitSpec, TwelveSixDecoder
from twelve_six.scaling_500k_evidence import (
    TARGET_PARAMETERS,
    _bpb,
    _model_state_sha256,
    _parameter_delta_stats,
    _parameter_snapshot,
    _target_spec,
)
from twelve_six.scaling_experiment import (
    PACKING_ID,
    _byte_stream,
    _canonical_hash,
    _file_sha256,
    _git_head,
    _make_batch,
    _read_jsonl,
    _validation_loss,
)
from twelve_six.tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from twelve_six.training import Trainer, TrainerConfig

PLAN_SCHEMA = "12-6.train50-schedule-plan.v1"
REPORT_SCHEMA = "12-6.train50-schedule-evidence.v1"
AUTHORITY = "LOCAL_FREE_467808_SCHEDULE_COMPARISON_NOT_PROMOTION_OR_PAID_COMPUTE_AUTHORIZATION"
SCHEDULES = ("constant_with_warmup", "cosine_with_warmup")
REPOSITORY = "Oleksii-debug/12-6-ai."


class ScheduleExperimentError(ValueError):
    """Raised when the TRAIN-50 experiment contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScheduleExperimentError(message)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _require(isinstance(plan, dict), "plan root must be an object")
    _require(plan.get("schema_version") == PLAN_SCHEMA, "unexpected plan schema")
    _require(plan.get("authority") == AUTHORITY, "unexpected plan authority")

    model = plan.get("model")
    _require(isinstance(model, dict), "model plan missing")
    _require(model.get("parameter_count") == TARGET_PARAMETERS, "parameter count drift")
    _require(model.get("vocab_size") == 256, "vocabulary drift")
    _require(model.get("max_seq_len") == 256, "context drift")

    for field in (
        "batch_size",
        "sequence_length",
        "tokens_per_optimizer_step",
        "planned_optimizer_steps",
        "planned_optimized_tokens",
        "warmup_steps",
        "validation_every_steps",
    ):
        value = plan.get(field)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"{field} invalid",
        )

    _require(plan["batch_size"] == 4, "batch_size must preserve RESEARCH41 control")
    _require(plan["sequence_length"] == 64, "sequence_length must preserve RESEARCH41 control")
    expected_tokens_per_step = plan["batch_size"] * (plan["sequence_length"] - 1)
    _require(
        plan["tokens_per_optimizer_step"] == expected_tokens_per_step,
        "token accounting drift",
    )
    _require(
        plan["planned_optimizer_steps"] * plan["tokens_per_optimizer_step"]
        == plan["planned_optimized_tokens"],
        "planned optimized-token budget must exactly equal whole optimizer steps",
    )
    _require(
        plan["warmup_steps"] < plan["planned_optimizer_steps"],
        "warmup consumes entire run",
    )
    _require(
        plan["planned_optimizer_steps"] % plan["validation_every_steps"] == 0,
        "validation cadence must divide the planned horizon",
    )
    _require(
        plan["warmup_steps"] * 20 == plan["planned_optimizer_steps"],
        "TRAIN-50 plan must retain the incumbent 5% warmup",
    )

    schedules = plan.get("schedules")
    _require(isinstance(schedules, list), "schedules missing")
    _require(
        tuple(schedules) == SCHEDULES,
        "exactly constant-with-warmup and cosine-with-warmup are allowed",
    )

    seeds = plan.get("seeds")
    _require(
        isinstance(seeds, list)
        and len(seeds) >= 2
        and len(set(seeds)) == len(seeds)
        and all(
            isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
            for seed in seeds
        ),
        "at least two unique non-negative seeds are required",
    )

    optimizer = plan.get("optimizer")
    _require(isinstance(optimizer, dict), "optimizer plan missing")
    _require(optimizer.get("name") == "AdamW", "only AdamW is allowed")
    _require(optimizer.get("peak_learning_rate") == 3e-4, "peak LR must remain 3e-4")
    _require(optimizer.get("betas") == [0.9, 0.95], "AdamW betas drift")
    _require(optimizer.get("eps") == 1e-8, "AdamW epsilon drift")
    _require(optimizer.get("weight_decay") == 0.0, "weight decay drift")
    _require(optimizer.get("gradient_clip_norm") == 1.0, "gradient clip drift")
    _require(optimizer.get("precision") == "fp32", "precision drift")

    fractions = plan.get("time_to_quality_fractions")
    _require(isinstance(fractions, list) and fractions, "time-to-quality fractions missing")
    _require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 < float(value) <= 1.0
            for value in fractions
        ),
        "time-to-quality fractions invalid",
    )
    _require(
        fractions == sorted(set(fractions)),
        "time-to-quality fractions must be sorted and unique",
    )

    truth = plan.get("truth_boundary")
    _require(isinstance(truth, dict), "truth boundary missing")
    _require(truth.get("paid_compute") is False, "paid compute must remain false")
    _require(
        truth.get("representative_corpus_quality_claim") is False,
        "representative-corpus claim forbidden",
    )
    return plan


def trainer_config_for(
    plan: Mapping[str, Any],
    *,
    schedule: str,
    seed: int,
) -> TrainerConfig:
    _require(schedule in SCHEDULES, f"unsupported schedule: {schedule}")
    optimizer = plan["optimizer"]
    return TrainerConfig(
        learning_rate=float(optimizer["peak_learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
        betas=(float(optimizer["betas"][0]), float(optimizer["betas"][1])),
        eps=float(optimizer["eps"]),
        max_steps=int(plan["planned_optimizer_steps"]),
        warmup_steps=int(plan["warmup_steps"]),
        scheduler="constant" if schedule == "constant_with_warmup" else "cosine",
        gradient_accumulation_steps=1,
        gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _batch_trace_sha256(
    train_stream: bytes,
    *,
    steps: int,
    batch_size: int,
    sequence_length: int,
) -> str:
    digest = hashlib.sha256()
    for step in range(steps):
        batch = _make_batch(
            train_stream,
            step=step,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        digest.update(batch.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _run_schedule(
    *,
    plan: Mapping[str, Any],
    schedule: str,
    seed: int,
    spec,
    init_spec: InitSpec,
    tokenizer: ByteTokenizer,
    train_stream: bytes,
    validation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    config = trainer_config_for(plan, schedule=schedule, seed=seed)
    model = TwelveSixDecoder(spec, init_spec)
    trainer = Trainer(model, config, device="cpu")

    initial_model_state_sha256 = _model_state_sha256(model)
    initial_parameters = _parameter_snapshot(model)
    previous_parameters = {
        name: value.clone() for name, value in initial_parameters.items()
    }
    initial_validation_loss, validation_tokens = _validation_loss(
        model,
        validation_records,
        tokenizer,
    )
    validation_curve: list[dict[str, Any]] = [
        {
            "optimizer_step": 0,
            "optimized_tokens": 0,
            "validation_loss": initial_validation_loss,
            "validation_bpb": _bpb(initial_validation_loss),
            "training_wall_seconds": 0.0,
        }
    ]
    train_curve: list[dict[str, Any]] = []
    training_wall_seconds = 0.0
    end_to_end_started = time.perf_counter()

    planned_steps = int(plan["planned_optimizer_steps"])
    batch_size = int(plan["batch_size"])
    sequence_length = int(plan["sequence_length"])
    validation_every = int(plan["validation_every_steps"])

    for step in range(planned_steps):
        batch = _make_batch(
            train_stream,
            step=step,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        step_started = time.perf_counter()
        metrics = trainer.train_microbatch({"input_ids": batch})
        step_wall_seconds = time.perf_counter() - step_started
        training_wall_seconds += step_wall_seconds
        if not metrics.optimizer_stepped or metrics.update_loss is None:
            raise RuntimeError("TRAIN-50 requires one optimizer update per batch")
        delta = _parameter_delta_stats(
            model,
            previous=previous_parameters,
            initial=initial_parameters,
        )
        previous_parameters = _parameter_snapshot(model)
        train_curve.append(
            {
                "optimizer_step": metrics.optimizer_step,
                "optimized_tokens": trainer.tokens_seen,
                "train_loss": metrics.update_loss,
                "train_bpb": _bpb(metrics.update_loss),
                "learning_rate": metrics.learning_rate,
                "grad_norm_pre_clip": metrics.grad_norm,
                "update_to_weight_ratio": delta["update_to_weight_ratio"],
                "update_l2": delta["update_l2"],
                "max_abs_update": delta["max_abs_update"],
                "step_wall_seconds": step_wall_seconds,
                "training_wall_seconds": training_wall_seconds,
            }
        )
        if metrics.optimizer_step % validation_every == 0:
            validation_loss, checked_tokens = _validation_loss(
                model,
                validation_records,
                tokenizer,
            )
            if checked_tokens != validation_tokens:
                raise RuntimeError("validation token count drifted")
            validation_curve.append(
                {
                    "optimizer_step": metrics.optimizer_step,
                    "optimized_tokens": trainer.tokens_seen,
                    "validation_loss": validation_loss,
                    "validation_bpb": _bpb(validation_loss),
                    "training_wall_seconds": training_wall_seconds,
                }
            )

    expected_tokens = int(plan["planned_optimized_tokens"])
    if trainer.optimizer_step != planned_steps or trainer.tokens_seen != expected_tokens:
        raise RuntimeError(
            "planned horizon/token budget drift: "
            f"steps={trainer.optimizer_step}/{planned_steps}, "
            f"tokens={trainer.tokens_seen}/{expected_tokens}"
        )
    if validation_curve[-1]["optimizer_step"] != planned_steps:
        raise RuntimeError("final validation point missing")

    return {
        "schedule": schedule,
        "seed": seed,
        "trainer_config": asdict(config),
        "initial_model_state_sha256": initial_model_state_sha256,
        "final_model_state_sha256": _model_state_sha256(model),
        "batch_trace_sha256": _batch_trace_sha256(
            train_stream,
            steps=planned_steps,
            batch_size=batch_size,
            sequence_length=sequence_length,
        ),
        "validation_tokens": validation_tokens,
        "train_curve": train_curve,
        "validation_curve": validation_curve,
        "summary": {
            "initial_validation_bpb": validation_curve[0]["validation_bpb"],
            "final_validation_loss": validation_curve[-1]["validation_loss"],
            "final_validation_bpb": validation_curve[-1]["validation_bpb"],
            "best_validation_bpb": min(
                point["validation_bpb"] for point in validation_curve
            ),
            "best_validation_step": min(
                validation_curve,
                key=lambda point: (
                    point["validation_bpb"],
                    point["optimizer_step"],
                ),
            )["optimizer_step"],
            "final_learning_rate_used": train_curve[-1]["learning_rate"],
            "median_update_to_weight_ratio": statistics.median(
                point["update_to_weight_ratio"] for point in train_curve
            ),
            "max_update_to_weight_ratio": max(
                point["update_to_weight_ratio"] for point in train_curve
            ),
            "training_wall_seconds": training_wall_seconds,
            "end_to_end_wall_seconds": time.perf_counter() - end_to_end_started,
            "optimized_tokens": trainer.tokens_seen,
            "optimizer_steps": trainer.optimizer_step,
        },
    }


def time_to_quality_for_seed(
    schedule_runs: Mapping[str, Mapping[str, Any]],
    fractions: list[float],
) -> dict[str, Any]:
    _require(
        set(schedule_runs) == set(SCHEDULES),
        "time-to-quality requires both schedules",
    )
    initial_values = [
        float(schedule_runs[name]["validation_curve"][0]["validation_bpb"])
        for name in SCHEDULES
    ]
    _require(
        initial_values[0] == initial_values[1],
        "same-seed initial validation BPB must match exactly",
    )
    initial_bpb = initial_values[0]
    final_bpb = {
        name: float(schedule_runs[name]["validation_curve"][-1]["validation_bpb"])
        for name in SCHEDULES
    }
    common_final_target_bpb = max(final_bpb.values())
    _require(
        common_final_target_bpb < initial_bpb,
        "both schedules must improve over initialization",
    )

    levels: list[dict[str, Any]] = []
    for fraction in fractions:
        threshold = initial_bpb - fraction * (initial_bpb - common_final_target_bpb)
        arrivals: dict[str, Any] = {}
        for schedule in SCHEDULES:
            match = next(
                (
                    point
                    for point in schedule_runs[schedule]["validation_curve"]
                    if float(point["validation_bpb"]) <= threshold
                ),
                None,
            )
            _require(
                match is not None,
                f"{schedule} did not reach common quality threshold",
            )
            arrivals[schedule] = {
                "optimizer_step": int(match["optimizer_step"]),
                "optimized_tokens": int(match["optimized_tokens"]),
                "training_wall_seconds": float(match["training_wall_seconds"]),
                "validation_bpb": float(match["validation_bpb"]),
            }
        levels.append(
            {
                "fraction_of_common_final_improvement": fraction,
                "threshold_validation_bpb": threshold,
                "arrivals": arrivals,
            }
        )
    return {
        "initial_validation_bpb": initial_bpb,
        "common_final_target_bpb": common_final_target_bpb,
        "schedule_final_bpb": final_bpb,
        "levels": levels,
    }


def _aggregate(
    report_runs: list[dict[str, Any]],
    fractions: list[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_schedule: dict[str, list[dict[str, Any]]] = {
        name: [] for name in SCHEDULES
    }
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for run in report_runs:
        schedule = str(run["schedule"])
        seed = int(run["seed"])
        by_schedule[schedule].append(run)
        by_seed.setdefault(seed, {})[schedule] = run

    final_quality: dict[str, Any] = {}
    for schedule in SCHEDULES:
        runs = by_schedule[schedule]
        final_bpbs = [
            float(run["summary"]["final_validation_bpb"]) for run in runs
        ]
        best_bpbs = [
            float(run["summary"]["best_validation_bpb"]) for run in runs
        ]
        final_quality[schedule] = {
            "seeds": [int(run["seed"]) for run in runs],
            "final_validation_bpb_by_seed": final_bpbs,
            "mean_final_validation_bpb": statistics.mean(final_bpbs),
            "mean_best_validation_bpb": statistics.mean(best_bpbs),
            "mean_training_wall_seconds": statistics.mean(
                float(run["summary"]["training_wall_seconds"])
                for run in runs
            ),
        }
    final_quality["cosine_minus_constant_mean_final_bpb"] = (
        final_quality["cosine_with_warmup"]["mean_final_validation_bpb"]
        - final_quality["constant_with_warmup"]["mean_final_validation_bpb"]
    )

    per_seed: dict[str, Any] = {}
    for seed, runs in sorted(by_seed.items()):
        per_seed[str(seed)] = time_to_quality_for_seed(runs, fractions)

    aggregate_levels: list[dict[str, Any]] = []
    for index, fraction in enumerate(fractions):
        row: dict[str, Any] = {
            "fraction_of_common_final_improvement": fraction
        }
        for schedule in SCHEDULES:
            arrivals = [
                per_seed[str(seed)]["levels"][index]["arrivals"][schedule]
                for seed in sorted(by_seed)
            ]
            row[schedule] = {
                "mean_optimizer_step": statistics.mean(
                    float(item["optimizer_step"]) for item in arrivals
                ),
                "mean_optimized_tokens": statistics.mean(
                    float(item["optimized_tokens"]) for item in arrivals
                ),
                "mean_training_wall_seconds": statistics.mean(
                    float(item["training_wall_seconds"]) for item in arrivals
                ),
            }
        aggregate_levels.append(row)
    return final_quality, {
        "per_seed": per_seed,
        "aggregate_levels": aggregate_levels,
    }


def run_schedule_experiment(
    *,
    repo_root: Path,
    source_sha: str,
    plan_path: Path,
    output_path: Path,
    torch_threads: int = 2,
) -> dict[str, Any]:
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("exact-checkout mismatch for TRAIN-50")
    if torch_threads <= 0:
        raise ValueError("torch_threads must be positive")
    plan = load_plan(plan_path)
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)

    tokenizer = ByteTokenizer()
    spec = _target_spec()
    init_spec = InitSpec()
    if spec.parameter_count() != TARGET_PARAMETERS:
        raise RuntimeError("467,808-parameter control drifted")

    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_records = _read_jsonl(train_path)
    validation_records = _read_jsonl(validation_path)
    train_ids = {str(row["id"]) for row in train_records}
    validation_ids = {str(row["id"]) for row in validation_records}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise RuntimeError(f"train/validation overlap: {overlap!r}")
    train_stream = _byte_stream(train_records, tokenizer)

    runs: list[dict[str, Any]] = []
    seeds = [int(seed) for seed in plan["seeds"]]
    # Reverse alternate-seed order to reduce systematic CPU warm-cache/order bias.
    for seed_index, seed in enumerate(seeds):
        order = list(
            SCHEDULES if seed_index % 2 == 0 else tuple(reversed(SCHEDULES))
        )
        seed_runs: dict[str, dict[str, Any]] = {}
        for schedule in order:
            result = _run_schedule(
                plan=plan,
                schedule=schedule,
                seed=seed,
                spec=spec,
                init_spec=init_spec,
                tokenizer=tokenizer,
                train_stream=train_stream,
                validation_records=validation_records,
            )
            seed_runs[schedule] = result
            runs.append(result)
        if (
            seed_runs[SCHEDULES[0]]["initial_model_state_sha256"]
            != seed_runs[SCHEDULES[1]]["initial_model_state_sha256"]
        ):
            raise RuntimeError("same-seed schedules did not start from identical weights")
        if (
            seed_runs[SCHEDULES[0]]["batch_trace_sha256"]
            != seed_runs[SCHEDULES[1]]["batch_trace_sha256"]
        ):
            raise RuntimeError("same-seed schedules did not consume identical batch traces")

    fractions = [float(value) for value in plan["time_to_quality_fractions"]]
    final_quality, time_to_quality = _aggregate(runs, fractions)
    common_trace_hashes = {run["batch_trace_sha256"] for run in runs}
    if len(common_trace_hashes) != 1:
        raise RuntimeError(
            "all schedules/seeds must consume the same deterministic batch trace"
        )

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": REPOSITORY, "git_sha": source_sha},
        "paid_compute": False,
        "plan": plan,
        "plan_file_sha256": _sha256_file(plan_path),
        "model": {
            "parameter_count": spec.parameter_count(),
            "modelspec_sha256": spec.identity_sha256(),
            "initspec_sha256": init_spec.identity_sha256(),
            "spec": spec.to_dict(),
        },
        "tokenizer": {
            "id": BYTE_TOKENIZER_VERSION,
            "config_sha256": BYTE_TOKENIZER_HASH,
            "vocab_sha256": BYTE_VOCAB_HASH,
        },
        "data": {
            "packing_id": PACKING_ID,
            "dataset_manifest_sha256": _file_sha256(manifest_path),
            "train_jsonl_sha256": _file_sha256(train_path),
            "validation_jsonl_sha256": _file_sha256(validation_path),
            "train_validation_record_overlap": overlap,
            "validation_never_optimized": True,
            "project_authored_tiny_repeated_fixture": True,
        },
        "common_batch_trace_sha256": next(iter(common_trace_hashes)),
        "runs": runs,
        "final_quality": final_quality,
        "time_to_quality": time_to_quality,
        "provisional_decision_rule": {
            "primary": "lower_mean_final_validation_bpb",
            "secondary": "lower_mean_optimized_tokens_to_common_quality_levels",
            "wall_time": (
                "diagnostic_cpu_measurement_not_transferable_accelerator_throughput"
            ),
            "campaign_scope": "provisional_100k_to_1m_default_only",
        },
        "truth_boundary": {
            "only_schedule_family_differs_within_each_seed": True,
            "full_scheduler_horizon_equals_executed_planned_run": True,
            "representative_corpus_quality_claim": False,
            "broad_scaling_claim": False,
            "paid_compute_authority": False,
            "promotion_authority": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def validate_report(
    report: Mapping[str, Any],
    *,
    expected_source_sha: str | None = None,
) -> None:
    _require(report.get("schema_version") == REPORT_SCHEMA, "unexpected report schema")
    _require(report.get("authority") == AUTHORITY, "unexpected report authority")
    _require(report.get("paid_compute") is False, "paid_compute must remain false")
    if expected_source_sha is not None:
        _require(
            report.get("source", {}).get("git_sha") == expected_source_sha,
            "source SHA mismatch",
        )
    _require(
        report.get("model", {}).get("parameter_count") == TARGET_PARAMETERS,
        "model parameter drift",
    )
    _require(
        report.get("tokenizer", {}).get("id") == BYTE_TOKENIZER_VERSION,
        "tokenizer drift",
    )
    _require(
        report.get("data", {}).get("train_validation_record_overlap") == [],
        "train/validation overlap",
    )
    _require(
        report.get("truth_boundary", {}).get(
            "full_scheduler_horizon_equals_executed_planned_run"
        )
        is True,
        "scheduler horizon proof missing",
    )

    plan = report.get("plan")
    _require(isinstance(plan, dict), "embedded plan missing")
    _require(plan.get("planned_optimizer_steps") == 1040, "planned horizon drift")
    _require(plan.get("planned_optimized_tokens") == 262080, "planned token budget drift")

    runs = report.get("runs")
    _require(isinstance(runs, list), "runs missing")
    expected_run_count = len(plan["seeds"]) * len(SCHEDULES)
    _require(len(runs) == expected_run_count, "run matrix incomplete")
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    trace_hashes: set[str] = set()
    for run in runs:
        _require(run.get("schedule") in SCHEDULES, "unexpected schedule family")
        summary = run.get("summary", {})
        _require(
            summary.get("optimizer_steps") == plan["planned_optimizer_steps"],
            "run ended before planned horizon",
        )
        _require(
            summary.get("optimized_tokens") == plan["planned_optimized_tokens"],
            "run token budget drift",
        )
        train_curve = run.get("train_curve")
        validation_curve = run.get("validation_curve")
        _require(
            isinstance(train_curve, list)
            and len(train_curve) == plan["planned_optimizer_steps"],
            "train curve incomplete",
        )
        _require(
            isinstance(validation_curve, list)
            and len(validation_curve)
            == plan["planned_optimizer_steps"] // plan["validation_every_steps"] + 1,
            "validation curve incomplete",
        )
        _require(
            all(
                math.isfinite(float(point["learning_rate"]))
                for point in train_curve
            ),
            "non-finite LR",
        )
        _require(
            all(math.isfinite(float(point["train_bpb"])) for point in train_curve),
            "non-finite train BPB",
        )
        _require(
            all(
                math.isfinite(float(point["update_to_weight_ratio"]))
                for point in train_curve
            ),
            "non-finite update ratio",
        )
        _require(
            all(
                math.isfinite(float(point["validation_bpb"]))
                for point in validation_curve
            ),
            "non-finite validation BPB",
        )
        seed = int(run["seed"])
        by_seed.setdefault(seed, {})[str(run["schedule"])] = run
        trace_hashes.add(str(run["batch_trace_sha256"]))

    _require(len(trace_hashes) == 1, "batch traces differ across runs")
    _require(
        report.get("common_batch_trace_sha256") in trace_hashes,
        "common batch trace hash mismatch",
    )
    for seed, schedule_runs in by_seed.items():
        _require(
            set(schedule_runs) == set(SCHEDULES),
            f"seed {seed} missing schedule arm",
        )
        left = schedule_runs[SCHEDULES[0]]
        right = schedule_runs[SCHEDULES[1]]
        _require(
            left["initial_model_state_sha256"]
            == right["initial_model_state_sha256"],
            "initial weight mismatch",
        )
        left_lrs = [
            float(point["learning_rate"])
            for point in left["train_curve"][: plan["warmup_steps"]]
        ]
        right_lrs = [
            float(point["learning_rate"])
            for point in right["train_curve"][: plan["warmup_steps"]]
        ]
        _require(left_lrs == right_lrs, "warmup LR traces differ")
        _require(
            float(left["train_curve"][-1]["learning_rate"])
            > float(right["train_curve"][-1]["learning_rate"]),
            "cosine arm did not decay below constant arm",
        )

    observed_hash = report.get("report_sha256")
    _require(isinstance(observed_hash, str), "report_sha256 missing")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    _require(_canonical_hash(unhashed) == observed_hash, "report_sha256 mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--source-sha", required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--torch-threads", type=int, default=2)
    validate = subparsers.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    args = parser.parse_args(argv)

    if args.command == "run":
        report = run_schedule_experiment(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            plan_path=args.plan,
            output_path=args.output,
            torch_threads=args.torch_threads,
        )
        validate_report(report, expected_source_sha=args.source_sha)
        print(
            json.dumps(
                {
                    "report_sha256": report["report_sha256"],
                    "parameters": report["model"]["parameter_count"],
                    "planned_optimizer_steps": report["plan"][
                        "planned_optimizer_steps"
                    ],
                    "planned_optimized_tokens": report["plan"][
                        "planned_optimized_tokens"
                    ],
                    "final_quality": report["final_quality"],
                    "time_to_quality": report["time_to_quality"][
                        "aggregate_levels"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_report(report, expected_source_sha=args.expected_source_sha)
    print(
        json.dumps(
            {"status": "PASS", "report_sha256": report["report_sha256"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
