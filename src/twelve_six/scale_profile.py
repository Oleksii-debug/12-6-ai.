"""Repeatable CPU scale profiling above S0 without owning the W5 S0 profiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch

from twelve_six.checkpoint.core import (
    CheckpointIdentity,
    detect_git_sha,
    hash_json,
    save_checkpoint,
    verify_checkpoint,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.training import Trainer, TrainerConfig

SCHEMA_VERSION = "12-6.scale-profile-matrix.v1"
STAGE_SCHEMA_VERSION = "12-6.scale-profile-stage.v1"
AUTHORITY = "LOCAL_FREE_CPU_SCALE_PROFILE_NOT_CAPACITY_OR_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
OBSERVED_STAGES = {
    "S1": "configs/stages/s1_100k.json",
    "S2": "configs/stages/s2_1m.json",
    "S3": "configs/stages/s3_10m.json",
}
ANALYTICAL_STAGE = "P100M_ANALYTICAL"
_HEX = frozenset("0123456789abcdef")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_sha(source_sha: str) -> None:
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 40
        or source_sha != source_sha.lower()
        or any(ch not in _HEX for ch in source_sha)
    ):
        raise ValueError("source_sha must be a full lowercase 40-hex Git SHA")


def _require_exact_checkout(source_sha: str) -> None:
    _validate_source_sha(source_sha)
    actual = detect_git_sha(_root())
    if actual != source_sha:
        raise RuntimeError(f"exact checkout mismatch: expected {source_sha}, observed {actual}")


def _rss_high_water_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value
    return value * 1024


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _summarize(samples: list[float]) -> dict[str, Any]:
    if not samples or any(value <= 0.0 or not math.isfinite(value) for value in samples):
        raise RuntimeError("timing samples must be finite and positive")
    return {
        "repetitions": len(samples),
        "seconds": {
            "min": min(samples),
            "median": statistics.median(samples),
            "max": max(samples),
        },
    }


def _measure(operation: Callable[[], Any], *, repetitions: int, warmups: int = 0) -> dict[str, Any]:
    if repetitions < 1 or warmups < 0:
        raise ValueError("repetitions must be >= 1 and warmups must be >= 0")
    for _ in range(warmups):
        operation()
    samples: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - start) / 1_000_000_000)
    return _summarize(samples)


def _trainer_config(*, seed: int, max_steps: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4,
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


def _synthetic_hash(stage: str, kind: str, payload: Any | None = None) -> str:
    return _canonical_hash(
        {
            "schema": "12-6.scale-profile-synthetic-identity.v1",
            "stage": stage,
            "kind": kind,
            "payload": payload,
        }
    )


def _checkpoint_identity(
    *,
    source_sha: str,
    stage: str,
    spec: ModelSpec,
    trainer: Trainer,
    config: TrainerConfig,
    sequence_length: int,
) -> CheckpointIdentity:
    optimizer_contract = {
        "name": "AdamW",
        "learning_rate": config.learning_rate,
        "betas": list(config.betas),
        "eps": config.eps,
        "weight_decay": config.weight_decay,
    }
    training_config = asdict(config)
    run_manifest = {
        "schema": "12-6.scale-profile-run.v1",
        "stage": stage,
        "source_sha": source_sha,
        "sequence_length": sequence_length,
        "seed": config.seed,
    }
    lock_index = _root() / "requirements/locks/index.json"
    lock_hash = _sha256_file(lock_index) if lock_index.exists() else None
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=_synthetic_hash(stage, "tokenizer"),
        tokenizer_vocab_hash=_synthetic_hash(stage, "tokenizer_vocab", spec.vocab_size),
        dataset_manifest_hash=_synthetic_hash(stage, "dataset", sequence_length),
        run_manifest_hash=hash_json(run_manifest),
        training_config=training_config,
        seed=config.seed,
        precision=config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer=optimizer_contract,
        scheduler=None,
        environment_lock_hash=lock_hash,
    )


def _stage_config_record(stage: str) -> tuple[Any, dict[str, Any]]:
    relative = OBSERVED_STAGES[stage]
    path = _root() / relative
    config = load_stage_config(path)
    if config.stage != stage:
        raise RuntimeError(f"stage config mismatch for {stage}: {config.stage}")
    return config, {
        "path": relative,
        "sha256": _sha256_file(path),
        "modelspec_sha256": config.model.identity_sha256(),
        "initspec_sha256": config.init.identity_sha256(),
    }


def _fixed_inputs(spec: ModelSpec, sequence_length: int) -> torch.Tensor:
    values = torch.arange(sequence_length, dtype=torch.long).remainder(spec.vocab_size)
    return values.unsqueeze(0)


def _profile_stage(
    *,
    stage: str,
    source_sha: str,
    sequence_length: int,
    repetitions: int,
    train_repetitions: int,
    checkpoint_repetitions: int,
    generation_repetitions: int,
    generation_new_tokens: int,
    torch_threads: int,
    seed: int,
) -> dict[str, Any]:
    _require_exact_checkout(source_sha)
    if stage not in OBSERVED_STAGES:
        raise ValueError(f"unsupported observed stage: {stage}")
    if torch_threads < 1:
        raise ValueError("torch_threads must be >= 1")
    torch.set_num_threads(torch_threads)
    config, config_identity = _stage_config_record(stage)
    spec = config.model
    init_spec = config.init
    effective_sequence = min(sequence_length, spec.max_seq_len)
    if effective_sequence < 2:
        raise ValueError("effective sequence length must be >= 2")
    if generation_new_tokens < 1:
        raise ValueError("generation_new_tokens must be >= 1")

    rss_before = _rss_high_water_bytes()
    construction_samples: list[float] = []
    for index in range(repetitions):
        torch.manual_seed(seed + index)
        start = time.perf_counter_ns()
        candidate = TwelveSixDecoder(spec, init_spec)
        construction_samples.append((time.perf_counter_ns() - start) / 1_000_000_000)
        del candidate
    construction = _summarize(construction_samples)

    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    input_ids = _fixed_inputs(spec, effective_sequence)
    batch = {"input_ids": input_ids, "labels": input_ids.clone()}

    model.eval()
    with torch.no_grad():
        forward = _measure(
            lambda: model(input_ids).logits,
            repetitions=repetitions,
            warmups=1,
        )
    forward_median = forward["seconds"]["median"]
    forward_tokens_per_second = effective_sequence / forward_median

    trainer_config = _trainer_config(seed=seed, max_steps=train_repetitions + 2)
    trainer = Trainer(model, trainer_config, device="cpu")
    trainer.train_microbatch(batch)
    train_samples: list[float] = []
    train_tokens = 0
    train_metrics: list[dict[str, Any]] = []
    for _ in range(train_repetitions):
        start = time.perf_counter_ns()
        metrics = trainer.train_microbatch(batch)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000_000
        train_samples.append(elapsed)
        train_tokens += metrics.tokens
        train_metrics.append(asdict(metrics))
    train = _summarize(train_samples)
    train_tokens_per_second = train_tokens / sum(train_samples)

    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    optimizer_tensor_bytes = _tensor_bytes(trainer.optimizer.state_dict())

    checkpoint_samples: list[float] = []
    checkpoint_bytes: list[int] = []
    verify_samples: list[float] = []
    with tempfile.TemporaryDirectory(prefix=f"twelve-six-{stage.lower()}-profile-") as temp:
        temp_root = Path(temp)
        identity = _checkpoint_identity(
            source_sha=source_sha,
            stage=stage,
            spec=spec,
            trainer=trainer,
            config=trainer_config,
            sequence_length=effective_sequence,
        )
        for index in range(checkpoint_repetitions):
            destination = temp_root / f"checkpoint-{index}"
            start = time.perf_counter_ns()
            save_checkpoint(
                destination,
                model=model,
                identity=identity,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                trainer_state={
                    "micro_step": trainer.micro_step,
                    "optimizer_step": trainer.optimizer_step,
                    "tokens_seen": trainer.tokens_seen,
                },
            )
            checkpoint_samples.append((time.perf_counter_ns() - start) / 1_000_000_000)
            checkpoint_bytes.append(_directory_bytes(destination))
            verify_start = time.perf_counter_ns()
            verify_checkpoint(destination)
            verify_samples.append((time.perf_counter_ns() - verify_start) / 1_000_000_000)
    checkpoint = _summarize(checkpoint_samples)
    checkpoint["bytes"] = {
        "min": min(checkpoint_bytes),
        "median": statistics.median(checkpoint_bytes),
        "max": max(checkpoint_bytes),
    }
    checkpoint_verify = _summarize(verify_samples)

    model.eval()
    prompt_length = min(16, max(1, spec.max_seq_len - generation_new_tokens))
    prompt = _fixed_inputs(spec, prompt_length)
    generation = _measure(
        lambda: model.generate(
            prompt,
            max_new_tokens=generation_new_tokens,
            do_sample=False,
        ),
        repetitions=generation_repetitions,
        warmups=1,
    )
    generation_tokens_per_second = generation_new_tokens / generation["seconds"]["median"]

    rss_after = _rss_high_water_bytes()
    return {
        "schema": STAGE_SCHEMA_VERSION,
        "origin": "OBSERVED",
        "stage": stage,
        "source_sha": source_sha,
        "config_identity": config_identity,
        "geometry": {
            "target_parameters": config.target_parameters,
            "parameter_count": config.expected_parameters,
            "vocab_size": spec.vocab_size,
            "max_seq_len": spec.max_seq_len,
            "d_model": spec.d_model,
            "n_layers": spec.n_layers,
            "n_heads": spec.n_heads,
            "n_kv_heads": spec.n_kv_heads,
            "head_dim": spec.head_dim,
            "d_ff": spec.d_ff,
        },
        "workload": {
            "batch_size": 1,
            "sequence_length": effective_sequence,
            "optimized_tokens_per_step": effective_sequence - 1,
            "generation_prompt_tokens": prompt_length,
            "generation_new_tokens": generation_new_tokens,
            "precision": "fp32",
            "device": "cpu",
            "seed": seed,
        },
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "measurements": {
            "construction": construction,
            "forward": {
                **forward,
                "input_tokens_per_second": forward_tokens_per_second,
            },
            "canonical_train_microbatch_forward_backward_update": {
                **train,
                "optimized_tokens_per_second": train_tokens_per_second,
                "first_timed_metrics": train_metrics[0],
                "last_timed_metrics": train_metrics[-1],
                "phase_decomposition": "NOT_EXPOSED_BY_PUBLIC_TRAINER_SEAM",
            },
            "checkpoint_save": checkpoint,
            "checkpoint_verify": checkpoint_verify,
            "greedy_generation_stateless": {
                **generation,
                "generated_tokens_per_second": generation_tokens_per_second,
            },
            "parameter_bytes": parameter_bytes,
            "optimizer_tensor_bytes": optimizer_tensor_bytes,
            "peak_rss_before_bytes": rss_before,
            "peak_rss_after_bytes": rss_after,
            "peak_rss_delta_bytes": (
                None if rss_before is None or rss_after is None else max(0, rss_after - rss_before)
            ),
        },
        "timing_metrics_in_deterministic_state_fingerprint": False,
    }


def _analytical_meta_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=8192,
        max_seq_len=2048,
        d_model=832,
        n_layers=10,
        n_heads=13,
        n_kv_heads=13,
        head_dim=64,
        d_ff=2624,
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10000.0,
        rope_rotary_dim=64,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )


def _power_law_extrapolate(
    rows: list[dict[str, Any]],
    value_getter: Callable[[dict[str, Any]], float],
    target_parameters: int,
) -> dict[str, Any]:
    points: list[tuple[float, float]] = []
    for row in rows:
        x = float(row["geometry"]["parameter_count"])
        y = float(value_getter(row))
        if x <= 0.0 or y <= 0.0 or not math.isfinite(y):
            raise ValueError("power-law inputs must be finite and positive")
        points.append((math.log(x), math.log(y)))
    x_mean = statistics.fmean(x for x, _ in points)
    y_mean = statistics.fmean(y for _, y in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0.0:
        raise ValueError("power-law fit requires distinct parameter counts")
    exponent = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    intercept = y_mean - exponent * x_mean
    estimate = math.exp(intercept + exponent * math.log(target_parameters))
    fitted = [math.exp(intercept + exponent * x) for x, _ in points]
    actual = [math.exp(y) for _, y in points]
    relative_errors = [abs(a - f) / a for a, f in zip(actual, fitted, strict=True)]
    return {
        "method": "LOG_LOG_POWER_LAW_FIT_OVER_OBSERVED_S1_S2_S3",
        "estimate": estimate,
        "exponent": exponent,
        "max_observed_fit_relative_error": max(relative_errors),
    }


def _meta_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    spec = _analytical_meta_spec()
    parameter_count = spec.parameter_count()
    parameter_bytes = parameter_count * 4
    optimizer_ratios = [
        row["measurements"]["optimizer_tensor_bytes"] / row["measurements"]["parameter_bytes"]
        for row in rows
    ]
    checkpoint_ratios = [
        row["measurements"]["checkpoint_save"]["bytes"]["median"]
        / row["measurements"]["parameter_bytes"]
        for row in rows
    ]
    optimizer_bytes = parameter_bytes * statistics.median(optimizer_ratios)
    checkpoint_bytes = parameter_bytes * statistics.median(checkpoint_ratios)

    construction_fit = _power_law_extrapolate(
        rows,
        lambda row: row["measurements"]["construction"]["seconds"]["median"],
        parameter_count,
    )
    forward_fit = _power_law_extrapolate(
        rows,
        lambda row: row["measurements"]["forward"]["seconds"]["median"],
        parameter_count,
    )
    train_fit = _power_law_extrapolate(
        rows,
        lambda row: row["measurements"][
            "canonical_train_microbatch_forward_backward_update"
        ]["seconds"]["median"],
        parameter_count,
    )
    checkpoint_fit = _power_law_extrapolate(
        rows,
        lambda row: row["measurements"]["checkpoint_save"]["seconds"]["median"],
        parameter_count,
    )
    generation_fit = _power_law_extrapolate(
        rows,
        lambda row: row["measurements"]["greedy_generation_stateless"]["seconds"]["median"],
        parameter_count,
    )
    sequence_length = rows[0]["workload"]["sequence_length"]
    train_tokens = sequence_length - 1
    generation_new_tokens = rows[0]["workload"]["generation_new_tokens"]
    return {
        "schema": STAGE_SCHEMA_VERSION,
        "origin": "EXTRAPOLATED_ANALYTICAL",
        "stage": ANALYTICAL_STAGE,
        "source_sha": rows[0]["source_sha"],
        "geometry": {
            "target_parameters": 100_000_000,
            "parameter_count": parameter_count,
            "vocab_size": spec.vocab_size,
            "max_seq_len": spec.max_seq_len,
            "d_model": spec.d_model,
            "n_layers": spec.n_layers,
            "n_heads": spec.n_heads,
            "n_kv_heads": spec.n_kv_heads,
            "head_dim": spec.head_dim,
            "d_ff": spec.d_ff,
            "modelspec_sha256": spec.identity_sha256(),
            "status": "ANALYTICAL_GEOMETRY_NOT_CANONICAL_STAGE_CONFIG",
        },
        "workload": {
            "batch_size": 1,
            "sequence_length": sequence_length,
            "optimized_tokens_per_step": train_tokens,
            "generation_new_tokens": generation_new_tokens,
            "precision": "fp32",
            "device": "cpu",
        },
        "estimates": {
            "parameter_bytes_fp32": parameter_bytes,
            "optimizer_tensor_bytes": optimizer_bytes,
            "checkpoint_bytes": checkpoint_bytes,
            "construction_seconds": construction_fit,
            "forward_seconds": forward_fit,
            "forward_input_tokens_per_second": {
                **forward_fit,
                "estimate": sequence_length / forward_fit["estimate"],
            },
            "canonical_train_step_seconds": train_fit,
            "optimized_tokens_per_second": {
                **train_fit,
                "estimate": train_tokens / train_fit["estimate"],
            },
            "checkpoint_save_seconds": checkpoint_fit,
            "greedy_generation_seconds": generation_fit,
            "generated_tokens_per_second": {
                **generation_fit,
                "estimate": generation_new_tokens / generation_fit["estimate"],
            },
            "peak_rss_bytes": None,
            "peak_rss_reason": (
                "NOT_EXTRAPOLATED_PROCESS_RSS_CONTAINS_RUNTIME_BASELINE_AND_ALLOCATOR_HISTORY"
            ),
        },
        "timing_metrics_in_deterministic_state_fingerprint": False,
    }


def _cpu_boundary(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    criteria = {
        "train_step_median_seconds_gte": 1.0,
        "optimized_tokens_per_second_lt": 2_000.0,
    }
    for row in rows:
        train = row["measurements"]["canonical_train_microbatch_forward_backward_update"]
        if (
            train["seconds"]["median"] >= criteria["train_step_median_seconds_gte"]
            or train["optimized_tokens_per_second"] < criteria["optimized_tokens_per_second_lt"]
        ):
            return {
                "classification": "OBSERVED_CPU_ITERATION_BOUNDARY",
                "earliest_stage": row["stage"],
                "parameter_count": row["geometry"]["parameter_count"],
                "criteria": criteria,
                "gpu_or_accelerator_engineering": "MANDATORY_FOR_SERIOUS_TRAINING",
                "distributed_training": (
                    "NOT_PROVEN_MANDATORY_BY_THIS_PROFILE; SINGLE_ACCELERATOR_CAPACITY_MUST_BE "
                    "CHECKED_WITH_ACTIVATIONS_AND_TARGET_PRECISION"
                ),
            }
    train_seconds = meta["estimates"]["canonical_train_step_seconds"]["estimate"]
    train_tps = meta["estimates"]["optimized_tokens_per_second"]["estimate"]
    if (
        train_seconds >= criteria["train_step_median_seconds_gte"]
        or train_tps < criteria["optimized_tokens_per_second_lt"]
    ):
        return {
            "classification": "EXTRAPOLATED_CPU_ITERATION_BOUNDARY",
            "earliest_stage": ANALYTICAL_STAGE,
            "parameter_count": meta["geometry"]["parameter_count"],
            "criteria": criteria,
            "gpu_or_accelerator_engineering": "MANDATORY_FOR_SERIOUS_TRAINING",
            "distributed_training": (
                "NOT_PROVEN_MANDATORY_BY_PARAMETER_STATE_AT_APPROX_100M; THROUGHPUT/ACTIVATION "
                "EVIDENCE REQUIRED"
            ),
        }
    return {
        "classification": "NO_BOUNDARY_REACHED_IN_PROFILED_RANGE",
        "earliest_stage": None,
        "criteria": criteria,
        "gpu_or_accelerator_engineering": "NOT_ESTABLISHED",
        "distributed_training": "NOT_ESTABLISHED",
    }


def _build_report(source_sha: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if [row["stage"] for row in rows] != ["S1", "S2", "S3"]:
        raise ValueError("rows must be ordered S1, S2, S3")
    meta = _meta_row(rows)
    report: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "scope": {
            "w5_s0_profiler": "NOT_MODIFIED_NOT_REEXECUTED",
            "observed_stages": ["S1", "S2", "S3"],
            "analytical_stages": [ANALYTICAL_STAGE],
            "observed_vs_extrapolated_separated": True,
            "broad_ci_required_by_profile_workflow": False,
            "timing_metrics_in_deterministic_state_fingerprint": False,
        },
        "observed": rows,
        "extrapolated": [meta],
        "decision_support": {
            "cpu_boundary": _cpu_boundary(rows, meta),
            "compute_plan_handoff": {
                "consumer": "C01/D13 compute planning",
                "status": "MEASURED_INPUT_READY_NOT_AUTOMATICALLY_COMPOSED",
                "c01_files_modified": False,
            },
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    return report


def validate_report(report: dict[str, Any], *, source_sha: str) -> None:
    _validate_source_sha(source_sha)
    if report.get("schema") != SCHEMA_VERSION:
        raise ValueError("unexpected scale profile schema")
    if report.get("authority") != AUTHORITY:
        raise ValueError("unexpected scale profile authority")
    if report.get("source_sha") != source_sha:
        raise ValueError("scale profile source_sha mismatch")
    observed = report.get("observed")
    extrapolated = report.get("extrapolated")
    observed_stages = [row.get("stage") for row in observed] if isinstance(observed, list) else []
    if observed_stages != ["S1", "S2", "S3"]:
        raise ValueError("observed scale rows must be exactly S1/S2/S3")
    if not isinstance(extrapolated, list) or len(extrapolated) != 1:
        raise ValueError("exactly one analytical row is required")
    expected_counts = {"S1": 107856, "S2": 1066112, "S3": 10059840}
    for row in observed:
        if row.get("origin") != "OBSERVED":
            raise ValueError("observed row origin mismatch")
        if row.get("source_sha") != source_sha:
            raise ValueError("observed row source mismatch")
        if row["geometry"]["parameter_count"] != expected_counts[row["stage"]]:
            raise ValueError("observed parameter count mismatch")
        if row.get("timing_metrics_in_deterministic_state_fingerprint") is not False:
            raise ValueError("timing metrics must stay outside deterministic state fingerprints")
        train = row["measurements"]["canonical_train_microbatch_forward_backward_update"]
        if train["phase_decomposition"] != "NOT_EXPOSED_BY_PUBLIC_TRAINER_SEAM":
            raise ValueError("train phase decomposition must not overclaim")
        if train["optimized_tokens_per_second"] <= 0:
            raise ValueError("observed train throughput must be positive")
        if row["measurements"]["parameter_bytes"] <= 0:
            raise ValueError("parameter bytes must be positive")
        if row["measurements"]["optimizer_tensor_bytes"] <= 0:
            raise ValueError("optimizer tensor bytes must be positive")
    meta = extrapolated[0]
    if meta.get("origin") != "EXTRAPOLATED_ANALYTICAL" or meta.get("stage") != ANALYTICAL_STAGE:
        raise ValueError("analytical row origin/stage mismatch")
    if meta["geometry"]["status"] != "ANALYTICAL_GEOMETRY_NOT_CANONICAL_STAGE_CONFIG":
        raise ValueError("analytical geometry must not claim canonical stage status")
    if not 95_000_000 <= meta["geometry"]["parameter_count"] <= 105_000_000:
        raise ValueError("analytical geometry must remain approximately 100M parameters")
    if meta.get("timing_metrics_in_deterministic_state_fingerprint") is not False:
        raise ValueError("analytical timing metrics must stay outside deterministic fingerprints")
    expected_hash = report.get("report_sha256")
    without_hash = dict(report)
    without_hash.pop("report_sha256", None)
    if expected_hash != _canonical_hash(without_hash):
        raise ValueError("scale profile report hash mismatch")


def _run_matrix(args: argparse.Namespace) -> None:
    _require_exact_checkout(args.source_sha)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="twelve-six-scale-matrix-") as temp:
        temp_root = Path(temp)
        for stage in ("S1", "S2", "S3"):
            output = temp_root / f"{stage.lower()}.json"
            command = [
                sys.executable,
                "-m",
                "twelve_six.scale_profile",
                "stage",
                "--stage",
                stage,
                "--source-sha",
                args.source_sha,
                "--output",
                str(output),
                "--sequence-length",
                str(args.sequence_length),
                "--repetitions",
                str(args.repetitions),
                "--train-repetitions",
                str(args.train_repetitions),
                "--checkpoint-repetitions",
                str(args.checkpoint_repetitions),
                "--generation-repetitions",
                str(args.generation_repetitions),
                "--generation-new-tokens",
                str(args.generation_new_tokens),
                "--torch-threads",
                str(args.torch_threads),
                "--seed",
                str(args.seed),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.stage_timeout_seconds,
                cwd=_root(),
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{stage} profile subprocess failed with {completed.returncode}: "
                    f"{completed.stderr[-4000:]}"
                )
            rows.append(json.loads(output.read_text(encoding="utf-8")))
    report = _build_report(args.source_sha, rows)
    validate_report(report, source_sha=args.source_sha)
    Path(args.output).write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _run_stage(args: argparse.Namespace) -> None:
    row = _profile_stage(
        stage=args.stage,
        source_sha=args.source_sha,
        sequence_length=args.sequence_length,
        repetitions=args.repetitions,
        train_repetitions=args.train_repetitions,
        checkpoint_repetitions=args.checkpoint_repetitions,
        generation_repetitions=args.generation_repetitions,
        generation_new_tokens=args.generation_new_tokens,
        torch_threads=args.torch_threads,
        seed=args.seed,
    )
    Path(args.output).write_text(
        json.dumps(row, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_cli(args: argparse.Namespace) -> None:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    validate_report(report, source_sha=args.source_sha)


def _add_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--train-repetitions", type=int, default=3)
    parser.add_argument("--checkpoint-repetitions", type=int, default=1)
    parser.add_argument("--generation-repetitions", type=int, default=2)
    parser.add_argument("--generation-new-tokens", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1337)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute the isolated S1/S2/S3 matrix")
    _add_profile_args(run)
    run.add_argument("--stage-timeout-seconds", type=int, default=900)
    run.set_defaults(handler=_run_matrix)

    stage = subparsers.add_parser("stage", help="execute one observed stage in an isolated process")
    stage.add_argument("--stage", choices=tuple(OBSERVED_STAGES), required=True)
    _add_profile_args(stage)
    stage.set_defaults(handler=_run_stage)

    validate = subparsers.add_parser("validate", help="validate a completed scale profile report")
    validate.add_argument("report")
    validate.add_argument("--source-sha", required=True)
    validate.set_defaults(handler=_validate_cli)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if getattr(args, "repetitions", 1) < 1:
        raise SystemExit("repetitions must be >= 1")
    if getattr(args, "train_repetitions", 1) < 1:
        raise SystemExit("train_repetitions must be >= 1")
    if getattr(args, "checkpoint_repetitions", 1) < 1:
        raise SystemExit("checkpoint_repetitions must be >= 1")
    if getattr(args, "generation_repetitions", 1) < 1:
        raise SystemExit("generation_repetitions must be >= 1")
    if getattr(args, "sequence_length", 2) < 2:
        raise SystemExit("sequence_length must be >= 2")
    args.handler(args)


if __name__ == "__main__":
    main()
