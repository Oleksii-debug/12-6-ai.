"""GPU-201 real 10M CUDA performance campaign with fail-closed CPU behavior."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch

from twelve_six.checkpoint import (
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    verify_checkpoint,
)
from twelve_six.milestone100_first_learned import EXPECTED_CORPUS_ID
from twelve_six.model import TwelveSixDecoder
from twelve_six.scale141_10m_continuation import (
    EXPECTED_INIT_SHA,
    EXPECTED_MODEL_SHA,
    EXPECTED_PARAMETERS,
    SEED,
    _model,
    _trainer_config,
)
from twelve_six.scale141_10m_runtime_v2 import SEQ as LEARNED_SEQUENCE_LENGTH
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer
from twelve_six.training.precision import resolve_precision_runtime
from twelve_six.training.single_gpu import seed_before_model_init

SCHEMA = "12-6.gpu201-10m-cuda-benchmark.v1"
WORKER_ID = "GPU-201-10M-CUDA-BENCHMARK"
GPU199_WORKER_ID = "GPU-199-CUDA-PRECISION-PILOT"
AUTHORITY = "TARGET_DEVICE_PERFORMANCE_EVIDENCE_NOT_MODEL_PROMOTION"
PROFILE_TOOL_BLOB_SHA = "17ac055ea2ae360e28fa608893f92ada3ddee4ec"
DEFAULT_WARMUP_STEPS = 8
DEFAULT_MEASURE_STEPS = 32
DEFAULT_MAX_HEADROOM_MICROBATCH = 16


class GPU201Error(RuntimeError):
    pass


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_blob(repo: Path, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"HEAD:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _smi(field: str) -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits", "-i", "0"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    values = result.stdout.strip().splitlines()
    return values[0].strip() if values else None


def _native_bf16() -> bool:
    probe = getattr(torch.cuda, "is_bf16_supported", None)
    if probe is None:
        return False
    try:
        return bool(probe(including_emulation=False))
    except TypeError:
        return bool(probe())


def runtime_identity() -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    visible = int(torch.cuda.device_count()) if available else 0
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_cuda_runtime": getattr(torch.version, "cuda", None),
        "cuda_available": available,
        "visible_cuda_devices": visible,
        "platform": platform.platform(),
    }
    if available:
        props = torch.cuda.get_device_properties(0)
        result.update(
            {
                "cuda_device_name": str(props.name),
                "cuda_capability": list(torch.cuda.get_device_capability(0)),
                "cuda_total_memory_bytes": int(props.total_memory),
                "native_bf16_supported": _native_bf16(),
                "nvidia_driver_version": _smi("driver_version"),
                "cuda_uuid": _smi("uuid"),
            }
        )
    return result


def no_gpu_report(source_sha: str) -> dict[str, Any]:
    report = {
        "schema": SCHEMA,
        "swarm_worker_id": WORKER_ID,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "status": "NOT_RUN_NO_GPU",
        "runtime": runtime_identity(),
        "benchmark_executed": False,
        "target_device_numbers_present": False,
        "cpu_extrapolation_present": False,
        "paid_compute": False,
        "torch_compile": {"enabled": False, "status": "NOT_RUN_NO_GPU"},
    }
    report["report_sha256"] = _hash_json(report)
    return report


def _hash_json(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _nested_string(value: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> str | None:
    for path in paths:
        cursor: Any = value
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]
        if isinstance(cursor, str) and cursor.strip():
            return cursor.strip()
    return None


def load_gpu199(path: Path, runtime: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GPU201Error("GPU-199 evidence is unreadable") from exc
    if not isinstance(value, dict):
        raise GPU201Error("GPU-199 evidence root must be an object")
    worker = _nested_string(value, (("swarm_worker_id",), ("worker_id",)))
    if worker != GPU199_WORKER_ID:
        raise GPU201Error("GPU-199 worker identity mismatch")
    status = _nested_string(value, (("status",), ("result", "status")))
    invalid = {None, "BLOCKED", "FAILED", "PREPARED_NOT_LAUNCHED"}
    if status in invalid or str(status).startswith("NOT_RUN"):
        raise GPU201Error("GPU-199 has no executed CUDA result")
    selected = _nested_string(
        value,
        (
            ("decision", "selected_precision"),
            ("result", "selected_precision"),
            ("selected_precision",),
        ),
    )
    if selected not in {"fp32", "bf16", "fp16"}:
        raise GPU201Error("GPU-199 selected precision is absent or unsupported")
    device_name = _nested_string(
        value,
        (("device", "cuda_name"), ("device", "name"), ("runtime", "cuda_device_name")),
    )
    if device_name != runtime.get("cuda_device_name"):
        raise GPU201Error("GPU-199 device binding does not match current CUDA device")
    return {
        "worker_id": worker,
        "status": status,
        "selected_precision": selected,
        "cuda_device_name": device_name,
        "evidence_sha256": _hash_file(path),
        "evidence_file": path.name,
    }


def load_environment(path: Path, source_sha: str, runtime: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GPU201Error("CUDA purpose environment evidence is unreadable") from exc
    if value.get("schema_version") != "12-6.purpose-environment-evidence.v1":
        raise GPU201Error("CUDA purpose environment schema mismatch")
    if value.get("profile_id") != "linux-x86_64-cuda-training":
        raise GPU201Error("GPU-201 requires linux-x86_64-cuda-training")
    if value.get("source_sha") != source_sha:
        raise GPU201Error("CUDA purpose environment source SHA mismatch")
    verification = value.get("verification", {})
    required_checks = (
        "exact_hash_install",
        "project_wheel_install",
        "registry_validation",
        "runtime_probe",
    )
    for key in required_checks:
        if verification.get(key) != "PASS":
            raise GPU201Error(f"CUDA purpose environment failed {key}")
    probe = value.get("runtime_probe", {})
    gpu_execution = str(probe.get("gpu_execution", ""))
    if probe.get("cuda_available") is not True or gpu_execution.startswith("NOT_RUN"):
        raise GPU201Error("CUDA purpose environment did not execute on a GPU")
    if probe.get("torch_cuda") != runtime.get("pytorch_cuda_runtime"):
        raise GPU201Error("CUDA purpose environment runtime mismatch")
    return {
        "profile_id": value["profile_id"],
        "profile_sha256": value.get("profile", {}).get("profile_sha256"),
        "source_sha": value["source_sha"],
        "torch_version": probe.get("torch_version"),
        "torch_cuda": probe.get("torch_cuda"),
        "gpu_execution": probe.get("gpu_execution"),
        "evidence_sha256": _hash_file(path),
        "evidence_file": path.name,
    }


def _load_profiler(repo: Path) -> ModuleType:
    path = repo / "tools" / "profile_perf148_10m.py"
    if not path.is_file():
        raise GPU201Error("accepted PERF-148 profiler is missing")
    if _git_blob(repo, "tools/profile_perf148_10m.py") != PROFILE_TOOL_BLOB_SHA:
        raise GPU201Error("accepted PERF-148 profiler blob identity drift")
    spec = importlib.util.spec_from_file_location("gpu201_perf148", path)
    if spec is None or spec.loader is None:
        raise GPU201Error("cannot import accepted PERF-148 profiler")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_current_recipe(profiler: ModuleType) -> None:
    def current_config(*, max_steps: int, precision: str, seed: int):
        base = _trainer_config()
        return replace(base, max_steps=max_steps, precision=precision, seed=seed)

    profiler._trainer_config = current_config


def _profile_args(
    repo: Path,
    source_sha: str,
    output: Path,
    precision: str,
    batch: int,
    warmup: int,
    steps: int,
):
    return SimpleNamespace(
        repo_root=repo,
        source_sha=source_sha,
        output=output,
        trace_output=None,
        device="cuda",
        precision=precision,
        batch_size=batch,
        sequence_length=LEARNED_SEQUENCE_LENGTH,
        warmup_steps=warmup,
        profile_steps=steps,
        seed=SEED,
        data_seed=201000,
        cpu_threads=2,
    )


def _region_seconds(report: dict[str, Any], key: str) -> float | None:
    for row in report["regions"]:
        if row["key"] == key:
            return float(row["device_total_us"]) / 1_000_000.0
    return None


def run_precision_trial(profiler: ModuleType, args, runtime: dict[str, Any]) -> dict[str, Any]:
    torch.cuda.empty_cache()
    report = profiler.run(args)
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    step_wall = [float(value) for value in report["execution"]["step_wall_seconds"]]
    data_wait = [float(value) for value in report["execution"]["data_wait_seconds"]]
    optimized_tokens = args.profile_steps * args.batch_size * (args.sequence_length - 1)
    end_to_end = sum(step_wall) + sum(data_wait)
    update_parts = [
        _region_seconds(report, "perf148::gradient_normalize_and_norm") or 0.0,
        _region_seconds(report, "perf148::gradient_clip") or 0.0,
        _region_seconds(report, "perf148::optimizer_step") or 0.0,
    ]
    return {
        "precision": args.precision,
        "shape": [args.batch_size, args.sequence_length],
        "warmup_steps": args.warmup_steps,
        "measured_optimizer_steps": args.profile_steps,
        "optimized_tokens_measured": optimized_tokens,
        "optimized_tokens_per_second": optimized_tokens / end_to_end,
        "training_step_tokens_per_second": optimized_tokens / sum(step_wall),
        "timing": {
            "data_wait_seconds_total": sum(data_wait),
            "data_wait_seconds_p50": statistics.median(data_wait),
            "trainer_step_seconds_total": sum(step_wall),
            "trainer_step_seconds_p50": statistics.median(step_wall),
            "forward_and_loss_seconds_total": _region_seconds(report, "perf148::forward_and_loss"),
            "backward_seconds_total": _region_seconds(report, "perf148::backward"),
            "update_components_seconds_total": sum(update_parts),
            "update_definition": "gradient normalize/norm + clip + optimizer.step",
        },
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "total_device_memory_bytes": runtime["cuda_total_memory_bytes"],
            "peak_allocated_fraction": peak_allocated / runtime["cuda_total_memory_bytes"],
            "peak_reserved_fraction": peak_reserved / runtime["cuda_total_memory_bytes"],
        },
        "checkpoint_save_seconds": report["execution"]["checkpoint_wall_seconds"],
        "trace": {
            "initial_parameter_sha256": report["trace"]["initial_parameter_sha256"],
            "final_parameter_sha256": report["trace"]["final_parameter_sha256"],
            "fixed_batch_sha256": report["trace"]["batch_sha256"],
            "role": "CONTROLLED_FIXED_TENSOR_TRACE_FOR_TARGET_DEVICE_TIMING",
        },
        "precision_runtime": resolve_precision_runtime(args.precision, "cuda:0").to_dict(),
    }


def semantic_control(profiler: ModuleType, args, expected_final_sha: str) -> dict[str, Any]:
    total = args.warmup_steps + args.profile_steps
    config = profiler._trainer_config(max_steps=total, precision=args.precision, seed=args.seed)
    seed_before_model_init(args.seed, torch.device("cuda:0"))
    model = TwelveSixDecoder(profiler.s3_current_model_spec(), profiler.s3_init_spec())
    trainer = Trainer(model, config, device="cuda:0")
    dataset = profiler.DeterministicBatchDataset(
        steps=total,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
    )
    for index in range(total):
        trainer.train_microbatch(dataset[index])
    torch.cuda.synchronize(0)
    observed = profiler._parameter_sha256(model)
    return {
        "canonical_trainer_final_parameter_sha256": observed,
        "profiled_final_parameter_sha256": expected_final_sha,
        "exact_parameter_parity": observed == expected_final_sha,
        "finite_parameters": all(torch.isfinite(p).all().item() for p in model.parameters()),
    }


def checkpoint_roundtrip(profiler: ModuleType, args, out: Path) -> dict[str, Any]:
    config = profiler._trainer_config(max_steps=2, precision=args.precision, seed=args.seed)
    seed_before_model_init(args.seed, torch.device("cuda:0"))
    model = TwelveSixDecoder(profiler.s3_current_model_spec(), profiler.s3_init_spec())
    trainer = Trainer(model, config, device="cuda:0")
    dataset = profiler.DeterministicBatchDataset(
        steps=2,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
    )
    for index in range(2):
        trainer.train_microbatch(dataset[index])
    trainer.assert_checkpoint_safe()
    identity = profiler._checkpoint_identity(config, trainer, args.source_sha)
    checkpoint = out / "checkpoint-roundtrip"
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    save_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        identity=identity,
        overwrite=True,
    )
    torch.cuda.synchronize(0)
    save_seconds = time.perf_counter() - started
    checked = verify_checkpoint(checkpoint)

    seed_before_model_init(args.seed, torch.device("cuda:0"))
    loaded_model = TwelveSixDecoder(profiler.s3_current_model_spec(), profiler.s3_init_spec())
    loaded_trainer = Trainer(loaded_model, config, device="cuda:0")
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    load_trainer_checkpoint(
        checkpoint,
        model=loaded_model,
        trainer=loaded_trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=args.source_sha,
        expected_model_spec_hash=EXPECTED_MODEL_SHA,
    )
    torch.cuda.synchronize(0)
    load_seconds = time.perf_counter() - started
    return {
        "checkpoint_id": checked["checkpoint_id"],
        "save_seconds": save_seconds,
        "load_seconds": load_seconds,
        "optimizer_step_match": loaded_trainer.optimizer_step == trainer.optimizer_step,
        "tokens_seen_match": loaded_trainer.tokens_seen == trainer.tokens_seen,
        "parameter_sha256_match": (
            profiler._parameter_sha256(loaded_model)
            == profiler._parameter_sha256(model)
        ),
    }


def headroom_probe(profiler: ModuleType, args, maximum: int) -> dict[str, Any]:
    rows = []
    largest = None
    microbatch = 1
    stopped = False
    while microbatch <= maximum:
        torch.cuda.empty_cache()
        config = profiler._trainer_config(max_steps=2, precision=args.precision, seed=args.seed)
        seed_before_model_init(args.seed, torch.device("cuda:0"))
        model = TwelveSixDecoder(profiler.s3_current_model_spec(), profiler.s3_init_spec())
        trainer = Trainer(model, config, device="cuda:0")
        dataset = profiler.DeterministicBatchDataset(
            steps=2,
            batch_size=microbatch,
            sequence_length=args.sequence_length,
            seed=args.data_seed,
        )
        torch.cuda.reset_peak_memory_stats(0)
        try:
            for index in range(2):
                trainer.train_microbatch(dataset[index])
            torch.cuda.synchronize(0)
            rows.append(
                {
                    "microbatch": microbatch,
                    "status": "PASS",
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                }
            )
            largest = microbatch
        except torch.cuda.OutOfMemoryError:
            rows.append({"microbatch": microbatch, "status": "OOM_FAIL_CLOSED"})
            stopped = True
            break
        finally:
            del trainer
            del model
            torch.cuda.empty_cache()
        microbatch *= 2
    return {
        "sequence_length": args.sequence_length,
        "precision": args.precision,
        "tested": rows,
        "largest_passing_microbatch": largest,
        "exact_upper_bound_observed": stopped,
        "status": "BOUND_BY_OOM" if stopped else "LOWER_BOUND_ONLY",
    }


def run_cuda(args, runtime: dict[str, Any]) -> dict[str, Any]:
    if runtime["visible_cuda_devices"] != 1:
        raise GPU201Error("exactly one visible CUDA device is required")
    if args.gpu199_report is None or args.environment_evidence is None:
        raise GPU201Error("CUDA execution requires GPU-199 and purpose-environment evidence")
    gpu199 = load_gpu199(args.gpu199_report, runtime)
    environment = load_environment(args.environment_evidence, args.source_sha, runtime)
    profiler = _load_profiler(args.repo_root)
    _install_current_recipe(profiler)

    spec, init, prepared = _model(args.repo_root)
    tokenizer = ByteTokenizer()
    if spec.identity_sha256() != EXPECTED_MODEL_SHA or init.identity_sha256() != EXPECTED_INIT_SHA:
        raise GPU201Error("current 10M model/init identity drift")
    if spec.parameter_count() != EXPECTED_PARAMETERS:
        raise GPU201Error("current 10M parameter count drift")

    selected = gpu199["selected_precision"]
    precisions = ["fp32"]
    if runtime["native_bf16_supported"]:
        precisions.append("bf16")
    if selected == "fp16":
        precisions.append("fp16")
    if selected not in precisions:
        raise GPU201Error("GPU-199 selected precision cannot execute on current GPU")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trials = {}
    for precision in precisions:
        profile_args = _profile_args(
            args.repo_root,
            args.source_sha,
            args.output_dir / f"profile-{precision}.json",
            precision,
            1,
            args.warmup_steps,
            args.measure_steps,
        )
        trial = run_precision_trial(profiler, profile_args, runtime)
        trial["semantic_control"] = semantic_control(
            profiler, profile_args, trial["trace"]["final_parameter_sha256"]
        )
        if not trial["semantic_control"]["exact_parameter_parity"]:
            raise GPU201Error("profiler path changed canonical Trainer semantics")
        trials[precision] = trial

    selected_args = _profile_args(
        args.repo_root,
        args.source_sha,
        args.output_dir / "selected-profile.json",
        selected,
        1,
        args.warmup_steps,
        args.measure_steps,
    )
    roundtrip = checkpoint_roundtrip(profiler, selected_args, args.output_dir)
    roundtrip_checks = (
        "optimizer_step_match",
        "tokens_seen_match",
        "parameter_sha256_match",
    )
    if not all(roundtrip[key] for key in roundtrip_checks):
        raise GPU201Error("checkpoint save/load parity failed")
    headroom = headroom_probe(profiler, selected_args, args.headroom_max_microbatch)

    base_config = _trainer_config()
    report = {
        "schema": SCHEMA,
        "swarm_worker_id": WORKER_ID,
        "authority": AUTHORITY,
        "source_sha": args.source_sha,
        "status": "PASS_TARGET_DEVICE_MEASURED",
        "runtime": runtime,
        "gpu199": gpu199,
        "environment": environment,
        "provenance": {
            "perf148_profiler_git_blob": PROFILE_TOOL_BLOB_SHA,
            "perf148_role": "ADDITIVE_MEASUREMENT_INSTRUMENTATION",
        },
        "science_binding": {
            "model_spec_sha256": spec.identity_sha256(),
            "init_spec_sha256": init.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "candidate_id": prepared["candidate"]["id"],
            "tokenizer_version": tokenizer.identity.version,
            "tokenizer_config_sha256": tokenizer.identity.config_sha256,
            "tokenizer_vocab_sha256": tokenizer.identity.vocab_sha256,
            "corpus_identity_sha256": EXPECTED_CORPUS_ID,
            "corpus_role_in_timing_trace": (
                "IDENTITY_BOUND_NOT_LOADED_CONTROLLED_FIXED_TENSOR_TRACE"
            ),
            "sequence_length": LEARNED_SEQUENCE_LENGTH,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": base_config.learning_rate,
                "betas": list(base_config.betas),
                "eps": base_config.eps,
                "weight_decay": base_config.weight_decay,
                "gradient_clip_norm": base_config.gradient_clip_norm,
            },
        },
        "selected_precision": selected,
        "precision_trials": trials,
        "checkpoint_roundtrip": roundtrip,
        "microbatch_headroom": headroom,
        "torch_compile": {
            "enabled": False,
            "status": "DISABLED_NOT_SEPARATELY_ACCEPTED",
        },
        "benchmark_executed": True,
        "target_device_numbers_present": True,
        "cpu_extrapolation_present": False,
        "paid_compute": False,
        "truth_boundary": {
            "foreign_pretrained_weights": False,
            "sft": False,
            "rlhf": False,
            "dpo": False,
            "intelligence_claim": False,
            "production_readiness_claim": False,
            "alignment_claim": False,
            "instruction_following_claim": False,
        },
    }
    report["report_sha256"] = _hash_json(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("gpu201-evidence"))
    parser.add_argument("--gpu199-report", type=Path)
    parser.add_argument("--environment-evidence", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--measure-steps", type=int, default=DEFAULT_MEASURE_STEPS)
    parser.add_argument(
        "--headroom-max-microbatch",
        type=int,
        default=DEFAULT_MAX_HEADROOM_MICROBATCH,
    )
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if _git_head(args.repo_root) != args.source_sha:
        raise GPU201Error("exact source SHA checkout mismatch")
    if args.warmup_steps < 4:
        raise GPU201Error("GPU-201 requires at least four warmup optimizer steps")
    if args.measure_steps < 16:
        raise GPU201Error("GPU-201 requires at least sixteen measured optimizer steps")
    if args.headroom_max_microbatch < 1:
        raise GPU201Error("headroom maximum microbatch must be positive")
    runtime = runtime_identity()
    report = (
        no_gpu_report(args.source_sha)
        if not runtime["cuda_available"]
        else run_cuda(args, runtime)
    )
    _write_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
