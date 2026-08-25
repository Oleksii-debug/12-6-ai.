"""Exact-parity LOCAL_FREE CPU training performance experiments.

This module deliberately keeps machine-local performance choices outside ModelSpec,
InitSpec, TrainerConfig, and checkpoint identity. Auto mode measures candidates on the
current machine; explicit mode verifies and records a caller-supplied profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .model import InitSpec, TwelveSixDecoder
from .scaling_experiment import (
    _byte_stream,
    _make_batch,
    _read_jsonl,
    _trainer_config,
    controlled_specs,
)
from .tokenization import ByteTokenizer
from .training import Trainer

SCHEMA = "12-6.cpu-training-performance.v1"
PROFILE_SCHEMA = "12-6.cpu-training-profile.v1"
AUTHORITY = "LOCAL_FREE_CPU_PERFORMANCE_EVIDENCE_NOT_MODEL_OR_TRAINING_SEMANTICS"
REFERENCE_BATCH_SIZE = 4
REFERENCE_SEQUENCE_LENGTH = 64
REFERENCE_TOKENS_PER_STEP = REFERENCE_BATCH_SIZE * (REFERENCE_SEQUENCE_LENGTH - 1)
_DEFAULT_SCALE_INDEXES = (0, 2, 3)
_SCALE_LABELS = {0: "100K", 2: "500K", 3: "1M"}
_EXPECTED_PARAMETERS = {0: 95_568, 2: 467_808, 3: 1_037_696}


class CPUPerformanceError(RuntimeError):
    """Raised when a performance profile cannot preserve exact training semantics."""


@dataclass(frozen=True, slots=True)
class CPUTrainingProfile:
    """Machine-local execution knobs; never part of model/training semantic identity."""

    torch_threads: int
    interop_threads: int
    dataloader_workers: int
    compile_model: bool = False
    batch_size: int = REFERENCE_BATCH_SIZE
    sequence_length: int = REFERENCE_SEQUENCE_LENGTH

    def __post_init__(self) -> None:
        if self.torch_threads <= 0 or self.interop_threads <= 0:
            raise ValueError("torch and inter-op thread counts must be positive")
        if self.dataloader_workers < 0:
            raise ValueError("dataloader_workers must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.sequence_length < 2 or self.sequence_length > 256:
            raise ValueError("sequence_length must be in [2, 256]")

    @property
    def valid_targets_per_step(self) -> int:
        return self.batch_size * (self.sequence_length - 1)

    @property
    def batch_shape(self) -> tuple[int, int]:
        return (self.batch_size, self.sequence_length)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = PROFILE_SCHEMA
        payload["valid_targets_per_step"] = self.valid_targets_per_step
        payload["persistent_workers"] = self.dataloader_workers > 0
        payload["profile_changes_model_identity"] = False
        payload["profile_changes_training_math"] = False
        return payload


class _DeterministicBatchDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        stream: bytes,
        *,
        steps: int,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        self.stream = stream
        self.steps = steps
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < self.steps:
            raise IndexError(index)
        return _make_batch(
            self.stream,
            step=index,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
        )


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_hash_update(digest: Any, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(struct.pack("!I", value.ndim))
    for dimension in value.shape:
        digest.update(struct.pack("!Q", int(dimension)))
    digest.update(value.numpy().tobytes(order="C"))


def _parameter_value_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    parameters = list(model.parameters())
    digest.update(struct.pack("!Q", len(parameters)))
    for parameter in parameters:
        _tensor_hash_update(digest, parameter)
    return digest.hexdigest()


def _batch_trace_sha256(batches: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("!Q", len(batches)))
    for batch in batches:
        _tensor_hash_update(digest, batch)
    return digest.hexdigest()


def _float_trace_sha256(values: list[float]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("!Q", len(values)))
    for value in values:
        digest.update(float(value).hex().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _text_trace_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("!Q", len(values)))
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _linux_cpu_details() -> dict[str, Any]:
    path = Path("/proc/cpuinfo")
    if not path.exists():
        return {"model_name": None, "physical_cores_detected": None}
    model_name: str | None = None
    physical_pairs: set[tuple[str, str]] = set()
    physical_id: str | None = None
    core_id: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
        if not raw_line.strip():
            if physical_id is not None and core_id is not None:
                physical_pairs.add((physical_id, core_id))
            physical_id = None
            core_id = None
            continue
        key, _, value = raw_line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "model name" and model_name is None:
            model_name = value
        elif key == "physical id":
            physical_id = value
        elif key == "core id":
            core_id = value
    return {
        "model_name": model_name,
        "physical_cores_detected": len(physical_pairs) if physical_pairs else None,
    }


def hardware_metadata() -> dict[str, Any]:
    linux = _linux_cpu_details()
    backends = {
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "mkl_available": bool(getattr(torch.backends, "mkl", None) and torch.backends.mkl.is_available()),
        "openmp_available": bool(
            getattr(torch.backends, "openmp", None) and torch.backends.openmp.is_available()
        ),
    }
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_model_name": linux["model_name"],
        "logical_cpus": os.cpu_count(),
        "physical_cores_detected": linux["physical_cores_detected"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_backends": backends,
        "environment_thread_hints": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _configure_worker_threads(profile: CPUTrainingProfile) -> None:
    torch.set_num_interop_threads(profile.interop_threads)
    torch.set_num_threads(profile.torch_threads)
    if torch.get_num_interop_threads() != profile.interop_threads:
        raise CPUPerformanceError("PyTorch inter-op thread setting did not apply exactly")
    if torch.get_num_threads() != profile.torch_threads:
        raise CPUPerformanceError("PyTorch intra-op thread setting did not apply exactly")


def _run_worker(
    *,
    repo_root: Path,
    spec_index: int,
    profile: CPUTrainingProfile,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and measured_steps positive")
    if spec_index not in _SCALE_LABELS:
        raise ValueError(f"unsupported PERF-58 scale index: {spec_index}")

    _configure_worker_threads(profile)
    torch.use_deterministic_algorithms(True, warn_only=False)
    total_steps = warmup_steps + measured_steps
    tokenizer = ByteTokenizer()
    train_records = _read_jsonl(repo_root / "data/s0/packaged/train.jsonl")
    train_stream = _byte_stream(train_records, tokenizer)
    spec = controlled_specs()[spec_index]
    if spec.parameter_count() != _EXPECTED_PARAMETERS[spec_index]:
        raise CPUPerformanceError("controlled scaling family parameter identity drift")

    torch.manual_seed(1337)
    model: torch.nn.Module = TwelveSixDecoder(spec, InitSpec())
    original_model = model
    compile_wrap_seconds = 0.0
    if profile.compile_model:
        if not hasattr(torch, "compile"):
            raise CPUPerformanceError("torch.compile is unavailable in this runtime")
        compile_started = time.perf_counter()
        model = torch.compile(model)
        compile_wrap_seconds = time.perf_counter() - compile_started

    trainer = Trainer(
        model,
        _trainer_config(max_steps=total_steps, seed=1337),
        device="cpu",
    )
    initial_parameter_sha256 = _parameter_value_sha256(original_model)
    dataset = _DeterministicBatchDataset(
        train_stream,
        steps=total_steps,
        batch_size=profile.batch_size,
        sequence_length=profile.sequence_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=profile.dataloader_workers,
        persistent_workers=profile.dataloader_workers > 0,
        pin_memory=False,
    )

    iterator_started = time.perf_counter()
    iterator = iter(loader)
    iterator_start_seconds = time.perf_counter() - iterator_started
    batch_values: list[torch.Tensor] = []
    losses: list[float] = []
    update_hashes: list[str] = []
    measured_step_seconds: list[float] = []
    measured_data_wait_seconds: list[float] = []
    warmup_wall_seconds = 0.0

    for step in range(total_steps):
        wait_started = time.perf_counter()
        batch = next(iterator)
        data_wait = time.perf_counter() - wait_started
        if not isinstance(batch, torch.Tensor):
            raise TypeError("deterministic DataLoader must yield Tensor batches")
        batch_values.append(batch.clone())

        step_started = time.perf_counter()
        metrics = trainer.train_microbatch({"input_ids": batch})
        train_seconds = time.perf_counter() - step_started
        losses.append(metrics.loss)
        update_hashes.append(_parameter_value_sha256(original_model))
        if step < warmup_steps:
            warmup_wall_seconds += data_wait + train_seconds
        else:
            measured_data_wait_seconds.append(data_wait)
            measured_step_seconds.append(train_seconds)

    trainer.assert_checkpoint_safe()
    final_parameter_sha256 = _parameter_value_sha256(original_model)
    full_step_seconds = [
        data_wait + train
        for data_wait, train in zip(
            measured_data_wait_seconds,
            measured_step_seconds,
            strict=True,
        )
    ]
    median_full_step = statistics.median(full_step_seconds)
    median_train_step = statistics.median(measured_step_seconds)
    median_data_wait = statistics.median(measured_data_wait_seconds)
    measured_wall = sum(full_step_seconds)
    measured_tokens = measured_steps * profile.valid_targets_per_step
    startup_seconds = compile_wrap_seconds + iterator_start_seconds + warmup_wall_seconds

    result = {
        "status": "PASS",
        "scale": _SCALE_LABELS[spec_index],
        "spec_index": spec_index,
        "parameters": spec.parameter_count(),
        "model_identity_sha256": spec.identity_sha256(),
        "profile": profile.to_dict(),
        "runtime_observed": {
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        },
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "optimizer_steps": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
        "initial_parameter_sha256": initial_parameter_sha256,
        "final_parameter_sha256": final_parameter_sha256,
        "batch_trace_sha256": _batch_trace_sha256(batch_values),
        "loss_trace_sha256": _float_trace_sha256(losses),
        "update_trace_sha256": _text_trace_sha256(update_hashes),
        "loss_trace_hex": [float(value).hex() for value in losses],
        "timing": {
            "compile_wrap_seconds": compile_wrap_seconds,
            "iterator_start_seconds": iterator_start_seconds,
            "warmup_wall_seconds": warmup_wall_seconds,
            "calibration_startup_seconds": startup_seconds,
            "measured_wall_seconds": measured_wall,
            "median_train_step_seconds": median_train_step,
            "median_data_wait_seconds": median_data_wait,
            "median_full_step_seconds": median_full_step,
            "min_full_step_seconds": min(full_step_seconds),
            "max_full_step_seconds": max(full_step_seconds),
            "measured_tokens_per_second": measured_tokens / measured_wall,
        },
    }
    return result


def _worker_command(
    *,
    repo_root: Path,
    spec_index: int,
    profile: CPUTrainingProfile,
    warmup_steps: int,
    measured_steps: int,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "twelve_six.cpu_training_perf",
        "worker",
        "--repo-root",
        str(repo_root),
        "--spec-index",
        str(spec_index),
        "--torch-threads",
        str(profile.torch_threads),
        "--interop-threads",
        str(profile.interop_threads),
        "--dataloader-workers",
        str(profile.dataloader_workers),
        "--batch-size",
        str(profile.batch_size),
        "--sequence-length",
        str(profile.sequence_length),
        "--warmup-steps",
        str(warmup_steps),
        "--measured-steps",
        str(measured_steps),
        "--output",
        str(output),
    ]
    if profile.compile_model:
        command.append("--compile-model")
    return command


def _run_candidate_subprocess(
    *,
    repo_root: Path,
    spec_index: int,
    profile: CPUTrainingProfile,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="perf58-") as temp_dir:
        output = Path(temp_dir) / "candidate.json"
        completed = subprocess.run(
            _worker_command(
                repo_root=repo_root,
                spec_index=spec_index,
                profile=profile,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
                output=output,
            ),
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0 or not output.exists():
            return {
                "status": "ERROR",
                "scale": _SCALE_LABELS[spec_index],
                "spec_index": spec_index,
                "parameters": _EXPECTED_PARAMETERS[spec_index],
                "profile": profile.to_dict(),
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-4000:],
                "stdout_tail": completed.stdout[-2000:],
            }
        payload = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("worker result must be a JSON object")
        return payload


def _parity_against(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("status") != "PASS":
        return {
            "exact": False,
            "reason": "candidate_execution_failed",
            "checks": {},
        }
    fields = (
        "model_identity_sha256",
        "initial_parameter_sha256",
        "batch_trace_sha256",
        "loss_trace_sha256",
        "update_trace_sha256",
        "final_parameter_sha256",
        "optimizer_steps",
        "tokens_seen",
    )
    checks = {field: candidate.get(field) == reference.get(field) for field in fields}
    return {
        "exact": all(checks.values()),
        "reason": "bitwise_equal" if all(checks.values()) else "exact_parity_mismatch",
        "checks": checks,
    }


def _projected_seconds(candidate: dict[str, Any], horizon_steps: int) -> float:
    timing = candidate["timing"]
    return float(timing["calibration_startup_seconds"]) + horizon_steps * float(
        timing["median_full_step_seconds"]
    )


def _choose_if_materially_faster(
    incumbent: dict[str, Any],
    challenger: dict[str, Any],
    *,
    horizon_steps: int,
    minimum_speedup: float,
) -> dict[str, Any]:
    if not challenger.get("parity", {}).get("exact", False):
        return incumbent
    incumbent_seconds = _projected_seconds(incumbent, horizon_steps)
    challenger_seconds = _projected_seconds(challenger, horizon_steps)
    if challenger_seconds <= incumbent_seconds / minimum_speedup:
        return challenger
    return incumbent


def _candidate_key(profile: CPUTrainingProfile) -> str:
    compile_name = "compile" if profile.compile_model else "eager"
    return (
        f"t{profile.torch_threads}-i{profile.interop_threads}-w{profile.dataloader_workers}-"
        f"{compile_name}-b{profile.batch_size}x{profile.sequence_length}"
    )


def _attach_parity(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = dict(candidate)
    result["candidate_id"] = _candidate_key(
        CPUTrainingProfile(
            torch_threads=int(candidate["profile"]["torch_threads"]),
            interop_threads=int(candidate["profile"]["interop_threads"]),
            dataloader_workers=int(candidate["profile"]["dataloader_workers"]),
            compile_model=bool(candidate["profile"]["compile_model"]),
            batch_size=int(candidate["profile"]["batch_size"]),
            sequence_length=int(candidate["profile"]["sequence_length"]),
        )
    )
    result["parity"] = _parity_against(reference, candidate)
    return result


def _bounded_thread_candidates(max_threads: int) -> tuple[int, ...]:
    logical = os.cpu_count() or 1
    cap = max(1, min(max_threads, logical))
    values = {1, cap}
    for value in (2, 4, 8):
        if value <= cap:
            values.add(value)
    return tuple(sorted(values))


def _bounded_interop_candidates(max_interop_threads: int) -> tuple[int, ...]:
    logical = os.cpu_count() or 1
    cap = max(1, min(max_interop_threads, logical))
    return tuple(value for value in (1, 2, 4) if value <= cap)


def _bounded_worker_candidates(max_workers: int) -> tuple[int, ...]:
    logical = os.cpu_count() or 1
    cap = max(0, min(max_workers, logical))
    values = {0, cap}
    for value in (1, 2, 4):
        if value <= cap:
            values.add(value)
    return tuple(sorted(values))


def _benchmark_scale_auto(
    *,
    repo_root: Path,
    spec_index: int,
    warmup_steps: int,
    measured_steps: int,
    horizon_steps: int,
    max_threads: int,
    max_interop_threads: int,
    max_workers: int,
    include_compile: bool,
    minimum_speedup: float,
) -> dict[str, Any]:
    reference_profile = CPUTrainingProfile(1, 1, 0, False)
    reference = _run_candidate_subprocess(
        repo_root=repo_root,
        spec_index=spec_index,
        profile=reference_profile,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )
    if reference.get("status") != "PASS":
        raise CPUPerformanceError(
            f"reference worker failed for {_SCALE_LABELS[spec_index]}: {reference.get('stderr_tail')}"
        )
    reference = _attach_parity(reference, reference)
    candidates: list[dict[str, Any]] = [reference]
    selected = reference

    for threads in _bounded_thread_candidates(max_threads):
        profile = CPUTrainingProfile(threads, 1, 0, False)
        if profile == reference_profile:
            continue
        candidate = _attach_parity(
            reference,
            _run_candidate_subprocess(
                repo_root=repo_root,
                spec_index=spec_index,
                profile=profile,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
            ),
        )
        candidates.append(candidate)
        selected = _choose_if_materially_faster(
            selected,
            candidate,
            horizon_steps=horizon_steps,
            minimum_speedup=minimum_speedup,
        )

    selected_profile = CPUTrainingProfile(
        int(selected["profile"]["torch_threads"]),
        1,
        0,
        False,
    )
    for interop in _bounded_interop_candidates(max_interop_threads):
        profile = CPUTrainingProfile(selected_profile.torch_threads, interop, 0, False)
        if any(item.get("candidate_id") == _candidate_key(profile) for item in candidates):
            continue
        candidate = _attach_parity(
            reference,
            _run_candidate_subprocess(
                repo_root=repo_root,
                spec_index=spec_index,
                profile=profile,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
            ),
        )
        candidates.append(candidate)
        selected = _choose_if_materially_faster(
            selected,
            candidate,
            horizon_steps=horizon_steps,
            minimum_speedup=minimum_speedup,
        )

    selected_profile = CPUTrainingProfile(
        int(selected["profile"]["torch_threads"]),
        int(selected["profile"]["interop_threads"]),
        0,
        False,
    )
    for workers in _bounded_worker_candidates(max_workers):
        profile = CPUTrainingProfile(
            selected_profile.torch_threads,
            selected_profile.interop_threads,
            workers,
            False,
        )
        if any(item.get("candidate_id") == _candidate_key(profile) for item in candidates):
            continue
        candidate = _attach_parity(
            reference,
            _run_candidate_subprocess(
                repo_root=repo_root,
                spec_index=spec_index,
                profile=profile,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
            ),
        )
        candidates.append(candidate)
        selected = _choose_if_materially_faster(
            selected,
            candidate,
            horizon_steps=horizon_steps,
            minimum_speedup=minimum_speedup,
        )

    if include_compile:
        profile = CPUTrainingProfile(
            int(selected["profile"]["torch_threads"]),
            int(selected["profile"]["interop_threads"]),
            int(selected["profile"]["dataloader_workers"]),
            True,
        )
        candidate = _attach_parity(
            reference,
            _run_candidate_subprocess(
                repo_root=repo_root,
                spec_index=spec_index,
                profile=profile,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
            ),
        )
        candidates.append(candidate)
        selected = _choose_if_materially_faster(
            selected,
            candidate,
            horizon_steps=horizon_steps,
            minimum_speedup=minimum_speedup,
        )

    baseline_projected = _projected_seconds(reference, horizon_steps)
    selected_projected = _projected_seconds(selected, horizon_steps)
    speedup = baseline_projected / selected_projected
    return {
        "scale": _SCALE_LABELS[spec_index],
        "spec_index": spec_index,
        "parameters": _EXPECTED_PARAMETERS[spec_index],
        "model_identity_sha256": reference["model_identity_sha256"],
        "reference_candidate_id": reference["candidate_id"],
        "selected_candidate_id": selected["candidate_id"],
        "selected_profile": selected["profile"],
        "selection_horizon_steps": horizon_steps,
        "minimum_material_speedup_x": minimum_speedup,
        "baseline_projected_seconds": baseline_projected,
        "selected_projected_seconds": selected_projected,
        "speedup_x": speedup,
        "baseline_median_full_step_seconds": reference["timing"]["median_full_step_seconds"],
        "selected_median_full_step_seconds": selected["timing"]["median_full_step_seconds"],
        "baseline_measured_tokens_per_second": reference["timing"]["measured_tokens_per_second"],
        "selected_measured_tokens_per_second": selected["timing"]["measured_tokens_per_second"],
        "exact_parity": bool(selected["parity"]["exact"]),
        "candidates": candidates,
    }


def _benchmark_scale_explicit(
    *,
    repo_root: Path,
    spec_index: int,
    profile: CPUTrainingProfile,
    warmup_steps: int,
    measured_steps: int,
    horizon_steps: int,
) -> dict[str, Any]:
    reference_profile = CPUTrainingProfile(1, 1, 0, False)
    reference = _run_candidate_subprocess(
        repo_root=repo_root,
        spec_index=spec_index,
        profile=reference_profile,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )
    if reference.get("status") != "PASS":
        raise CPUPerformanceError("reference worker failed")
    reference = _attach_parity(reference, reference)
    candidate = reference
    if profile != reference_profile:
        candidate = _attach_parity(
            reference,
            _run_candidate_subprocess(
                repo_root=repo_root,
                spec_index=spec_index,
                profile=profile,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
            ),
        )
    if not candidate.get("parity", {}).get("exact", False):
        raise CPUPerformanceError(
            f"explicit profile failed exact parity at {_SCALE_LABELS[spec_index]}: "
            f"{candidate.get('parity')}"
        )
    baseline_projected = _projected_seconds(reference, horizon_steps)
    selected_projected = _projected_seconds(candidate, horizon_steps)
    return {
        "scale": _SCALE_LABELS[spec_index],
        "spec_index": spec_index,
        "parameters": _EXPECTED_PARAMETERS[spec_index],
        "model_identity_sha256": reference["model_identity_sha256"],
        "reference_candidate_id": reference["candidate_id"],
        "selected_candidate_id": candidate["candidate_id"],
        "selected_profile": candidate["profile"],
        "selection_horizon_steps": horizon_steps,
        "baseline_projected_seconds": baseline_projected,
        "selected_projected_seconds": selected_projected,
        "speedup_x": baseline_projected / selected_projected,
        "baseline_median_full_step_seconds": reference["timing"]["median_full_step_seconds"],
        "selected_median_full_step_seconds": candidate["timing"]["median_full_step_seconds"],
        "baseline_measured_tokens_per_second": reference["timing"]["measured_tokens_per_second"],
        "selected_measured_tokens_per_second": candidate["timing"]["measured_tokens_per_second"],
        "exact_parity": True,
        "candidates": [reference] if candidate is reference else [reference, candidate],
    }


def _batch_shape_probes(
    *,
    repo_root: Path,
    warmup_steps: int,
    measured_steps: int,
) -> list[dict[str, Any]]:
    spec_index = 2
    reference_profile = CPUTrainingProfile(1, 1, 0, False)
    reference = _run_candidate_subprocess(
        repo_root=repo_root,
        spec_index=spec_index,
        profile=reference_profile,
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
    )
    if reference.get("status") != "PASS":
        raise CPUPerformanceError("batch-shape reference worker failed")
    probes: list[dict[str, Any]] = []
    for batch_size, sequence_length in ((6, 43), (12, 22)):
        profile = CPUTrainingProfile(1, 1, 0, False, batch_size, sequence_length)
        candidate = _run_candidate_subprocess(
            repo_root=repo_root,
            spec_index=spec_index,
            profile=profile,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
        )
        parity = _parity_against(reference, candidate)
        probes.append(
            {
                "profile": profile.to_dict(),
                "same_valid_targets_per_step": (
                    profile.valid_targets_per_step == REFERENCE_TOKENS_PER_STEP
                ),
                "measured": candidate,
                "parity": parity,
                "eligible_for_semantics_preserving_auto_selection": False,
                "reason": (
                    "Changing [batch,time] changes causal attention grouping and data boundaries; "
                    "equal target count is not equal training math. The exact guard must reject it."
                ),
            }
        )
    return probes


def run_performance_experiment(
    *,
    repo_root: Path,
    source_sha: str,
    output_path: Path,
    mode: str,
    warmup_steps: int,
    measured_steps: int,
    horizon_steps: int,
    max_threads: int,
    max_interop_threads: int,
    max_workers: int,
    include_compile: bool,
    minimum_speedup: float,
    explicit_profile: CPUTrainingProfile | None = None,
) -> dict[str, Any]:
    if mode not in {"auto", "explicit"}:
        raise ValueError("mode must be auto or explicit")
    if len(source_sha) != 40 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ValueError("source_sha must be lowercase 40-hex")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if not 1.0 <= minimum_speedup <= 2.0:
        raise ValueError("minimum_speedup must be in [1.0, 2.0]")
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if observed != source_sha:
        raise CPUPerformanceError(
            f"exact-checkout mismatch: expected {source_sha}, observed {observed}"
        )
    if mode == "explicit" and explicit_profile is None:
        raise ValueError("explicit mode requires explicit_profile")

    scales: list[dict[str, Any]] = []
    for spec_index in _DEFAULT_SCALE_INDEXES:
        if mode == "auto":
            scale = _benchmark_scale_auto(
                repo_root=repo_root,
                spec_index=spec_index,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
                horizon_steps=horizon_steps,
                max_threads=max_threads,
                max_interop_threads=max_interop_threads,
                max_workers=max_workers,
                include_compile=include_compile,
                minimum_speedup=minimum_speedup,
            )
        else:
            assert explicit_profile is not None
            scale = _benchmark_scale_explicit(
                repo_root=repo_root,
                spec_index=spec_index,
                profile=explicit_profile,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
                horizon_steps=horizon_steps,
            )
        scales.append(scale)

    shape_probes = _batch_shape_probes(
        repo_root=repo_root,
        warmup_steps=max(0, min(warmup_steps, 1)),
        measured_steps=max(1, min(measured_steps, 3)),
    )
    compile_measurements = [
        candidate
        for scale in scales
        for candidate in scale["candidates"]
        if candidate.get("profile", {}).get("compile_model") is True
    ]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
        },
        "execution": {
            "mode": mode,
            "paid_compute": False,
            "device": "cpu",
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "selection_horizon_steps": horizon_steps,
            "minimum_material_speedup_x": minimum_speedup,
            "compile_requested": include_compile,
            "compile_candidates_measured": len(compile_measurements),
        },
        "hardware": hardware_metadata(),
        "selection_policy": {
            "reference": CPUTrainingProfile(1, 1, 0, False).to_dict(),
            "axes": [
                "torch_threads",
                "interop_threads",
                "dataloader_workers",
                "compile_model_if_requested",
            ],
            "fresh_process_per_candidate": True,
            "exact_parity_fields": [
                "model_identity_sha256",
                "initial_parameter_sha256",
                "batch_trace_sha256",
                "loss_trace_sha256",
                "update_trace_sha256",
                "final_parameter_sha256",
                "optimizer_steps",
                "tokens_seen",
            ],
            "timing_selection": (
                "calibration startup plus median full step projected to selection horizon; "
                "challenger must meet minimum material speedup"
            ),
            "hardware_specific_defaults_written_globally": False,
        },
        "scales": scales,
        "batch_shape_evaluation": shape_probes,
        "explicit_replay_profiles": {
            scale["scale"]: scale["selected_profile"] for scale in scales
        },
        "truth_boundary": {
            "model_identity_changed": False,
            "training_math_changed": False,
            "optimizer_semantics_changed": False,
            "data_order_changed_for_selected_profiles": False,
            "hardware_result_universalized": False,
            "gpu_behavior_inferred": False,
            "paid_compute_used": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def validate_report(report: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("CPU performance report schema/authority mismatch")
    source = report.get("source")
    if not isinstance(source, dict) or source.get("repository") != "Oleksii-debug/12-6-ai.":
        raise ValueError("unexpected repository identity")
    source_sha = source.get("git_sha")
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ValueError("CPU performance source SHA mismatch")
    execution = report.get("execution")
    if not isinstance(execution, dict) or execution.get("paid_compute") is not False:
        raise ValueError("CPU performance report may not claim paid compute")
    scales = report.get("scales")
    if not isinstance(scales, list) or [item.get("scale") for item in scales] != [
        "100K",
        "500K",
        "1M",
    ]:
        raise ValueError("expected PERF-58 100K/500K/1M scale measurements")
    specs = controlled_specs()
    for scale in scales:
        spec_index = int(scale["spec_index"])
        if int(scale["parameters"]) != _EXPECTED_PARAMETERS[spec_index]:
            raise ValueError("PERF-58 parameter family drift")
        if scale["model_identity_sha256"] != specs[spec_index].identity_sha256():
            raise ValueError("performance profile changed or misbound model identity")
        if scale.get("exact_parity") is not True:
            raise ValueError("selected CPU profile lacks exact loss/update parity")
        selected_id = scale.get("selected_candidate_id")
        selected = [
            candidate
            for candidate in scale.get("candidates", [])
            if candidate.get("candidate_id") == selected_id
        ]
        if len(selected) != 1 or selected[0].get("parity", {}).get("exact") is not True:
            raise ValueError("selected profile is not exactly parity-qualified")
        if float(scale.get("speedup_x", 0.0)) <= 0.0:
            raise ValueError("invalid measured speedup")
    shape_probes = report.get("batch_shape_evaluation")
    if not isinstance(shape_probes, list) or len(shape_probes) < 2:
        raise ValueError("batch-shape evaluation is missing")
    for probe in shape_probes:
        if probe.get("eligible_for_semantics_preserving_auto_selection") is not False:
            raise ValueError("batch-shape semantic boundary weakened")
        if probe.get("same_valid_targets_per_step") is not True:
            raise ValueError("batch-shape probes must hold valid-target count fixed")
    truth = report.get("truth_boundary")
    if not isinstance(truth, dict) or any(value is not False for value in truth.values()):
        raise ValueError("CPU performance truth boundary was weakened")
    supplied_hash = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if supplied_hash != _canonical_hash(unsigned):
        raise ValueError("CPU performance report self-hash mismatch")


def _add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--dataloader-workers", type=int, default=0)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--batch-size", type=int, default=REFERENCE_BATCH_SIZE)
    parser.add_argument("--sequence-length", type=int, default=REFERENCE_SEQUENCE_LENGTH)


def _profile_from_args(args: argparse.Namespace) -> CPUTrainingProfile:
    return CPUTrainingProfile(
        args.torch_threads,
        args.interop_threads,
        args.dataloader_workers,
        args.compile_model,
        args.batch_size,
        args.sequence_length,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--repo-root", type=Path, default=Path("."))
    worker.add_argument("--spec-index", type=int, required=True)
    worker.add_argument("--warmup-steps", type=int, default=2)
    worker.add_argument("--measured-steps", type=int, default=8)
    worker.add_argument("--output", type=Path, required=True)
    _add_profile_arguments(worker)

    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--source-sha", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--mode", choices=("auto", "explicit"), default="auto")
    run.add_argument("--warmup-steps", type=int, default=2)
    run.add_argument("--measured-steps", type=int, default=8)
    run.add_argument("--selection-horizon-steps", type=int, default=256)
    run.add_argument("--max-threads", type=int, default=8)
    run.add_argument("--max-interop-threads", type=int, default=2)
    run.add_argument("--max-workers", type=int, default=2)
    run.add_argument("--include-compile", action="store_true")
    run.add_argument("--minimum-speedup", type=float, default=1.03)
    _add_profile_arguments(run)

    validate = subparsers.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "worker":
        result = _run_worker(
            repo_root=args.repo_root.resolve(),
            spec_index=args.spec_index,
            profile=_profile_from_args(args),
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0
    if args.command == "run":
        profile = _profile_from_args(args) if args.mode == "explicit" else None
        report = run_performance_experiment(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            output_path=args.output,
            mode=args.mode,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            horizon_steps=args.selection_horizon_steps,
            max_threads=args.max_threads,
            max_interop_threads=args.max_interop_threads,
            max_workers=args.max_workers,
            include_compile=args.include_compile,
            minimum_speedup=args.minimum_speedup,
            explicit_profile=profile,
        )
        validate_report(report, expected_source_sha=args.source_sha)
        print(
            json.dumps(
                {
                    "scales": [
                        {
                            "scale": scale["scale"],
                            "parameters": scale["parameters"],
                            "selected_profile": scale["selected_profile"],
                            "speedup_x": scale["speedup_x"],
                        }
                        for scale in report["scales"]
                    ],
                    "report_sha256": report["report_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("report must be a JSON object")
    validate_report(report, expected_source_sha=args.expected_source_sha)
    print(f"{SCHEMA}: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
