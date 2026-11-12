"""Observe native training resources without changing the scientific recipe."""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Any

import torch

from ..checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from ..model import TwelveSixDecoder, count_trainable_parameters
from .trainer import Batch, Trainer


class ResourceAdmissionError(RuntimeError):
    """The requested native training path cannot fit the observed host."""


@dataclass(frozen=True, slots=True)
class HostResources:
    device: str
    host_available_bytes: int | None
    device_available_bytes: int | None
    checkpoint_volume_free_bytes: int


def _host_memory_available() -> int | None:
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_uint32),
                ("load", ctypes.c_uint32),
                ("total_physical", ctypes.c_uint64),
                ("available_physical", ctypes.c_uint64),
                ("total_page", ctypes.c_uint64),
                ("available_page", ctypes.c_uint64),
                ("total_virtual", ctypes.c_uint64),
                ("available_virtual", ctypes.c_uint64),
                ("extended_virtual", ctypes.c_uint64),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx(
            ctypes.byref(status)
        ):
            return int(status.available_physical)
        return None
    try:
        available = os.sysconf("SC_AVPHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return int(available * size) if available >= 0 and size > 0 else None
    except (AttributeError, OSError, ValueError):
        return None


def observe_host_resources(checkpoint_root: Path, device: str) -> HostResources:
    """Read available memory and the actual checkpoint volume before allocation."""
    try:
        target = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ResourceAdmissionError(f"invalid training device {device!r}") from exc
    host = _host_memory_available()
    if target.type == "cpu":
        device_available = host
    elif target.type == "cuda":
        try:
            device_available = (
                int(torch.cuda.mem_get_info(target)[0]) if torch.cuda.is_available() else None
            )
        except (RuntimeError, ValueError) as exc:
            raise ResourceAdmissionError(f"could not observe available memory on {target}") from exc
    else:
        raise ResourceAdmissionError(
            f"native resource observation is unavailable for {target.type}"
        )
    ancestor = Path(checkpoint_root).absolute()
    while not ancestor.exists():
        if ancestor == ancestor.parent:
            raise ResourceAdmissionError(f"checkpoint volume is unavailable for {checkpoint_root}")
        ancestor = ancestor.parent
    return HostResources(str(target), host, device_available, shutil.disk_usage(ancestor).free)


def assess_resource_envelope(
    parameter_count: int,
    resources: HostResources,
    *,
    remaining_updates: int,
    initial_checkpoint_needed: bool,
) -> dict[str, Any]:
    """Screen the native FP32-parameter AdamW path against arithmetic floors."""
    if type(parameter_count) is not int or parameter_count < 1:
        raise ValueError("parameter count must be a positive integer")
    if type(remaining_updates) is not int or remaining_updates < 0:
        raise ValueError("remaining updates must be a nonnegative integer")
    if type(initial_checkpoint_needed) is not bool:
        raise ValueError("initial checkpoint requirement must be boolean")
    if type(resources.checkpoint_volume_free_bytes) is not int:
        raise ValueError("checkpoint volume observation must be an integer")
    for amount in (
        resources.host_available_bytes,
        resources.device_available_bytes,
        resources.checkpoint_volume_free_bytes,
    ):
        if amount is not None and (type(amount) is not int or amount < 0):
            raise ValueError("resource observations must be nonnegative byte counts")
    weight_bytes = 4 * parameter_count
    training_bytes = 16 * parameter_count
    # The current native save holds live weights/moments plus the two moment
    # copies made by Trainer.state_dict and dataclass serialization.
    save_device_bytes = 28 * parameter_count
    host_staging_bytes = 12 * parameter_count
    checkpoint_bytes = 12 * parameter_count
    unpruned_bytes = remaining_updates * checkpoint_bytes
    if initial_checkpoint_needed:
        unpruned_bytes += weight_bytes
    reasons = []
    if resources.device_available_bytes is None:
        reasons.append("DEVICE_MEMORY_UNOBSERVED")
    elif resources.device_available_bytes < training_bytes:
        reasons.append("DEVICE_MEMORY_BELOW_FP32_ADAMW_FLOOR")
    elif remaining_updates and resources.device_available_bytes < save_device_bytes:
        reasons.append("DEVICE_MEMORY_BELOW_NATIVE_SAVE_FLOOR")
    if resources.device.startswith("cuda"):
        if resources.host_available_bytes is None:
            reasons.append("HOST_STAGING_MEMORY_UNOBSERVED")
        elif resources.host_available_bytes < host_staging_bytes:
            reasons.append("HOST_MEMORY_BELOW_CHECKPOINT_STAGING_FLOOR")
    if resources.checkpoint_volume_free_bytes < unpruned_bytes:
        reasons.append("DISK_BELOW_UNPRUNED_CHECKPOINT_FLOOR")
    return {
        "evidence_class": "OBSERVED_CAPACITY_WITH_ARITHMETIC_FLOORS",
        "parameter_count": parameter_count,
        "remaining_updates": remaining_updates,
        "resources": asdict(resources),
        "fp32_adamw_device_bytes": training_bytes,
        "native_checkpoint_save_device_bytes": save_device_bytes,
        "checkpoint_payload_floor_bytes": checkpoint_bytes,
        "unpruned_checkpoint_volume_floor_bytes": unpruned_bytes,
        "denials": reasons,
    }


def process_peak_working_set_bytes() -> int | None:
    """Read the current process lifetime peak from the native OS counter."""
    if os.name == "nt":

        class Counters(ctypes.Structure):
            _fields_ = [
                ("size", ctypes.c_uint32),
                ("page_faults", ctypes.c_uint32),
                ("peak_working_set", ctypes.c_size_t),
                ("working_set", ctypes.c_size_t),
                ("peak_paged_pool", ctypes.c_size_t),
                ("paged_pool", ctypes.c_size_t),
                ("peak_nonpaged_pool", ctypes.c_size_t),
                ("nonpaged_pool", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t),
                ("peak_pagefile", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.size = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        process = kernel.GetCurrentProcess()
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Counters),
            ctypes.c_uint32,
        ]
        if psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.size):
            return int(counters.peak_working_set)
        return None
    try:
        import resource

        maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(maximum if sys.platform == "darwin" else maximum * 1024)
    except (ImportError, AttributeError, OSError):
        return None


def benchmark_optimizer_boundary(
    trainer: Trainer,
    fresh_trainer: Callable[[], Trainer],
    identity: CheckpointIdentity,
    batches: Iterable[Batch],
    checkpoint: Path,
    *,
    projection_tokens: int,
) -> dict[str, Any]:
    """Measure one bounded optimizer step and native checkpoint/restart."""
    if type(projection_tokens) is not int or projection_tokens < 1:
        raise ValueError("projection tokens must be a positive integer")
    if trainer.optimizer_step or trainer.micro_step or identity.step or identity.tokens_seen:
        raise ValueError("boundary benchmark requires fresh trainer and checkpoint identity")
    if (
        not isinstance(trainer.model, TwelveSixDecoder)
        or hash_json(trainer.model.spec.to_dict()) != hash_json(identity.model_spec)
        or count_trainable_parameters(trainer.model) != identity.parameter_count
    ):
        raise ValueError("boundary benchmark model differs from the declared ModelSpec")
    if checkpoint.exists() or checkpoint.is_symlink():
        raise FileExistsError(f"benchmark checkpoint already exists: {checkpoint}")
    if not checkpoint.parent.is_dir():
        raise FileNotFoundError(f"benchmark checkpoint parent is unavailable: {checkpoint.parent}")
    accumulation = trainer.config.gradient_accumulation_steps
    window = tuple(islice(batches, accumulation))
    if len(window) != accumulation:
        raise ValueError("benchmark requires one complete gradient accumulation group")
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    start = time.perf_counter_ns()
    metrics = None
    for batch in window:
        metrics = trainer.train_microbatch(batch)
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    step_ns = time.perf_counter_ns() - start
    if metrics is None or not metrics.optimizer_stepped or trainer.optimizer_step != 1:
        raise RuntimeError("boundary benchmark did not complete exactly one optimizer step")
    progress = replace(identity, step=1, tokens_seen=trainer.tokens_seen)
    start = time.perf_counter_ns()
    saved = save_trainer_checkpoint(
        checkpoint, model=trainer.model, trainer=trainer, identity=progress
    )
    write_ns = time.perf_counter_ns() - start
    target = fresh_trainer()
    if (
        target is trainer
        or target.model is trainer.model
        or target.optimizer_step
        or target.micro_step
    ):
        raise ValueError("benchmark restart requires a distinct fresh trainer")
    start = time.perf_counter_ns()
    load_trainer_checkpoint(
        checkpoint,
        model=target.model,
        trainer=target,
        expected_checkpoint_id=saved["checkpoint_id"],
        expected_run_manifest_hash=identity.run_manifest_hash,
        expected_step=1,
        expected_tokens_seen=trainer.tokens_seen,
    )
    read_ns = time.perf_counter_ns() - start
    if target.optimizer_step != 1 or target.tokens_seen != trainer.tokens_seen:
        raise RuntimeError("benchmark restart did not restore the exact boundary")
    checkpoint_bytes = sum(record["bytes"] for record in saved["files"].values())
    tokens_per_second = trainer.tokens_seen * 1_000_000_000 / max(step_ns, 1)
    return {
        "evidence_class": "MEASURED_SINGLE_OPTIMIZER_BOUNDARY",
        "run_manifest_sha256": identity.run_manifest_hash,
        "model_spec_sha256": hash_json(identity.model_spec),
        "trainer_config_sha256": hash_json(asdict(trainer.config)),
        "parameter_count": identity.parameter_count,
        "optimizer_step_ns": step_ns,
        "tokens_seen": trainer.tokens_seen,
        "tokens_per_second": tokens_per_second,
        "same_fixture_projection_tokens": projection_tokens,
        "same_fixture_projected_seconds": projection_tokens / tokens_per_second,
        "checkpoint_id": saved["checkpoint_id"],
        "checkpoint_payload_bytes": checkpoint_bytes,
        "checkpoint_write_ns": write_ns,
        "checkpoint_restore_ns": read_ns,
        "checkpoint_write_payload_bytes_per_second": checkpoint_bytes
        * 1_000_000_000
        / max(write_ns, 1),
        "checkpoint_restore_payload_bytes_per_second": checkpoint_bytes
        * 1_000_000_000
        / max(read_ns, 1),
        "process_peak_working_set_bytes": process_peak_working_set_bytes(),
        "ssd_wear": "UNOBSERVED",
        "sustained_cpu_throttling": "UNOBSERVED",
    }
