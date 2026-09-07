"""Single-accelerator execution helpers layered on the canonical D02 Trainer.

Precision mode semantics stay owned by ``training.precision``. This module owns
one-device selection, host/device transfer, device-bound measurements, seed setup
before scratch model construction, and fail-closed CUDA OOM handling.
"""

from __future__ import annotations

import os
import random
import resource
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .trainer import StepMetrics, Trainer

Batch = Mapping[str, Tensor]


class SingleDeviceOOMError(RuntimeError):
    """Raised after an OOM invalidates the current in-memory training attempt."""


@dataclass(frozen=True, slots=True)
class DeviceRuntime:
    requested: str
    resolved: str
    device_type: str
    visible_cuda_devices: int
    cuda_name: str | None
    cuda_capability: tuple[int, int] | None
    total_memory_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BatchTransferMetrics:
    requested_non_blocking: bool
    effective_non_blocking: bool
    all_cpu_sources_pinned: bool
    bytes_moved: int


@dataclass(frozen=True, slots=True)
class SingleDeviceStepMetrics:
    trainer: StepMetrics
    transfer: BatchTransferMetrics
    transfer_seconds: float
    train_seconds: float
    total_seconds: float
    tokens_per_second: float
    timing_authority: str
    process_rss_bytes: int | None
    cuda_memory_allocated_bytes: int | None
    cuda_memory_reserved_bytes: int | None
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None


def resolve_single_device(
    requested: str = "auto",
    *,
    allow_cpu_fallback: bool = True,
    require_single_visible_cuda: bool = True,
) -> tuple[torch.device, DeviceRuntime]:
    """Resolve CPU or exactly one visible CUDA accelerator before model creation."""

    if not isinstance(requested, str) or not requested.strip():
        raise TypeError("requested device must be a non-empty string")
    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            normalized = "cuda:0"
        elif allow_cpu_fallback:
            normalized = "cpu"
        else:
            raise RuntimeError("CUDA was required by policy but no CUDA device is available")

    device = torch.device(normalized)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("single-device training currently supports only cpu or cuda")

    visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if device.type == "cpu":
        return device, DeviceRuntime(
            requested=requested,
            resolved="cpu",
            device_type="cpu",
            visible_cuda_devices=visible,
            cuda_name=None,
            cuda_capability=None,
            total_memory_bytes=None,
        )

    if not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    index = 0 if device.index is None else device.index
    if index < 0 or index >= visible:
        raise ValueError(f"requested CUDA index {index} is outside visible device count {visible}")
    if require_single_visible_cuda and visible != 1:
        raise RuntimeError(
            "single-GPU checkpoint/RNG semantics require exactly one visible CUDA device; "
            "launch with CUDA_VISIBLE_DEVICES=<one-device-index>"
        )

    resolved = torch.device("cuda", index)
    properties = torch.cuda.get_device_properties(resolved)
    capability = torch.cuda.get_device_capability(resolved)
    return resolved, DeviceRuntime(
        requested=requested,
        resolved=str(resolved),
        device_type="cuda",
        visible_cuda_devices=visible,
        cuda_name=str(properties.name),
        cuda_capability=(int(capability[0]), int(capability[1])),
        total_memory_bytes=int(properties.total_memory),
    )


def seed_before_model_init(seed: int, device: torch.device) -> None:
    """Seed project RNG streams before caller constructs a random-init Base model."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cannot seed requested CUDA device because CUDA is unavailable")
        torch.cuda.manual_seed_all(seed)


def _tensor_bytes(tensor: Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def move_batch_to_device(
    batch: Batch,
    device: torch.device,
    *,
    non_blocking: bool = True,
) -> tuple[dict[str, Tensor], BatchTransferMetrics]:
    """Move one flat Trainer batch and report whether H2D can be asynchronous."""

    if not isinstance(batch, Mapping) or not batch:
        raise ValueError("batch must be a non-empty tensor mapping")
    cpu_sources: list[Tensor] = []
    bytes_moved = 0
    for name, tensor in batch.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise TypeError("single-device batches must map string keys to torch.Tensor values")
        if tensor.device != device:
            bytes_moved += _tensor_bytes(tensor)
        if tensor.device.type == "cpu":
            cpu_sources.append(tensor)

    all_pinned = bool(cpu_sources) and all(tensor.is_pinned() for tensor in cpu_sources)
    effective_non_blocking = bool(
        non_blocking and device.type == "cuda" and cpu_sources and all_pinned
    )
    moved = {
        name: tensor.to(device, non_blocking=effective_non_blocking)
        for name, tensor in batch.items()
    }
    return moved, BatchTransferMetrics(
        requested_non_blocking=bool(non_blocking),
        effective_non_blocking=effective_non_blocking,
        all_cpu_sources_pinned=all_pinned,
        bytes_moved=bytes_moved,
    )


def build_synthetic_lm_batch(
    *,
    vocab_size: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
    pin_memory: bool = False,
) -> dict[str, Tensor]:
    """Build a deterministic mechanics fixture; never treat it as stage data evidence."""

    for name, value, minimum in (
        ("vocab_size", vocab_size, 2),
        ("batch_size", batch_size, 1),
        ("sequence_length", sequence_length, 2),
        ("seed", seed, 0),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    labels = input_ids.clone()
    if pin_memory:
        input_ids = input_ids.pin_memory()
        labels = labels.pin_memory()
    return {"input_ids": input_ids, "labels": labels}


def _process_rss_bytes() -> int | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, ValueError):
        return None
    multiplier = 1 if sys.platform == "darwin" else 1024
    return int(usage * multiplier)


def model_storage_dtypes(model: nn.Module) -> list[str]:
    return sorted({str(parameter.dtype) for parameter in model.parameters()})


def optimizer_state_dtypes(trainer: Trainer) -> list[str]:
    dtypes: set[str] = set()
    for state in trainer.optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor):
                dtypes.add(str(value.dtype))
    return sorted(dtypes)


class SingleDeviceStepRunner:
    """Measure one-device Trainer transitions and poison this runner on CUDA OOM."""

    def __init__(
        self,
        trainer: Trainer,
        *,
        non_blocking_transfer: bool = True,
        synchronize_for_metrics: bool = True,
    ) -> None:
        if trainer.device.type not in {"cpu", "cuda"}:
            raise ValueError("single-device runner supports only cpu or cuda Trainer devices")
        self.trainer = trainer
        self.non_blocking_transfer = bool(non_blocking_transfer)
        self.synchronize_for_metrics = bool(synchronize_for_metrics)
        self.failed = False

    def _synchronize(self) -> None:
        if self.trainer.device.type == "cuda" and self.synchronize_for_metrics:
            torch.cuda.synchronize(self.trainer.device)

    def _poison_after_oom(self, exc: BaseException) -> SingleDeviceOOMError:
        self.failed = True
        self.trainer.optimizer.zero_grad(set_to_none=True)
        if self.trainer.device.type == "cuda":
            torch.cuda.empty_cache()
        return SingleDeviceOOMError(
            "CUDA OOM: do not retry the same in-memory Trainer transition. Restore the "
            "last verified checkpoint into fresh objects and lower microbatch size and/or "
            "sequence length."
        )

    def train_microbatch(self, batch: Batch) -> SingleDeviceStepMetrics:
        if self.failed:
            raise RuntimeError(
                "single-device runner is invalid after OOM; restore a verified checkpoint "
                "into a fresh model/Trainer before continuing"
            )
        if self.trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.trainer.device)

        try:
            self._synchronize()
            total_start = time.perf_counter()
            transfer_start = total_start
            moved, transfer = move_batch_to_device(
                batch,
                self.trainer.device,
                non_blocking=self.non_blocking_transfer,
            )
            self._synchronize()
            transfer_end = time.perf_counter()

            train_start = transfer_end
            metrics = self.trainer.train_microbatch(moved)
            self._synchronize()
            train_end = time.perf_counter()
        except torch.cuda.OutOfMemoryError as exc:
            raise self._poison_after_oom(exc) from exc

        train_seconds = train_end - train_start
        token_rate = metrics.tokens / train_seconds if train_seconds > 0 else float("inf")
        cuda_values: tuple[int | None, int | None, int | None, int | None]
        if self.trainer.device.type == "cuda":
            cuda_values = (
                int(torch.cuda.memory_allocated(self.trainer.device)),
                int(torch.cuda.memory_reserved(self.trainer.device)),
                int(torch.cuda.max_memory_allocated(self.trainer.device)),
                int(torch.cuda.max_memory_reserved(self.trainer.device)),
            )
        else:
            cuda_values = (None, None, None, None)

        return SingleDeviceStepMetrics(
            trainer=metrics,
            transfer=transfer,
            transfer_seconds=transfer_end - transfer_start,
            train_seconds=train_seconds,
            total_seconds=train_end - total_start,
            tokens_per_second=token_rate,
            timing_authority=(
                "DEVICE_SYNCHRONIZED_WALL_TIME"
                if self.trainer.device.type == "cuda" and self.synchronize_for_metrics
                else "HOST_WALL_TIME"
            ),
            process_rss_bytes=_process_rss_bytes(),
            cuda_memory_allocated_bytes=cuda_values[0],
            cuda_memory_reserved_bytes=cuda_values[1],
            cuda_peak_allocated_bytes=cuda_values[2],
            cuda_peak_reserved_bytes=cuda_values[3],
        )


def greedy_inference_after_training(
    model: nn.Module,
    prompt_ids: Tensor,
    *,
    device: torch.device,
    max_new_tokens: int,
) -> Tensor:
    """Run post-training greedy inference in model storage dtype without autocast."""

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not hasattr(model, "generate"):
        raise TypeError("model must provide generate() for the single-GPU pilot")
    model.eval()
    prompt = prompt_ids.to(device)
    with torch.inference_mode():
        generated = model.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return generated.detach().cpu()


def launch_environment() -> dict[str, str | None]:
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
    }
