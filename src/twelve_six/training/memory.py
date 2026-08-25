"""Runtime tensor-state accounting for the live training stack.

This module intentionally separates exact tensor accounting from process RSS. Tensor bytes are
portable properties of the current model/optimizer state; RSS includes Python, allocator,
kernels, shared libraries, and retained arenas and must be interpreted as process telemetry.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass(frozen=True, slots=True)
class TrainingTensorMemory:
    parameter_bytes: int
    gradient_bytes: int
    adam_moment_bytes: int
    optimizer_other_tensor_bytes: int
    scaler_tensor_bytes: int

    @property
    def total_tensor_bytes(self) -> int:
        return (
            self.parameter_bytes
            + self.gradient_bytes
            + self.adam_moment_bytes
            + self.optimizer_other_tensor_bytes
            + self.scaler_tensor_bytes
        )


def tensor_nbytes(tensor: Tensor) -> int:
    """Return logical storage bytes for a tensor value."""
    return tensor.numel() * tensor.element_size()


def _unique_parameters(model: nn.Module):
    seen: set[int] = set()
    for parameter in model.parameters():
        marker = id(parameter)
        if marker in seen:
            continue
        seen.add(marker)
        yield parameter


def parameter_tensor_bytes(model: nn.Module) -> int:
    return sum(tensor_nbytes(parameter) for parameter in _unique_parameters(model))


def gradient_tensor_bytes(model: nn.Module) -> int:
    return sum(
        tensor_nbytes(parameter.grad)
        for parameter in _unique_parameters(model)
        if parameter.grad is not None
    )


def _tree_tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    if isinstance(value, Tensor):
        marker = id(value)
        if marker in seen:
            return 0
        seen.add(marker)
        return tensor_nbytes(value)
    if isinstance(value, dict):
        return sum(_tree_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tree_tensor_bytes(item, seen) for item in value)
    return 0


def optimizer_tensor_bytes(optimizer: Optimizer) -> tuple[int, int]:
    """Return ``(Adam moments, other optimizer tensors)`` in bytes.

    AdamW's ``exp_avg`` and ``exp_avg_sq`` are reported separately because they are the persistent
    O(parameters) optimizer state. Per-parameter scalar ``step`` tensors and any future auxiliary
    tensors remain visible as ``optimizer_other_tensor_bytes`` rather than being silently lost.
    """
    moments = 0
    other = 0
    seen: set[int] = set()
    for state in optimizer.state.values():
        for name, value in state.items():
            if not isinstance(value, Tensor):
                continue
            marker = id(value)
            if marker in seen:
                continue
            seen.add(marker)
            size = tensor_nbytes(value)
            if name in {"exp_avg", "exp_avg_sq"}:
                moments += size
            else:
                other += size
    return moments, other


def scaler_tensor_bytes(scaler: Any | None) -> int:
    if scaler is None:
        return 0
    return _tree_tensor_bytes(scaler.state_dict())


def measure_training_tensor_memory(
    model: nn.Module,
    optimizer: Optimizer,
    scaler: Any | None = None,
) -> TrainingTensorMemory:
    moments, optimizer_other = optimizer_tensor_bytes(optimizer)
    return TrainingTensorMemory(
        parameter_bytes=parameter_tensor_bytes(model),
        gradient_bytes=gradient_tensor_bytes(model),
        adam_moment_bytes=moments,
        optimizer_other_tensor_bytes=optimizer_other,
        scaler_tensor_bytes=scaler_tensor_bytes(scaler),
    )


def process_rss_bytes() -> int:
    """Return current process RSS on Linux without adding a runtime dependency."""
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
    except (OSError, IndexError, ValueError) as exc:
        raise RuntimeError("current RSS telemetry requires Linux /proc/self/statm") from exc
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def scaler_state_metadata(scaler: Any | None) -> dict[str, Any]:
    if scaler is None:
        return {"present": False, "enabled": False, "state_keys": [], "tensor_bytes": 0}
    state = scaler.state_dict()
    enabled = bool(scaler.is_enabled()) if hasattr(scaler, "is_enabled") else bool(state)
    return {
        "present": True,
        "enabled": enabled,
        "state_keys": sorted(str(key) for key in state),
        "tensor_bytes": _tree_tensor_bytes(state),
    }
