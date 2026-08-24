"""Small runtime measurements for validating planning algebra without claiming GPU capacity."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TensorByteMeasurement:
    tensor_count: int
    element_count: int
    total_bytes: int


def measure_tensor_bytes(tensors: Iterable[Any]) -> TensorByteMeasurement:
    """Measure unique tensor objects by numel/element_size; works with torch-like tensors."""

    seen: set[int] = set()
    tensor_count = 0
    element_count = 0
    total_bytes = 0
    for tensor in tensors:
        identity = id(tensor)
        if identity in seen:
            continue
        seen.add(identity)
        if not hasattr(tensor, "numel") or not hasattr(tensor, "element_size"):
            raise TypeError("measurement inputs must provide numel() and element_size()")
        elements = int(tensor.numel())
        element_size = int(tensor.element_size())
        if elements < 0 or element_size < 0:
            raise ValueError("tensor byte metadata must be non-negative")
        tensor_count += 1
        element_count += elements
        total_bytes += elements * element_size
    return TensorByteMeasurement(
        tensor_count=tensor_count,
        element_count=element_count,
        total_bytes=total_bytes,
    )


def measure_model_parameter_bytes(model: Any) -> TensorByteMeasurement:
    if not hasattr(model, "parameters"):
        raise TypeError("model must provide parameters()")
    return measure_tensor_bytes(model.parameters())
