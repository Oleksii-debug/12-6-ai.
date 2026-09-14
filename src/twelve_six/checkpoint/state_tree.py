"""Portable serialization for nested trainer/optimizer/RNG state.

Tensor leaves are separated from the JSON structure and stored in SafeTensors.
The tree codec intentionally rejects arbitrary Python objects instead of falling
back to pickle so checkpoint loading remains data-only.
"""

from __future__ import annotations

import base64
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


class StateTreeError(TypeError):
    """Raised when a state tree contains a value that cannot be serialized safely."""


@dataclass(frozen=True)
class PackedStateTree:
    """JSON-compatible structure plus NumPy tensor payloads."""

    tree: Any
    tensors: dict[str, np.ndarray]


def _is_torch_tensor(value: Any) -> bool:
    cls = value.__class__
    return cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}


def _tensor_to_numpy(value: Any) -> tuple[np.ndarray, str, str | None]:
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value), "numpy", None
    if _is_torch_tensor(value):
        detached = value.detach().cpu().contiguous()
        # bfloat16 cannot be converted directly to NumPy. Store raw uint16 bits.
        if str(detached.dtype) == "torch.bfloat16":
            array = detached.view(importlib.import_module("torch").uint16).numpy().copy()
            return array, "torch_bfloat16", str(value.device)
        return detached.numpy().copy(), "torch", str(value.device)
    raise StateTreeError(f"unsupported tensor type: {type(value)!r}")


def _pack(value: Any, tensors: dict[str, np.ndarray], path: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"__kind__": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, np.generic):
        return {
            "__kind__": "numpy_scalar",
            "dtype": str(value.dtype),
            "value": value.item(),
        }
    if isinstance(value, np.ndarray) or _is_torch_tensor(value):
        key = f"tensor_{len(tensors):08d}"
        array, backend, device = _tensor_to_numpy(value)
        tensors[key] = array
        return {
            "__kind__": "tensor",
            "key": key,
            "backend": backend,
            "device": device,
        }
    if isinstance(value, tuple):
        return {"__kind__": "tuple", "items": [_pack(v, tensors, f"{path}[]") for v in value]}
    if isinstance(value, list):
        return {"__kind__": "list", "items": [_pack(v, tensors, f"{path}[]") for v in value]}
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            packed_key = _pack(key, tensors, f"{path}.<key>")
            packed_value = _pack(item, tensors, f"{path}.{key!r}")
            items.append([packed_key, packed_value])
        return {"__kind__": "mapping", "items": items}
    raise StateTreeError(f"unsupported state value at {path}: {type(value)!r}")


def pack_state_tree(value: Any) -> PackedStateTree:
    """Split a nested state object into JSON-safe structure and tensor leaves."""

    tensors: dict[str, np.ndarray] = {}
    return PackedStateTree(tree=_pack(value, tensors, "$"), tensors=tensors)


def _restore_tensor(array: np.ndarray, backend: str, device: str | None) -> Any:
    if backend == "numpy":
        return array.copy()
    if backend in {"torch", "torch_bfloat16"}:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise StateTreeError("torch is required to restore torch tensor state") from exc
        tensor = torch.from_numpy(array.copy())
        if backend == "torch_bfloat16":
            tensor = tensor.view(torch.bfloat16)
        if device and device != "cpu":
            try:
                tensor = tensor.to(device=device)
            except (RuntimeError, AssertionError):
                # Checkpoints remain portable when the recorded accelerator is unavailable.
                tensor = tensor.cpu()
        return tensor
    raise StateTreeError(f"unknown tensor backend {backend!r}")


def _unpack(value: Any, tensors: Mapping[str, np.ndarray]) -> Any:
    if not isinstance(value, dict) or "__kind__" not in value:
        return value
    kind = value["__kind__"]
    if kind == "bytes":
        return base64.b64decode(value["base64"].encode("ascii"))
    if kind == "numpy_scalar":
        return np.asarray(value["value"], dtype=value["dtype"])[()]
    if kind == "tensor":
        key = value["key"]
        if key not in tensors:
            raise StateTreeError(f"missing tensor payload {key!r}")
        return _restore_tensor(tensors[key], value["backend"], value.get("device"))
    if kind == "tuple":
        return tuple(_unpack(item, tensors) for item in value["items"])
    if kind == "list":
        return [_unpack(item, tensors) for item in value["items"]]
    if kind == "mapping":
        return {_unpack(key, tensors): _unpack(item, tensors) for key, item in value["items"]}
    raise StateTreeError(f"unknown state-tree kind {kind!r}")


def unpack_state_tree(tree: Any, tensors: Mapping[str, np.ndarray]) -> Any:
    """Reconstruct a nested state object from its JSON structure and tensor leaves."""

    return _unpack(tree, tensors)
