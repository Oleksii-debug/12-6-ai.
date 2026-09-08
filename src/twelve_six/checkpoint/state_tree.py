"""Portable serialization for nested trainer/optimizer/RNG state.

Tensor leaves are separated from the JSON structure and stored in SafeTensors.
The tree codec intentionally rejects arbitrary Python objects instead of falling
back to pickle so checkpoint loading remains data-only.
"""

from __future__ import annotations

import base64
import binascii
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
        return {"__kind__": "numpy_scalar", "dtype": str(value.dtype), "value": value.item()}
    if isinstance(value, np.ndarray) or _is_torch_tensor(value):
        key = f"tensor_{len(tensors):08d}"
        array, backend, device = _tensor_to_numpy(value)
        tensors[key] = array
        return {"__kind__": "tensor", "key": key, "backend": backend, "device": device}
    if isinstance(value, tuple):
        return {"__kind__": "tuple", "items": [_pack(v, tensors, f"{path}[]") for v in value]}
    if isinstance(value, list):
        return {"__kind__": "list", "items": [_pack(v, tensors, f"{path}[]") for v in value]}
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            items.append([_pack(key, tensors, f"{path}.<key>"), _pack(item, tensors, f"{path}.{key!r}")])
        return {"__kind__": "mapping", "items": items}
    raise StateTreeError(f"unsupported state value at {path}: {type(value)!r}")


def pack_state_tree(value: Any) -> PackedStateTree:
    tensors: dict[str, np.ndarray] = {}
    return PackedStateTree(tree=_pack(value, tensors, "$"), tensors=tensors)


def _require_keys(value: dict[str, Any], kind: str, exact: set[str]) -> None:
    actual = set(value)
    if actual != exact:
        raise StateTreeError(
            f"noncanonical {kind} node fields: missing={sorted(exact - actual)}, unknown={sorted(actual - exact)}"
        )


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
            if array.dtype != np.uint16:
                raise StateTreeError("torch_bfloat16 tensor payload must use uint16 storage")
            tensor = tensor.view(torch.bfloat16)
        if device and device != "cpu":
            try:
                tensor = tensor.to(device=device)
            except (RuntimeError, AssertionError):
                tensor = tensor.cpu()
        return tensor
    raise StateTreeError(f"unknown tensor backend {backend!r}")


def _unpack(value: Any, tensors: Mapping[str, np.ndarray], used: set[str], path: str) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if not isinstance(value, dict):
        raise StateTreeError(f"noncanonical raw container at {path}: {type(value)!r}")
    if "__kind__" not in value:
        raise StateTreeError(f"noncanonical mapping node without __kind__ at {path}")
    kind = value["__kind__"]
    if not isinstance(kind, str):
        raise StateTreeError(f"state-tree kind must be a string at {path}")
    if kind == "bytes":
        _require_keys(value, kind, {"__kind__", "base64"})
        encoded = value["base64"]
        if not isinstance(encoded, str):
            raise StateTreeError("bytes base64 payload must be a string")
        try:
            return base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
            raise StateTreeError("bytes base64 payload is invalid") from exc
    if kind == "numpy_scalar":
        _require_keys(value, kind, {"__kind__", "dtype", "value"})
        dtype_name = value["dtype"]
        if not isinstance(dtype_name, str):
            raise StateTreeError("numpy scalar dtype must be a string")
        try:
            dtype = np.dtype(dtype_name)
        except TypeError as exc:
            raise StateTreeError(f"invalid numpy scalar dtype {dtype_name!r}") from exc
        if dtype.hasobject:
            raise StateTreeError("object numpy scalar dtype is not permitted")
        try:
            return np.asarray(value["value"], dtype=dtype)[()]
        except (TypeError, ValueError, OverflowError) as exc:
            raise StateTreeError("numpy scalar payload is incompatible with its dtype") from exc
    if kind == "tensor":
        _require_keys(value, kind, {"__kind__", "key", "backend", "device"})
        key, backend, device = value["key"], value["backend"], value["device"]
        if not isinstance(key, str) or not key:
            raise StateTreeError("tensor key must be a non-empty string")
        if not isinstance(backend, str):
            raise StateTreeError("tensor backend must be a string")
        if device is not None and not isinstance(device, str):
            raise StateTreeError("tensor device must be a string or None")
        if key not in tensors:
            raise StateTreeError(f"missing tensor payload {key!r}")
        if key in used:
            raise StateTreeError(f"tensor payload {key!r} is referenced more than once")
        used.add(key)
        return _restore_tensor(tensors[key], backend, device)
    if kind in {"tuple", "list"}:
        _require_keys(value, kind, {"__kind__", "items"})
        items = value["items"]
        if not isinstance(items, list):
            raise StateTreeError(f"{kind} items must be a list")
        restored = [_unpack(item, tensors, used, f"{path}[{index}]") for index, item in enumerate(items)]
        return tuple(restored) if kind == "tuple" else restored
    if kind == "mapping":
        _require_keys(value, kind, {"__kind__", "items"})
        items = value["items"]
        if not isinstance(items, list):
            raise StateTreeError("mapping items must be a list")
        restored: dict[Any, Any] = {}
        for index, pair in enumerate(items):
            if not isinstance(pair, list) or len(pair) != 2:
                raise StateTreeError(f"mapping item {index} must be an exact key/value pair")
            key = _unpack(pair[0], tensors, used, f"{path}.<key:{index}>")
            item = _unpack(pair[1], tensors, used, f"{path}[{index}]")
            try:
                if key in restored:
                    raise StateTreeError(f"duplicate mapping key at {path}: {key!r}")
                restored[key] = item
            except TypeError as exc:
                raise StateTreeError(f"unhashable mapping key at {path}: {type(key)!r}") from exc
        return restored
    raise StateTreeError(f"unknown state-tree kind {kind!r}")


def unpack_state_tree(tree: Any, tensors: Mapping[str, np.ndarray]) -> Any:
    """Reconstruct a nested state object from its canonical JSON/tensor encoding."""
    if not isinstance(tensors, Mapping):
        raise StateTreeError("tensor payload collection must be a mapping")
    if any(not isinstance(key, str) or not key for key in tensors):
        raise StateTreeError("tensor payload keys must be non-empty strings")
    used: set[str] = set()
    restored = _unpack(tree, tensors, used, "$")
    unreferenced = sorted(set(tensors) - used)
    if unreferenced:
        raise StateTreeError(f"unreferenced tensor payloads: {unreferenced}")
    return restored
