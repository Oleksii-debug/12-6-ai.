"""Fail-closed D05 checkpoint compatibility facade.

``_core_v1`` owns checkpoint-v1 serialization, integrity, immutable snapshots,
and the closed-world manifest contract. This facade adds target compatibility
checks that must complete before any live model, optimizer, or scheduler state
is mutated.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import _core_v1 as _impl

_original_load_verified_checkpoint = _impl.load_verified_checkpoint


def _is_torch_tensor(value: Any) -> bool:
    cls = value.__class__
    return cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}


def _strict_materialize_for_target(array: np.ndarray, target: Any) -> Any:
    """Materialize one model tensor without implicit checkpoint dtype conversion."""

    if isinstance(target, np.ndarray):
        if tuple(target.shape) != tuple(array.shape):
            raise _impl.CheckpointCompatibilityError(
                f"shape mismatch: checkpoint {tuple(array.shape)} vs "
                f"target {tuple(target.shape)}"
            )
        if np.dtype(array.dtype) != np.dtype(target.dtype):
            raise _impl.CheckpointCompatibilityError(
                f"dtype mismatch: checkpoint {array.dtype} vs target {target.dtype}"
            )
        return array.copy()

    if _is_torch_tensor(target):
        torch = importlib.import_module("torch")
        if str(target.dtype) == "torch.bfloat16":
            if np.dtype(array.dtype) != np.dtype(np.uint16):
                raise _impl.CheckpointCompatibilityError(
                    "dtype mismatch: BF16 targets require checkpoint uint16 bit storage; "
                    f"got {array.dtype}"
                )
            tensor = torch.from_numpy(array.copy()).view(torch.bfloat16)
        else:
            try:
                tensor = torch.from_numpy(array.copy())
            except (TypeError, ValueError, RuntimeError) as exc:
                raise _impl.CheckpointCompatibilityError(
                    f"checkpoint dtype {array.dtype} cannot materialize target {target.dtype}"
                ) from exc
            if tensor.dtype != target.dtype:
                raise _impl.CheckpointCompatibilityError(
                    f"dtype mismatch: checkpoint {tensor.dtype} vs target {target.dtype}"
                )
        if tuple(target.shape) != tuple(tensor.shape):
            raise _impl.CheckpointCompatibilityError(
                f"shape mismatch: checkpoint {tuple(tensor.shape)} vs "
                f"target {tuple(target.shape)}"
            )
        return tensor.to(device=target.device)

    raise _impl.CheckpointCompatibilityError(
        f"unsupported target tensor type {type(target)!r}"
    )


def _state_tensor_dtype_compatible(value: Any, parameter: Any) -> bool:
    """Return whether a tensor-valued optimizer slot is safe for the parameter."""

    torch = importlib.import_module("torch")
    if not value.dtype.is_floating_point and not value.dtype.is_complex:
        return value.dtype == parameter.dtype
    if parameter.dtype in {torch.float16, torch.bfloat16}:
        return value.dtype in {parameter.dtype, torch.float32}
    return value.dtype == parameter.dtype


def _validate_torch_optimizer_slot(value: Any, parameter: Any, *, path: str) -> None:
    if _is_torch_tensor(value):
        leaf_name = path.rsplit(".", 1)[-1]
        if leaf_name == "step":
            if value.ndim != 0:
                raise _impl.CheckpointCompatibilityError(
                    f"{path} must be a scalar tensor"
                )
            return
        if tuple(value.shape) != tuple(parameter.shape):
            raise _impl.CheckpointCompatibilityError(
                f"{path} shape {tuple(value.shape)} does not match parameter "
                f"shape {tuple(parameter.shape)}"
            )
        if not _state_tensor_dtype_compatible(value, parameter):
            raise _impl.CheckpointCompatibilityError(
                f"{path} dtype {value.dtype} is incompatible with parameter dtype "
                f"{parameter.dtype}"
            )
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_torch_optimizer_slot(item, parameter, path=f"{path}.{key}")
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_torch_optimizer_slot(item, parameter, path=f"{path}[{index}]")


def _preflight_torch_optimizer_state(
    optimizer: Any,
    checkpoint_state: Mapping[str, Any],
) -> None:
    state = checkpoint_state.get("state")
    groups = checkpoint_state.get("param_groups")
    if not isinstance(state, Mapping) or not isinstance(groups, list):
        raise _impl.CheckpointCompatibilityError(
            "checkpoint torch optimizer state_dict is structurally invalid"
        )

    live_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(live_groups, list) or len(groups) != len(live_groups):
        raise _impl.CheckpointCompatibilityError(
            "checkpoint optimizer param-group count differs from target"
        )

    target_serialized = optimizer.state_dict()
    target_groups = target_serialized.get("param_groups")
    if not isinstance(target_groups, list) or len(target_groups) != len(groups):
        raise _impl.CheckpointCompatibilityError(
            "target optimizer state_dict has invalid param_groups"
        )

    checkpoint_id_to_parameter: dict[Any, Any] = {}
    group_triples = zip(groups, live_groups, target_groups, strict=True)
    for group_index, (saved_group, live_group, target_group) in enumerate(group_triples):
        if not isinstance(saved_group, Mapping) or not isinstance(live_group, Mapping):
            raise _impl.CheckpointCompatibilityError(
                f"optimizer param_groups[{group_index}] must be mappings"
            )
        if not isinstance(target_group, Mapping):
            raise _impl.CheckpointCompatibilityError(
                f"target optimizer param_groups[{group_index}] must be a mapping"
            )

        saved_ids = saved_group.get("params")
        live_params = live_group.get("params")
        target_ids = target_group.get("params")
        if not isinstance(saved_ids, list) or not isinstance(live_params, list):
            raise _impl.CheckpointCompatibilityError(
                f"optimizer param_groups[{group_index}].params must be lists"
            )
        if not isinstance(target_ids, list):
            raise _impl.CheckpointCompatibilityError(
                f"target param_groups[{group_index}].params must be a list"
            )
        if len(saved_ids) != len(live_params) or len(saved_ids) != len(target_ids):
            raise _impl.CheckpointCompatibilityError(
                f"optimizer param_groups[{group_index}] parameter count differs from target"
            )

        saved_keys = set(saved_group)
        target_keys = set(target_group)
        if saved_keys != target_keys:
            raise _impl.CheckpointCompatibilityError(
                f"optimizer param_groups[{group_index}] keys differ: "
                f"checkpoint={sorted(saved_keys)}, target={sorted(target_keys)}"
            )

        for saved_id, parameter in zip(saved_ids, live_params, strict=True):
            if saved_id in checkpoint_id_to_parameter:
                raise _impl.CheckpointCompatibilityError(
                    f"optimizer checkpoint parameter id {saved_id!r} is duplicated"
                )
            if not _is_torch_tensor(parameter):
                raise _impl.CheckpointCompatibilityError(
                    "torch optimizer param_groups contain a non-tensor parameter"
                )
            checkpoint_id_to_parameter[saved_id] = parameter

    for parameter_id, slot in state.items():
        if parameter_id not in checkpoint_id_to_parameter:
            raise _impl.CheckpointCompatibilityError(
                f"optimizer state refers to unknown parameter id {parameter_id!r}"
            )
        if not isinstance(slot, Mapping):
            raise _impl.CheckpointCompatibilityError(
                f"optimizer state[{parameter_id!r}] must be a mapping"
            )
        parameter = checkpoint_id_to_parameter[parameter_id]
        for key, value in slot.items():
            _validate_torch_optimizer_slot(
                value,
                parameter,
                path=f"optimizer.state[{parameter_id!r}].{key}",
            )


def _validate_generic_state_shape(checkpoint: Any, current: Any, *, path: str) -> None:
    if isinstance(checkpoint, np.ndarray) and isinstance(current, np.ndarray):
        if checkpoint.shape != current.shape:
            raise _impl.CheckpointCompatibilityError(
                f"{path} shape {checkpoint.shape} does not match target {current.shape}"
            )
        if checkpoint.dtype != current.dtype:
            raise _impl.CheckpointCompatibilityError(
                f"{path} dtype {checkpoint.dtype} does not match target {current.dtype}"
            )
        return

    if _is_torch_tensor(checkpoint) and _is_torch_tensor(current):
        if checkpoint.shape != current.shape or checkpoint.dtype != current.dtype:
            raise _impl.CheckpointCompatibilityError(
                f"{path} tensor metadata does not match target state"
            )
        return

    if isinstance(checkpoint, Mapping) and isinstance(current, Mapping):
        if set(checkpoint) != set(current):
            raise _impl.CheckpointCompatibilityError(
                f"{path} mapping keys differ from target state"
            )
        for key in checkpoint:
            _validate_generic_state_shape(
                checkpoint[key],
                current[key],
                path=f"{path}.{key}",
            )
        return

    if isinstance(checkpoint, (list, tuple)) and isinstance(current, (list, tuple)):
        if len(checkpoint) != len(current):
            raise _impl.CheckpointCompatibilityError(
                f"{path} sequence length differs from target state"
            )
        for index, (saved, target) in enumerate(zip(checkpoint, current, strict=True)):
            _validate_generic_state_shape(saved, target, path=f"{path}[{index}]")


def _preflight_optimizer_state(optimizer: Any, checkpoint_state: Any) -> None:
    if not isinstance(checkpoint_state, Mapping):
        raise _impl.CheckpointCompatibilityError(
            "checkpoint optimizer state must be a mapping"
        )

    live_groups = getattr(optimizer, "param_groups", None)
    looks_like_torch_optimizer = isinstance(live_groups, list) and all(
        isinstance(group, Mapping)
        and all(_is_torch_tensor(param) for param in group.get("params", []))
        for group in live_groups
    )
    if looks_like_torch_optimizer:
        _preflight_torch_optimizer_state(optimizer, checkpoint_state)
        return

    current = optimizer.state_dict()
    if not isinstance(current, Mapping):
        raise _impl.CheckpointCompatibilityError(
            "target optimizer state_dict must be a mapping"
        )
    _validate_generic_state_shape(checkpoint_state, current, path="optimizer")


def _preflight_scheduler_state(scheduler: Any, checkpoint_state: Any) -> None:
    current = scheduler.state_dict()
    if not isinstance(checkpoint_state, Mapping) or not isinstance(current, Mapping):
        raise _impl.CheckpointCompatibilityError(
            "checkpoint and target scheduler state must be mappings"
        )
    _validate_generic_state_shape(checkpoint_state, current, path="scheduler")


def _hardened_load_verified_checkpoint(
    verified: Any,
    *,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    strict_model: bool = True,
    restore_rng: bool = True,
    expected_git_sha: str | None = None,
    expected_model_spec_hash: str | None = None,
    expected_tokenizer_hash: str | None = None,
    expected_tokenizer_vocab_hash: str | None = None,
    expected_dataset_manifest_hash: str | None = None,
    expected_run_manifest_hash: str | None = None,
) -> Any:
    """Preflight mutable optimizer/scheduler targets before first live mutation."""

    _, combined_state = _impl._decode_verified_state(verified)
    if optimizer is not None:
        optimizer_state = combined_state.get("optimizer")
        if optimizer_state is None:
            raise _impl.CheckpointCompatibilityError(
                "optimizer was requested but checkpoint has no optimizer state"
            )
        _preflight_optimizer_state(optimizer, optimizer_state)

    if scheduler is not None:
        scheduler_state = combined_state.get("scheduler")
        if scheduler_state is None:
            raise _impl.CheckpointCompatibilityError(
                "scheduler was requested but checkpoint has no scheduler state"
            )
        _preflight_scheduler_state(scheduler, scheduler_state)

    return _original_load_verified_checkpoint(
        verified,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        strict_model=strict_model,
        restore_rng=restore_rng,
        expected_git_sha=expected_git_sha,
        expected_model_spec_hash=expected_model_spec_hash,
        expected_tokenizer_hash=expected_tokenizer_hash,
        expected_tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
        expected_dataset_manifest_hash=expected_dataset_manifest_hash,
        expected_run_manifest_hash=expected_run_manifest_hash,
    )


# Functions defined in ``_core_v1`` resolve these module globals at call time.
# Patching them here therefore hardens both direct snapshot loads and the
# existing load_checkpoint() path without duplicating serialization logic.
_impl._materialize_for_target = _strict_materialize_for_target
_impl.load_verified_checkpoint = _hardened_load_verified_checkpoint

# Preserve ``twelve_six.checkpoint.core`` as the stable import surface.
for _export_name in dir(_impl):
    if not _export_name.startswith("__"):
        globals()[_export_name] = getattr(_impl, _export_name)
