"""Fail-closed compatibility hardening for checkpoint-v1 loads.

This module closes corruption classes found by NEXT100-075 without changing the
serialized checkpoint-v1 format.  It installs stricter semantic validation and
preflight checks before any live model/optimizer/scheduler mutation occurs.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import core as _core

_ORIGINAL_LOAD_VERIFIED_CHECKPOINT = _core.load_verified_checkpoint
_ORIGINAL_VALIDATE_MANIFEST_IDENTITY = _core._validate_manifest_identity
_INSTALLED = False


def _require_non_negative_int(identity: Mapping[str, Any], field: str) -> int:
    value = identity.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _core.CheckpointIntegrityError(
            f"identity.{field} must be a non-negative integer"
        )
    return value


def _validate_manifest_identity_fail_closed(identity: Any) -> None:
    """Validate derived hashes plus the semantic invariants used at save time."""

    _ORIGINAL_VALIDATE_MANIFEST_IDENTITY(identity)
    if not isinstance(identity, Mapping):
        raise _core.CheckpointIntegrityError("manifest identity must be a mapping")

    parameter_count = identity.get("parameter_count")
    if (
        not isinstance(parameter_count, int)
        or isinstance(parameter_count, bool)
        or parameter_count <= 0
    ):
        raise _core.CheckpointIntegrityError(
            "identity.parameter_count must be a positive integer"
        )

    _require_non_negative_int(identity, "seed")
    _require_non_negative_int(identity, "step")
    _require_non_negative_int(identity, "tokens_seen")

    model_spec = identity.get("model_spec")
    if not isinstance(model_spec, Mapping) or not model_spec:
        raise _core.CheckpointIntegrityError(
            "identity.model_spec must be a non-empty mapping"
        )
    training_config = identity.get("training_config")
    if not isinstance(training_config, Mapping) or not training_config:
        raise _core.CheckpointIntegrityError(
            "identity.training_config must be a non-empty mapping"
        )
    optimizer = identity.get("optimizer")
    if not isinstance(optimizer, Mapping) or not optimizer:
        raise _core.CheckpointIntegrityError(
            "identity.optimizer must be a non-empty mapping"
        )
    scheduler = identity.get("scheduler")
    if scheduler is not None and (
        not isinstance(scheduler, Mapping) or not scheduler
    ):
        raise _core.CheckpointIntegrityError(
            "identity.scheduler must be a non-empty mapping or null"
        )
    precision = identity.get("precision")
    if not isinstance(precision, str) or not precision.strip():
        raise _core.CheckpointIntegrityError(
            "identity.precision must be a non-empty string"
        )


def _materialize_for_target_fail_closed(array: np.ndarray, target: Any) -> Any:
    """Materialize one tensor only when source and target dtypes are exact.

    BF16 is the single representation exception: checkpoint-v1 stores PyTorch
    bfloat16 tensor payload bits as NumPy uint16 because NumPy has no native
    bfloat16 dtype in the supported environment.
    """

    if isinstance(target, np.ndarray):
        if tuple(target.shape) != tuple(array.shape):
            raise _core.CheckpointCompatibilityError(
                f"shape mismatch: checkpoint {tuple(array.shape)} vs target {tuple(target.shape)}"
            )
        if array.dtype != target.dtype:
            raise _core.CheckpointCompatibilityError(
                f"dtype mismatch: checkpoint {array.dtype} vs target {target.dtype}"
            )
        return array.copy()

    cls = target.__class__
    if cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}:
        torch = importlib.import_module("torch")
        if str(target.dtype) == "torch.bfloat16":
            if array.dtype != np.dtype(np.uint16):
                raise _core.CheckpointCompatibilityError(
                    "dtype mismatch: bfloat16 target requires checkpoint uint16 bit storage"
                )
            tensor = torch.from_numpy(array.copy()).view(torch.bfloat16)
        else:
            try:
                tensor = torch.from_numpy(array.copy())
            except (TypeError, ValueError, RuntimeError) as exc:
                raise _core.CheckpointCompatibilityError(
                    f"checkpoint dtype {array.dtype} cannot materialize for target {target.dtype}"
                ) from exc
            if tensor.dtype != target.dtype:
                raise _core.CheckpointCompatibilityError(
                    f"dtype mismatch: checkpoint {tensor.dtype} vs target {target.dtype}"
                )
        if tuple(target.shape) != tuple(tensor.shape):
            raise _core.CheckpointCompatibilityError(
                f"shape mismatch: checkpoint {tuple(tensor.shape)} vs target {tuple(target.shape)}"
            )
        return tensor.to(device=target.device)

    raise _core.CheckpointCompatibilityError(
        f"unsupported target tensor type {type(target)!r}"
    )


def _is_tensor_like(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    cls = value.__class__
    return cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value.shape)


def _dtype_name(value: Any) -> str:
    return str(value.dtype)


def _numel(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.size)
    return int(value.numel())


def _preflight_tree_compatibility(
    current: Any,
    incoming: Any,
    *,
    label: str,
    path: str = "state",
) -> None:
    """Compare state structure and tensor geometry without mutating the target."""

    if _is_tensor_like(current):
        if not _is_tensor_like(incoming):
            raise _core.CheckpointCompatibilityError(
                f"{label} {path} tensor/type mismatch"
            )
        if _shape(current) != _shape(incoming):
            raise _core.CheckpointCompatibilityError(
                f"{label} {path} shape mismatch: checkpoint {_shape(incoming)} vs target {_shape(current)}"
            )
        if _dtype_name(current) != _dtype_name(incoming):
            raise _core.CheckpointCompatibilityError(
                f"{label} {path} dtype mismatch: checkpoint {_dtype_name(incoming)} vs target {_dtype_name(current)}"
            )
        return

    if isinstance(current, Mapping):
        if not isinstance(incoming, Mapping):
            raise _core.CheckpointCompatibilityError(
                f"{label} {path} mapping/type mismatch"
            )
        if current and set(current) != set(incoming):
            missing = sorted(str(key) for key in set(current) - set(incoming))
            unexpected = sorted(str(key) for key in set(incoming) - set(current))
            raise _core.CheckpointCompatibilityError(
                f"{label} {path} keys differ: missing={missing}, unexpected={unexpected}"
            )
        for key in current.keys() & incoming.keys():
            _preflight_tree_compatibility(
                current[key], incoming[key], label=label, path=f"{path}.{key}"
            )
        return

    if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
        if not isinstance(incoming, Sequence) or isinstance(
            incoming, (str, bytes, bytearray)
        ):
            raise _core.CheckpointCompatibilityError(
                f"{label} {path} sequence/type mismatch"
            )
        if len(current) != len(incoming):
            raise _core.CheckpointCompatibilityError(
                f"{label} {path} length mismatch: checkpoint {len(incoming)} vs target {len(current)}"
            )
        for index, (current_item, incoming_item) in enumerate(zip(current, incoming)):
            _preflight_tree_compatibility(
                current_item,
                incoming_item,
                label=label,
                path=f"{path}[{index}]",
            )


def _preflight_torch_optimizer_state(optimizer: Any, incoming: Any) -> None:
    """Bind each serialized per-parameter tensor to live parameter geometry."""

    if not isinstance(incoming, Mapping):
        raise _core.CheckpointCompatibilityError(
            "optimizer checkpoint state must be a mapping"
        )
    incoming_state = incoming.get("state")
    incoming_groups = incoming.get("param_groups")
    if not isinstance(incoming_state, Mapping) or not isinstance(incoming_groups, list):
        raise _core.CheckpointCompatibilityError(
            "PyTorch optimizer state must contain state and param_groups"
        )
    live_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(live_groups, list) or len(live_groups) != len(incoming_groups):
        raise _core.CheckpointCompatibilityError(
            "optimizer parameter-group count differs from checkpoint"
        )

    checkpoint_id_to_param: dict[Any, Any] = {}
    for group_index, (live_group, saved_group) in enumerate(
        zip(live_groups, incoming_groups)
    ):
        live_params = live_group.get("params") if isinstance(live_group, Mapping) else None
        saved_params = saved_group.get("params") if isinstance(saved_group, Mapping) else None
        if not isinstance(live_params, list) or not isinstance(saved_params, list):
            raise _core.CheckpointCompatibilityError(
                f"optimizer param_group[{group_index}] params are invalid"
            )
        if len(live_params) != len(saved_params):
            raise _core.CheckpointCompatibilityError(
                f"optimizer param_group[{group_index}] parameter count mismatch"
            )
        for saved_id, live_param in zip(saved_params, live_params):
            if saved_id in checkpoint_id_to_param:
                raise _core.CheckpointCompatibilityError(
                    f"optimizer checkpoint parameter id {saved_id!r} is duplicated"
                )
            checkpoint_id_to_param[saved_id] = live_param

    unknown = set(incoming_state) - set(checkpoint_id_to_param)
    if unknown:
        raise _core.CheckpointCompatibilityError(
            f"optimizer checkpoint state references unknown parameters: {sorted(map(str, unknown))}"
        )

    def check_leaf(value: Any, *, parameter: Any, state_key: str, path: str) -> None:
        if _is_tensor_like(value):
            shape = _shape(value)
            parameter_shape = _shape(parameter)
            if shape == () or (state_key == "step" and _numel(value) == 1):
                return
            if shape != parameter_shape:
                raise _core.CheckpointCompatibilityError(
                    f"optimizer {path} shape mismatch: checkpoint {shape} vs parameter {parameter_shape}"
                )
            return
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                check_leaf(
                    child,
                    parameter=parameter,
                    state_key=str(child_key),
                    path=f"{path}.{child_key}",
                )
            return
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, child in enumerate(value):
                check_leaf(
                    child,
                    parameter=parameter,
                    state_key=state_key,
                    path=f"{path}[{index}]",
                )

    for saved_id, per_parameter_state in incoming_state.items():
        if not isinstance(per_parameter_state, Mapping):
            raise _core.CheckpointCompatibilityError(
                f"optimizer state for parameter {saved_id!r} must be a mapping"
            )
        parameter = checkpoint_id_to_param[saved_id]
        for state_key, value in per_parameter_state.items():
            check_leaf(
                value,
                parameter=parameter,
                state_key=str(state_key),
                path=f"state[{saved_id!r}].{state_key}",
            )


def _preflight_load_state(target: Any, incoming: Any, *, label: str) -> None:
    if not hasattr(target, "state_dict") or not hasattr(target, "load_state_dict"):
        raise _core.CheckpointCompatibilityError(
            f"{label} must provide state_dict() and load_state_dict()"
        )

    current = target.state_dict()
    if label == "optimizer" and hasattr(target, "param_groups"):
        _preflight_torch_optimizer_state(target, incoming)
    else:
        _preflight_tree_compatibility(current, incoming, label=label)

    try:
        probe = copy.deepcopy(target)
        probe.load_state_dict(copy.deepcopy(incoming))
    except Exception as exc:
        raise _core.CheckpointCompatibilityError(
            f"{label} checkpoint state cannot be loaded into an isolated target copy"
        ) from exc


def load_verified_checkpoint(
    verified: _core.VerifiedCheckpoint,
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
) -> _core.LoadResult:
    """Preflight all requested mutable targets, then delegate the atomic load."""

    _validate_manifest_identity_fail_closed(verified.manifest.get("identity"))
    _arrays, combined_state = _core._decode_verified_state(verified)

    if optimizer is not None:
        optimizer_state = combined_state.get("optimizer")
        if optimizer_state is None:
            raise _core.CheckpointCompatibilityError(
                "optimizer was requested but checkpoint has no optimizer state"
            )
        _preflight_load_state(optimizer, optimizer_state, label="optimizer")
    if scheduler is not None:
        scheduler_state = combined_state.get("scheduler")
        if scheduler_state is None:
            raise _core.CheckpointCompatibilityError(
                "scheduler was requested but checkpoint has no scheduler state"
            )
        _preflight_load_state(scheduler, scheduler_state, label="scheduler")

    return _ORIGINAL_LOAD_VERIFIED_CHECKPOINT(
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


def load_checkpoint(
    directory: str | Any,
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
) -> _core.LoadResult:
    """Verify a byte snapshot, hard-preflight it, and only then mutate targets."""

    verified = _core.prepare_checkpoint_load(directory)
    return load_verified_checkpoint(
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


def install_checkpoint_hardening() -> None:
    """Install fail-closed checks into the core module exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return
    _core._validate_manifest_identity = _validate_manifest_identity_fail_closed
    _core._materialize_for_target = _materialize_for_target_fail_closed
    _core.load_verified_checkpoint = load_verified_checkpoint
    _core.load_checkpoint = load_checkpoint
    _INSTALLED = True
