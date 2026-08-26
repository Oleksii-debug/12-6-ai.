"""Fail-closed semantic hardening for checkpoint-v1 loads.

This module closes corruption cases that are checksum-valid but semantically
invalid.  It is installed by :mod:`twelve_six.checkpoint` before public
checkpoint functions are re-exported so both package and direct core imports
share the hardened behavior.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np


def _is_torch_tensor(value: Any) -> bool:
    cls = value.__class__
    return cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}


def _is_torch_optimizer(value: Any) -> bool:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return False
    return isinstance(value, torch.optim.Optimizer)


def _is_torch_scheduler(value: Any) -> bool:
    module = value.__class__.__module__
    if module.startswith("torch.optim.lr_scheduler"):
        return True
    optimizer = getattr(value, "optimizer", None)
    return optimizer is not None and _is_torch_optimizer(optimizer)


def _fail(core: Any, message: str) -> None:
    raise core.CheckpointCompatibilityError(message)


def _validate_manifest_scalars(core: Any, identity: Mapping[str, Any]) -> None:
    for name in ("model_spec", "training_config", "optimizer", "environment"):
        value = identity.get(name)
        if not isinstance(value, Mapping) or not value:
            raise core.CheckpointIntegrityError(f"identity.{name} must be a non-empty mapping")

    scheduler = identity.get("scheduler")
    if scheduler is not None and (not isinstance(scheduler, Mapping) or not scheduler):
        raise core.CheckpointIntegrityError(
            "identity.scheduler must be a non-empty mapping or null"
        )

    parameter_count = identity.get("parameter_count")
    if (
        not isinstance(parameter_count, int)
        or isinstance(parameter_count, bool)
        or parameter_count <= 0
    ):
        raise core.CheckpointIntegrityError(
            "identity.parameter_count must be a positive integer"
        )

    for name in ("seed", "step", "tokens_seen"):
        value = identity.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise core.CheckpointIntegrityError(
                f"identity.{name} must be a non-negative integer"
            )

    precision = identity.get("precision")
    if not isinstance(precision, str) or not precision.strip():
        raise core.CheckpointIntegrityError("identity.precision must be a non-empty string")


def _strict_materialize(core: Any, array: np.ndarray, target: Any) -> Any:
    if isinstance(target, np.ndarray):
        if tuple(target.shape) != tuple(array.shape):
            _fail(
                core,
                f"shape mismatch: checkpoint {tuple(array.shape)} vs target {tuple(target.shape)}",
            )
        if array.dtype != target.dtype:
            _fail(
                core,
                f"dtype mismatch: checkpoint {array.dtype} vs target {target.dtype}",
            )
        return array.copy()

    if _is_torch_tensor(target):
        torch = importlib.import_module("torch")
        if tuple(target.shape) != tuple(array.shape):
            _fail(
                core,
                f"shape mismatch: checkpoint {tuple(array.shape)} vs target {tuple(target.shape)}",
            )

        if str(target.dtype) == "torch.bfloat16":
            if array.dtype != np.dtype(np.uint16):
                _fail(
                    core,
                    f"dtype mismatch: checkpoint {array.dtype} vs target torch.bfloat16 storage uint16",
                )
            tensor = torch.from_numpy(array.copy()).view(torch.bfloat16)
        else:
            try:
                tensor = torch.from_numpy(array.copy())
            except (TypeError, ValueError, RuntimeError) as exc:
                raise core.CheckpointCompatibilityError(
                    f"checkpoint dtype {array.dtype} cannot materialize for target {target.dtype}"
                ) from exc
            if tensor.dtype != target.dtype:
                _fail(
                    core,
                    f"dtype mismatch: checkpoint {tensor.dtype} vs target {target.dtype}",
                )
        return tensor.to(device=target.device)

    _fail(core, f"unsupported target tensor type {type(target)!r}")
    raise AssertionError("unreachable")


def _tensor_signature(value: Any) -> tuple[tuple[int, ...], str] | None:
    if isinstance(value, np.ndarray):
        return tuple(value.shape), str(value.dtype)
    if _is_torch_tensor(value):
        return tuple(value.shape), str(value.dtype)
    return None


def _validate_state_shape_tree(core: Any, current: Any, incoming: Any, path: str) -> None:
    """Validate structural/tensor compatibility without applying state."""

    current_sig = _tensor_signature(current)
    incoming_sig = _tensor_signature(incoming)
    if current_sig is not None or incoming_sig is not None:
        if current_sig is None or incoming_sig is None:
            _fail(core, f"{path} tensor/non-tensor mismatch")
        if current_sig != incoming_sig:
            _fail(
                core,
                f"{path} tensor signature mismatch: checkpoint {incoming_sig} vs target {current_sig}",
            )
        return

    if isinstance(current, Mapping):
        if not isinstance(incoming, Mapping):
            _fail(core, f"{path} must be a mapping")
        current_keys = set(current)
        incoming_keys = set(incoming)
        if current_keys != incoming_keys:
            _fail(
                core,
                f"{path} keys differ: missing={sorted(current_keys - incoming_keys, key=str)}, "
                f"unexpected={sorted(incoming_keys - current_keys, key=str)}",
            )
        for key in current_keys:
            _validate_state_shape_tree(
                core,
                current[key],
                incoming[key],
                f"{path}.{key}",
            )
        return

    if isinstance(current, (list, tuple)):
        if not isinstance(incoming, type(current)) or len(current) != len(incoming):
            _fail(core, f"{path} sequence structure mismatch")
        for index, (current_item, incoming_item) in enumerate(zip(current, incoming)):
            _validate_state_shape_tree(
                core,
                current_item,
                incoming_item,
                f"{path}[{index}]",
            )


def _validate_torch_state_leaf(core: Any, value: Any, parameter: Any, path: str) -> None:
    signature = _tensor_signature(value)
    if signature is not None:
        shape, _ = signature
        if len(shape) == 0:
            return
        if not _is_torch_tensor(value):
            _fail(core, f"{path} must be a torch tensor for a torch optimizer")
        expected_shape = tuple(parameter.shape)
        if shape != expected_shape:
            _fail(
                core,
                f"{path} shape mismatch: checkpoint {shape} vs parameter {expected_shape}",
            )

        torch = importlib.import_module("torch")
        if parameter.is_floating_point():
            allowed_dtypes = {parameter.dtype}
            if parameter.dtype in {torch.float16, torch.bfloat16}:
                allowed_dtypes.add(torch.float32)
            if value.dtype not in allowed_dtypes:
                expected = ", ".join(sorted(str(item) for item in allowed_dtypes))
                _fail(
                    core,
                    f"{path} dtype mismatch: checkpoint {value.dtype}; expected one of {expected}",
                )
        elif value.dtype != parameter.dtype:
            _fail(
                core,
                f"{path} dtype mismatch: checkpoint {value.dtype} vs parameter {parameter.dtype}",
            )
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_torch_state_leaf(core, item, parameter, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_torch_state_leaf(core, item, parameter, f"{path}[{index}]")


def _preflight_torch_optimizer(core: Any, optimizer: Any, incoming: Any) -> None:
    if not isinstance(incoming, Mapping):
        _fail(core, "checkpoint optimizer state must be a mapping")
    state = incoming.get("state")
    saved_groups = incoming.get("param_groups")
    if not isinstance(state, Mapping) or not isinstance(saved_groups, (list, tuple)):
        _fail(core, "checkpoint torch optimizer state must contain state and param_groups")

    live_groups = optimizer.param_groups
    if len(saved_groups) != len(live_groups):
        _fail(
            core,
            f"optimizer param-group count mismatch: checkpoint {len(saved_groups)} vs target {len(live_groups)}",
        )

    id_to_parameter: dict[Any, Any] = {}
    for group_index, (saved_group, live_group) in enumerate(zip(saved_groups, live_groups)):
        if not isinstance(saved_group, Mapping):
            _fail(core, f"optimizer param_groups[{group_index}] must be a mapping")
        saved_params = saved_group.get("params")
        live_params = live_group.get("params")
        if not isinstance(saved_params, (list, tuple)) or not isinstance(live_params, (list, tuple)):
            _fail(core, f"optimizer param_groups[{group_index}].params must be a sequence")
        if len(saved_params) != len(live_params):
            _fail(
                core,
                f"optimizer param_groups[{group_index}] size mismatch: "
                f"checkpoint {len(saved_params)} vs target {len(live_params)}",
            )
        for saved_id, parameter in zip(saved_params, live_params):
            try:
                existing = id_to_parameter.get(saved_id)
            except TypeError:
                _fail(core, f"optimizer parameter id {saved_id!r} is not hashable")
            if existing is not None and existing is not parameter:
                _fail(core, f"optimizer parameter id {saved_id!r} is duplicated")
            id_to_parameter[saved_id] = parameter

    for saved_id, parameter_state in state.items():
        try:
            parameter = id_to_parameter[saved_id]
        except (KeyError, TypeError):
            _fail(core, f"optimizer state contains orphan parameter id {saved_id!r}")
        if not isinstance(parameter_state, Mapping):
            _fail(core, f"optimizer state[{saved_id!r}] must be a mapping")
        _validate_torch_state_leaf(
            core,
            parameter_state,
            parameter,
            f"optimizer.state[{saved_id!r}]",
        )


def _preflight_generic_stateful(core: Any, obj: Any, incoming: Any, role: str) -> None:
    if not hasattr(obj, "state_dict") or not hasattr(obj, "load_state_dict"):
        _fail(core, f"{role} must provide state_dict() and load_state_dict()")
    current = obj.state_dict()
    _validate_state_shape_tree(core, current, incoming, role)

    if _is_torch_scheduler(obj):
        # Deep-copying a torch scheduler may duplicate its optimizer and model.
        # Structural validation is sufficient for scheduler state, whose tensors
        # (if any) must already match the live scheduler state signature.
        return

    try:
        probe = copy.deepcopy(obj)
        probe.load_state_dict(copy.deepcopy(incoming))
    except Exception as exc:
        raise core.CheckpointCompatibilityError(
            f"checkpoint {role} state cannot be loaded into a detached probe"
        ) from exc


def _preflight_optimizer(core: Any, optimizer: Any, incoming: Any) -> None:
    if _is_torch_optimizer(optimizer):
        if not hasattr(optimizer, "load_state_dict"):
            _fail(core, "optimizer must provide load_state_dict()")
        _preflight_torch_optimizer(core, optimizer, incoming)
        return
    _preflight_generic_stateful(core, optimizer, incoming, "optimizer")


def _hardened_load_verified_checkpoint(
    core: Any,
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
    manifest = verified.manifest
    core.assert_identity(
        manifest,
        git_sha=expected_git_sha,
        model_spec_hash=expected_model_spec_hash,
        tokenizer_hash=expected_tokenizer_hash,
        tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
        dataset_manifest_hash=expected_dataset_manifest_hash,
        run_manifest_hash=expected_run_manifest_hash,
    )
    arrays, combined_state = core._decode_verified_state(verified)
    materialized = core._prepare_model_weights(model, arrays, strict_model)

    if optimizer is not None:
        optimizer_state = combined_state.get("optimizer")
        if optimizer_state is None:
            raise core.CheckpointCompatibilityError(
                "optimizer was requested but checkpoint has no optimizer state"
            )
        _preflight_optimizer(core, optimizer, optimizer_state)

    if scheduler is not None:
        scheduler_state = combined_state.get("scheduler")
        if scheduler_state is None:
            raise core.CheckpointCompatibilityError(
                "scheduler was requested but checkpoint has no scheduler state"
            )
        _preflight_generic_stateful(core, scheduler, scheduler_state, "scheduler")

    if restore_rng:
        core._preflight_rng_state(combined_state["rng"])

    # Every checksum, identity invariant, model tensor signature, requested
    # optimizer/scheduler state, and RNG stream is validated before mutation.
    core._apply_model_weights(model, materialized, strict_model)
    if optimizer is not None:
        optimizer.load_state_dict(combined_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(combined_state["scheduler"])
    if restore_rng:
        core.restore_rng_state(combined_state["rng"])

    return core.LoadResult(
        manifest=copy.deepcopy(manifest),
        trainer_state=combined_state.get("trainer", {}),
        rng_state=combined_state["rng"],
    )


def install(core: Any) -> None:
    """Install D05 hardening into the checkpoint core exactly once."""

    if getattr(core, "_D05_INTEGRITY_HARDENING_INSTALLED", False):
        return

    original_validate_manifest_identity = core._validate_manifest_identity

    def validate_manifest_identity(identity: Any) -> None:
        original_validate_manifest_identity(identity)
        _validate_manifest_scalars(core, identity)

    def materialize_for_target(array: np.ndarray, target: Any) -> Any:
        return _strict_materialize(core, array, target)

    def load_verified_checkpoint(verified: Any, **kwargs: Any) -> Any:
        return _hardened_load_verified_checkpoint(core, verified, **kwargs)

    core._validate_manifest_identity = validate_manifest_identity
    core._materialize_for_target = materialize_for_target
    core.load_verified_checkpoint = load_verified_checkpoint
    core._D05_INTEGRITY_HARDENING_INSTALLED = True
