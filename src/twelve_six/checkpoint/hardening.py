"""Fail-closed checkpoint load hardening for D05.

This module installs strict runtime guards around checkpoint-v1 without changing
its on-disk format.  The guards close three corruption classes found by the
MODEL-341 20M checkpoint red-team:

* model tensor dtype mismatches must not be silently cast;
* optimizer state must be structurally compatible before model mutation;
* manifest scalar/counter invariants must remain valid even after a consistent
  manifest/checkpoint-id rebind.

The package initializer installs these guards before trainer_adapter imports the
core load functions, so both package-level and ``checkpoint.core`` consumers use
the hardened implementation.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import core as _core

_ORIGINAL_VALIDATE_MANIFEST_IDENTITY = _core._validate_manifest_identity
_SCALAR_OPTIMIZER_STATE_NAMES = frozenset({"step", "eta", "mu", "mu_product"})


def _fail(message: str) -> None:
    raise _core.CheckpointCompatibilityError(message)


def _require_non_negative_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _core.CheckpointIntegrityError(f"{field} must be a non-negative integer")
    return value


def _validate_manifest_identity(identity: Any) -> None:
    """Extend the v1 identity verifier with scalar/schema invariants."""

    _ORIGINAL_VALIDATE_MANIFEST_IDENTITY(identity)
    assert isinstance(identity, Mapping)  # established by the original validator

    if not isinstance(identity.get("model_spec"), Mapping) or not identity["model_spec"]:
        raise _core.CheckpointIntegrityError("identity.model_spec must be a non-empty mapping")
    if not isinstance(identity.get("training_config"), Mapping) or not identity["training_config"]:
        raise _core.CheckpointIntegrityError(
            "identity.training_config must be a non-empty mapping"
        )
    if not isinstance(identity.get("optimizer"), Mapping) or not identity["optimizer"]:
        raise _core.CheckpointIntegrityError("identity.optimizer must be a non-empty mapping")
    scheduler = identity.get("scheduler")
    if scheduler is not None and (not isinstance(scheduler, Mapping) or not scheduler):
        raise _core.CheckpointIntegrityError(
            "identity.scheduler must be a non-empty mapping or null"
        )

    parameter_count = identity.get("parameter_count")
    if (
        not isinstance(parameter_count, int)
        or isinstance(parameter_count, bool)
        or parameter_count <= 0
    ):
        raise _core.CheckpointIntegrityError(
            "identity.parameter_count must be a positive integer"
        )
    _require_non_negative_int(identity.get("seed"), field="identity.seed")
    _require_non_negative_int(identity.get("step"), field="identity.step")
    _require_non_negative_int(identity.get("tokens_seen"), field="identity.tokens_seen")

    precision = identity.get("precision")
    if not isinstance(precision, str) or not precision.strip():
        raise _core.CheckpointIntegrityError(
            "identity.precision must be a non-empty string"
        )


def _materialize_for_target(array: np.ndarray, target: Any) -> Any:
    """Materialize only exact dtype-compatible tensors; never cast corruption."""

    if isinstance(target, np.ndarray):
        if tuple(target.shape) != tuple(array.shape):
            _fail(
                f"shape mismatch: checkpoint {tuple(array.shape)} vs "
                f"target {tuple(target.shape)}"
            )
        if np.dtype(array.dtype) != np.dtype(target.dtype):
            _fail(
                f"dtype mismatch: checkpoint {array.dtype} vs target {target.dtype}"
            )
        return array.copy()

    cls = target.__class__
    if cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}:
        torch = __import__("torch")
        if tuple(target.shape) != tuple(array.shape):
            _fail(
                f"shape mismatch: checkpoint {tuple(array.shape)} vs "
                f"target {tuple(target.shape)}"
            )

        if str(target.dtype) == "torch.bfloat16":
            if np.dtype(array.dtype) != np.dtype(np.uint16):
                _fail(
                    "dtype mismatch: bfloat16 target requires the explicit "
                    f"uint16 checkpoint representation, got {array.dtype}"
                )
            tensor = torch.from_numpy(array.copy()).view(torch.bfloat16)
        else:
            try:
                target_numpy_dtype = target.detach().cpu().numpy().dtype
            except (RuntimeError, TypeError) as exc:
                raise _core.CheckpointCompatibilityError(
                    f"cannot determine target tensor dtype for {target.dtype}"
                ) from exc
            if np.dtype(array.dtype) != np.dtype(target_numpy_dtype):
                _fail(
                    f"dtype mismatch: checkpoint {array.dtype} vs "
                    f"target {target_numpy_dtype}"
                )
            tensor = torch.from_numpy(array.copy())
            if tensor.dtype != target.dtype:
                _fail(
                    f"dtype mismatch after exact materialization: checkpoint "
                    f"{tensor.dtype} vs target {target.dtype}"
                )
        return tensor.to(device=target.device)

    _fail(f"unsupported target tensor type {type(target)!r}")
    raise AssertionError("unreachable")


def _validate_scalar_optimizer_value(value: Any, *, field: str) -> None:
    if hasattr(value, "numel") and callable(value.numel):
        if int(value.numel()) != 1:
            _fail(f"optimizer scalar state {field} must contain exactly one value")
        try:
            scalar = value.detach().cpu().item()
        except (AttributeError, RuntimeError, ValueError, TypeError) as exc:
            raise _core.CheckpointCompatibilityError(
                f"optimizer scalar state {field} cannot be decoded"
            ) from exc
    else:
        scalar = value

    if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
        _fail(f"optimizer scalar state {field} must be numeric")
    if isinstance(scalar, float) and not math.isfinite(scalar):
        _fail(f"optimizer scalar state {field} must be finite")
    if scalar < 0:
        _fail(f"optimizer scalar state {field} must be non-negative")


def _validate_optimizer_value(param: Any, field: str, value: Any) -> None:
    cls = value.__class__
    is_torch_tensor = cls.__module__.startswith("torch") and cls.__name__ in {
        "Tensor",
        "Parameter",
    }
    if is_torch_tensor:
        if field in _SCALAR_OPTIMIZER_STATE_NAMES:
            _validate_scalar_optimizer_value(value, field=field)
            return
        if tuple(value.shape) != tuple(param.shape):
            _fail(
                f"optimizer state shape mismatch for {field}: checkpoint "
                f"{tuple(value.shape)} vs parameter {tuple(param.shape)}"
            )
        if getattr(param, "is_floating_point", lambda: False)() and not (
            value.is_floating_point() or value.is_complex()
        ):
            _fail(
                f"optimizer state dtype is incompatible for {field}: "
                f"checkpoint {value.dtype} vs parameter {param.dtype}"
            )
        return

    if isinstance(value, np.ndarray):
        if field in _SCALAR_OPTIMIZER_STATE_NAMES:
            _validate_scalar_optimizer_value(value.item() if value.size == 1 else value, field=field)
            return
        if tuple(value.shape) != tuple(param.shape):
            _fail(
                f"optimizer state shape mismatch for {field}: checkpoint "
                f"{tuple(value.shape)} vs parameter {tuple(param.shape)}"
            )
        return

    if field in _SCALAR_OPTIMIZER_STATE_NAMES:
        _validate_scalar_optimizer_value(value, field=field)
        return

    if isinstance(value, Mapping):
        for nested_name, nested_value in value.items():
            _validate_optimizer_value(param, f"{field}.{nested_name}", nested_value)
        return
    if isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(value):
            _validate_optimizer_value(param, f"{field}[{index}]", nested_value)
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    _fail(f"unsupported optimizer state value for {field}: {type(value)!r}")


def _preflight_optimizer_state(optimizer: Any, state: Any) -> None:
    """Validate a PyTorch-style optimizer state without mutating live targets."""

    if not isinstance(state, Mapping):
        _fail("checkpoint optimizer state must be a mapping")
    if not hasattr(optimizer, "param_groups") or not hasattr(optimizer, "state_dict"):
        _fail("optimizer must expose param_groups and state_dict() for safe preflight")

    checkpoint_groups = state.get("param_groups")
    checkpoint_state = state.get("state")
    live_groups = optimizer.param_groups
    if not isinstance(checkpoint_groups, (list, tuple)):
        _fail("checkpoint optimizer param_groups must be a list")
    if not isinstance(checkpoint_state, Mapping):
        _fail("checkpoint optimizer state table must be a mapping")
    if len(checkpoint_groups) != len(live_groups):
        _fail(
            "optimizer parameter-group count mismatch: checkpoint "
            f"{len(checkpoint_groups)} vs target {len(live_groups)}"
        )

    saved_to_live: dict[Any, Any] = {}
    for group_index, (saved_group, live_group) in enumerate(
        zip(checkpoint_groups, live_groups, strict=True)
    ):
        if not isinstance(saved_group, Mapping) or not isinstance(live_group, Mapping):
            _fail(f"optimizer parameter group {group_index} must be a mapping")
        saved_params = saved_group.get("params")
        live_params = live_group.get("params")
        if not isinstance(saved_params, (list, tuple)) or not isinstance(
            live_params, (list, tuple)
        ):
            _fail(f"optimizer parameter group {group_index} has invalid params")
        if len(saved_params) != len(live_params):
            _fail(
                f"optimizer parameter count mismatch in group {group_index}: "
                f"checkpoint {len(saved_params)} vs target {len(live_params)}"
            )
        for saved_id, live_param in zip(saved_params, live_params, strict=True):
            if saved_id in saved_to_live and saved_to_live[saved_id] is not live_param:
                _fail(f"optimizer parameter id {saved_id!r} is mapped more than once")
            saved_to_live[saved_id] = live_param

    unknown_state_ids = set(checkpoint_state) - set(saved_to_live)
    if unknown_state_ids:
        _fail(
            "optimizer state contains unknown parameter ids: "
            f"{sorted(unknown_state_ids, key=repr)}"
        )

    for saved_id, saved_entry in checkpoint_state.items():
        if not isinstance(saved_entry, Mapping):
            _fail(f"optimizer state for parameter {saved_id!r} must be a mapping")
        live_param = saved_to_live[saved_id]
        for field, value in saved_entry.items():
            _validate_optimizer_value(live_param, str(field), value)


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
    """Preflight all known unsafe continuation paths before first mutation."""

    manifest = verified.manifest
    _core.assert_identity(
        manifest,
        git_sha=expected_git_sha,
        model_spec_hash=expected_model_spec_hash,
        tokenizer_hash=expected_tokenizer_hash,
        tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
        dataset_manifest_hash=expected_dataset_manifest_hash,
        run_manifest_hash=expected_run_manifest_hash,
    )
    arrays, combined_state = _core._decode_verified_state(verified)
    materialized = _core._prepare_model_weights(model, arrays, strict_model)

    if optimizer is not None:
        optimizer_state = combined_state.get("optimizer")
        if optimizer_state is None:
            _fail("optimizer was requested but checkpoint has no optimizer state")
        _preflight_optimizer_state(optimizer, optimizer_state)
    if scheduler is not None and combined_state.get("scheduler") is None:
        _fail("scheduler was requested but checkpoint has no scheduler state")
    if restore_rng:
        _core._preflight_rng_state(combined_state["rng"])

    # All byte integrity, identity, model shape/dtype, optimizer structure and RNG
    # checks complete before the first live target mutation.
    _core._apply_model_weights(model, materialized, strict_model)
    if optimizer is not None:
        optimizer.load_state_dict(combined_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(combined_state["scheduler"])
    if restore_rng:
        _core.restore_rng_state(combined_state["rng"])
    return _core.LoadResult(
        manifest=copy.deepcopy(manifest),
        trainer_state=combined_state.get("trainer", {}),
        rng_state=combined_state["rng"],
    )


def load_checkpoint(
    directory: str | _core.Path,
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
    """Install hardened helpers/loaders into the already-loaded core module."""

    _core._validate_manifest_identity = _validate_manifest_identity
    _core._materialize_for_target = _materialize_for_target
    _core.load_verified_checkpoint = load_verified_checkpoint
    _core.load_checkpoint = load_checkpoint
