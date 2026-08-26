"""Fail-closed D05 checkpoint load hardening.

This module wraps the checkpoint-v1 loader with corruption preflights that must
complete before the production core mutates model, optimizer, scheduler, or RNG
state. It intentionally keeps the serialized checkpoint format unchanged.

The module also installs the hardened load/verify entry points back onto
``checkpoint.core`` at import time. The original core implementations are kept
as private delegates so callers cannot bypass the preflight merely by importing
``twelve_six.checkpoint.core`` directly.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load as load_safetensors_bytes

from . import core as _core

CheckpointCompatibilityError = _core.CheckpointCompatibilityError
CheckpointIntegrityError = _core.CheckpointIntegrityError
LoadResult = _core.LoadResult
VerifiedCheckpoint = _core.VerifiedCheckpoint

# Freeze the verified production implementations before installing the hardened
# entry points back onto the core module. Hardened wrappers delegate only to
# these saved callables, which prevents recursion after installation.
_CORE_PREPARE_CHECKPOINT_LOAD = _core.prepare_checkpoint_load
_CORE_LOAD_VERIFIED_CHECKPOINT = _core.load_verified_checkpoint


def _validate_identity_counters(manifest: Mapping[str, Any]) -> None:
    """Reject malformed or negative resume counters before target mutation."""

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise CheckpointIntegrityError("manifest identity must be a mapping")
    for field in ("step", "tokens_seen"):
        value = identity.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CheckpointIntegrityError(
                f"identity.{field} must be a non-negative integer"
            )


def _validate_model_tensor_dtypes(
    model: Any,
    arrays: Mapping[str, np.ndarray],
    *,
    strict: bool,
) -> None:
    """Require exact checkpoint/target tensor dtype compatibility.

    SafeTensors cannot encode BF16 through NumPy directly, so checkpoint-v1
    deliberately stores BF16 payload bits as uint16. That is the only accepted
    representation exception; all other dtypes must match exactly.
    """

    if not hasattr(model, "state_dict"):
        raise CheckpointCompatibilityError("model must provide state_dict()")
    target_state = model.state_dict()
    if not isinstance(target_state, Mapping):
        raise CheckpointCompatibilityError("model.state_dict() must be a mapping")

    target_keys = set(target_state)
    source_keys = set(arrays)
    if strict and target_keys != source_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise CheckpointCompatibilityError(
            f"state_dict keys differ: missing={missing}, unexpected={unexpected}"
        )

    for name in target_keys & source_keys:
        array = arrays[name]
        target = target_state[name]
        if isinstance(target, np.ndarray):
            if array.dtype != target.dtype:
                raise CheckpointCompatibilityError(
                    f"dtype mismatch for {name}: checkpoint {array.dtype} vs target {target.dtype}"
                )
            continue

        cls = target.__class__
        if not (
            cls.__module__.startswith("torch")
            and cls.__name__ in {"Tensor", "Parameter"}
        ):
            raise CheckpointCompatibilityError(
                f"unsupported target tensor type for {name}: {type(target)!r}"
            )

        import torch

        if str(target.dtype) == "torch.bfloat16":
            if array.dtype != np.dtype(np.uint16):
                raise CheckpointCompatibilityError(
                    f"dtype mismatch for {name}: checkpoint {array.dtype} vs target torch.bfloat16 "
                    "(expected uint16 BF16 bit representation)"
                )
            continue

        try:
            source_dtype = torch.from_numpy(np.empty((), dtype=array.dtype)).dtype
        except (TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                f"checkpoint dtype for {name} cannot be materialized by torch: {array.dtype}"
            ) from exc
        if source_dtype != target.dtype:
            raise CheckpointCompatibilityError(
                f"dtype mismatch for {name}: checkpoint {source_dtype} vs target {target.dtype}"
            )


def _iter_state_tensors(value: Any, path: str = "state"):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_state_tensors(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_state_tensors(item, f"{path}[{index}]")
        return
    cls = value.__class__
    if cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}:
        yield path, value


def _validate_loaded_optimizer_semantics(optimizer: Any) -> None:
    """Check loaded tensor states against the parameter they will update."""

    if not hasattr(optimizer, "param_groups") or not hasattr(optimizer, "state"):
        raise CheckpointCompatibilityError(
            "optimizer must expose param_groups and state for safe checkpoint preflight"
        )

    for group_index, group in enumerate(optimizer.param_groups):
        params = group.get("params") if isinstance(group, Mapping) else None
        if not isinstance(params, (list, tuple)):
            raise CheckpointCompatibilityError(
                f"optimizer param group {group_index} has invalid params inventory"
            )
        for param_index, param in enumerate(params):
            state = optimizer.state.get(param, {})
            if not isinstance(state, Mapping):
                raise CheckpointCompatibilityError(
                    f"optimizer state for group {group_index} param {param_index} must be a mapping"
                )
            for state_path, tensor in _iter_state_tensors(state):
                # Optimizer step counters are commonly scalar tensors and are
                # intentionally exempt from parameter-shape checks.
                if tensor.ndim == 0 or tensor.numel() == 1:
                    continue
                if tuple(tensor.shape) != tuple(param.shape):
                    raise CheckpointCompatibilityError(
                        "optimizer state tensor shape mismatch at "
                        f"group {group_index} param {param_index} {state_path}: "
                        f"state {tuple(tensor.shape)} vs parameter {tuple(param.shape)}"
                    )
                if tensor.is_floating_point():
                    import torch

                    allowed = {param.dtype, torch.float32}
                    if tensor.dtype not in allowed:
                        raise CheckpointCompatibilityError(
                            "optimizer state tensor dtype mismatch at "
                            f"group {group_index} param {param_index} {state_path}: "
                            f"state {tensor.dtype} vs allowed "
                            f"{sorted(str(item) for item in allowed)}"
                        )


def _preflight_optimizer_state(optimizer: Any, state: Any) -> None:
    """Load optimizer state into an isolated copy and validate update semantics."""

    if not hasattr(optimizer, "load_state_dict"):
        raise CheckpointCompatibilityError("optimizer must provide load_state_dict()")
    try:
        probe = copy.deepcopy(optimizer)
    except Exception as exc:
        raise CheckpointCompatibilityError(
            "optimizer cannot be isolated for fail-closed checkpoint preflight"
        ) from exc
    try:
        probe.load_state_dict(copy.deepcopy(state))
    except Exception as exc:
        raise CheckpointCompatibilityError(
            "checkpoint optimizer state is incompatible with the target optimizer"
        ) from exc
    _validate_loaded_optimizer_semantics(probe)


def _preflight_stateful_object(obj: Any, state: Any, *, label: str) -> None:
    if not hasattr(obj, "load_state_dict"):
        raise CheckpointCompatibilityError(f"{label} must provide load_state_dict()")
    try:
        probe = copy.deepcopy(obj)
        probe.load_state_dict(copy.deepcopy(state))
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint {label} state is incompatible with the target {label}"
        ) from exc


def _decode_weights(verified: VerifiedCheckpoint) -> dict[str, np.ndarray]:
    try:
        return load_safetensors_bytes(verified._artifacts[_core.WEIGHTS_NAME])
    except Exception as exc:
        raise CheckpointIntegrityError("weights.safetensors cannot be decoded") from exc


def prepare_checkpoint_load(directory: str | Path) -> VerifiedCheckpoint:
    verified = _CORE_PREPARE_CHECKPOINT_LOAD(directory)
    _validate_identity_counters(verified.manifest)
    return verified


def verify_checkpoint(directory: str | Path) -> dict[str, Any]:
    return prepare_checkpoint_load(directory).manifest


def load_verified_checkpoint(
    verified: VerifiedCheckpoint,
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
) -> LoadResult:
    """Run corruption preflights, then delegate mutation to checkpoint-v1 core."""

    manifest = verified.manifest
    _validate_identity_counters(manifest)
    arrays = _decode_weights(verified)
    _validate_model_tensor_dtypes(model, arrays, strict=strict_model)

    _, combined_state = _core._decode_verified_state(verified)
    if optimizer is not None:
        optimizer_state = combined_state.get("optimizer")
        if optimizer_state is None:
            raise CheckpointCompatibilityError(
                "optimizer was requested but checkpoint has no optimizer state"
            )
        _preflight_optimizer_state(optimizer, optimizer_state)
    if scheduler is not None:
        scheduler_state = combined_state.get("scheduler")
        if scheduler_state is None:
            raise CheckpointCompatibilityError(
                "scheduler was requested but checkpoint has no scheduler state"
            )
        _preflight_stateful_object(scheduler, scheduler_state, label="scheduler")

    return _CORE_LOAD_VERIFIED_CHECKPOINT(
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
    directory: str | Path,
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
) -> LoadResult:
    verified = prepare_checkpoint_load(directory)
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


def _install_core_entrypoints() -> None:
    """Make direct core imports use the same hardened production load path."""

    _core.prepare_checkpoint_load = prepare_checkpoint_load
    _core.verify_checkpoint = verify_checkpoint
    _core.load_verified_checkpoint = load_verified_checkpoint
    _core.load_checkpoint = load_checkpoint


_install_core_entrypoints()
