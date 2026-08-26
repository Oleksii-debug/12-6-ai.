"""Fail-closed checkpoint compatibility preflight for D05.

This module layers semantic compatibility checks on top of the immutable-byte
verification implemented by :mod:`twelve_six.checkpoint.core`.  The checks are
intentionally completed before any live model, optimizer, scheduler, trainer,
or RNG state is mutated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import core as _core

CheckpointCompatibilityError = _core.CheckpointCompatibilityError
CheckpointIntegrityError = _core.CheckpointIntegrityError
LoadResult = _core.LoadResult
VerifiedCheckpoint = _core.VerifiedCheckpoint


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_loaded_identity(manifest: Mapping[str, Any]) -> None:
    """Re-apply scalar/container invariants that save-time identity enforces."""

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise CheckpointIntegrityError("manifest identity must be a mapping")
    if not isinstance(identity.get("model_spec"), Mapping) or not identity["model_spec"]:
        raise CheckpointIntegrityError("identity.model_spec must be a non-empty mapping")
    if not isinstance(identity.get("training_config"), Mapping) or not identity["training_config"]:
        raise CheckpointIntegrityError("identity.training_config must be a non-empty mapping")
    if not isinstance(identity.get("optimizer"), Mapping) or not identity["optimizer"]:
        raise CheckpointIntegrityError("identity.optimizer must be a non-empty mapping")
    scheduler = identity.get("scheduler")
    if scheduler is not None and (not isinstance(scheduler, Mapping) or not scheduler):
        raise CheckpointIntegrityError("identity.scheduler must be a non-empty mapping or null")

    parameter_count = identity.get("parameter_count")
    if (
        not isinstance(parameter_count, int)
        or isinstance(parameter_count, bool)
        or parameter_count <= 0
    ):
        raise CheckpointIntegrityError("identity.parameter_count must be a positive integer")
    if not _is_non_negative_int(identity.get("seed")):
        raise CheckpointIntegrityError("identity.seed must be a non-negative integer")
    precision = identity.get("precision")
    if not isinstance(precision, str) or not precision.strip():
        raise CheckpointIntegrityError("identity.precision must be a non-empty string")
    if not _is_non_negative_int(identity.get("step")):
        raise CheckpointIntegrityError("identity.step must be a non-negative integer")
    if not _is_non_negative_int(identity.get("tokens_seen")):
        raise CheckpointIntegrityError("identity.tokens_seen must be a non-negative integer")


def _numpy_dtype_for_torch_target(target: Any) -> np.dtype[Any]:
    if str(target.dtype) == "torch.bfloat16":
        return np.dtype(np.uint16)
    try:
        return target.detach().cpu().numpy().dtype
    except (AttributeError, TypeError) as exc:
        raise CheckpointCompatibilityError(
            f"cannot derive checkpoint dtype for target tensor {type(target)!r}"
        ) from exc


def _validate_model_tensor(array: np.ndarray, target: Any, *, name: str) -> None:
    if isinstance(target, np.ndarray):
        expected_dtype = target.dtype
        expected_shape = tuple(target.shape)
    else:
        cls = target.__class__
        if not (cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}):
            raise CheckpointCompatibilityError(
                f"unsupported target tensor type for {name!r}: {type(target)!r}"
            )
        expected_dtype = _numpy_dtype_for_torch_target(target)
        expected_shape = tuple(target.shape)

    if tuple(array.shape) != expected_shape:
        raise CheckpointCompatibilityError(
            f"shape mismatch for {name!r}: checkpoint {tuple(array.shape)} vs target {expected_shape}"
        )
    if array.dtype != expected_dtype:
        raise CheckpointCompatibilityError(
            f"dtype mismatch for {name!r}: checkpoint {array.dtype} vs target {expected_dtype}"
        )


def _preflight_model_state(
    model: Any, arrays: Mapping[str, np.ndarray], *, strict: bool
) -> None:
    if not hasattr(model, "state_dict"):
        raise CheckpointCompatibilityError("model must provide state_dict()")
    target_state = model.state_dict()
    if not isinstance(target_state, Mapping) or not target_state:
        raise CheckpointCompatibilityError("model.state_dict() must be a non-empty mapping")

    target_keys = set(target_state)
    source_keys = set(arrays)
    if strict and target_keys != source_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise CheckpointCompatibilityError(
            f"state_dict keys differ: missing={missing}, unexpected={unexpected}"
        )
    for name in target_keys & source_keys:
        _validate_model_tensor(arrays[name], target_state[name], name=str(name))


def _is_torch_tensor(value: Any) -> bool:
    cls = value.__class__
    return cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}


def _validate_torch_state_tensor(value: Any, parameter: Any, *, field: str) -> None:
    if not _is_torch_tensor(value):
        return
    if value.ndim == 0:
        if field == "step":
            try:
                scalar = float(value.detach().cpu().item())
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                raise CheckpointCompatibilityError("optimizer step tensor is invalid") from exc
            if not np.isfinite(scalar) or scalar < 0:
                raise CheckpointCompatibilityError("optimizer step must be finite and non-negative")
        return
    if tuple(value.shape) != tuple(parameter.shape):
        raise CheckpointCompatibilityError(
            f"optimizer state {field!r} shape mismatch: checkpoint {tuple(value.shape)} "
            f"vs parameter {tuple(parameter.shape)}"
        )
    if value.dtype != parameter.dtype:
        raise CheckpointCompatibilityError(
            f"optimizer state {field!r} dtype mismatch: checkpoint {value.dtype} "
            f"vs parameter {parameter.dtype}"
        )


def _preflight_torch_optimizer_state(optimizer: Any, state: Mapping[str, Any]) -> None:
    optimizer_name = optimizer.__class__.__name__
    if optimizer_name not in {"AdamW", "SGD"}:
        raise CheckpointCompatibilityError(
            "fail-closed optimizer resume preflight currently supports only AdamW and SGD; "
            f"got {optimizer.__class__.__module__}.{optimizer_name}"
        )

    source_state = state.get("state")
    source_groups = state.get("param_groups")
    if not isinstance(source_state, Mapping) or not isinstance(source_groups, Sequence):
        raise CheckpointCompatibilityError(
            "torch optimizer checkpoint must contain mapping 'state' and sequence 'param_groups'"
        )
    live_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(live_groups, Sequence) or len(source_groups) != len(live_groups):
        raise CheckpointCompatibilityError("optimizer parameter-group count mismatch")

    id_to_parameter: dict[Any, Any] = {}
    for group_index, (source_group, live_group) in enumerate(zip(source_groups, live_groups, strict=True)):
        if not isinstance(source_group, Mapping) or not isinstance(live_group, Mapping):
            raise CheckpointCompatibilityError("optimizer parameter group must be a mapping")
        source_ids = source_group.get("params")
        live_parameters = live_group.get("params")
        if not isinstance(source_ids, Sequence) or not isinstance(live_parameters, Sequence):
            raise CheckpointCompatibilityError("optimizer parameter group params must be sequences")
        if len(source_ids) != len(live_parameters):
            raise CheckpointCompatibilityError(
                f"optimizer parameter count mismatch in group {group_index}: "
                f"checkpoint {len(source_ids)} vs target {len(live_parameters)}"
            )
        for source_id, parameter in zip(source_ids, live_parameters, strict=True):
            if source_id in id_to_parameter:
                raise CheckpointCompatibilityError(
                    f"optimizer checkpoint reuses parameter id {source_id!r}"
                )
            id_to_parameter[source_id] = parameter

    unknown_state_ids = set(source_state) - set(id_to_parameter)
    if unknown_state_ids:
        raise CheckpointCompatibilityError(
            f"optimizer state references unknown parameter ids: {sorted(unknown_state_ids, key=repr)}"
        )

    for parameter_id, parameter_state in source_state.items():
        if not isinstance(parameter_state, Mapping):
            raise CheckpointCompatibilityError(
                f"optimizer state for parameter id {parameter_id!r} must be a mapping"
            )
        parameter = id_to_parameter[parameter_id]
        for field, value in parameter_state.items():
            _validate_torch_state_tensor(value, parameter, field=str(field))


def _preflight_generic_tensor_tree(
    checkpoint_value: Any, target_value: Any, *, path: str
) -> None:
    if isinstance(checkpoint_value, np.ndarray):
        if not isinstance(target_value, np.ndarray):
            raise CheckpointCompatibilityError(
                f"optimizer tensor {path} has no compatible initialized target state"
            )
        if checkpoint_value.shape != target_value.shape:
            raise CheckpointCompatibilityError(
                f"optimizer state {path} shape mismatch: checkpoint {checkpoint_value.shape} "
                f"vs target {target_value.shape}"
            )
        if checkpoint_value.dtype != target_value.dtype:
            raise CheckpointCompatibilityError(
                f"optimizer state {path} dtype mismatch: checkpoint {checkpoint_value.dtype} "
                f"vs target {target_value.dtype}"
            )
        return
    if _is_torch_tensor(checkpoint_value):
        if not _is_torch_tensor(target_value):
            raise CheckpointCompatibilityError(
                f"optimizer tensor {path} has no compatible initialized target state"
            )
        if checkpoint_value.shape != target_value.shape or checkpoint_value.dtype != target_value.dtype:
            raise CheckpointCompatibilityError(
                f"optimizer state {path} tensor metadata mismatch"
            )
        return
    if isinstance(checkpoint_value, Mapping):
        if not isinstance(target_value, Mapping):
            return
        for key, value in checkpoint_value.items():
            if key in target_value:
                _preflight_generic_tensor_tree(value, target_value[key], path=f"{path}.{key}")
        return
    if isinstance(checkpoint_value, (list, tuple)) and isinstance(target_value, (list, tuple)):
        for index, (value, target) in enumerate(zip(checkpoint_value, target_value)):
            _preflight_generic_tensor_tree(value, target, path=f"{path}[{index}]")


def preflight_optimizer_state(optimizer: Any, state: Any) -> None:
    """Validate optimizer state metadata before any live checkpoint mutation."""

    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError("optimizer checkpoint state must be a mapping")
    module_name = optimizer.__class__.__module__
    if module_name.startswith("torch.optim"):
        _preflight_torch_optimizer_state(optimizer, state)
        return
    if not hasattr(optimizer, "state_dict"):
        raise CheckpointCompatibilityError("optimizer must provide state_dict()")
    target_state = optimizer.state_dict()
    if not isinstance(target_state, Mapping):
        raise CheckpointCompatibilityError("optimizer.state_dict() must be a mapping")
    _preflight_generic_tensor_tree(state, target_state, path="optimizer")


def preflight_trainer_state(trainer: Any, trainer_state: Any) -> None:
    """Validate D02 trainer counters and optimizer tensors before model mutation."""

    if not isinstance(trainer_state, Mapping):
        raise CheckpointCompatibilityError("trainer checkpoint state must be a mapping")
    for field in ("micro_step", "optimizer_step", "tokens_seen"):
        if not _is_non_negative_int(trainer_state.get(field)):
            raise CheckpointCompatibilityError(
                f"trainer {field} must be a non-negative integer"
            )
    checkpoint_optimizer = trainer_state.get("optimizer")
    live_optimizer = getattr(trainer, "optimizer", None)
    if live_optimizer is None or checkpoint_optimizer is None:
        raise CheckpointCompatibilityError("trainer checkpoint optimizer state is missing")
    preflight_optimizer_state(live_optimizer, checkpoint_optimizer)


def _decode_and_preflight(
    verified: VerifiedCheckpoint,
    *,
    model: Any,
    optimizer: Any | None,
    strict_model: bool,
) -> Mapping[str, Any]:
    arrays, combined_state = _core._decode_verified_state(verified)
    _preflight_model_state(model, arrays, strict=strict_model)
    if optimizer is not None:
        optimizer_state = combined_state.get("optimizer")
        if optimizer_state is None:
            raise CheckpointCompatibilityError(
                "optimizer was requested but checkpoint has no optimizer state"
            )
        preflight_optimizer_state(optimizer, optimizer_state)
    return combined_state


def prepare_checkpoint_load(directory: str | Path) -> VerifiedCheckpoint:
    verified = _core.prepare_checkpoint_load(directory)
    _validate_loaded_identity(verified.manifest)
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
    expected_step: int | None = None,
    expected_tokens_seen: int | None = None,
) -> LoadResult:
    """Semantically preflight one verified checkpoint, then delegate mutation."""

    manifest = verified.manifest
    _validate_loaded_identity(manifest)
    identity = manifest["identity"]
    counter_expectations = {
        "step": expected_step,
        "tokens_seen": expected_tokens_seen,
    }
    mismatches = {
        key: {"expected": expected, "actual": identity.get(key)}
        for key, expected in counter_expectations.items()
        if expected is not None and identity.get(key) != expected
    }
    if mismatches:
        raise CheckpointCompatibilityError(
            f"checkpoint counter identity mismatch: {mismatches}"
        )

    _decode_and_preflight(
        verified,
        model=model,
        optimizer=optimizer,
        strict_model=strict_model,
    )
    return _core.load_verified_checkpoint(
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
    expected_step: int | None = None,
    expected_tokens_seen: int | None = None,
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
        expected_step=expected_step,
        expected_tokens_seen=expected_tokens_seen,
    )
