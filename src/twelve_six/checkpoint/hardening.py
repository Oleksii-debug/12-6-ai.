"""Fail-closed compatibility preflight for checkpoint-v1 loading.

This module hardens the D05 loader without changing the checkpoint wire format.
It is installed by :mod:`twelve_six.checkpoint` before public loader symbols are
exported. The checks are deliberately conservative: ambiguous optimizer state
is rejected before any live model, optimizer, scheduler, or RNG mutation.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import core as _core

_INSTALLED = False
_ORIGINAL_VALIDATE_MANIFEST_IDENTITY = _core._validate_manifest_identity


def _semantic_checkpoint_identity(identity: Mapping[str, Any]) -> _core.CheckpointIdentity:
    """Rebuild the public identity contract from a verified manifest record."""

    try:
        return _core.CheckpointIdentity(
            git_sha=identity.get("git_sha"),
            model_spec=identity.get("model_spec"),
            parameter_count=identity.get("parameter_count"),
            tokenizer_hash=identity.get("tokenizer_hash"),
            tokenizer_vocab_hash=identity.get("tokenizer_vocab_hash"),
            dataset_manifest_hash=identity.get("dataset_manifest_hash"),
            run_manifest_hash=identity.get("run_manifest_hash"),
            training_config=identity.get("training_config"),
            seed=identity.get("seed"),
            precision=identity.get("precision"),
            step=identity.get("step"),
            tokens_seen=identity.get("tokens_seen"),
            optimizer=identity.get("optimizer"),
            scheduler=identity.get("scheduler"),
            environment_lock_hash=identity.get("environment_lock_hash"),
        )
    except TypeError as exc:
        raise _core.CheckpointIntegrityError("manifest identity fields are invalid") from exc


def _validate_manifest_identity(identity: Any) -> None:
    """Validate hashes and all semantic scalar/container identity invariants."""

    _ORIGINAL_VALIDATE_MANIFEST_IDENTITY(identity)
    assert isinstance(identity, Mapping)  # established by the original validator
    try:
        _semantic_checkpoint_identity(identity).validate()
    except (TypeError, ValueError) as exc:
        raise _core.CheckpointIntegrityError(f"invalid checkpoint identity: {exc}") from exc


def _torch_target_storage_dtype(target: Any) -> np.dtype[Any]:
    torch = importlib.import_module("torch")
    if target.dtype == torch.bfloat16:
        return np.dtype(np.uint16)
    try:
        return target.detach().cpu().contiguous().numpy().dtype
    except (RuntimeError, TypeError) as exc:
        raise _core.CheckpointCompatibilityError(
            f"target dtype {target.dtype} has no supported SafeTensors NumPy representation"
        ) from exc


def _materialize_for_target(array: np.ndarray, target: Any) -> Any:
    """Require exact source storage dtype and shape; never silently cast weights."""

    source_shape = tuple(array.shape)
    if isinstance(target, np.ndarray):
        target_shape = tuple(target.shape)
        if source_shape != target_shape:
            raise _core.CheckpointCompatibilityError(
                f"shape mismatch: checkpoint {source_shape} vs target {target_shape}"
            )
        if np.dtype(array.dtype) != np.dtype(target.dtype):
            raise _core.CheckpointCompatibilityError(
                f"dtype mismatch: checkpoint {array.dtype} vs target {target.dtype}"
            )
        return array.copy()

    cls = target.__class__
    if cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}:
        expected_storage_dtype = _torch_target_storage_dtype(target)
        if np.dtype(array.dtype) != expected_storage_dtype:
            raise _core.CheckpointCompatibilityError(
                "dtype mismatch: checkpoint storage "
                f"{array.dtype} vs target {target.dtype} "
                f"(expected storage {expected_storage_dtype})"
            )
        torch = importlib.import_module("torch")
        tensor = torch.from_numpy(array.copy())
        if str(target.dtype) == "torch.bfloat16":
            tensor = tensor.view(torch.bfloat16)
        if tuple(tensor.shape) != tuple(target.shape):
            raise _core.CheckpointCompatibilityError(
                f"shape mismatch: checkpoint {tuple(tensor.shape)} vs target {tuple(target.shape)}"
            )
        if tensor.dtype != target.dtype:
            raise _core.CheckpointCompatibilityError(
                "dtype mismatch after materialization: checkpoint "
                f"{tensor.dtype} vs target {target.dtype}"
            )
        return tensor.to(device=target.device)

    raise _core.CheckpointCompatibilityError(f"unsupported target tensor type {type(target)!r}")


def _is_tensor(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    cls = value.__class__
    return cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}


def _tensor_shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def _tensor_numel(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.size)
    return int(value.numel())


def _walk_optimizer_state_tensors(value: Any, path: tuple[str, ...] = ()):
    if _is_tensor(value):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_optimizer_state_tensors(item, (*path, str(key)))
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            yield from _walk_optimizer_state_tensors(item, (*path, str(index)))


def _is_scalar_step_tensor(path: tuple[str, ...], value: Any) -> bool:
    if not path:
        return False
    key = path[-1].lower()
    return key in {"step", "steps"} and _tensor_numel(value) == 1


def _preflight_optimizer_state(optimizer: Any, state: Any) -> None:
    """Prove optimizer state compatibility on an isolated copy before mutation."""

    if not isinstance(state, Mapping):
        raise _core.CheckpointCompatibilityError("checkpoint optimizer state must be a mapping")
    if not hasattr(optimizer, "load_state_dict") or not hasattr(optimizer, "state_dict"):
        raise _core.CheckpointCompatibilityError(
            "optimizer must provide state_dict() and load_state_dict()"
        )

    checkpoint_groups = state.get("param_groups")
    checkpoint_state = state.get("state")
    live_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(checkpoint_groups, list) or not isinstance(checkpoint_state, Mapping):
        raise _core.CheckpointCompatibilityError(
            "checkpoint optimizer state must contain param_groups list and state mapping"
        )
    if not isinstance(live_groups, list) or len(checkpoint_groups) != len(live_groups):
        raise _core.CheckpointCompatibilityError("optimizer param-group count mismatch")

    parameter_map: dict[Any, Any] = {}
    for group_index, (checkpoint_group, live_group) in enumerate(
        zip(checkpoint_groups, live_groups, strict=True)
    ):
        if not isinstance(checkpoint_group, Mapping) or not isinstance(live_group, Mapping):
            raise _core.CheckpointCompatibilityError(
                f"optimizer param group {group_index} is structurally invalid"
            )
        checkpoint_params = checkpoint_group.get("params")
        live_params = live_group.get("params")
        if not isinstance(checkpoint_params, list) or not isinstance(live_params, list):
            raise _core.CheckpointCompatibilityError(
                f"optimizer param group {group_index} params must be lists"
            )
        if len(checkpoint_params) != len(live_params):
            raise _core.CheckpointCompatibilityError(
                f"optimizer param group {group_index} parameter count mismatch"
            )
        for checkpoint_param_id, live_param in zip(checkpoint_params, live_params, strict=True):
            if checkpoint_param_id in parameter_map:
                raise _core.CheckpointCompatibilityError(
                    f"optimizer checkpoint parameter id is duplicated: {checkpoint_param_id!r}"
                )
            parameter_map[checkpoint_param_id] = live_param

    unknown_state_ids = set(checkpoint_state) - set(parameter_map)
    if unknown_state_ids:
        raise _core.CheckpointCompatibilityError(
            "optimizer state references unknown parameter ids: "
            f"{sorted(map(str, unknown_state_ids))}"
        )

    for checkpoint_param_id, param_state in checkpoint_state.items():
        live_param = parameter_map[checkpoint_param_id]
        expected_shape = tuple(int(item) for item in live_param.shape)
        for path, tensor in _walk_optimizer_state_tensors(param_state):
            if _is_scalar_step_tensor(path, tensor):
                continue
            actual_shape = _tensor_shape(tensor)
            if actual_shape != expected_shape:
                location = ".".join(path) or "<tensor>"
                raise _core.CheckpointCompatibilityError(
                    "optimizer state tensor shape mismatch for parameter "
                    f"{checkpoint_param_id!r} at {location}: checkpoint {actual_shape} "
                    f"vs parameter {expected_shape}"
                )

    try:
        probe = copy.deepcopy(optimizer)
    except Exception as exc:
        raise _core.CheckpointCompatibilityError(
            "optimizer cannot be isolated for fail-closed checkpoint preflight"
        ) from exc
    try:
        probe.load_state_dict(copy.deepcopy(state))
    except Exception as exc:
        raise _core.CheckpointCompatibilityError(
            "checkpoint optimizer state cannot be loaded into an isolated optimizer"
        ) from exc


def _preflight_stateful_loader(target: Any, state: Any, *, label: str) -> None:
    if not hasattr(target, "load_state_dict"):
        raise _core.CheckpointCompatibilityError(f"{label} must provide load_state_dict()")
    try:
        probe = copy.deepcopy(target)
    except Exception as exc:
        raise _core.CheckpointCompatibilityError(
            f"{label} cannot be isolated for fail-closed checkpoint preflight"
        ) from exc
    try:
        probe.load_state_dict(copy.deepcopy(state))
    except Exception as exc:
        raise _core.CheckpointCompatibilityError(
            f"checkpoint {label} state cannot be loaded into an isolated target"
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
    """Fully preflight a verified snapshot before mutating any requested target."""

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

    optimizer_state = combined_state.get("optimizer")
    scheduler_state = combined_state.get("scheduler")
    if optimizer is not None:
        if optimizer_state is None:
            raise _core.CheckpointCompatibilityError(
                "optimizer was requested but checkpoint has no optimizer state"
            )
        _preflight_optimizer_state(optimizer, optimizer_state)
    if scheduler is not None:
        if scheduler_state is None:
            raise _core.CheckpointCompatibilityError(
                "scheduler was requested but checkpoint has no scheduler state"
            )
        _preflight_stateful_loader(scheduler, scheduler_state, label="scheduler")
    if restore_rng:
        _core._preflight_rng_state(combined_state["rng"])

    _core._apply_model_weights(model, materialized, strict_model)
    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler.load_state_dict(scheduler_state)
    if restore_rng:
        _core.restore_rng_state(combined_state["rng"])
    return _core.LoadResult(
        manifest=copy.deepcopy(manifest),
        trainer_state=combined_state.get("trainer", {}),
        rng_state=combined_state["rng"],
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
    """Install hardened D05 semantics on the normal package import path once."""

    global _INSTALLED
    if _INSTALLED:
        return
    _core._validate_manifest_identity = _validate_manifest_identity
    _core._materialize_for_target = _materialize_for_target
    _core.load_verified_checkpoint = load_verified_checkpoint
    _core.load_checkpoint = load_checkpoint
    _INSTALLED = True
