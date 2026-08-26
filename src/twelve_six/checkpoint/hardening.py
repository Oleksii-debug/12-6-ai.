"""Fail-closed D05 checkpoint preflight hardening.

This module is deliberately small and compatibility-oriented.  It tightens the
checkpoint-v1 load path without changing the on-disk format:

* model tensors must match target dtype exactly (except the existing BF16/raw
  uint16 representation);
* manifest scalar/counter invariants are re-validated at load time; and
* PyTorch optimizer tensor state is shape/dtype checked against the live
  parameter geometry before any model mutation.

``install_checkpoint_hardening`` patches the private materialization primitive
and verified-load entrypoint in :mod:`twelve_six.checkpoint.core`.  Importing the
``twelve_six.checkpoint`` package installs the hardening before public checkpoint
symbols and the trainer adapter are exported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from . import core as _core

_ORIGINAL_LOAD_VERIFIED_CHECKPOINT = _core.load_verified_checkpoint
_INSTALLED = False


def _is_torch_tensor(value: Any) -> bool:
    cls = value.__class__
    return cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}


def _strict_materialize_for_target(array: np.ndarray, target: Any) -> Any:
    """Materialize only representation-compatible checkpoint tensors.

    D05 is an integrity boundary, not a conversion API.  Silent dtype casts can
    turn corrupted bytes into apparently valid state, so dtype differences fail
    closed.  BF16 remains the one explicit representation exception because the
    checkpoint format stores BF16 tensor bits as NumPy uint16.
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
        torch = __import__("torch")
        if tuple(target.shape) != tuple(array.shape):
            raise _core.CheckpointCompatibilityError(
                f"shape mismatch: checkpoint {tuple(array.shape)} vs target {tuple(target.shape)}"
            )

        if str(target.dtype) == "torch.bfloat16":
            if array.dtype != np.uint16:
                raise _core.CheckpointCompatibilityError(
                    "dtype mismatch: BF16 target requires checkpoint uint16 raw-bit representation"
                )
            tensor = torch.from_numpy(array.copy()).view(torch.bfloat16)
            return tensor.to(device=target.device)

        expected_dtype = target.detach().cpu().numpy().dtype
        if array.dtype != expected_dtype:
            raise _core.CheckpointCompatibilityError(
                f"dtype mismatch: checkpoint {array.dtype} vs target {target.dtype}"
            )
        tensor = torch.from_numpy(array.copy())
        return tensor.to(device=target.device)

    raise _core.CheckpointCompatibilityError(f"unsupported target tensor type {type(target)!r}")


def _require_non_negative_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise _core.CheckpointIntegrityError(f"identity.{field} must be a non-negative integer")
    return value


def _validate_manifest_scalar_invariants(manifest: Mapping[str, Any]) -> None:
    """Validate scalar identity semantics that hashes alone cannot prove."""

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise _core.CheckpointIntegrityError("manifest identity must be a mapping")

    parameter_count = identity.get("parameter_count")
    if (
        not isinstance(parameter_count, int)
        or isinstance(parameter_count, bool)
        or parameter_count <= 0
    ):
        raise _core.CheckpointIntegrityError("identity.parameter_count must be a positive integer")

    _require_non_negative_int(identity.get("seed"), field="seed")
    _require_non_negative_int(identity.get("step"), field="step")
    _require_non_negative_int(identity.get("tokens_seen"), field="tokens_seen")

    precision = identity.get("precision")
    if not isinstance(precision, str) or not precision.strip():
        raise _core.CheckpointIntegrityError("identity.precision must be a non-empty string")

    for field in ("model_spec", "training_config", "optimizer"):
        value = identity.get(field)
        if not isinstance(value, Mapping) or not value:
            raise _core.CheckpointIntegrityError(f"identity.{field} must be a non-empty mapping")
    scheduler = identity.get("scheduler")
    if scheduler is not None and (not isinstance(scheduler, Mapping) or not scheduler):
        raise _core.CheckpointIntegrityError(
            "identity.scheduler must be a non-empty mapping or None"
        )


def _optimizer_state_dtype_compatible(state_tensor: Any, parameter: Any) -> bool:
    """Return whether a floating optimizer tensor has a safe parameter dtype."""

    if not _is_torch_tensor(state_tensor) or not _is_torch_tensor(parameter):
        return True
    if not state_tensor.is_floating_point() or not parameter.is_floating_point():
        return True
    if state_tensor.dtype == parameter.dtype:
        return True

    torch = __import__("torch")
    # FP32 optimizer/master state for lower-precision parameters is a supported
    # mixed-precision pattern.  Other silent dtype changes are rejected.
    return parameter.dtype in {torch.float16, torch.bfloat16} and state_tensor.dtype == torch.float32


def _preflight_optimizer_state(optimizer: Any, state: Any) -> None:
    """Validate optimizer state geometry before ``optimizer.load_state_dict``.

    PyTorch intentionally accepts some malformed state at load time and only
    fails on the next optimizer step.  We map serialized parameter IDs to the
    live parameter objects by param-group position, then reject non-scalar tensor
    state whose shape or floating dtype cannot be applied to that parameter.
    """

    if not isinstance(state, Mapping):
        raise _core.CheckpointCompatibilityError("checkpoint optimizer state must be a mapping")
    saved_groups = state.get("param_groups")
    saved_state = state.get("state")
    live_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(saved_groups, Sequence) or isinstance(saved_groups, (str, bytes)):
        raise _core.CheckpointCompatibilityError(
            "checkpoint optimizer param_groups must be a sequence"
        )
    if not isinstance(saved_state, Mapping):
        raise _core.CheckpointCompatibilityError("checkpoint optimizer state table must be a mapping")
    if not isinstance(live_groups, list):
        raise _core.CheckpointCompatibilityError("optimizer must expose param_groups")
    if len(saved_groups) != len(live_groups):
        raise _core.CheckpointCompatibilityError(
            "checkpoint optimizer param-group count differs from target optimizer"
        )

    parameter_by_saved_id: dict[Any, Any] = {}
    for group_index, (saved_group, live_group) in enumerate(zip(saved_groups, live_groups)):
        if not isinstance(saved_group, Mapping) or not isinstance(live_group, Mapping):
            raise _core.CheckpointCompatibilityError(
                f"optimizer param group {group_index} is structurally invalid"
            )
        saved_params = saved_group.get("params")
        live_params = live_group.get("params")
        if not isinstance(saved_params, Sequence) or isinstance(saved_params, (str, bytes)):
            raise _core.CheckpointCompatibilityError(
                f"checkpoint optimizer group {group_index} params must be a sequence"
            )
        if not isinstance(live_params, Sequence) or isinstance(live_params, (str, bytes)):
            raise _core.CheckpointCompatibilityError(
                f"target optimizer group {group_index} params must be a sequence"
            )
        if len(saved_params) != len(live_params):
            raise _core.CheckpointCompatibilityError(
                f"optimizer group {group_index} parameter count differs from target"
            )
        for saved_id, parameter in zip(saved_params, live_params):
            prior = parameter_by_saved_id.get(saved_id)
            if prior is not None and prior is not parameter:
                raise _core.CheckpointCompatibilityError(
                    f"checkpoint optimizer parameter id {saved_id!r} is reused inconsistently"
                )
            parameter_by_saved_id[saved_id] = parameter

    unexpected_ids = set(saved_state) - set(parameter_by_saved_id)
    if unexpected_ids:
        raise _core.CheckpointCompatibilityError(
            f"checkpoint optimizer state contains unknown parameter ids: {sorted(unexpected_ids, key=str)}"
        )

    for saved_id, parameter_state in saved_state.items():
        if not isinstance(parameter_state, Mapping):
            raise _core.CheckpointCompatibilityError(
                f"optimizer state for parameter {saved_id!r} must be a mapping"
            )
        parameter = parameter_by_saved_id[saved_id]
        parameter_shape = tuple(getattr(parameter, "shape", ()))
        for state_name, value in parameter_state.items():
            if _is_torch_tensor(value):
                # Scalar counters (for example Adam's step) are parameter-independent.
                if value.ndim == 0:
                    continue
                if tuple(value.shape) != parameter_shape:
                    raise _core.CheckpointCompatibilityError(
                        "optimizer tensor shape mismatch for "
                        f"parameter {saved_id!r} state {state_name!r}: "
                        f"checkpoint {tuple(value.shape)} vs target {parameter_shape}"
                    )
                if not _optimizer_state_dtype_compatible(value, parameter):
                    raise _core.CheckpointCompatibilityError(
                        "optimizer tensor dtype mismatch for "
                        f"parameter {saved_id!r} state {state_name!r}: "
                        f"checkpoint {value.dtype} vs target {getattr(parameter, 'dtype', None)}"
                    )
            elif isinstance(value, np.ndarray):
                if value.ndim == 0:
                    continue
                if tuple(value.shape) != parameter_shape:
                    raise _core.CheckpointCompatibilityError(
                        "optimizer array shape mismatch for "
                        f"parameter {saved_id!r} state {state_name!r}: "
                        f"checkpoint {tuple(value.shape)} vs target {parameter_shape}"
                    )


def preflight_trainer_state(trainer: Any, state: Any) -> None:
    """Validate 12-6 trainer-owned resume state before model mutation."""

    if is_dataclass(state) and not isinstance(state, type):
        state = asdict(state)
    if not isinstance(state, Mapping):
        raise _core.CheckpointCompatibilityError("checkpoint trainer state must be a mapping")

    for field in ("micro_step", "optimizer_step", "tokens_seen"):
        value = state.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise _core.CheckpointCompatibilityError(
                f"trainer {field} must be a non-negative integer"
            )

    live_config = getattr(trainer, "config", None)
    if is_dataclass(live_config) and not isinstance(live_config, type):
        live_config = asdict(live_config)
    elif hasattr(live_config, "model_dump"):
        live_config = live_config.model_dump(mode="python")
    checkpoint_config = state.get("config")
    if live_config is not None and checkpoint_config != live_config:
        raise _core.CheckpointCompatibilityError("trainer config mismatch; refusing unsafe resume")

    if isinstance(live_config, Mapping):
        accumulation = live_config.get("gradient_accumulation_steps")
        max_steps = live_config.get("max_steps")
        if isinstance(accumulation, int) and accumulation > 0:
            expected_micro = state["optimizer_step"] * accumulation
            if state["micro_step"] != expected_micro:
                raise _core.CheckpointCompatibilityError(
                    "checkpoint is not at a complete committed accumulation boundary: "
                    f"micro_step={state['micro_step']}, expected={expected_micro}"
                )
        if isinstance(max_steps, int) and state["optimizer_step"] > max_steps:
            raise _core.CheckpointCompatibilityError(
                "checkpoint optimizer_step exceeds configured max_steps"
            )

    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        raise _core.CheckpointCompatibilityError("trainer must expose optimizer for resume preflight")
    _preflight_optimizer_state(optimizer, state.get("optimizer"))

    scheduler = getattr(trainer, "scheduler", None)
    checkpoint_scheduler = state.get("scheduler")
    if (checkpoint_scheduler is None) != (scheduler is None):
        raise _core.CheckpointCompatibilityError("scheduler state/config mismatch")


def _hardened_load_verified_checkpoint(
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
    """Run semantic D05 preflight before delegating to checkpoint-v1 loader."""

    manifest = verified.manifest
    _validate_manifest_scalar_invariants(manifest)
    _, combined_state = _core._decode_verified_state(verified)
    if optimizer is not None:
        optimizer_state = combined_state.get("optimizer")
        if optimizer_state is None:
            raise _core.CheckpointCompatibilityError(
                "optimizer was requested but checkpoint has no optimizer state"
            )
        _preflight_optimizer_state(optimizer, optimizer_state)

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


def install_checkpoint_hardening() -> None:
    """Install D05 semantic preflights exactly once for all package consumers."""

    global _INSTALLED
    if _INSTALLED:
        return
    _core._materialize_for_target = _strict_materialize_for_target
    _core.load_verified_checkpoint = _hardened_load_verified_checkpoint
    _INSTALLED = True


__all__ = ["install_checkpoint_hardening", "preflight_trainer_state"]
