"""Fail-closed D05 checkpoint preflight hardening.

This module tightens checkpoint-v1 without changing the on-disk format:

* model tensors must match target dtype exactly, except the established BF16
  raw-uint16 representation;
* manifest scalar/counter invariants are enforced during verification, not only
  after a load has begun; and
* optimizer state is validated before live model mutation. PyTorch optimizers
  receive parameter-aware geometry checks while generic state-dict optimizers
  retain compatibility through structural validation plus a detached load probe.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from . import core as _core

_ORIGINAL_LOAD_VERIFIED_CHECKPOINT = _core.load_verified_checkpoint
_ORIGINAL_VALIDATE_MANIFEST_IDENTITY = _core._validate_manifest_identity
_INSTALLED = False


def _is_torch_tensor(value: Any) -> bool:
    cls = value.__class__
    return cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}


def _strict_materialize_for_target(array: np.ndarray, target: Any) -> Any:
    """Materialize only representation-compatible checkpoint tensors."""

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


def _hardened_validate_manifest_identity(identity: Any) -> None:
    """Preserve hash checks and add semantic identity invariants at verify time."""

    _ORIGINAL_VALIDATE_MANIFEST_IDENTITY(identity)
    _validate_manifest_scalar_invariants({"identity": identity})


def _optimizer_state_dtype_compatible(state_tensor: Any, parameter: Any) -> bool:
    """Return whether a floating optimizer tensor has a safe parameter dtype."""

    if not _is_torch_tensor(state_tensor) or not _is_torch_tensor(parameter):
        return True
    if not state_tensor.is_floating_point() or not parameter.is_floating_point():
        return True
    if state_tensor.dtype == parameter.dtype:
        return True

    torch = __import__("torch")
    # FP32 master/optimizer state for lower-precision parameters is a supported
    # mixed-precision pattern. Other silent floating dtype changes are rejected.
    return parameter.dtype in {torch.float16, torch.bfloat16} and state_tensor.dtype == torch.float32


def _preflight_torch_optimizer_state(optimizer: Any, state: Mapping[str, Any]) -> None:
    """Validate PyTorch optimizer geometry before ``load_state_dict``."""

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
            "checkpoint optimizer state contains unknown parameter ids: "
            f"{sorted(unexpected_ids, key=str)}"
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


def _validate_generic_state_shape(saved: Any, live: Any, *, path: str) -> None:
    """Validate generic state-dict tensor/array structure without comparing values."""

    if isinstance(live, np.ndarray):
        if not isinstance(saved, np.ndarray):
            raise _core.CheckpointCompatibilityError(f"{path} must be a NumPy array")
        if tuple(saved.shape) != tuple(live.shape):
            raise _core.CheckpointCompatibilityError(
                f"{path} shape mismatch: checkpoint {tuple(saved.shape)} vs target {tuple(live.shape)}"
            )
        if saved.dtype != live.dtype:
            raise _core.CheckpointCompatibilityError(
                f"{path} dtype mismatch: checkpoint {saved.dtype} vs target {live.dtype}"
            )
        return

    if _is_torch_tensor(live):
        if not _is_torch_tensor(saved):
            raise _core.CheckpointCompatibilityError(f"{path} must be a torch tensor")
        if tuple(saved.shape) != tuple(live.shape):
            raise _core.CheckpointCompatibilityError(
                f"{path} shape mismatch: checkpoint {tuple(saved.shape)} vs target {tuple(live.shape)}"
            )
        if saved.dtype != live.dtype:
            raise _core.CheckpointCompatibilityError(
                f"{path} dtype mismatch: checkpoint {saved.dtype} vs target {live.dtype}"
            )
        return

    if isinstance(live, Mapping):
        if not isinstance(saved, Mapping):
            raise _core.CheckpointCompatibilityError(f"{path} must be a mapping")
        if set(saved) != set(live):
            raise _core.CheckpointCompatibilityError(
                f"{path} keys mismatch: checkpoint {sorted(saved, key=str)} vs "
                f"target {sorted(live, key=str)}"
            )
        for key in live:
            _validate_generic_state_shape(saved[key], live[key], path=f"{path}.{key}")
        return

    if isinstance(live, (list, tuple)):
        if not isinstance(saved, type(live)) or len(saved) != len(live):
            raise _core.CheckpointCompatibilityError(f"{path} sequence structure mismatch")
        for index, (saved_item, live_item) in enumerate(zip(saved, live)):
            _validate_generic_state_shape(saved_item, live_item, path=f"{path}[{index}]")
        return

    # Scalar values are allowed to differ (for example LR at resume). Their
    # representation class still must be compatible with the live state owner.
    if live is not None and saved is not None and type(saved) is not type(live):
        raise _core.CheckpointCompatibilityError(
            f"{path} scalar type mismatch: checkpoint {type(saved).__name__} vs "
            f"target {type(live).__name__}"
        )


def _preflight_generic_optimizer_state(optimizer: Any, state: Mapping[str, Any]) -> None:
    if not hasattr(optimizer, "state_dict") or not hasattr(optimizer, "load_state_dict"):
        raise _core.CheckpointCompatibilityError(
            "optimizer must provide state_dict() and load_state_dict()"
        )
    try:
        live_state = optimizer.state_dict()
    except Exception as exc:
        raise _core.CheckpointCompatibilityError("target optimizer state_dict() failed") from exc
    _validate_generic_state_shape(state, live_state, path="optimizer")

    # Probe the public loader on a detached owner so validation cannot mutate the
    # live optimizer or model. This preserves compatibility with project-local
    # state-dict optimizers such as the NumPy MomentumSGD checkpoint fixture.
    try:
        probe = copy.deepcopy(optimizer)
        probe.load_state_dict(copy.deepcopy(state))
    except Exception as exc:
        raise _core.CheckpointCompatibilityError(
            "checkpoint optimizer state is rejected by a detached load probe"
        ) from exc


def _preflight_optimizer_state(optimizer: Any, state: Any) -> None:
    """Validate optimizer state before model mutation, preserving generic support."""

    if not isinstance(state, Mapping):
        raise _core.CheckpointCompatibilityError("checkpoint optimizer state must be a mapping")

    live_groups = getattr(optimizer, "param_groups", None)
    if isinstance(live_groups, list):
        _preflight_torch_optimizer_state(optimizer, state)
        return
    _preflight_generic_optimizer_state(optimizer, state)


def _probe_generic_trainer_state(trainer: Any, state: Mapping[str, Any]) -> None:
    try:
        probe = copy.deepcopy(trainer)
        probe.load_state_dict(copy.deepcopy(state))
    except Exception as exc:
        raise _core.CheckpointCompatibilityError(
            "checkpoint trainer state is rejected by a detached load probe"
        ) from exc


def preflight_trainer_state(trainer: Any, state: Any) -> None:
    """Validate trainer-owned resume state before model mutation."""

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
        _probe_generic_trainer_state(trainer, state)
        return

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
    """Run semantic D05 preflights before delegating to checkpoint-v1 loader."""

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
    _core._validate_manifest_identity = _hardened_validate_manifest_identity
    _core.load_verified_checkpoint = _hardened_load_verified_checkpoint
    _INSTALLED = True


__all__ = ["install_checkpoint_hardening", "preflight_trainer_state"]
