"""Checkpoint v1 facade with fail-closed load hardening.

The original checkpoint-v1 implementation is retained in ``_core_impl`` so
this change can harden corruption handling without rewriting the established
serialization format. Public APIs remain available from this module.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import _core_impl as _impl


# Preserve the complete checkpoint-v1 API surface, including constants and
# internal helpers used by the existing tests, then override the vulnerable
# load paths below.
for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)


_ORIGINAL_VALIDATE_MANIFEST_IDENTITY = _impl._validate_manifest_identity
_SCALAR_OPTIMIZER_STATE_KEYS = frozenset(
    {"step", "steps", "step_count", "num_steps", "num_updates"}
)


def _require_manifest_int(
    identity: Mapping[str, Any],
    field: str,
    *,
    minimum: int,
) -> int:
    value = identity.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        bound = "positive" if minimum == 1 else "non-negative"
        raise CheckpointIntegrityError(f"identity.{field} must be a {bound} integer")
    return value


def _validate_manifest_identity(identity: Any) -> None:
    """Validate hashed identity payloads and scalar invariants on load."""

    _ORIGINAL_VALIDATE_MANIFEST_IDENTITY(identity)
    assert isinstance(identity, Mapping)

    for field in ("model_spec", "training_config", "optimizer"):
        value = identity.get(field)
        if not isinstance(value, Mapping) or not value:
            raise CheckpointIntegrityError(f"identity.{field} must be a non-empty mapping")
    scheduler = identity.get("scheduler")
    if scheduler is not None and (not isinstance(scheduler, Mapping) or not scheduler):
        raise CheckpointIntegrityError("identity.scheduler must be a non-empty mapping or null")
    environment = identity.get("environment")
    if not isinstance(environment, Mapping) or not environment:
        raise CheckpointIntegrityError("identity.environment must be a non-empty mapping")

    _require_manifest_int(identity, "parameter_count", minimum=1)
    _require_manifest_int(identity, "seed", minimum=0)
    _require_manifest_int(identity, "step", minimum=0)
    _require_manifest_int(identity, "tokens_seen", minimum=0)
    precision = identity.get("precision")
    if not isinstance(precision, str) or not precision.strip():
        raise CheckpointIntegrityError("identity.precision must be a non-empty string")


def _materialize_for_target(array: np.ndarray, target: Any) -> Any:
    """Materialize a checkpoint tensor only when shape and dtype are exact."""

    if isinstance(target, np.ndarray):
        if tuple(target.shape) != tuple(array.shape):
            raise CheckpointCompatibilityError(
                f"shape mismatch: checkpoint {tuple(array.shape)} vs target {tuple(target.shape)}"
            )
        if np.dtype(array.dtype) != np.dtype(target.dtype):
            raise CheckpointCompatibilityError(
                f"dtype mismatch: checkpoint {array.dtype} vs target {target.dtype}"
            )
        return array.copy()

    cls = target.__class__
    if cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}:
        torch = _impl.importlib.import_module("torch")
        if tuple(target.shape) != tuple(array.shape):
            raise CheckpointCompatibilityError(
                f"shape mismatch: checkpoint {tuple(array.shape)} vs target {tuple(target.shape)}"
            )

        if str(target.dtype) == "torch.bfloat16":
            if np.dtype(array.dtype) != np.dtype(np.uint16):
                raise CheckpointCompatibilityError(
                    "dtype mismatch: bfloat16 target requires checkpoint uint16 bit storage"
                )
            tensor = torch.from_numpy(array.copy()).view(torch.bfloat16)
        else:
            try:
                tensor = torch.from_numpy(array.copy())
            except (TypeError, ValueError, RuntimeError) as exc:
                raise CheckpointCompatibilityError(
                    f"checkpoint dtype {array.dtype} cannot be materialized for torch target"
                ) from exc
            if tensor.dtype != target.dtype:
                raise CheckpointCompatibilityError(
                    f"dtype mismatch: checkpoint {tensor.dtype} vs target {target.dtype}"
                )

        return tensor.to(device=target.device)

    raise CheckpointCompatibilityError(f"unsupported target tensor type {type(target)!r}")


def _is_tensor_state(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    cls = value.__class__
    return cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}


def _tensor_shape(value: Any) -> tuple[int, ...]:
    return tuple(value.shape)


def _tensor_dtype(value: Any) -> str:
    return str(value.dtype)


def _tensor_numel(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.size)
    return int(value.numel())


def _preflight_generic_state_target(target: Any, state: Any, *, label: str) -> None:
    """Exercise a generic state loader on an isolated copy before live mutation."""

    if not hasattr(target, "load_state_dict"):
        raise CheckpointCompatibilityError(f"{label} must provide load_state_dict()")
    try:
        probe = copy.deepcopy(target)
        probe.load_state_dict(copy.deepcopy(state))
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint {label} state is incompatible with the requested target"
        ) from exc


def _preflight_torch_optimizer_state(optimizer: Any, state: Any) -> None:
    """Validate PyTorch optimizer state shape/dtype semantics without mutation."""

    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError("checkpoint optimizer state must be a mapping")
    checkpoint_groups = state.get("param_groups")
    checkpoint_state = state.get("state")
    live_groups = getattr(optimizer, "param_groups", None)
    live_state = getattr(optimizer, "state", None)
    if not isinstance(checkpoint_groups, (list, tuple)) or not isinstance(checkpoint_state, Mapping):
        raise CheckpointCompatibilityError(
            "checkpoint optimizer state must contain param_groups and state"
        )
    if not isinstance(live_groups, (list, tuple)) or not isinstance(live_state, Mapping):
        raise CheckpointCompatibilityError(
            "requested torch optimizer does not expose compatible param_groups/state"
        )
    if len(checkpoint_groups) != len(live_groups):
        raise CheckpointCompatibilityError(
            "optimizer param-group count differs between checkpoint and target"
        )

    checkpoint_ids: list[Any] = []
    bindings: list[tuple[Any, Any]] = []
    for group_index, (checkpoint_group, live_group) in enumerate(
        zip(checkpoint_groups, live_groups, strict=True)
    ):
        if not isinstance(checkpoint_group, Mapping) or not isinstance(live_group, Mapping):
            raise CheckpointCompatibilityError(
                f"optimizer param group {group_index} must be a mapping"
            )
        checkpoint_params = checkpoint_group.get("params")
        live_params = live_group.get("params")
        if not isinstance(checkpoint_params, (list, tuple)) or not isinstance(
            live_params, (list, tuple)
        ):
            raise CheckpointCompatibilityError(
                f"optimizer param group {group_index} has invalid params"
            )
        if len(checkpoint_params) != len(live_params):
            raise CheckpointCompatibilityError(
                f"optimizer param group {group_index} parameter count differs"
            )
        for checkpoint_id, live_param in zip(checkpoint_params, live_params, strict=True):
            checkpoint_ids.append(checkpoint_id)
            bindings.append((checkpoint_id, live_param))

    try:
        unique_ids = set(checkpoint_ids)
    except TypeError as exc:
        raise CheckpointCompatibilityError("optimizer parameter identifiers must be hashable") from exc
    if len(unique_ids) != len(checkpoint_ids):
        raise CheckpointCompatibilityError("optimizer parameter identifiers must be unique")
    extra_state_ids = set(checkpoint_state) - unique_ids
    if extra_state_ids:
        raise CheckpointCompatibilityError(
            f"optimizer state contains unbound parameter ids: {sorted(extra_state_ids, key=str)}"
        )

    for checkpoint_id, live_param in bindings:
        parameter_state = checkpoint_state.get(checkpoint_id, {})
        if not isinstance(parameter_state, Mapping):
            raise CheckpointCompatibilityError(
                f"optimizer state for parameter {checkpoint_id!r} must be a mapping"
            )
        reference_state = live_state.get(live_param, {})
        if not isinstance(reference_state, Mapping):
            reference_state = {}
        parameter_shape = tuple(live_param.shape)

        for state_name, state_value in parameter_state.items():
            if not _is_tensor_state(state_value):
                continue
            reference_value = reference_state.get(state_name)
            if _is_tensor_state(reference_value):
                if _tensor_shape(state_value) != _tensor_shape(reference_value):
                    raise CheckpointCompatibilityError(
                        "optimizer state shape mismatch for "
                        f"{state_name!r}: checkpoint {_tensor_shape(state_value)} vs "
                        f"target {_tensor_shape(reference_value)}"
                    )
                if _tensor_dtype(state_value) != _tensor_dtype(reference_value):
                    raise CheckpointCompatibilityError(
                        "optimizer state dtype mismatch for "
                        f"{state_name!r}: checkpoint {_tensor_dtype(state_value)} vs "
                        f"target {_tensor_dtype(reference_value)}"
                    )
                continue

            state_key = str(state_name).lower()
            if state_key in _SCALAR_OPTIMIZER_STATE_KEYS:
                if _tensor_numel(state_value) != 1:
                    raise CheckpointCompatibilityError(
                        f"optimizer scalar state {state_name!r} must contain exactly one value"
                    )
                continue
            if _tensor_shape(state_value) != parameter_shape:
                raise CheckpointCompatibilityError(
                    "optimizer state shape mismatch for "
                    f"{state_name!r}: checkpoint {_tensor_shape(state_value)} vs "
                    f"parameter {parameter_shape}"
                )


def _preflight_optimizer_state(optimizer: Any, state: Any) -> None:
    """Fail closed on optimizer corruption before model/optimizer mutation."""

    module = optimizer.__class__.__module__
    if module.startswith("torch.optim"):
        _preflight_torch_optimizer_state(optimizer, state)
        return
    _preflight_generic_state_target(optimizer, state, label="optimizer")


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
    """Preflight every requested state component before the first live mutation."""

    manifest = verified.manifest
    _impl.assert_identity(
        manifest,
        git_sha=expected_git_sha,
        model_spec_hash=expected_model_spec_hash,
        tokenizer_hash=expected_tokenizer_hash,
        tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
        dataset_manifest_hash=expected_dataset_manifest_hash,
        run_manifest_hash=expected_run_manifest_hash,
    )
    arrays, combined_state = _impl._decode_verified_state(verified)
    materialized = _impl._prepare_model_weights(model, arrays, strict_model)

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
        _preflight_generic_state_target(scheduler, scheduler_state, label="scheduler")
    if restore_rng:
        _impl._preflight_rng_state(combined_state["rng"])

    # Every integrity, identity, model, optimizer, scheduler and RNG check above
    # completes before the first live target is mutated.
    _impl._apply_model_weights(model, materialized, strict_model)
    if optimizer is not None:
        optimizer.load_state_dict(combined_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(combined_state["scheduler"])
    if restore_rng:
        _impl.restore_rng_state(combined_state["rng"])
    return LoadResult(
        manifest=copy.deepcopy(manifest),
        trainer_state=combined_state.get("trainer", {}),
        rng_state=combined_state["rng"],
    )


def load_checkpoint(
    directory: str | _impl.Path,
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
    """Snapshot+verify exact checkpoint bytes, then restore requested targets."""

    verified = _impl.prepare_checkpoint_load(directory)
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


# Existing implementation functions resolve their globals in _core_impl at
# runtime. Installing the validators here hardens verify/save/preflight callers
# there as well, while the public load entry points use the transactional facade.
_impl._validate_manifest_identity = _validate_manifest_identity
_impl._materialize_for_target = _materialize_for_target
_impl.load_verified_checkpoint = load_verified_checkpoint
_impl.load_checkpoint = load_checkpoint
