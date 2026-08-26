"""Fail-closed checkpoint load hardening for resume-critical state.

This module is installed by :mod:`twelve_six.checkpoint` after the core module
loads. Keeping the guards here makes the security boundary auditable while
preserving the checkpoint-v1 on-disk format.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from . import core as _core

_ORIGINAL_VALIDATE_MANIFEST_IDENTITY = _core._validate_manifest_identity
_INSTALLED = False


def _validate_manifest_identity(identity: Any) -> None:
    """Validate hashes plus scalar/domain invariants required for safe resume."""

    _ORIGINAL_VALIDATE_MANIFEST_IDENTITY(identity)
    if not isinstance(identity, Mapping):
        raise _core.CheckpointIntegrityError("manifest identity must be a mapping")
    try:
        candidate = _core.CheckpointIdentity(
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
        candidate.validate()
    except (TypeError, ValueError) as exc:
        raise _core.CheckpointIntegrityError(
            f"manifest identity scalar invariant failed: {exc}"
        ) from exc


def _materialize_for_target(array: np.ndarray, target: Any) -> Any:
    """Require exact model tensor shape/dtype before moving bytes to a target."""

    if tuple(target.shape) != tuple(array.shape):
        raise _core.CheckpointCompatibilityError(
            f"shape mismatch: checkpoint {tuple(array.shape)} vs target {tuple(target.shape)}"
        )

    if isinstance(target, np.ndarray):
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
                    "dtype mismatch: BF16 targets require checkpoint uint16 bit representation"
                )
            tensor = torch.from_numpy(array.copy()).view(torch.bfloat16)
        else:
            try:
                tensor = torch.from_numpy(array.copy())
            except (TypeError, ValueError, RuntimeError) as exc:
                raise _core.CheckpointCompatibilityError(
                    f"checkpoint dtype {array.dtype} cannot materialize as a torch tensor"
                ) from exc
            if tensor.dtype != target.dtype:
                raise _core.CheckpointCompatibilityError(
                    f"dtype mismatch: checkpoint {tensor.dtype} vs target {target.dtype}"
                )
        return tensor.to(device=target.device)

    raise _core.CheckpointCompatibilityError(
        f"unsupported target tensor type {type(target)!r}"
    )


def _optimizer_state_error(path: str, detail: str) -> None:
    raise _core.CheckpointCompatibilityError(
        f"optimizer state incompatible at {path}: {detail}"
    )


def _validate_torch_optimizer_value(
    value: Any,
    *,
    param: Any,
    path: str,
    leaf_name: str | None,
    torch: Any,
) -> None:
    """Validate tensor leaves PyTorch itself may otherwise cast or accept."""

    if torch.is_tensor(value):
        if leaf_name == "step":
            if value.numel() != 1:
                _optimizer_state_error(path, "step tensor must contain exactly one value")
            try:
                scalar = value.detach().cpu().item()
            except (RuntimeError, ValueError) as exc:
                raise _core.CheckpointCompatibilityError(
                    f"optimizer state incompatible at {path}: unreadable step tensor"
                ) from exc
            if isinstance(scalar, (int, float)) and scalar < 0:
                _optimizer_state_error(path, "step must be non-negative")
            return

        if value.ndim == 0:
            return
        if tuple(value.shape) != tuple(param.shape):
            _optimizer_state_error(
                path,
                f"tensor shape {tuple(value.shape)} does not match parameter {tuple(param.shape)}",
            )
        if param.is_floating_point() and value.is_floating_point() and value.dtype != param.dtype:
            _optimizer_state_error(
                path,
                f"tensor dtype {value.dtype} does not match parameter {param.dtype}",
            )
        if param.is_complex() and value.is_complex() and value.dtype != param.dtype:
            _optimizer_state_error(
                path,
                f"tensor dtype {value.dtype} does not match parameter {param.dtype}",
            )
        return

    if isinstance(value, np.ndarray):
        _optimizer_state_error(path, "NumPy tensor leaf is invalid for a torch optimizer")

    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_torch_optimizer_value(
                item,
                param=param,
                path=f"{path}.{key}",
                leaf_name=str(key),
                torch=torch,
            )
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_torch_optimizer_value(
                item,
                param=param,
                path=f"{path}[{index}]",
                leaf_name=leaf_name,
                torch=torch,
            )


def _preflight_optimizer_state(optimizer: Any, state: Any) -> None:
    """Validate torch optimizer binding without mutating the live optimizer."""

    if not isinstance(state, Mapping):
        raise _core.CheckpointCompatibilityError(
            "checkpoint optimizer state must be a mapping"
        )
    if not hasattr(optimizer, "load_state_dict") or not hasattr(optimizer, "state_dict"):
        raise _core.CheckpointCompatibilityError(
            "optimizer must provide state_dict() and load_state_dict()"
        )

    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return
    if not isinstance(optimizer, torch.optim.Optimizer):
        return

    saved_groups = state.get("param_groups")
    saved_state = state.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_state, Mapping):
        raise _core.CheckpointCompatibilityError(
            "torch optimizer state requires list param_groups and mapping state"
        )

    live_groups = optimizer.param_groups
    if len(saved_groups) != len(live_groups):
        raise _core.CheckpointCompatibilityError(
            "optimizer parameter-group count differs from checkpoint"
        )

    saved_ids: list[int] = []
    live_params: list[Any] = []
    for group_index, (saved_group, live_group) in enumerate(
        zip(saved_groups, live_groups, strict=True)
    ):
        if not isinstance(saved_group, Mapping):
            _optimizer_state_error(
                f"param_groups[{group_index}]", "saved parameter group must be a mapping"
            )
        group_ids = saved_group.get("params")
        group_params = live_group.get("params")
        if not isinstance(group_ids, (list, tuple)) or not isinstance(
            group_params, (list, tuple)
        ):
            _optimizer_state_error(
                f"param_groups[{group_index}]", "params must be a sequence"
            )
        if len(group_ids) != len(group_params):
            _optimizer_state_error(
                f"param_groups[{group_index}]",
                "saved/live parameter counts differ",
            )
        names = saved_group.get("param_names")
        if names is not None and (
            not isinstance(names, (list, tuple)) or len(names) != len(group_ids)
        ):
            _optimizer_state_error(
                f"param_groups[{group_index}].param_names",
                "param_names must align one-to-one with params",
            )
        for saved_id, live_param in zip(group_ids, group_params, strict=True):
            if (
                not isinstance(saved_id, int)
                or isinstance(saved_id, bool)
                or saved_id < 0
            ):
                _optimizer_state_error(
                    f"param_groups[{group_index}].params",
                    "parameter IDs must be non-negative integers",
                )
            if not torch.is_tensor(live_param):
                _optimizer_state_error(
                    f"param_groups[{group_index}].params",
                    f"live optimizer parameter is not a tensor: {type(live_param)!r}",
                )
            saved_ids.append(saved_id)
            live_params.append(live_param)

    if len(set(saved_ids)) != len(saved_ids):
        raise _core.CheckpointCompatibilityError(
            "optimizer checkpoint contains duplicate parameter IDs"
        )
    id_map = dict(zip(saved_ids, live_params, strict=True))

    for saved_id, param_state in saved_state.items():
        if saved_id not in id_map:
            _optimizer_state_error(
                f"state[{saved_id!r}]",
                "state refers to a parameter ID absent from param_groups",
            )
        if not isinstance(param_state, Mapping):
            _optimizer_state_error(
                f"state[{saved_id!r}]", "per-parameter state must be a mapping"
            )
        param = id_map[saved_id]
        for key, value in param_state.items():
            _validate_torch_optimizer_value(
                value,
                param=param,
                path=f"state[{saved_id!r}].{key}",
                leaf_name=str(key),
                torch=torch,
            )


def _target_parameter_count(model: Any) -> int | None:
    if hasattr(model, "parameters"):
        try:
            return sum(int(parameter.numel()) for parameter in model.parameters())
        except (AttributeError, TypeError):
            pass
    if not hasattr(model, "state_dict"):
        return None
    try:
        state = model.state_dict()
    except Exception:
        return None
    if not isinstance(state, Mapping):
        return None
    count = 0
    for value in state.values():
        if isinstance(value, np.ndarray):
            count += int(value.size)
            continue
        if hasattr(value, "numel"):
            try:
                count += int(value.numel())
            except (AttributeError, TypeError):
                return None
        else:
            return None
    return count


def _assert_resume_identity(
    manifest: Mapping[str, Any],
    *,
    model: Any,
    strict_model: bool,
    expected_parameter_count: int | None,
    expected_step: int | None,
    expected_tokens_seen: int | None,
) -> None:
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise _core.CheckpointCompatibilityError("verified checkpoint identity is missing")

    checks = {
        "parameter_count": (expected_parameter_count, identity.get("parameter_count")),
        "step": (expected_step, identity.get("step")),
        "tokens_seen": (expected_tokens_seen, identity.get("tokens_seen")),
    }
    mismatches = {
        name: {"expected": expected, "actual": actual}
        for name, (expected, actual) in checks.items()
        if expected is not None and expected != actual
    }
    if mismatches:
        raise _core.CheckpointCompatibilityError(
            f"checkpoint resume identity mismatch: {mismatches}"
        )

    if strict_model:
        actual_count = _target_parameter_count(model)
        recorded_count = identity.get("parameter_count")
        if actual_count is not None and actual_count != recorded_count:
            raise _core.CheckpointCompatibilityError(
                "checkpoint parameter_count does not match the target model: "
                f"checkpoint={recorded_count}, target={actual_count}"
            )


def load_verified_checkpoint(
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
    expected_parameter_count: int | None = None,
    expected_step: int | None = None,
    expected_tokens_seen: int | None = None,
) -> Any:
    """Preflight all resume-critical compatibility before first live mutation."""

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
    _assert_resume_identity(
        manifest,
        model=model,
        strict_model=strict_model,
        expected_parameter_count=expected_parameter_count,
        expected_step=expected_step,
        expected_tokens_seen=expected_tokens_seen,
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
    if scheduler is not None and scheduler_state is None:
        raise _core.CheckpointCompatibilityError(
            "scheduler was requested but checkpoint has no scheduler state"
        )
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
        manifest=__import__("copy").deepcopy(manifest),
        trainer_state=combined_state.get("trainer", {}),
        rng_state=combined_state["rng"],
    )


def load_checkpoint(
    directory: Any,
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
    expected_parameter_count: int | None = None,
    expected_step: int | None = None,
    expected_tokens_seen: int | None = None,
) -> Any:
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
        expected_parameter_count=expected_parameter_count,
        expected_step=expected_step,
        expected_tokens_seen=expected_tokens_seen,
    )


def install_checkpoint_hardening() -> None:
    """Install fail-closed guards into the already-loaded checkpoint core."""

    global _INSTALLED
    if _INSTALLED:
        return
    _core._validate_manifest_identity = _validate_manifest_identity
    _core._materialize_for_target = _materialize_for_target
    _core._preflight_optimizer_state = _preflight_optimizer_state
    _core.load_verified_checkpoint = load_verified_checkpoint
    _core.load_checkpoint = load_checkpoint
    _INSTALLED = True
