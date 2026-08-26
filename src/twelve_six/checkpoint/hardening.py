"""Fail-closed D05 checkpoint compatibility hardening.

This module is installed by :mod:`twelve_six.checkpoint` after the v1 core
module is imported. It keeps the checkpoint wire format unchanged while
closing compatibility gaps that must be rejected before the first live model,
optimizer, scheduler, or RNG mutation.
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


def _iter_tensor_leaves(value: Any, path: str):
    if _is_torch_tensor(value) or isinstance(value, np.ndarray):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_tensor_leaves(item, f"{path}.{key!r}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_tensor_leaves(item, f"{path}[{index}]")


def _validate_optimizer_tensor(core: Any, tensor: Any, parameter: Any, path: str) -> None:
    if _is_torch_tensor(tensor) and _is_torch_tensor(parameter):
        # Scalar tensor slots such as Adam/AdamW ``step`` are metadata, not
        # parameter-shaped optimizer buffers.
        if tensor.ndim == 0:
            return
        if tuple(tensor.shape) != tuple(parameter.shape):
            raise core.CheckpointCompatibilityError(
                "optimizer tensor shape mismatch before target mutation: "
                f"{path} checkpoint={tuple(tensor.shape)} target={tuple(parameter.shape)}"
            )
        if tensor.dtype != parameter.dtype:
            raise core.CheckpointCompatibilityError(
                "optimizer tensor dtype mismatch before target mutation: "
                f"{path} checkpoint={tensor.dtype} target={parameter.dtype}"
            )
        return
    if isinstance(tensor, np.ndarray) and isinstance(parameter, np.ndarray):
        if tensor.ndim == 0:
            return
        if tuple(tensor.shape) != tuple(parameter.shape):
            raise core.CheckpointCompatibilityError(
                "optimizer array shape mismatch before target mutation: "
                f"{path} checkpoint={tuple(tensor.shape)} target={tuple(parameter.shape)}"
            )
        if tensor.dtype != parameter.dtype:
            raise core.CheckpointCompatibilityError(
                "optimizer array dtype mismatch before target mutation: "
                f"{path} checkpoint={tensor.dtype} target={parameter.dtype}"
            )
        return
    # PyTorch optimizers are the production D05 target. A tensor state paired
    # with an unrecognised parameter representation is not safe to reinterpret.
    raise core.CheckpointCompatibilityError(
        f"optimizer tensor target type is unsupported at {path}: {type(parameter)!r}"
    )


def _preflight_optimizer_state(core: Any, optimizer: Any, state: Any) -> None:
    if not isinstance(state, Mapping):
        raise core.CheckpointCompatibilityError("checkpoint optimizer state must be a mapping")
    saved_state = state.get("state")
    saved_groups = state.get("param_groups")
    target_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(saved_state, Mapping) or not isinstance(saved_groups, list):
        raise core.CheckpointCompatibilityError(
            "checkpoint optimizer state must contain mapping state and list param_groups"
        )
    if not isinstance(target_groups, list) or len(saved_groups) != len(target_groups):
        raise core.CheckpointCompatibilityError(
            "checkpoint optimizer parameter-group count differs from target optimizer"
        )

    saved_to_live: dict[Any, Any] = {}
    for group_index, (saved_group, target_group) in enumerate(zip(saved_groups, target_groups)):
        if not isinstance(saved_group, Mapping) or not isinstance(target_group, Mapping):
            raise core.CheckpointCompatibilityError(
                f"optimizer parameter group {group_index} is structurally invalid"
            )
        saved_params = saved_group.get("params")
        target_params = target_group.get("params")
        if not isinstance(saved_params, list) or not isinstance(target_params, list):
            raise core.CheckpointCompatibilityError(
                f"optimizer parameter group {group_index} must contain parameter lists"
            )
        if len(saved_params) != len(target_params):
            raise core.CheckpointCompatibilityError(
                "checkpoint optimizer parameter count differs from target optimizer in "
                f"group {group_index}"
            )
        for saved_id, live_parameter in zip(saved_params, target_params):
            if saved_id in saved_to_live:
                raise core.CheckpointCompatibilityError(
                    f"checkpoint optimizer parameter id is duplicated: {saved_id!r}"
                )
            saved_to_live[saved_id] = live_parameter

    unknown_state_ids = [saved_id for saved_id in saved_state if saved_id not in saved_to_live]
    if unknown_state_ids:
        raise core.CheckpointCompatibilityError(
            "checkpoint optimizer state contains unknown parameter ids: "
            f"{unknown_state_ids!r}"
        )

    for saved_id, parameter_state in saved_state.items():
        if not isinstance(parameter_state, Mapping):
            raise core.CheckpointCompatibilityError(
                f"checkpoint optimizer state for parameter {saved_id!r} must be a mapping"
            )
        parameter = saved_to_live[saved_id]
        for path, tensor in _iter_tensor_leaves(
            parameter_state, f"optimizer.state[{saved_id!r}]"
        ):
            _validate_optimizer_tensor(core, tensor, parameter, path)

    # Exercise the optimizer's own loader on an isolated copy as an additional
    # generic compatibility check. This happens before any live model state is
    # changed and catches group/schema constraints owned by the optimizer class.
    try:
        probe = copy.deepcopy(optimizer)
        probe.load_state_dict(copy.deepcopy(state))
    except Exception as exc:
        raise core.CheckpointCompatibilityError(
            "checkpoint optimizer state is rejected by an isolated target optimizer"
        ) from exc


def _preflight_stateful_target(core: Any, target: Any, state: Any, *, label: str) -> None:
    try:
        probe = copy.deepcopy(target)
        probe.load_state_dict(copy.deepcopy(state))
    except Exception as exc:
        raise core.CheckpointCompatibilityError(
            f"checkpoint {label} state is rejected before target mutation"
        ) from exc


def install_checkpoint_hardening(core: Any) -> None:
    """Install D05 fail-closed validators into the already-imported core module."""

    if getattr(core, "_D05_HARDENING_INSTALLED", False):
        return

    original_validate_manifest_identity = core._validate_manifest_identity
    original_assert_identity = core.assert_identity

    def hardened_materialize_for_target(array: np.ndarray, target: Any) -> Any:
        if isinstance(target, np.ndarray):
            if tuple(target.shape) != tuple(array.shape):
                raise core.CheckpointCompatibilityError(
                    f"shape mismatch: checkpoint {tuple(array.shape)} vs target {tuple(target.shape)}"
                )
            if array.dtype != target.dtype:
                raise core.CheckpointCompatibilityError(
                    f"dtype mismatch: checkpoint {array.dtype} vs target {target.dtype}"
                )
            return array.copy()

        cls = target.__class__
        if cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}:
            torch = importlib.import_module("torch")
            if str(target.dtype) == "torch.bfloat16":
                if array.dtype != np.dtype(np.uint16):
                    raise core.CheckpointCompatibilityError(
                        "dtype mismatch: bfloat16 target requires uint16 checkpoint representation, "
                        f"got {array.dtype}"
                    )
                tensor = torch.from_numpy(array.copy()).view(torch.bfloat16)
            else:
                try:
                    expected_numpy_dtype = torch.empty((), dtype=target.dtype).numpy().dtype
                except (TypeError, RuntimeError) as exc:
                    raise core.CheckpointCompatibilityError(
                        f"target dtype cannot be represented safely in checkpoint v1: {target.dtype}"
                    ) from exc
                if array.dtype != expected_numpy_dtype:
                    raise core.CheckpointCompatibilityError(
                        f"dtype mismatch: checkpoint {array.dtype} vs target {target.dtype}"
                    )
                tensor = torch.from_numpy(array.copy())
            if tuple(target.shape) != tuple(tensor.shape):
                raise core.CheckpointCompatibilityError(
                    f"shape mismatch: checkpoint {tuple(tensor.shape)} vs target {tuple(target.shape)}"
                )
            return tensor.to(device=target.device)
        raise core.CheckpointCompatibilityError(f"unsupported target tensor type {type(target)!r}")

    def hardened_validate_manifest_identity(identity: Any) -> None:
        original_validate_manifest_identity(identity)
        try:
            core.CheckpointIdentity(
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
            ).validate()
        except (TypeError, ValueError) as exc:
            raise core.CheckpointIntegrityError(f"invalid manifest identity: {exc}") from exc

    def hardened_assert_identity(
        manifest: Mapping[str, Any],
        *,
        git_sha: str | None = None,
        model_spec_hash: str | None = None,
        tokenizer_hash: str | None = None,
        tokenizer_vocab_hash: str | None = None,
        dataset_manifest_hash: str | None = None,
        run_manifest_hash: str | None = None,
        step: int | None = None,
        tokens_seen: int | None = None,
    ) -> None:
        original_assert_identity(
            manifest,
            git_sha=git_sha,
            model_spec_hash=model_spec_hash,
            tokenizer_hash=tokenizer_hash,
            tokenizer_vocab_hash=tokenizer_vocab_hash,
            dataset_manifest_hash=dataset_manifest_hash,
            run_manifest_hash=run_manifest_hash,
        )
        identity = manifest["identity"]
        expected = {"step": step, "tokens_seen": tokens_seen}
        mismatches = {
            key: {"expected": value, "actual": identity.get(key)}
            for key, value in expected.items()
            if value is not None and identity.get(key) != value
        }
        if mismatches:
            raise core.CheckpointCompatibilityError(
                f"checkpoint counter identity mismatch: {mismatches}"
            )

    def hardened_load_verified_checkpoint(
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
        expected_step: int | None = None,
        expected_tokens_seen: int | None = None,
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
            step=expected_step,
            tokens_seen=expected_tokens_seen,
        )
        arrays, combined_state = core._decode_verified_state(verified)
        materialized = core._prepare_model_weights(model, arrays, strict_model)

        optimizer_state = combined_state.get("optimizer")
        scheduler_state = combined_state.get("scheduler")
        if optimizer is not None:
            if optimizer_state is None:
                raise core.CheckpointCompatibilityError(
                    "optimizer was requested but checkpoint has no optimizer state"
                )
            _preflight_optimizer_state(core, optimizer, optimizer_state)
        if scheduler is not None:
            if scheduler_state is None:
                raise core.CheckpointCompatibilityError(
                    "scheduler was requested but checkpoint has no scheduler state"
                )
            _preflight_stateful_target(core, scheduler, scheduler_state, label="scheduler")
        if restore_rng:
            core._preflight_rng_state(combined_state["rng"])

        # Every compatibility check above completes before the first mutation.
        core._apply_model_weights(model, materialized, strict_model)
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_state)
        if scheduler is not None:
            scheduler.load_state_dict(scheduler_state)
        if restore_rng:
            core.restore_rng_state(combined_state["rng"])
        return core.LoadResult(
            manifest=copy.deepcopy(manifest),
            trainer_state=combined_state.get("trainer", {}),
            rng_state=combined_state["rng"],
        )

    def hardened_load_checkpoint(
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
        expected_step: int | None = None,
        expected_tokens_seen: int | None = None,
    ) -> Any:
        verified = core.prepare_checkpoint_load(directory)
        return core.load_verified_checkpoint(
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

    core._materialize_for_target = hardened_materialize_for_target
    core._validate_manifest_identity = hardened_validate_manifest_identity
    core.assert_identity = hardened_assert_identity
    core.load_verified_checkpoint = hardened_load_verified_checkpoint
    core.load_checkpoint = hardened_load_checkpoint
    core._D05_HARDENING_INSTALLED = True
