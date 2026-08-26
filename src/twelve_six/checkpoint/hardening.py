"""Fail-closed checkpoint preflight used by public and trainer resume paths.

This module strengthens checkpoint-v1 without changing its on-disk format. It
rejects semantically incompatible manifest counters, model dtypes and optimizer
state before the first live target mutation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import core as _core


def _require_integer(value: Any, *, field: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise _core.CheckpointIntegrityError(f"identity.{field} must be a {qualifier} integer")
    return value


def _validate_manifest_semantics(manifest: Mapping[str, Any]) -> None:
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise _core.CheckpointIntegrityError("manifest identity must be a mapping")

    _require_integer(identity.get("parameter_count"), field="parameter_count", minimum=1)
    _require_integer(identity.get("seed"), field="seed", minimum=0)
    _require_integer(identity.get("step"), field="step", minimum=0)
    _require_integer(identity.get("tokens_seen"), field="tokens_seen", minimum=0)

    precision = identity.get("precision")
    if not isinstance(precision, str) or not precision.strip():
        raise _core.CheckpointIntegrityError("identity.precision must be a non-empty string")

    for field in ("model_spec", "training_config", "optimizer", "environment"):
        value = identity.get(field)
        if not isinstance(value, Mapping) or not value:
            raise _core.CheckpointIntegrityError(f"identity.{field} must be a non-empty mapping")
    scheduler = identity.get("scheduler")
    if scheduler is not None and (not isinstance(scheduler, Mapping) or not scheduler):
        raise _core.CheckpointIntegrityError(
            "identity.scheduler must be a non-empty mapping or null"
        )


def _torch_tensor(value: Any) -> bool:
    cls = value.__class__
    return cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}


def _materialize_for_target(array: np.ndarray, target: Any) -> Any:
    """Materialize a checkpoint tensor only when shape and dtype are exact.

    bfloat16 is the sole storage exception: checkpoint-v1 stores its raw bits as
    uint16 because NumPy cannot represent torch.bfloat16 directly.
    """

    if isinstance(target, np.ndarray):
        if tuple(target.shape) != tuple(array.shape):
            raise _core.CheckpointCompatibilityError(
                f"shape mismatch: checkpoint {tuple(array.shape)} vs target {tuple(target.shape)}"
            )
        if target.dtype != array.dtype:
            raise _core.CheckpointCompatibilityError(
                f"dtype mismatch: checkpoint {array.dtype} vs target {target.dtype}"
            )
        return array.copy()

    if _torch_tensor(target):
        torch = __import__("torch")
        if str(target.dtype) == "torch.bfloat16" and array.dtype == np.uint16:
            tensor = torch.from_numpy(array.copy()).view(torch.bfloat16)
        else:
            try:
                tensor = torch.from_numpy(array.copy())
            except (TypeError, RuntimeError) as exc:
                raise _core.CheckpointCompatibilityError(
                    f"checkpoint dtype {array.dtype} cannot materialize for torch target"
                ) from exc
            if tensor.dtype != target.dtype:
                raise _core.CheckpointCompatibilityError(
                    f"dtype mismatch: checkpoint {tensor.dtype} vs target {target.dtype}"
                )
        if tuple(target.shape) != tuple(tensor.shape):
            raise _core.CheckpointCompatibilityError(
                f"shape mismatch: checkpoint {tuple(tensor.shape)} vs target {tuple(target.shape)}"
            )
        return tensor.to(device=target.device)

    raise _core.CheckpointCompatibilityError(f"unsupported target tensor type {type(target)!r}")


def _prepare_model_weights(
    model: Any, arrays: Mapping[str, np.ndarray], strict: bool
) -> dict[str, Any]:
    if not hasattr(model, "state_dict"):
        raise TypeError("model must provide state_dict()")
    target_state = model.state_dict()
    if not isinstance(target_state, Mapping) or not target_state:
        raise _core.CheckpointCompatibilityError("model.state_dict() must be a non-empty mapping")
    target_keys = set(target_state)
    source_keys = set(arrays)
    if strict and target_keys != source_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise _core.CheckpointCompatibilityError(
            f"state_dict keys differ: missing={missing}, unexpected={unexpected}"
        )
    return {
        name: _materialize_for_target(arrays[name], target_state[name])
        for name in target_state.keys() & arrays.keys()
    }


def _validate_tensor_against_parameter(value: Any, parameter: Any, *, path: str) -> None:
    if not _torch_tensor(value):
        return
    if value.numel() == 1:
        return
    if tuple(value.shape) != tuple(parameter.shape):
        raise _core.CheckpointCompatibilityError(
            f"optimizer state shape mismatch at {path}: "
            f"checkpoint {tuple(value.shape)} vs parameter {tuple(parameter.shape)}"
        )

    torch = __import__("torch")
    source_dtype = value.dtype
    target_dtype = parameter.dtype
    allowed_fp32_master = (
        source_dtype == torch.float32
        and target_dtype in {torch.float16, torch.bfloat16}
    )
    if source_dtype != target_dtype and not allowed_fp32_master:
        raise _core.CheckpointCompatibilityError(
            f"optimizer state dtype mismatch at {path}: "
            f"checkpoint {source_dtype} vs parameter {target_dtype}"
        )


def _walk_parameter_state(value: Any, parameter: Any, *, path: str) -> None:
    if _torch_tensor(value):
        _validate_tensor_against_parameter(value, parameter, path=path)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _walk_parameter_state(item, parameter, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_parameter_state(item, parameter, path=f"{path}[{index}]")


def _compare_existing_tensor_metadata(reference: Any, candidate: Any, *, path: str) -> None:
    if isinstance(reference, np.ndarray):
        if not isinstance(candidate, np.ndarray):
            raise _core.CheckpointCompatibilityError(f"{path} tensor backend mismatch")
        if reference.shape != candidate.shape:
            raise _core.CheckpointCompatibilityError(
                f"{path} shape mismatch: checkpoint {candidate.shape} vs target {reference.shape}"
            )
        if reference.dtype != candidate.dtype:
            raise _core.CheckpointCompatibilityError(
                f"{path} dtype mismatch: checkpoint {candidate.dtype} vs target {reference.dtype}"
            )
        return
    if _torch_tensor(reference):
        if not _torch_tensor(candidate):
            raise _core.CheckpointCompatibilityError(f"{path} tensor backend mismatch")
        if tuple(reference.shape) != tuple(candidate.shape):
            raise _core.CheckpointCompatibilityError(
                f"{path} shape mismatch: checkpoint {tuple(candidate.shape)} "
                f"vs target {tuple(reference.shape)}"
            )
        if reference.dtype != candidate.dtype:
            raise _core.CheckpointCompatibilityError(
                f"{path} dtype mismatch: checkpoint {candidate.dtype} vs target {reference.dtype}"
            )
        return
    if isinstance(reference, Mapping) and isinstance(candidate, Mapping):
        for key in reference.keys() & candidate.keys():
            _compare_existing_tensor_metadata(
                reference[key], candidate[key], path=f"{path}.{key}"
            )
        return
    if isinstance(reference, (list, tuple)) and isinstance(candidate, (list, tuple)):
        for index, (left, right) in enumerate(zip(reference, candidate, strict=False)):
            _compare_existing_tensor_metadata(left, right, path=f"{path}[{index}]")


def preflight_optimizer_state(optimizer: Any, state: Any) -> None:
    """Validate optimizer state geometry without mutating the live optimizer.

    PyTorch documents that optimizer parameter IDs are zipped to live parameters
    in order during load without additional verification. We therefore validate
    the saved per-parameter tensor geometry against those live parameters first.
    """

    if not isinstance(state, Mapping):
        raise _core.CheckpointCompatibilityError("checkpoint optimizer state must be a mapping")
    if not hasattr(optimizer, "state_dict") or not hasattr(optimizer, "load_state_dict"):
        raise _core.CheckpointCompatibilityError(
            "optimizer must provide state_dict() and load_state_dict()"
        )

    live_groups = getattr(optimizer, "param_groups", None)
    saved_groups = state.get("param_groups")
    saved_state = state.get("state")
    if isinstance(live_groups, list) and isinstance(saved_groups, list) and isinstance(saved_state, Mapping):
        if len(saved_groups) != len(live_groups):
            raise _core.CheckpointCompatibilityError("optimizer parameter-group count mismatch")

        id_to_parameter: dict[Any, Any] = {}
        for group_index, (saved_group, live_group) in enumerate(zip(saved_groups, live_groups, strict=True)):
            if not isinstance(saved_group, Mapping) or not isinstance(live_group, Mapping):
                raise _core.CheckpointCompatibilityError(
                    f"optimizer param_group[{group_index}] must be a mapping"
                )
            saved_params = saved_group.get("params")
            live_params = live_group.get("params")
            if not isinstance(saved_params, list) or not isinstance(live_params, list):
                raise _core.CheckpointCompatibilityError(
                    f"optimizer param_group[{group_index}].params must be a list"
                )
            if len(saved_params) != len(live_params):
                raise _core.CheckpointCompatibilityError(
                    f"optimizer param_group[{group_index}] parameter count mismatch"
                )
            for saved_id, parameter in zip(saved_params, live_params, strict=True):
                if saved_id in id_to_parameter:
                    raise _core.CheckpointCompatibilityError(
                        f"optimizer parameter id {saved_id!r} is duplicated"
                    )
                id_to_parameter[saved_id] = parameter

        unknown = set(saved_state) - set(id_to_parameter)
        if unknown:
            raise _core.CheckpointCompatibilityError(
                f"optimizer state contains unknown parameter ids: {sorted(unknown, key=repr)}"
            )
        for saved_id, parameter_state in saved_state.items():
            if not isinstance(parameter_state, Mapping):
                raise _core.CheckpointCompatibilityError(
                    f"optimizer state for parameter {saved_id!r} must be a mapping"
                )
            _walk_parameter_state(
                parameter_state,
                id_to_parameter[saved_id],
                path=f"optimizer.state[{saved_id!r}]",
            )
        return

    reference = optimizer.state_dict()
    if not isinstance(reference, Mapping):
        raise _core.CheckpointCompatibilityError("optimizer.state_dict() must return a mapping")
    _compare_existing_tensor_metadata(reference, state, path="optimizer")


def _preflight_loadable_state(obj: Any, state: Any, *, label: str) -> None:
    if not hasattr(obj, "state_dict") or not hasattr(obj, "load_state_dict"):
        raise _core.CheckpointCompatibilityError(
            f"{label} must provide state_dict() and load_state_dict()"
        )
    reference = obj.state_dict()
    if not isinstance(reference, Mapping) or not isinstance(state, Mapping):
        raise _core.CheckpointCompatibilityError(f"checkpoint {label} state must be a mapping")
    _compare_existing_tensor_metadata(reference, state, path=label)


def _config_mapping(config: Any) -> Mapping[str, Any] | None:
    if is_dataclass(config) and not isinstance(config, type):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "model_dump"):
        dumped = config.model_dump(mode="python")
        return dumped if isinstance(dumped, Mapping) else None
    if hasattr(config, "to_dict"):
        dumped = config.to_dict()
        return dumped if isinstance(dumped, Mapping) else None
    return None


def preflight_trainer_state(trainer: Any, state: Any) -> None:
    """Validate the repository Trainer resume payload before model mutation."""

    if not isinstance(state, Mapping):
        raise _core.CheckpointCompatibilityError("checkpoint trainer state must be a mapping")
    current_config = _config_mapping(getattr(trainer, "config", None))
    saved_config = state.get("config")
    if current_config is not None and saved_config != current_config:
        raise _core.CheckpointCompatibilityError("trainer config mismatch; refusing unsafe resume")

    micro_step = state.get("micro_step")
    optimizer_step = state.get("optimizer_step")
    tokens_seen = state.get("tokens_seen")
    for field, value in (
        ("micro_step", micro_step),
        ("optimizer_step", optimizer_step),
        ("tokens_seen", tokens_seen),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise _core.CheckpointCompatibilityError(
                f"trainer {field} must be a non-negative integer"
            )

    if current_config is not None:
        accumulation = current_config.get("gradient_accumulation_steps")
        if isinstance(accumulation, int) and accumulation > 0:
            expected_micro = optimizer_step * accumulation
            if micro_step != expected_micro:
                raise _core.CheckpointCompatibilityError(
                    "checkpoint is not at a complete committed accumulation boundary: "
                    f"micro_step={micro_step}, expected={expected_micro}"
                )
        max_steps = current_config.get("max_steps")
        if isinstance(max_steps, int) and optimizer_step > max_steps:
            raise _core.CheckpointCompatibilityError(
                "checkpoint optimizer_step exceeds configured max_steps"
            )

    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        raise _core.CheckpointCompatibilityError("trainer has no optimizer to restore")
    preflight_optimizer_state(optimizer, state.get("optimizer"))

    scheduler = getattr(trainer, "scheduler", None)
    saved_scheduler = state.get("scheduler")
    if (scheduler is None) != (saved_scheduler is None):
        raise _core.CheckpointCompatibilityError("scheduler state/config mismatch")
    if scheduler is not None:
        _preflight_loadable_state(scheduler, saved_scheduler, label="scheduler")

    scaler = getattr(trainer, "scaler", None)
    saved_scaler = state.get("scaler")
    if scaler is not None and saved_scaler is not None:
        _preflight_loadable_state(scaler, saved_scaler, label="scaler")


def verify_checkpoint(directory: str | Path) -> dict[str, Any]:
    verified = _core.prepare_checkpoint_load(directory)
    manifest = verified.manifest
    _validate_manifest_semantics(manifest)
    return manifest


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
    """Strictly preflight a verified snapshot, then mutate requested targets."""

    manifest = verified.manifest
    _validate_manifest_semantics(manifest)
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
    materialized = _prepare_model_weights(model, arrays, strict_model)

    optimizer_state = combined_state.get("optimizer")
    scheduler_state = combined_state.get("scheduler")
    if optimizer is not None:
        if optimizer_state is None:
            raise _core.CheckpointCompatibilityError(
                "optimizer was requested but checkpoint has no optimizer state"
            )
        preflight_optimizer_state(optimizer, optimizer_state)
    if scheduler is not None:
        if scheduler_state is None:
            raise _core.CheckpointCompatibilityError(
                "scheduler was requested but checkpoint has no scheduler state"
            )
        _preflight_loadable_state(scheduler, scheduler_state, label="scheduler")
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
