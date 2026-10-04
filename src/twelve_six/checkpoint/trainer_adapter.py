"""Trainer-owned checkpoint adapter.

D02 owns trainer semantics. D05 only converts a trainer's public state_dict()
contract into the data-only checkpoint format and gives the decoded state back
to trainer.load_state_dict(). This avoids duplicating optimizer/scheduler/scaler
ownership inside the checkpoint API.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from . import core as _core
from .core import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    LoadResult,
    capture_rng_state,
    _apply_model_weights,
    _decode_verified_state,
    _preflight_optimizer_state,
    _preflight_rng_state,
    _prepare_model_weights,
    _semantic_stateful_probe,
    assert_identity,
    prepare_checkpoint_load,
    restore_rng_state,
    save_checkpoint,
)

_CANONICAL_TRAINER_STATE_FIELDS = frozenset(
    {
        "micro_step",
        "optimizer_step",
        "tokens_seen",
        "optimizer",
        "scheduler",
        "scaler",
        "config",
    }
)


def _trainer_state_as_mapping(state: Any) -> Mapping[str, Any]:
    if is_dataclass(state) and not isinstance(state, type):
        return asdict(state)
    if isinstance(state, Mapping):
        return dict(state)
    raise TypeError(
        "trainer.state_dict() must return a dataclass instance or mapping "
        "for data-only serialization"
    )


def _assert_bound_metadata(
    manifest: Mapping[str, Any],
    *,
    expected_init_spec_hash: str | None,
    expected_split_identity: str | None,
    expected_packing_hash: str | None,
    expected_packing_version: str | None,
    expected_training_config_hash: str | None,
    expected_environment_lock_hash: str | None,
    expected_seed: int | None,
) -> None:
    """Check canonical run-binding fields after verification and before mutation."""

    expectations = (
        expected_init_spec_hash,
        expected_split_identity,
        expected_packing_hash,
        expected_packing_version,
        expected_training_config_hash,
        expected_environment_lock_hash,
        expected_seed,
    )
    if all(value is None for value in expectations):
        return

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise CheckpointCompatibilityError("verified checkpoint identity is missing")
    training_config = identity.get("training_config")
    if not isinstance(training_config, Mapping):
        raise CheckpointCompatibilityError("verified checkpoint training_config is missing")

    need_data = any(
        value is not None
        for value in (expected_split_identity, expected_packing_hash, expected_packing_version)
    )
    data = training_config.get("data")
    if need_data and not isinstance(data, Mapping):
        raise CheckpointCompatibilityError("verified checkpoint bound data identity is missing")
    if not isinstance(data, Mapping):
        data = {}

    checks = {
        "init_spec_hash": (expected_init_spec_hash, training_config.get("init_spec_sha256")),
        "split_identity": (expected_split_identity, data.get("split_identity")),
        "packing_hash": (expected_packing_hash, data.get("packing_sha256")),
        "packing_version": (expected_packing_version, data.get("packing_version")),
        "training_config_hash": (
            expected_training_config_hash,
            identity.get("training_config_hash"),
        ),
        "environment_lock_hash": (
            expected_environment_lock_hash,
            identity.get("environment_lock_hash"),
        ),
        "seed": (expected_seed, identity.get("seed")),
    }
    mismatches = {
        name: {"expected": expected, "actual": actual}
        for name, (expected, actual) in checks.items()
        if expected is not None and expected != actual
    }
    if mismatches:
        raise CheckpointCompatibilityError(f"checkpoint canonical binding mismatch: {mismatches}")


def _validate_state_schema(expected: Any, actual: Any, *, path: str) -> None:
    """Validate a state-dict payload without mutating or copying model-scale objects."""

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise CheckpointCompatibilityError(f"{path} must be a mapping")
        expected_keys = set(expected)
        actual_keys = set(actual)
        if actual_keys != expected_keys:
            missing = sorted(map(str, expected_keys - actual_keys))
            unexpected = sorted(map(str, actual_keys - expected_keys))
            raise CheckpointCompatibilityError(
                f"{path} keys differ: missing={missing}, unexpected={unexpected}"
            )
        for key, expected_value in expected.items():
            _validate_state_schema(
                expected_value,
                actual[key],
                path=f"{path}.{key}",
            )
        return

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise CheckpointCompatibilityError(
                f"{path} list geometry mismatch: expected length {len(expected)}"
            )
        for index, (expected_value, actual_value) in enumerate(
            zip(expected, actual, strict=True)
        ):
            _validate_state_schema(
                expected_value,
                actual_value,
                path=f"{path}[{index}]",
            )
        return

    if isinstance(expected, tuple):
        if not isinstance(actual, tuple) or len(actual) != len(expected):
            raise CheckpointCompatibilityError(
                f"{path} tuple geometry mismatch: expected length {len(expected)}"
            )
        for index, (expected_value, actual_value) in enumerate(
            zip(expected, actual, strict=True)
        ):
            _validate_state_schema(
                expected_value,
                actual_value,
                path=f"{path}[{index}]",
            )
        return

    expected_cls = expected.__class__ if expected is not None else None
    actual_cls = actual.__class__ if actual is not None else None
    expected_is_tensor = bool(
        expected_cls is not None
        and expected_cls.__module__.startswith("torch")
        and expected_cls.__name__ in {"Tensor", "Parameter"}
    )
    actual_is_tensor = bool(
        actual_cls is not None
        and actual_cls.__module__.startswith("torch")
        and actual_cls.__name__ in {"Tensor", "Parameter"}
    )
    if expected_is_tensor:
        if not actual_is_tensor:
            raise CheckpointCompatibilityError(f"{path} must be a torch tensor")
        if tuple(actual.shape) != tuple(expected.shape) or actual.dtype != expected.dtype:
            raise CheckpointCompatibilityError(
                f"{path} tensor metadata mismatch: checkpoint "
                f"shape={tuple(actual.shape)} dtype={actual.dtype}, live "
                f"shape={tuple(expected.shape)} dtype={expected.dtype}"
            )
        return

    if expected is None:
        if actual is not None:
            raise CheckpointCompatibilityError(f"{path} must be None")
        return

    if isinstance(expected, bool):
        compatible = isinstance(actual, bool)
    elif isinstance(expected, int):
        compatible = isinstance(actual, int) and not isinstance(actual, bool)
    elif isinstance(expected, float):
        compatible = isinstance(actual, float)
    else:
        compatible = type(actual) is type(expected)
    if not compatible:
        raise CheckpointCompatibilityError(
            f"{path} type mismatch: checkpoint {type(actual).__name__}, "
            f"live {type(expected).__name__}"
        )


def _preflight_stateful_component(component: Any | None, state: Any, *, label: str) -> None:
    """Check scheduler/scaler schema and load semantics before live mutation."""

    if (state is None) != (component is None):
        raise CheckpointCompatibilityError(f"{label} state/config mismatch")
    if component is None:
        return
    if not hasattr(component, "state_dict") or not hasattr(component, "load_state_dict"):
        raise CheckpointCompatibilityError(
            f"{label} must provide state_dict/load_state_dict"
        )
    live_state = component.state_dict()
    if not isinstance(live_state, Mapping):
        raise CheckpointCompatibilityError(f"live {label} state must be a mapping")
    _validate_state_schema(live_state, state, path=f"{label} state")
    _semantic_stateful_probe(component, state, label=label)


def _assert_trainer_model_binding(model: Any, trainer: Any) -> None:
    """Refuse mismatched D02 model/optimizer owners before saving or restoring."""

    if (
        hasattr(trainer, "_failure_reason")
        and hasattr(trainer, "_update_incomplete")
        and hasattr(trainer, "model")
        and trainer.model is not model
    ):
        raise CheckpointCompatibilityError(
            "canonical trainer owns a different model than the checkpoint target"
        )


def _preflight_trainer_target(trainer: Any) -> None:
    """Reject a D02 trainer target that its own loader would refuse after mutation."""

    if not (
        hasattr(trainer, "_failure_reason")
        and hasattr(trainer, "_update_incomplete")
    ):
        return
    if trainer._failure_reason is not None:
        raise CheckpointCompatibilityError(
            "checkpoint restore requires a fresh trainer; target trainer is poisoned"
        )
    if trainer._update_incomplete:
        raise CheckpointCompatibilityError(
            "checkpoint restore requires a fresh trainer; target trainer has an incomplete update"
        )
    # D02 refuses restoration to a trainer which has already consumed data,
    # has pending accumulation, or retains gradients. Check the same live
    # conditions before opening a checkpoint or changing model weights.
    if any(
        getattr(trainer, field, 0) != 0
        for field in (
            "micro_step",
            "optimizer_step",
            "tokens_seen",
            "_pending_tokens",
            "_pending_loss_sum",
        )
    ):
        raise CheckpointCompatibilityError(
            "checkpoint restore requires a fresh trainer with no consumed exposure"
        )
    model = getattr(trainer, "model", None)
    parameters = getattr(model, "parameters", None)
    if callable(parameters) and any(
        getattr(parameter, "grad", None) is not None for parameter in parameters()
    ):
        raise CheckpointCompatibilityError(
            "checkpoint restore requires a fresh trainer with no pending gradients"
        )


def _preflight_trainer_state_without_rng_guard(
    trainer: Any,
    state: Any,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Validate trainer-owned resume state before checkpoint model mutation.

    ``Trainer.load_state_dict`` correctly rejects invalid counters/configuration,
    but the adapter used to call it only after model weights had already been
    restored. This mirrors those fail-closed checks and reuses D05 optimizer
    geometry validation so a bad trainer-owned AdamW/SGD state cannot partially
    restore a live model before failing.

    Checkpoint-v1 also supports generic trainer-owned state adapters that do not
    expose a public ``optimizer`` attribute. Those retain compatibility through
    an isolated deep-copy load probe so the live trainer and model remain untouched.
    """

    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError("checkpoint trainer state must be a mapping")

    _preflight_trainer_target(trainer)

    # Canonical D02 Trainer and its scale subclasses construct TrainerState(**state)
    # during the real load. Extra keys therefore fail only at that final call unless
    # the adapter mirrors the exact schema now, before model/RNG mutation.
    if hasattr(trainer, "_failure_reason") and hasattr(trainer, "_update_incomplete"):
        actual_fields = set(state)
        if actual_fields != _CANONICAL_TRAINER_STATE_FIELDS:
            missing = sorted(_CANONICAL_TRAINER_STATE_FIELDS - actual_fields)
            unexpected = sorted(actual_fields - _CANONICAL_TRAINER_STATE_FIELDS)
            raise CheckpointCompatibilityError(
                "canonical trainer state keys differ: "
                f"missing={missing}, unexpected={unexpected}"
            )

    for field in ("micro_step", "optimizer_step", "tokens_seen"):
        value = state.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CheckpointCompatibilityError(
                f"trainer {field} must be a non-negative integer"
            )

    if manifest is not None:
        identity = manifest.get("identity")
        if not isinstance(identity, Mapping):
            raise CheckpointCompatibilityError("verified checkpoint identity is missing")
        if state["optimizer_step"] != identity.get("step"):
            raise CheckpointCompatibilityError(
                "trainer optimizer_step disagrees with checkpoint identity.step"
            )
        if state["tokens_seen"] != identity.get("tokens_seen"):
            raise CheckpointCompatibilityError(
                "trainer tokens_seen disagrees with checkpoint identity.tokens_seen"
            )

    live_config = getattr(trainer, "config", None)
    if is_dataclass(live_config) and not isinstance(live_config, type):
        live_config = asdict(live_config)
    elif hasattr(live_config, "model_dump"):
        live_config = live_config.model_dump(mode="python")

    checkpoint_config = state.get("config")
    if live_config is not None and checkpoint_config != live_config:
        raise CheckpointCompatibilityError("trainer config mismatch; refusing unsafe resume")

    if isinstance(live_config, Mapping):
        accumulation = live_config.get("gradient_accumulation_steps")
        max_steps = live_config.get("max_steps")
        if (
            isinstance(accumulation, int)
            and not isinstance(accumulation, bool)
            and accumulation > 0
        ):
            expected_micro_steps = state["optimizer_step"] * accumulation
            if state["micro_step"] != expected_micro_steps:
                raise CheckpointCompatibilityError(
                    "checkpoint is not at a complete committed accumulation boundary: "
                    f"micro_step={state['micro_step']}, expected={expected_micro_steps}"
                )
        if (
            isinstance(max_steps, int)
            and not isinstance(max_steps, bool)
            and state["optimizer_step"] > max_steps
        ):
            raise CheckpointCompatibilityError(
                "checkpoint optimizer_step exceeds configured max_steps"
            )

    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        if not hasattr(trainer, "load_state_dict"):
            raise CheckpointCompatibilityError("trainer must provide load_state_dict")
        try:
            probe = copy.deepcopy(trainer)
            probe.load_state_dict(copy.deepcopy(state))
        except Exception as exc:
            raise CheckpointCompatibilityError(
                "checkpoint trainer state failed isolated compatibility preflight"
            ) from exc
        return

    _preflight_optimizer_state(optimizer, state.get("optimizer"))
    _preflight_stateful_component(
        getattr(trainer, "scheduler", None),
        state.get("scheduler"),
        label="scheduler",
    )
    _preflight_stateful_component(
        getattr(trainer, "scaler", None),
        state.get("scaler"),
        label="scaler",
    )



def _preflight_trainer_state(
    trainer: Any,
    state: Any,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    """Keep detached loader probes from advancing live process RNG streams.

    A deep-copied trainer, optimizer, scheduler or scaler can still call the
    module-global Python, NumPy or torch generators. Whether semantic probing
    succeeds or rejects a checkpoint, it must not silently alter future draws.
    The actual loader runs later in the guarded model -> trainer -> RNG region.
    """

    ambient = capture_rng_state()
    torch_state = ambient.get("torch")
    warn_only = None
    if torch_state is not None:
        torch = importlib.import_module("torch")
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        _preflight_trainer_state_without_rng_guard(
            trainer, state, manifest=manifest,
        )
    finally:
        # Ambient probe rollback is not the application-stage RNG restore.
        # If it fails, the live RNG is ambiguous even though the model has not
        # been loaded. Refuse future work on a canonical D02 trainer.
        try:
            try:
                _core.restore_rng_state(ambient)
            except BaseException as rng_exc:
                # A secondary policy rollback fault must not hide the primary
                # failed/interrupted RNG rollback or its preflight context.
                if warn_only is not None:
                    try:
                        torch.use_deterministic_algorithms(
                            bool(torch_state["deterministic_algorithms"]),
                            warn_only=warn_only,
                        )
                    except BaseException as mode_exc:
                        rng_exc.add_note(
                            "PyTorch preflight-mode rollback also failed: "
                            f"{mode_exc!r}"
                        )
                raise
            else:
                if warn_only is not None:
                    torch.use_deterministic_algorithms(
                        bool(torch_state["deterministic_algorithms"]),
                        warn_only=warn_only,
                    )
        except BaseException:
            if hasattr(trainer, "_failure_reason") and hasattr(trainer, "_update_incomplete"):
                if trainer._failure_reason is None:
                    trainer._failure_reason = "checkpoint_preflight_rng_rollback_failed"
                trainer._update_incomplete = True
            raise



def _assert_live_d02_determinism(trainer: Any) -> bool | None:
    """Reject ambient torch policy drift before exporting or restoring D02."""

    if not (
        hasattr(trainer, "_failure_reason")
        and hasattr(trainer, "_update_incomplete")
    ):
        return None
    config = getattr(trainer, "config", None)
    enabled = getattr(config, "deterministic_algorithms", None)
    warn_only = getattr(config, "deterministic_warn_only", None)
    if type(enabled) is not bool or type(warn_only) is not bool:
        return None
    torch = importlib.import_module("torch")
    if (
        torch.are_deterministic_algorithms_enabled() != enabled
        or torch.is_deterministic_algorithms_warn_only_enabled() != warn_only
    ):
        raise CheckpointCompatibilityError(
            "live torch deterministic policy disagrees with canonical trainer configuration"
        )
    return enabled


def _assert_d02_checkpoint_rng_policy(
    trainer: Any, rng_state: Mapping[str, Any],
) -> None:
    """Bind captured deterministic enablement to the canonical D02 config."""

    configured = _assert_live_d02_determinism(trainer)
    if configured is None:
        return
    torch_state = rng_state.get("torch") if isinstance(rng_state, Mapping) else None
    if (
        not isinstance(torch_state, Mapping)
        or type(torch_state.get("deterministic_algorithms")) is not bool
    ):
        raise CheckpointCompatibilityError(
            "canonical trainer requires an exact torch deterministic_algorithms "
            "checkpoint RNG policy"
        )
    if torch_state["deterministic_algorithms"] != configured:
        raise CheckpointCompatibilityError(
            "checkpoint torch deterministic_algorithms disagrees with "
            "canonical trainer configuration"
        )


def _restore_checkpoint_rng_preserving_warn_only(
    state: Mapping[str, Any],
    *,
    restore: Any,
) -> None:
    """Do not erase the live PyTorch warn-only policy on checkpoint RNG replay.

    The V1 RNG snapshot records deterministic enablement, but not warn_only.
    A canonical D02 Trainer has already configured its validated policy; the
    core RNG restore defaults warn_only to False even when it was True.
    """

    torch_state = state.get("torch")
    warn_only = None
    if torch_state:
        torch = importlib.import_module("torch")
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        restore(state)
        if warn_only is not None:
            torch.use_deterministic_algorithms(
                torch.are_deterministic_algorithms_enabled(), warn_only=warn_only,
            )
    except BaseException as exc:
        # Final replay or its follow-up policy application can partially
        # change process-global PyTorch execution
        # mode before it fails. The target trainer is poisoned by the caller,
        # but unrelated trainers must not inherit a half-applied mode.
        if warn_only is not None:
            try:
                torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            except BaseException as mode_exc:
                # Never replace the primary interrupted/failed RNG restore.
                if hasattr(exc, "add_note"):
                    exc.add_note(
                        "PyTorch deterministic-mode rollback also failed: "
                        f"{mode_exc!r}"
                    )
        raise


def save_trainer_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    trainer: Any,
    identity: CheckpointIdentity,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Save model + trainer-owned optimizer/scheduler/scaler/counter state."""

    if not hasattr(trainer, "state_dict"):
        raise TypeError("trainer must provide state_dict()")
    _assert_trainer_model_binding(model, trainer)
    state = _trainer_state_as_mapping(trainer.state_dict())
    # A canonical D02 checkpoint should never be produced under a different
    # ambient PyTorch policy than the validated trainer configuration.
    _assert_live_d02_determinism(trainer)
    return save_checkpoint(
        directory,
        model=model,
        trainer_state=state,
        identity=identity,
        overwrite=overwrite,
    )


def load_trainer_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    trainer: Any,
    strict_model: bool = True,
    restore_rng: bool = True,
    expected_git_sha: str | None = None,
    expected_model_spec_hash: str | None = None,
    expected_init_spec_hash: str | None = None,
    expected_tokenizer_hash: str | None = None,
    expected_tokenizer_vocab_hash: str | None = None,
    expected_dataset_manifest_hash: str | None = None,
    expected_split_identity: str | None = None,
    expected_packing_hash: str | None = None,
    expected_packing_version: str | None = None,
    expected_run_manifest_hash: str | None = None,
    expected_training_config_hash: str | None = None,
    expected_environment_lock_hash: str | None = None,
    expected_seed: int | None = None,
) -> LoadResult:
    """Verify and decode one snapshot once, then restore fresh D02 targets.

    Canonical identity checks, trainer-state preflight, model materialization,
    RNG preflight, and the actual load all consume one decoded verified snapshot.
    No checkpoint artifact is reopened or decoded a second time before mutation.
    """

    if not hasattr(trainer, "load_state_dict"):
        raise TypeError("trainer must provide load_state_dict()")

    _assert_trainer_model_binding(model, trainer)
    _preflight_trainer_target(trainer)
    verified = prepare_checkpoint_load(directory)
    manifest = verified.manifest
    _assert_bound_metadata(
        manifest,
        expected_init_spec_hash=expected_init_spec_hash,
        expected_split_identity=expected_split_identity,
        expected_packing_hash=expected_packing_hash,
        expected_packing_version=expected_packing_version,
        expected_training_config_hash=expected_training_config_hash,
        expected_environment_lock_hash=expected_environment_lock_hash,
        expected_seed=expected_seed,
    )
    assert_identity(
        manifest,
        git_sha=expected_git_sha,
        model_spec_hash=expected_model_spec_hash,
        tokenizer_hash=expected_tokenizer_hash,
        tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
        dataset_manifest_hash=expected_dataset_manifest_hash,
        run_manifest_hash=expected_run_manifest_hash,
    )

    arrays, combined_state = _decode_verified_state(verified)
    del verified

    trainer_state = combined_state.get("trainer")
    _preflight_trainer_state(
        trainer,
        trainer_state,
        manifest=manifest,
    )
    if restore_rng:
        _preflight_rng_state(combined_state["rng"])
        _assert_d02_checkpoint_rng_policy(trainer, combined_state["rng"])
    else:
        _assert_live_d02_determinism(trainer)
    materialized = _prepare_model_weights(model, arrays, strict_model)

    # The decoded source weights are no longer needed after target materialization.
    # Releasing them before the first mutation keeps resume peak memory bounded as
    # the same checkpoint path scales from 20M toward 100M and 1B parameters.
    del arrays

    # State loaders may draw from process RNG even when they succeed.
    # Failed application may leave a mixed model/optimizer state, so canonical
    # D02 targets must require a fresh instance and verified checkpoint.
    try:
        _apply_model_weights(model, materialized, strict_model)
        trainer.load_state_dict(trainer_state)
        if restore_rng:
            _restore_checkpoint_rng_preserving_warn_only(
                combined_state["rng"], restore=restore_rng_state,
            )
    except BaseException:
        if hasattr(trainer, "_failure_reason") and hasattr(trainer, "_update_incomplete"):
            if trainer._failure_reason is None:
                trainer._failure_reason = "checkpoint_restore_apply_failed"
            trainer._update_incomplete = True
        raise
    return LoadResult(
        manifest=copy.deepcopy(manifest),
        trainer_state=trainer_state,
        rng_state=combined_state["rng"],
    )
