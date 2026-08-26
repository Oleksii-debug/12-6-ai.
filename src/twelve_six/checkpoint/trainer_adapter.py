"""Trainer-owned checkpoint adapter.

D02 owns trainer semantics. D05 only converts a trainer's public state_dict()
contract into the data-only checkpoint format and gives the decoded state back
to trainer.load_state_dict(). This avoids duplicating optimizer/scheduler/scaler
ownership inside the checkpoint API.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .core import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    LoadResult,
    _decode_verified_state,
    _preflight_optimizer_state,
    load_verified_checkpoint,
    prepare_checkpoint_load,
    save_checkpoint,
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
    """Check scheduler/scaler state schema without cloning model-scale optimizer state."""

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


def _preflight_trainer_state(
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
    state = _trainer_state_as_mapping(trainer.state_dict())
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
    """Verify one exact byte snapshot, bind it, then restore fresh D02 targets.

    The canonical nested identity checks, trainer-state preflight, and actual load
    consume the same verified byte snapshot. The source directory is never
    re-opened between run-binding verification and model mutation.
    """

    if not hasattr(trainer, "load_state_dict"):
        raise TypeError("trainer must provide load_state_dict()")

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

    # The trainer owns optimizer/scheduler/counter state, so validate that exact
    # decoded snapshot before load_verified_checkpoint is allowed to touch model
    # weights. This closes the same deferred optimizer-failure class for the
    # production Trainer adapter path used by long-running campaigns.
    _, combined_state = _decode_verified_state(verified)
    _preflight_trainer_state(
        trainer,
        combined_state.get("trainer"),
        manifest=manifest,
    )

    result = load_verified_checkpoint(
        verified,
        model=model,
        strict_model=strict_model,
        restore_rng=restore_rng,
        expected_git_sha=expected_git_sha,
        expected_model_spec_hash=expected_model_spec_hash,
        expected_tokenizer_hash=expected_tokenizer_hash,
        expected_tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
        expected_dataset_manifest_hash=expected_dataset_manifest_hash,
        expected_run_manifest_hash=expected_run_manifest_hash,
    )
    trainer.load_state_dict(result.trainer_state)
    return result
