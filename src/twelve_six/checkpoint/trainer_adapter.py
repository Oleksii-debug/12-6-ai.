"""Trainer-owned checkpoint adapter.

D02 owns trainer semantics. D05 only converts a trainer's public state_dict()
contract into the data-only checkpoint format and gives the decoded state back
to trainer.load_state_dict(). This avoids duplicating optimizer/scheduler/scaler
ownership inside the checkpoint API.
"""

from __future__ import annotations

import copy
import importlib
import inspect
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
    _bind_model_state_loader,
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
from .expected_binding import (
    _validate_expected_canonical_binding,
    _validate_expected_core_identity,
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


def _typed_config_equal(left: Any, right: Any) -> bool:
    """Reject Python numeric/bool aliases in canonical checkpoint configuration."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if left.keys() != right.keys():
            return False
        return all(_typed_config_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return False
        return all(_typed_config_equal(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


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
    if (
        not callable(getattr(component, "state_dict", None))
        or not callable(getattr(component, "load_state_dict", None))
    ):
        raise CheckpointCompatibilityError(
            f"{label} must provide state_dict/load_state_dict"
        )
    live_state = component.state_dict()
    if not isinstance(live_state, Mapping):
        raise CheckpointCompatibilityError(f"live {label} state must be a mapping")
    _validate_state_schema(live_state, state, path=f"{label} state")
    _semantic_stateful_probe(component, state, label=label)


def _is_canonical_d02(trainer: Any) -> bool:
    """Identify the D02 recovery protocol without executing custom descriptors."""

    try:
        attrs = vars(trainer)
    except TypeError:
        return False
    return "_failure_reason" in attrs and "_update_incomplete" in attrs


def _poison_canonical_d02(trainer: Any, reason: str) -> None:
    """Record fail-closed recovery state without invoking custom descriptors."""

    if not _is_canonical_d02(trainer):
        return
    attrs = vars(trainer)
    if attrs.get("_failure_reason") is None:
        attrs["_failure_reason"] = reason
    attrs["_update_incomplete"] = True


def _poison_canonical_restore_failure(
    trainer: Any,
    *,
    expected_canonical: bool,
    reason: str,
    exc: BaseException,
) -> None:
    """Poison an entry-canonical target even if a hook deleted its markers."""

    if not expected_canonical:
        return
    try:
        attrs = vars(trainer)
    except TypeError as poison_exc:
        exc.add_note(
            "canonical trainer recovery-state access also failed: "
            f"{poison_exc!r}"
        )
        return
    if attrs.get("_failure_reason") is None:
        attrs["_failure_reason"] = reason
    attrs["_update_incomplete"] = True


def _snapshot_trainer_restore_bindings(
    trainer: Any,
) -> tuple[bool, dict[str, Any]]:
    """Pin canonical restore component identities without descriptor dispatch."""

    canonical_d02 = _is_canonical_d02(trainer)
    if not canonical_d02:
        return False, {}
    attrs = vars(trainer)
    return True, {
        field: attrs[field]
        for field in ("model", "optimizer", "scheduler", "scaler", "config")
        if field in attrs
    }


def _assert_trainer_restore_bindings(
    trainer: Any,
    snapshot: tuple[bool, dict[str, Any]],
) -> None:
    """Reject safety classification or restore-component identity drift."""

    expected_canonical, bindings = snapshot
    if _is_canonical_d02(trainer) != expected_canonical:
        raise CheckpointCompatibilityError(
            "trainer safety classification changed during checkpoint restore"
        )
    if not expected_canonical:
        return
    try:
        attrs = vars(trainer)
    except TypeError as exc:
        raise CheckpointCompatibilityError(
            "canonical trainer does not expose instance recovery state"
        ) from exc
    sentinel = object()
    for field, expected in bindings.items():
        if attrs.get(field, sentinel) is not expected:
            raise CheckpointCompatibilityError(
                f"canonical trainer {field} binding changed during checkpoint restore"
            )


def _assert_trainer_model_binding(model: Any, trainer: Any) -> None:
    """Refuse mismatched D02 model owners without executing custom descriptors."""

    if not _is_canonical_d02(trainer):
        return
    attrs = vars(trainer)
    if "model" in attrs and attrs["model"] is not model:
        raise CheckpointCompatibilityError(
            "canonical trainer owns a different model than the checkpoint target"
        )


def _effective_strict_model(trainer: Any, strict_model: bool) -> bool:
    """Canonical D02 resume must restore every persistent model-state key.

    Non-strict loading is still supported for generic checkpoint adapters.
    D02's optimizer counters must never be credited to a partially restored
    model, even when the caller explicitly requests strict_model=False.
    """

    return strict_model or _is_canonical_d02(trainer)


def _bind_trainer_state_loader(trainer: Any) -> Any:
    loader = getattr(trainer, "load_state_dict", None)
    if not callable(loader):
        raise TypeError("trainer must provide load_state_dict()")

    # Canonical D02 must fail closed before model mutation when its restore
    # invocation cannot accept the one authoritative trainer-state payload.
    # Generic adapters retain the historical permissive callable contract.
    canonical_d02 = _is_canonical_d02(trainer)
    if canonical_d02:
        try:
            signature = inspect.signature(loader)
            signature.bind({})
        except (TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                "trainer load_state_dict cannot safely bind checkpoint state"
            ) from exc
    return loader


def _preflight_trainer_target(trainer: Any) -> None:
    """Reject a D02 trainer target that its own loader would refuse after mutation."""

    if not _is_canonical_d02(trainer):
        return
    initial_attrs = vars(trainer)
    if initial_attrs.get("_failure_reason") is not None:
        raise CheckpointCompatibilityError(
            "checkpoint restore requires a fresh trainer; target trainer is poisoned"
        )
    if initial_attrs.get("_update_incomplete"):
        raise CheckpointCompatibilityError(
            "checkpoint restore requires a fresh trainer; target trainer has an incomplete update"
        )

    # Freeze canonical component identities before *any* effectful authority
    # descriptor lookup. A getter that rebinds one of these objects must be
    # observed as drift by the descriptor-free final snapshot below.
    model = initial_attrs.get("model")
    optimizer = initial_attrs.get("optimizer")
    config = initial_attrs.get("config")
    scheduler = initial_attrs.get("scheduler")
    scaler = initial_attrs.get("scaler")

    # Bind every effectful authority/interface lookup before the final
    # freshness snapshot. Descriptor/proxy lookup itself may execute user code;
    # any such side effect must therefore be visible to the checks below.
    authorities: dict[str, Any] = {}
    for authority, label in (
        ("_require_finite_auxiliary_state", "auxiliary-state"),
        ("_require_safe_optimizer_hyperparameters", "optimizer-hyperparameter"),
        ("_require_finite_committed_update", "committed-update"),
        ("_require_no_residual_model_gradients", "gradient-cleanliness"),
        ("_require_deterministic_policy", "deterministic-policy"),
        ("_require_optimizer_parameter_coverage", "optimizer-coverage"),
    ):
        bound = getattr(trainer, authority, None)
        if not callable(bound):
            raise CheckpointCompatibilityError(
                f"canonical trainer {label} authority unavailable"
            )
        authorities[authority] = bound

    parameters = getattr(model, "parameters", None)
    zero_grad = getattr(optimizer, "zero_grad", None) if optimizer is not None else None
    if optimizer is not None and not callable(zero_grad):
        raise CheckpointCompatibilityError(
            "canonical trainer optimizer zero_grad unavailable"
        )

    # D02 validates optimizer/model parameter ownership during its actual load.
    # Invoke the already-bound authority before the final freshness snapshot so
    # an effectful custom authority cannot mutate the target after that snapshot.
    try:
        authorities["_require_optimizer_parameter_coverage"]()
    except Exception as exc:
        raise CheckpointCompatibilityError(
            "checkpoint restore requires valid optimizer ownership of model parameters"
        ) from exc

    pending_gradient = bool(
        callable(parameters)
        and any(
            getattr(parameter, "grad", None) is not None
            for parameter in parameters()
        )
    )

    # Global deterministic mode is a pure target compatibility precondition.
    # Run it after the effectful bindings/calls above, then close with the
    # freshness/identity checks that gate checkpoint I/O and model mutation.
    _assert_live_d02_determinism(trainer)
    # Canonical D02 stores its recovery flags, counters and restore components
    # as instance attributes. Take one descriptor-free final snapshot so a
    # late property/proxy read cannot mutate an earlier checked field.
    try:
        live_attrs = vars(trainer)
    except TypeError as exc:
        raise CheckpointCompatibilityError(
            "canonical trainer does not expose instance recovery state"
        ) from exc
    if live_attrs.get("_failure_reason") is not None:
        raise CheckpointCompatibilityError(
            "checkpoint restore requires a fresh trainer; target trainer is poisoned"
        )
    if live_attrs.get("_update_incomplete"):
        raise CheckpointCompatibilityError(
            "checkpoint restore requires a fresh trainer; target trainer has an incomplete update"
        )
    if any(
        live_attrs.get(field, 0) != 0
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
    for name, expected in (
        ("model", model),
        ("optimizer", optimizer),
        ("config", config),
        ("scheduler", scheduler),
        ("scaler", scaler),
    ):
        if live_attrs.get(name) is not expected:
            raise CheckpointCompatibilityError(
                f"checkpoint restore target {name} changed during preflight"
            )
    if pending_gradient:
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
    canonical_d02 = _is_canonical_d02(trainer)
    if canonical_d02:
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
        # Canonical D02 requires exact Python ints; generic adapters retain
        # their prior non-bool int-subclass compatibility.
        valid_type = (
            type(value) is int
            if canonical_d02
            else isinstance(value, int) and not isinstance(value, bool)
        )
        if not valid_type or value < 0:
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
    if live_config is not None:
        # D02's typed checkpoint contract must agree with direct Trainer restore
        # before D05 applies model weights or optimizer moments. Preserve generic
        # adapter compatibility when its own state loader defines loose equality.
        matches = (
            _typed_config_equal(checkpoint_config, live_config)
            if canonical_d02
            else checkpoint_config == live_config
        )
        if not matches:
            raise CheckpointCompatibilityError(
                "trainer config mismatch; refusing unsafe resume"
            )

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

    # A canonical D02 trainer refuses non-finite committed moments only after
    # loading them. Reject poisoned tensor leaves here, while the live model,
    # optimizer, counters and RNG are still untouched. Use D02's own recursive
    # numerical contract rather than introducing a different finiteness policy.
    if canonical_d02:
        require_finite = getattr(trainer, "_require_finite_state_tree", None)
        if not callable(require_finite):
            raise CheckpointCompatibilityError(
                "canonical trainer numeric-state authority unavailable"
            )
        for field in ("optimizer", "scheduler", "scaler"):
            try:
                require_finite(state.get(field), f"checkpoint {field}")
            except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
                raise CheckpointCompatibilityError(
                    f"checkpoint trainer {field} has non-finite or invalid numeric state"
                ) from exc

    # A finite optimizer group can still encode an invalid update contract
    # (for example negative decay/LR, zero eps or beta outside [0, 1)).
    safe_optimizer_check = getattr(
        trainer, "_require_safe_optimizer_hyperparameters", None
    )
    if canonical_d02:
        if not callable(safe_optimizer_check):
            raise CheckpointCompatibilityError(
                "canonical trainer optimizer-hyperparameter authority unavailable"
            )
        try:
            safe_optimizer_check(state.get("optimizer"))
        except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                "checkpoint trainer optimizer hyperparameters invalid"
            ) from exc

    # Shared D02 authority must reject finite but forged scheduler state
    # BEFORE either D05 public loader can apply model weights or restore RNG.
    # Generic third-party trainer adapters retain their original semantics.
    chronology_check = getattr(trainer, "_require_checkpoint_scheduler_chronology", None)
    if canonical_d02:
        if not callable(chronology_check):
            raise CheckpointCompatibilityError(
                "canonical trainer scheduler authority unavailable"
            )
        try:
            chronology_check(
                state.get("scheduler"), state["optimizer_step"], state.get("optimizer"),
            )
        except (ArithmeticError, ValueError, TypeError, RuntimeError) as exc:
            raise CheckpointCompatibilityError(
                "checkpoint trainer scheduler chronology mismatch"
            ) from exc

    # Native GradScaler accepts finite but invalid statistics in a detached
    # load probe. Mirror D02's single authority before model/RNG application.
    if canonical_d02:
        scaler_check = getattr(trainer, "_require_checkpoint_scaler_state", None)
        if not callable(scaler_check):
            raise CheckpointCompatibilityError(
                "canonical trainer scaler authority unavailable"
            )
        try:
            scaler_check(state.get("scaler"))
        except (ArithmeticError, ValueError, TypeError, RuntimeError) as exc:
            raise CheckpointCompatibilityError(
                "checkpoint trainer scaler statistics invalid"
            ) from exc

    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        if not callable(getattr(trainer, "load_state_dict", None)):
            raise CheckpointCompatibilityError("trainer must provide load_state_dict")
        try:
            probe = copy.deepcopy(trainer)
            probe.load_state_dict(copy.deepcopy(state))
        except Exception as exc:
            raise CheckpointCompatibilityError(
                "checkpoint trainer state failed isolated compatibility preflight"
            ) from exc
        return

    # D02's authoritative names bind serialized optimizer slots to live
    # parameters before model weights or optimizer moments can be applied.
    order_check = getattr(trainer, "_require_optimizer_state_parameter_order", None)
    if canonical_d02 and not callable(order_check):
        raise CheckpointCompatibilityError(
            "canonical trainer optimizer-order authority unavailable"
        )
    if callable(order_check):
        try:
            order_check(state.get("optimizer"))
        except (ValueError, TypeError) as exc:
            raise CheckpointCompatibilityError(
                "checkpoint optimizer parameter order/identity mismatch"
            ) from exc
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

    expected_canonical = _is_canonical_d02(trainer)
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
                _restore_ambient_rng_after_failed_apply(ambient, rng_exc)
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
                        _restore_initial_torch_policy(
                            (bool(torch_state["deterministic_algorithms"]), warn_only),
                            rng_exc,
                        )
                raise
            else:
                if warn_only is not None:
                    try:
                        torch.use_deterministic_algorithms(
                            bool(torch_state["deterministic_algorithms"]),
                            warn_only=warn_only,
                        )
                    except BaseException as mode_exc:
                        _restore_initial_torch_policy(
                            (bool(torch_state["deterministic_algorithms"]), warn_only),
                            mode_exc,
                        )
                        raise
        except BaseException as exc:
            _poison_canonical_restore_failure(
                trainer,
                expected_canonical=expected_canonical,
                reason="checkpoint_preflight_rng_rollback_failed",
                exc=exc,
            )
            raise



def _assert_live_d02_determinism(trainer: Any) -> bool | None:
    """Reject ambient torch policy drift before exporting or restoring D02."""

    if not _is_canonical_d02(trainer):
        return None
    config = vars(trainer).get("config")
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
    """Require an exact canonical D02 replay, not merely checksum-valid RNG."""

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

    # A sealed V1 artifact can be valid while omitting one or more streams.
    # Replaying only the available streams silently changes the next batch.
    missing = sorted({"python", "numpy"} - rng_state.keys())
    if missing:
        raise CheckpointCompatibilityError(
            f"canonical trainer checkpoint is missing RNG streams: {missing}"
        )
    if "cpu" not in torch_state or "cuda" not in torch_state:
        raise CheckpointCompatibilityError(
            "canonical trainer checkpoint is missing torch CPU/CUDA RNG streams"
        )
    cuda_states = torch_state["cuda"]
    if not isinstance(cuda_states, list):
        raise CheckpointCompatibilityError(
            "canonical trainer CUDA RNG streams must be a list"
        )
    torch = importlib.import_module("torch")
    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda_states) != device_count:
        raise CheckpointCompatibilityError(
            "canonical trainer CUDA RNG device count differs from checkpoint; "
            "load with restore_rng=False to opt out of exact replay"
        )
    for index, cuda_state in enumerate(cuda_states):
        try:
            probe = torch.Generator(device=f"cuda:{index}")
            probe.set_state(cuda_state.cpu())
        except (AttributeError, RuntimeError, TypeError) as exc:
            raise CheckpointCompatibilityError(
                f"canonical trainer CUDA RNG state for device {index} is invalid"
            ) from exc


def _snapshot_torch_policy(state: Mapping[str, Any]) -> tuple[bool, bool] | None:
    """Pin the live policy before any model or trainer loader can mutate it."""

    if not state.get("torch"):
        return None
    torch = importlib.import_module("torch")
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )


def _restore_initial_torch_policy(
    initial_policy: tuple[bool, bool] | None, exc: BaseException,
) -> None:
    """Best-effort rollback without masking an application-stage failure."""

    if initial_policy is None:
        return
    try:
        torch = importlib.import_module("torch")
        torch.use_deterministic_algorithms(
            initial_policy[0], warn_only=initial_policy[1],
        )
    except BaseException as mode_exc:
        exc.add_note(f"PyTorch deterministic-mode rollback also failed: {mode_exc!r}")


def _restore_ambient_rng_after_failed_apply(
    ambient: Mapping[str, Any], exc: BaseException,
) -> None:
    """Recover independent streams even if one ambient rollback setter fails."""

    try:
        _core.restore_rng_state(ambient)
        return
    except BaseException as rng_exc:
        exc.add_note(f"Ambient RNG rollback also failed: {rng_exc!r}")

    # core.restore_rng_state stops at its first failed setter. Retry each
    # independent family separately so a Python/NumPy failure cannot also
    # strand an otherwise recoverable torch CPU/CUDA stream.
    if "python" in ambient:
        try:
            _core.random.setstate(ambient["python"])
        except BaseException as rollback_exc:
            exc.add_note(f"Python RNG rollback also failed: {rollback_exc!r}")
    if "numpy" in ambient:
        try:
            _core.np.random.set_state(ambient["numpy"])
        except BaseException as rollback_exc:
            exc.add_note(f"NumPy RNG rollback also failed: {rollback_exc!r}")

    torch_state = ambient.get("torch")
    if not isinstance(torch_state, Mapping):
        return
    try:
        torch = importlib.import_module("torch")
    except BaseException as rollback_exc:
        exc.add_note(f"PyTorch RNG rollback unavailable: {rollback_exc!r}")
        return
    if "cpu" in torch_state:
        try:
            torch.set_rng_state(torch_state["cpu"].cpu())
        except BaseException as rollback_exc:
            exc.add_note(f"PyTorch CPU RNG rollback also failed: {rollback_exc!r}")
    for index, cuda_state in enumerate(torch_state.get("cuda", ())):
        try:
            torch.cuda.set_rng_state(cuda_state.cpu(), device=index)
        except BaseException as rollback_exc:
            exc.add_note(
                f"PyTorch CUDA RNG rollback on device {index} also failed: "
                f"{rollback_exc!r}"
            )


def _restore_preapply_process_state(
    ambient: Mapping[str, Any],
    policy: tuple[bool, bool] | None,
    trainer: Any,
) -> None:
    """Make effectful pre-application inspection observationally RNG-neutral."""

    expected_canonical = _is_canonical_d02(trainer)
    try:
        _core.restore_rng_state(ambient)
        if policy is not None:
            torch = importlib.import_module("torch")
            torch.use_deterministic_algorithms(
                policy[0],
                warn_only=policy[1],
            )
    except BaseException as exc:
        _restore_ambient_rng_after_failed_apply(ambient, exc)
        _restore_initial_torch_policy(policy, exc)
        _poison_canonical_restore_failure(
            trainer,
            expected_canonical=expected_canonical,
            reason="checkpoint_preapply_rng_rollback_failed",
            exc=exc,
        )
        raise


def _restore_checkpoint_rng_preserving_warn_only(
    state: Mapping[str, Any],
    *,
    restore: Any,
    initial_policy: tuple[bool, bool] | None = None,
) -> None:
    """Do not erase the live PyTorch warn-only policy on checkpoint RNG replay.

    The V1 RNG snapshot records deterministic enablement, but not warn_only.
    A canonical D02 Trainer has already configured its validated policy; the
    core RNG restore defaults warn_only to False even when it was True.
    """

    policy = initial_policy
    if policy is None:
        policy = _snapshot_torch_policy(state)
    try:
        restore(state)
        if policy is not None:
            torch = importlib.import_module("torch")
            torch.use_deterministic_algorithms(
                torch.are_deterministic_algorithms_enabled(),
                warn_only=policy[1],
            )
    except BaseException as exc:
        # Model/trainer loaders may already have changed process-global mode.
        # Roll back to the pre-application policy, not to that later value.
        _restore_initial_torch_policy(policy, exc)
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

    if not callable(getattr(trainer, "state_dict", None)):
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

    restore_bindings = _snapshot_trainer_restore_bindings(trainer)
    prebind_ambient = capture_rng_state()
    prebind_policy = _snapshot_torch_policy(prebind_ambient)
    try:
        load_trainer_state = _bind_trainer_state_loader(trainer)
    finally:
        _restore_preapply_process_state(
            prebind_ambient,
            prebind_policy,
            trainer,
        )
    _assert_trainer_restore_bindings(trainer, restore_bindings)

    _validate_expected_core_identity(
        expected_git_sha=expected_git_sha,
        expected_model_spec_hash=expected_model_spec_hash,
        expected_tokenizer_hash=expected_tokenizer_hash,
        expected_tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
        expected_dataset_manifest_hash=expected_dataset_manifest_hash,
        expected_run_manifest_hash=expected_run_manifest_hash,
    )
    _validate_expected_canonical_binding(
        expected_init_spec_hash=expected_init_spec_hash,
        expected_split_identity=expected_split_identity,
        expected_packing_hash=expected_packing_hash,
        expected_packing_version=expected_packing_version,
        expected_training_config_hash=expected_training_config_hash,
        expected_environment_lock_hash=expected_environment_lock_hash,
        expected_seed=expected_seed,
    )
    preio_ambient = capture_rng_state()
    preio_policy = _snapshot_torch_policy(preio_ambient)
    try:
        _assert_trainer_model_binding(model, trainer)
        _preflight_trainer_target(trainer)
    finally:
        _restore_preapply_process_state(
            preio_ambient,
            preio_policy,
            trainer,
        )
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
    _assert_trainer_restore_bindings(trainer, restore_bindings)
    if restore_rng:
        _preflight_rng_state(combined_state["rng"])
        _assert_d02_checkpoint_rng_policy(trainer, combined_state["rng"])
    else:
        _assert_live_d02_determinism(trainer)
    strict_model = _effective_strict_model(trainer, strict_model)
    preapply_ambient = capture_rng_state()
    preapply_policy = _snapshot_torch_policy(preapply_ambient)
    try:
        # Loader lookup/signature inspection can execute descriptors or proxies.
        # Bind both effectful restore interfaces before model materialization, then
        # revalidate the canonical target. No loader attribute is reopened later.
        # The trainer loader was already bound once under process-state guard
        # before checkpoint I/O. Reuse it so stateful descriptors cannot execute
        # a second time between final target validation and application.
        model_apply = _bind_model_state_loader(model, strict_model)
        _assert_trainer_restore_bindings(trainer, restore_bindings)
        _assert_trainer_model_binding(model, trainer)
        _preflight_trainer_state(trainer, trainer_state, manifest=manifest)
        _assert_trainer_restore_bindings(trainer, restore_bindings)
        _assert_trainer_model_binding(model, trainer)
        _preflight_trainer_target(trainer)
        _assert_trainer_restore_bindings(trainer, restore_bindings)
        materialized = _prepare_model_weights(model, arrays, strict_model)
        _assert_trainer_restore_bindings(trainer, restore_bindings)

        # The decoded source weights are no longer needed after target
        # materialization. Release them before the first live mutation so peak
        # resume memory stays bounded as checkpoint scale increases.
        del arrays

        # Materialization can execute model.state_dict() and custom tensor/device
        # conversion hooks. Revalidate the live target, but never reopen either
        # already-bound restore interface.
        _assert_trainer_model_binding(model, trainer)
        _preflight_trainer_state(trainer, trainer_state, manifest=manifest)
        _assert_trainer_restore_bindings(trainer, restore_bindings)
        _assert_trainer_model_binding(model, trainer)
        _preflight_trainer_target(trainer)
        _assert_trainer_restore_bindings(trainer, restore_bindings)
    finally:
        _restore_preapply_process_state(
            preapply_ambient,
            preapply_policy,
            trainer,
        )

    policy_before_apply = _snapshot_torch_policy(combined_state["rng"])
    ambient_before_apply = capture_rng_state()
    # An integrity-valid opt-out snapshot may omit torch; failure rollback
    # must still recover the live process-global deterministic/warn-only mode.
    rollback_policy = (
        policy_before_apply
        if policy_before_apply is not None
        else _snapshot_torch_policy(ambient_before_apply)
    )

    # State loaders may draw from process RNG even when they succeed.
    # Failed application may leave a mixed model/optimizer state, so canonical
    # D02 targets must require a fresh instance and verified checkpoint.
    try:
        model_apply(materialized)
        _assert_trainer_restore_bindings(trainer, restore_bindings)
        _assert_trainer_model_binding(model, trainer)
        load_trainer_state(trainer_state)
        _assert_trainer_restore_bindings(trainer, restore_bindings)
        if restore_rng:
            _restore_checkpoint_rng_preserving_warn_only(
                combined_state["rng"],
                restore=restore_rng_state,
                initial_policy=policy_before_apply,
            )
        else:
            _assert_live_d02_determinism(trainer)
    except BaseException as exc:
        try:
            _restore_ambient_rng_after_failed_apply(ambient_before_apply, exc)
        finally:
            _restore_initial_torch_policy(rollback_policy, exc)
        _poison_canonical_restore_failure(
            trainer,
            expected_canonical=restore_bindings[0],
            reason="checkpoint_restore_apply_failed",
            exc=exc,
        )
        raise
    return LoadResult(
        manifest=copy.deepcopy(manifest),
        trainer_state=trainer_state,
        rng_state=combined_state["rng"],
    )
