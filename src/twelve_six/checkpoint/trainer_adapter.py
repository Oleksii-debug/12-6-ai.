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
from types import FunctionType, GetSetDescriptorType, MemberDescriptorType
from typing import Any

from ..training.config import TrainerConfig as _CanonicalTrainerConfig
from ..training.trainer import Trainer as _CanonicalTrainer
from . import core as _core
from .core import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    LoadResult,
    _bind_model_state_loader,
    _decode_verified_state,
    _preflight_optimizer_state,
    _preflight_rng_state,
    _prepare_model_weights,
    _semantic_stateful_probe,
    assert_identity,
    capture_rng_state,
    prepare_checkpoint_load,
    restore_rng_state,
    save_checkpoint,
)
from .expected_binding import (
    _validate_expected_canonical_binding,
    _validate_expected_core_identity,
)

_NATIVE_D02_CHECKPOINT_SAFETY_AUTHORITIES = (
    _CanonicalTrainer._CHECKPOINT_SAFETY_AUTHORITIES
)
_NATIVE_D02_CHECKPOINT_STORAGE_FIELDS = (
    _CanonicalTrainer._CHECKPOINT_STORAGE_FIELDS
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


def _is_native_d02(trainer: Any) -> bool:
    """Recognize real D02 lineage without dispatching a custom metaclass."""

    trainer_type = type(trainer)
    mro = type.__getattribute__(trainer_type, "__mro__")
    return _CanonicalTrainer in mro


def _trainer_instance_attrs(trainer: Any) -> dict[str, Any]:
    """Read native Trainer instance storage without subclass descriptor dispatch."""

    if not _is_native_d02(trainer):
        return vars(trainer)

    namespace = type.__getattribute__(_CanonicalTrainer, "__dict__")
    descriptor = namespace.get("__dict__")
    if not isinstance(descriptor, GetSetDescriptorType):
        raise TypeError("canonical Trainer instance dictionary authority unavailable")
    try:
        attrs = descriptor.__get__(trainer, type(trainer))
    except (AttributeError, TypeError) as exc:
        raise TypeError("native D02 trainer instance storage unavailable") from exc
    if type(attrs) is not dict:
        raise TypeError("native D02 trainer instance storage must be a dictionary")
    return attrs


def _is_canonical_d02(trainer: Any) -> bool:
    """Identify the D02 recovery protocol without native subclass dispatch."""

    try:
        attrs = _trainer_instance_attrs(trainer)
    except TypeError:
        return False
    return "_failure_reason" in attrs and "_update_incomplete" in attrs


def _require_canonical_d02_markers(trainer: Any) -> bool:
    """Prevent a real Trainer from downgrading into generic adapter semantics."""

    canonical_d02 = _is_canonical_d02(trainer)
    if _is_native_d02(trainer) and not canonical_d02:
        raise CheckpointCompatibilityError(
            "native D02 trainer recovery markers are unavailable"
        )
    return canonical_d02


def _assert_native_d02_checkpoint_safety_lineage(trainer: Any) -> None:
    """Reject subclass or instance replacement of first-party safety code."""

    if not _is_native_d02(trainer):
        return
    try:
        instance_attrs = _trainer_instance_attrs(trainer)
    except TypeError as exc:
        raise CheckpointCompatibilityError(
            "native D02 trainer does not expose checkpoint safety state"
        ) from exc
    for name in _NATIVE_D02_CHECKPOINT_SAFETY_AUTHORITIES:
        if name in instance_attrs:
            raise CheckpointCompatibilityError(
                f"native D02 safety authority must remain canonical: {name}"
            )
        canonical = inspect.getattr_static(_CanonicalTrainer, name, None)
        resolved = inspect.getattr_static(type(trainer), name, None)
        if canonical is None or resolved is not canonical:
            raise CheckpointCompatibilityError(
                f"native D02 safety authority must remain canonical: {name}"
            )
    for name in _NATIVE_D02_CHECKPOINT_STORAGE_FIELDS:
        canonical = inspect.getattr_static(_CanonicalTrainer, name, None)
        resolved = inspect.getattr_static(type(trainer), name, None)
        if resolved is not canonical:
            raise CheckpointCompatibilityError(
                "native D02 restore storage descriptor must remain canonical: "
                f"{name}"
            )


def _run_native_d02_checkpoint_safety(trainer: Any) -> None:
    """Run canonical checkpoint-safe validation outside subclass export hooks."""

    if not _is_native_d02(trainer):
        return
    _assert_native_d02_checkpoint_safety_lineage(trainer)
    canonical = inspect.getattr_static(
        _CanonicalTrainer,
        "assert_checkpoint_safe",
        None,
    )
    if not isinstance(canonical, FunctionType):
        raise CheckpointCompatibilityError(
            "native D02 checkpoint safety authority is unavailable"
        )
    canonical.__get__(trainer, type(trainer))()
    _assert_native_d02_checkpoint_safety_lineage(trainer)


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
        attrs = _trainer_instance_attrs(trainer)
    except TypeError as poison_exc:
        _add_failure_note_preserving_primary(
            exc,
            "canonical trainer recovery-state access also failed: "
            f"{poison_exc!r}"
        )
        return
    if attrs.get("_failure_reason") is None:
        attrs["_failure_reason"] = reason
    attrs["_update_incomplete"] = True


def _snapshot_restore_contract_value(value: Any, *, path: str) -> Any:
    """Copy small restore-contract data without invoking arbitrary object hooks."""

    if value is None or type(value) in {bool, int, float, str, bytes}:
        return value
    if type(value) is tuple:
        return tuple(
            _snapshot_restore_contract_value(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        )
    if type(value) is list:
        return [
            _snapshot_restore_contract_value(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if type(value) is dict:
        copied: dict[str, Any] = {}
        for key, child in value.items():
            if type(key) is not str:
                raise CheckpointCompatibilityError(
                    f"{path} contains a non-string restore-contract key"
                )
            copied[key] = _snapshot_restore_contract_value(
                child,
                path=f"{path}.{key}",
            )
        return copied
    raise CheckpointCompatibilityError(
        f"{path} contains unsupported restore-contract data"
    )


def _restore_contract_equal(left: Any, right: Any) -> bool:
    """Compare small restore-contract data without custom equality hooks."""

    if type(left) is not type(right):
        return False
    if left is None:
        return True
    if type(left) in {bool, int, str, bytes}:
        return bool(left == right)
    if type(left) is float:
        return left.hex() == right.hex()
    if type(left) is tuple:
        return len(left) == len(right) and all(
            _restore_contract_equal(a, b)
            for a, b in zip(left, right, strict=True)
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _restore_contract_equal(a, b)
            for a, b in zip(left, right, strict=True)
        )
    if type(left) is dict:
        if left.keys() != right.keys():
            return False
        return all(
            _restore_contract_equal(left[key], right[key])
            for key in left
        )
    return False


def _snapshot_native_d02_config(config: Any) -> dict[str, Any]:
    """Snapshot native TrainerConfig without executing mutable field descriptors."""

    config_type = type(config)
    try:
        type_attrs = type.__getattribute__(config_type, "__dict__")
    except (AttributeError, TypeError) as exc:
        raise CheckpointCompatibilityError(
            "native D02 trainer config type is unavailable"
        ) from exc
    raw_fields = type_attrs.get("__dataclass_fields__")
    if config_type is not _CanonicalTrainerConfig or type(raw_fields) is not dict:
        raise CheckpointCompatibilityError(
            "native D02 trainer config must remain canonical TrainerConfig"
        )

    snapshot: dict[str, Any] = {}
    for field_name in raw_fields:
        if type(field_name) is not str:
            raise CheckpointCompatibilityError(
                "native D02 trainer config contains an invalid field name"
            )
        descriptor = type_attrs.get(field_name)
        if not isinstance(descriptor, MemberDescriptorType):
            raise CheckpointCompatibilityError(
                "native D02 trainer config fields must remain inert slots"
            )
        try:
            value = descriptor.__get__(config, config_type)
        except BaseException as exc:
            raise CheckpointCompatibilityError(
                f"native D02 config field unavailable: {field_name}"
            ) from exc
        snapshot[field_name] = _snapshot_restore_contract_value(
            value,
            path=f"native D02 config.{field_name}",
        )
    return snapshot


def _snapshot_native_d02_model_contract(
    trainer: Any,
) -> tuple[tuple[Any, ...], ...]:
    """Pin non-serialized model topology/trainability across restore callouts."""

    attrs = _trainer_instance_attrs(trainer)
    model = attrs.get("model")
    torch = importlib.import_module("torch")
    module_type = torch.nn.Module
    parameter_type = torch.nn.Parameter
    tensor_type = torch.Tensor
    if not isinstance(model, module_type):
        raise CheckpointCompatibilityError(
            "native D02 model contract requires a torch module"
        )

    requires_grad_descriptor = inspect.getattr_static(
        tensor_type,
        "requires_grad",
        None,
    )
    if not isinstance(requires_grad_descriptor, GetSetDescriptorType):
        raise CheckpointCompatibilityError(
            "native D02 tensor trainability authority is unavailable"
        )

    entries: list[tuple[Any, ...]] = []
    seen: set[int] = set()
    active: set[int] = set()

    def walk(module: Any, prefix: str) -> None:
        module_id = id(module)
        if module_id in active:
            raise CheckpointCompatibilityError(
                "native D02 model contract contains a module cycle"
            )
        if module_id in seen:
            return
        active.add(module_id)
        seen.add(module_id)
        try:
            try:
                module_attrs = _CanonicalTrainer._raw_instance_dict(
                    module,
                    module_type,
                    label="checkpoint model contract module",
                )
            except (AttributeError, RuntimeError, TypeError) as exc:
                raise CheckpointCompatibilityError(
                    "native D02 model contract storage is unavailable"
                ) from exc

            training = module_attrs.get("training")
            parameters = module_attrs.get("_parameters")
            buffers = module_attrs.get("_buffers")
            children = module_attrs.get("_modules")
            non_persistent = module_attrs.get("_non_persistent_buffers_set")
            if type(training) is not bool:
                raise CheckpointCompatibilityError(
                    "native D02 model contract training flag is invalid"
                )
            if (
                type(parameters) is not dict
                or type(buffers) is not dict
                or type(children) is not dict
                or type(non_persistent) is not set
                or any(type(name) is not str for name in non_persistent)
            ):
                raise CheckpointCompatibilityError(
                    "native D02 model contract registries are not canonical"
                )

            entries.append(
                (
                    "module",
                    prefix,
                    module_id,
                    _CanonicalTrainer._type_identity(module),
                    training,
                    tuple(sorted(non_persistent)),
                )
            )

            for name, parameter in parameters.items():
                if type(name) is not str:
                    raise CheckpointCompatibilityError(
                        "native D02 model parameter name is not canonical"
                    )
                full_name = f"{prefix}{name}"
                if parameter is None:
                    entries.append(("parameter", full_name, None, None, None))
                    continue
                if not isinstance(parameter, parameter_type):
                    raise CheckpointCompatibilityError(
                        "native D02 model parameter binding is not canonical"
                    )
                try:
                    requires_grad = requires_grad_descriptor.__get__(
                        parameter,
                        type(parameter),
                    )
                except (AttributeError, RuntimeError, TypeError) as exc:
                    raise CheckpointCompatibilityError(
                        f"native D02 parameter trainability is unavailable: {full_name}"
                    ) from exc
                if type(requires_grad) is not bool:
                    raise CheckpointCompatibilityError(
                        f"native D02 parameter trainability is invalid: {full_name}"
                    )
                entries.append(
                    (
                        "parameter",
                        full_name,
                        id(parameter),
                        _CanonicalTrainer._type_identity(parameter),
                        requires_grad,
                    )
                )

            for name, buffer in buffers.items():
                if type(name) is not str:
                    raise CheckpointCompatibilityError(
                        "native D02 model buffer name is not canonical"
                    )
                full_name = f"{prefix}{name}"
                if buffer is None:
                    entries.append(
                        ("buffer", full_name, None, None, name in non_persistent)
                    )
                    continue
                if not isinstance(buffer, tensor_type):
                    raise CheckpointCompatibilityError(
                        "native D02 model buffer binding is not canonical"
                    )
                entries.append(
                    (
                        "buffer",
                        full_name,
                        id(buffer),
                        _CanonicalTrainer._type_identity(buffer),
                        name in non_persistent,
                    )
                )

            for name, child in children.items():
                if type(name) is not str:
                    raise CheckpointCompatibilityError(
                        "native D02 model child name is not canonical"
                    )
                full_name = f"{prefix}{name}"
                if child is None:
                    entries.append(("child", full_name, None, None))
                    continue
                if not isinstance(child, module_type):
                    raise CheckpointCompatibilityError(
                        "native D02 model child binding is not canonical"
                    )
                entries.append(
                    (
                        "child",
                        full_name,
                        id(child),
                        _CanonicalTrainer._type_identity(child),
                    )
                )
                walk(child, f"{full_name}.")
        finally:
            active.remove(module_id)

    walk(model, "")
    return tuple(entries)


def _snapshot_trainer_restore_bindings(
    trainer: Any,
) -> tuple[bool, dict[str, Any]]:
    """Pin canonical restore component identities without descriptor dispatch."""

    native_d02 = _is_native_d02(trainer)
    canonical_d02 = _require_canonical_d02_markers(trainer)
    if not canonical_d02:
        return False, {}
    if native_d02:
        _assert_native_d02_checkpoint_safety_lineage(trainer)
    attrs = _trainer_instance_attrs(trainer)
    component_fields = ("model", "optimizer", "scheduler", "scaler", "config")
    if native_d02:
        binding_fields = (*component_fields, "device")
        missing_bindings = [field for field in binding_fields if field not in attrs]
        if missing_bindings:
            raise CheckpointCompatibilityError(
                "native D02 trainer is missing restore binding fields: "
                f"{missing_bindings}"
            )
        bindings = {field: attrs[field] for field in binding_fields}
    else:
        bindings = {
            field: attrs[field]
            for field in component_fields
            if field in attrs
        }
    policy_fields = (
        "_canonical_default_schedule",
        "_canonical_unscheduled_default_optimizer",
        "_canonical_default_optimizer_options",
    )
    if native_d02:
        missing_policies = [field for field in policy_fields if field not in attrs]
        if missing_policies:
            raise CheckpointCompatibilityError(
                "native D02 trainer is missing restore policy fields: "
                f"{missing_policies}"
            )
        policies = {
            field: _snapshot_restore_contract_value(
                attrs[field],
                path=f"native D02 {field}",
            )
            for field in policy_fields
        }
    else:
        policies = {}
    config = (
        _snapshot_native_d02_config(bindings["config"])
        if native_d02 and "config" in bindings
        else None
    )
    model_contract = (
        _snapshot_native_d02_model_contract(trainer)
        if native_d02
        else None
    )
    return True, {
        "native_d02": native_d02,
        "bindings": bindings,
        "policies": policies,
        "config": config,
        "model_contract": model_contract,
    }


def _assert_trainer_restore_bindings(
    trainer: Any,
    snapshot: tuple[bool, dict[str, Any]],
) -> None:
    """Reject safety classification or restore-component identity drift."""

    expected_canonical, snapshot_state = snapshot
    current_canonical = _is_canonical_d02(trainer)
    if current_canonical != expected_canonical:
        exc = CheckpointCompatibilityError(
            "trainer safety classification changed during checkpoint restore"
        )
        _poison_canonical_restore_failure(
            trainer,
            expected_canonical=expected_canonical,
            reason="checkpoint_restore_target_drift",
            exc=exc,
        )
        raise exc
    if not expected_canonical:
        return
    current_native = _is_native_d02(trainer)
    if current_native != snapshot_state["native_d02"]:
        exc = CheckpointCompatibilityError(
            "trainer native D02 classification changed during checkpoint restore"
        )
        _poison_canonical_restore_failure(
            trainer,
            expected_canonical=expected_canonical,
            reason="checkpoint_restore_target_drift",
            exc=exc,
        )
        raise exc
    if current_native:
        _assert_native_d02_checkpoint_safety_lineage(trainer)
    try:
        attrs = _trainer_instance_attrs(trainer)
    except TypeError as exc:
        raise CheckpointCompatibilityError(
            "canonical trainer does not expose instance recovery state"
        ) from exc
    sentinel = object()
    for field, expected in snapshot_state["bindings"].items():
        if attrs.get(field, sentinel) is not expected:
            raise CheckpointCompatibilityError(
                f"canonical trainer {field} binding changed during checkpoint restore"
            )
    expected_config = snapshot_state["config"]
    if expected_config is not None:
        current_config = attrs.get("config", sentinel)
        try:
            current_config = _snapshot_native_d02_config(current_config)
        except CheckpointCompatibilityError as exc:
            raise CheckpointCompatibilityError(
                "canonical trainer config changed during checkpoint restore"
            ) from exc
        if not _restore_contract_equal(current_config, expected_config):
            raise CheckpointCompatibilityError(
                "canonical trainer config changed during checkpoint restore"
            )
    expected_model_contract = snapshot_state["model_contract"]
    if expected_model_contract is not None:
        current_model_contract = _snapshot_native_d02_model_contract(trainer)
        if not _restore_contract_equal(current_model_contract, expected_model_contract):
            raise CheckpointCompatibilityError(
                "canonical trainer model contract changed during checkpoint restore"
            )
    for field, expected in snapshot_state["policies"].items():
        if (
            field not in attrs
            or not _restore_contract_equal(attrs[field], expected)
        ):
            raise CheckpointCompatibilityError(
                f"canonical trainer {field} policy changed during checkpoint restore"
            )


def _note_restore_binding_drift(
    trainer: Any,
    snapshot: tuple[bool, dict[str, Any]],
    exc: BaseException,
) -> None:
    """Preserve a primary callout error while recording target drift."""

    try:
        _assert_trainer_restore_bindings(trainer, snapshot)
    except CheckpointCompatibilityError as drift_exc:
        _add_failure_note_preserving_primary(
            exc,
            "trainer restore target drift also detected: "
            f"{drift_exc}"
        )


def _assert_native_d02_model_training_mode(model: Any, trainer: Any) -> None:
    """Require the non-serialized nn.Module training-mode resume invariant."""

    if not _is_native_d02(trainer):
        return
    torch = importlib.import_module("torch")
    module_type = torch.nn.Module
    if not isinstance(model, module_type):
        raise CheckpointCompatibilityError(
            "native D02 checkpoint model is not a torch module"
        )

    seen: set[int] = set()
    active: set[int] = set()

    def walk(module: Any, path: str) -> None:
        module_id = id(module)
        if module_id in active:
            raise CheckpointCompatibilityError(
                "native D02 checkpoint model module graph contains a cycle"
            )
        if module_id in seen:
            return
        active.add(module_id)
        seen.add(module_id)
        try:
            try:
                attrs = _CanonicalTrainer._raw_instance_dict(
                    module,
                    module_type,
                    label="checkpoint model module",
                )
            except (AttributeError, RuntimeError, TypeError) as exc:
                raise CheckpointCompatibilityError(
                    "native D02 checkpoint model does not expose training mode"
                ) from exc
            if attrs.get("training") is not True:
                raise CheckpointCompatibilityError(
                    "native D02 checkpoint restore requires model training mode"
                )
            children = attrs.get("_modules")
            if type(children) is not dict:
                raise CheckpointCompatibilityError(
                    "native D02 checkpoint model child registry is not canonical"
                )
            for name, child in children.items():
                if type(name) is not str:
                    raise CheckpointCompatibilityError(
                        "native D02 checkpoint model child name is not canonical"
                    )
                if child is None:
                    continue
                if not isinstance(child, module_type):
                    raise CheckpointCompatibilityError(
                        "native D02 checkpoint model child is not a torch module"
                    )
                walk(child, f"{path}.{name}" if path else name)
        finally:
            active.remove(module_id)

    walk(model, "")


def _assert_trainer_model_binding(model: Any, trainer: Any) -> None:
    """Refuse mismatched D02 model owners without executing custom descriptors."""

    if not _require_canonical_d02_markers(trainer):
        return
    attrs = _trainer_instance_attrs(trainer)
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

    return strict_model or _is_native_d02(trainer) or _is_canonical_d02(trainer)


def _bind_exact_native_d02_authority(
    trainer: Any,
    name: str,
    *,
    positional_args: int,
) -> Any | None:
    """Bind one exact first-party D02 safety authority without subclass dispatch."""

    if not _is_native_d02(trainer):
        return None
    canonical = inspect.getattr_static(_CanonicalTrainer, name, None)
    resolved = inspect.getattr_static(type(trainer), name, None)
    if not isinstance(canonical, FunctionType) or resolved is not canonical:
        raise CheckpointCompatibilityError(
            f"native D02 safety authority must remain canonical: {name}"
        )
    bound = canonical.__get__(trainer, type(trainer))
    try:
        inspect.signature(bound).bind(*([None] * positional_args))
    except (TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(
            f"native D02 safety authority cannot bind safely: {name}"
        ) from exc
    return bound


def _bind_native_export_live_authorities(trainer: Any) -> dict[str, Any]:
    """Bind exact D02 exported-vs-live authorities without subclass overrides."""

    if not _is_native_d02(trainer):
        return {}
    canonical_leaf_equal = inspect.getattr_static(
        _CanonicalTrainer,
        "_exact_export_leaf_equal",
        None,
    )
    resolved_leaf_equal = inspect.getattr_static(
        type(trainer),
        "_exact_export_leaf_equal",
        None,
    )
    if resolved_leaf_equal is not canonical_leaf_equal:
        raise CheckpointCompatibilityError(
            "native D02 safety authority must remain canonical: "
            "_exact_export_leaf_equal"
        )
    authorities: dict[str, Any] = {}
    for name in (
        "_require_exported_scheduler_matches_live",
        "_require_exported_scaler_matches_live",
        "_require_exported_optimizer_matches_live",
    ):
        bound = _bind_exact_native_d02_authority(
            trainer,
            name,
            positional_args=1,
        )
        assert bound is not None
        authorities[name] = bound
    return authorities


def _bind_native_model_export_validator(trainer: Any) -> Any | None:
    """Bind exact D02 staged-model/live-state equality authority."""

    return _bind_exact_native_d02_authority(
        trainer,
        "_require_exported_model_matches_live",
        positional_args=1,
    )


def _bind_native_model_export_fingerprint(trainer: Any) -> Any | None:
    """Bind the exact first-party model fingerprint authority."""

    return _bind_exact_native_d02_authority(
        trainer,
        "_model_export_fingerprint",
        positional_args=0,
    )


def _bind_native_auxiliary_fingerprint(trainer: Any) -> Any | None:
    """Bind the exact first-party inert auxiliary fingerprint authority."""

    return _bind_exact_native_d02_authority(
        trainer,
        "_checkpoint_auxiliary_fingerprint",
        positional_args=0,
    )


def _bind_trainer_state_exporter(trainer: Any) -> Any:
    """Bind one checkpoint exporter without executing native instance lookup."""

    if _is_native_d02(trainer):
        try:
            instance_attrs = _trainer_instance_attrs(trainer)
        except TypeError as exc:
            raise CheckpointCompatibilityError(
                "native D02 trainer does not expose checkpoint exporter state"
            ) from exc
        if "state_dict" in instance_attrs:
            raise CheckpointCompatibilityError(
                "native D02 trainer state_dict must remain class-bound"
            )
        class_exporter = inspect.getattr_static(
            type(trainer),
            "state_dict",
            None,
        )
        if not isinstance(class_exporter, FunctionType):
            raise CheckpointCompatibilityError(
                "native D02 trainer state_dict must remain class-bound"
            )
        exporter = class_exporter.__get__(trainer, type(trainer))
        try:
            inspect.signature(exporter).bind()
        except (TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError(
                "trainer state_dict cannot safely bind checkpoint export"
            ) from exc
        return exporter

    exporter = getattr(trainer, "state_dict", None)
    if not callable(exporter):
        raise TypeError("trainer must provide state_dict()")
    return exporter


def _bind_trainer_state_loader(trainer: Any) -> Any:
    # Canonical D02 must fail closed before model mutation when its restore
    # invocation cannot accept the one authoritative trainer-state payload.
    # Native binding avoids executing an instance shadow or __getattribute__.
    # Generic adapters retain the historical permissive callable contract.
    canonical_d02 = _require_canonical_d02_markers(trainer)
    if _is_native_d02(trainer):
        try:
            instance_attrs = _trainer_instance_attrs(trainer)
        except TypeError as exc:
            raise CheckpointCompatibilityError(
                "native D02 trainer does not expose checkpoint loader state"
            ) from exc
        if "load_state_dict" in instance_attrs:
            raise CheckpointCompatibilityError(
                "native D02 trainer load_state_dict must remain class-bound"
            )
        class_loader = inspect.getattr_static(
            type(trainer),
            "load_state_dict",
            None,
        )
        if not isinstance(class_loader, FunctionType):
            raise CheckpointCompatibilityError(
                "native D02 trainer load_state_dict must remain class-bound"
            )
        loader = class_loader.__get__(trainer, type(trainer))
    else:
        loader = getattr(trainer, "load_state_dict", None)
        if not callable(loader):
            raise TypeError("trainer must provide load_state_dict()")

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

    if not _require_canonical_d02_markers(trainer):
        return
    initial_attrs = _trainer_instance_attrs(trainer)
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
    _assert_native_d02_model_training_mode(model, trainer)
    scheduler = initial_attrs.get("scheduler")
    scaler = initial_attrs.get("scaler")
    native_d02 = _is_native_d02(trainer)
    config_snapshot = (
        _snapshot_native_d02_config(config)
        if native_d02
        else None
    )

    # Bind every effectful authority/interface lookup before the final
    # freshness snapshot. Descriptor/proxy lookup itself may execute user code;
    # any such side effect must therefore be visible to the checks below.
    authorities: dict[str, Any] = {}
    if native_d02:
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

    zero_grad = getattr(optimizer, "zero_grad", None) if optimizer is not None else None
    if optimizer is not None and not callable(zero_grad):
        raise CheckpointCompatibilityError(
            "canonical trainer optimizer zero_grad unavailable"
        )

    # D02 validates optimizer/model parameter ownership during its actual load.
    # Invoke the already-bound authority before the final freshness snapshot so
    # an effectful custom authority cannot mutate the target after that snapshot.
    if native_d02:
        try:
            authorities["_require_optimizer_parameter_coverage"]()
        except Exception as exc:
            raise CheckpointCompatibilityError(
                "checkpoint restore requires valid optimizer ownership of model parameters"
            ) from exc

    if native_d02:
        try:
            authorities["_require_optimizer_parameter_coverage"]()
        except Exception as exc:
            raise CheckpointCompatibilityError(
                "checkpoint restore requires stable optimizer ownership of model parameters"
            ) from exc
        try:
            authorities["_require_no_residual_model_gradients"]()
        except Exception as exc:
            raise CheckpointCompatibilityError(
                "checkpoint restore requires a fresh trainer with no pending gradients"
            ) from exc

    # Global deterministic mode is a pure target compatibility precondition.
    # Run it after the effectful bindings/calls above, then close with the
    # freshness/identity checks that gate checkpoint I/O and model mutation.
    _assert_live_d02_determinism(trainer)
    # Canonical D02 stores its recovery flags, counters and restore components
    # as instance attributes. Take one descriptor-free final snapshot so a
    # late property/proxy read cannot mutate an earlier checked field.
    try:
        live_attrs = _trainer_instance_attrs(trainer)
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
    if (
        native_d02
        and not _restore_contract_equal(
            _snapshot_native_d02_config(config),
            config_snapshot,
        )
    ):
        raise CheckpointCompatibilityError(
            "checkpoint restore target config changed during preflight"
        )
    _assert_native_d02_model_training_mode(model, trainer)


def _assert_native_d02_postload_snapshot(trainer: Any, state: Any) -> None:
    """Verify descriptor-free committed D02 state after effectful post-load work."""

    if not _is_native_d02(trainer):
        return
    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError("checkpoint trainer state must be a mapping")
    attrs = _trainer_instance_attrs(trainer)
    if (
        attrs.get("_failure_reason") is not None
        or attrs.get("_update_incomplete")
    ):
        raise CheckpointCompatibilityError(
            "canonical trainer remained poisoned after checkpoint restore"
        )
    for field in ("micro_step", "optimizer_step", "tokens_seen"):
        live = attrs.get(field)
        expected = state.get(field)
        if type(live) is not int or live != expected:
            raise CheckpointCompatibilityError(
                f"canonical trainer post-load {field} disagrees with checkpoint"
            )
    if (
        attrs.get("_pending_tokens") != 0
        or attrs.get("_pending_loss_sum") != 0.0
    ):
        raise CheckpointCompatibilityError(
            "canonical trainer retained pending accumulation after checkpoint restore"
        )

    live_config = _snapshot_native_d02_config(attrs.get("config"))
    if not _typed_config_equal(state.get("config"), live_config):
        raise CheckpointCompatibilityError(
            "canonical trainer post-load config disagrees with checkpoint"
        )


def _assert_native_d02_inert_determinism(trainer: Any) -> None:
    """Verify native D02 torch policy without dispatching trainer safety hooks."""

    if not _is_native_d02(trainer):
        return
    attrs = _trainer_instance_attrs(trainer)
    config_state = _snapshot_native_d02_config(attrs.get("config"))
    enabled = config_state.get("deterministic_algorithms")
    warn_only = config_state.get("deterministic_warn_only")
    if type(enabled) is not bool or type(warn_only) is not bool:
        raise CheckpointCompatibilityError(
            "native D02 deterministic policy configuration is invalid"
        )
    torch = importlib.import_module("torch")
    if (
        torch.are_deterministic_algorithms_enabled() != enabled
        or torch.is_deterministic_algorithms_warn_only_enabled() != warn_only
    ):
        raise CheckpointCompatibilityError(
            "live torch deterministic policy disagrees with canonical trainer configuration"
        )


def _assert_native_d02_inert_live_state(
    trainer: Any,
    state: Any,
    *,
    model_fingerprint: Any | None,
    sealed_model_fingerprint: str | None,
    auxiliary_fingerprint: Any | None,
    sealed_auxiliary_fingerprint: str | None,
    phase: str,
) -> None:
    """Seal native committed state using only descriptor-free/raw observers."""

    if not _is_native_d02(trainer):
        return
    if (
        model_fingerprint is None
        or sealed_model_fingerprint is None
        or auxiliary_fingerprint is None
        or sealed_auxiliary_fingerprint is None
    ):
        raise CheckpointCompatibilityError(
            "native D02 inert exact-state authority is unavailable"
        )
    _assert_native_d02_postload_snapshot(trainer, state)
    _assert_native_d02_inert_determinism(trainer)
    try:
        current_model_fingerprint = model_fingerprint()
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"canonical trainer model changed during {phase}"
        ) from exc
    if current_model_fingerprint != sealed_model_fingerprint:
        raise CheckpointCompatibilityError(
            f"canonical trainer model changed during {phase}"
        )
    try:
        current_auxiliary_fingerprint = auxiliary_fingerprint()
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"canonical trainer auxiliary state changed during {phase}"
        ) from exc
    if current_auxiliary_fingerprint != sealed_auxiliary_fingerprint:
        raise CheckpointCompatibilityError(
            f"canonical trainer auxiliary state changed during {phase}"
        )
    _assert_native_d02_inert_determinism(trainer)
    _assert_native_d02_postload_snapshot(trainer, state)


def _assert_native_d02_exact_live_state(
    trainer: Any,
    state: Any,
    *,
    model_fingerprint: Any | None,
    sealed_model_fingerprint: str | None,
    auxiliary_fingerprint: Any | None,
    sealed_auxiliary_fingerprint: str | None,
    export_live_authorities: Mapping[str, Any],
    phase: str,
) -> None:
    """Prove native model and auxiliary state still equal the accepted snapshot."""

    if not _is_native_d02(trainer):
        return
    if (
        model_fingerprint is None
        or sealed_model_fingerprint is None
        or auxiliary_fingerprint is None
        or sealed_auxiliary_fingerprint is None
    ):
        raise CheckpointCompatibilityError(
            "native D02 exact-state authority is unavailable"
        )

    _assert_native_d02_inert_live_state(
        trainer,
        state,
        model_fingerprint=model_fingerprint,
        sealed_model_fingerprint=sealed_model_fingerprint,
        auxiliary_fingerprint=auxiliary_fingerprint,
        sealed_auxiliary_fingerprint=sealed_auxiliary_fingerprint,
        phase=phase,
    )
    try:
        export_live_authorities[
            "_require_exported_scheduler_matches_live"
        ](state.get("scheduler"))
        export_live_authorities[
            "_require_exported_scaler_matches_live"
        ](state.get("scaler"))
        export_live_authorities[
            "_require_exported_optimizer_matches_live"
        ](state.get("optimizer"))
    except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(
            f"canonical trainer auxiliary state changed during {phase}"
        ) from exc

    # Close the effectful comparison chain with raw observers only.
    _assert_native_d02_inert_live_state(
        trainer,
        state,
        model_fingerprint=model_fingerprint,
        sealed_model_fingerprint=sealed_model_fingerprint,
        auxiliary_fingerprint=auxiliary_fingerprint,
        sealed_auxiliary_fingerprint=sealed_auxiliary_fingerprint,
        phase=phase,
    )


def _postflight_trainer_state(trainer: Any, state: Any) -> None:
    """Verify native D02 live state across effectful post-load authorities."""

    if not _is_native_d02(trainer):
        return
    _assert_native_d02_postload_snapshot(trainer, state)

    for authority, label in (
        ("_require_optimizer_parameter_coverage", "optimizer coverage"),
        ("_require_finite_auxiliary_state", "auxiliary state"),
        ("_require_finite_committed_update", "committed update"),
        ("_require_no_residual_model_gradients", "gradient cleanliness"),
        ("_require_deterministic_policy", "deterministic policy"),
    ):
        check = getattr(trainer, authority, None)
        # Descriptor lookup itself can execute arbitrary code. Recheck the
        # committed state before invoking the returned authority.
        _assert_native_d02_postload_snapshot(trainer, state)
        if not callable(check):
            raise CheckpointCompatibilityError(
                f"canonical trainer post-load {label} authority unavailable"
            )
        try:
            check()
        except Exception as exc:
            raise CheckpointCompatibilityError(
                f"canonical trainer post-load {label} invalid"
            ) from exc
        # The authority can also mutate counters/config/pending accounting while
        # returning success. Never report a clean resume after such drift.
        _assert_native_d02_postload_snapshot(trainer, state)


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
    canonical_d02 = _require_canonical_d02_markers(trainer)
    native_d02 = _is_native_d02(trainer)
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
        _assert_native_checkpoint_config_identity(
            trainer,
            state,
            identity=identity,
        )

    if native_d02:
        live_attrs = _trainer_instance_attrs(trainer)
        live_config = _snapshot_native_d02_config(live_attrs["config"])
    else:
        live_attrs = None
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
    if native_d02:
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
    if native_d02:
        safe_optimizer_check = getattr(
            trainer, "_require_safe_optimizer_hyperparameters", None
        )
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
    if native_d02:
        chronology_check = getattr(
            trainer,
            "_require_checkpoint_scheduler_chronology",
            None,
        )
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
    if native_d02:
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

    optimizer = (
        live_attrs["optimizer"]
        if native_d02 and live_attrs is not None
        else getattr(trainer, "optimizer", None)
    )
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
    if native_d02 and not callable(order_check):
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
    scheduler = (
        live_attrs["scheduler"]
        if native_d02 and live_attrs is not None
        else getattr(trainer, "scheduler", None)
    )
    scaler = (
        live_attrs["scaler"]
        if native_d02 and live_attrs is not None
        else getattr(trainer, "scaler", None)
    )
    _preflight_stateful_component(
        scheduler,
        state.get("scheduler"),
        label="scheduler",
    )
    _preflight_stateful_component(
        scaler,
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

    restore_bindings = _snapshot_trainer_restore_bindings(trainer)
    expected_canonical = restore_bindings[0]
    ambient = capture_rng_state()
    execution_mode = _snapshot_torch_execution_mode()
    torch_state = ambient.get("torch")
    warn_only = None
    if torch_state is not None:
        torch = importlib.import_module("torch")
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        try:
            _preflight_trainer_state_without_rng_guard(
                trainer, state, manifest=manifest,
            )
        except BaseException as exc:
            _note_restore_binding_drift(
                trainer,
                restore_bindings,
                exc,
            )
            raise
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
                    except BaseException as mode_exc:  # noqa: BLE001
                        _add_failure_note_preserving_primary(
                            rng_exc,
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
                _assert_ambient_process_state_stable(
                    ambient,
                    expected_canonical=expected_canonical,
                )
                _assert_torch_execution_mode_stable(
                    execution_mode,
                    expected_canonical=expected_canonical,
                )
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

    if not _require_canonical_d02_markers(trainer):
        return None
    config = _trainer_instance_attrs(trainer).get("config")
    if _is_native_d02(trainer):
        config_state = _snapshot_native_d02_config(config)
        enabled = config_state.get("deterministic_algorithms")
        warn_only = config_state.get("deterministic_warn_only")
    else:
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

    config = _trainer_instance_attrs(trainer).get("config")
    if _is_native_d02(trainer):
        configured_warn_only = _snapshot_native_d02_config(config).get(
            "deterministic_warn_only"
        )
    else:
        configured_warn_only = getattr(config, "deterministic_warn_only", None)
    checkpoint_warn_only = torch_state.get("deterministic_warn_only")
    if checkpoint_warn_only is not None and (
        type(checkpoint_warn_only) is not bool
        or checkpoint_warn_only != configured_warn_only
    ):
        raise CheckpointCompatibilityError(
            "checkpoint torch deterministic_warn_only disagrees with "
            "canonical trainer configuration"
        )

    missing_numeric_policy = sorted(
        {
            "default_dtype",
            "float32_matmul_precision",
            "cudnn_allow_tf32",
            "cudnn_enabled",
            "cudnn_deterministic",
            "cudnn_benchmark",
        }
        - torch_state.keys()
    )
    if missing_numeric_policy:
        raise CheckpointCompatibilityError(
            "canonical trainer checkpoint is missing torch numeric RNG policy "
            f"fields: {missing_numeric_policy}; load with restore_rng=False to "
            "opt out of exact replay"
        )
    cuda_environment = torch_state.get("cuda_environment")
    if not isinstance(cuda_environment, Mapping):
        raise CheckpointCompatibilityError(
            "canonical trainer checkpoint is missing CUDA process environment; "
            "load with restore_rng=False to opt out of exact replay"
        )
    _core._assert_torch_process_environment_matches(cuda_environment)

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


def _assert_checkpoint_process_environment_stable(
    state: Mapping[str, Any],
    *,
    expected_canonical: bool,
) -> None:
    """Fail closed if effectful restore work drifts the bound CUDA environment."""

    if not expected_canonical:
        return
    torch_state = state.get("torch")
    if not isinstance(torch_state, Mapping):
        raise CheckpointCompatibilityError(
            "canonical trainer checkpoint torch process state is unavailable"
        )
    cuda_environment = torch_state.get("cuda_environment")
    if not isinstance(cuda_environment, Mapping):
        raise CheckpointCompatibilityError(
            "canonical trainer checkpoint CUDA process environment is unavailable"
        )
    _core._assert_torch_process_environment_matches(cuda_environment)


def _assert_checkpoint_numeric_policy_stable(
    state: Mapping[str, Any],
    *,
    expected_canonical: bool,
) -> None:
    """Require live floating-point process policy to equal the checkpoint."""

    if not expected_canonical:
        return
    torch_state = state.get("torch")
    if not isinstance(torch_state, Mapping):
        raise CheckpointCompatibilityError(
            "canonical trainer checkpoint torch process state is unavailable"
        )
    required = (
        "default_dtype",
        "float32_matmul_precision",
        "cudnn_allow_tf32",
        "cudnn_enabled",
        "cudnn_deterministic",
        "cudnn_benchmark",
    )
    missing = [field for field in required if field not in torch_state]
    if missing:
        raise CheckpointCompatibilityError(
            "canonical trainer checkpoint numeric policy is incomplete: "
            f"{missing}"
        )
    torch = importlib.import_module("torch")
    live = {
        "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    drifted = [field for field in required if live[field] != torch_state[field]]
    if drifted:
        raise CheckpointCompatibilityError(
            "checkpoint torch numeric policy differs from the live process: "
            f"{drifted}"
        )


def _assert_ambient_process_state_stable(
    state: Mapping[str, Any],
    *,
    expected_canonical: bool,
) -> None:
    """Require restore work to preserve caller-owned numeric/CUDA process state."""

    if not expected_canonical:
        return
    try:
        _assert_checkpoint_process_environment_stable(
            state,
            expected_canonical=True,
        )
        _assert_checkpoint_numeric_policy_stable(
            state,
            expected_canonical=True,
        )
    except CheckpointCompatibilityError as exc:
        raise CheckpointCompatibilityError(
            "live ambient torch/CUDA process state changed during checkpoint restore"
        ) from exc


def _snapshot_torch_execution_mode() -> tuple[bool, bool]:
    """Snapshot caller-owned thread-local autograd/inference mode."""

    torch = importlib.import_module("torch")
    return (
        bool(torch.is_grad_enabled()),
        bool(torch.is_inference_mode_enabled()),
    )


def _assert_torch_execution_mode_stable(
    expected: tuple[bool, bool],
    *,
    expected_canonical: bool,
) -> None:
    """Reject restore callbacks that leak caller-owned torch execution mode."""

    if not expected_canonical:
        return
    live = _snapshot_torch_execution_mode()
    if live != expected:
        raise CheckpointCompatibilityError(
            "live torch autograd/inference mode changed during checkpoint restore"
        )


def _add_failure_note_preserving_primary(
    exc: BaseException,
    note: str,
) -> None:
    """Best-effort note attachment that cannot replace the primary failure."""

    try:
        BaseException.add_note(exc, note)
    except BaseException:  # noqa: BLE001 - diagnostics must never mask failure
        return


def _note_torch_execution_mode_drift(
    expected: tuple[bool, bool],
    exc: BaseException,
    *,
    operation: str,
    expected_canonical: bool,
) -> None:
    """Attach canonical execution-mode drift evidence without masking failure."""

    if not expected_canonical:
        return
    try:
        live = _snapshot_torch_execution_mode()
    except BaseException as mode_exc:  # noqa: BLE001 - preserve primary failure
        _add_failure_note_preserving_primary(
            exc,
            f"{operation} execution-mode drift check also failed: {mode_exc!r}",
        )
        return
    if live != expected:
        _add_failure_note_preserving_primary(
            exc,
            f"{operation} also leaked caller-owned torch execution mode: "
            f"expected grad_enabled={expected[0]}, inference_mode={expected[1]}; "
            f"observed grad_enabled={live[0]}, inference_mode={live[1]}",
        )


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
    except BaseException as mode_exc:  # noqa: BLE001
        _add_failure_note_preserving_primary(
            exc,
            f"PyTorch deterministic-mode rollback also failed: {mode_exc!r}",
        )


def _restore_ambient_rng_after_failed_apply(
    ambient: Mapping[str, Any], exc: BaseException,
) -> None:
    """Recover independent streams even if one ambient rollback setter fails."""

    try:
        _core.restore_rng_state(ambient)
        return
    except BaseException as rng_exc:  # noqa: BLE001
        _add_failure_note_preserving_primary(exc, f"Ambient RNG rollback also failed: {rng_exc!r}")

    # core.restore_rng_state stops at its first failed setter. Retry each
    # independent family separately so a Python/NumPy failure cannot also
    # strand an otherwise recoverable torch CPU/CUDA stream.
    if "python" in ambient:
        try:
            _core.random.setstate(ambient["python"])
        except BaseException as rollback_exc:  # noqa: BLE001
            _add_failure_note_preserving_primary(
                exc,
                f"Python RNG rollback also failed: {rollback_exc!r}",
            )
    if "numpy" in ambient:
        try:
            _core.np.random.set_state(ambient["numpy"])
        except BaseException as rollback_exc:  # noqa: BLE001
            _add_failure_note_preserving_primary(
                exc,
                f"NumPy RNG rollback also failed: {rollback_exc!r}",
            )

    torch_state = ambient.get("torch")
    if not isinstance(torch_state, Mapping):
        return
    try:
        torch = importlib.import_module("torch")
    except BaseException as rollback_exc:  # noqa: BLE001
        _add_failure_note_preserving_primary(
            exc,
            f"PyTorch RNG rollback unavailable: {rollback_exc!r}",
        )
        return
    if "cpu" in torch_state:
        try:
            torch.set_rng_state(torch_state["cpu"].cpu())
        except BaseException as rollback_exc:  # noqa: BLE001
            _add_failure_note_preserving_primary(
                exc,
                f"PyTorch CPU RNG rollback also failed: {rollback_exc!r}",
            )
    for index, cuda_state in enumerate(torch_state.get("cuda", ())):
        try:
            torch.cuda.set_rng_state(cuda_state.cpu(), device=index)
        except BaseException as rollback_exc:  # noqa: BLE001
            _add_failure_note_preserving_primary(
                exc,
                f"PyTorch CUDA RNG rollback on device {index} also failed: "
                f"{rollback_exc!r}"
            )
    if "default_dtype" in torch_state:
        try:
            _core._restore_torch_default_dtype(
                torch,
                torch_state["default_dtype"],
            )
        except BaseException as rollback_exc:  # noqa: BLE001
            _add_failure_note_preserving_primary(
                exc,
                f"PyTorch default-dtype rollback also failed: {rollback_exc!r}"
            )
    if "float32_matmul_precision" in torch_state:
        try:
            _core._restore_torch_matmul_precision(
                torch,
                torch_state["float32_matmul_precision"],
            )
        except BaseException as rollback_exc:  # noqa: BLE001
            _add_failure_note_preserving_primary(
                exc,
                "PyTorch float32-matmul-precision rollback also failed: "
                f"{rollback_exc!r}"
            )
    if "cudnn_allow_tf32" in torch_state:
        try:
            _core._restore_torch_cudnn_allow_tf32(
                torch,
                torch_state["cudnn_allow_tf32"],
            )
        except BaseException as rollback_exc:  # noqa: BLE001
            _add_failure_note_preserving_primary(
                exc,
                "PyTorch cuDNN TF32 rollback also failed: "
                f"{rollback_exc!r}"
            )
    cudnn_rollbacks = (
        ("cudnn_enabled", _core._restore_torch_cudnn_enabled, "enabled"),
        (
            "cudnn_deterministic",
            _core._restore_torch_cudnn_deterministic,
            "deterministic",
        ),
        ("cudnn_benchmark", _core._restore_torch_cudnn_benchmark, "benchmark"),
    )
    for field, restore, label in cudnn_rollbacks:
        if field not in torch_state:
            continue
        try:
            restore(torch, torch_state[field])
        except BaseException as rollback_exc:  # noqa: BLE001
            _add_failure_note_preserving_primary(
                exc,
                f"PyTorch cuDNN {label} rollback also failed: {rollback_exc!r}"
            )


def _restore_preapply_process_state(
    ambient: Mapping[str, Any],
    policy: tuple[bool, bool] | None,
    trainer: Any,
    *,
    execution_mode: tuple[bool, bool] | None = None,
    expected_canonical: bool | None = None,
) -> None:
    """Make effectful pre-application inspection observationally RNG-neutral."""

    if expected_canonical is None:
        expected_canonical = _is_canonical_d02(trainer) or _is_native_d02(trainer)
    try:
        _core.restore_rng_state(ambient)
        if policy is not None:
            torch = importlib.import_module("torch")
            torch.use_deterministic_algorithms(
                policy[0],
                warn_only=policy[1],
            )
        _assert_ambient_process_state_stable(
            ambient,
            expected_canonical=expected_canonical,
        )
        if execution_mode is not None:
            _assert_torch_execution_mode_stable(
                execution_mode,
                expected_canonical=expected_canonical,
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


def _restore_checkpoint_numeric_policy_for_apply(
    state: Mapping[str, Any],
) -> None:
    """Apply saved floating-point process policy before effectful loaders."""

    torch_state = state.get("torch")
    if not isinstance(torch_state, Mapping):
        return
    torch = importlib.import_module("torch")
    if "default_dtype" in torch_state:
        _core._restore_torch_default_dtype(
            torch,
            torch_state["default_dtype"],
        )
    if "float32_matmul_precision" in torch_state:
        _core._restore_torch_matmul_precision(
            torch,
            torch_state["float32_matmul_precision"],
        )
    if "cudnn_allow_tf32" in torch_state:
        _core._restore_torch_cudnn_allow_tf32(
            torch,
            torch_state["cudnn_allow_tf32"],
        )
    if "cudnn_enabled" in torch_state:
        _core._restore_torch_cudnn_enabled(
            torch,
            torch_state["cudnn_enabled"],
        )
    if "cudnn_deterministic" in torch_state:
        _core._restore_torch_cudnn_deterministic(
            torch,
            torch_state["cudnn_deterministic"],
        )
    if "cudnn_benchmark" in torch_state:
        _core._restore_torch_cudnn_benchmark(
            torch,
            torch_state["cudnn_benchmark"],
        )


def _restore_checkpoint_rng_preserving_warn_only(
    state: Mapping[str, Any],
    *,
    restore: Any,
    initial_policy: tuple[bool, bool] | None = None,
) -> None:
    """Preserve the validated live PyTorch policy across RNG replay.

    Legacy V1 RNG snapshots can omit warn-only mode. A canonical D02 Trainer
    has already configured its validated policy, so preserve that policy after
    replay; newer snapshots are preflight-checked against it before apply.
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


def _assert_native_checkpoint_config_identity(
    trainer: Any,
    state: Mapping[str, Any],
    *,
    identity: Mapping[str, Any] | CheckpointIdentity,
) -> None:
    """Bind native seed/precision identity to serialized TrainerConfig."""

    if not _is_native_d02(trainer):
        return
    config = state.get("config")
    if not isinstance(config, Mapping):
        raise CheckpointCompatibilityError(
            "native checkpoint trainer config identity is unavailable"
        )
    if isinstance(identity, Mapping):
        identity_seed = identity.get("seed")
        identity_precision = identity.get("precision")
    else:
        identity_seed = identity.seed
        identity_precision = identity.precision
    mismatches = {}
    for field, expected, actual, expected_type in (
        ("seed", identity_seed, config.get("seed"), int),
        ("precision", identity_precision, config.get("precision"), str),
    ):
        typed = (
            type(expected) is expected_type
            and type(actual) is expected_type
            and expected == actual
        )
        if not typed:
            mismatches[field] = {"identity": expected, "trainer": actual}
    if mismatches:
        raise CheckpointCompatibilityError(
            f"checkpoint native config identity mismatch: {mismatches}"
        )


def _assert_native_checkpoint_save_progress(
    trainer: Any,
    state: Mapping[str, Any],
    identity: CheckpointIdentity,
) -> None:
    """Bind native trainer counters to the manifest identity before file I/O."""

    if not _is_native_d02(trainer):
        return
    checks = {
        "step": (identity.step, state.get("optimizer_step")),
        "tokens_seen": (identity.tokens_seen, state.get("tokens_seen")),
    }
    mismatches = {
        field: {"identity": expected, "trainer": actual}
        for field, (expected, actual) in checks.items()
        if type(expected) is not int or type(actual) is not int or actual != expected
    }
    if mismatches:
        raise CheckpointCompatibilityError(
            f"checkpoint save progress identity mismatch: {mismatches}"
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

    save_bindings = _snapshot_trainer_restore_bindings(trainer)
    export_trainer_state = _bind_trainer_state_exporter(trainer)
    model_export_authority = _bind_native_model_export_validator(trainer)
    model_fingerprint = _bind_native_model_export_fingerprint(trainer)
    auxiliary_fingerprint = _bind_native_auxiliary_fingerprint(trainer)
    export_live_authorities = _bind_native_export_live_authorities(trainer)
    _assert_trainer_model_binding(model, trainer)
    _assert_native_d02_model_training_mode(model, trainer)
    # Reject a pre-existing process-policy mismatch before any effectful export.
    _assert_live_d02_determinism(trainer)

    if save_bindings[0]:
        export_ambient = capture_rng_state()
        export_policy = _snapshot_torch_policy(export_ambient)
        export_execution_mode = _snapshot_torch_execution_mode()
        export_started = False
        try:
            entry_model_fingerprint = (
                model_fingerprint()
                if model_fingerprint is not None
                else None
            )
            entry_auxiliary_fingerprint = (
                auxiliary_fingerprint()
                if auxiliary_fingerprint is not None
                else None
            )
            export_started = True
            state = _trainer_state_as_mapping(export_trainer_state())
            _run_native_d02_checkpoint_safety(trainer)
            sealed_model_fingerprint = (
                model_fingerprint()
                if model_fingerprint is not None
                else None
            )
            sealed_auxiliary_fingerprint = (
                auxiliary_fingerprint()
                if auxiliary_fingerprint is not None
                else None
            )
            if sealed_model_fingerprint != entry_model_fingerprint:
                raise CheckpointCompatibilityError(
                    "canonical trainer model changed during checkpoint export"
                )
            if sealed_auxiliary_fingerprint != entry_auxiliary_fingerprint:
                raise CheckpointCompatibilityError(
                    "canonical trainer auxiliary state changed during checkpoint export"
                )
        except BaseException as exc:
            _note_restore_binding_drift(trainer, save_bindings, exc)
            if export_started:
                _poison_canonical_restore_failure(
                    trainer,
                    expected_canonical=save_bindings[0],
                    reason="checkpoint_export_state_drift",
                    exc=exc,
                )
            raise
        finally:
            _restore_preapply_process_state(
                export_ambient,
                export_policy,
                trainer,
                execution_mode=export_execution_mode,
                expected_canonical=save_bindings[0],
            )
        try:
            _assert_trainer_restore_bindings(trainer, save_bindings)
            _assert_trainer_model_binding(model, trainer)
            _assert_native_d02_model_training_mode(model, trainer)
            _assert_native_d02_postload_snapshot(trainer, state)
            _assert_live_d02_determinism(trainer)
        except BaseException as exc:
            _poison_canonical_restore_failure(
                trainer,
                expected_canonical=save_bindings[0],
                reason="checkpoint_export_state_drift",
                exc=exc,
            )
            raise
    else:
        state = _trainer_state_as_mapping(export_trainer_state())
        sealed_model_fingerprint = None
        sealed_auxiliary_fingerprint = None

    _assert_trainer_restore_bindings(trainer, save_bindings)
    _assert_native_checkpoint_save_progress(trainer, state, identity)
    _assert_native_checkpoint_config_identity(
        trainer,
        state,
        identity=identity,
    )

    def assert_export_execution_mode() -> None:
        if not save_bindings[0]:
            return
        _assert_torch_execution_mode_stable(
            export_execution_mode,
            expected_canonical=True,
        )

    def prepublish_validator() -> None:
        if not save_bindings[0]:
            return
        try:
            assert_export_execution_mode()
            _assert_trainer_restore_bindings(trainer, save_bindings)
            _assert_trainer_model_binding(model, trainer)
            _assert_native_d02_model_training_mode(model, trainer)
            _assert_native_d02_postload_snapshot(trainer, state)
            _assert_live_d02_determinism(trainer)
            _assert_native_d02_exact_live_state(
                trainer,
                state,
                model_fingerprint=model_fingerprint,
                sealed_model_fingerprint=sealed_model_fingerprint,
                auxiliary_fingerprint=auxiliary_fingerprint,
                sealed_auxiliary_fingerprint=sealed_auxiliary_fingerprint,
                export_live_authorities=export_live_authorities,
                phase="checkpoint publication",
            )
            _assert_trainer_restore_bindings(trainer, save_bindings)
            _assert_trainer_model_binding(model, trainer)
            _assert_native_d02_model_training_mode(model, trainer)
            _assert_live_d02_determinism(trainer)
            assert_export_execution_mode()
        except BaseException as exc:
            _poison_canonical_restore_failure(
                trainer,
                expected_canonical=save_bindings[0],
                reason="checkpoint_export_state_drift",
                exc=exc,
            )
            raise

    def model_export_validator(exported: Mapping[str, Any]) -> None:
        try:
            assert_export_execution_mode()
            if model_export_authority is None:
                return
            _assert_trainer_restore_bindings(trainer, save_bindings)
            _assert_trainer_model_binding(model, trainer)
            _assert_native_d02_model_training_mode(model, trainer)
            _assert_native_d02_inert_live_state(
                trainer,
                state,
                model_fingerprint=model_fingerprint,
                sealed_model_fingerprint=sealed_model_fingerprint,
                auxiliary_fingerprint=auxiliary_fingerprint,
                sealed_auxiliary_fingerprint=sealed_auxiliary_fingerprint,
                phase="checkpoint model serialization",
            )
            model_export_authority(exported)
            assert_export_execution_mode()
        except BaseException as exc:
            _poison_canonical_restore_failure(
                trainer,
                expected_canonical=save_bindings[0],
                reason="checkpoint_export_state_drift",
                exc=exc,
            )
            if isinstance(exc, CheckpointCompatibilityError):
                raise
            raise CheckpointCompatibilityError(
                "checkpoint staged model export differs from live model state"
            ) from exc

    def post_rng_prepublish_validator() -> None:
        if not save_bindings[0]:
            return
        try:
            assert_export_execution_mode()
            _assert_trainer_restore_bindings(trainer, save_bindings)
            _assert_trainer_model_binding(model, trainer)
            _assert_native_d02_model_training_mode(model, trainer)
            _assert_native_d02_inert_live_state(
                trainer,
                state,
                model_fingerprint=model_fingerprint,
                sealed_model_fingerprint=sealed_model_fingerprint,
                auxiliary_fingerprint=auxiliary_fingerprint,
                sealed_auxiliary_fingerprint=sealed_auxiliary_fingerprint,
                phase="final checkpoint publication seal",
            )
            assert_export_execution_mode()
        except BaseException as exc:
            _poison_canonical_restore_failure(
                trainer,
                expected_canonical=save_bindings[0],
                reason="checkpoint_export_state_drift",
                exc=exc,
            )
            raise

    try:
        return save_checkpoint(
            directory,
            model=model,
            trainer_state=state,
            identity=identity,
            overwrite=overwrite,
            model_export_validator=model_export_validator,
            prepublish_validator=prepublish_validator,
            post_rng_prepublish_validator=post_rng_prepublish_validator,
        )
    except BaseException as exc:
        if save_bindings[0]:
            try:
                assert_export_execution_mode()
            except CheckpointCompatibilityError as mode_exc:
                _poison_canonical_restore_failure(
                    trainer,
                    expected_canonical=True,
                    reason="checkpoint_export_state_drift",
                    exc=mode_exc,
                )
                _add_failure_note_preserving_primary(
                    exc,
                    "checkpoint save also leaked caller-owned torch execution "
                    f"mode: {mode_exc}"
                )
        raise


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
    prebind_execution_mode = _snapshot_torch_execution_mode()
    try:
        load_trainer_state = _bind_trainer_state_loader(trainer)
        model_apply_authority = _bind_native_model_export_validator(trainer)
        model_fingerprint = _bind_native_model_export_fingerprint(trainer)
        auxiliary_fingerprint = _bind_native_auxiliary_fingerprint(trainer)
        restore_live_authorities = _bind_native_export_live_authorities(trainer)
    except BaseException as exc:
        _note_restore_binding_drift(trainer, restore_bindings, exc)
        raise
    finally:
        _restore_preapply_process_state(
            prebind_ambient,
            prebind_policy,
            trainer,
            execution_mode=prebind_execution_mode,
            expected_canonical=restore_bindings[0],
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
    preio_execution_mode = _snapshot_torch_execution_mode()
    try:
        _assert_trainer_model_binding(model, trainer)
        _preflight_trainer_target(trainer)
    except BaseException as exc:
        _note_restore_binding_drift(trainer, restore_bindings, exc)
        raise
    finally:
        _restore_preapply_process_state(
            preio_ambient,
            preio_policy,
            trainer,
            execution_mode=preio_execution_mode,
            expected_canonical=restore_bindings[0],
        )
    _assert_trainer_restore_bindings(trainer, restore_bindings)
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
    preapply_execution_mode = _snapshot_torch_execution_mode()
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
    except BaseException as exc:
        _note_restore_binding_drift(trainer, restore_bindings, exc)
        raise
    finally:
        _restore_preapply_process_state(
            preapply_ambient,
            preapply_policy,
            trainer,
            execution_mode=preapply_execution_mode,
            expected_canonical=restore_bindings[0],
        )
    _assert_trainer_restore_bindings(trainer, restore_bindings)
    if restore_rng:
        try:
            _assert_checkpoint_process_environment_stable(
                combined_state["rng"],
                expected_canonical=restore_bindings[0],
            )
        except BaseException as exc:
            _poison_canonical_restore_failure(
                trainer,
                expected_canonical=restore_bindings[0],
                reason="checkpoint_preapply_process_environment_drift",
                exc=exc,
            )
            raise

    policy_before_apply = _snapshot_torch_policy(combined_state["rng"])
    ambient_before_apply = capture_rng_state()
    execution_mode_before_apply = _snapshot_torch_execution_mode()
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
        if restore_rng:
            _restore_checkpoint_numeric_policy_for_apply(
                combined_state["rng"],
            )
            _assert_checkpoint_process_environment_stable(
                combined_state["rng"],
                expected_canonical=restore_bindings[0],
            )
            _assert_checkpoint_numeric_policy_stable(
                combined_state["rng"],
                expected_canonical=restore_bindings[0],
            )
        model_apply(materialized)
        if restore_rng:
            _assert_checkpoint_process_environment_stable(
                combined_state["rng"],
                expected_canonical=restore_bindings[0],
            )
            _assert_checkpoint_numeric_policy_stable(
                combined_state["rng"],
                expected_canonical=restore_bindings[0],
            )
        else:
            _assert_ambient_process_state_stable(
                ambient_before_apply,
                expected_canonical=restore_bindings[0],
            )
        _assert_torch_execution_mode_stable(
            execution_mode_before_apply,
            expected_canonical=restore_bindings[0],
        )
        if model_apply_authority is not None:
            try:
                model_apply_authority(materialized)
            except (ArithmeticError, RuntimeError, TypeError, ValueError) as exc:
                raise CheckpointCompatibilityError(
                    "checkpoint model load differs from verified model state"
                ) from exc
        # No later stage needs the materialized model-scale copy. Release it
        # before optimizer/scheduler/scaler restoration to reduce peak resume memory.
        del materialized
        sealed_model_fingerprint = (
            model_fingerprint()
            if model_fingerprint is not None
            else None
        )
        _assert_trainer_restore_bindings(trainer, restore_bindings)
        _assert_trainer_model_binding(model, trainer)
        _assert_native_d02_model_training_mode(model, trainer)
        load_trainer_state(trainer_state)
        if restore_rng:
            _assert_checkpoint_process_environment_stable(
                combined_state["rng"],
                expected_canonical=restore_bindings[0],
            )
            _assert_checkpoint_numeric_policy_stable(
                combined_state["rng"],
                expected_canonical=restore_bindings[0],
            )
        else:
            _assert_ambient_process_state_stable(
                ambient_before_apply,
                expected_canonical=restore_bindings[0],
            )
        _assert_torch_execution_mode_stable(
            execution_mode_before_apply,
            expected_canonical=restore_bindings[0],
        )
        sealed_auxiliary_fingerprint = (
            auxiliary_fingerprint()
            if auxiliary_fingerprint is not None
            else None
        )
        _assert_trainer_restore_bindings(trainer, restore_bindings)
        _assert_native_d02_model_training_mode(model, trainer)
        _postflight_trainer_state(trainer, trainer_state)
        _assert_trainer_restore_bindings(trainer, restore_bindings)
        _assert_native_d02_model_training_mode(model, trainer)
        _assert_native_d02_exact_live_state(
            trainer,
            trainer_state,
            model_fingerprint=model_fingerprint,
            sealed_model_fingerprint=sealed_model_fingerprint,
            auxiliary_fingerprint=auxiliary_fingerprint,
            sealed_auxiliary_fingerprint=sealed_auxiliary_fingerprint,
            export_live_authorities=restore_live_authorities,
            phase="checkpoint restore",
        )
        if restore_rng:
            _restore_checkpoint_rng_preserving_warn_only(
                combined_state["rng"],
                restore=restore_rng_state,
                initial_policy=policy_before_apply,
            )
            _assert_checkpoint_process_environment_stable(
                combined_state["rng"],
                expected_canonical=restore_bindings[0],
            )
            _assert_checkpoint_numeric_policy_stable(
                combined_state["rng"],
                expected_canonical=restore_bindings[0],
            )
        else:
            _assert_live_d02_determinism(trainer)

        # The final RNG replay / opt-out policy check is itself effectful.
        # Seal the already-restored native D02 state one last time before
        # reporting success so a late callout cannot consume exposure or
        # change restore ownership after postflight.
        _assert_trainer_restore_bindings(trainer, restore_bindings)
        _assert_trainer_model_binding(model, trainer)
        _assert_native_d02_model_training_mode(model, trainer)
        _assert_native_d02_inert_live_state(
            trainer,
            trainer_state,
            model_fingerprint=model_fingerprint,
            sealed_model_fingerprint=sealed_model_fingerprint,
            auxiliary_fingerprint=auxiliary_fingerprint,
            sealed_auxiliary_fingerprint=sealed_auxiliary_fingerprint,
            phase="final checkpoint restore seal",
        )
        if not restore_rng:
            _assert_ambient_process_state_stable(
                ambient_before_apply,
                expected_canonical=restore_bindings[0],
            )
        _assert_torch_execution_mode_stable(
            execution_mode_before_apply,
            expected_canonical=restore_bindings[0],
        )
    except BaseException as exc:
        _note_torch_execution_mode_drift(
            execution_mode_before_apply,
            exc,
            operation="checkpoint restore apply",
            expected_canonical=restore_bindings[0],
        )
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
