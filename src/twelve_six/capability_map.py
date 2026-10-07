from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_REGISTRY_BYTES = 1024 * 1024
_PACKAGE_ROOT = Path(__file__).resolve().parent


def _git_subprocess_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


def _require_repository_module_origin(
    module: ModuleType,
    *,
    _sealed_package_root: Path = _PACKAGE_ROOT,
) -> Path:
    """Bind an imported twelve_six module to its canonical repository source path."""

    if not _is_exact_type(module, ModuleType):
        raise ValueError("component contract owner must be an exact Python module")
    module_name = getattr(module, "__name__", None)
    if not _is_exact_type(module_name, str) or not (
        module_name == "twelve_six" or module_name.startswith("twelve_six.")
    ):
        raise ValueError("component contract owner module is not canonical twelve_six code")

    module_file = getattr(module, "__file__", None)
    module_spec = getattr(module, "__spec__", None)
    spec_origin = getattr(module_spec, "origin", None)
    if not _is_exact_type(module_file, str) or not _is_exact_type(spec_origin, str):
        raise ValueError("component contract repository-owned module origin is unavailable")

    source_path = Path(module_file).resolve()
    spec_path = Path(spec_origin).resolve()
    if source_path != spec_path or not source_path.is_file():
        raise ValueError("component contract repository-owned module origin mismatch")

    relative_parts = module_name.split(".")[1:]
    module_stem = _sealed_package_root.joinpath(*relative_parts)
    expected_paths = {
        module_stem.with_suffix(".py").resolve(),
        (module_stem / "__init__.py").resolve(),
    }
    if source_path not in expected_paths:
        raise ValueError("component contract repository-owned module origin is outside package")
    try:
        source_path.relative_to(_sealed_package_root)
    except ValueError as exc:
        raise ValueError(
            "component contract repository-owned module origin escapes package root"
        ) from exc
    return source_path


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object member")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number is not allowed")
    return parsed


def _strict_json_object(data: bytes) -> dict[str, Any]:
    if not _is_exact_type(data, bytes):
        raise ValueError("capability registry input must be bytes")
    if len(data) > _MAX_REGISTRY_BYTES:
        raise ValueError("capability registry exceeds maximum encoded size")
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("capability registry is not strict unambiguous UTF-8 JSON") from exc
    if not _is_exact_type(value, dict):
        raise ValueError("capability registry root must be an object")
    return value


def _require_exact_fields(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not _is_exact_type(value, dict) or set(value) != expected:
        raise ValueError(f"{label} schema is non-canonical")
    return value


def _require_id(name: str, value: object) -> str:
    if not _is_exact_type(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical identifier")
    return value


def _require_text(name: str, value: object) -> str:
    if not _is_exact_type(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if not _is_exact_type(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if not _is_exact_type(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _is_exact_type(value: object, expected: type[object]) -> bool:
    # Registry identity schemas reject behavioral subclasses that can reseal serialization.
    return type(value) is expected


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class CapabilityStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class TestLevel(str, Enum):
    COMPONENT = "component"
    INTEGRATION = "integration"
    END_TO_END = "end_to_end"


def _capability_status_wire_value(value: object) -> str:
    if not _is_exact_type(value, CapabilityStatus):
        raise ValueError("status must be a CapabilityStatus")
    raw_value = str.__str__(value)
    stored_value = object.__getattribute__(value, "_value_")
    if not _is_exact_type(stored_value, str) or stored_value != raw_value:
        raise ValueError("status wire value is non-canonical")
    return raw_value


def _test_level_wire_value(value: object) -> str:
    if not _is_exact_type(value, TestLevel):
        raise ValueError("level must be a TestLevel")
    raw_value = str.__str__(value)
    stored_value = object.__getattribute__(value, "_value_")
    if not _is_exact_type(stored_value, str) or stored_value != raw_value:
        raise ValueError("level wire value is non-canonical")
    return raw_value


_CANONICAL_CAPABILITY_STATUSES = tuple(CapabilityStatus)
_CANONICAL_CAPABILITY_STATUS_VALUES = tuple(
    _capability_status_wire_value(item) for item in _CANONICAL_CAPABILITY_STATUSES
)
_CANONICAL_TEST_LEVELS = tuple(TestLevel)
_CANONICAL_TEST_LEVEL_VALUES = tuple(
    _test_level_wire_value(item) for item in _CANONICAL_TEST_LEVELS
)

_SEALED_TEST_LEVEL_WIRE = _test_level_wire_value
_SEALED_REQUIRED_LEVEL_VALUES = (
    _CANONICAL_TEST_LEVEL_VALUES[0],
    _CANONICAL_TEST_LEVEL_VALUES[1],
)


def _require_capability_status(
    value: object,
    _sealed_type: type[CapabilityStatus] = CapabilityStatus,
    _sealed_statuses: tuple[CapabilityStatus, ...] = _CANONICAL_CAPABILITY_STATUSES,
    _sealed_values: tuple[str, ...] = _CANONICAL_CAPABILITY_STATUS_VALUES,
) -> CapabilityStatus:
    if type(value) is not _sealed_type:
        raise ValueError("status must be a CapabilityStatus")
    for index, canonical in enumerate(_sealed_statuses):
        if value is canonical:
            if _capability_status_wire_value(canonical) != _sealed_values[index]:
                raise ValueError("status wire value is non-canonical")
            return canonical
    raise ValueError("status must be a canonical CapabilityStatus")


def _require_test_level(
    value: object,
    _sealed_type: type[TestLevel] = TestLevel,
    _sealed_levels: tuple[TestLevel, ...] = _CANONICAL_TEST_LEVELS,
    _sealed_values: tuple[str, ...] = _CANONICAL_TEST_LEVEL_VALUES,
) -> TestLevel:
    if type(value) is not _sealed_type:
        raise ValueError("level must be a TestLevel")
    for index, canonical in enumerate(_sealed_levels):
        if value is canonical:
            if _test_level_wire_value(canonical) != _sealed_values[index]:
                raise ValueError("level wire value is non-canonical")
            return canonical
    raise ValueError("level must be a canonical TestLevel")


def _capability_status_from_wire_value(
    value: object,
    _sealed_statuses: tuple[CapabilityStatus, ...] = _CANONICAL_CAPABILITY_STATUSES,
    _sealed_values: tuple[str, ...] = _CANONICAL_CAPABILITY_STATUS_VALUES,
    _sealed_validator: Any = _require_capability_status,
) -> CapabilityStatus:
    if type(value) is not str:
        raise ValueError("status must be an exact string")
    try:
        index = _sealed_values.index(value)
    except ValueError as exc:
        raise ValueError("status wire value is unsupported") from exc
    return _sealed_validator(_sealed_statuses[index])


def _test_level_from_wire_value(
    value: object,
    _sealed_levels: tuple[TestLevel, ...] = _CANONICAL_TEST_LEVELS,
    _sealed_values: tuple[str, ...] = _CANONICAL_TEST_LEVEL_VALUES,
    _sealed_validator: Any = _require_test_level,
) -> TestLevel:
    if type(value) is not str:
        raise ValueError("level must be an exact string")
    try:
        index = _sealed_values.index(value)
    except ValueError as exc:
        raise ValueError("level wire value is unsupported") from exc
    return _sealed_validator(_sealed_levels[index])


def _build_component_contract_authority() -> Any:
    # Capture the import/origin machinery used by the acceptance authority.
    # Rebinding module globals after import must not manufacture a repository-owned
    # component contract.
    sealed_import_module = importlib.import_module
    sealed_getsourcefile = inspect.getsourcefile
    sealed_module_type = ModuleType
    sealed_path_type = Path
    sealed_package_root = _PACKAGE_ROOT.resolve()

    def require_module_origin(module: ModuleType, expected_name: str) -> Path:
        if type(module) is not sealed_module_type:
            raise ValueError("component contract owner must be an exact Python module")
        module_name = getattr(module, "__name__", None)
        if type(module_name) is not str or module_name != expected_name:
            raise ValueError("component contract import resolved unexpected module")
        if not (
            module_name == "twelve_six" or module_name.startswith("twelve_six.")
        ):
            raise ValueError("component contract owner module is not canonical twelve_six code")

        module_file = getattr(module, "__file__", None)
        module_spec = getattr(module, "__spec__", None)
        spec_origin = getattr(module_spec, "origin", None)
        if type(module_file) is not str or type(spec_origin) is not str:
            raise ValueError("component contract repository-owned module origin is unavailable")

        source_path = sealed_path_type(module_file).resolve()
        spec_path = sealed_path_type(spec_origin).resolve()
        if source_path != spec_path or not source_path.is_file():
            raise ValueError("component contract repository-owned module origin mismatch")

        relative_parts = module_name.split(".")[1:]
        module_stem = sealed_package_root.joinpath(*relative_parts)
        expected_paths = {
            module_stem.with_suffix(".py").resolve(),
            (module_stem / "__init__.py").resolve(),
        }
        if source_path not in expected_paths:
            raise ValueError("component contract repository-owned module origin is outside package")
        try:
            source_path.relative_to(sealed_package_root)
        except ValueError as exc:
            raise ValueError(
                "component contract repository-owned module origin escapes package root"
            ) from exc
        return source_path

    def resolve(component_contract: str) -> object:
        if type(component_contract) is not str or not component_contract.strip():
            raise ValueError("component_contract must be non-empty text")
        contract = component_contract
        if not contract.startswith("twelve_six."):
            raise ValueError(
                "AVAILABLE component contract must be repository-owned twelve_six Python"
            )

        parts = contract.split(".")
        for index in range(len(parts), 0, -1):
            module_name = ".".join(parts[:index])
            try:
                imported_module = sealed_import_module(module_name)
            except ModuleNotFoundError as exc:
                missing_name = exc.name
                if (
                    type(missing_name) is not str
                    or not (
                        module_name == missing_name
                        or module_name.startswith(f"{missing_name}.")
                    )
                ):
                    raise ValueError(
                        f"component contract import failed inside module: {module_name}"
                    ) from exc
                continue

            require_module_origin(imported_module, module_name)
            resolved: object = imported_module
            for attribute in parts[index:]:
                if not hasattr(resolved, attribute):
                    raise ValueError(
                        f"component contract attribute does not exist: {contract}"
                    )
                resolved = getattr(resolved, attribute)

            if type(resolved) is sealed_module_type:
                owner = resolved
                owner_name = getattr(owner, "__name__", None)
                if type(owner_name) is not str:
                    raise ValueError(
                        "component contract owner module name is unavailable"
                    )
            else:
                owner_name = getattr(resolved, "__module__", None)
                if type(owner_name) is not str or not (
                    owner_name == "twelve_six"
                    or owner_name.startswith("twelve_six.")
                ):
                    raise ValueError(
                        "AVAILABLE component contract must resolve to "
                        "repository-owned twelve_six code"
                    )
                try:
                    owner = sealed_import_module(owner_name)
                except (ImportError, ValueError) as exc:
                    raise ValueError(
                        "AVAILABLE component contract owner module cannot be resolved"
                    ) from exc

            owner_source_path = require_module_origin(owner, owner_name)
            if type(resolved) is not sealed_module_type:
                try:
                    resolved_source = sealed_getsourcefile(resolved)
                except (OSError, TypeError) as exc:
                    raise ValueError(
                        "component contract object source is unavailable"
                    ) from exc
                if type(resolved_source) is not str:
                    raise ValueError(
                        "component contract object source is unavailable"
                    )
                if sealed_path_type(resolved_source).resolve() != owner_source_path:
                    raise ValueError(
                        "component contract object source does not match owner module"
                    )
            return resolved
        raise ValueError(f"component contract module does not exist: {contract}")

    return resolve


_resolve_component_contract_authority = _build_component_contract_authority()

# Seal the repository-owned contract resolver against rebinding of public/module
# helpers. Stored registry authority captures this closure at definition time.
_SEALED_RESOLVE_COMPONENT_CONTRACT = _resolve_component_contract_authority


@dataclass(frozen=True, slots=True)
class EnvironmentSupport:
    environment_id: str
    supported: bool

    def __post_init__(self) -> None:
        if (
            type(self.environment_id) is not str
            or not 1 <= len(self.environment_id) <= 96
            or self.environment_id[0] not in "abcdefghijklmnopqrstuvwxyz"
            or any(
                char not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
                for char in self.environment_id[1:]
            )
        ):
            raise ValueError("environment_id must be a canonical identifier")
        if type(self.supported) is not bool:
            raise ValueError("supported must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return _environment_support_payload_from_stored_state(self)


@dataclass(frozen=True, slots=True)
class TestVector:
    vector_id: str
    level: TestLevel
    command: str

    def __post_init__(self) -> None:
        if (
            type(self.vector_id) is not str
            or not 1 <= len(self.vector_id) <= 96
            or self.vector_id[0] not in "abcdefghijklmnopqrstuvwxyz"
            or any(
                char not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
                for char in self.vector_id[1:]
            )
        ):
            raise ValueError("vector_id must be a canonical identifier")
        _require_test_level(self.level)
        _require_text("command", self.command)
        tokens = self.command.split()
        if (
            tokens[:2] != ["pytest", "-q"]
            or len(tokens) < 3
            or self.command != " ".join(tokens)
        ):
            raise ValueError("test vector command must be canonical pytest -q test paths")
        for token in tokens[2:]:
            parts = token.split("/")
            if (
                "\\" in token
                or token.startswith("/")
                or len(parts) < 2
                or parts[0] != "tests"
                or any(part in {"", ".", ".."} for part in parts)
                or not parts[-1].endswith(".py")
                or parts[-1] == ".py"
            ):
                raise ValueError(
                    "test vector command may reference only canonical tests/*.py paths"
                )

    def to_dict(self) -> dict[str, Any]:
        return _test_vector_payload_from_stored_state(self)


@dataclass(frozen=True, slots=True)
class EvidenceTarget:
    evidence_id: str
    target: str

    def __post_init__(self) -> None:
        if (
            type(self.evidence_id) is not str
            or not 1 <= len(self.evidence_id) <= 96
            or self.evidence_id[0] not in "abcdefghijklmnopqrstuvwxyz"
            or any(
                char not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
                for char in self.evidence_id[1:]
            )
        ):
            raise ValueError("evidence_id must be a canonical identifier")
        if type(self.target) is not str or not self.target.strip():
            raise ValueError("target must be non-empty text")

    def to_dict(self) -> dict[str, str]:
        return _evidence_target_payload_from_stored_state(self)


@dataclass(frozen=True, slots=True)
class Capability:
    capability_id: str
    schema_version: int
    status: CapabilityStatus
    component_contract: str
    dependencies: tuple[str, ...]
    journey_ids: tuple[str, ...]
    environments: tuple[EnvironmentSupport, ...]
    test_vectors: tuple[TestVector, ...]
    evidence_targets: tuple[EvidenceTarget, ...]
    integrated_result: str | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if (
            type(self.capability_id) is not str
            or not 1 <= len(self.capability_id) <= 96
            or self.capability_id[0] not in "abcdefghijklmnopqrstuvwxyz"
            or any(
                char not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
                for char in self.capability_id[1:]
            )
        ):
            raise ValueError("capability_id must be a canonical identifier")
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("schema_version must be a positive integer")
        _require_capability_status(self.status)
        if (
            type(self.component_contract) is not str
            or not self.component_contract.strip()
        ):
            raise ValueError("component_contract must be non-empty text")

        for name, values in (
            ("dependencies", self.dependencies),
            ("journey_ids", self.journey_ids),
        ):
            if type(values) is not tuple:
                raise ValueError(f"{name} must be an immutable tuple")
            for value in values:
                if (
                    type(value) is not str
                    or not 1 <= len(value) <= 96
                    or value[0] not in "abcdefghijklmnopqrstuvwxyz"
                    or any(
                        char not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
                        for char in value[1:]
                    )
                ):
                    raise ValueError(f"{name} must be a canonical identifier")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")

        if not self.journey_ids:
            raise ValueError("capability must bind at least one user/operator journey")

        if type(self.environments) is not tuple or any(
            type(item) is not EnvironmentSupport for item in self.environments
        ):
            raise ValueError("environments must contain EnvironmentSupport values")
        if type(self.test_vectors) is not tuple or any(
            type(item) is not TestVector for item in self.test_vectors
        ):
            raise ValueError("test_vectors must contain TestVector values")
        if type(self.evidence_targets) is not tuple or any(
            type(item) is not EvidenceTarget for item in self.evidence_targets
        ):
            raise ValueError("evidence_targets must contain EvidenceTarget values")
        for item in self.environments:
            EnvironmentSupport.__post_init__(item)
        for item in self.test_vectors:
            TestVector.__post_init__(item)
        for item in self.evidence_targets:
            EvidenceTarget.__post_init__(item)

        environment_ids = [item.environment_id for item in self.environments]
        vector_ids = [item.vector_id for item in self.test_vectors]
        evidence_ids = [item.evidence_id for item in self.evidence_targets]
        if len(environment_ids) != len(set(environment_ids)):
            raise ValueError("environment ids must be unique")
        if len(vector_ids) != len(set(vector_ids)):
            raise ValueError("test vector ids must be unique")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence target ids must be unique")

        if self.status is CapabilityStatus.AVAILABLE:
            if self.unavailable_reason is not None:
                raise ValueError("AVAILABLE capability cannot have unavailable_reason")
            if (
                type(self.integrated_result) is not str
                or not self.integrated_result.strip()
            ):
                raise ValueError("integrated_result must be non-empty text")
            if not any(item.supported for item in self.environments):
                raise ValueError("AVAILABLE capability needs a supported environment")
            # The required acceptance levels are part of Section-2 authority.
            # Do not read module-global "sealed" aliases here: they can be rebound
            # after import. Nested TestVector state is revalidated separately on
            # stored-state paths, and the canonical wire values are fixed literals.
            levels = {str.__str__(item.level) for item in self.test_vectors}
            if any(
                required_level not in levels
                for required_level in ("component", "integration")
            ):
                raise ValueError(
                    "AVAILABLE capability needs component and integration test vectors"
                )
            if not self.evidence_targets:
                raise ValueError("AVAILABLE capability needs an evidence target")
        else:
            if (
                type(self.unavailable_reason) is not str
                or not self.unavailable_reason.strip()
            ):
                raise ValueError("unavailable_reason must be non-empty text")
            if self.integrated_result is not None:
                raise ValueError("UNAVAILABLE capability cannot claim an integrated_result")

    def to_dict(self) -> dict[str, Any]:
        return _capability_payload_from_stored_state(self)


@dataclass(frozen=True, slots=True)
class Journey:
    journey_id: str
    title: str
    capability_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.journey_id) is not str
            or not 1 <= len(self.journey_id) <= 96
            or self.journey_id[0] not in "abcdefghijklmnopqrstuvwxyz"
            or any(
                char not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
                for char in self.journey_id[1:]
            )
        ):
            raise ValueError("journey_id must be a canonical identifier")
        if type(self.title) is not str or not self.title.strip():
            raise ValueError("title must be non-empty text")
        if type(self.capability_ids) is not tuple or not self.capability_ids:
            raise ValueError("journey capability_ids must be a non-empty tuple")
        for capability_id in self.capability_ids:
            if (
                type(capability_id) is not str
                or not 1 <= len(capability_id) <= 96
                or capability_id[0] not in "abcdefghijklmnopqrstuvwxyz"
                or any(
                    char not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
                    for char in capability_id[1:]
                )
            ):
                raise ValueError("journey capability_id must be a canonical identifier")
        if len(self.capability_ids) != len(set(self.capability_ids)):
            raise ValueError("journey capability_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return _journey_payload_from_stored_state(self)


@dataclass(frozen=True, slots=True)
class SourceSurface:
    path: str
    capability_id: str
    origin: str

    def __post_init__(self) -> None:
        _require_text("source surface path", self.path)
        if (
            type(self.capability_id) is not str
            or not 1 <= len(self.capability_id) <= 96
            or self.capability_id[0] not in "abcdefghijklmnopqrstuvwxyz"
            or any(
                char not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
                for char in self.capability_id[1:]
            )
        ):
            raise ValueError("source surface capability_id must be a canonical identifier")
        if type(self.origin) is not str or self.origin not in {
            "accepted_main",
            "stacked_candidate",
            "modified_candidate",
        }:
            raise ValueError("source surface origin is unsupported")
        parts = self.path.split("/")
        if (
            "\\" in self.path
            or self.path.startswith("/")
            or len(parts) < 3
            or parts[:2] != ["src", "twelve_six"]
            or any(part in {"", ".", ".."} for part in parts)
            or not parts[-1].endswith(".py")
            or parts[-1] == ".py"
        ):
            raise ValueError(
                "source surface path must be a canonical Python path under src/twelve_six"
            )

    def to_dict(self) -> dict[str, str]:
        return _source_surface_payload_from_stored_state(self)


@dataclass(frozen=True, slots=True)
class SourceSurfaceInventory:
    schema_version: int
    observed_main_sha: str
    observed_main_tree_sha: str
    source_root: str
    source_surface_count: int
    accepted_main_surface_count: int
    candidate_overlay_surface_count: int
    surfaces: tuple[SourceSurface, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("schema_version must be a positive integer")
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("unsupported SourceSurfaceInventory schema_version")
        for field_name, value in (
            ("observed_main_sha", self.observed_main_sha),
            ("observed_main_tree_sha", self.observed_main_tree_sha),
        ):
            if type(value) is not str or _SHA40_RE.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a lowercase 40-hex Git SHA")
        _require_text("source_root", self.source_root)
        if self.source_root != "src/twelve_six":
            raise ValueError("source_root must be canonical src/twelve_six")
        if type(self.source_surface_count) is not int or self.source_surface_count <= 0:
            raise ValueError("source_surface_count must be a positive integer")
        if (
            type(self.accepted_main_surface_count) is not int
            or self.accepted_main_surface_count <= 0
        ):
            raise ValueError("accepted_main_surface_count must be a positive integer")
        if (
            type(self.candidate_overlay_surface_count) is not int
            or self.candidate_overlay_surface_count < 0
        ):
            raise ValueError(
                "candidate_overlay_surface_count must be a non-negative integer"
            )
        if type(self.surfaces) is not tuple or not self.surfaces:
            raise ValueError("surfaces must be a non-empty tuple")
        if any(type(item) is not SourceSurface for item in self.surfaces):
            raise ValueError("surfaces must contain only SourceSurface values")
        for item in self.surfaces:
            SourceSurface.__post_init__(item)
        if self.source_surface_count != len(self.surfaces):
            raise ValueError("source_surface_count does not match surfaces")
        accepted_count = sum(item.origin == "accepted_main" for item in self.surfaces)
        candidate_count = sum(item.origin != "accepted_main" for item in self.surfaces)
        if accepted_count != self.accepted_main_surface_count:
            raise ValueError("accepted_main_surface_count does not match surfaces")
        if candidate_count != self.candidate_overlay_surface_count:
            raise ValueError("candidate_overlay_surface_count does not match surfaces")
        if accepted_count + candidate_count != self.source_surface_count:
            raise ValueError("source surface origin counts do not cover the inventory")
        paths = [item.path for item in self.surfaces]
        if paths != sorted(paths):
            raise ValueError("source surfaces must be in canonical path order")
        if len(paths) != len(set(paths)):
            raise ValueError("source surface paths must be unique")

    def to_dict(self) -> dict[str, Any]:
        return _source_surface_inventory_payload_from_stored_state(self)

    def identity_sha256(self) -> str:
        return _source_surface_inventory_identity_from_stored_state(self)


def _build_capability_registry_post_init() -> Any:
    # The canonical constructor validator closes over authority at definition time.
    # The public dataclass hook therefore exposes no caller-supplied validator
    # parameter and later module-global resolver rebinding cannot replace it.
    sealed_resolve_component_contract = _SEALED_RESOLVE_COMPONENT_CONTRACT
    sealed_capability_type = Capability
    sealed_journey_type = Journey
    sealed_environment_type = EnvironmentSupport
    sealed_test_vector_type = TestVector
    sealed_evidence_type = EvidenceTarget
    sealed_capability_validate = Capability.__post_init__
    sealed_journey_validate = Journey.__post_init__
    sealed_environment_validate = EnvironmentSupport.__post_init__
    sealed_test_vector_validate = TestVector.__post_init__
    sealed_evidence_validate = EvidenceTarget.__post_init__
    sealed_require_capability_status = _require_capability_status
    sealed_require_test_level = _require_test_level
    sealed_available_status = CapabilityStatus.AVAILABLE
    sealed_unavailable_status = CapabilityStatus.UNAVAILABLE
    sealed_sha40_fullmatch = _SHA40_RE.fullmatch

    def validate(self: Any) -> None:
        if type(self.schema_version) is not int or self.schema_version <= 0:
            raise ValueError("schema_version must be a positive integer")
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise ValueError("unsupported CapabilityRegistry schema_version")
        if type(self.observed_main_sha) is not str or sealed_sha40_fullmatch(
            self.observed_main_sha
        ) is None:
            raise ValueError("observed_main_sha must be a lowercase 40-hex Git SHA")
        if (
            type(self.observed_main_ci_run_id) is not int
            or self.observed_main_ci_run_id <= 0
        ):
            raise ValueError("observed_main_ci_run_id must be a positive integer")
        if (
            type(self.observed_main_ci_conclusion) is not str
            or not self.observed_main_ci_conclusion.strip()
        ):
            raise ValueError("observed_main_ci_conclusion must be non-empty text")
        if self.observed_main_ci_conclusion != "success":
            raise ValueError("observed main CI must be terminal success")

        if type(self.capabilities) is not tuple or not self.capabilities:
            raise ValueError("capabilities must be a non-empty tuple")
        if any(type(item) is not sealed_capability_type for item in self.capabilities):
            raise ValueError("capabilities must contain only Capability values")
        if type(self.journeys) is not tuple or not self.journeys:
            raise ValueError("journeys must be a non-empty tuple")
        if any(type(item) is not sealed_journey_type for item in self.journeys):
            raise ValueError("journeys must contain only Journey values")
        for item in self.capabilities:
            sealed_require_capability_status(item.status)
            for environment in item.environments:
                if type(environment) is not sealed_environment_type:
                    raise ValueError(
                        "environments must contain EnvironmentSupport values"
                    )
                sealed_environment_validate(environment)
            for vector in item.test_vectors:
                if type(vector) is not sealed_test_vector_type:
                    raise ValueError("test_vectors must contain TestVector values")
                sealed_require_test_level(vector.level)
                sealed_test_vector_validate(vector)
            for evidence in item.evidence_targets:
                if type(evidence) is not sealed_evidence_type:
                    raise ValueError(
                        "evidence_targets must contain EvidenceTarget values"
                    )
                sealed_evidence_validate(evidence)
            sealed_capability_validate(item)
        for item in self.journeys:
            sealed_journey_validate(item)

        by_capability = {item.capability_id: item for item in self.capabilities}
        by_journey = {item.journey_id: item for item in self.journeys}
        if len(by_capability) != len(self.capabilities):
            raise ValueError("capability ids must be unique")
        if len(by_journey) != len(self.journeys):
            raise ValueError("journey ids must be unique")

        expected_main_ci_target = f"github-actions:{self.observed_main_ci_run_id}"
        for capability in self.capabilities:
            if capability.status is sealed_available_status:
                sealed_resolve_component_contract(capability.component_contract)
                main_ci_targets = [
                    target.target
                    for target in capability.evidence_targets
                    if target.evidence_id == "main-ci"
                ]
                if main_ci_targets != [expected_main_ci_target]:
                    raise ValueError(
                        f"{capability.capability_id} must bind exact observed main CI"
                    )
            for dependency_id in capability.dependencies:
                if dependency_id not in by_capability:
                    raise ValueError(
                        f"{capability.capability_id} has unknown dependency {dependency_id}"
                    )
                if (
                    capability.status is sealed_available_status
                    and by_capability[dependency_id].status is sealed_unavailable_status
                ):
                    raise ValueError(
                        f"AVAILABLE capability depends on UNAVAILABLE {dependency_id}"
                    )
            for journey_id in capability.journey_ids:
                if journey_id not in by_journey:
                    raise ValueError(
                        f"{capability.capability_id} has unknown journey {journey_id}"
                    )
                if capability.capability_id not in by_journey[journey_id].capability_ids:
                    raise ValueError(
                        f"{capability.capability_id} is not listed by journey {journey_id}"
                    )

        for journey in self.journeys:
            for capability_id in journey.capability_ids:
                if capability_id not in by_capability:
                    raise ValueError(
                        f"{journey.journey_id} references unknown capability {capability_id}"
                    )
                if journey.journey_id not in by_capability[capability_id].journey_ids:
                    raise ValueError(
                        f"{journey.journey_id} is not back-bound by {capability_id}"
                    )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(capability_id: str) -> None:
            if capability_id in visited:
                return
            if capability_id in visiting:
                raise ValueError(f"capability dependency cycle at {capability_id}")
            visiting.add(capability_id)
            for dependency_id in by_capability[capability_id].dependencies:
                visit(dependency_id)
            visiting.remove(capability_id)
            visited.add(capability_id)

        for capability_id in by_capability:
            visit(capability_id)

    return validate


_CAPABILITY_REGISTRY_POST_INIT = _build_capability_registry_post_init()


@dataclass(frozen=True, slots=True)
class CapabilityRegistry:
    schema_version: int
    observed_main_sha: str
    observed_main_ci_run_id: int
    observed_main_ci_conclusion: str
    capabilities: tuple[Capability, ...]
    journeys: tuple[Journey, ...]

    __post_init__ = _CAPABILITY_REGISTRY_POST_INIT

    @staticmethod
    def _reject_dependency_cycles(by_capability: dict[str, Capability]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(capability_id: str) -> None:
            if capability_id in visited:
                return
            if capability_id in visiting:
                raise ValueError(f"capability dependency cycle at {capability_id}")
            visiting.add(capability_id)
            for dependency_id in by_capability[capability_id].dependencies:
                visit(dependency_id)
            visiting.remove(capability_id)
            visited.add(capability_id)

        for capability_id in by_capability:
            visit(capability_id)

    def capability(self, capability_id: str) -> Capability:
        _validate_capability_registry_stored(self)
        _require_id("capability_id", capability_id)
        for capability in self.capabilities:
            if capability.capability_id == capability_id:
                return capability
        raise KeyError(capability_id)

    def journey_available(self, journey_id: str) -> bool:
        _validate_capability_registry_stored(self)
        _require_id("journey_id", journey_id)
        journey = next(
            (item for item in self.journeys if item.journey_id == journey_id),
            None,
        )
        if journey is None:
            raise KeyError(journey_id)
        by_capability = {item.capability_id: item for item in self.capabilities}
        return all(
            by_capability[capability_id].status is CapabilityStatus.AVAILABLE
            for capability_id in journey.capability_ids
        )

    def acceptance_path(self, capability_id: str) -> dict[str, Any]:
        _validate_capability_registry_stored(self)
        _require_id("capability_id", capability_id)
        capability = next(
            (
                item
                for item in self.capabilities
                if item.capability_id == capability_id
            ),
            None,
        )
        if capability is None:
            raise KeyError(capability_id)
        if capability.status is CapabilityStatus.UNAVAILABLE:
            return {
                "capability_id": capability.capability_id,
                "status": _capability_status_wire_value(capability.status),
                "component_contract": capability.component_contract,
                "unavailable_reason": capability.unavailable_reason,
                "integrated_result": None,
            }
        return {
            "capability_id": capability.capability_id,
            "status": _capability_status_wire_value(capability.status),
            "component_contract": capability.component_contract,
            "test_vectors": [
                _test_vector_payload_from_stored_state(item)
                for item in capability.test_vectors
            ],
            "evidence_targets": [
                _evidence_target_payload_from_stored_state(item)
                for item in capability.evidence_targets
            ],
            "integrated_result": capability.integrated_result,
        }

    def to_dict(self) -> dict[str, Any]:
        return _capability_registry_payload_from_stored_state(self)

    def identity_sha256(self) -> str:
        return _capability_registry_identity_from_stored_state(self)


def _environment_support_payload_from_stored_state(
    value: EnvironmentSupport,
    _sealed_validate=EnvironmentSupport.__post_init__,
) -> dict[str, Any]:
    _sealed_validate(value)
    return {"environment_id": value.environment_id, "supported": value.supported}


def _test_vector_payload_from_stored_state(
    value: TestVector,
    _sealed_validate=TestVector.__post_init__,
    _sealed_level_wire=_test_level_wire_value,
) -> dict[str, Any]:
    _sealed_validate(value)
    return {
        "vector_id": value.vector_id,
        "level": _sealed_level_wire(value.level),
        "command": value.command,
    }


def _evidence_target_payload_from_stored_state(
    value: EvidenceTarget,
    _sealed_validate=EvidenceTarget.__post_init__,
) -> dict[str, str]:
    _sealed_validate(value)
    return {"evidence_id": value.evidence_id, "target": value.target}


def _capability_payload_from_stored_state(
    value: Capability,
    _sealed_validate=Capability.__post_init__,
    _sealed_status_wire=_capability_status_wire_value,
    _sealed_environment_payload=_environment_support_payload_from_stored_state,
    _sealed_vector_payload=_test_vector_payload_from_stored_state,
    _sealed_evidence_payload=_evidence_target_payload_from_stored_state,
) -> dict[str, Any]:
    _sealed_validate(value)
    return {
        "capability_id": value.capability_id,
        "schema_version": value.schema_version,
        "status": _sealed_status_wire(value.status),
        "component_contract": value.component_contract,
        "dependencies": list(value.dependencies),
        "journey_ids": list(value.journey_ids),
        "environments": [
            _sealed_environment_payload(item) for item in value.environments
        ],
        "test_vectors": [
            _sealed_vector_payload(item) for item in value.test_vectors
        ],
        "evidence_targets": [
            _sealed_evidence_payload(item) for item in value.evidence_targets
        ],
        "integrated_result": value.integrated_result,
        "unavailable_reason": value.unavailable_reason,
    }


def _journey_payload_from_stored_state(
    value: Journey,
    _sealed_validate=Journey.__post_init__,
) -> dict[str, Any]:
    _sealed_validate(value)
    return {
        "journey_id": value.journey_id,
        "title": value.title,
        "capability_ids": list(value.capability_ids),
    }


def _source_surface_payload_from_stored_state(
    value: SourceSurface,
    _sealed_validate=SourceSurface.__post_init__,
) -> dict[str, str]:
    _sealed_validate(value)
    return {
        "path": value.path,
        "capability_id": value.capability_id,
        "origin": value.origin,
    }


def _source_surface_inventory_payload_from_stored_state(
    value: SourceSurfaceInventory,
    _sealed_validate=SourceSurfaceInventory.__post_init__,
    _sealed_surface_payload=_source_surface_payload_from_stored_state,
) -> dict[str, Any]:
    # Validate nested authority-bearing records through the sealed stored-state
    # path before the aggregate validator.  The aggregate dataclass validator
    # intentionally performs count checks too, but its nested method lookup can
    # be monkeypatched after construction; ordering the sealed payload checks
    # first prevents such rebinding from masking the actual invalid surface.
    surfaces = [_sealed_surface_payload(item) for item in value.surfaces]
    _sealed_validate(value)
    return {
        "schema_version": value.schema_version,
        "observed_main_sha": value.observed_main_sha,
        "observed_main_tree_sha": value.observed_main_tree_sha,
        "source_root": value.source_root,
        "source_surface_count": value.source_surface_count,
        "accepted_main_surface_count": value.accepted_main_surface_count,
        "candidate_overlay_surface_count": value.candidate_overlay_surface_count,
        "surfaces": surfaces,
    }


def _source_surface_inventory_identity_from_stored_state(
    value: SourceSurfaceInventory,
    _sealed_hash=_canonical_sha256,
    _sealed_payload=_source_surface_inventory_payload_from_stored_state,
) -> str:
    return _sealed_hash(_sealed_payload(value))


def _validate_capability_registry_stored(
    value: CapabilityRegistry,
    _sealed_registry_validate=CapabilityRegistry.__post_init__,
    _sealed_cycle_check=CapabilityRegistry._reject_dependency_cycles,
    _sealed_capability_payload=_capability_payload_from_stored_state,
    _sealed_journey_payload=_journey_payload_from_stored_state,
    _sealed_resolve_component_contract=_SEALED_RESOLVE_COMPONENT_CONTRACT,
) -> None:
    _sealed_registry_validate(value)
    for capability in value.capabilities:
        if capability.status is CapabilityStatus.AVAILABLE:
            _sealed_resolve_component_contract(capability.component_contract)
    for capability in value.capabilities:
        _sealed_capability_payload(capability)
    for journey in value.journeys:
        _sealed_journey_payload(journey)
    _sealed_cycle_check(
        {capability.capability_id: capability for capability in value.capabilities}
    )


def _capability_registry_payload_from_stored_state(
    value: CapabilityRegistry,
    _sealed_validate=_validate_capability_registry_stored,
    _sealed_capability_payload=_capability_payload_from_stored_state,
    _sealed_journey_payload=_journey_payload_from_stored_state,
) -> dict[str, Any]:
    _sealed_validate(value)
    return {
        "schema_version": value.schema_version,
        "observed_main_sha": value.observed_main_sha,
        "observed_main_ci": {
            "run_id": value.observed_main_ci_run_id,
            "conclusion": value.observed_main_ci_conclusion,
        },
        "capabilities": [
            _sealed_capability_payload(item) for item in value.capabilities
        ],
        "journeys": [_sealed_journey_payload(item) for item in value.journeys],
    }


def _capability_registry_identity_from_stored_state(
    value: CapabilityRegistry,
    _sealed_hash=_canonical_sha256,
    _sealed_payload=_capability_registry_payload_from_stored_state,
) -> str:
    return _sealed_hash(_sealed_payload(value))


def resolve_component_contract(component_contract: str) -> object:
    """Resolve one repository-owned Python module or module attribute fail-closed."""

    return _SEALED_RESOLVE_COMPONENT_CONTRACT(component_contract)


def validate_available_component_contracts(registry: CapabilityRegistry) -> None:
    """Prove every AVAILABLE capability begins at a live repository contract."""

    if not _is_exact_type(registry, CapabilityRegistry):
        raise ValueError("registry must be a CapabilityRegistry")
    _validate_capability_registry_stored(registry)
    for capability in registry.capabilities:
        if capability.status is CapabilityStatus.AVAILABLE:
            _SEALED_RESOLVE_COMPONENT_CONTRACT(capability.component_contract)


def load_capability_registry(path: str | Path) -> CapabilityRegistry:
    payload = _strict_json_object(Path(path).read_bytes())
    if set(payload) != {
        "schema_version",
        "observed_main_sha",
        "observed_main_ci",
        "capabilities",
        "journeys",
    }:
        raise ValueError("capability registry top-level schema is non-canonical")

    ci = _require_exact_fields(
        payload["observed_main_ci"],
        {"run_id", "conclusion"},
        "observed_main_ci",
    )
    raw_capabilities = payload["capabilities"]
    raw_journeys = payload["journeys"]
    if not _is_exact_type(raw_capabilities, list):
        raise ValueError("capabilities must be a JSON array")
    if not _is_exact_type(raw_journeys, list):
        raise ValueError("journeys must be a JSON array")

    capabilities = []
    capability_fields = {
        "capability_id",
        "schema_version",
        "status",
        "component_contract",
        "dependencies",
        "journey_ids",
        "environments",
        "test_vectors",
        "evidence_targets",
        "integrated_result",
        "unavailable_reason",
    }
    for raw_item in raw_capabilities:
        item = _require_exact_fields(raw_item, capability_fields, "capability")
        for list_field in (
            "dependencies",
            "journey_ids",
            "environments",
            "test_vectors",
            "evidence_targets",
        ):
            if not _is_exact_type(item[list_field], list):
                raise ValueError(f"capability.{list_field} must be a JSON array")
        environments = tuple(
            EnvironmentSupport(
                **_require_exact_fields(
                    environment,
                    {"environment_id", "supported"},
                    "environment",
                )
            )
            for environment in item["environments"]
        )
        test_vectors = tuple(
            TestVector(
                vector_id=vector["vector_id"],
                level=_test_level_from_wire_value(vector["level"]),
                command=vector["command"],
            )
            for raw_vector in item["test_vectors"]
            for vector in [
                _require_exact_fields(
                    raw_vector,
                    {"vector_id", "level", "command"},
                    "test_vector",
                )
            ]
        )
        evidence_targets = tuple(
            EvidenceTarget(
                **_require_exact_fields(
                    target,
                    {"evidence_id", "target"},
                    "evidence_target",
                )
            )
            for target in item["evidence_targets"]
        )
        capabilities.append(
            Capability(
                capability_id=item["capability_id"],
                schema_version=item["schema_version"],
                status=_capability_status_from_wire_value(item["status"]),
                component_contract=item["component_contract"],
                dependencies=tuple(item["dependencies"]),
                journey_ids=tuple(item["journey_ids"]),
                environments=environments,
                test_vectors=test_vectors,
                evidence_targets=evidence_targets,
                integrated_result=item["integrated_result"],
                unavailable_reason=item["unavailable_reason"],
            )
        )

    journeys_list = []
    for raw_item in raw_journeys:
        item = _require_exact_fields(
            raw_item,
            {"journey_id", "title", "capability_ids"},
            "journey",
        )
        if not _is_exact_type(item["capability_ids"], list):
            raise ValueError("journey.capability_ids must be a JSON array")
        journeys_list.append(
            Journey(
                journey_id=item["journey_id"],
                title=item["title"],
                capability_ids=tuple(item["capability_ids"]),
            )
        )
    journeys = tuple(journeys_list)
    registry = CapabilityRegistry(
        schema_version=payload["schema_version"],
        observed_main_sha=payload["observed_main_sha"],
        observed_main_ci_run_id=ci["run_id"],
        observed_main_ci_conclusion=ci["conclusion"],
        capabilities=tuple(capabilities),
        journeys=journeys,
    )
    _validate_capability_registry_stored(registry)
    return registry


def load_source_surface_inventory(path: str | Path) -> SourceSurfaceInventory:
    payload = _strict_json_object(Path(path).read_bytes())
    payload = _require_exact_fields(
        payload,
        {
            "schema_version",
            "observed_main_sha",
            "observed_main_tree_sha",
            "source_root",
            "source_surface_count",
            "accepted_main_surface_count",
            "candidate_overlay_surface_count",
            "surfaces",
        },
        "source_surface_inventory",
    )
    raw_surfaces = payload["surfaces"]
    if not _is_exact_type(raw_surfaces, list):
        raise ValueError("source_surface_inventory.surfaces must be a JSON array")
    surfaces = tuple(
        SourceSurface(
            **_require_exact_fields(
                item,
                {"path", "capability_id", "origin"},
                "source_surface",
            )
        )
        for item in raw_surfaces
    )
    return SourceSurfaceInventory(
        schema_version=payload["schema_version"],
        observed_main_sha=payload["observed_main_sha"],
        observed_main_tree_sha=payload["observed_main_tree_sha"],
        source_root=payload["source_root"],
        source_surface_count=payload["source_surface_count"],
        accepted_main_surface_count=payload["accepted_main_surface_count"],
        candidate_overlay_surface_count=payload["candidate_overlay_surface_count"],
        surfaces=surfaces,
    )


def _python_source_blob_map(
    repo_root: Path,
    treeish: str,
    source_root: str,
) -> dict[str, str]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "ls-tree",
            "-rz",
            treeish,
            "--",
            source_root,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_git_subprocess_env(),
    )
    if completed.returncode != 0:
        raise ValueError(f"cannot enumerate source blobs for {treeish}")
    if completed.stdout and not completed.stdout.endswith("\0"):
        raise ValueError("git ls-tree source output is missing its NUL delimiter")

    blobs: dict[str, str] = {}
    prefix = f"{source_root}/"
    records = completed.stdout[:-1].split("\0") if completed.stdout else []
    for line in records:
        try:
            metadata, path = line.split("\t", 1)
            mode, kind, blob_sha = metadata.split()
        except ValueError as exc:
            raise ValueError("git ls-tree emitted a non-canonical source record") from exc
        if not path.startswith(prefix) or not path.endswith(".py"):
            continue
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError("source surface must be a regular Git blob")
        if _SHA40_RE.fullmatch(blob_sha) is None:
            raise ValueError("git ls-tree emitted a malformed source blob SHA")
        blobs[path] = f"{mode}:{blob_sha}"
    return blobs


def _changed_existing_source_paths(
    accepted_main_blobs: dict[str, str],
    checkout_blobs: dict[str, str],
) -> set[str]:
    return {
        path
        for path, blob_sha in accepted_main_blobs.items()
        if path in checkout_blobs and checkout_blobs[path] != blob_sha
    }


def _worktree_python_source_drift(
    repo_root: Path,
    source_root: str,
) -> set[str]:
    """Return tracked Python-source paths whose worktree bytes/mode differ from HEAD."""

    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "diff",
            "--name-only",
            "-z",
            "HEAD",
            "--",
            source_root,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_git_subprocess_env(),
    )
    if completed.returncode != 0:
        raise ValueError("cannot inspect Python source worktree drift")
    if completed.stdout and not completed.stdout.endswith("\0"):
        raise ValueError("git diff source output is missing its NUL delimiter")

    prefix = f"{source_root}/"
    changed: set[str] = set()
    paths = completed.stdout[:-1].split("\0") if completed.stdout else []
    for path in paths:
        candidate = PurePosixPath(path)
        if (
            "\\" in path
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != path
        ):
            raise ValueError("git diff emitted a non-canonical source path")
        if path.startswith(prefix) and path.endswith(".py"):
            changed.add(path)
    return changed


def _baseline_source_capability_map(
    repo_root: Path,
    observed_main_sha: str,
) -> dict[str, str]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "show",
            f"{observed_main_sha}:configs/control/product_source_surface_inventory_v1.json",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_git_subprocess_env(),
    )
    if completed.returncode != 0:
        raise ValueError("cannot read accepted-main source capability inventory")
    try:
        payload = _strict_json_object(completed.stdout.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValueError(
            "accepted-main source capability inventory is not canonical UTF-8"
        ) from exc
    payload = _require_exact_fields(
        payload,
        {
            "schema_version",
            "observed_main_sha",
            "observed_main_tree_sha",
            "source_root",
            "source_surface_count",
            "accepted_main_surface_count",
            "candidate_overlay_surface_count",
            "surfaces",
        },
        "accepted_main_source_surface_inventory",
    )
    raw_surfaces = payload["surfaces"]
    if not _is_exact_type(raw_surfaces, list):
        raise ValueError("accepted-main source surfaces must be a JSON array")

    mapping: dict[str, str] = {}
    for raw_surface in raw_surfaces:
        item = _require_exact_fields(
            raw_surface,
            {"path", "capability_id", "origin"},
            "accepted_main_source_surface",
        )
        surface = SourceSurface(**item)
        if surface.path in mapping:
            raise ValueError("accepted-main source surface paths must be unique")
        mapping[surface.path] = surface.capability_id
    return mapping


def validate_source_surface_coverage(
    registry: CapabilityRegistry,
    inventory: SourceSurfaceInventory,
    *,
    repo_root: str | Path,
) -> None:
    if not _is_exact_type(registry, CapabilityRegistry):
        raise ValueError("registry must be a CapabilityRegistry")
    if not _is_exact_type(inventory, SourceSurfaceInventory):
        raise ValueError("inventory must be a SourceSurfaceInventory")
    _validate_capability_registry_stored(registry)
    _source_surface_inventory_payload_from_stored_state(inventory)
    if registry.observed_main_sha != inventory.observed_main_sha:
        raise ValueError("capability and source inventories observe different main SHAs")

    by_capability = {
        item.capability_id: item for item in registry.capabilities
    }
    known_capability_ids = set(by_capability)
    mapped_capability_ids = {item.capability_id for item in inventory.surfaces}
    unknown = sorted(mapped_capability_ids.difference(known_capability_ids))
    if unknown:
        raise ValueError(f"source inventory maps unknown capability ids: {unknown}")

    premature_candidate_acceptance = sorted(
        f"{surface.path}->{surface.capability_id}"
        for surface in inventory.surfaces
        if surface.origin != "accepted_main"
        and by_capability[surface.capability_id].status
        is not CapabilityStatus.UNAVAILABLE
    )
    if premature_candidate_acceptance:
        raise ValueError(
            "candidate source surfaces must map to UNAVAILABLE capabilities "
            "until integrated: "
            f"{premature_candidate_acceptance}"
        )

    root = Path(repo_root)
    worktree_drift = _worktree_python_source_drift(root, inventory.source_root)
    if worktree_drift:
        raise ValueError(
            "source worktree differs from committed candidate HEAD: "
            f"{sorted(worktree_drift)}"
        )

    tree_check = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "show",
            "-s",
            "--format=%T",
            inventory.observed_main_sha,
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_git_subprocess_env(),
    )
    if tree_check.returncode != 0:
        raise ValueError("cannot resolve observed_main_sha in repository checkout")
    resolved_tree_sha = tree_check.stdout.strip()
    if resolved_tree_sha != inventory.observed_main_tree_sha:
        raise ValueError(
            "source inventory observed_main_tree_sha does not match observed_main_sha"
        )

    baseline_capability_map = _baseline_source_capability_map(
        root,
        inventory.observed_main_sha,
    )
    candidate_capability_map = {
        item.path: item.capability_id for item in inventory.surfaces
    }
    capability_mapping_drift = sorted(
        path
        for path, capability_id in baseline_capability_map.items()
        if path in candidate_capability_map
        and candidate_capability_map[path] != capability_id
    )
    if capability_mapping_drift:
        raise ValueError(
            "source capability mapping drift from accepted predecessor: "
            f"{capability_mapping_drift}"
        )

    accepted_main_blobs = _python_source_blob_map(
        root,
        inventory.observed_main_tree_sha,
        inventory.source_root,
    )
    accepted_main_actual = sorted(accepted_main_blobs)
    accepted_main_expected = [
        item.path
        for item in inventory.surfaces
        if item.origin in {"accepted_main", "modified_candidate"}
    ]
    if accepted_main_actual != accepted_main_expected:
        missing = sorted(set(accepted_main_actual).difference(accepted_main_expected))
        stale = sorted(set(accepted_main_expected).difference(accepted_main_actual))
        raise ValueError(
            "accepted-main source inventory drift: "
            f"unmapped_main={missing}, stale_main_inventory={stale}"
        )

    checkout_blobs = _python_source_blob_map(root, "HEAD", inventory.source_root)
    changed_existing = _changed_existing_source_paths(
        accepted_main_blobs,
        checkout_blobs,
    )
    modified_expected = {
        item.path
        for item in inventory.surfaces
        if item.origin == "modified_candidate"
    }
    if changed_existing != modified_expected:
        raise ValueError(
            "modified candidate source classification drift: "
            f"unmapped_modified={sorted(changed_existing - modified_expected)}, "
            f"stale_modified={sorted(modified_expected - changed_existing)}"
        )

    source_root = root / inventory.source_root
    checkout_actual = sorted(
        path.relative_to(root).as_posix()
        for path in source_root.rglob("*.py")
        if path.is_file()
    )
    checkout_expected = [item.path for item in inventory.surfaces]
    if checkout_actual != checkout_expected:
        missing = sorted(set(checkout_actual).difference(checkout_expected))
        stale = sorted(set(checkout_expected).difference(checkout_actual))
        raise ValueError(
            "stacked source inventory drift: "
            f"unmapped_checkout={missing}, stale_checkout_inventory={stale}"
        )

    overlay_actual = sorted(set(checkout_actual).difference(accepted_main_actual))
    overlay_expected = [
        item.path for item in inventory.surfaces if item.origin == "stacked_candidate"
    ]
    if overlay_actual != overlay_expected:
        raise ValueError("stacked candidate surface classification is non-canonical")

    journey_ids = {item.journey_id for item in registry.journeys}
    for capability_id in mapped_capability_ids:
        capability = by_capability[capability_id]
        if not capability.journey_ids:
            raise ValueError(
                f"source-mapped capability lacks a user/operator journey: {capability_id}"
            )
        if any(journey_id not in journey_ids for journey_id in capability.journey_ids):
            raise ValueError(
                f"source-mapped capability has an unknown journey: {capability_id}"
            )
