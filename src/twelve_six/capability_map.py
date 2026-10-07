from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_REGISTRY_BYTES = 1024 * 1024


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
    if not isinstance(data, bytes):
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
    if not isinstance(value, dict):
        raise ValueError("capability registry root must be an object")
    return value


def _require_exact_fields(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} schema is non-canonical")
    return value


def _require_id(name: str, value: object) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical identifier")
    return value


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


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


@dataclass(frozen=True, slots=True)
class EnvironmentSupport:
    environment_id: str
    supported: bool

    def __post_init__(self) -> None:
        _require_id("environment_id", self.environment_id)
        if not isinstance(self.supported, bool):
            raise ValueError("supported must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {"environment_id": self.environment_id, "supported": self.supported}


@dataclass(frozen=True, slots=True)
class TestVector:
    vector_id: str
    level: TestLevel
    command: str

    def __post_init__(self) -> None:
        _require_id("vector_id", self.vector_id)
        if not isinstance(self.level, TestLevel):
            raise ValueError("level must be a TestLevel")
        _require_text("command", self.command)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "level": self.level.value,
            "command": self.command,
        }


@dataclass(frozen=True, slots=True)
class EvidenceTarget:
    evidence_id: str
    target: str

    def __post_init__(self) -> None:
        _require_id("evidence_id", self.evidence_id)
        _require_text("target", self.target)

    def to_dict(self) -> dict[str, str]:
        return {"evidence_id": self.evidence_id, "target": self.target}


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
        _require_id("capability_id", self.capability_id)
        _require_positive_int("schema_version", self.schema_version)
        if not isinstance(self.status, CapabilityStatus):
            raise ValueError("status must be a CapabilityStatus")
        _require_text("component_contract", self.component_contract)

        for name, values in (
            ("dependencies", self.dependencies),
            ("journey_ids", self.journey_ids),
        ):
            if not isinstance(values, tuple):
                raise ValueError(f"{name} must be an immutable tuple")
            for value in values:
                _require_id(name, value)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")

        if not isinstance(self.environments, tuple) or any(
            not isinstance(item, EnvironmentSupport) for item in self.environments
        ):
            raise ValueError("environments must contain EnvironmentSupport values")
        if not isinstance(self.test_vectors, tuple) or any(
            not isinstance(item, TestVector) for item in self.test_vectors
        ):
            raise ValueError("test_vectors must contain TestVector values")
        if not isinstance(self.evidence_targets, tuple) or any(
            not isinstance(item, EvidenceTarget) for item in self.evidence_targets
        ):
            raise ValueError("evidence_targets must contain EvidenceTarget values")

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
            _require_text("integrated_result", self.integrated_result)
            if not any(item.supported for item in self.environments):
                raise ValueError("AVAILABLE capability needs a supported environment")
            levels = {item.level for item in self.test_vectors}
            if TestLevel.COMPONENT not in levels or TestLevel.INTEGRATION not in levels:
                raise ValueError(
                    "AVAILABLE capability needs component and integration test vectors"
                )
            if not self.evidence_targets:
                raise ValueError("AVAILABLE capability needs an evidence target")
        else:
            _require_text("unavailable_reason", self.unavailable_reason)
            if self.integrated_result is not None:
                raise ValueError("UNAVAILABLE capability cannot claim an integrated_result")

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "schema_version": self.schema_version,
            "status": self.status.value,
            "component_contract": self.component_contract,
            "dependencies": list(self.dependencies),
            "journey_ids": list(self.journey_ids),
            "environments": [item.to_dict() for item in self.environments],
            "test_vectors": [item.to_dict() for item in self.test_vectors],
            "evidence_targets": [item.to_dict() for item in self.evidence_targets],
            "integrated_result": self.integrated_result,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class Journey:
    journey_id: str
    title: str
    capability_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_id("journey_id", self.journey_id)
        _require_text("title", self.title)
        if not isinstance(self.capability_ids, tuple) or not self.capability_ids:
            raise ValueError("journey capability_ids must be a non-empty tuple")
        for capability_id in self.capability_ids:
            _require_id("journey capability_id", capability_id)
        if len(self.capability_ids) != len(set(self.capability_ids)):
            raise ValueError("journey capability_ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "journey_id": self.journey_id,
            "title": self.title,
            "capability_ids": list(self.capability_ids),
        }


@dataclass(frozen=True, slots=True)
class SourceSurface:
    path: str
    capability_id: str
    origin: str

    def __post_init__(self) -> None:
        _require_text("source surface path", self.path)
        _require_id("source surface capability_id", self.capability_id)
        if self.origin not in {"accepted_main", "stacked_candidate"}:
            raise ValueError("source surface origin is unsupported")
        if not self.path.startswith("src/twelve_six/") or not self.path.endswith(".py"):
            raise ValueError("source surface path must be a Python path under src/twelve_six")

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "capability_id": self.capability_id,
            "origin": self.origin,
        }


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
        _require_positive_int("schema_version", self.schema_version)
        if self.schema_version != 1:
            raise ValueError("unsupported SourceSurfaceInventory schema_version")
        for field_name, value in (
            ("observed_main_sha", self.observed_main_sha),
            ("observed_main_tree_sha", self.observed_main_tree_sha),
        ):
            if not isinstance(value, str) or _SHA40_RE.fullmatch(value) is None:
                raise ValueError(f"{field_name} must be a lowercase 40-hex Git SHA")
        if self.source_root != "src/twelve_six":
            raise ValueError("source_root must be canonical src/twelve_six")
        _require_positive_int("source_surface_count", self.source_surface_count)
        _require_positive_int(
            "accepted_main_surface_count", self.accepted_main_surface_count
        )
        _require_positive_int(
            "candidate_overlay_surface_count", self.candidate_overlay_surface_count
        )
        if not isinstance(self.surfaces, tuple) or not self.surfaces:
            raise ValueError("surfaces must be a non-empty tuple")
        if any(not isinstance(item, SourceSurface) for item in self.surfaces):
            raise ValueError("surfaces must contain only SourceSurface values")
        if self.source_surface_count != len(self.surfaces):
            raise ValueError("source_surface_count does not match surfaces")
        accepted_count = sum(item.origin == "accepted_main" for item in self.surfaces)
        candidate_count = sum(item.origin == "stacked_candidate" for item in self.surfaces)
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
        return {
            "schema_version": self.schema_version,
            "observed_main_sha": self.observed_main_sha,
            "observed_main_tree_sha": self.observed_main_tree_sha,
            "source_root": self.source_root,
            "source_surface_count": self.source_surface_count,
            "accepted_main_surface_count": self.accepted_main_surface_count,
            "candidate_overlay_surface_count": self.candidate_overlay_surface_count,
            "surfaces": [item.to_dict() for item in self.surfaces],
        }

    def identity_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class CapabilityRegistry:
    schema_version: int
    observed_main_sha: str
    observed_main_ci_run_id: int
    observed_main_ci_conclusion: str
    capabilities: tuple[Capability, ...]
    journeys: tuple[Journey, ...]

    def __post_init__(self) -> None:
        _require_positive_int("schema_version", self.schema_version)
        if self.schema_version != 1:
            raise ValueError("unsupported CapabilityRegistry schema_version")
        if not isinstance(self.observed_main_sha, str) or _SHA40_RE.fullmatch(
            self.observed_main_sha
        ) is None:
            raise ValueError("observed_main_sha must be a lowercase 40-hex Git SHA")
        _require_positive_int("observed_main_ci_run_id", self.observed_main_ci_run_id)
        if self.observed_main_ci_conclusion != "success":
            raise ValueError("observed main CI must be terminal success")

        if not isinstance(self.capabilities, tuple) or not self.capabilities:
            raise ValueError("capabilities must be a non-empty tuple")
        if any(not isinstance(item, Capability) for item in self.capabilities):
            raise ValueError("capabilities must contain only Capability values")
        if not isinstance(self.journeys, tuple) or not self.journeys:
            raise ValueError("journeys must be a non-empty tuple")
        if any(not isinstance(item, Journey) for item in self.journeys):
            raise ValueError("journeys must contain only Journey values")

        by_capability = {item.capability_id: item for item in self.capabilities}
        by_journey = {item.journey_id: item for item in self.journeys}
        if len(by_capability) != len(self.capabilities):
            raise ValueError("capability ids must be unique")
        if len(by_journey) != len(self.journeys):
            raise ValueError("journey ids must be unique")

        for capability in self.capabilities:
            for dependency_id in capability.dependencies:
                if dependency_id not in by_capability:
                    raise ValueError(
                        f"{capability.capability_id} has unknown dependency {dependency_id}"
                    )
                if (
                    capability.status is CapabilityStatus.AVAILABLE
                    and by_capability[dependency_id].status is CapabilityStatus.UNAVAILABLE
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

        self._reject_dependency_cycles(by_capability)

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
        _require_id("capability_id", capability_id)
        for capability in self.capabilities:
            if capability.capability_id == capability_id:
                return capability
        raise KeyError(capability_id)

    def journey_available(self, journey_id: str) -> bool:
        _require_id("journey_id", journey_id)
        journey = next(
            (item for item in self.journeys if item.journey_id == journey_id),
            None,
        )
        if journey is None:
            raise KeyError(journey_id)
        return all(
            self.capability(capability_id).status is CapabilityStatus.AVAILABLE
            for capability_id in journey.capability_ids
        )

    def acceptance_path(self, capability_id: str) -> dict[str, Any]:
        capability = self.capability(capability_id)
        if capability.status is CapabilityStatus.UNAVAILABLE:
            return {
                "capability_id": capability.capability_id,
                "status": capability.status.value,
                "component_contract": capability.component_contract,
                "unavailable_reason": capability.unavailable_reason,
                "integrated_result": None,
            }
        return {
            "capability_id": capability.capability_id,
            "status": capability.status.value,
            "component_contract": capability.component_contract,
            "test_vectors": [item.to_dict() for item in capability.test_vectors],
            "evidence_targets": [item.to_dict() for item in capability.evidence_targets],
            "integrated_result": capability.integrated_result,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "observed_main_sha": self.observed_main_sha,
            "observed_main_ci": {
                "run_id": self.observed_main_ci_run_id,
                "conclusion": self.observed_main_ci_conclusion,
            },
            "capabilities": [item.to_dict() for item in self.capabilities],
            "journeys": [item.to_dict() for item in self.journeys],
        }

    def identity_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def resolve_component_contract(component_contract: str) -> object:
    """Resolve one repository-owned Python module or module attribute fail-closed."""

    contract = _require_text("component_contract", component_contract)
    if not contract.startswith("twelve_six."):
        raise ValueError(
            "AVAILABLE component contract must be repository-owned twelve_six Python"
        )

    parts = contract.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            resolved: object = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name != module_name:
                raise ValueError(
                    f"component contract import failed inside module: {module_name}"
                ) from exc
            continue
        for attribute in parts[index:]:
            if not hasattr(resolved, attribute):
                raise ValueError(
                    f"component contract attribute does not exist: {contract}"
                )
            resolved = getattr(resolved, attribute)
        return resolved
    raise ValueError(f"component contract module does not exist: {contract}")


def validate_available_component_contracts(registry: CapabilityRegistry) -> None:
    """Prove every AVAILABLE capability begins at a live repository contract."""

    if not isinstance(registry, CapabilityRegistry):
        raise ValueError("registry must be a CapabilityRegistry")
    for capability in registry.capabilities:
        if capability.status is CapabilityStatus.AVAILABLE:
            resolve_component_contract(capability.component_contract)


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
    if not isinstance(raw_capabilities, list):
        raise ValueError("capabilities must be a JSON array")
    if not isinstance(raw_journeys, list):
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
            if not isinstance(item[list_field], list):
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
                level=TestLevel(vector["level"]),
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
                status=CapabilityStatus(item["status"]),
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
        if not isinstance(item["capability_ids"], list):
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
    validate_available_component_contracts(registry)
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
    if not isinstance(raw_surfaces, list):
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


def validate_source_surface_coverage(
    registry: CapabilityRegistry,
    inventory: SourceSurfaceInventory,
    *,
    repo_root: str | Path,
) -> None:
    if not isinstance(registry, CapabilityRegistry):
        raise ValueError("registry must be a CapabilityRegistry")
    if not isinstance(inventory, SourceSurfaceInventory):
        raise ValueError("inventory must be a SourceSurfaceInventory")
    if registry.observed_main_sha != inventory.observed_main_sha:
        raise ValueError("capability and source inventories observe different main SHAs")

    known_capability_ids = {item.capability_id for item in registry.capabilities}
    mapped_capability_ids = {item.capability_id for item in inventory.surfaces}
    unknown = sorted(mapped_capability_ids.difference(known_capability_ids))
    if unknown:
        raise ValueError(f"source inventory maps unknown capability ids: {unknown}")

    root = Path(repo_root)
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
    )
    if tree_check.returncode != 0:
        raise ValueError("cannot resolve observed_main_sha in repository checkout")
    resolved_tree_sha = tree_check.stdout.strip()
    if resolved_tree_sha != inventory.observed_main_tree_sha:
        raise ValueError(
            "source inventory observed_main_tree_sha does not match observed_main_sha"
        )

    listing = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-tree",
            "-r",
            "--name-only",
            inventory.observed_main_tree_sha,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if listing.returncode != 0:
        raise ValueError("cannot enumerate observed main source tree")

    prefix = f"{inventory.source_root}/"
    accepted_main_actual = sorted(
        line.strip()
        for line in listing.stdout.splitlines()
        if line.strip().startswith(prefix) and line.strip().endswith(".py")
    )
    accepted_main_expected = [
        item.path for item in inventory.surfaces if item.origin == "accepted_main"
    ]
    if accepted_main_actual != accepted_main_expected:
        missing = sorted(set(accepted_main_actual).difference(accepted_main_expected))
        stale = sorted(set(accepted_main_expected).difference(accepted_main_actual))
        raise ValueError(
            "accepted-main source inventory drift: "
            f"unmapped_main={missing}, stale_main_inventory={stale}"
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
        capability = registry.capability(capability_id)
        if not capability.journey_ids:
            raise ValueError(
                f"source-mapped capability lacks a user/operator journey: {capability_id}"
            )
        if any(journey_id not in journey_ids for journey_id in capability.journey_ids):
            raise ValueError(
                f"source-mapped capability has an unknown journey: {capability_id}"
            )
