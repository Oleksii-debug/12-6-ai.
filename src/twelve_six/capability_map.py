from __future__ import annotations

import hashlib
import json
import math
import re
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
class CapabilityRegistry:
    schema_version: int
    observed_main_sha: str
    observed_main_ci_run_id: int
    observed_main_ci_conclusion: str
    capabilities: tuple[Capability, ...]
    journeys: tuple[Journey, ...]

    def __post_init__(self) -> None:
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
    return CapabilityRegistry(
        schema_version=payload["schema_version"],
        observed_main_sha=payload["observed_main_sha"],
        observed_main_ci_run_id=ci["run_id"],
        observed_main_ci_conclusion=ci["conclusion"],
        capabilities=tuple(capabilities),
        journeys=journeys,
    )
