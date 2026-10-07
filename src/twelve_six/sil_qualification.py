from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from twelve_six.capability_map import (
    CapabilityRegistry,
    CapabilityStatus,
    TestLevel,
    load_capability_registry,
)
from twelve_six.model import InitSpec, ModelSpec

_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_MAX_SCENARIO_BYTES = 64 * 1024
_MAX_EVIDENCE_BYTES = 4 * 1024 * 1024
_MAX_ENVIRONMENT_RECEIPT_BYTES = 256 * 1024

_SIL_ENVIRONMENT_LOCK_SOURCE_COMMIT = "029514654829cebc149cff6fc1fea2a8ba4fa566"
_SIL_ENVIRONMENT_LOCKS = (
    (
        "toolchain",
        "requirements/locks/linux-x86_64/toolchain.lock.txt",
        "06b3f872e648e0b7f3cc33a92bd8b04709dea5d9a9821df4f2e8141f646cdcf3",
    ),
    (
        "cpu_runtime",
        "requirements/execution/linux-x86_64/cpu-runtime.lock.txt",
        "03e08dd06ff446651dcc6950d0f433325bb32261d3e2406b34506cd00e1be52a",
    ),
    (
        "dev",
        "requirements/locks/linux-x86_64/dev.lock.txt",
        "1869e05c5eeaf056813df1c99dc63c7e5febb9794351ec640d265f62efaeac28",
    ),
)
_SIL_ENVIRONMENT_PACKAGES = (
    ("filelock", "3.32.4"),
    ("fsspec", "2026.7.0"),
    ("iniconfig", "2.3.0"),
    ("jinja2", "3.1.6"),
    ("markupsafe", "3.0.3"),
    ("mpmath", "1.3.0"),
    ("networkx", "3.6.1"),
    ("numpy", "2.4.6"),
    ("packaging", "26.3"),
    ("pip", "26.2.1"),
    ("pluggy", "1.6.0"),
    ("pygments", "2.21.0"),
    ("pytest", "9.1.1"),
    ("ruff", "0.16.4"),
    ("safetensors", "0.8.0"),
    ("setuptools", "84.0.0"),
    ("sympy", "1.14.0"),
    ("torch", "2.13.0+cpu"),
    ("twelve-six-ai", "0.2.0.dev0"),
    ("typing-extensions", "4.16.0"),
    ("wheel", "0.48.0"),
)
_DIST_NAME_RE = re.compile(r"[-_.]+")

# Section 3.1 requires journey-level end-to-end execution, not merely a bag of
# integration-green capabilities.  This sealed policy is the explicit accepted
# declaration of which ordered integration steps jointly constitute each
# currently AVAILABLE journey under one shared SIL input envelope.  build_sil_plan()
# captures this object as a default argument so a module-global rebind cannot
# reseal end-to-end semantics in an already-loaded validator.
_CANONICAL_JOURNEY_E2E_VECTOR_POLICY = (
    ("developer-model-contract", ("model-resource-receipt",)),
    (
        "researcher-prepare-training-inputs",
        ("tokenizer-migration", "packing-stream"),
    ),
    ("operator-resume-checkpoint", ("checkpoint-roundtrip",)),
    ("operator-windows-cli", ("windows-cli-packaging",)),
    (
        "developer-replace-cognitive-core",
        ("section0-product-stack", "section1-stack"),
    ),
    ("researcher-training-mechanics", ("trainer-preflight",)),
    ("data-curator-governance", ("data-clean-dedup-decontam",)),
    ("operator-learned20m-readiness", ("learned20m-lease-control",)),
    ("operator-portable-run", ("portable-run-binding",)),
    ("operator-scale141-recovery", ("scale141-content-addressed",)),
    ("researcher-split-validation", ("split-robustness-manifest",)),
    ("maintainer-project-control", ("swarm-protocol",)),
    ("maintainer-capability-qualification", ("section2-stack",)),
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob_sha1(value: bytes) -> str:
    """Return the repository's SHA-1 Git blob identity for exact worktree bytes."""

    header = f"blob {len(value)}\0".encode("ascii")
    return hashlib.sha1(header + value, usedforsecurity=False).hexdigest()


def build_package_manifest_bytes(repo_root: str | Path) -> bytes:
    """Bind the editable package to exact tracked source and packaged research configs."""

    root = Path(repo_root)
    listed = subprocess.run(
        [
            "git",
            "ls-files",
            "--stage",
            "-z",
            "--",
            "pyproject.toml",
            "src/twelve_six",
            "configs/research",
        ],
        cwd=root,
        env=_qualification_subprocess_env(),
        stdin=subprocess.DEVNULL,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if listed.returncode != 0:
        raise ValueError("cannot enumerate tracked package source files")
    if listed.stdout and not listed.stdout.endswith("\0"):
        raise ValueError("tracked package source listing is missing its NUL delimiter")

    entries: list[tuple[str, str, str]] = []
    records = listed.stdout[:-1].split("\0") if listed.stdout else []
    for line in records:
        try:
            metadata, relative_path = line.split("\t", 1)
            mode, blob_sha, stage = metadata.split()
        except ValueError as exc:
            raise ValueError("tracked package source record is non-canonical") from exc
        if stage != "0":
            raise ValueError("tracked package source has a non-zero Git index stage")
        if mode not in {"100644", "100755"}:
            raise ValueError("tracked package source must be a regular Git file")
        if _SHA40_RE.fullmatch(blob_sha) is None:
            raise ValueError("tracked package source has a malformed Git blob SHA")
        canonical_path = PurePosixPath(relative_path)
        if (
            "\\" in relative_path
            or canonical_path.is_absolute()
            or ".." in canonical_path.parts
            or canonical_path.as_posix() != relative_path
        ):
            raise ValueError("tracked package source path is non-canonical")
        entries.append((relative_path, mode, blob_sha))

    entries.sort(key=lambda item: item[0])
    paths = [item[0] for item in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("tracked package source paths must be unique")
    if "pyproject.toml" not in paths:
        raise ValueError("tracked package manifest is missing pyproject.toml")
    if not any(path.startswith("src/twelve_six/") for path in paths):
        raise ValueError("tracked package manifest has no twelve_six package source")

    files = []
    for relative_path, mode, blob_sha in entries:
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"tracked package source is not a file: {relative_path}")
        raw = path.read_bytes()
        if _git_blob_sha1(raw) != blob_sha:
            raise ValueError(
                f"tracked package source bytes do not match Git blob: {relative_path}"
            )
        files.append(
            {
                "path": relative_path,
                "git_mode": mode,
                "git_blob_sha": blob_sha,
                "bytes": len(raw),
                "sha256": _sha256_bytes(raw),
            }
        )
    return _canonical_json_bytes(
        {
            "schema_version": "12-6.package-source-manifest.v1",
            "files": files,
        }
    )


def _require_sha256(name: str, value: object) -> str:
    if not _is_exact_type(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _require_git_sha(name: str, value: object) -> str:
    if not _is_exact_type(value, str) or _SHA40_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase 40-hex Git SHA")
    return value


def _is_exact_type(value: object, expected: type[object]) -> bool:
    # SIL evidence schemas reject behavioral subclasses that can reseal validated state.
    return type(value) is expected


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object member")
        value[key] = item
    return value


def _strict_json_object(data: bytes, *, maximum_bytes: int, label: str) -> dict[str, Any]:
    if not _is_exact_type(data, bytes):
        raise ValueError(f"{label} input must be bytes")
    if len(data) > maximum_bytes:
        raise ValueError(f"{label} exceeds maximum encoded size")
    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant is not allowed: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} is not strict unambiguous UTF-8 JSON") from exc
    if not _is_exact_type(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return value


def _canonical_distribution_name(value: str) -> str:
    return _DIST_NAME_RE.sub("-", value.strip()).lower()


def _build_sil_environment_receipt_authority():
    # Capture the qualified lock-source contract once. Later module-global rebinding
    # must not be able to mint a different self-consistent "canonical" environment.
    sealed_lock_source_commit = _SIL_ENVIRONMENT_LOCK_SOURCE_COMMIT
    sealed_locks = _SIL_ENVIRONMENT_LOCKS
    sealed_packages = _SIL_ENVIRONMENT_PACKAGES
    sealed_hash = _canonical_sha256

    def canonical() -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": "12-6.sil-environment-receipt.v1",
            "python": {
                "implementation": "cpython",
                "version": "3.11.16",
            },
            "lock_source_commit": sealed_lock_source_commit,
            "locks": [
                {
                    "role": role,
                    "path": path,
                    "sha256": digest,
                }
                for role, path, digest in sealed_locks
            ],
            "packages": [
                {"name": name, "version": version}
                for name, version in sealed_packages
            ],
        }
        payload["identity_sha256"] = sealed_hash(payload)
        return payload

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        expected = canonical()
        if payload != expected:
            raise ValueError(
                "SIL environment receipt does not match exact pinned lock-source contract"
            )
        return payload

    return canonical, validate


(
    canonical_sil_environment_receipt_v1,
    _validate_sil_environment_receipt,
) = _build_sil_environment_receipt_authority()


def load_sil_environment_receipt(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    payload = _strict_json_object(
        raw,
        maximum_bytes=_MAX_ENVIRONMENT_RECEIPT_BYTES,
        label="SIL environment receipt",
    )
    if raw != _canonical_json_bytes(payload) + b"\n":
        raise ValueError("SIL environment receipt bytes are non-canonical")
    return _validate_sil_environment_receipt(payload)


def _installed_distribution_versions() -> dict[str, str]:
    observed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("installed distribution has no canonical Name metadata")
        name = _canonical_distribution_name(raw_name)
        if name in observed:
            raise ValueError(f"duplicate installed distribution name: {name}")
        observed[name] = distribution.version
    return observed


def require_current_sil_environment(
    receipt: dict[str, Any],
) -> dict[str, Any]:
    receipt = _validate_sil_environment_receipt(receipt)
    if sys.implementation.name != receipt["python"]["implementation"]:
        raise ValueError("SIL Python implementation does not match environment receipt")
    if platform.python_version() != receipt["python"]["version"]:
        raise ValueError("SIL Python version does not match environment receipt")
    expected_packages = {
        item["name"]: item["version"] for item in receipt["packages"]
    }
    observed_packages = _installed_distribution_versions()
    if observed_packages != expected_packages:
        missing = sorted(set(expected_packages) - set(observed_packages))
        unexpected = sorted(set(observed_packages) - set(expected_packages))
        version_drift = sorted(
            name
            for name in set(expected_packages) & set(observed_packages)
            if expected_packages[name] != observed_packages[name]
        )
        raise ValueError(
            "SIL installed distribution set does not match environment receipt: "
            f"missing={missing}, unexpected={unexpected}, "
            f"version_drift={version_drift}"
        )
    return receipt


@dataclass(frozen=True, slots=True)
class SILScenario:
    schema_version: int
    scenario_id: str
    journey_selector: str
    fixture_policy: str
    synthetic_data_utf8: str
    timeout_seconds_per_vector: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported SILScenario schema_version")
        if not _is_exact_type(self.scenario_id, str) or _ID_RE.fullmatch(self.scenario_id) is None:
            raise ValueError("scenario_id must be a canonical identifier")
        if (
            not _is_exact_type(self.journey_selector, str)
            or self.journey_selector != "ALL_AVAILABLE"
        ):
            raise ValueError("journey_selector must be ALL_AVAILABLE")
        if (
            not _is_exact_type(self.fixture_policy, str)
            or self.fixture_policy != "DETERMINISTIC_SYNTHETIC"
        ):
            raise ValueError("fixture_policy must be DETERMINISTIC_SYNTHETIC")
        if not _is_exact_type(self.synthetic_data_utf8, str) or not self.synthetic_data_utf8:
            raise ValueError("synthetic_data_utf8 must be non-empty text")
        if len(self.synthetic_data_utf8.encode("utf-8")) > 64 * 1024:
            raise ValueError("synthetic_data_utf8 is too large")
        if (
            type(self.timeout_seconds_per_vector) is not int
            or not 1 <= self.timeout_seconds_per_vector <= 900
        ):
            raise ValueError("timeout_seconds_per_vector must be an integer in [1, 900]")

    def to_dict(self) -> dict[str, Any]:
        SILScenario.__post_init__(self)
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "journey_selector": self.journey_selector,
            "fixture_policy": self.fixture_policy,
            "synthetic_data_utf8": self.synthetic_data_utf8,
            "timeout_seconds_per_vector": self.timeout_seconds_per_vector,
        }

    def identity_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def load_sil_scenario(path: str | Path) -> SILScenario:
    payload = _strict_json_object(
        Path(path).read_bytes(),
        maximum_bytes=_MAX_SCENARIO_BYTES,
        label="SIL scenario",
    )
    expected = {
        "schema_version",
        "scenario_id",
        "journey_selector",
        "fixture_policy",
        "synthetic_data_utf8",
        "timeout_seconds_per_vector",
    }
    if set(payload) != expected:
        raise ValueError("SIL scenario fields are non-canonical")
    return SILScenario(**payload)


@dataclass(frozen=True, slots=True)
class PlannedVector:
    journey_id: str
    capability_id: str
    vector_id: str
    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("journey_id", self.journey_id),
            ("capability_id", self.capability_id),
            ("vector_id", self.vector_id),
        ):
            if not _is_exact_type(value, str) or _ID_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a canonical identifier")
        if (
            not _is_exact_type(self.argv, tuple)
            or len(self.argv) < 5
            or self.argv[:4] != (sys.executable, "-m", "pytest", "-q")
        ):
            raise ValueError("planned vector argv must be canonical no-shell pytest argv")
        for token in self.argv[4:]:
            if not _is_exact_type(token, str) or not token:
                raise ValueError("planned vector test path must be non-empty text")
            path = PurePosixPath(token)
            if (
                "\\" in token
                or token.startswith("-")
                or path.is_absolute()
                or ".." in path.parts
                or len(path.parts) < 2
                or path.parts[0] != "tests"
                or path.suffix != ".py"
                or path.as_posix() != token
            ):
                raise ValueError("planned vector may reference only canonical tests/*.py paths")

    def to_dict(self) -> dict[str, Any]:
        PlannedVector.__post_init__(self)
        return {
            "journey_id": self.journey_id,
            "capability_id": self.capability_id,
            "vector_id": self.vector_id,
            "argv": list(self.argv),
        }


@dataclass(frozen=True, slots=True)
class UnavailableJourney:
    journey_id: str
    blocking_capability_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _is_exact_type(self.journey_id, str) or _ID_RE.fullmatch(self.journey_id) is None:
            raise ValueError("unavailable journey_id must be a canonical identifier")
        if (
            not _is_exact_type(self.blocking_capability_ids, tuple)
            or not self.blocking_capability_ids
            or not _is_exact_type(self.reasons, tuple)
            or len(self.reasons) != len(self.blocking_capability_ids)
        ):
            raise ValueError("unavailable journey blockers and reasons are non-canonical")
        for capability_id in self.blocking_capability_ids:
            if not _is_exact_type(capability_id, str) or _ID_RE.fullmatch(capability_id) is None:
                raise ValueError("blocking capability id must be canonical")
        if len(set(self.blocking_capability_ids)) != len(self.blocking_capability_ids):
            raise ValueError("blocking capability ids must be unique")
        if any(not _is_exact_type(reason, str) or not reason.strip() for reason in self.reasons):
            raise ValueError("unavailable journey reasons must be non-empty text")

    def to_dict(self) -> dict[str, Any]:
        UnavailableJourney.__post_init__(self)
        return {
            "journey_id": self.journey_id,
            "blocking_capability_ids": list(self.blocking_capability_ids),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class SILPlan:
    available_journey_ids: tuple[str, ...]
    unavailable_journeys: tuple[UnavailableJourney, ...]
    journey_end_to_end_contracts: tuple[tuple[str, tuple[str, ...]], ...]
    vectors: tuple[PlannedVector, ...]

    def __post_init__(self) -> None:
        if (
            not _is_exact_type(self.available_journey_ids, tuple)
            or not self.available_journey_ids
            or any(
                not _is_exact_type(journey_id, str)
                or _ID_RE.fullmatch(journey_id) is None
                for journey_id in self.available_journey_ids
            )
            or len(set(self.available_journey_ids)) != len(self.available_journey_ids)
        ):
            raise ValueError("SIL plan AVAILABLE journey ids are non-canonical")
        if (
            not _is_exact_type(self.unavailable_journeys, tuple)
            or any(
                not _is_exact_type(item, UnavailableJourney)
                for item in self.unavailable_journeys
            )
        ):
            raise ValueError("SIL plan unavailable journeys are non-canonical")
        unavailable_ids = tuple(item.journey_id for item in self.unavailable_journeys)
        if len(set(unavailable_ids)) != len(unavailable_ids):
            raise ValueError("SIL plan unavailable journey ids must be unique")
        if set(unavailable_ids).intersection(self.available_journey_ids):
            raise ValueError("SIL plan journey cannot be both AVAILABLE and UNAVAILABLE")
        if (
            not _is_exact_type(self.vectors, tuple)
            or not self.vectors
            or any(not _is_exact_type(item, PlannedVector) for item in self.vectors)
        ):
            raise ValueError("SIL plan integration vectors are non-canonical")
        for item in self.unavailable_journeys:
            UnavailableJourney.__post_init__(item)
        for item in self.vectors:
            PlannedVector.__post_init__(item)
        if any(
            vector.journey_id not in self.available_journey_ids
            for vector in self.vectors
        ):
            raise ValueError("SIL plan vector references a non-AVAILABLE journey")
        if (
            not _is_exact_type(self.journey_end_to_end_contracts, tuple)
            or not self.journey_end_to_end_contracts
        ):
            raise ValueError("SIL plan needs explicit end-to-end journey contracts")
        contract_journey_ids: list[str] = []
        for item in self.journey_end_to_end_contracts:
            if (
                not _is_exact_type(item, tuple)
                or len(item) != 2
                or not _is_exact_type(item[0], str)
                or _ID_RE.fullmatch(item[0]) is None
                or not _is_exact_type(item[1], tuple)
                or not item[1]
                or any(
                    not _is_exact_type(vector_id, str)
                    or _ID_RE.fullmatch(vector_id) is None
                    for vector_id in item[1]
                )
                or len(set(item[1])) != len(item[1])
            ):
                raise ValueError("SIL end-to-end journey contract is non-canonical")
            contract_journey_ids.append(item[0])
        if tuple(contract_journey_ids) != self.available_journey_ids:
            raise ValueError(
                "SIL end-to-end contracts must exactly match AVAILABLE journey order"
            )
        for journey_id, declared_vector_ids in self.journey_end_to_end_contracts:
            observed_vector_ids = tuple(
                vector.vector_id
                for vector in self.vectors
                if vector.journey_id == journey_id
            )
            if observed_vector_ids != declared_vector_ids:
                raise ValueError(
                    "SIL plan actual vectors do not match declared end-to-end contract: "
                    f"{journey_id}"
                )

    def to_dict(self) -> dict[str, Any]:
        SILPlan.__post_init__(self)
        return {
            "available_journey_ids": list(self.available_journey_ids),
            "unavailable_journeys": [item.to_dict() for item in self.unavailable_journeys],
            "journey_end_to_end_contracts": [
                {
                    "journey_id": journey_id,
                    "execution_mode": "SEQUENTIAL_SHARED_INPUT_ENVELOPE",
                    "completion_rule": "ALL_DECLARED_STEPS_PASS_IN_ORDER",
                    "vector_ids": list(vector_ids),
                }
                for journey_id, vector_ids in self.journey_end_to_end_contracts
            ],
            "vectors": [item.to_dict() for item in self.vectors],
        }


def _build_sil_object_stored_state_authority():
    scenario_validator = SILScenario.__post_init__
    planned_vector_validator = PlannedVector.__post_init__
    unavailable_journey_validator = UnavailableJourney.__post_init__
    plan_validator = SILPlan.__post_init__

    def validate_scenario(value: SILScenario) -> None:
        if not _is_exact_type(value, SILScenario):
            raise ValueError("scenario must be a SILScenario")
        scenario_validator(value)

    def scenario_payload(value: SILScenario) -> dict[str, Any]:
        validate_scenario(value)
        return {
            "schema_version": value.schema_version,
            "scenario_id": value.scenario_id,
            "journey_selector": value.journey_selector,
            "fixture_policy": value.fixture_policy,
            "synthetic_data_utf8": value.synthetic_data_utf8,
            "timeout_seconds_per_vector": value.timeout_seconds_per_vector,
        }

    def scenario_identity(value: SILScenario) -> str:
        return _canonical_sha256(scenario_payload(value))

    def validate_planned_vector(value: PlannedVector) -> None:
        if not _is_exact_type(value, PlannedVector):
            raise ValueError("SIL plan vectors must be exact PlannedVector values")
        planned_vector_validator(value)

    def planned_vector_payload(value: PlannedVector) -> dict[str, Any]:
        validate_planned_vector(value)
        return {
            "journey_id": value.journey_id,
            "capability_id": value.capability_id,
            "vector_id": value.vector_id,
            "argv": list(value.argv),
        }

    def validate_unavailable_journey(value: UnavailableJourney) -> None:
        if not _is_exact_type(value, UnavailableJourney):
            raise ValueError(
                "SIL plan unavailable journeys must be exact UnavailableJourney values"
            )
        unavailable_journey_validator(value)

    def unavailable_journey_payload(
        value: UnavailableJourney,
    ) -> dict[str, Any]:
        validate_unavailable_journey(value)
        return {
            "journey_id": value.journey_id,
            "blocking_capability_ids": list(value.blocking_capability_ids),
            "reasons": list(value.reasons),
        }

    def validate_plan(value: SILPlan) -> None:
        if not _is_exact_type(value, SILPlan):
            raise ValueError("plan must be a SILPlan")
        plan_validator(value)
        for item in value.unavailable_journeys:
            validate_unavailable_journey(item)
        for item in value.vectors:
            validate_planned_vector(item)

    def plan_payload(value: SILPlan) -> dict[str, Any]:
        validate_plan(value)
        return {
            "available_journey_ids": list(value.available_journey_ids),
            "unavailable_journeys": [
                unavailable_journey_payload(item)
                for item in value.unavailable_journeys
            ],
            "journey_end_to_end_contracts": [
                {
                    "journey_id": journey_id,
                    "execution_mode": "SEQUENTIAL_SHARED_INPUT_ENVELOPE",
                    "completion_rule": "ALL_DECLARED_STEPS_PASS_IN_ORDER",
                    "vector_ids": list(vector_ids),
                }
                for journey_id, vector_ids in value.journey_end_to_end_contracts
            ],
            "vectors": [planned_vector_payload(item) for item in value.vectors],
        }

    return validate_scenario, scenario_identity, validate_plan, plan_payload


(
    _validate_sil_scenario_stored,
    _sil_scenario_identity_from_stored_state,
    _validate_sil_plan_stored,
    _sil_plan_payload_from_stored_state,
) = _build_sil_object_stored_state_authority()


def parse_vector_command(command: str) -> tuple[str, ...]:
    if not _is_exact_type(command, str) or not command.strip():
        raise ValueError("integration vector command must be non-empty")
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError("integration vector command is not shell-tokenizable") from exc
    if len(tokens) < 3 or tokens[:2] != ["pytest", "-q"]:
        raise ValueError("SIL integration vectors must use 'pytest -q <test files>'")
    test_paths = tokens[2:]
    for token in test_paths:
        if token.startswith("-") or any(char in token for char in (";", "|", "&", ">", "<", "$")):
            raise ValueError("SIL integration vector contains shell/control syntax")
        path = PurePosixPath(token)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("SIL integration test path must stay inside the repository")
        if len(path.parts) < 2 or path.parts[0] != "tests" or path.suffix != ".py":
            raise ValueError("SIL integration vectors may reference only tests/*.py files")
    return (sys.executable, "-m", "pytest", "-q", *test_paths)


def _build_sil_plan_with_policy(
    registry: CapabilityRegistry,
    scenario: SILScenario,
    *,
    e2e_policy: tuple[tuple[str, tuple[str, ...]], ...],
    _registry_validator: Callable[[CapabilityRegistry], None] = (
        CapabilityRegistry.__post_init__
    ),
    _scenario_validator: Callable[[SILScenario], None] = (
        _validate_sil_scenario_stored
    ),
) -> SILPlan:
    if not _is_exact_type(registry, CapabilityRegistry):
        raise ValueError("registry must be a CapabilityRegistry")
    if not _is_exact_type(scenario, SILScenario):
        raise ValueError("scenario must be a SILScenario")
    _registry_validator(registry)
    _scenario_validator(scenario)
    if scenario.journey_selector != "ALL_AVAILABLE":
        raise ValueError("unsupported journey selection")

    if (
        not _is_exact_type(e2e_policy, tuple)
        or any(
            not _is_exact_type(item, tuple)
            or len(item) != 2
            or not _is_exact_type(item[0], str)
            or not _is_exact_type(item[1], tuple)
            for item in e2e_policy
        )
    ):
        raise ValueError("SIL end-to-end policy is non-canonical")
    policy_journey_ids = tuple(item[0] for item in e2e_policy)
    if len(set(policy_journey_ids)) != len(policy_journey_ids):
        raise ValueError("SIL end-to-end policy journey ids must be unique")
    policy_by_journey = dict(e2e_policy)

    available: list[str] = []
    unavailable: list[UnavailableJourney] = []
    journey_contracts: list[tuple[str, tuple[str, ...]]] = []
    vectors: list[PlannedVector] = []

    for journey in registry.journeys:
        if registry.journey_available(journey.journey_id):
            available.append(journey.journey_id)
            declared_vector_ids = policy_by_journey.get(journey.journey_id)
            if declared_vector_ids is None:
                raise ValueError(
                    f"AVAILABLE journey lacks explicit end-to-end contract: "
                    f"{journey.journey_id}"
                )
            journey_vectors: list[PlannedVector] = []
            for capability_id in journey.capability_ids:
                capability = registry.capability(capability_id)
                if capability.status is not CapabilityStatus.AVAILABLE:
                    raise ValueError("available journey contains an unavailable capability")
                integration_vectors = tuple(
                    vector
                    for vector in capability.test_vectors
                    if vector.level is TestLevel.INTEGRATION
                )
                if not integration_vectors:
                    raise ValueError(
                        f"AVAILABLE capability lacks an integration vector: {capability_id}"
                    )
                for vector in integration_vectors:
                    journey_vectors.append(
                        PlannedVector(
                            journey_id=journey.journey_id,
                            capability_id=capability_id,
                            vector_id=vector.vector_id,
                            argv=parse_vector_command(vector.command),
                        )
                    )
            observed_vector_ids = tuple(
                vector.vector_id for vector in journey_vectors
            )
            if observed_vector_ids != declared_vector_ids:
                raise ValueError(
                    "AVAILABLE journey integration vectors do not match its explicit "
                    f"end-to-end contract: {journey.journey_id}"
                )
            journey_contracts.append(
                (journey.journey_id, declared_vector_ids)
            )
            vectors.extend(journey_vectors)
            continue

        blockers = tuple(
            registry.capability(capability_id)
            for capability_id in journey.capability_ids
            if registry.capability(capability_id).status is CapabilityStatus.UNAVAILABLE
        )
        if not blockers:
            raise ValueError("unavailable journey has no explicit unavailable capability")
        unavailable.append(
            UnavailableJourney(
                journey_id=journey.journey_id,
                blocking_capability_ids=tuple(item.capability_id for item in blockers),
                reasons=tuple(item.unavailable_reason or "" for item in blockers),
            )
        )

    if tuple(available) != policy_journey_ids:
        raise ValueError(
            "SIL end-to-end policy must exactly match the current AVAILABLE journeys"
        )

    return SILPlan(
        available_journey_ids=tuple(available),
        unavailable_journeys=tuple(unavailable),
        journey_end_to_end_contracts=tuple(journey_contracts),
        vectors=tuple(vectors),
    )


def _build_sil_plan_authority():
    # Public planning authority closes over the repository-owned end-to-end
    # journey policy. Tests may exercise the underscore-prefixed helper with
    # alternate policies, but production callers cannot inject plan authority.
    sealed_impl = _build_sil_plan_with_policy
    sealed_policy = _CANONICAL_JOURNEY_E2E_VECTOR_POLICY
    sealed_validate_scenario = _validate_sil_scenario_stored
    sealed_validate_plan = _validate_sil_plan_stored

    def canonical(
        registry: CapabilityRegistry,
        scenario: SILScenario,
    ) -> SILPlan:
        sealed_validate_scenario(scenario)
        plan = sealed_impl(
            registry,
            scenario,
            e2e_policy=sealed_policy,
        )
        sealed_validate_plan(plan)
        return plan

    return canonical


build_sil_plan = _build_sil_plan_authority()


@dataclass(frozen=True, slots=True)
class GitState:
    sha: str
    tracked_clean: bool

    def __post_init__(self) -> None:
        _require_git_sha("GitState.sha", self.sha)
        if not _is_exact_type(self.tracked_clean, bool):
            raise ValueError("GitState.tracked_clean must be boolean")


def _qualification_subprocess_env() -> dict[str, str]:
    """Build the bounded environment used by SIL Git and journey subprocesses."""

    blocked_prefixes = ("GIT_", "PYTHON", "PYTEST")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(blocked_prefixes)
    }
    env.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONUTF8": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        }
    )
    return env


def probe_git_state(repo_root: str | Path) -> GitState:
    root = Path(repo_root)
    sha_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        env=_qualification_subprocess_env(),
        stdin=subprocess.DEVNULL,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    sha = sha_result.stdout.strip()
    _require_git_sha("observed git SHA", sha)

    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        env=_qualification_subprocess_env(),
        stdin=subprocess.DEVNULL,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    if status.returncode != 0:
        raise ValueError("cannot establish SIL checkout cleanliness")
    return GitState(sha=sha, tracked_clean=not status.stdout.strip())


_INPUT_VERIFICATION_PREFIX = "SIL_INPUT_VERIFIED_SHA256="
_EXECUTION_WRAPPER = (
    "import hashlib,os,sys\n"
    "expected=sys.argv[1]\n"
    "raw=sys.argv[2].encode('utf-8')\n"
    "actual=hashlib.sha256(raw).hexdigest()\n"
    "if actual != expected: raise SystemExit(86)\n"
    "print('SIL_INPUT_VERIFIED_SHA256='+actual, flush=True)\n"
    "os.environ['TWELVE_SIX_SIL_INPUT_IDENTITY_SHA256']=actual\n"
    "os.environ['TWELVE_SIX_SIL_INPUT_ENVELOPE_JSON']=raw.decode('utf-8')\n"
    "os.execv(sys.argv[3], sys.argv[3:])\n"
)


@dataclass(frozen=True, slots=True)
class CommandExecution:
    return_code: int
    stdout: str
    stderr: str
    duration_ms: int
    consumed_input_identity_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.return_code) is not int:
            raise ValueError("CommandExecution.return_code must be an integer")
        if not _is_exact_type(self.stdout, str) or not _is_exact_type(self.stderr, str):
            raise ValueError("CommandExecution stdout/stderr must be text")
        if type(self.duration_ms) is not int or self.duration_ms < 0:
            raise ValueError("CommandExecution.duration_ms must be a non-negative integer")
        if self.consumed_input_identity_sha256 is not None:
            _require_sha256(
                "CommandExecution.consumed_input_identity_sha256",
                self.consumed_input_identity_sha256,
            )


CommandRunner = Callable[
    [tuple[str, ...], Path, int, bytes, str],
    CommandExecution,
]
GitProbe = Callable[[str | Path], GitState]


def _require_exact_clean_git_state_with_probe(
    repo_root: str | Path,
    expected_git_sha: str,
    *,
    git_probe: GitProbe,
) -> GitState:
    """Internal test harness for the canonical exact-git-state authority."""

    expected = _require_git_sha("expected_git_sha", expected_git_sha)
    state = git_probe(repo_root)
    if not _is_exact_type(state, GitState):
        raise ValueError("git probe must return GitState")
    git_state_validator(state)
    if state.sha != expected:
        raise ValueError(
            f"exact-head mismatch: expected {expected}, observed {state.sha}"
        )
    if not state.tracked_clean:
        raise ValueError("SIL checkout is dirty")
    return state


def _build_exact_git_state_authority():
    # Capture the real probe and validator in a closure.  Later module-global
    # rebinding cannot replace the canonical production authority, and callers
    # cannot supply their own probe through the public function signature.
    sealed_probe = probe_git_state
    sealed_impl = _require_exact_clean_git_state_with_probe

    def canonical(
        repo_root: str | Path,
        expected_git_sha: str,
    ) -> GitState:
        return sealed_impl(
            repo_root,
            expected_git_sha,
            git_probe=sealed_probe,
        )

    return canonical


require_exact_clean_git_state = _build_exact_git_state_authority()


def run_command(
    argv: tuple[str, ...],
    cwd: Path,
    timeout_seconds: int,
    input_envelope_bytes: bytes,
    expected_input_identity_sha256: str,
) -> CommandExecution:
    expected_identity = _require_sha256(
        "expected_input_identity_sha256",
        expected_input_identity_sha256,
    )
    if not _is_exact_type(input_envelope_bytes, bytes) or not input_envelope_bytes:
        raise ValueError("input_envelope_bytes must be non-empty bytes")
    actual_identity = _sha256_bytes(input_envelope_bytes)
    if actual_identity != expected_identity:
        raise ValueError("SIL input envelope identity mismatch before execution")
    try:
        envelope_text = input_envelope_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("SIL input envelope must be strict UTF-8") from exc

    wrapped_argv = (
        sys.executable,
        "-c",
        _EXECUTION_WRAPPER,
        expected_identity,
        envelope_text,
        *argv,
    )
    started = time.monotonic_ns()
    try:
        result = subprocess.run(
            list(wrapped_argv),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=_qualification_subprocess_env(),
            stdin=subprocess.DEVNULL,
            shell=False,
        )
        return_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr += f"\nSIL_TIMEOUT_AFTER_SECONDS={timeout_seconds}\n"

    verification_line = f"{_INPUT_VERIFICATION_PREFIX}{expected_identity}\n"
    consumed_identity = expected_identity if stdout.startswith(verification_line) else None
    duration_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
    return CommandExecution(
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=duration_ms,
        consumed_input_identity_sha256=consumed_identity,
    )


def _synthetic_model_identities() -> tuple[str, str]:
    model = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        rope_rotary_dim=16,
    )
    init = InitSpec()
    return model.identity_sha256(), init.identity_sha256()


def _qualify_sil_with_backends(
    *,
    repo_root: str | Path,
    expected_git_sha: str,
    registry: CapabilityRegistry,
    scenario: SILScenario,
    package_bytes: bytes,
    environment_receipt: dict[str, Any],
    command_runner: CommandRunner,
    git_probe: GitProbe,
    package_manifest_builder: Callable[[str | Path], bytes] = build_package_manifest_bytes,
    plan_builder: Callable[[CapabilityRegistry, SILScenario], SILPlan] = build_sil_plan,
    registry_identity_builder: Callable[[CapabilityRegistry], str] = (
        CapabilityRegistry.identity_sha256
    ),
    scenario_identity_builder: Callable[[SILScenario], str] = (
        _sil_scenario_identity_from_stored_state
    ),
    plan_payload_builder: Callable[[SILPlan], dict[str, Any]] = (
        _sil_plan_payload_from_stored_state
    ),
    git_state_validator: Callable[[GitState], None] = GitState.__post_init__,
    command_execution_validator: Callable[[CommandExecution], None] = (
        CommandExecution.__post_init__
    ),
) -> tuple[dict[str, Any], str]:
    """Internal deterministic harness; not a canonical evidence authority."""
    expected_git_sha = _require_git_sha("expected_git_sha", expected_git_sha)
    if not _is_exact_type(package_bytes, bytes) or not package_bytes:
        raise ValueError("package_bytes must be non-empty bytes")
    environment_receipt = _validate_sil_environment_receipt(environment_receipt)

    root = Path(repo_root)
    expected_package_bytes = package_manifest_builder(root)
    if package_bytes != expected_package_bytes:
        raise ValueError(
            "package_bytes do not match exact tracked package source manifest"
        )

    state = git_probe(root)
    if not _is_exact_type(state, GitState):
        raise ValueError("git probe must return exact GitState")
    GitState.__post_init__(state)
    if state.sha != expected_git_sha:
        raise ValueError(
            f"exact-head mismatch: expected {expected_git_sha}, observed {state.sha}"
        )
    if not state.tracked_clean:
        raise ValueError("tracked checkout is dirty before SIL execution")

    plan = plan_builder(registry, scenario)
    package_identity = _sha256_bytes(package_bytes)
    environment_identity = environment_receipt["identity_sha256"]
    registry_identity = registry_identity_builder(registry)
    model_identity, init_identity = _synthetic_model_identities()
    data_identity = _sha256_bytes(scenario.synthetic_data_utf8.encode("utf-8"))
    scenario_identity = scenario_identity_builder(scenario)
    plan_payload = plan_payload_builder(plan)

    input_envelope = {
        "schema_version": "12-6.github-sil-input.v1",
        "git_sha": expected_git_sha,
        "package_identity_sha256": package_identity,
        "environment_identity_sha256": environment_identity,
        "capability_registry_identity_sha256": registry_identity,
        "model_spec_identity_sha256": model_identity,
        "init_spec_identity_sha256": init_identity,
        "data_identity_sha256": data_identity,
        "scenario_identity_sha256": scenario_identity,
        "synthetic_data_utf8": scenario.synthetic_data_utf8,
        "plan": plan_payload,
    }
    input_envelope_bytes = _canonical_json_bytes(input_envelope)
    input_identity = _sha256_bytes(input_envelope_bytes)

    started_unix_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    executions: list[dict[str, Any]] = []
    log_records: list[dict[str, Any]] = []

    for vector in plan.vectors:
        pre_vector_state = git_probe(root)
        if not _is_exact_type(pre_vector_state, GitState):
            raise ValueError("git probe must return exact GitState")
        git_state_validator(pre_vector_state)
        if pre_vector_state.sha != expected_git_sha:
            raise ValueError(
                "exact-head changed before SIL vector execution: "
                f"expected {expected_git_sha}, observed {pre_vector_state.sha}"
            )
        if not pre_vector_state.tracked_clean:
            raise ValueError("tracked checkout is dirty before SIL vector execution")

        result = command_runner(
            vector.argv,
            root,
            scenario.timeout_seconds_per_vector,
            input_envelope_bytes,
            input_identity,
        )
        if not _is_exact_type(result, CommandExecution):
            raise ValueError("command runner must return exact CommandExecution")
        command_execution_validator(result)

        post_vector_state = git_probe(root)
        if not _is_exact_type(post_vector_state, GitState):
            raise ValueError("git probe must return exact GitState")
        git_state_validator(post_vector_state)
        if post_vector_state.sha != expected_git_sha:
            raise ValueError(
                "exact-head changed during SIL vector execution: "
                f"expected {expected_git_sha}, observed {post_vector_state.sha}"
            )
        if not post_vector_state.tracked_clean:
            raise ValueError("tracked checkout became dirty during SIL vector execution")

        execution = {
            "journey_id": vector.journey_id,
            "capability_id": vector.capability_id,
            "vector_id": vector.vector_id,
            "argv": list(vector.argv),
            "return_code": result.return_code,
            "expected_input_identity_sha256": input_identity,
            "consumed_input_identity_sha256": result.consumed_input_identity_sha256,
            "stdout_sha256": _sha256_bytes(result.stdout.encode("utf-8")),
            "stderr_sha256": _sha256_bytes(result.stderr.encode("utf-8")),
            "duration_ms": result.duration_ms,
        }
        executions.append(execution)
        log_records.append(
            {
                "journey_id": vector.journey_id,
                "capability_id": vector.capability_id,
                "vector_id": vector.vector_id,
                "argv": list(vector.argv),
                "return_code": result.return_code,
                "expected_input_identity_sha256": input_identity,
                "consumed_input_identity_sha256": result.consumed_input_identity_sha256,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_ms": result.duration_ms,
            }
        )

    final_state = git_probe(root)
    if not _is_exact_type(final_state, GitState):
        raise ValueError("git probe must return exact GitState")
    git_state_validator(final_state)
    if final_state.sha != expected_git_sha:
        raise ValueError(
            "exact-head changed before SIL evidence sealing: "
            f"expected {expected_git_sha}, observed {final_state.sha}"
        )
    if not final_state.tracked_clean:
        raise ValueError("tracked checkout is dirty before SIL evidence sealing")

    finished_monotonic_ns = time.monotonic_ns()
    finished_unix_ns = time.time_ns()
    log_text = "".join(
        _canonical_json_bytes(record).decode("utf-8") + "\n"
        for record in log_records
    )
    log_identity = _sha256_bytes(log_text.encode("utf-8"))
    unavailable_payload = plan_payload["unavailable_journeys"]
    output_identity = _canonical_sha256(
        {
            "available_journey_ids": list(plan.available_journey_ids),
            "unavailable_journeys": unavailable_payload,
            "executions": executions,
        }
    )
    verdict = (
        "PASS"
        if executions
        and all(
            item["return_code"] == 0
            and item["expected_input_identity_sha256"] == input_identity
            and item["consumed_input_identity_sha256"] == input_identity
            for item in executions
        )
        else "FAIL"
    )

    evidence: dict[str, Any] = {
        "schema_version": "12-6.github-sil-evidence.v1",
        "git_sha": expected_git_sha,
        "package_identity_sha256": package_identity,
        "environment_identity_sha256": environment_identity,
        "capability_registry_identity_sha256": registry_identity,
        "model_spec_identity_sha256": model_identity,
        "init_spec_identity_sha256": init_identity,
        "data_identity_sha256": data_identity,
        "scenario_id": scenario.scenario_id,
        "scenario_identity_sha256": scenario_identity,
        "fixture_policy": scenario.fixture_policy,
        "input_identity_sha256": input_identity,
        "available_journey_ids": list(plan.available_journey_ids),
        "unavailable_journeys": unavailable_payload,
        "executions": executions,
        "output_identity_sha256": output_identity,
        "log_sha256": log_identity,
        "timings": {
            "started_unix_ns": started_unix_ns,
            "finished_unix_ns": finished_unix_ns,
            "duration_ms": max(
                0,
                (finished_monotonic_ns - started_monotonic_ns) // 1_000_000,
            ),
        },
        "verdict": verdict,
        "scientific_boundary": {
            "corpus_admission_authorized": False,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
        },
    }
    evidence["evidence_identity_sha256"] = _canonical_sha256(evidence)
    return evidence, log_text


def _build_qualify_sil_authority():
    # Production SIL evidence must always use the repository-owned execution
    # and Git probes.  Capture all three callables so neither caller-supplied
    # callbacks nor later module-global rebinding can fabricate a canonical PASS.
    sealed_impl = _qualify_sil_with_backends
    sealed_runner = run_command
    sealed_probe = probe_git_state
    sealed_package_manifest_builder = build_package_manifest_bytes
    sealed_plan_builder = build_sil_plan
    sealed_registry_identity_builder = CapabilityRegistry.identity_sha256
    sealed_scenario_identity_builder = _sil_scenario_identity_from_stored_state
    sealed_plan_payload_builder = _sil_plan_payload_from_stored_state
    sealed_git_state_validator = GitState.__post_init__
    sealed_command_execution_validator = CommandExecution.__post_init__

    def canonical(
        *,
        repo_root: str | Path,
        expected_git_sha: str,
        registry: CapabilityRegistry,
        scenario: SILScenario,
        package_bytes: bytes,
        environment_receipt: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        return sealed_impl(
            repo_root=repo_root,
            expected_git_sha=expected_git_sha,
            registry=registry,
            scenario=scenario,
            package_bytes=package_bytes,
            environment_receipt=environment_receipt,
            command_runner=sealed_runner,
            git_probe=sealed_probe,
            package_manifest_builder=sealed_package_manifest_builder,
            plan_builder=sealed_plan_builder,
            registry_identity_builder=sealed_registry_identity_builder,
            scenario_identity_builder=sealed_scenario_identity_builder,
            plan_payload_builder=sealed_plan_payload_builder,
            git_state_validator=sealed_git_state_validator,
            command_execution_validator=sealed_command_execution_validator,
        )

    return canonical


qualify_sil = _build_qualify_sil_authority()


_EVIDENCE_FIELDS = {
    "schema_version",
    "git_sha",
    "package_identity_sha256",
    "environment_identity_sha256",
    "capability_registry_identity_sha256",
    "model_spec_identity_sha256",
    "init_spec_identity_sha256",
    "data_identity_sha256",
    "scenario_id",
    "scenario_identity_sha256",
    "fixture_policy",
    "input_identity_sha256",
    "available_journey_ids",
    "unavailable_journeys",
    "executions",
    "output_identity_sha256",
    "log_sha256",
    "timings",
    "verdict",
    "scientific_boundary",
    "evidence_identity_sha256",
}


_MAX_LOG_BYTES = 16 * 1024 * 1024
_LOG_RECORD_FIELDS = {
    "journey_id",
    "capability_id",
    "vector_id",
    "argv",
    "return_code",
    "expected_input_identity_sha256",
    "consumed_input_identity_sha256",
    "stdout",
    "stderr",
    "duration_ms",
}


def _load_sil_log_records(log_bytes: bytes) -> list[dict[str, Any]]:
    if not _is_exact_type(log_bytes, bytes) or not log_bytes:
        raise ValueError("SIL log must be non-empty bytes")
    if len(log_bytes) > _MAX_LOG_BYTES:
        raise ValueError("SIL log exceeds maximum encoded size")
    if not log_bytes.endswith(b"\n"):
        raise ValueError("SIL log must end with a canonical newline")

    records: list[dict[str, Any]] = []
    for index, line in enumerate(log_bytes.splitlines(), start=1):
        if not line:
            raise ValueError("SIL log contains an empty record")
        record = _strict_json_object(
            line,
            maximum_bytes=_MAX_LOG_BYTES,
            label=f"SIL log record {index}",
        )
        if set(record) != _LOG_RECORD_FIELDS:
            raise ValueError("SIL log record fields are non-canonical")
        records.append(record)
    if not records:
        raise ValueError("SIL log contains no execution records")
    return records


def _verify_sil_evidence_with_authorities(
    evidence_path: str | Path,
    log_path: str | Path,
    *,
    expected_package_bytes: bytes,
    expected_environment_receipt: dict[str, Any],
    expected_registry: CapabilityRegistry,
    expected_scenario: SILScenario,
    expected_git_sha: str,
    require_pass: bool = True,
    plan_builder: Callable[[CapabilityRegistry, SILScenario], SILPlan],
    registry_identity_builder: Callable[[CapabilityRegistry], str],
    scenario_identity_builder: Callable[[SILScenario], str],
    plan_payload_builder: Callable[[SILPlan], dict[str, Any]],
) -> dict[str, Any]:
    payload = _strict_json_object(
        Path(evidence_path).read_bytes(),
        maximum_bytes=_MAX_EVIDENCE_BYTES,
        label="SIL evidence",
    )
    if set(payload) != _EVIDENCE_FIELDS:
        raise ValueError("SIL evidence fields are non-canonical")
    if payload["schema_version"] != "12-6.github-sil-evidence.v1":
        raise ValueError("unsupported SIL evidence schema_version")
    _require_git_sha("evidence git_sha", payload["git_sha"])
    if payload["git_sha"] != _require_git_sha("expected_git_sha", expected_git_sha):
        raise ValueError("SIL evidence Git SHA mismatch")
    for field in (
        "package_identity_sha256",
        "environment_identity_sha256",
        "capability_registry_identity_sha256",
        "model_spec_identity_sha256",
        "init_spec_identity_sha256",
        "data_identity_sha256",
        "scenario_identity_sha256",
        "input_identity_sha256",
        "output_identity_sha256",
        "log_sha256",
        "evidence_identity_sha256",
    ):
        _require_sha256(field, payload[field])

    if not _is_exact_type(expected_package_bytes, bytes) or not expected_package_bytes:
        raise ValueError("expected_package_bytes must be non-empty bytes")
    expected_environment_receipt = _validate_sil_environment_receipt(
        expected_environment_receipt
    )
    if not _is_exact_type(expected_registry, CapabilityRegistry):
        raise ValueError("expected_registry must be a CapabilityRegistry")
    if not _is_exact_type(expected_scenario, SILScenario):
        raise ValueError("expected_scenario must be a SILScenario")

    expected_model_identity, expected_init_identity = _synthetic_model_identities()
    expected_identities = {
        "package_identity_sha256": _sha256_bytes(expected_package_bytes),
        "environment_identity_sha256": expected_environment_receipt[
            "identity_sha256"
        ],
        "capability_registry_identity_sha256": registry_identity_builder(
            expected_registry
        ),
        "model_spec_identity_sha256": expected_model_identity,
        "init_spec_identity_sha256": expected_init_identity,
        "data_identity_sha256": _sha256_bytes(
            expected_scenario.synthetic_data_utf8.encode("utf-8")
        ),
        "scenario_identity_sha256": scenario_identity_builder(expected_scenario),
    }
    for field, expected_value in expected_identities.items():
        if payload[field] != expected_value:
            raise ValueError(
                f"SIL evidence {field} does not match exact verifier authority"
            )

    if payload["scenario_id"] != expected_scenario.scenario_id:
        raise ValueError("SIL evidence scenario_id does not match exact scenario")
    if payload["fixture_policy"] != expected_scenario.fixture_policy:
        raise ValueError("SIL evidence fixture policy does not match exact scenario")

    available_journey_ids = payload["available_journey_ids"]
    unavailable_journeys = payload["unavailable_journeys"]
    executions = payload["executions"]
    if not _is_exact_type(available_journey_ids, list) or not available_journey_ids:
        raise ValueError("SIL evidence needs available journeys")
    if not _is_exact_type(unavailable_journeys, list):
        raise ValueError("SIL evidence unavailable_journeys must be an array")
    if not _is_exact_type(executions, list) or not executions:
        raise ValueError("SIL evidence needs executed integration vectors")

    expected_plan = plan_builder(expected_registry, expected_scenario)
    expected_plan_payload = plan_payload_builder(expected_plan)
    expected_available = list(expected_plan.available_journey_ids)
    expected_unavailable = expected_plan_payload["unavailable_journeys"]
    if available_journey_ids != expected_available:
        raise ValueError("SIL evidence available journeys do not match exact registry")
    if unavailable_journeys != expected_unavailable:
        raise ValueError("SIL evidence unavailable journeys do not match exact registry")
    if len(executions) != len(expected_plan.vectors):
        raise ValueError("SIL evidence execution count does not match exact plan")

    execution_fields = {
        "journey_id",
        "capability_id",
        "vector_id",
        "argv",
        "return_code",
        "expected_input_identity_sha256",
        "consumed_input_identity_sha256",
        "stdout_sha256",
        "stderr_sha256",
        "duration_ms",
    }
    for execution, planned in zip(executions, expected_plan.vectors, strict=True):
        if type(execution) is not dict or set(execution) != execution_fields:
            raise ValueError("SIL execution record schema is non-canonical")
        if (
            execution["journey_id"] != planned.journey_id
            or execution["capability_id"] != planned.capability_id
            or execution["vector_id"] != planned.vector_id
            or execution["argv"] != list(planned.argv)
        ):
            raise ValueError("SIL execution record does not match exact plan")
        if type(execution["return_code"]) is not int:
            raise ValueError("SIL execution return_code must be an integer")
        if type(execution["duration_ms"]) is not int or execution["duration_ms"] < 0:
            raise ValueError("SIL execution duration_ms must be a non-negative integer")
        if execution["expected_input_identity_sha256"] != payload["input_identity_sha256"]:
            raise ValueError("SIL execution expected input identity mismatch")
        consumed_identity = execution["consumed_input_identity_sha256"]
        if consumed_identity is not None:
            _require_sha256("consumed_input_identity_sha256", consumed_identity)
        _require_sha256("stdout_sha256", execution["stdout_sha256"])
        _require_sha256("stderr_sha256", execution["stderr_sha256"])

    expected_input_envelope = {
        "schema_version": "12-6.github-sil-input.v1",
        "git_sha": payload["git_sha"],
        "package_identity_sha256": expected_identities["package_identity_sha256"],
        "environment_identity_sha256": expected_identities[
            "environment_identity_sha256"
        ],
        "capability_registry_identity_sha256": expected_identities[
            "capability_registry_identity_sha256"
        ],
        "model_spec_identity_sha256": expected_model_identity,
        "init_spec_identity_sha256": expected_init_identity,
        "data_identity_sha256": expected_identities["data_identity_sha256"],
        "scenario_identity_sha256": expected_identities["scenario_identity_sha256"],
        "synthetic_data_utf8": expected_scenario.synthetic_data_utf8,
        "plan": expected_plan_payload,
    }
    expected_input_identity = _sha256_bytes(
        _canonical_json_bytes(expected_input_envelope)
    )
    if payload["input_identity_sha256"] != expected_input_identity:
        raise ValueError("SIL input identity does not match exact verifier authority")

    timings = payload["timings"]
    if type(timings) is not dict or set(timings) != {
        "started_unix_ns",
        "finished_unix_ns",
        "duration_ms",
    }:
        raise ValueError("SIL evidence timings schema is non-canonical")
    for field in ("started_unix_ns", "finished_unix_ns", "duration_ms"):
        if type(timings[field]) is not int or timings[field] < 0:
            raise ValueError(f"SIL evidence timing {field} must be a non-negative integer")
    if timings["finished_unix_ns"] < timings["started_unix_ns"]:
        raise ValueError("SIL evidence wall-clock timings are inverted")
    execution_duration_sum = sum(
        execution["duration_ms"] for execution in executions
    )
    if timings["duration_ms"] < execution_duration_sum:
        raise ValueError(
            "SIL evidence total duration is shorter than sequential execution durations"
        )

    expected_verdict = (
        "PASS"
        if all(
            execution["return_code"] == 0
            and execution["expected_input_identity_sha256"]
            == payload["input_identity_sha256"]
            and execution["consumed_input_identity_sha256"]
            == payload["input_identity_sha256"]
            for execution in executions
        )
        else "FAIL"
    )
    if payload["verdict"] != expected_verdict:
        raise ValueError("SIL evidence verdict does not match execution results")
    if require_pass and payload["verdict"] != "PASS":
        raise ValueError("SIL evidence is not PASS")

    scientific_boundary = payload["scientific_boundary"]
    expected_boundary = {
        "corpus_admission_authorized": False,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights_used": False,
    }
    if scientific_boundary != expected_boundary:
        raise ValueError("SIL evidence widened the scientific boundary")

    log_bytes = Path(log_path).read_bytes()
    if _sha256_bytes(log_bytes) != payload["log_sha256"]:
        raise ValueError("SIL log identity mismatch")
    log_records = _load_sil_log_records(log_bytes)
    if len(log_records) != len(executions):
        raise ValueError("SIL log execution count does not match evidence")
    for log_record, execution in zip(log_records, executions, strict=True):
        for field in (
            "journey_id",
            "capability_id",
            "vector_id",
            "argv",
            "return_code",
            "expected_input_identity_sha256",
            "consumed_input_identity_sha256",
            "duration_ms",
        ):
            if log_record[field] != execution[field]:
                raise ValueError(f"SIL log {field} does not match evidence")
        if not _is_exact_type(log_record["stdout"], str) or not _is_exact_type(
            log_record["stderr"], str
        ):
            raise ValueError("SIL log stdout/stderr must be text")
        if _sha256_bytes(log_record["stdout"].encode("utf-8")) != execution[
            "stdout_sha256"
        ]:
            raise ValueError("SIL log stdout identity does not match evidence")
        if _sha256_bytes(log_record["stderr"].encode("utf-8")) != execution[
            "stderr_sha256"
        ]:
            raise ValueError("SIL log stderr identity does not match evidence")

    expected_output = _canonical_sha256(
        {
            "available_journey_ids": payload["available_journey_ids"],
            "unavailable_journeys": payload["unavailable_journeys"],
            "executions": executions,
        }
    )
    if expected_output != payload["output_identity_sha256"]:
        raise ValueError("SIL output identity mismatch")

    evidence_identity = payload["evidence_identity_sha256"]
    unsigned = dict(payload)
    del unsigned["evidence_identity_sha256"]
    if _canonical_sha256(unsigned) != evidence_identity:
        raise ValueError("SIL evidence identity mismatch")
    return payload


def _build_verify_sil_evidence_authority():
    sealed_impl = _verify_sil_evidence_with_authorities
    sealed_plan_builder = build_sil_plan
    sealed_registry_identity_builder = CapabilityRegistry.identity_sha256
    sealed_scenario_identity_builder = _sil_scenario_identity_from_stored_state
    sealed_plan_payload_builder = _sil_plan_payload_from_stored_state

    def canonical(
        evidence_path: str | Path,
        log_path: str | Path,
        *,
        expected_package_bytes: bytes,
        expected_environment_receipt: dict[str, Any],
        expected_registry: CapabilityRegistry,
        expected_scenario: SILScenario,
        expected_git_sha: str,
        require_pass: bool = True,
    ) -> dict[str, Any]:
        return sealed_impl(
            evidence_path,
            log_path,
            expected_package_bytes=expected_package_bytes,
            expected_environment_receipt=expected_environment_receipt,
            expected_registry=expected_registry,
            expected_scenario=expected_scenario,
            expected_git_sha=expected_git_sha,
            require_pass=require_pass,
            plan_builder=sealed_plan_builder,
            registry_identity_builder=sealed_registry_identity_builder,
            scenario_identity_builder=sealed_scenario_identity_builder,
            plan_payload_builder=sealed_plan_payload_builder,
        )

    return canonical


verify_sil_evidence = _build_verify_sil_evidence_authority()


def _write_run_outputs(
    evidence: dict[str, Any],
    log_text: str,
    *,
    evidence_output: str | Path,
    log_output: str | Path,
) -> None:
    evidence_path = Path(evidence_output)
    log_path = Path(log_output)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(_canonical_json_bytes(evidence) + b"\n")
    log_path.write_text(log_text, encoding="utf-8")


def _environment_receipt_cli(args: argparse.Namespace) -> int:
    receipt = canonical_sil_environment_receipt_v1()
    require_current_sil_environment(receipt)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_json_bytes(receipt) + b"\n")
    print(
        json.dumps(
            {
                "identity_sha256": receipt["identity_sha256"],
                "lock_source_commit": receipt["lock_source_commit"],
                "package_count": len(receipt["packages"]),
            },
            sort_keys=True,
        )
    )
    return 0


def _run_cli(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve()
    environment_receipt = load_sil_environment_receipt(args.environment_receipt)
    require_current_sil_environment(environment_receipt)
    registry = load_capability_registry(args.capability_registry)
    scenario = load_sil_scenario(args.scenario)
    package_bytes = build_package_manifest_bytes(root)
    evidence, log_text = qualify_sil(
        repo_root=root,
        expected_git_sha=args.expected_git_sha,
        registry=registry,
        scenario=scenario,
        package_bytes=package_bytes,
        environment_receipt=environment_receipt,
    )
    _write_run_outputs(
        evidence,
        log_text,
        evidence_output=args.evidence_output,
        log_output=args.log_output,
    )
    print(
        json.dumps(
            {
                "git_sha": evidence["git_sha"],
                "scenario_id": evidence["scenario_id"],
                "available_journeys": len(evidence["available_journey_ids"]),
                "executions": len(evidence["executions"]),
                "verdict": evidence["verdict"],
                "evidence_identity_sha256": evidence["evidence_identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if evidence["verdict"] == "PASS" else 1


def _verify_cli(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve()
    require_exact_clean_git_state(root, args.expected_git_sha)
    environment_receipt = load_sil_environment_receipt(args.environment_receipt)
    require_current_sil_environment(environment_receipt)
    registry = load_capability_registry(args.capability_registry)
    scenario = load_sil_scenario(args.scenario)
    package_bytes = build_package_manifest_bytes(root)
    evidence = verify_sil_evidence(
        args.evidence,
        args.log,
        expected_package_bytes=package_bytes,
        expected_environment_receipt=environment_receipt,
        expected_registry=registry,
        expected_scenario=scenario,
        expected_git_sha=args.expected_git_sha,
        require_pass=True,
    )
    require_exact_clean_git_state(root, args.expected_git_sha)
    print(
        json.dumps(
            {
                "git_sha": evidence["git_sha"],
                "verdict": evidence["verdict"],
                "evidence_identity_sha256": evidence["evidence_identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or verify the 12-6 GitHub Software-in-the-Loop qualification."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    environment_parser = subparsers.add_parser("environment-receipt")
    environment_parser.add_argument("--output", required=True)
    environment_parser.set_defaults(func=_environment_receipt_cli)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repo-root", required=True)
    run_parser.add_argument("--expected-git-sha", required=True)
    run_parser.add_argument("--capability-registry", required=True)
    run_parser.add_argument("--scenario", required=True)
    run_parser.add_argument("--environment-receipt", required=True)
    run_parser.add_argument("--evidence-output", required=True)
    run_parser.add_argument("--log-output", required=True)
    run_parser.set_defaults(func=_run_cli)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--repo-root", required=True)
    verify_parser.add_argument("--capability-registry", required=True)
    verify_parser.add_argument("--scenario", required=True)
    verify_parser.add_argument("--environment-receipt", required=True)
    verify_parser.add_argument("--evidence", required=True)
    verify_parser.add_argument("--log", required=True)
    verify_parser.add_argument("--expected-git-sha", required=True)
    verify_parser.set_defaults(func=_verify_cli)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
