from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

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
    ("researcher-training-mechanics", ("trainer-preflight",)),
    ("data-curator-governance", ("data-clean-dedup-decontam",)),
    ("operator-learned20m-readiness", ("learned20m-lease-control",)),
    ("operator-portable-run", ("portable-run-binding",)),
    ("operator-scale141-recovery", ("scale141-content-addressed",)),
    ("researcher-split-validation", ("split-robustness-manifest",)),
    ("maintainer-project-control", ("swarm-protocol",)),
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
        check=False,
        capture_output=True,
        text=True,
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
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _require_git_sha(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA40_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase 40-hex Git SHA")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object member")
        value[key] = item
    return value


def _strict_json_object(data: bytes, *, maximum_bytes: int, label: str) -> dict[str, Any]:
    if not isinstance(data, bytes):
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
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return value


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
        if not isinstance(self.scenario_id, str) or _ID_RE.fullmatch(self.scenario_id) is None:
            raise ValueError("scenario_id must be a canonical identifier")
        if self.journey_selector != "ALL_AVAILABLE":
            raise ValueError("journey_selector must be ALL_AVAILABLE")
        if self.fixture_policy != "DETERMINISTIC_SYNTHETIC":
            raise ValueError("fixture_policy must be DETERMINISTIC_SYNTHETIC")
        if not isinstance(self.synthetic_data_utf8, str) or not self.synthetic_data_utf8:
            raise ValueError("synthetic_data_utf8 must be non-empty text")
        if len(self.synthetic_data_utf8.encode("utf-8")) > 64 * 1024:
            raise ValueError("synthetic_data_utf8 is too large")
        if (
            type(self.timeout_seconds_per_vector) is not int
            or not 1 <= self.timeout_seconds_per_vector <= 900
        ):
            raise ValueError("timeout_seconds_per_vector must be an integer in [1, 900]")

    def to_dict(self) -> dict[str, Any]:
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

    def to_dict(self) -> dict[str, Any]:
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

    def to_dict(self) -> dict[str, Any]:
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
        if not self.available_journey_ids:
            raise ValueError("SIL plan needs at least one AVAILABLE journey")
        if not self.vectors:
            raise ValueError("SIL plan needs at least one integration vector")
        if (
            not isinstance(self.journey_end_to_end_contracts, tuple)
            or not self.journey_end_to_end_contracts
        ):
            raise ValueError("SIL plan needs explicit end-to-end journey contracts")
        contract_journey_ids: list[str] = []
        for item in self.journey_end_to_end_contracts:
            if (
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or _ID_RE.fullmatch(item[0]) is None
                or not isinstance(item[1], tuple)
                or not item[1]
                or any(
                    not isinstance(vector_id, str)
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

    def to_dict(self) -> dict[str, Any]:
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


def parse_vector_command(command: str) -> tuple[str, ...]:
    if not isinstance(command, str) or not command.strip():
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


def build_sil_plan(
    registry: CapabilityRegistry,
    scenario: SILScenario,
    _sealed_e2e_policy: tuple[tuple[str, tuple[str, ...]], ...] = (
        _CANONICAL_JOURNEY_E2E_VECTOR_POLICY
    ),
) -> SILPlan:
    if not isinstance(registry, CapabilityRegistry):
        raise ValueError("registry must be a CapabilityRegistry")
    if not isinstance(scenario, SILScenario):
        raise ValueError("scenario must be a SILScenario")
    if scenario.journey_selector != "ALL_AVAILABLE":
        raise ValueError("unsupported journey selection")

    if (
        not isinstance(_sealed_e2e_policy, tuple)
        or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], tuple)
            for item in _sealed_e2e_policy
        )
    ):
        raise ValueError("SIL end-to-end policy is non-canonical")
    policy_journey_ids = tuple(item[0] for item in _sealed_e2e_policy)
    if len(set(policy_journey_ids)) != len(policy_journey_ids):
        raise ValueError("SIL end-to-end policy journey ids must be unique")
    policy_by_journey = dict(_sealed_e2e_policy)

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


@dataclass(frozen=True, slots=True)
class GitState:
    sha: str
    tracked_clean: bool


def probe_git_state(repo_root: str | Path) -> GitState:
    root = Path(repo_root)
    sha_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    sha = sha_result.stdout.strip()
    _require_git_sha("observed git SHA", sha)

    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
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


CommandRunner = Callable[
    [tuple[str, ...], Path, int, bytes, str],
    CommandExecution,
]
GitProbe = Callable[[str | Path], GitState]


def require_exact_clean_git_state(
    repo_root: str | Path,
    expected_git_sha: str,
    *,
    git_probe: GitProbe | None = None,
) -> GitState:
    expected = _require_git_sha("expected_git_sha", expected_git_sha)
    probe = probe_git_state if git_probe is None else git_probe
    state = probe(repo_root)
    if not isinstance(state, GitState):
        raise ValueError("git probe must return GitState")
    if state.sha != expected:
        raise ValueError(
            f"exact-head mismatch: expected {expected}, observed {state.sha}"
        )
    if not state.tracked_clean:
        raise ValueError("SIL checkout is dirty")
    return state


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
    if not isinstance(input_envelope_bytes, bytes) or not input_envelope_bytes:
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
            env=os.environ.copy(),
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


def qualify_sil(
    *,
    repo_root: str | Path,
    expected_git_sha: str,
    registry: CapabilityRegistry,
    scenario: SILScenario,
    package_bytes: bytes,
    command_runner: CommandRunner = run_command,
    git_probe: GitProbe = probe_git_state,
) -> tuple[dict[str, Any], str]:
    expected_git_sha = _require_git_sha("expected_git_sha", expected_git_sha)
    if not isinstance(package_bytes, bytes) or not package_bytes:
        raise ValueError("package_bytes must be non-empty bytes")

    root = Path(repo_root)
    expected_package_bytes = build_package_manifest_bytes(root)
    if package_bytes != expected_package_bytes:
        raise ValueError(
            "package_bytes do not match exact tracked package source manifest"
        )

    state = git_probe(root)
    if state.sha != expected_git_sha:
        raise ValueError(
            f"exact-head mismatch: expected {expected_git_sha}, observed {state.sha}"
        )
    if not state.tracked_clean:
        raise ValueError("tracked checkout is dirty before SIL execution")

    plan = build_sil_plan(registry, scenario)
    package_identity = _sha256_bytes(package_bytes)
    registry_identity = registry.identity_sha256()
    model_identity, init_identity = _synthetic_model_identities()
    data_identity = _sha256_bytes(scenario.synthetic_data_utf8.encode("utf-8"))
    scenario_identity = scenario.identity_sha256()

    input_envelope = {
        "schema_version": "12-6.github-sil-input.v1",
        "git_sha": expected_git_sha,
        "package_identity_sha256": package_identity,
        "capability_registry_identity_sha256": registry_identity,
        "model_spec_identity_sha256": model_identity,
        "init_spec_identity_sha256": init_identity,
        "data_identity_sha256": data_identity,
        "scenario_identity_sha256": scenario_identity,
        "synthetic_data_utf8": scenario.synthetic_data_utf8,
        "plan": plan.to_dict(),
    }
    input_envelope_bytes = _canonical_json_bytes(input_envelope)
    input_identity = _sha256_bytes(input_envelope_bytes)

    started_unix_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    executions: list[dict[str, Any]] = []
    log_records: list[dict[str, Any]] = []

    for vector in plan.vectors:
        pre_vector_state = git_probe(root)
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

        post_vector_state = git_probe(root)
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

    finished_monotonic_ns = time.monotonic_ns()
    finished_unix_ns = time.time_ns()
    log_text = "".join(
        _canonical_json_bytes(record).decode("utf-8") + "\n"
        for record in log_records
    )
    log_identity = _sha256_bytes(log_text.encode("utf-8"))
    unavailable_payload = [item.to_dict() for item in plan.unavailable_journeys]
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


_EVIDENCE_FIELDS = {
    "schema_version",
    "git_sha",
    "package_identity_sha256",
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
    if not isinstance(log_bytes, bytes) or not log_bytes:
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


def verify_sil_evidence(
    evidence_path: str | Path,
    log_path: str | Path,
    *,
    expected_package_bytes: bytes,
    expected_registry: CapabilityRegistry,
    expected_scenario: SILScenario,
    expected_git_sha: str | None = None,
    require_pass: bool = True,
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
    if expected_git_sha is not None and payload["git_sha"] != _require_git_sha(
        "expected_git_sha", expected_git_sha
    ):
        raise ValueError("SIL evidence Git SHA mismatch")
    for field in (
        "package_identity_sha256",
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

    if not isinstance(expected_package_bytes, bytes) or not expected_package_bytes:
        raise ValueError("expected_package_bytes must be non-empty bytes")
    if not isinstance(expected_registry, CapabilityRegistry):
        raise ValueError("expected_registry must be a CapabilityRegistry")
    if not isinstance(expected_scenario, SILScenario):
        raise ValueError("expected_scenario must be a SILScenario")

    expected_model_identity, expected_init_identity = _synthetic_model_identities()
    expected_identities = {
        "package_identity_sha256": _sha256_bytes(expected_package_bytes),
        "capability_registry_identity_sha256": expected_registry.identity_sha256(),
        "model_spec_identity_sha256": expected_model_identity,
        "init_spec_identity_sha256": expected_init_identity,
        "data_identity_sha256": _sha256_bytes(
            expected_scenario.synthetic_data_utf8.encode("utf-8")
        ),
        "scenario_identity_sha256": expected_scenario.identity_sha256(),
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
    if not isinstance(available_journey_ids, list) or not available_journey_ids:
        raise ValueError("SIL evidence needs available journeys")
    if not isinstance(unavailable_journeys, list):
        raise ValueError("SIL evidence unavailable_journeys must be an array")
    if not isinstance(executions, list) or not executions:
        raise ValueError("SIL evidence needs executed integration vectors")

    expected_plan = build_sil_plan(expected_registry, expected_scenario)
    expected_available = list(expected_plan.available_journey_ids)
    expected_unavailable = [
        item.to_dict() for item in expected_plan.unavailable_journeys
    ]
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
        "capability_registry_identity_sha256": expected_identities[
            "capability_registry_identity_sha256"
        ],
        "model_spec_identity_sha256": expected_model_identity,
        "init_spec_identity_sha256": expected_init_identity,
        "data_identity_sha256": expected_identities["data_identity_sha256"],
        "scenario_identity_sha256": expected_identities["scenario_identity_sha256"],
        "synthetic_data_utf8": expected_scenario.synthetic_data_utf8,
        "plan": expected_plan.to_dict(),
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
        if not isinstance(log_record["stdout"], str) or not isinstance(
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


def _run_cli(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve()
    registry = load_capability_registry(args.capability_registry)
    scenario = load_sil_scenario(args.scenario)
    package_bytes = build_package_manifest_bytes(root)
    evidence, log_text = qualify_sil(
        repo_root=root,
        expected_git_sha=args.expected_git_sha,
        registry=registry,
        scenario=scenario,
        package_bytes=package_bytes,
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
    registry = load_capability_registry(args.capability_registry)
    scenario = load_sil_scenario(args.scenario)
    package_bytes = build_package_manifest_bytes(root)
    evidence = verify_sil_evidence(
        args.evidence,
        args.log,
        expected_package_bytes=package_bytes,
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

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repo-root", required=True)
    run_parser.add_argument("--expected-git-sha", required=True)
    run_parser.add_argument("--capability-registry", required=True)
    run_parser.add_argument("--scenario", required=True)
    run_parser.add_argument("--evidence-output", required=True)
    run_parser.add_argument("--log-output", required=True)
    run_parser.set_defaults(func=_run_cli)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--repo-root", required=True)
    verify_parser.add_argument("--capability-registry", required=True)
    verify_parser.add_argument("--scenario", required=True)
    verify_parser.add_argument("--evidence", required=True)
    verify_parser.add_argument("--log", required=True)
    verify_parser.add_argument("--expected-git-sha", required=True)
    verify_parser.set_defaults(func=_verify_cli)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
