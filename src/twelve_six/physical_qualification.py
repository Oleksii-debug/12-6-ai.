from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from twelve_six.sil_qualification import GitProbe, GitState, probe_git_state


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_ACTIONS = 32
_MAX_ACTION_TIMEOUT_SECONDS = 900
_MAX_OUTPUT_BYTES = 1024 * 1024
_MAX_ARTIFACTS = 64
_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_MAX_PACKET_LIFETIME_SECONDS = 24 * 60 * 60


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _require_git_sha(name: str, value: object) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase 40-hex Git SHA")
    return value


def _require_id(name: str, value: object) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical identifier")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object member")
        value[key] = item
    return value


def _strict_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds maximum encoded size")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant is not allowed: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} is not strict unambiguous UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return value


def _safe_repo_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty POSIX repository path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} must stay inside the repository")
    return value


class ExecutionMode(str, Enum):
    SIMULATION = "SIMULATION"
    REAL_HOST = "REAL_HOST"


class ResourceKind(str, Enum):
    CPU = "CPU"
    GPU = "GPU"
    RAM = "RAM"
    DISK = "DISK"
    NETWORK = "NETWORK"
    MODEL = "MODEL"
    PROVIDER = "PROVIDER"


class ResourceObservation(str, Enum):
    REAL_PROBED = "REAL_PROBED"
    NOT_PRESENT = "NOT_PRESENT"
    NOT_PROBED = "NOT_PROBED"
    SIMULATED = "SIMULATED"


class ActionVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class QualificationVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SIMULATION_PASS = "SIMULATION_PASS"


@dataclass(frozen=True, slots=True)
class QualificationAction:
    action_id: str
    pytest_targets: tuple[str, ...]
    timeout_seconds: int
    max_output_bytes: int
    required_resources: tuple[ResourceKind, ...]

    def __post_init__(self) -> None:
        _require_id("action_id", self.action_id)
        if not self.pytest_targets or len(self.pytest_targets) > 64:
            raise ValueError("pytest_targets must contain 1..64 targets")
        for target in self.pytest_targets:
            _safe_repo_path(target, label="pytest target")
            if not target.startswith("tests/"):
                raise ValueError("physical qualification pytest targets must live under tests/")
        if (
            type(self.timeout_seconds) is not int
            or self.timeout_seconds <= 0
            or self.timeout_seconds > _MAX_ACTION_TIMEOUT_SECONDS
        ):
            raise ValueError("action timeout exceeds the bounded policy")
        if (
            type(self.max_output_bytes) is not int
            or self.max_output_bytes <= 0
            or self.max_output_bytes > _MAX_OUTPUT_BYTES
        ):
            raise ValueError("action output bound exceeds the bounded policy")
        if len(set(self.required_resources)) != len(self.required_resources):
            raise ValueError("required_resources must be unique")
        if tuple(sorted(self.required_resources, key=lambda item: item.value)) != (
            self.required_resources
        ):
            raise ValueError("required_resources must use canonical lexical order")

    def logical_argv(self) -> tuple[str, ...]:
        return ("python", "-m", "pytest", "-q", *self.pytest_targets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "pytest_targets": list(self.pytest_targets),
            "timeout_seconds": self.timeout_seconds,
            "max_output_bytes": self.max_output_bytes,
            "required_resources": [item.value for item in self.required_resources],
        }


@dataclass(frozen=True, slots=True)
class QualificationPacket:
    schema_version: str
    packet_id: str
    target_git_sha: str
    agent_source_sha256: str
    execution_mode: ExecutionMode
    allowed_os_families: tuple[str, ...]
    not_before_epoch_seconds: int
    expires_epoch_seconds: int
    actions: tuple[QualificationAction, ...]
    artifact_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "12-6.physical-qualification-packet.v1":
            raise ValueError("unsupported physical qualification packet schema")
        _require_id("packet_id", self.packet_id)
        _require_git_sha("target_git_sha", self.target_git_sha)
        _require_sha256("agent_source_sha256", self.agent_source_sha256)
        if not self.allowed_os_families:
            raise ValueError("allowed_os_families cannot be empty")
        if tuple(sorted(set(self.allowed_os_families))) != self.allowed_os_families:
            raise ValueError("allowed_os_families must be canonical unique lexical order")
        if not set(self.allowed_os_families).issubset({"LINUX", "WINDOWS"}):
            raise ValueError("only LINUX/WINDOWS hosts are allowed in Section-5 v1")
        for name, value in (
            ("not_before_epoch_seconds", self.not_before_epoch_seconds),
            ("expires_epoch_seconds", self.expires_epoch_seconds),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        lifetime = self.expires_epoch_seconds - self.not_before_epoch_seconds
        if lifetime <= 0 or lifetime > _MAX_PACKET_LIFETIME_SECONDS:
            raise ValueError("qualification packet lifetime is invalid or unbounded")
        if not self.actions or len(self.actions) > _MAX_ACTIONS:
            raise ValueError("qualification packet must contain 1..32 actions")
        action_ids = [item.action_id for item in self.actions]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("qualification action ids must be unique")
        if action_ids != sorted(action_ids):
            raise ValueError("qualification actions must use canonical action_id order")
        if len(self.artifact_paths) > _MAX_ARTIFACTS:
            raise ValueError("qualification artifact count exceeds policy")
        for item in self.artifact_paths:
            _safe_repo_path(item, label="artifact path")
        if tuple(sorted(set(self.artifact_paths))) != self.artifact_paths:
            raise ValueError("artifact_paths must be canonical unique lexical order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "packet_id": self.packet_id,
            "target_git_sha": self.target_git_sha,
            "agent_source_sha256": self.agent_source_sha256,
            "execution_mode": self.execution_mode.value,
            "allowed_os_families": list(self.allowed_os_families),
            "not_before_epoch_seconds": self.not_before_epoch_seconds,
            "expires_epoch_seconds": self.expires_epoch_seconds,
            "actions": [item.to_dict() for item in self.actions],
            "artifact_paths": list(self.artifact_paths),
        }

    def identity_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class VerifiedSignedPacket:
    packet: QualificationPacket
    signing_key_id: str
    signature_sha256: str
    signed_bundle_identity_sha256: str


SignatureVerifier = Callable[[str, bytes, bytes], bool]


def _action_from_dict(value: object) -> QualificationAction:
    expected = {
        "action_id",
        "pytest_targets",
        "timeout_seconds",
        "max_output_bytes",
        "required_resources",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("qualification action fields are non-canonical")
    targets = value["pytest_targets"]
    resources = value["required_resources"]
    if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
        raise ValueError("pytest_targets must be a string array")
    if not isinstance(resources, list) or not all(isinstance(item, str) for item in resources):
        raise ValueError("required_resources must be a string array")
    return QualificationAction(
        action_id=value["action_id"],
        pytest_targets=tuple(targets),
        timeout_seconds=value["timeout_seconds"],
        max_output_bytes=value["max_output_bytes"],
        required_resources=tuple(ResourceKind(item) for item in resources),
    )


def _packet_from_dict(value: object) -> QualificationPacket:
    expected = {
        "schema_version",
        "packet_id",
        "target_git_sha",
        "agent_source_sha256",
        "execution_mode",
        "allowed_os_families",
        "not_before_epoch_seconds",
        "expires_epoch_seconds",
        "actions",
        "artifact_paths",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("qualification packet fields are non-canonical")
    allowed = value["allowed_os_families"]
    actions = value["actions"]
    artifacts = value["artifact_paths"]
    if not isinstance(allowed, list) or not all(isinstance(item, str) for item in allowed):
        raise ValueError("allowed_os_families must be a string array")
    if not isinstance(actions, list):
        raise ValueError("actions must be an array")
    if not isinstance(artifacts, list) or not all(isinstance(item, str) for item in artifacts):
        raise ValueError("artifact_paths must be a string array")
    return QualificationPacket(
        schema_version=value["schema_version"],
        packet_id=value["packet_id"],
        target_git_sha=value["target_git_sha"],
        agent_source_sha256=value["agent_source_sha256"],
        execution_mode=ExecutionMode(value["execution_mode"]),
        allowed_os_families=tuple(allowed),
        not_before_epoch_seconds=value["not_before_epoch_seconds"],
        expires_epoch_seconds=value["expires_epoch_seconds"],
        actions=tuple(_action_from_dict(item) for item in actions),
        artifact_paths=tuple(artifacts),
    )


def load_verified_signed_packet(
    path: str | Path,
    *,
    signature_verifier: SignatureVerifier,
    now_epoch_seconds: int | None = None,
) -> VerifiedSignedPacket:
    payload = _strict_json_object(path, label="signed physical qualification packet")
    if set(payload) != {"schema_version", "packet", "signature"}:
        raise ValueError("signed packet fields are non-canonical")
    if payload["schema_version"] != "12-6.signed-physical-qualification-packet.v1":
        raise ValueError("unsupported signed packet schema")
    signature = payload["signature"]
    if not isinstance(signature, dict) or set(signature) != {
        "algorithm",
        "key_id",
        "signature_b64",
    }:
        raise ValueError("signed packet signature fields are non-canonical")
    if signature["algorithm"] != "ED25519":
        raise ValueError("Section-5 v1 accepts only ED25519 detached signatures")
    key_id = _require_id("signing key id", signature["key_id"])
    try:
        signature_bytes = base64.b64decode(
            signature["signature_b64"],
            validate=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("signature_b64 is not strict base64") from exc
    if len(signature_bytes) != 64:
        raise ValueError("ED25519 signature must contain exactly 64 bytes")
    packet = _packet_from_dict(payload["packet"])
    message = _canonical_json_bytes(packet.to_dict())
    if not signature_verifier(key_id, message, signature_bytes):
        raise ValueError("qualification packet signature verification failed")
    now = int(time.time()) if now_epoch_seconds is None else now_epoch_seconds
    if type(now) is not int:
        raise ValueError("now_epoch_seconds must be an integer")
    if now < packet.not_before_epoch_seconds or now > packet.expires_epoch_seconds:
        raise ValueError("qualification packet is outside its validity window")
    return VerifiedSignedPacket(
        packet=packet,
        signing_key_id=key_id,
        signature_sha256=_sha256_bytes(signature_bytes),
        signed_bundle_identity_sha256=_sha256_bytes(_canonical_json_bytes(payload)),
    )


@dataclass(frozen=True, slots=True)
class HostInventory:
    os_family: str
    platform_system: str
    platform_release: str
    machine: str
    python_version: str
    cpu_logical_count: int | None
    ram_total_bytes: int | None
    disk_total_bytes: int
    disk_free_bytes: int
    torch_version: str | None
    cuda_available: bool
    cuda_device_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.os_family not in {"WINDOWS", "LINUX", "OTHER"}:
            raise ValueError("invalid host os_family")
        for name, value in (
            ("disk_total_bytes", self.disk_total_bytes),
            ("disk_free_bytes", self.disk_free_bytes),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.cpu_logical_count is not None and (
            type(self.cpu_logical_count) is not int or self.cpu_logical_count <= 0
        ):
            raise ValueError("cpu_logical_count must be positive when present")
        if self.ram_total_bytes is not None and (
            type(self.ram_total_bytes) is not int or self.ram_total_bytes <= 0
        ):
            raise ValueError("ram_total_bytes must be positive when present")
        if type(self.cuda_available) is not bool:
            raise ValueError("cuda_available must be a bool")
        if not self.cuda_available and self.cuda_device_names:
            raise ValueError("CUDA device names require cuda_available=true")

    def to_dict(self) -> dict[str, Any]:
        return {
            "os_family": self.os_family,
            "platform_system": self.platform_system,
            "platform_release": self.platform_release,
            "machine": self.machine,
            "python_version": self.python_version,
            "cpu_logical_count": self.cpu_logical_count,
            "ram_total_bytes": self.ram_total_bytes,
            "disk_total_bytes": self.disk_total_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_device_names": list(self.cuda_device_names),
        }

    def identity_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_dict()))


def _probe_ram_total_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_phys", ctypes.c_ulonglong),
                    ("avail_phys", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("avail_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("avail_virtual", ctypes.c_ulonglong),
                    ("avail_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_phys)
            return None
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        if type(page_size) is int and type(pages) is int and page_size > 0 and pages > 0:
            return page_size * pages
    except (AttributeError, OSError, ValueError):
        return None
    return None


def inventory_host(repo_root: str | Path) -> HostInventory:
    system = platform.system()
    os_family = {"Windows": "WINDOWS", "Linux": "LINUX"}.get(system, "OTHER")
    disk = shutil.disk_usage(Path(repo_root).resolve())
    torch_version: str | None = None
    cuda_available = False
    cuda_names: tuple[str, ...] = ()
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_names = tuple(
                str(torch.cuda.get_device_name(index))
                for index in range(torch.cuda.device_count())
            )
    except (ImportError, RuntimeError):
        pass
    return HostInventory(
        os_family=os_family,
        platform_system=system,
        platform_release=platform.release(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        cpu_logical_count=os.cpu_count(),
        ram_total_bytes=_probe_ram_total_bytes(),
        disk_total_bytes=int(disk.total),
        disk_free_bytes=int(disk.free),
        torch_version=torch_version,
        cuda_available=cuda_available,
        cuda_device_names=cuda_names,
    )


def resource_observations(
    inventory: HostInventory,
    *,
    execution_mode: ExecutionMode,
) -> dict[ResourceKind, ResourceObservation]:
    if execution_mode is ExecutionMode.SIMULATION:
        return {item: ResourceObservation.SIMULATED for item in ResourceKind}
    return {
        ResourceKind.CPU: (
            ResourceObservation.REAL_PROBED
            if inventory.cpu_logical_count is not None
            else ResourceObservation.NOT_PRESENT
        ),
        ResourceKind.GPU: (
            ResourceObservation.REAL_PROBED
            if inventory.cuda_available and inventory.cuda_device_names
            else ResourceObservation.NOT_PRESENT
        ),
        ResourceKind.RAM: (
            ResourceObservation.REAL_PROBED
            if inventory.ram_total_bytes is not None
            else ResourceObservation.NOT_PRESENT
        ),
        ResourceKind.DISK: ResourceObservation.REAL_PROBED,
        ResourceKind.NETWORK: ResourceObservation.NOT_PROBED,
        ResourceKind.MODEL: ResourceObservation.NOT_PROBED,
        ResourceKind.PROVIDER: ResourceObservation.NOT_PROBED,
    }


@dataclass(frozen=True, slots=True)
class ActionExecution:
    return_code: int
    stdout: bytes
    stderr: bytes
    duration_ms: int

    def __post_init__(self) -> None:
        if type(self.return_code) is not int:
            raise ValueError("return_code must be an integer")
        if not isinstance(self.stdout, bytes) or not isinstance(self.stderr, bytes):
            raise ValueError("action output must be bytes")
        if type(self.duration_ms) is not int or self.duration_ms < 0:
            raise ValueError("duration_ms must be a non-negative integer")


ActionRunner = Callable[[QualificationAction, Path], ActionExecution]


def run_bounded_pytest(action: QualificationAction, repo_root: Path) -> ActionExecution:
    argv = (sys.executable, "-m", "pytest", "-q", *action.pytest_targets)
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(
            argv,
            cwd=repo_root,
            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=action.timeout_seconds,
            check=False,
            shell=False,
        )
        return_code = int(completed.returncode)
        stdout = bytes(completed.stdout)
        stderr = bytes(completed.stderr)
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        stdout = bytes(exc.stdout or b"")
        stderr = bytes(exc.stderr or b"")
    duration_ms = (time.monotonic_ns() - started) // 1_000_000
    return ActionExecution(
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=int(duration_ms),
    )


def _collect_artifacts(repo_root: Path, paths: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    collected: list[dict[str, Any]] = []
    total_bytes = 0
    root = repo_root.resolve()
    for relative in paths:
        target = root / relative
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"qualification artifact is not a regular file: {relative}")
        resolved = target.resolve()
        if root not in resolved.parents:
            raise ValueError(f"qualification artifact escapes repository: {relative}")
        raw = target.read_bytes()
        total_bytes += len(raw)
        if total_bytes > _MAX_ARTIFACT_BYTES:
            raise ValueError("qualification artifacts exceed total byte bound")
        collected.append(
            {
                "path": relative,
                "bytes": len(raw),
                "sha256": _sha256_bytes(raw),
            }
        )
    return tuple(collected)


def _bounded_output(raw: bytes, limit: int) -> tuple[bytes, bool]:
    return raw[:limit], len(raw) > limit


def _logical_log_bytes(log_rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) + b"\n" for row in log_rows)


def execute_qualification(
    verified: VerifiedSignedPacket,
    *,
    repo_root: str | Path,
    host_inventory: HostInventory | None = None,
    action_runner: ActionRunner = run_bounded_pytest,
    git_probe: GitProbe = probe_git_state,
    agent_source_bytes: bytes | None = None,
) -> tuple[dict[str, Any], bytes]:
    packet = verified.packet
    root = Path(repo_root).resolve()
    source_bytes = Path(__file__).read_bytes() if agent_source_bytes is None else agent_source_bytes
    source_sha = _sha256_bytes(source_bytes)
    if source_sha != packet.agent_source_sha256:
        raise ValueError("signed packet does not authorize this qualification agent source")

    inventory = inventory_host(root) if host_inventory is None else host_inventory
    resources = resource_observations(inventory, execution_mode=packet.execution_mode)
    reasons: list[str] = []
    if inventory.os_family not in packet.allowed_os_families:
        reasons.append("host OS family is not allowed by the signed packet")

    required = {
        resource
        for action in packet.actions
        for resource in action.required_resources
    }
    if packet.execution_mode is ExecutionMode.REAL_HOST:
        for resource in sorted(required, key=lambda item: item.value):
            if resources[resource] is not ResourceObservation.REAL_PROBED:
                reasons.append(f"required real resource is not proven: {resource.value}")

    initial_state = git_probe(root)
    if initial_state.sha != packet.target_git_sha:
        raise ValueError("physical agent checkout does not match signed target Git SHA")
    if not initial_state.tracked_clean:
        raise ValueError("physical agent checkout is dirty before qualification")

    action_evidence: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    all_actions_pass = True

    for action in packet.actions:
        pre_state = git_probe(root)
        if pre_state.sha != packet.target_git_sha or not pre_state.tracked_clean:
            raise ValueError("checkout changed before physical qualification action")

        execution = action_runner(action, root)
        stdout, stdout_truncated = _bounded_output(execution.stdout, action.max_output_bytes)
        stderr, stderr_truncated = _bounded_output(execution.stderr, action.max_output_bytes)

        post_state = git_probe(root)
        exact_clean = (
            post_state.sha == packet.target_git_sha
            and post_state.tracked_clean
        )
        passed = (
            execution.return_code == 0
            and not stdout_truncated
            and not stderr_truncated
            and exact_clean
        )
        if not passed:
            all_actions_pass = False

        row = {
            "action_id": action.action_id,
            "stdout_b64": base64.b64encode(stdout).decode("ascii"),
            "stderr_b64": base64.b64encode(stderr).decode("ascii"),
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
        log_rows.append(row)
        action_evidence.append(
            {
                "action_id": action.action_id,
                "argv": list(action.logical_argv()),
                "return_code": execution.return_code,
                "duration_ms": execution.duration_ms,
                "stdout_sha256": _sha256_bytes(stdout),
                "stderr_sha256": _sha256_bytes(stderr),
                "stdout_bytes": len(stdout),
                "stderr_bytes": len(stderr),
                "stdout_truncated": stdout_truncated,
                "stderr_truncated": stderr_truncated,
                "pre_git_sha": pre_state.sha,
                "pre_tracked_clean": pre_state.tracked_clean,
                "post_git_sha": post_state.sha,
                "post_tracked_clean": post_state.tracked_clean,
                "verdict": ActionVerdict.PASS.value if passed else ActionVerdict.FAIL.value,
            }
        )

    artifacts = _collect_artifacts(root, packet.artifact_paths)
    log_bytes = _logical_log_bytes(log_rows)

    if reasons or not all_actions_pass:
        verdict = QualificationVerdict.FAIL
    elif packet.execution_mode is ExecutionMode.SIMULATION:
        verdict = QualificationVerdict.SIMULATION_PASS
    else:
        verdict = QualificationVerdict.PASS

    body = {
        "schema_version": "12-6.physical-qualification-evidence.v1",
        "packet_identity_sha256": packet.identity_sha256(),
        "signed_bundle_identity_sha256": verified.signed_bundle_identity_sha256,
        "signing_key_id": verified.signing_key_id,
        "signature_sha256": verified.signature_sha256,
        "agent_source_sha256": source_sha,
        "target_git_sha": packet.target_git_sha,
        "execution_mode": packet.execution_mode.value,
        "host_inventory": inventory.to_dict(),
        "host_inventory_identity_sha256": inventory.identity_sha256(),
        "resource_observations": {
            item.value: resources[item].value for item in sorted(ResourceKind, key=lambda x: x.value)
        },
        "actions": action_evidence,
        "artifacts": list(artifacts),
        "log_sha256": _sha256_bytes(log_bytes),
        "log_bytes": len(log_bytes),
        "verdict": verdict.value,
        "reasons": reasons,
    }
    body["evidence_identity_sha256"] = _sha256_bytes(_canonical_json_bytes(body))
    return body, log_bytes


def write_evidence_bundle(
    evidence_path: str | Path,
    log_path: str | Path,
    *,
    evidence: dict[str, Any],
    log_bytes: bytes,
) -> None:
    expected_identity = evidence.get("evidence_identity_sha256")
    body = dict(evidence)
    body.pop("evidence_identity_sha256", None)
    if expected_identity != _sha256_bytes(_canonical_json_bytes(body)):
        raise ValueError("evidence identity does not match evidence body")
    if evidence.get("log_sha256") != _sha256_bytes(log_bytes):
        raise ValueError("log hash does not match evidence")
    if evidence.get("log_bytes") != len(log_bytes):
        raise ValueError("log byte count does not match evidence")
    Path(evidence_path).write_bytes(_canonical_json_bytes(evidence) + b"\n")
    Path(log_path).write_bytes(log_bytes)


def verify_qualification_evidence(
    evidence_path: str | Path,
    log_path: str | Path,
    *,
    verified_packet: VerifiedSignedPacket,
    agent_source_bytes: bytes,
    artifact_root: str | Path,
    require_real_pass: bool,
) -> dict[str, Any]:
    evidence = _strict_json_object(evidence_path, label="physical qualification evidence")
    expected_fields = {
        "schema_version",
        "packet_identity_sha256",
        "signed_bundle_identity_sha256",
        "signing_key_id",
        "signature_sha256",
        "agent_source_sha256",
        "target_git_sha",
        "execution_mode",
        "host_inventory",
        "host_inventory_identity_sha256",
        "resource_observations",
        "actions",
        "artifacts",
        "log_sha256",
        "log_bytes",
        "verdict",
        "reasons",
        "evidence_identity_sha256",
    }
    if set(evidence) != expected_fields:
        raise ValueError("physical evidence fields are non-canonical")
    if evidence["schema_version"] != "12-6.physical-qualification-evidence.v1":
        raise ValueError("unsupported physical evidence schema")

    claimed_identity = _require_sha256(
        "evidence_identity_sha256",
        evidence["evidence_identity_sha256"],
    )
    identity_body = dict(evidence)
    del identity_body["evidence_identity_sha256"]
    if _sha256_bytes(_canonical_json_bytes(identity_body)) != claimed_identity:
        raise ValueError("physical evidence identity mismatch")

    packet = verified_packet.packet
    if evidence["packet_identity_sha256"] != packet.identity_sha256():
        raise ValueError("physical evidence is bound to a different packet")
    if (
        evidence["signed_bundle_identity_sha256"]
        != verified_packet.signed_bundle_identity_sha256
    ):
        raise ValueError("physical evidence is bound to a different signed packet")
    if evidence["signing_key_id"] != verified_packet.signing_key_id:
        raise ValueError("physical evidence signing key id mismatch")
    if evidence["signature_sha256"] != verified_packet.signature_sha256:
        raise ValueError("physical evidence signature identity mismatch")
    if evidence["target_git_sha"] != packet.target_git_sha:
        raise ValueError("physical evidence target Git SHA mismatch")
    if evidence["agent_source_sha256"] != _sha256_bytes(agent_source_bytes):
        raise ValueError("physical evidence agent source hash mismatch")

    inventory_payload = evidence["host_inventory"]
    if not isinstance(inventory_payload, dict):
        raise ValueError("host_inventory must be an object")
    if (
        _sha256_bytes(_canonical_json_bytes(inventory_payload))
        != evidence["host_inventory_identity_sha256"]
    ):
        raise ValueError("host inventory identity mismatch")

    mode = ExecutionMode(evidence["execution_mode"])
    if mode is not packet.execution_mode:
        raise ValueError("physical evidence execution mode mismatch")
    verdict = QualificationVerdict(evidence["verdict"])
    if mode is ExecutionMode.SIMULATION and verdict is QualificationVerdict.PASS:
        raise ValueError("simulation evidence cannot claim physical PASS")
    if require_real_pass and (
        mode is not ExecutionMode.REAL_HOST or verdict is not QualificationVerdict.PASS
    ):
        raise ValueError("real physical PASS is required")

    resource_payload = evidence["resource_observations"]
    if not isinstance(resource_payload, dict) or set(resource_payload) != {
        item.value for item in ResourceKind
    }:
        raise ValueError("resource observations are incomplete")
    resource_values = {
        ResourceKind(key): ResourceObservation(value)
        for key, value in resource_payload.items()
    }
    required = {
        resource
        for action in packet.actions
        for resource in action.required_resources
    }
    if verdict is QualificationVerdict.PASS:
        for resource in required:
            if resource_values[resource] is not ResourceObservation.REAL_PROBED:
                raise ValueError("physical PASS contains an unproven required resource")

    log_bytes = Path(log_path).read_bytes()
    if _sha256_bytes(log_bytes) != evidence["log_sha256"]:
        raise ValueError("physical evidence log hash mismatch")
    if len(log_bytes) != evidence["log_bytes"]:
        raise ValueError("physical evidence log byte count mismatch")
    log_rows: list[dict[str, Any]] = []
    for raw_line in log_bytes.splitlines():
        try:
            row = json.loads(
                raw_line.decode("utf-8", errors="strict"),
                object_pairs_hook=_unique_json_object,
                parse_constant=lambda item: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant is not allowed: {item}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError("physical log is not canonical JSONL") from exc
        if not isinstance(row, dict):
            raise ValueError("physical log row must be an object")
        log_rows.append(row)

    actions = evidence["actions"]
    if not isinstance(actions, list) or len(actions) != len(packet.actions):
        raise ValueError("physical evidence action count mismatch")
    if len(log_rows) != len(packet.actions):
        raise ValueError("physical log action count mismatch")
    for expected, actual, log_row in zip(packet.actions, actions, log_rows, strict=True):
        if not isinstance(actual, dict):
            raise ValueError("physical action evidence must be an object")
        if actual.get("action_id") != expected.action_id:
            raise ValueError("physical action evidence order/id mismatch")
        if actual.get("argv") != list(expected.logical_argv()):
            raise ValueError("physical action argv mismatch")
        if actual.get("pre_git_sha") != packet.target_git_sha:
            raise ValueError("physical action pre-run Git SHA mismatch")
        if actual.get("post_git_sha") != packet.target_git_sha:
            raise ValueError("physical action post-run Git SHA mismatch")
        if actual.get("pre_tracked_clean") is not True:
            raise ValueError("physical action started from a dirty checkout")
        if actual.get("post_tracked_clean") is not True:
            raise ValueError("physical action dirtied the checkout")
        if log_row.get("action_id") != expected.action_id:
            raise ValueError("physical log action id mismatch")
        try:
            stdout = base64.b64decode(log_row.get("stdout_b64"), validate=True)
            stderr = base64.b64decode(log_row.get("stderr_b64"), validate=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("physical log output is not strict base64") from exc
        if _sha256_bytes(stdout) != actual.get("stdout_sha256"):
            raise ValueError("physical stdout hash mismatch")
        if _sha256_bytes(stderr) != actual.get("stderr_sha256"):
            raise ValueError("physical stderr hash mismatch")
        if len(stdout) != actual.get("stdout_bytes") or len(stderr) != actual.get(
            "stderr_bytes"
        ):
            raise ValueError("physical action output byte count mismatch")
        if actual.get("verdict") == ActionVerdict.PASS.value and (
            actual.get("stdout_truncated") is not False
            or actual.get("stderr_truncated") is not False
        ):
            raise ValueError("physical PASS action contains truncated output")

    artifacts = evidence["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(packet.artifact_paths):
        raise ValueError("physical artifact evidence count mismatch")
    root = Path(artifact_root).resolve()
    by_path = {item.get("path"): item for item in artifacts if isinstance(item, dict)}
    if set(by_path) != set(packet.artifact_paths):
        raise ValueError("physical artifact evidence path mismatch")
    total_bytes = 0
    for relative in packet.artifact_paths:
        target = root / relative
        if target.is_symlink() or not target.is_file():
            raise ValueError("physical artifact is missing or not a regular file")
        resolved = target.resolve()
        if root not in resolved.parents:
            raise ValueError("physical artifact escapes repository")
        raw = target.read_bytes()
        total_bytes += len(raw)
        if total_bytes > _MAX_ARTIFACT_BYTES:
            raise ValueError("physical artifacts exceed byte bound")
        item = by_path[relative]
        if item.get("bytes") != len(raw) or item.get("sha256") != _sha256_bytes(raw):
            raise ValueError("physical artifact hash/size mismatch")

    return evidence
