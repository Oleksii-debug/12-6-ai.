"""Windows LOCAL_FREE operator preflight and authenticated safe-stop requests.

Passing this preflight never authorizes launch or training. A stop marker is only
an operator request for the canonical trainer to consume at a checkpoint-safe
boundary; it is not evidence that a checkpoint or optimizer update occurred.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import shutil
import stat
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from twelve_six.portable_run_packet import validate_portable_run_contract

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "configs/research/r01_windows_local_free_operator_v1.json"
DEFAULT_PACKET = ROOT / "configs/research/r01_portable_local_free_run_packet_v1.json"
DEFAULT_STATE_DIR = ROOT / ".twelve-six-local"
PROFILE_SCHEMA_VERSION = 1
PROFILE_ID = "R01-WINDOWS-LOCAL-FREE-OPERATOR-V1"
PACKET_SCHEMA_VERSION = 1
PACKET_ID = "R01-LEARNED20M-PORTABLE-RUN-PACKET-V1"
EXIT_OK = 0
EXIT_BLOCKED = 2
EXIT_ERROR = 3

# Leaf-local facts only. Corpus/source provenance is deliberately not asserted
# by this machine/operator preflight.
_TRUTH_BOUNDARY = {
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed_on_real_targets": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
}


class OperatorPreflightError(RuntimeError):
    """Fail-closed input, path, or marker error."""


@dataclass(frozen=True)
class MachineFacts:
    os_name: str
    os_release: str
    machine: str
    logical_cpus: int
    ram_bytes: int
    free_disk_bytes: int
    python_version: str
    windows_version_major: int | None = None
    windows_version_minor: int | None = None
    windows_version_build: int | None = None


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate_object_key:{key}")
        result[key] = value
    return result


def _json_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non_finite_number:{token}")
    return value


def _json_reject_constant(token: str) -> Any:
    raise ValueError(f"non_finite_number:{token}")


def _strict_json_loads(raw: str) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_json_object_no_duplicates,
        parse_float=_json_finite_float,
        parse_constant=_json_reject_constant,
    )


def _read_json_file(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = _strict_json_loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OperatorPreflightError(f"invalid_or_unreadable_json:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise OperatorPreflightError(f"json_root_must_be_object:{path}")
    return value, hashlib.sha256(raw).hexdigest()


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def validate_operator_profile(
    profile: Mapping[str, Any], packet: Mapping[str, Any]
) -> list[str]:
    """Validate the operator contract without manufacturing launch authority."""
    errors: list[str] = []
    _expect(
        errors,
        _strict_int(profile.get("schema_version"))
        and profile.get("schema_version") == PROFILE_SCHEMA_VERSION,
        "profile_schema_version_mismatch",
    )
    _expect(errors, profile.get("profile_id") == PROFILE_ID, "profile_id_mismatch")
    _expect(
        errors,
        profile.get("status") == "OPERATOR_PREFLIGHT_ONLY",
        "profile_status_must_be_operator_preflight_only",
    )
    _expect(
        errors,
        profile.get("execution_mode") == "LOCAL_FREE",
        "execution_mode_must_be_local_free",
    )

    portable = profile.get("portable_packet")
    portable = portable if isinstance(portable, Mapping) else {}
    _expect(
        errors,
        portable.get("packet_id") == PACKET_ID,
        "portable_packet_id_binding_mismatch",
    )
    _expect(
        errors,
        _strict_int(portable.get("schema_version"))
        and portable.get("schema_version") == PACKET_SCHEMA_VERSION,
        "portable_packet_schema_binding_mismatch",
    )
    _expect(errors, packet.get("packet_id") == PACKET_ID, "packet_id_mismatch")
    _expect(
        errors,
        _strict_int(packet.get("schema_version"))
        and packet.get("schema_version") == PACKET_SCHEMA_VERSION,
        "packet_schema_version_mismatch",
    )

    if isinstance(packet, dict):
        portable_errors = validate_portable_run_contract(packet)
    else:
        portable_errors = validate_portable_run_contract(dict(packet))
    errors.extend(f"portable_packet_contract:{item}" for item in portable_errors)

    resource = packet.get("resource")
    resource = resource if isinstance(resource, Mapping) else {}
    _expect(
        errors,
        resource.get("resource_class") == "LOCAL_FREE",
        "packet_resource_class_must_be_local_free",
    )
    _expect(
        errors,
        resource.get("materially_paid") is False,
        "packet_materially_paid_must_be_false",
    )
    cost = resource.get("maximum_cost_usd")
    _expect(
        errors,
        type(cost) in (int, float) and math.isfinite(float(cost)) and cost == 0,
        "packet_maximum_cost_must_be_zero",
    )

    policy = profile.get("machine_policy")
    policy = policy if isinstance(policy, Mapping) else {}
    _expect(errors, policy.get("required_os") == "Windows", "required_os_must_be_windows")
    _expect(errors, policy.get("cpu_path_required") is True, "cpu_path_must_be_required")
    _expect(errors, policy.get("gpu_required") is False, "gpu_must_not_be_required")
    _expect(
        errors,
        policy.get("threshold_semantics")
        == "OPERATOR_ADMISSION_FLOORS_NOT_FULL_TRAINING_SUFFICIENCY",
        "threshold_semantics_mismatch",
    )

    version_rules = (
        ("minimum_windows_major", 0),
        ("minimum_windows_minor", 0),
        ("minimum_windows_build", 1),
    )
    for key, minimum in version_rules:
        value = policy.get(key)
        _expect(
            errors,
            _strict_int(value) and value >= minimum,
            f"{key}_must_be_integer_at_least_{minimum}",
        )

    numeric_pairs = (
        ("minimum_logical_cpus", "warn_below_logical_cpus"),
        ("minimum_ram_bytes", "warn_below_ram_bytes"),
        ("minimum_free_disk_bytes", "warn_below_free_disk_bytes"),
    )
    for floor_key, warning_key in numeric_pairs:
        floor = policy.get(floor_key)
        _expect(
            errors,
            _strict_int(floor) and floor > 0,
            f"{floor_key}_must_be_positive_integer",
        )
        warning = policy.get(warning_key)
        _expect(
            errors,
            _strict_int(warning) and warning > 0,
            f"{warning_key}_must_be_positive_integer",
        )
        if _strict_int(floor) and _strict_int(warning):
            _expect(errors, warning >= floor, f"{warning_key}_below_floor")

    targets = profile.get("targets")
    targets = targets if isinstance(targets, Mapping) else {}
    for target in ("20m", "100m"):
        value = targets.get(target)
        if not isinstance(value, Mapping):
            errors.append(f"target_{target}_missing")
            continue
        _expect(
            errors,
            value.get("operator_preflight_only") is True,
            f"target_{target}_must_be_operator_preflight_only",
        )
        _expect(
            errors,
            value.get("launch_authorized_by_profile") is False,
            f"target_{target}_must_not_authorize_launch",
        )
        _expect(
            errors,
            value.get("training_authorized_by_profile") is False,
            f"target_{target}_must_not_authorize_training",
        )
    hundred = targets.get("100m")
    if isinstance(hundred, Mapping):
        _expect(
            errors,
            hundred.get("qualification_only") is True,
            "target_100m_must_be_qualification_only",
        )
        _expect(
            errors,
            hundred.get("runtime_or_speed_guaranteed") is False,
            "target_100m_must_not_guarantee_runtime_or_speed",
        )

    stop = profile.get("safe_stop")
    stop = stop if isinstance(stop, Mapping) else {}
    _expect(
        errors,
        stop.get("marker_filename") == "STOP_REQUEST.json",
        "safe_stop_marker_filename_mismatch",
    )
    _expect(
        errors,
        stop.get("request_kind") == "SAFE_STOP_AT_NEXT_CHECKPOINT",
        "safe_stop_request_kind_mismatch",
    )
    _expect(
        errors,
        stop.get("consumer") == "CANONICAL_TRAINER_ONLY",
        "safe_stop_consumer_mismatch",
    )
    for key in (
        "content_authenticated",
        "idempotent",
        "does_not_claim_checkpoint_or_training_success",
    ):
        _expect(errors, stop.get(key) is True, f"safe_stop_{key}_must_be_true")

    _expect(
        errors,
        profile.get("truth_boundary") == _TRUTH_BOUNDARY,
        "profile_truth_boundary_mismatch",
    )
    return sorted(set(errors))


def _windows_total_ram_bytes() -> int:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    kernel32 = getattr(getattr(ctypes, "windll", None), "kernel32", None)
    if kernel32 is None or not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OperatorPreflightError("windows_ram_query_failed")
    return int(status.ullTotalPhys)


def _windows_native_version() -> tuple[int, int, int]:
    getwindowsversion = getattr(sys, "getwindowsversion", None)
    if getwindowsversion is None:
        raise OperatorPreflightError("windows_native_version_query_unavailable")
    version = getwindowsversion()
    values = (
        getattr(version, "major", None),
        getattr(version, "minor", None),
        getattr(version, "build", None),
    )
    if not all(_strict_int(value) and value >= 0 for value in values):
        raise OperatorPreflightError("windows_native_version_invalid")
    return int(values[0]), int(values[1]), int(values[2])


def collect_machine_facts(disk_path: Path) -> MachineFacts:
    os_name = platform.system()
    try:
        free_disk = int(shutil.disk_usage(disk_path).free)
    except OSError as exc:
        raise OperatorPreflightError(f"disk_query_failed:{disk_path}:{exc}") from exc
    logical = os.cpu_count()
    logical = logical if _strict_int(logical) else 0
    windows_version: tuple[int | None, int | None, int | None] = (None, None, None)
    if os_name == "Windows":
        windows_version = _windows_native_version()
    return MachineFacts(
        os_name=os_name,
        os_release=platform.release(),
        machine=platform.machine(),
        logical_cpus=int(logical),
        ram_bytes=_windows_total_ram_bytes() if os_name == "Windows" else 0,
        free_disk_bytes=free_disk,
        python_version=platform.python_version(),
        windows_version_major=windows_version[0],
        windows_version_minor=windows_version[1],
        windows_version_build=windows_version[2],
    )


def assess_machine(
    profile: Mapping[str, Any],
    packet: Mapping[str, Any],
    facts: MachineFacts,
    *,
    target: str,
) -> dict[str, Any]:
    errors = validate_operator_profile(profile, packet)
    if target not in {"20m", "100m"}:
        errors.append("target_must_be_20m_or_100m")
    policy = profile.get("machine_policy")
    policy = policy if isinstance(policy, Mapping) else {}
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    def add(name: str, passed: bool, observed: Any, required: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": passed is True,
                "observed": observed,
                "required": required,
            }
        )

    os_ok = facts.os_name == "Windows"
    add("os", os_ok, facts.os_name, "Windows")

    required_version = (
        policy.get("minimum_windows_major"),
        policy.get("minimum_windows_minor"),
        policy.get("minimum_windows_build"),
    )
    observed_version = (
        facts.windows_version_major,
        facts.windows_version_minor,
        facts.windows_version_build,
    )
    version_types_valid = all(_strict_int(value) and value >= 0 for value in observed_version)
    required_types_valid = (
        _strict_int(required_version[0])
        and required_version[0] >= 0
        and _strict_int(required_version[1])
        and required_version[1] >= 0
        and _strict_int(required_version[2])
        and required_version[2] > 0
    )
    version_ok = (
        os_ok
        and version_types_valid
        and required_types_valid
        and observed_version >= required_version
    )
    add(
        "windows_native_version",
        version_ok,
        {
            "major": facts.windows_version_major,
            "minor": facts.windows_version_minor,
            "build": facts.windows_version_build,
            "display_release": facts.os_release,
        },
        {
            "minimum_major": required_version[0],
            "minimum_minor": required_version[1],
            "minimum_build": required_version[2],
        },
    )

    for name, observed, floor_key, warn_key in (
        (
            "logical_cpus",
            facts.logical_cpus,
            "minimum_logical_cpus",
            "warn_below_logical_cpus",
        ),
        ("ram_bytes", facts.ram_bytes, "minimum_ram_bytes", "warn_below_ram_bytes"),
        (
            "free_disk_bytes",
            facts.free_disk_bytes,
            "minimum_free_disk_bytes",
            "warn_below_free_disk_bytes",
        ),
    ):
        floor = policy.get(floor_key)
        warning = policy.get(warn_key)
        passed = (
            _strict_int(observed)
            and observed > 0
            and _strict_int(floor)
            and observed >= floor
        )
        add(name, passed, observed, floor)
        if passed and _strict_int(warning) and observed < warning:
            warnings.append(f"{name}_below_recommended")

    target_cfg = profile.get("targets")
    target_cfg = target_cfg.get(target, {}) if isinstance(target_cfg, Mapping) else {}
    qualification_only = (
        isinstance(target_cfg, Mapping)
        and target_cfg.get("qualification_only") is True
    )
    all_pass = not errors and all(item["passed"] for item in checks)
    status = (
        "BLOCKED"
        if not all_pass
        else "QUALIFICATION_ONLY"
        if qualification_only
        else "PREFLIGHT_PASS"
    )
    return {
        "schema": "12-6.windows-local-free-operator-preflight.v1",
        "status": status,
        "target": target,
        "operator_preflight_passed": status in {"PREFLIGHT_PASS", "QUALIFICATION_ONLY"},
        "qualification_only": qualification_only,
        "launch_authorized": False,
        "training_authorized": False,
        "full_training_sufficiency_claimed": False,
        "machine": asdict(facts),
        "checks": checks,
        "warnings": sorted(set(warnings)),
        "contract_errors": sorted(set(errors)),
        "truth_boundary": dict(_TRUTH_BOUNDARY),
    }


def _ensure_state_dir(path: Path, *, create: bool) -> Path:
    path = path.absolute()
    if path.exists() or path.is_symlink():
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise OperatorPreflightError("state_dir_must_not_be_symlink")
        if not stat.S_ISDIR(mode):
            raise OperatorPreflightError("state_dir_must_be_directory")
        return path
    if not create:
        raise OperatorPreflightError("state_dir_missing")
    parent = path.parent
    if not parent.exists() or parent.is_symlink() or not parent.is_dir():
        raise OperatorPreflightError("state_dir_parent_must_be_existing_real_directory")
    try:
        path.mkdir(mode=0o700)
    except OSError as exc:
        raise OperatorPreflightError(f"state_dir_create_failed:{exc}") from exc
    return path


def _marker_payload(
    *,
    profile_sha256: str,
    packet_sha256: str,
    target: str,
    requested_at_utc: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "12-6.windows-local-free-safe-stop-request.v1",
        "request": "SAFE_STOP_AT_NEXT_CHECKPOINT",
        "target": target,
        "profile_sha256": profile_sha256,
        "portable_packet_sha256": packet_sha256,
        "requested_at_utc": requested_at_utc,
        "consumer": "CANONICAL_TRAINER_ONLY",
        "checkpoint_or_training_success_claimed": False,
        "truth_boundary": dict(_TRUTH_BOUNDARY),
    }
    payload["marker_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_existing_marker(
    marker: Mapping[str, Any],
    *,
    profile_sha256: str,
    packet_sha256: str,
    target: str,
) -> list[str]:
    errors: list[str] = []
    unsigned = dict(marker)
    digest = unsigned.pop("marker_sha256", None)
    _expect(
        errors,
        isinstance(digest, str) and digest == _canonical_sha256(unsigned),
        "safe_stop_marker_sha256_invalid",
    )
    expected = {
        "schema": "12-6.windows-local-free-safe-stop-request.v1",
        "request": "SAFE_STOP_AT_NEXT_CHECKPOINT",
        "target": target,
        "profile_sha256": profile_sha256,
        "portable_packet_sha256": packet_sha256,
        "consumer": "CANONICAL_TRAINER_ONLY",
        "checkpoint_or_training_success_claimed": False,
        "truth_boundary": _TRUTH_BOUNDARY,
    }
    for key, value in expected.items():
        _expect(errors, marker.get(key) == value, f"safe_stop_{key}_mismatch")
    return sorted(set(errors))


def read_stop_status(
    state_dir: Path,
    *,
    profile_sha256: str,
    packet_sha256: str,
    target: str,
) -> dict[str, Any]:
    state = _ensure_state_dir(state_dir, create=False)
    marker_path = state / "STOP_REQUEST.json"
    if not marker_path.exists() and not marker_path.is_symlink():
        return {"status": "NOT_REQUESTED", "marker": None, "errors": []}
    mode = marker_path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        return {
            "status": "INVALID",
            "marker": None,
            "errors": ["safe_stop_marker_must_be_regular_file"],
        }
    try:
        marker = _strict_json_loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {
            "status": "INVALID",
            "marker": None,
            "errors": ["safe_stop_marker_invalid_json"],
        }
    if not isinstance(marker, dict):
        return {
            "status": "INVALID",
            "marker": None,
            "errors": ["safe_stop_marker_root_invalid"],
        }
    errors = _validate_existing_marker(
        marker,
        profile_sha256=profile_sha256,
        packet_sha256=packet_sha256,
        target=target,
    )
    return {
        "status": "REQUESTED" if not errors else "INVALID",
        "marker": marker if not errors else None,
        "errors": errors,
    }


def request_safe_stop(
    state_dir: Path,
    *,
    profile_sha256: str,
    packet_sha256: str,
    target: str,
    requested_at_utc: str | None = None,
) -> dict[str, Any]:
    state = _ensure_state_dir(state_dir, create=True)
    marker_path = state / "STOP_REQUEST.json"
    payload = _marker_payload(
        profile_sha256=profile_sha256,
        packet_sha256=packet_sha256,
        target=target,
        requested_at_utc=requested_at_utc
        or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    raw = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False).encode()
        + b"\n"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(marker_path, flags, 0o600)
    except FileExistsError:
        existing = read_stop_status(
            state,
            profile_sha256=profile_sha256,
            packet_sha256=packet_sha256,
            target=target,
        )
        if existing["status"] != "REQUESTED":
            raise OperatorPreflightError(
                "existing_safe_stop_marker_invalid:" + ",".join(existing["errors"])
            )
        return {
            "status": "ALREADY_REQUESTED",
            "marker_path": str(marker_path),
            "marker_sha256": existing["marker"]["marker_sha256"],
        }
    except OSError as exc:
        raise OperatorPreflightError(f"safe_stop_create_failed:{exc}") from exc
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OperatorPreflightError("safe_stop_short_write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return {
        "status": "REQUESTED",
        "marker_path": str(marker_path),
        "marker_sha256": payload["marker_sha256"],
    }


def _load_bound_inputs(
    profile_path: Path, packet_path: Path
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    profile, profile_sha = _read_json_file(profile_path)
    packet, packet_sha = _read_json_file(packet_path)
    errors = validate_operator_profile(profile, packet)
    if errors:
        raise OperatorPreflightError("contract_invalid:" + ",".join(errors))
    return profile, profile_sha, packet, packet_sha


def _single_line_text(value: Any) -> str:
    """Escape control characters while preserving printable Unicode for NVDA."""
    rendered: list[str] = []
    for char in str(value):
        if char.isprintable():
            rendered.append(char)
            continue
        codepoint = ord(char)
        if codepoint <= 0xFF:
            rendered.append(f"\\x{codepoint:02x}")
        elif codepoint <= 0xFFFF:
            rendered.append(f"\\u{codepoint:04x}")
        else:
            rendered.append(f"\\U{codepoint:08x}")
    return "".join(rendered)


def _render_text(result: Mapping[str, Any]) -> str:
    lines = [f"OPERATOR_STATUS: {result.get('status')}"]
    if "error" in result:
        lines.append(f"ERROR: {result['error']}")
    for key, label in (
        ("target", "TARGET"),
        ("launch_authorized", "LAUNCH_AUTHORIZED"),
        ("training_authorized", "TRAINING_AUTHORIZED"),
    ):
        if key in result:
            value = result.get(key)
            value = str(value).lower() if isinstance(value, bool) else value
            lines.append(f"{label}: {value}")
    safe_stop = result.get("safe_stop")
    if isinstance(safe_stop, Mapping):
        if "status" in safe_stop:
            lines.append(f"SAFE_STOP_STATUS: {safe_stop.get('status')}")
        safe_stop_errors = safe_stop.get("errors", [])
        if isinstance(safe_stop_errors, list):
            lines.extend(f"SAFE_STOP_ERROR: {item}" for item in safe_stop_errors)
    for item in result.get("checks", []):
        if isinstance(item, Mapping):
            verdict = "PASS" if item.get("passed") is True else "FAIL"
            lines.append(
                f"CHECK {item.get('name')}: {verdict} "
                f"(observed={item.get('observed')!r}, required={item.get('required')!r})"
            )
    lines.extend(f"WARNING: {item}" for item in result.get("warnings", []))
    lines.extend(f"ERROR: {item}" for item in result.get("contract_errors", []))
    if "marker_path" in result:
        lines.append(f"STOP_MARKER: {result['marker_path']}")
    return "\n".join(_single_line_text(line) for line in lines)


def _print_result(result: Mapping[str, Any], *, as_json: bool) -> None:
    print(
        json.dumps(result, sort_keys=True, ensure_ascii=False)
        if as_json
        else _render_text(result)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="12-6 Windows LOCAL_FREE operator preflight/status/safe-stop"
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--json", action="store_true")
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "status", "request-stop"):
        sub = subs.add_parser(command)
        sub.add_argument("--target", choices=("20m", "100m"), default="20m")
    args = parser.parse_args(argv)

    try:
        profile, profile_sha, packet, packet_sha = _load_bound_inputs(
            args.profile, args.packet
        )
        if args.command == "request-stop":
            result = request_safe_stop(
                args.state_dir,
                profile_sha256=profile_sha,
                packet_sha256=packet_sha,
                target=args.target,
            )
            result.update(
                {
                    "target": args.target,
                    "launch_authorized": False,
                    "training_authorized": False,
                    "truth_boundary": dict(_TRUTH_BOUNDARY),
                }
            )
            _print_result(result, as_json=args.json)
            return EXIT_OK

        disk_path = args.state_dir if args.state_dir.exists() else args.state_dir.parent
        result = assess_machine(
            profile,
            packet,
            collect_machine_facts(disk_path),
            target=args.target,
        )
        if args.command == "status":
            result["safe_stop"] = (
                read_stop_status(
                    args.state_dir,
                    profile_sha256=profile_sha,
                    packet_sha256=packet_sha,
                    target=args.target,
                )
                if args.state_dir.exists() or args.state_dir.is_symlink()
                else {"status": "NOT_REQUESTED", "marker": None, "errors": []}
            )
            if result["safe_stop"]["status"] == "INVALID":
                result["status"] = "BLOCKED"
                result["operator_preflight_passed"] = False
        _print_result(result, as_json=args.json)
        return (
            EXIT_OK
            if result["status"] in {"PREFLIGHT_PASS", "QUALIFICATION_ONLY"}
            else EXIT_BLOCKED
        )
    except (OSError, OperatorPreflightError) as exc:
        result = {
            "status": "ERROR",
            "error": str(exc),
            "launch_authorized": False,
            "training_authorized": False,
            "truth_boundary": dict(_TRUTH_BOUNDARY),
        }
        _print_result(result, as_json=args.json)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())