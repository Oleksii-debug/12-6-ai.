"""Windows LOCAL_FREE operator preflight and safe-stop request protocol.

This module is deliberately authority-neutral. Passing machine checks does not
authorize launch or training, and a stop marker is only an operator request for
the canonical trainer to consume at a checkpoint-safe boundary.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import stat
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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

_TRUTH_BOUNDARY = {
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed_on_real_targets": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
    "external_llm_or_api_used_for_data_or_intelligence": False,
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


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json_file(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise OperatorPreflightError(f"cannot_read_json:{path}:{exc}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperatorPreflightError(f"invalid_json:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise OperatorPreflightError(f"json_root_must_be_object:{path}")
    return value, _file_sha256(raw)


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def validate_operator_profile(
    profile: Mapping[str, Any], packet: Mapping[str, Any]
) -> list[str]:
    """Validate profile shape without turning preflight into launch authority."""
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
    if not isinstance(portable, Mapping):
        errors.append("portable_packet_binding_missing")
        portable = {}
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

    resource = packet.get("resource")
    if not isinstance(resource, Mapping):
        errors.append("packet_resource_missing")
    else:
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
        _expect(
            errors,
            type(resource.get("maximum_cost_usd")) in (int, float)
            and resource.get("maximum_cost_usd") == 0,
            "packet_maximum_cost_must_be_zero",
        )

    policy = profile.get("machine_policy")
    if not isinstance(policy, Mapping):
        errors.append("machine_policy_missing")
        policy = {}
    _expect(errors, policy.get("required_os") == "Windows", "required_os_must_be_windows")
    for key in (
        "minimum_windows_release",
        "minimum_logical_cpus",
        "minimum_ram_bytes",
        "minimum_free_disk_bytes",
        "warn_below_logical_cpus",
        "warn_below_ram_bytes",
        "warn_below_free_disk_bytes",
    ):
        _expect(
            errors,
            _strict_int(policy.get(key)) and policy.get(key) > 0,
            f"{key}_must_be_positive_integer",
        )
    _expect(errors, policy.get("cpu_path_required") is True, "cpu_path_must_be_required")
    _expect(errors, policy.get("gpu_required") is False, "gpu_must_not_be_required")
    _expect(
        errors,
        policy.get("threshold_semantics")
        == "OPERATOR_ADMISSION_FLOORS_NOT_FULL_TRAINING_SUFFICIENCY",
        "threshold_semantics_mismatch",
    )
    for floor, warning in (
        ("minimum_logical_cpus", "warn_below_logical_cpus"),
        ("minimum_ram_bytes", "warn_below_ram_bytes"),
        ("minimum_free_disk_bytes", "warn_below_free_disk_bytes"),
    ):
        if _strict_int(policy.get(floor)) and _strict_int(policy.get(warning)):
            _expect(errors, policy[warning] >= policy[floor], f"{warning}_below_floor")

    targets = profile.get("targets")
    if not isinstance(targets, Mapping):
        errors.append("targets_missing")
        targets = {}
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
    if not isinstance(stop, Mapping):
        errors.append("safe_stop_missing")
    else:
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
        for key in (
            "content_authenticated",
            "idempotent",
            "does_not_claim_checkpoint_or_training_success",
        ):
            _expect(errors, stop.get(key) is True, f"safe_stop_{key}_must_be_true")

    truth = profile.get("truth_boundary")
    _expect(errors, truth == _TRUTH_BOUNDARY, "profile_truth_boundary_mismatch")
    return sorted(set(errors))


def _windows_release_major(value: str) -> int | None:
    prefix = value.strip().split(".", 1)[0]
    return int(prefix) if prefix.isdigit() else None


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
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    kernel32 = getattr(getattr(ctypes, "windll", None), "kernel32", None)
    if kernel32 is None or not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OperatorPreflightError("windows_ram_query_failed")
    return int(status.ullTotalPhys)


def collect_machine_facts(disk_path: Path) -> MachineFacts:
    """Collect only local, non-network machine facts."""
    os_name = platform.system()
    ram = 0 if os_name != "Windows" else _windows_total_ram_bytes()
    try:
        free_disk = int(shutil.disk_usage(disk_path).free)
    except OSError as exc:
        raise OperatorPreflightError(f"disk_query_failed:{disk_path}:{exc}") from exc
    logical = os.cpu_count()
    return MachineFacts(
        os_name=os_name,
        os_release=platform.release(),
        machine=platform.machine(),
        logical_cpus=int(logical)
        if isinstance(logical, int) and not isinstance(logical, bool)
        else 0,
        ram_bytes=ram,
        free_disk_bytes=free_disk,
        python_version=platform.python_version(),
    )


def assess_machine(
    profile: Mapping[str, Any],
    packet: Mapping[str, Any],
    facts: MachineFacts,
    *,
    target: str,
) -> dict[str, Any]:
    """Return deterministic fail-closed operator preflight assessment."""
    errors = validate_operator_profile(profile, packet)
    if target not in {"20m", "100m"}:
        errors.append("target_must_be_20m_or_100m")

    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    policy = profile.get("machine_policy")
    if not isinstance(policy, Mapping):
        policy = {}

    def add_check(name: str, passed: bool, observed: Any, required: Any) -> None:
        checks.append(
            {
                "name": name,
                "passed": passed is True,
                "observed": observed,
                "required": required,
            }
        )

    os_ok = facts.os_name == "Windows"
    add_check("os", os_ok, facts.os_name, "Windows")

    major = _windows_release_major(facts.os_release) if os_ok else None
    min_release = policy.get("minimum_windows_release")
    release_ok = _strict_int(min_release) and major is not None and major >= min_release
    add_check("windows_release", bool(release_ok), facts.os_release, min_release)

    def numeric_check(name: str, observed: Any, floor_key: str, warn_key: str) -> None:
        floor = policy.get(floor_key)
        warn = policy.get(warn_key)
        observed_valid = _strict_int(observed) and observed > 0
        floor_valid = _strict_int(floor) and floor > 0
        passed = observed_valid and floor_valid and observed >= floor
        add_check(name, bool(passed), observed, floor)
        if passed and _strict_int(warn) and observed < warn:
            warnings.append(f"{name}_below_recommended")

    numeric_check(
        "logical_cpus",
        facts.logical_cpus,
        "minimum_logical_cpus",
        "warn_below_logical_cpus",
    )
    numeric_check("ram_bytes", facts.ram_bytes, "minimum_ram_bytes", "warn_below_ram_bytes")
    numeric_check(
        "free_disk_bytes",
        facts.free_disk_bytes,
        "minimum_free_disk_bytes",
        "warn_below_free_disk_bytes",
    )

    all_checks_pass = all(item["passed"] is True for item in checks)
    contract_valid = not errors
    target_cfg = profile.get("targets", {}).get(target, {})
    qualification_only = bool(
        isinstance(target_cfg, Mapping) and target_cfg.get("qualification_only") is True
    )

    if not contract_valid or not all_checks_pass:
        status = "BLOCKED"
    elif qualification_only:
        status = "QUALIFICATION_ONLY"
    else:
        status = "PREFLIGHT_PASS"

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
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise OperatorPreflightError(f"state_dir_lstat_failed:{exc}") from exc
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
    digest = marker.get("marker_sha256")
    unsigned = dict(marker)
    unsigned.pop("marker_sha256", None)
    _expect(
        errors,
        isinstance(digest, str) and digest == _canonical_sha256(unsigned),
        "safe_stop_marker_sha256_invalid",
    )
    _expect(
        errors,
        marker.get("schema") == "12-6.windows-local-free-safe-stop-request.v1",
        "safe_stop_marker_schema_mismatch",
    )
    _expect(
        errors,
        marker.get("request") == "SAFE_STOP_AT_NEXT_CHECKPOINT",
        "safe_stop_request_mismatch",
    )
    _expect(errors, marker.get("target") == target, "safe_stop_target_mismatch")
    _expect(
        errors,
        marker.get("profile_sha256") == profile_sha256,
        "safe_stop_profile_binding_mismatch",
    )
    _expect(
        errors,
        marker.get("portable_packet_sha256") == packet_sha256,
        "safe_stop_packet_binding_mismatch",
    )
    _expect(
        errors,
        marker.get("consumer") == "CANONICAL_TRAINER_ONLY",
        "safe_stop_consumer_mismatch",
    )
    _expect(
        errors,
        marker.get("checkpoint_or_training_success_claimed") is False,
        "safe_stop_must_not_claim_success",
    )
    _expect(
        errors,
        marker.get("truth_boundary") == _TRUTH_BOUNDARY,
        "safe_stop_truth_boundary_mismatch",
    )
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
    try:
        mode = marker_path.lstat().st_mode
    except OSError as exc:
        raise OperatorPreflightError(f"stop_marker_lstat_failed:{exc}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        return {
            "status": "INVALID",
            "marker": None,
            "errors": ["safe_stop_marker_must_be_regular_file"],
        }
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
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
    timestamp = requested_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = _marker_payload(
        profile_sha256=profile_sha256,
        packet_sha256=packet_sha256,
        target=target,
        requested_at_utc=timestamp,
    )
    raw = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8")
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


def _render_text(result: Mapping[str, Any]) -> str:
    lines = [f"OPERATOR_STATUS: {result.get('status')}"]
    if "target" in result:
        lines.append(f"TARGET: {result.get('target')}")
    if "launch_authorized" in result:
        lines.append(
            f"LAUNCH_AUTHORIZED: {str(result.get('launch_authorized')).lower()}"
        )
    if "training_authorized" in result:
        lines.append(
            f"TRAINING_AUTHORIZED: {str(result.get('training_authorized')).lower()}"
        )
    checks = result.get("checks")
    if isinstance(checks, list):
        for item in checks:
            if isinstance(item, Mapping):
                lines.append(
                    f"CHECK {item.get('name')}: "
                    f"{'PASS' if item.get('passed') is True else 'FAIL'} "
                    f"(observed={item.get('observed')!r}, required={item.get('required')!r})"
                )
    warnings = result.get("warnings")
    if isinstance(warnings, list):
        for warning in warnings:
            lines.append(f"WARNING: {warning}")
    errors = result.get("contract_errors")
    if isinstance(errors, list):
        for error in errors:
            lines.append(f"ERROR: {error}")
    if "marker_path" in result:
        lines.append(f"STOP_MARKER: {result.get('marker_path')}")
    return "\n".join(lines)


def _print_result(result: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    else:
        print(_render_text(result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="12-6 Windows LOCAL_FREE operator preflight/status/safe-stop"
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--json", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("verify", "status", "request-stop"):
        sub = subparsers.add_parser(command)
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
            result = {
                **result,
                "target": args.target,
                "launch_authorized": False,
                "training_authorized": False,
                "truth_boundary": dict(_TRUTH_BOUNDARY),
            }
            _print_result(result, as_json=args.json)
            return EXIT_OK

        disk_path = args.state_dir if args.state_dir.exists() else args.state_dir.parent
        facts = collect_machine_facts(disk_path)
        assessment = assess_machine(profile, packet, facts, target=args.target)
        if args.command == "status":
            if args.state_dir.exists() or args.state_dir.is_symlink():
                stop = read_stop_status(
                    args.state_dir,
                    profile_sha256=profile_sha,
                    packet_sha256=packet_sha,
                    target=args.target,
                )
            else:
                stop = {"status": "NOT_REQUESTED", "marker": None, "errors": []}
            assessment["safe_stop"] = stop
            if stop["status"] == "INVALID":
                assessment["status"] = "BLOCKED"
                assessment["operator_preflight_passed"] = False
        _print_result(assessment, as_json=args.json)
        return (
            EXIT_OK
            if assessment["status"] in {"PREFLIGHT_PASS", "QUALIFICATION_ONLY"}
            else EXIT_BLOCKED
        )
    except OperatorPreflightError as exc:
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
