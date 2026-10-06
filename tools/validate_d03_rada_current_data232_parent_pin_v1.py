#!/usr/bin/env python3
"""Validate and safely extract the terminal DATA-232 parent artifact pin.

The pin is intentionally created only after independent artifact readback. This
tool rejects placeholder, widened, path-aliased, or transport-drifted evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "12-6.d03-rada-current-data232-parent-artifact-pin.v1"
PARENT_EXECUTION_HEAD = "a7982bdfd1650b062808024856e13b1d8916a634"
EXPECTED_ARTIFACT_NAME = f"d03-rada-current-data232-{PARENT_EXECUTION_HEAD}"
SHA64 = re.compile(r"^[0-9a-f]{64}$")
LOGICAL_FILES = {
    "inventory": ("inventory_identity_sha256", "parent-inventory.json"),
    "handoff": ("handoff_identity_sha256", "parent-handoff.json"),
    "report": ("report_sha256", "parent-report.json"),
    "execution_evidence": (
        "execution_identity_sha256",
        "parent-execution-evidence.json",
    ),
    "result": ("result_identity_sha256", "parent-result.json"),
    "proof": ("proof_identity_sha256", "parent-two-clean-proof.json"),
}


class ParentPinError(RuntimeError):
    """Raised when the independently pinned parent artifact cannot be trusted."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ParentPinError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_sha256(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and SHA64.fullmatch(value) is not None,
        f"{label} must be 64 lowercase hex",
    )
    return value


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ParentPinError(f"non-finite JSON constant rejected: {value}")


def strict_loads(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ParentPinError(f"{label}: strict JSON decode failed") from exc
    require(isinstance(value, dict), f"{label}: root must be an object")
    return value


def _safe_member(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} missing")
    require("\\" not in value, f"{label} must use ZIP forward slashes")
    pure = PurePosixPath(value)
    require(not pure.is_absolute(), f"{label} must be relative")
    require(
        all(part not in {"", ".", ".."} for part in pure.parts),
        f"{label} contains unsafe path component",
    )
    return value


def validate_pin(pin: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "parent_execution_head_sha",
        "workflow_run_id",
        "artifact_id",
        "artifact_name",
        "artifact_zip_sha256",
        "allowed_members",
        "files",
        "qualification",
    }
    require(set(pin) == required, "parent pin root schema is not closed")
    require(pin.get("schema_version") == SCHEMA, "parent pin schema drift")
    require(
        pin.get("parent_execution_head_sha") == PARENT_EXECUTION_HEAD,
        "parent execution HEAD drift",
    )
    for field in ("workflow_run_id", "artifact_id"):
        require(
            type(pin.get(field)) is int and pin[field] > 0,
            f"{field} must be a positive integer",
        )
    require(pin.get("artifact_name") == EXPECTED_ARTIFACT_NAME, "artifact name drift")
    require_sha256(pin.get("artifact_zip_sha256"), "artifact ZIP SHA-256")

    allowed = pin.get("allowed_members")
    require(isinstance(allowed, list) and bool(allowed), "allowed_members missing")
    normalized_allowed = [_safe_member(value, "allowed member") for value in allowed]
    require(
        normalized_allowed == sorted(normalized_allowed),
        "allowed_members must be sorted",
    )
    require(
        len(normalized_allowed) == len(set(normalized_allowed)),
        "allowed_members contains duplicates",
    )

    files = pin.get("files")
    require(isinstance(files, dict) and set(files) == set(LOGICAL_FILES), "files schema drift")
    used_members: set[str] = set()
    for logical, (identity_field, _output_name) in LOGICAL_FILES.items():
        row = files[logical]
        require(
            isinstance(row, dict)
            and set(row) == {"member", "file_sha256", "identity_sha256"},
            f"files.{logical} schema drift",
        )
        member = _safe_member(row.get("member"), f"files.{logical}.member")
        require(member in normalized_allowed, f"files.{logical} member is not allowed")
        require(member not in used_members, "logical files reuse one ZIP member")
        used_members.add(member)
        require_sha256(row.get("file_sha256"), f"files.{logical}.file_sha256")
        require_sha256(row.get("identity_sha256"), f"files.{logical}.identity_sha256")
        require(identity_field.endswith("sha256"), "internal identity field configuration drift")

    qualification = pin.get("qualification")
    require(
        qualification == {
            "fresh_execution_count": 2,
            "independent_runner_jobs": True,
            "two_fresh_processes_byte_identical": True,
            "durable_evidence_hash_only": True,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "tokenizer_fit_authorized": False,
            "training_executed": False,
            "learned_weights_created": False,
            "paid_compute_used": False,
            "scale_promotion_authorized": False,
        },
        "parent qualification boundary drift",
    )
    return pin


def load_pin(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "pin file invalid")
    return validate_pin(strict_loads(path.read_bytes(), "parent pin"))


def _zip_members(archive: zipfile.ZipFile) -> list[str]:
    observed: list[str] = []
    for info in archive.infolist():
        require(not info.is_dir(), "artifact ZIP contains a directory member")
        name = _safe_member(info.filename, "artifact ZIP member")
        mode = (info.external_attr >> 16) & 0xFFFF
        require(not stat.S_ISLNK(mode), f"artifact ZIP symlink rejected: {name}")
        observed.append(name)
    require(len(observed) == len(set(observed)), "artifact ZIP contains duplicate members")
    return sorted(observed)


def _write_new(path: Path, payload: bytes) -> None:
    require(not path.is_symlink(), f"output path is a symlink: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"output already exists: {path.name}")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def extract(pin_path: Path, artifact_zip: Path, output_dir: Path) -> dict[str, Any]:
    pin = load_pin(pin_path)
    require(
        artifact_zip.is_file() and not artifact_zip.is_symlink(),
        "artifact ZIP path invalid",
    )
    archive_raw = artifact_zip.read_bytes()
    require(
        sha256(archive_raw) == pin["artifact_zip_sha256"],
        "artifact ZIP SHA-256 drift",
    )
    require(not output_dir.is_symlink(), "output directory must not be a symlink")
    if output_dir.exists():
        require(output_dir.is_dir(), "output path is not a directory")
        require(not any(output_dir.iterdir()), "output directory must be empty")
    else:
        output_dir.mkdir(parents=True)

    extracted: dict[str, Any] = {}
    with zipfile.ZipFile(artifact_zip, "r") as archive:
        observed = _zip_members(archive)
        require(observed == pin["allowed_members"], "artifact ZIP member set drift")
        for logical, (identity_field, output_name) in LOGICAL_FILES.items():
            row = pin["files"][logical]
            payload = archive.read(row["member"])
            require(
                sha256(payload) == row["file_sha256"],
                f"{logical} file SHA-256 drift",
            )
            document = strict_loads(payload, logical)
            require(
                document.get(identity_field) == row["identity_sha256"],
                f"{logical} identity drift",
            )
            destination = output_dir / output_name
            _write_new(destination, payload)
            extracted[logical] = {
                "member": row["member"],
                "output": output_name,
                "file_sha256": row["file_sha256"],
                "identity_sha256": row["identity_sha256"],
            }

    manifest = {
        "schema_version": SCHEMA,
        "parent_execution_head_sha": PARENT_EXECUTION_HEAD,
        "workflow_run_id": pin["workflow_run_id"],
        "artifact_id": pin["artifact_id"],
        "artifact_name": pin["artifact_name"],
        "artifact_zip_sha256": pin["artifact_zip_sha256"],
        "files": extracted,
    }
    _write_new(output_dir / "validated-parent-manifest.json", canonical(manifest) + b"\n")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = result.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", allow_abbrev=False)
    validate.add_argument("--pin", type=Path, required=True)
    extract_parser = sub.add_parser("extract", allow_abbrev=False)
    extract_parser.add_argument("--pin", type=Path, required=True)
    extract_parser.add_argument("--artifact-zip", type=Path, required=True)
    extract_parser.add_argument("--output-dir", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "validate":
            pin = load_pin(args.pin)
            print(
                "D03_RADA_DATA232_PARENT_PIN=PASS "
                + str(pin["artifact_id"])
                + " "
                + pin["artifact_zip_sha256"]
            )
        else:
            manifest = extract(args.pin, args.artifact_zip, args.output_dir)
            print(
                "D03_RADA_DATA232_PARENT_ARTIFACT=PASS "
                + str(manifest["artifact_id"])
                + " "
                + manifest["artifact_zip_sha256"]
            )
    except (OSError, RuntimeError, UnicodeError, ValueError, zipfile.BadZipFile) as exc:
        detail = " ".join(str(exc).split())[:500]
        print(f"D03_RADA_DATA232_PARENT_ARTIFACT=BLOCKED: {detail}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
