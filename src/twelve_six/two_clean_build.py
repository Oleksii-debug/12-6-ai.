"""Current-main successor for deterministic two-clean-build verification.

This module deliberately does not build, tokenize, split, pack, or authorize a corpus.
It compares two caller-supplied clean build roots byte-for-byte and binds the result
to exact upstream scientific identities. The historical NEXT100-070 comparator
semantics are retained without importing its stale corpus/gate snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
BINDING_ID = "R01-TWO-CLEAN-BUILD-V1"
REPORT_SCHEMA = "12-6.r01-two-clean-build-report.v1"

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_KEYS = {
    "source_git_sha",
    "corpus_manifest_sha256",
    "split_sha256",
    "tokenizer_sha256",
    "packing_sha256",
    "build_config_sha256",
}
_TRUTH_KEYS = {
    "execution_class",
    "clean_build_count",
    "shared_mutable_cache_allowed",
    "semantic_packing_validation_separate",
    "tokenizer_fit_executed_by_verifier",
    "final_test_accessed",
    "model_training_executed",
    "paid_compute_used",
    "unique_loss_ledger_authorized_by_verifier",
}


class CleanBuildError(ValueError):
    """Raised when a clean-build contract or build root fails closed."""


@dataclass(frozen=True)
class BuildTree:
    """Content-only view of one build root; host/path metadata is excluded."""

    entries: tuple[dict[str, Any], ...]
    total_bytes: int
    tree_sha256: str


@dataclass(frozen=True)
class CleanBuildAssessment:
    """Two-root comparison result without downstream training authority."""

    identical: bool
    blockers: tuple[str, ...]
    report: dict[str, Any] | None


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON encoding used for stable identities and retained reports."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (payload + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and _GIT_SHA_RE.fullmatch(value) is not None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _exact_keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CleanBuildError(f"{name}_must_be_object")
    missing = expected - set(value)
    unexpected = set(value) - expected
    if missing:
        raise CleanBuildError(f"{name}_missing:{','.join(sorted(missing))}")
    if unexpected:
        raise CleanBuildError(f"{name}_unexpected:{','.join(sorted(unexpected))}")
    return value


def validate_binding(data: Any) -> list[str]:
    """Validate the checked-in binding/template without touching build payloads."""
    if not isinstance(data, dict):
        return ["binding_root_must_be_object"]

    errors: list[str] = []
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if data.get("binding_id") != BINDING_ID:
        errors.append("binding_id_mismatch")

    status_value = data.get("status")
    if status_value not in {"BLOCKED_TEMPLATE", "READY_CANDIDATE"}:
        errors.append("status_invalid")
    template = status_value == "BLOCKED_TEMPLATE"

    try:
        identities = _exact_keys(data.get("identities"), _IDENTITY_KEYS, "identities")
    except CleanBuildError as exc:
        errors.append(str(exc))
        identities = {}

    for key in sorted(_IDENTITY_KEYS):
        value = identities.get(key)
        valid = _is_git_sha(value) if key == "source_git_sha" else _is_sha256(value)
        if template:
            if value is not None and not valid:
                errors.append(f"{key}_invalid")
        elif not valid:
            errors.append(f"{key}_invalid")

    try:
        boundary = _exact_keys(data.get("truth_boundary"), _TRUTH_KEYS, "truth_boundary")
    except CleanBuildError as exc:
        errors.append(str(exc))
        boundary = {}

    if boundary.get("execution_class") != "LOCAL_FREE":
        errors.append("execution_class_must_be_local_free")
    if boundary.get("clean_build_count") != 2:
        errors.append("clean_build_count_must_be_two")
    if boundary.get("shared_mutable_cache_allowed") is not False:
        errors.append("shared_mutable_cache_must_be_false")
    if boundary.get("semantic_packing_validation_separate") is not True:
        errors.append("semantic_packing_validation_must_remain_separate")
    for key in (
        "tokenizer_fit_executed_by_verifier",
        "final_test_accessed",
        "model_training_executed",
        "paid_compute_used",
        "unique_loss_ledger_authorized_by_verifier",
    ):
        if boundary.get(key) is not False:
            errors.append(f"truth_boundary_{key}_must_be_false")

    allowed_root = {
        "schema_version",
        "binding_id",
        "status",
        "identities",
        "truth_boundary",
    }
    missing_root = allowed_root - set(data)
    unexpected_root = set(data) - allowed_root
    errors.extend(f"root_{key}_missing" for key in sorted(missing_root))
    errors.extend(f"root_{key}_unexpected" for key in sorted(unexpected_root))
    return sorted(set(errors))


def _regular_file_bytes(path: Path) -> bytes:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        raise CleanBuildError(f"symlink_forbidden:{path.name}")
    if not stat.S_ISREG(mode):
        raise CleanBuildError(f"non_regular_file_forbidden:{path.name}")
    return path.read_bytes()


def scan_build_root(root: Path) -> BuildTree:
    """Hash every regular file and reject namespace ambiguity or empty roots."""
    if root.is_symlink():
        raise CleanBuildError("root_symlink_forbidden")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise CleanBuildError("build_root_must_be_directory")

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(resolved, followlinks=False):
        current = Path(dirpath)
        for dirname in sorted(dirnames):
            candidate = current / dirname
            if candidate.is_symlink():
                rel = candidate.relative_to(resolved).as_posix()
                raise CleanBuildError(f"symlink_forbidden:{rel}")
        for filename in sorted(filenames):
            path = current / filename
            rel = path.relative_to(resolved).as_posix()
            data = _regular_file_bytes(path)
            entries.append(
                {
                    "path": rel,
                    "bytes": len(data),
                    "sha256": sha256_bytes(data),
                }
            )
            total_bytes += len(data)

    entries.sort(key=lambda row: row["path"])
    if not entries:
        raise CleanBuildError("build_root_empty")
    if total_bytes <= 0:
        raise CleanBuildError("build_root_zero_bytes")

    tree_payload = {
        "entries": entries,
        "file_count": len(entries),
        "total_bytes": total_bytes,
    }
    return BuildTree(
        entries=tuple(entries),
        total_bytes=total_bytes,
        tree_sha256=sha256_bytes(canonical_json_bytes(tree_payload)),
    )


def _build_report(binding: dict[str, Any], tree: BuildTree) -> dict[str, Any]:
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS_BYTE_IDENTICAL_TWO_CLEAN_BUILDS",
        "binding_id": binding["binding_id"],
        "identities": dict(binding["identities"]),
        "file_count": len(tree.entries),
        "total_bytes": tree.total_bytes,
        "entries": [dict(row) for row in tree.entries],
        "build_tree_sha256": tree.tree_sha256,
        "two_clean_builds_identical": True,
        "truth_boundary": dict(binding["truth_boundary"]),
        "report_sha256": None,
    }
    hash_input = dict(report)
    hash_input.pop("report_sha256")
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(hash_input))
    return report


def compare_clean_builds(
    binding: dict[str, Any], root_a: Path, root_b: Path
) -> CleanBuildAssessment:
    """Compare two clean roots and retain only root-independent evidence."""
    binding_errors = validate_binding(binding)
    if binding_errors:
        return CleanBuildAssessment(False, tuple(binding_errors), None)
    if binding.get("status") != "READY_CANDIDATE":
        return CleanBuildAssessment(False, ("binding_status_not_ready_candidate",), None)

    try:
        resolved_a = root_a.resolve(strict=True)
        resolved_b = root_b.resolve(strict=True)
        if resolved_a == resolved_b:
            raise CleanBuildError("clean_build_roots_must_be_distinct")
        if resolved_a in resolved_b.parents or resolved_b in resolved_a.parents:
            raise CleanBuildError("clean_build_roots_must_not_be_nested")
        tree_a = scan_build_root(root_a)
        tree_b = scan_build_root(root_b)
    except (CleanBuildError, FileNotFoundError) as exc:
        return CleanBuildAssessment(False, (str(exc),), None)

    if tree_a.entries != tree_b.entries:
        paths_a = {row["path"] for row in tree_a.entries}
        paths_b = {row["path"] for row in tree_b.entries}
        blocker = (
            "build_path_set_mismatch" if paths_a != paths_b else "build_content_mismatch"
        )
        return CleanBuildAssessment(False, (blocker,), None)
    if tree_a.tree_sha256 != tree_b.tree_sha256:
        return CleanBuildAssessment(False, ("build_tree_identity_mismatch",), None)

    return CleanBuildAssessment(True, (), _build_report(binding, tree_a))


def _validate_report_entries(entries: Any, errors: list[str]) -> list[dict[str, Any]]:
    if not isinstance(entries, list) or not entries:
        errors.append("report_entries_missing")
        return []

    valid_rows: list[dict[str, Any]] = []
    for index, row in enumerate(entries):
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            errors.append(f"report_entry_{index}_invalid")
            continue
        path = row.get("path")
        byte_count = row.get("bytes")
        digest = row.get("sha256")
        if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
            errors.append(f"report_entry_{index}_path_invalid")
            continue
        if any(part in {"", ".", ".."} for part in path.split("/")):
            errors.append(f"report_entry_{index}_path_invalid")
            continue
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            errors.append(f"report_entry_{index}_bytes_invalid")
            continue
        if not _is_sha256(digest):
            errors.append(f"report_entry_{index}_sha256_invalid")
            continue
        valid_rows.append(row)
    return valid_rows


def validate_report(report: Any) -> list[str]:
    """Revalidate a retained report without needing the original build roots."""
    if not isinstance(report, dict):
        return ["report_root_must_be_object"]
    errors: list[str] = []
    if report.get("schema_version") != REPORT_SCHEMA:
        errors.append("report_schema_version_mismatch")
    if report.get("status") != "PASS_BYTE_IDENTICAL_TWO_CLEAN_BUILDS":
        errors.append("report_status_not_pass")
    if report.get("binding_id") != BINDING_ID:
        errors.append("report_binding_id_mismatch")
    if report.get("two_clean_builds_identical") is not True:
        errors.append("report_two_clean_builds_not_identical")
    if not _is_sha256(report.get("build_tree_sha256")):
        errors.append("report_build_tree_sha256_invalid")
    if not _is_sha256(report.get("report_sha256")):
        errors.append("report_sha256_invalid")

    entries = report.get("entries")
    valid_rows = _validate_report_entries(entries, errors)
    if valid_rows and isinstance(entries, list) and len(valid_rows) == len(entries):
        if report.get("file_count") != len(entries):
            errors.append("report_file_count_mismatch")
        if entries != sorted(entries, key=lambda row: row["path"]):
            errors.append("report_entries_not_sorted")
        paths = [row["path"] for row in entries]
        if len(paths) != len(set(paths)):
            errors.append("report_entry_paths_not_unique")
        if report.get("total_bytes") != sum(row["bytes"] for row in entries):
            errors.append("report_total_bytes_mismatch")
        tree_payload = {
            "entries": entries,
            "file_count": report.get("file_count"),
            "total_bytes": report.get("total_bytes"),
        }
        expected_tree_sha = sha256_bytes(canonical_json_bytes(tree_payload))
        if report.get("build_tree_sha256") != expected_tree_sha:
            errors.append("report_build_tree_identity_mismatch")

    reconstructed_binding = {
        "schema_version": SCHEMA_VERSION,
        "binding_id": report.get("binding_id"),
        "status": "READY_CANDIDATE",
        "identities": report.get("identities"),
        "truth_boundary": report.get("truth_boundary"),
    }
    errors.extend(validate_binding(reconstructed_binding))

    report_sha = report.get("report_sha256")
    if _is_sha256(report_sha):
        payload = dict(report)
        payload.pop("report_sha256", None)
        if report_sha != sha256_bytes(canonical_json_bytes(payload)):
            errors.append("report_self_hash_mismatch")

    return sorted(set(errors))
