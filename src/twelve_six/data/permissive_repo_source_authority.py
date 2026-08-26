from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_PARTS = {"test", "tests", "vendor", "vendored", "third_party", "extern", "external"}
_EXPECTED_UPSTREAM = {
    "repository": "scipy/scipy",
    "tag": "v1.18.0",
    "annotated_tag_sha": "7adb8c972443f664b9395a0e6e8e0283e9b4faff",
    "commit_sha": "54ef5423f2e4376230ec3bfda6912a07a50958e3",
    "tree_sha": "efc619e51d474a41624a09f4b73241b43dc1fc2a",
}
_REQUIRED_GATES = {
    "requires_materialization",
    "requires_materialized_license_scan",
    "requires_normalization",
    "requires_global_cross_source_dedup",
    "requires_balance_diversity_retest",
    "requires_decontamination_before_training",
}


class SourceAuthorityError(ValueError):
    """Raised when a source-authority document is not safe to consume."""


def authority_identity(document: dict[str, Any]) -> str:
    payload = dict(document)
    payload.pop("authority_sha256", None)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_source_authority(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != 1:
        raise SourceAuthorityError("unsupported schema_version")
    if document.get("authority_id") != "scipy-v1.18.0-bounded-first-party-v1":
        raise SourceAuthorityError("unexpected authority_id")
    if document.get("source_family") != "code.scipy.project":
        raise SourceAuthorityError("unexpected source_family")

    claimed_identity = document.get("authority_sha256")
    actual_identity = authority_identity(document)
    if claimed_identity != actual_identity:
        raise SourceAuthorityError("authority_sha256 mismatch")

    if document.get("upstream") != _EXPECTED_UPSTREAM:
        raise SourceAuthorityError("upstream pin drift")

    license_info = document.get("license")
    if license_info != {
        "root_spdx": "BSD-3-Clause",
        "root_path": "LICENSE.txt",
        "bundled_license_path": "LICENSES_bundled.txt",
        "whole_repository_credit_forbidden": True,
    }:
        raise SourceAuthorityError("license boundary drift")

    purpose = document.get("purpose")
    if purpose != {"training_allowed": True, "evaluation_allowed": False}:
        raise SourceAuthorityError("purpose boundary drift")

    gates = document.get("gates")
    if not isinstance(gates, dict) or set(gates) != _REQUIRED_GATES:
        raise SourceAuthorityError("gate set drift")
    if not all(gates.values()):
        raise SourceAuthorityError("all downstream gates must remain fail-closed")

    allowlist = document.get("allowlist")
    if not isinstance(allowlist, list) or not allowlist:
        raise SourceAuthorityError("allowlist must be non-empty")

    commit_sha = _EXPECTED_UPSTREAM["commit_sha"]
    seen_paths: set[str] = set()
    candidate_bytes = 0
    for entry in allowlist:
        if not isinstance(entry, dict):
            raise SourceAuthorityError("allowlist entry must be an object")
        if set(entry) != {"path", "git_blob_sha1", "raw_bytes", "raw_url"}:
            raise SourceAuthorityError("allowlist entry schema drift")

        path = entry["path"]
        if not isinstance(path, str):
            raise SourceAuthorityError("allowlist path must be a string")
        posix = PurePosixPath(path)
        if posix.is_absolute() or ".." in posix.parts:
            raise SourceAuthorityError(f"unsafe allowlist path: {path}")
        if len(posix.parts) < 3 or posix.parts[:2] != ("scipy", "optimize"):
            raise SourceAuthorityError(f"path outside bounded scipy/optimize scope: {path}")
        if posix.suffix != ".py":
            raise SourceAuthorityError(f"non-Python path is not allowed: {path}")
        if any(part.lower() in _FORBIDDEN_PARTS for part in posix.parts):
            raise SourceAuthorityError(f"forbidden provenance path: {path}")
        if path in seen_paths:
            raise SourceAuthorityError(f"duplicate allowlist path: {path}")
        seen_paths.add(path)

        blob_sha = entry["git_blob_sha1"]
        if not isinstance(blob_sha, str) or not _HEX40.fullmatch(blob_sha):
            raise SourceAuthorityError(f"invalid Git blob SHA-1 for {path}")
        raw_bytes = entry["raw_bytes"]
        if not isinstance(raw_bytes, int) or isinstance(raw_bytes, bool) or raw_bytes <= 0:
            raise SourceAuthorityError(f"invalid raw byte count for {path}")
        expected_url = f"https://raw.githubusercontent.com/scipy/scipy/{commit_sha}/{path}"
        if entry["raw_url"] != expected_url:
            raise SourceAuthorityError(f"raw URL is not exact-commit pinned for {path}")
        candidate_bytes += raw_bytes

    capacity = document.get("capacity")
    if not isinstance(capacity, dict):
        raise SourceAuthorityError("capacity must be an object")
    if capacity.get("candidate_raw_bytes") != candidate_bytes:
        raise SourceAuthorityError("candidate byte arithmetic drift")
    if capacity.get("canonical_credit_bytes") != 0:
        raise SourceAuthorityError("canonical credit must remain zero pre-materialization")
    if capacity.get("basis") != "raw_bytes_pre_normalization_pre_global_dedup":
        raise SourceAuthorityError("capacity basis drift")
    if capacity.get("materialization_status") != "not_materialized":
        raise SourceAuthorityError("authority cannot claim materialization")

    return {
        "authority_id": document["authority_id"],
        "authority_sha256": actual_identity,
        "files": len(allowlist),
        "candidate_raw_bytes": candidate_bytes,
        "canonical_credit_bytes": 0,
        "ready_for_corpus_credit": False,
    }


def load_and_validate_source_authority(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SourceAuthorityError("authority root must be an object")
    return validate_source_authority(document)
