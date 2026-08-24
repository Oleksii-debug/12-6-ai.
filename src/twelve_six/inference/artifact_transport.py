"""Fail-closed transport manifest for retained S0 inference artifacts.

The retained checkpoint builder validates bytes before GitHub Actions upload. This
module adds a narrower boundary: bind the exact extracted byte tree so a fresh
consumer can prove that the downloaded artifact still contains the same checkpoint
and generation evidence. It delegates checkpoint and generation semantics to the
existing D05/D07 validators.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.checkpoint import sha256_file, verify_checkpoint

from .s0_artifact import validate_s0_generation_artifact

SCHEMA_VERSION = "12-6.s0-artifact-transport.v1"
AUTHORITY = "ARTIFACT_BYTE_TRANSPORT_EVIDENCE_NOT_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
CHECKPOINT_RELATIVE_PATH = "checkpoint"
EVIDENCE_RELATIVE_PATH = "s0-generation-evidence.json"
REQUIRED_CHECKPOINT_FILES = (
    "MANIFEST.json",
    "MANIFEST.sha256",
    "state.json",
    "state.safetensors",
    "weights.safetensors",
)
_HEX = frozenset("0123456789abcdef")


class S0ArtifactTransportError(ValueError):
    """Raised when a transported S0 artifact fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S0ArtifactTransportError(message)


def _is_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and set(value) <= _HEX
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _HEX
    )


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _validate_relative_path(value: str) -> str:
    path = Path(value)
    _require(value == path.as_posix(), f"non-canonical artifact path: {value!r}")
    _require(not path.is_absolute(), f"absolute artifact path rejected: {value!r}")
    _require(value not in {"", "."}, "empty artifact path rejected")
    _require(".." not in path.parts, f"parent traversal rejected: {value!r}")
    return value


def _inventory(root: Path) -> dict[str, list[dict[str, Any]] | list[str]]:
    _require(root.exists(), f"artifact payload root does not exist: {root}")
    _require(not root.is_symlink(), "artifact payload root must not be a symlink")
    _require(root.is_dir(), "artifact payload root must be a directory")

    files: list[dict[str, Any]] = []
    directories: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current = Path(dirpath)
        for name in sorted(dirnames):
            path = current / name
            _require(not path.is_symlink(), f"artifact directory symlink rejected: {path}")
            mode = path.lstat().st_mode
            _require(stat.S_ISDIR(mode), f"non-directory artifact node rejected: {path}")
            relative = _validate_relative_path(path.relative_to(root).as_posix())
            directories.append(relative)
        for name in sorted(filenames):
            path = current / name
            _require(not path.is_symlink(), f"artifact file symlink rejected: {path}")
            status = path.lstat()
            _require(stat.S_ISREG(status.st_mode), f"non-regular artifact file rejected: {path}")
            relative = _validate_relative_path(path.relative_to(root).as_posix())
            files.append(
                {
                    "path": relative,
                    "bytes": status.st_size,
                    "sha256": sha256_file(path),
                }
            )
    directories.sort()
    files.sort(key=lambda item: str(item["path"]))
    _require(bool(files), "artifact payload must contain at least one file")
    return {"directories": directories, "files": files}


def _required_paths() -> tuple[str, ...]:
    checkpoint_paths = tuple(
        f"{CHECKPOINT_RELATIVE_PATH}/{name}" for name in REQUIRED_CHECKPOINT_FILES
    )
    return (*checkpoint_paths, EVIDENCE_RELATIVE_PATH)


def _validate_inventory_shape(inventory: object) -> Mapping[str, Any]:
    _require(isinstance(inventory, Mapping), "artifact inventory must be a mapping")
    _require(set(inventory) == {"directories", "files"}, "artifact inventory schema drift")
    directories = inventory.get("directories")
    files = inventory.get("files")
    _require(isinstance(directories, list), "artifact directories must be a list")
    _require(isinstance(files, list) and files, "artifact files must be a non-empty list")
    _require(directories == sorted(directories), "artifact directories must be sorted")
    for value in directories:
        _require(isinstance(value, str), "artifact directory path must be text")
        _validate_relative_path(value)
    file_paths: list[str] = []
    for item in files:
        _require(isinstance(item, Mapping), "artifact file record must be a mapping")
        _require(set(item) == {"path", "bytes", "sha256"}, "artifact file schema drift")
        path = item.get("path")
        _require(isinstance(path, str), "artifact file path must be text")
        _validate_relative_path(path)
        size = item.get("bytes")
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            "artifact file byte count must be a non-negative integer",
        )
        _require(_is_sha256(item.get("sha256")), "artifact file SHA-256 is invalid")
        file_paths.append(path)
    _require(file_paths == sorted(file_paths), "artifact files must be sorted")
    _require(len(file_paths) == len(set(file_paths)), "duplicate artifact file path")
    for required in _required_paths():
        _require(required in file_paths, f"required artifact file missing: {required}")
    return inventory


def build_s0_artifact_transport_manifest(
    payload_root: str | Path,
    *,
    source_sha: str,
) -> dict[str, Any]:
    """Bind one validated retained S0 payload before transport."""

    _require(_is_git_sha(source_sha), "source_sha must be a full lowercase 40-hex Git SHA")
    root = Path(payload_root).resolve()
    checkpoint_path = root / CHECKPOINT_RELATIVE_PATH
    evidence_path = root / EVIDENCE_RELATIVE_PATH
    _require(not checkpoint_path.is_symlink(), "checkpoint path must not be a symlink")
    _require(not evidence_path.is_symlink(), "generation evidence path must not be a symlink")

    checkpoint = verify_checkpoint(checkpoint_path)
    _require(
        checkpoint["identity"]["git_sha"] == source_sha,
        "checkpoint source SHA does not match transport source",
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    _require(isinstance(evidence, Mapping), "generation evidence must be a JSON object")
    validate_s0_generation_artifact(evidence, checkpoint_path=checkpoint_path)
    _require(evidence.get("candidate_sha") == source_sha, "evidence source SHA mismatch")
    evidence_hash = sha256_file(evidence_path)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "checkpoint": {
            "relative_path": CHECKPOINT_RELATIVE_PATH,
            "checkpoint_id": checkpoint["checkpoint_id"],
            "git_sha": checkpoint["identity"]["git_sha"],
        },
        "generation_evidence": {
            "relative_path": EVIDENCE_RELATIVE_PATH,
            "file_sha256": evidence_hash,
            "evidence_sha256": evidence.get("evidence_sha256"),
        },
        "inventory": _inventory(root),
        "transport_claims": {
            "file_bytes_and_sizes_bound": True,
            "directory_topology_bound": True,
            "posix_mode_preservation_claimed": False,
            "artifact_archive_digest_equivalence_claimed": False,
            "promotion_authority": False,
        },
    }
    _validate_inventory_shape(payload["inventory"])
    _require(
        _is_sha256(payload["generation_evidence"]["evidence_sha256"]),
        "generation evidence self-hash is invalid",
    )
    payload["manifest_sha256"] = _canonical_hash(payload)
    return payload


def validate_s0_artifact_transport_manifest(
    payload_root: str | Path,
    manifest: Mapping[str, Any],
    *,
    expected_source_sha: str | None = None,
) -> dict[str, Any]:
    """Validate extracted artifact bytes in a fresh consumer environment."""

    report = dict(manifest)
    _require(report.get("schema_version") == SCHEMA_VERSION, "artifact transport schema mismatch")
    _require(report.get("authority") == AUTHORITY, "artifact transport authority mismatch")
    _require(report.get("repository") == REPOSITORY, "artifact transport repository mismatch")
    source_sha = report.get("source_sha")
    _require(_is_git_sha(source_sha), "artifact transport source SHA is invalid")
    if expected_source_sha is not None:
        _require(_is_git_sha(expected_source_sha), "expected source SHA is invalid")
        _require(source_sha == expected_source_sha, "artifact transport source SHA mismatch")

    stored_hash = report.pop("manifest_sha256", None)
    _require(_is_sha256(stored_hash), "artifact transport manifest SHA-256 is invalid")
    _require(stored_hash == _canonical_hash(report), "artifact transport manifest self-hash mismatch")
    report["manifest_sha256"] = stored_hash

    expected_inventory = _validate_inventory_shape(report.get("inventory"))
    root = Path(payload_root).resolve()
    actual_inventory = _inventory(root)
    _require(actual_inventory == expected_inventory, "downloaded artifact byte inventory mismatch")

    checkpoint_record = report.get("checkpoint")
    _require(isinstance(checkpoint_record, Mapping), "checkpoint transport binding missing")
    _require(
        set(checkpoint_record) == {"relative_path", "checkpoint_id", "git_sha"},
        "checkpoint transport schema drift",
    )
    _require(
        checkpoint_record.get("relative_path") == CHECKPOINT_RELATIVE_PATH,
        "checkpoint relative path drift",
    )
    _require(_is_sha256(checkpoint_record.get("checkpoint_id")), "checkpoint_id is invalid")
    _require(checkpoint_record.get("git_sha") == source_sha, "checkpoint Git binding mismatch")

    checkpoint_path = root / CHECKPOINT_RELATIVE_PATH
    checkpoint = verify_checkpoint(checkpoint_path)
    _require(
        checkpoint["checkpoint_id"] == checkpoint_record["checkpoint_id"],
        "downloaded checkpoint_id differs from producer binding",
    )
    _require(
        checkpoint["identity"]["git_sha"] == source_sha,
        "downloaded checkpoint Git SHA differs from transport binding",
    )

    evidence_record = report.get("generation_evidence")
    _require(isinstance(evidence_record, Mapping), "generation evidence transport binding missing")
    _require(
        set(evidence_record) == {"relative_path", "file_sha256", "evidence_sha256"},
        "generation evidence transport schema drift",
    )
    _require(
        evidence_record.get("relative_path") == EVIDENCE_RELATIVE_PATH,
        "generation evidence relative path drift",
    )
    _require(_is_sha256(evidence_record.get("file_sha256")), "generation evidence file hash invalid")
    _require(_is_sha256(evidence_record.get("evidence_sha256")), "generation evidence self-hash invalid")
    evidence_path = root / EVIDENCE_RELATIVE_PATH
    _require(
        sha256_file(evidence_path) == evidence_record["file_sha256"],
        "downloaded generation evidence file hash mismatch",
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    _require(isinstance(evidence, Mapping), "downloaded generation evidence must be an object")
    _require(
        evidence.get("evidence_sha256") == evidence_record["evidence_sha256"],
        "downloaded generation evidence self-hash binding mismatch",
    )
    validate_s0_generation_artifact(evidence, checkpoint_path=checkpoint_path)
    _require(evidence.get("candidate_sha") == source_sha, "downloaded generation evidence source drift")

    claims = report.get("transport_claims")
    _require(isinstance(claims, Mapping), "transport claims missing")
    _require(
        claims
        == {
            "file_bytes_and_sizes_bound": True,
            "directory_topology_bound": True,
            "posix_mode_preservation_claimed": False,
            "artifact_archive_digest_equivalence_claimed": False,
            "promotion_authority": False,
        },
        "artifact transport truth boundary drift",
    )
    return report
