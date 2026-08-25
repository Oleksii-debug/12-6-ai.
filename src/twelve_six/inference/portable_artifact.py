"""Fail-closed validation for retained S0 inference artifact bundles.

The retained-evidence producer may run on Linux, but downstream consumers should
not trust an uploaded artifact merely because GitHub transported it. This module
recomputes the bundle manifest, rejects unsafe/untracked paths, verifies the D05
checkpoint, and cross-binds inference/CLI/server evidence to one exact source SHA.

It intentionally does not claim Windows/NVDA compatibility. The current D08 lock
set has Linux profiles only, so Windows execution remains blocked until a
hash-locked Windows runtime profile exists.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from twelve_six.checkpoint import CheckpointError, hash_json, sha256_file, verify_checkpoint
from twelve_six.training.s0_evidence_contract import (
    S0EvidenceContractError,
    validate_locked_environment_evidence,
)

ARTIFACT_SCHEMA = "12-6.s0-retained-inference-artifact.v1"
VALIDATION_SCHEMA = "12-6.s0-portable-inference-validation.v1"
INFERENCE_EVIDENCE_SCHEMA = "12-6.s0-retained-inference-evidence.v1"
REPOSITORY = "Oleksii-debug/12-6-ai."
WINDOWS_RUNTIME_STATUS = "BLOCKED_BY_MISSING_HASH_LOCKED_WINDOWS_RUNTIME"
WINDOWS_NVDA_STATUS = "NOT_TESTED"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_PATHS = frozenset(
    {
        "locked-environment-linux-x86_64.json",
        "runtime/inference_evidence.json",
        "runtime/checkpoint/manifest.json",
        "runtime/checkpoint/MANIFEST.sha256",
        "runtime/checkpoint/weights.safetensors",
        "runtime/checkpoint/state.safetensors",
        "runtime/checkpoint/state.json",
        "cli-prompt.json",
        "cli-stdin.json",
        "server-response.json",
    }
)


class PortableArtifactError(RuntimeError):
    """Raised when a retained inference bundle cannot be trusted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PortableArtifactError(message)


def _load_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PortableArtifactError(f"{field} is not valid UTF-8 JSON: {exc}") from exc
    _require(isinstance(value, dict), f"{field} must contain a JSON object")
    return value


def _safe_relative_path(value: Any) -> str:
    _require(isinstance(value, str) and bool(value), "artifact file path must be non-empty text")
    _require("\\" not in value, f"artifact path must use POSIX separators: {value!r}")
    path = PurePosixPath(value)
    _require(not path.is_absolute(), f"artifact path must be relative: {value!r}")
    _require(
        all(part not in {"", ".", ".."} for part in path.parts),
        f"artifact path contains unsafe traversal: {value!r}",
    )
    normalized = path.as_posix()
    _require(normalized == value, f"artifact path is not canonically normalized: {value!r}")
    return normalized


def _validate_sha256(value: Any, *, field: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{field} must be a lowercase SHA-256",
    )
    return value


def _validate_source_sha(value: Any) -> str:
    _require(
        isinstance(value, str) and _GIT_SHA.fullmatch(value) is not None,
        "artifact source_sha must be a full lowercase Git SHA",
    )
    return value


def _inventory_files(root: Path) -> set[str]:
    _require(root.exists(), "artifact root does not exist")
    _require(root.is_dir() and not root.is_symlink(), "artifact root must be a real directory")
    files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise PortableArtifactError(f"artifact bundle contains symlink: {relative}")
        if path.is_dir():
            continue
        _require(path.is_file(), f"artifact bundle contains non-regular entry: {relative}")
        files.add(relative)
    return files


def validate_portable_artifact_manifest(
    artifact_root: str | Path,
    *,
    expected_source_sha: str | None = None,
) -> dict[str, Any]:
    """Validate manifest structure plus every bundle byte without trusting filenames."""

    root = Path(artifact_root).resolve()
    manifest_path = root / "artifact-manifest.json"
    manifest = _load_object(manifest_path, field="artifact manifest")

    _require(manifest.get("schema") == ARTIFACT_SCHEMA, "wrong portable artifact schema")
    source_sha = _validate_source_sha(manifest.get("source_sha"))
    if expected_source_sha is not None:
        _require(source_sha == expected_source_sha, "portable artifact source SHA mismatch")
    _require(manifest.get("promotion_claim") is False, "portable artifact must not self-promote")
    _require(
        manifest.get("windows_nvda_live_pass") is False,
        "portable artifact cannot assert Windows/NVDA PASS before platform execution",
    )

    claimed_manifest_hash = _validate_sha256(
        manifest.get("manifest_sha256"),
        field="artifact manifest manifest_sha256",
    )
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    _require(
        hash_json(unhashed) == claimed_manifest_hash,
        "portable artifact manifest self-hash mismatch",
    )

    raw_files = manifest.get("files")
    _require(
        isinstance(raw_files, list) and bool(raw_files),
        "artifact manifest files must be non-empty",
    )
    records: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(raw_files):
        _require(isinstance(raw_record, Mapping), f"artifact file record {index} must be an object")
        path = _safe_relative_path(raw_record.get("path"))
        _require(
            path != "artifact-manifest.json",
            "artifact manifest must not recursively list itself",
        )
        _require(path not in records, f"duplicate artifact path: {path}")
        size = raw_record.get("bytes")
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            f"artifact file {path} bytes must be a non-negative integer",
        )
        digest = _validate_sha256(raw_record.get("sha256"), field=f"artifact file {path} sha256")
        records[path] = {"path": path, "bytes": size, "sha256": digest}

    _require(
        _REQUIRED_PATHS <= records.keys(),
        "portable artifact is missing required runtime files",
    )
    wheel_paths = sorted(
        path for path in records if path.startswith("dist/") and path.endswith(".whl")
    )
    _require(len(wheel_paths) == 1, "portable artifact must contain exactly one project wheel")

    actual_files = _inventory_files(root)
    expected_files = set(records) | {"artifact-manifest.json"}
    missing = sorted(expected_files - actual_files)
    extra = sorted(actual_files - expected_files)
    _require(not missing, f"portable artifact is missing manifested files: {missing}")
    _require(not extra, f"portable artifact contains unmanifested files: {extra}")

    for path, record in records.items():
        physical = root.joinpath(*PurePosixPath(path).parts)
        _require(
            physical.is_file() and not physical.is_symlink(),
            f"artifact file is not regular: {path}",
        )
        _require(physical.stat().st_size == record["bytes"], f"artifact file size mismatch: {path}")
        _require(
            sha256_file(physical) == record["sha256"],
            f"artifact file checksum mismatch: {path}",
        )

    wheel = records[wheel_paths[0]]
    return {
        "source_sha": source_sha,
        "artifact_manifest_sha256": claimed_manifest_hash,
        "file_count": len(records),
        "wheel": dict(wheel),
        "records": records,
    }


def _validate_inference_report(
    report: Mapping[str, Any],
    *,
    source_sha: str,
    checkpoint_id: str,
) -> str:
    _require(report.get("schema") == INFERENCE_EVIDENCE_SCHEMA, "wrong retained inference schema")
    _require(report.get("status") == "PASS", "retained inference evidence is not PASS")

    claimed_hash = _validate_sha256(report.get("report_sha256"), field="inference report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    _require(hash_json(unhashed) == claimed_hash, "retained inference report self-hash mismatch")

    candidate = report.get("candidate")
    _require(isinstance(candidate, Mapping), "retained inference candidate block missing")
    _require(candidate.get("repository") == REPOSITORY, "retained inference repository mismatch")
    _require(candidate.get("sha") == source_sha, "retained inference source SHA mismatch")
    _require(candidate.get("canonical_base") == "random_init", "retained Base is not random_init")
    _require(candidate.get("pretraining_only") is True, "retained Base is not pretraining-only")
    _require(
        candidate.get("foreign_pretrained_weights") is False,
        "retained evidence reports foreign pretrained weights",
    )
    _require(
        candidate.get("behavioral_alignment_weights") is False,
        "retained evidence reports behavioral/alignment weights",
    )

    checkpoint = report.get("checkpoint")
    _require(isinstance(checkpoint, Mapping), "retained inference checkpoint block missing")
    _require(checkpoint.get("checkpoint_id") == checkpoint_id, "inference/checkpoint ID mismatch")
    _require(
        checkpoint.get("serialization_pickle") is False,
        "portable checkpoint must remain pickle-free",
    )
    _require(
        checkpoint.get("corrupt_checkpoint_rejected") is True,
        "retained evidence did not prove corrupt-checkpoint rejection",
    )

    inference = report.get("inference")
    _require(isinstance(inference, Mapping), "retained inference block missing")
    parity = inference.get("parity")
    _require(
        isinstance(parity, Mapping) and parity.get("passed") is True,
        "retained parity is not PASS",
    )
    _require(parity.get("max_abs_error") == 0.0, "retained parity is not exact")
    _require(parity.get("max_rel_error") == 0.0, "retained parity is not exact")
    _require(
        inference.get("openai_compatible_raw_completion_equal") is True,
        "raw completion response diverged from canonical generation",
    )
    _require(
        inference.get("chat_semantics") is False,
        "retained artifact must remain raw Base completion",
    )

    artifact = report.get("artifact")
    _require(isinstance(artifact, Mapping), "retained artifact block missing")
    _require(
        artifact.get("retained_for_external_execution") is True,
        "retained checkpoint is not marked for external execution",
    )
    _require(
        artifact.get("windows_nvda_live_pass") is False,
        "Windows/NVDA PASS is not yet authorized",
    )

    boundary = report.get("truth_boundary")
    _require(isinstance(boundary, Mapping), "retained truth boundary missing")
    _require(boundary.get("paid_compute") is False, "retained evidence reports paid compute")
    _require(
        boundary.get("candidate_or_stable_promotion") is False,
        "retained evidence attempts candidate/STABLE promotion",
    )
    _require(
        boundary.get("windows_nvda_live_execution") == WINDOWS_NVDA_STATUS,
        "retained evidence overclaims Windows/NVDA execution",
    )
    return claimed_hash


def _validate_cli_payload(
    payload: Mapping[str, Any],
    *,
    source_sha: str,
    checkpoint_id: str,
    field: str,
) -> None:
    backend = payload.get("backend")
    _require(isinstance(backend, Mapping), f"{field} backend diagnostics missing")
    _require(backend.get("backend") == "first_party_torch", f"{field} used wrong backend")
    _require(backend.get("git_sha") == source_sha, f"{field} source SHA mismatch")
    _require(backend.get("checkpoint_id") == checkpoint_id, f"{field} checkpoint ID mismatch")
    _require(payload.get("mode") == "greedy", f"{field} must exercise deterministic greedy mode")


def validate_portable_runtime_artifact(
    artifact_root: str | Path,
    *,
    expected_source_sha: str | None = None,
) -> dict[str, Any]:
    """Deep-validate one retained artifact and emit a Windows handoff contract."""

    root = Path(artifact_root).resolve()
    manifest_result = validate_portable_artifact_manifest(
        root,
        expected_source_sha=expected_source_sha,
    )
    source_sha = manifest_result["source_sha"]

    try:
        checkpoint_manifest = verify_checkpoint(root / "runtime/checkpoint")
    except CheckpointError as exc:
        raise PortableArtifactError(f"portable checkpoint verification failed: {exc}") from exc
    checkpoint_id = checkpoint_manifest.get("checkpoint_id")
    _require(isinstance(checkpoint_id, str) and bool(checkpoint_id), "checkpoint ID is missing")
    identity = checkpoint_manifest.get("identity")
    _require(isinstance(identity, Mapping), "checkpoint identity block missing")
    _require(identity.get("git_sha") == source_sha, "checkpoint source SHA mismatch")

    inference_report = _load_object(
        root / "runtime/inference_evidence.json",
        field="retained inference evidence",
    )
    inference_report_sha256 = _validate_inference_report(
        inference_report,
        source_sha=source_sha,
        checkpoint_id=checkpoint_id,
    )

    locked_environment = _load_object(
        root / "locked-environment-linux-x86_64.json",
        field="locked environment evidence",
    )
    try:
        environment_binding = validate_locked_environment_evidence(
            locked_environment,
            source_sha=source_sha,
        )
    except S0EvidenceContractError as exc:
        raise PortableArtifactError(
            f"locked environment evidence failed validation: {exc}"
        ) from exc

    for filename in ("cli-prompt.json", "cli-stdin.json"):
        payload = _load_object(root / filename, field=filename)
        _validate_cli_payload(
            payload,
            source_sha=source_sha,
            checkpoint_id=checkpoint_id,
            field=filename,
        )

    server = _load_object(root / "server-response.json", field="server response")
    _require(
        server.get("object") == "text_completion",
        "server response is not a text completion",
    )
    _require(server.get("model") == "12-6-base-s0", "server response model identity mismatch")
    choices = server.get("choices")
    _require(isinstance(choices, list) and len(choices) == 1, "server response choices are invalid")

    report: dict[str, Any] = {
        "schema": VALIDATION_SCHEMA,
        "status": "PASS",
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "artifact_manifest_sha256": manifest_result["artifact_manifest_sha256"],
        "artifact_file_count": manifest_result["file_count"],
        "wheel": manifest_result["wheel"],
        "checkpoint_id": checkpoint_id,
        "inference_report_sha256": inference_report_sha256,
        "locked_environment": environment_binding,
        "windows_handoff": {
            "artifact_only": True,
            "repository_checkout_required": False,
            "required_dependency_profile": "windows-x86_64",
            "hash_locked_windows_profile_available": False,
            "runtime_status": WINDOWS_RUNTIME_STATUS,
            "nvda_status": WINDOWS_NVDA_STATUS,
        },
        "authority": {
            "local_free_or_free_hosted_only": True,
            "promotion_claim": False,
            "audit_verdict": False,
        },
    }
    report["validation_sha256"] = hash_json(report)
    return report
