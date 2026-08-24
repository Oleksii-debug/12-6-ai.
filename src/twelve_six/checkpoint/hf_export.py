"""Conservative Hugging Face-style directory export for verified 12-6 checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .core import (
    MANIFEST_NAME,
    WEIGHTS_NAME,
    CheckpointIntegrityError,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    verify_checkpoint,
)

EXPORT_ATTESTATION_NAME = "12-6-export.json"
PARITY_REQUEST_NAME = "12-6-parity-request.json"
EXPORTED_WEIGHTS_NAME = "model.safetensors"
EXPORTED_CONFIG_NAME = "config.json"
EXPORTED_SOURCE_MANIFEST_NAME = "12-6-checkpoint-manifest.json"
ParityHook = Callable[[Path, Path], Mapping[str, Any]]

_BASE_EXPORT_NAMES = frozenset(
    {
        EXPORTED_WEIGHTS_NAME,
        EXPORTED_CONFIG_NAME,
        EXPORTED_SOURCE_MANIFEST_NAME,
    }
)
_EXPORT_NAMES = frozenset(
    {
        *_BASE_EXPORT_NAMES,
        EXPORT_ATTESTATION_NAME,
        PARITY_REQUEST_NAME,
    }
)
_REQUIRED_CHECKS = [
    "prompt_token_identity",
    "next_token_logit_parity",
    "greedy_generation_parity",
]
_COMPATIBILITY = {
    "layout": "HF_STYLE_SAFETENSORS_DIRECTORY",
    "weights": "EXACT_CANONICAL_BYTE_COPY",
    "transformers_architecture": "NOT_CLAIMED",
    "runtime_logit_generation_parity": "NOT_TESTED",
}


def _require_directory(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(f"{label} directory is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise CheckpointIntegrityError(
            f"{label} must be a real directory, not a symlink or special file: {path}"
        )


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise CheckpointIntegrityError(
            f"{label} must be a regular file, not a symlink or special file: {path}"
        )


def _require_inventory(directory: Path, expected: frozenset[str], *, label: str) -> None:
    _require_directory(directory, label=label)
    observed = {entry.name for entry in directory.iterdir()}
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise CheckpointIntegrityError(
            f"{label} inventory mismatch: missing={missing}, extra={extra}"
        )
    for name in expected:
        _require_regular_file(directory / name, label=f"{label} payload {name}")


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    _require_regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"{label} must contain a JSON object")
    return value


def _artifact_record(path: Path) -> dict[str, Any]:
    _require_regular_file(path, label=f"export payload {path.name}")
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _verify_static_export_payloads(source: Path, export_dir: Path) -> dict[str, str]:
    _require_inventory(export_dir, _BASE_EXPORT_NAMES, label="staged HF-style export")
    source_manifest = verify_checkpoint(source)
    source_weights_sha = sha256_file(source / WEIGHTS_NAME)
    exported_weights_sha = sha256_file(export_dir / EXPORTED_WEIGHTS_NAME)
    if exported_weights_sha != source_weights_sha:
        raise CheckpointIntegrityError(
            "HF-style export weight copy changed canonical SafeTensors bytes"
        )
    source_manifest_sha = sha256_file(source / MANIFEST_NAME)
    exported_manifest_sha = sha256_file(export_dir / EXPORTED_SOURCE_MANIFEST_NAME)
    if exported_manifest_sha != source_manifest_sha:
        raise CheckpointIntegrityError(
            "HF-style export source-manifest copy changed canonical checkpoint provenance"
        )
    return {
        "checkpoint_id": source_manifest["checkpoint_id"],
        "source_weights_sha256": source_weights_sha,
        "source_manifest_sha256": source_manifest_sha,
        "config_sha256": sha256_file(export_dir / EXPORTED_CONFIG_NAME),
    }


def verify_hf_export(
    checkpoint_dir: str | Path,
    export_dir: str | Path,
) -> dict[str, Any]:
    """Verify a complete HF-style export against its canonical checkpoint.

    This verifier is intentionally first-party and conversion-focused. It proves
    exact canonical SafeTensors/provenance bytes plus internal export integrity;
    it does not claim that Transformers can instantiate or execute the model.
    """

    source = Path(checkpoint_dir)
    destination = Path(export_dir)
    _require_inventory(destination, _EXPORT_NAMES, label="HF-style export")
    source_manifest = verify_checkpoint(source)

    source_weights_sha = sha256_file(source / WEIGHTS_NAME)
    source_manifest_sha = sha256_file(source / MANIFEST_NAME)
    model_sha = sha256_file(destination / EXPORTED_WEIGHTS_NAME)
    config_sha = sha256_file(destination / EXPORTED_CONFIG_NAME)
    source_manifest_copy_sha = sha256_file(destination / EXPORTED_SOURCE_MANIFEST_NAME)
    parity_sha = sha256_file(destination / PARITY_REQUEST_NAME)

    if model_sha != source_weights_sha:
        raise CheckpointIntegrityError(
            "HF-style export model.safetensors no longer matches canonical weights"
        )
    if source_manifest_copy_sha != source_manifest_sha:
        raise CheckpointIntegrityError(
            "HF-style export checkpoint-manifest copy no longer matches canonical provenance"
        )

    parity = _load_json_mapping(
        destination / PARITY_REQUEST_NAME,
        label="HF-style export parity request",
    )
    expected_statuses = {"NOT_TESTED", "EXTERNAL_EVIDENCE_ATTACHED"}
    if parity.get("schema") != "12-6.export-parity-request.v1":
        raise CheckpointIntegrityError("HF-style export parity schema mismatch")
    if parity.get("status") not in expected_statuses:
        raise CheckpointIntegrityError("HF-style export parity status is invalid")
    if parity.get("checkpoint_id") != source_manifest["checkpoint_id"]:
        raise CheckpointIntegrityError("HF-style export parity checkpoint_id mismatch")
    if parity.get("reference_weights_sha256") != source_weights_sha:
        raise CheckpointIntegrityError("HF-style export parity reference weight hash mismatch")
    if parity.get("candidate_weights_sha256") != model_sha:
        raise CheckpointIntegrityError("HF-style export parity candidate weight hash mismatch")
    if parity.get("candidate_config_sha256") != config_sha:
        raise CheckpointIntegrityError("HF-style export parity config hash mismatch")
    if parity.get("required_checks") != _REQUIRED_CHECKS:
        raise CheckpointIntegrityError("HF-style export parity required-check set mismatch")
    if parity.get("authority") != "D07_or_independent_parity_harness":
        raise CheckpointIntegrityError("HF-style export parity authority mismatch")
    hook_result = parity.get("hook_result")
    if parity["status"] == "NOT_TESTED" and hook_result is not None:
        raise CheckpointIntegrityError("NOT_TESTED parity cannot carry hook_result evidence")
    if parity["status"] == "EXTERNAL_EVIDENCE_ATTACHED" and not isinstance(
        hook_result, dict
    ):
        raise CheckpointIntegrityError(
            "EXTERNAL_EVIDENCE_ATTACHED parity requires a JSON-object hook_result"
        )

    attestation = _load_json_mapping(
        destination / EXPORT_ATTESTATION_NAME,
        label="HF-style export attestation",
    )
    if attestation.get("schema") != "12-6.hf-style-export.v1":
        raise CheckpointIntegrityError("HF-style export attestation schema mismatch")
    if attestation.get("checkpoint_id") != source_manifest["checkpoint_id"]:
        raise CheckpointIntegrityError("HF-style export attestation checkpoint_id mismatch")
    if attestation.get("source_manifest_sha256") != source_manifest_sha:
        raise CheckpointIntegrityError("HF-style export source manifest hash mismatch")
    if attestation.get("model_safetensors_sha256") != model_sha:
        raise CheckpointIntegrityError("HF-style export model hash mismatch")
    if attestation.get("config_sha256") != config_sha:
        raise CheckpointIntegrityError("HF-style export config hash mismatch")
    if attestation.get("compatibility") != _COMPATIBILITY:
        raise CheckpointIntegrityError("HF-style export compatibility boundary mismatch")

    expected_files = {
        EXPORTED_WEIGHTS_NAME: _artifact_record(destination / EXPORTED_WEIGHTS_NAME),
        EXPORTED_CONFIG_NAME: _artifact_record(destination / EXPORTED_CONFIG_NAME),
        EXPORTED_SOURCE_MANIFEST_NAME: _artifact_record(
            destination / EXPORTED_SOURCE_MANIFEST_NAME
        ),
        PARITY_REQUEST_NAME: _artifact_record(destination / PARITY_REQUEST_NAME),
    }
    if attestation.get("files") != expected_files:
        raise CheckpointIntegrityError("HF-style export attested file inventory/hash mismatch")

    attestation_without_self_hash = dict(attestation)
    claimed_self_hash = attestation_without_self_hash.pop("attestation_sha256", None)
    expected_self_hash = sha256_bytes(canonical_json_bytes(attestation_without_self_hash))
    if claimed_self_hash != expected_self_hash:
        raise CheckpointIntegrityError("HF-style export attestation self-hash mismatch")
    if expected_files[PARITY_REQUEST_NAME]["sha256"] != parity_sha:
        raise CheckpointIntegrityError("HF-style export parity request hash mismatch")
    return attestation


def _publish_staged_export(staging: Path, destination: Path) -> None:
    """Publish a verified staging directory without deleting an existing artifact."""

    try:
        destination.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"export destination already exists and is immutable: {destination}"
        ) from exc

    publish_complete = False
    try:
        # The attestation is moved last and acts as the completion marker. A reader
        # that verifies the exact inventory will fail closed during this brief phase.
        publish_order = sorted(_EXPORT_NAMES - {EXPORT_ATTESTATION_NAME})
        for name in publish_order:
            os.replace(staging / name, destination / name)
        os.replace(
            staging / EXPORT_ATTESTATION_NAME,
            destination / EXPORT_ATTESTATION_NAME,
        )
        staging.rmdir()
        publish_complete = True
    finally:
        if not publish_complete:
            # destination was created by this call, so cleanup cannot destroy a prior
            # artifact. A process-level crash can still leave an incomplete directory;
            # later calls reject that directory rather than overwriting it.
            shutil.rmtree(destination, ignore_errors=True)


def export_hf_directory(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    hf_config: Mapping[str, Any],
    overwrite: bool = False,
    parity_hook: ParityHook | None = None,
) -> Path:
    """Create a verified HF-style single-file SafeTensors/config layout.

    Guarantees:
    - the source checkpoint is verified before and after external hook execution;
    - ``model.safetensors`` is an exact byte copy of canonical checkpoint weights;
    - config, provenance, parity-request and file hashes are attested;
    - the export is fully built and verified in staging before fail-closed publish;
    - an existing destination is immutable and is never recursively deleted.

    ``overwrite`` is retained for API compatibility only. Passing ``True`` does
    not authorize destructive replacement of an existing export; callers must
    choose a fresh destination.

    Non-guarantees are equally explicit: an HF-style directory is *not* a claim
    that ``transformers.AutoModel`` can instantiate the 12-6 architecture. Runtime
    logit/generation parity remains ``NOT_TESTED`` unless an external D07-owned
    parity hook is supplied. D05 records that hook result but does not promote it
    into architecture compatibility authority.
    """

    source = Path(checkpoint_dir)
    source_manifest = verify_checkpoint(source)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        overwrite_note = " (overwrite=True does not bypass immutability)" if overwrite else ""
        raise FileExistsError(
            f"export destination already exists and is immutable: {destination}"
            f"{overwrite_note}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    source_resolved = source.resolve(strict=True)
    destination_abs = destination.parent.resolve(strict=True) / destination.name
    if destination_abs == source_resolved or source_resolved in destination_abs.parents:
        raise ValueError("export destination cannot be the checkpoint or live inside it")

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    published = False
    final_verified = False
    try:
        exported_weights = staging / EXPORTED_WEIGHTS_NAME
        exported_config = staging / EXPORTED_CONFIG_NAME
        exported_source_manifest = staging / EXPORTED_SOURCE_MANIFEST_NAME

        shutil.copy2(source / WEIGHTS_NAME, exported_weights)
        exported_config.write_bytes(canonical_json_bytes(dict(hf_config)) + b"\n")
        shutil.copy2(source / MANIFEST_NAME, exported_source_manifest)

        static = _verify_static_export_payloads(source, staging)
        parity_request: dict[str, Any] = {
            "schema": "12-6.export-parity-request.v1",
            "status": "NOT_TESTED",
            "checkpoint_id": source_manifest["checkpoint_id"],
            "reference_weights_sha256": static["source_weights_sha256"],
            "candidate_weights_sha256": static["source_weights_sha256"],
            "candidate_config_sha256": static["config_sha256"],
            "required_checks": list(_REQUIRED_CHECKS),
            "authority": "D07_or_independent_parity_harness",
            "hook_result": None,
        }
        if parity_hook is not None:
            result = parity_hook(source, staging)
            if not isinstance(result, Mapping):
                raise TypeError("parity_hook must return a mapping")
            # A hook is evidence-producing, not trusted artifact-authoring code.
            # Re-verify source + staged immutable payloads after it returns.
            static_after_hook = _verify_static_export_payloads(source, staging)
            if static_after_hook != static:
                raise CheckpointIntegrityError(
                    "parity_hook changed source or staged export identity"
                )
            parity_request["hook_result"] = dict(result)
            parity_request["status"] = "EXTERNAL_EVIDENCE_ATTACHED"

        (staging / PARITY_REQUEST_NAME).write_bytes(
            canonical_json_bytes(parity_request) + b"\n"
        )
        files = {
            EXPORTED_WEIGHTS_NAME: _artifact_record(staging / EXPORTED_WEIGHTS_NAME),
            EXPORTED_CONFIG_NAME: _artifact_record(staging / EXPORTED_CONFIG_NAME),
            EXPORTED_SOURCE_MANIFEST_NAME: _artifact_record(
                staging / EXPORTED_SOURCE_MANIFEST_NAME
            ),
            PARITY_REQUEST_NAME: _artifact_record(staging / PARITY_REQUEST_NAME),
        }
        attestation: dict[str, Any] = {
            "schema": "12-6.hf-style-export.v1",
            "checkpoint_id": source_manifest["checkpoint_id"],
            "source_manifest_sha256": static["source_manifest_sha256"],
            "model_safetensors_sha256": static["source_weights_sha256"],
            "config_sha256": static["config_sha256"],
            "files": files,
            "compatibility": dict(_COMPATIBILITY),
        }
        attestation["attestation_sha256"] = sha256_bytes(
            canonical_json_bytes(attestation)
        )
        (staging / EXPORT_ATTESTATION_NAME).write_bytes(
            canonical_json_bytes(attestation) + b"\n"
        )

        verify_hf_export(source, staging)
        _publish_staged_export(staging, destination)
        published = True
        verify_hf_export(source, destination)
        final_verified = True
        return destination
    finally:
        if published and not final_verified:
            shutil.rmtree(destination, ignore_errors=True)
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging, ignore_errors=True)
