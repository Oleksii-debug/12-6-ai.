"""Conservative, transaction-safe Hugging Face-style export for verified checkpoints."""

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
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST_NAME,
    WEIGHTS_NAME,
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    canonical_json_bytes,
    hash_json,
    prepare_checkpoint_load,
    sha256_bytes,
)

EXPORT_ATTESTATION_NAME = "12-6-export.json"
EXPORT_CHECKSUM_NAME = "12-6-export.sha256"
PARITY_REQUEST_NAME = "12-6-parity-request.json"
EXPORTED_WEIGHTS_NAME = "model.safetensors"
EXPORTED_CONFIG_NAME = "config.json"
EXPORTED_SOURCE_MANIFEST_NAME = "12-6-checkpoint-manifest.json"
_EXPORT_FILES = frozenset(
    {
        EXPORTED_WEIGHTS_NAME,
        EXPORTED_CONFIG_NAME,
        EXPORTED_SOURCE_MANIFEST_NAME,
        EXPORT_ATTESTATION_NAME,
        EXPORT_CHECKSUM_NAME,
        PARITY_REQUEST_NAME,
    }
)
_REQUIRED_PARITY_CHECKS = [
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
ParityHook = Callable[[Path, Path], Mapping[str, Any]]


def _read_regular_bytes(root: Path, name: str) -> bytes:
    path = root / name
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(f"missing HF-style export artifact: {name}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CheckpointIntegrityError(
            f"HF-style export artifact must be a regular non-symlink file: {name}"
        )

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CheckpointIntegrityError(
            f"cannot safely open HF-style export artifact: {name}"
        ) from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise CheckpointIntegrityError(f"HF-style export artifact changed type: {name}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise CheckpointIntegrityError(
                f"HF-style export artifact changed while opening: {name}"
            )
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(fd)


def _read_export_snapshot(root: Path) -> dict[str, bytes]:
    try:
        root_stat = root.lstat()
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(f"HF-style export directory does not exist: {root}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise CheckpointIntegrityError(
            "HF-style export root must be a real directory, not a symlink"
        )
    names = {entry.name for entry in root.iterdir()}
    if names != _EXPORT_FILES:
        missing = sorted(_EXPORT_FILES - names)
        unexpected = sorted(names - _EXPORT_FILES)
        raise CheckpointIntegrityError(
            f"HF-style export inventory mismatch: missing={missing}, unexpected={unexpected}"
        )
    return {name: _read_regular_bytes(root, name) for name in sorted(_EXPORT_FILES)}


def _json_object(data: bytes, *, artifact: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(f"{artifact} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"{artifact} must contain a JSON object")
    return value


def verify_hf_directory(directory: str | Path) -> dict[str, Any]:
    """Verify one exact HF-style export directory without trusting path metadata."""

    payloads = _read_export_snapshot(Path(directory))
    try:
        checksum_parts = payloads[EXPORT_CHECKSUM_NAME].decode("ascii").strip().split()
    except UnicodeDecodeError as exc:
        raise CheckpointIntegrityError(f"{EXPORT_CHECKSUM_NAME} must be ASCII") from exc
    if len(checksum_parts) != 2 or checksum_parts[1] != EXPORT_ATTESTATION_NAME:
        raise CheckpointIntegrityError(f"invalid {EXPORT_CHECKSUM_NAME} format")
    if checksum_parts[0] != sha256_bytes(payloads[EXPORT_ATTESTATION_NAME]):
        raise CheckpointIntegrityError("HF-style export attestation checksum mismatch")

    source_manifest = _json_object(
        payloads[EXPORTED_SOURCE_MANIFEST_NAME],
        artifact=EXPORTED_SOURCE_MANIFEST_NAME,
    )
    if (
        source_manifest.get("format") != FORMAT_NAME
        or source_manifest.get("format_version") != FORMAT_VERSION
    ):
        raise CheckpointCompatibilityError(
            "exported source manifest has unsupported checkpoint format"
        )
    identity = source_manifest.get("identity")
    files = source_manifest.get("files")
    if not isinstance(identity, dict) or not isinstance(files, dict):
        raise CheckpointIntegrityError(
            "exported source manifest is missing identity/files mappings"
        )
    checkpoint_id = hash_json({"identity": identity, "files": files})
    if source_manifest.get("checkpoint_id") != checkpoint_id:
        raise CheckpointIntegrityError(
            "exported source manifest checkpoint_id is self-inconsistent"
        )
    weights_record = files.get(WEIGHTS_NAME)
    if not isinstance(weights_record, dict):
        raise CheckpointIntegrityError("source manifest is missing canonical weights record")

    weights_sha = sha256_bytes(payloads[EXPORTED_WEIGHTS_NAME])
    config_sha = sha256_bytes(payloads[EXPORTED_CONFIG_NAME])
    source_manifest_sha = sha256_bytes(payloads[EXPORTED_SOURCE_MANIFEST_NAME])
    parity_sha = sha256_bytes(payloads[PARITY_REQUEST_NAME])
    if weights_record.get("sha256") != weights_sha:
        raise CheckpointIntegrityError(
            "exported model.safetensors differs from canonical weights hash"
        )
    if weights_record.get("bytes") != len(payloads[EXPORTED_WEIGHTS_NAME]):
        raise CheckpointIntegrityError(
            "exported model.safetensors differs from canonical byte length"
        )

    parity = _json_object(payloads[PARITY_REQUEST_NAME], artifact=PARITY_REQUEST_NAME)
    if parity.get("schema") != "12-6.export-parity-request.v2":
        raise CheckpointCompatibilityError("unsupported export parity request schema")
    expected_parity = {
        "checkpoint_id": checkpoint_id,
        "reference_weights_sha256": weights_sha,
        "candidate_weights_sha256": weights_sha,
        "candidate_config_sha256": config_sha,
        "required_checks": _REQUIRED_PARITY_CHECKS,
        "authority": "D07_or_independent_parity_harness",
    }
    for field, expected in expected_parity.items():
        if parity.get(field) != expected:
            raise CheckpointIntegrityError(f"export parity request {field} mismatch")
    status = parity.get("status")
    hook_result = parity.get("hook_result")
    if status == "NOT_TESTED":
        if hook_result is not None:
            raise CheckpointIntegrityError("NOT_TESTED parity request cannot attach hook evidence")
    elif status == "EXTERNAL_EVIDENCE_ATTACHED":
        if not isinstance(hook_result, dict):
            raise CheckpointIntegrityError(
                "EXTERNAL_EVIDENCE_ATTACHED parity request requires mapping evidence"
            )
    else:
        raise CheckpointIntegrityError(f"unsupported export parity status: {status!r}")

    attestation = _json_object(
        payloads[EXPORT_ATTESTATION_NAME], artifact=EXPORT_ATTESTATION_NAME
    )
    if attestation.get("schema") != "12-6.hf-style-export.v2":
        raise CheckpointCompatibilityError("unsupported HF-style export attestation schema")
    if attestation.get("compatibility") != _COMPATIBILITY:
        raise CheckpointIntegrityError("HF-style export compatibility claims changed unexpectedly")
    expected_attestation = {
        "checkpoint_id": checkpoint_id,
        "source_manifest_sha256": source_manifest_sha,
        "model_safetensors_sha256": weights_sha,
        "config_sha256": config_sha,
        "parity_request_sha256": parity_sha,
    }
    for field, expected in expected_attestation.items():
        if attestation.get(field) != expected:
            raise CheckpointIntegrityError(f"HF-style export attestation {field} mismatch")
    return attestation


def export_hf_directory(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    hf_config: Mapping[str, Any],
    overwrite: bool = False,
    parity_hook: ParityHook | None = None,
) -> Path:
    """Create an immutable, verified HF-style SafeTensors directory.

    Source checkpoint bytes are snapshotted and verified once through D05's
    transactional loader, then the export is built from that exact in-memory
    snapshot. The complete export is verified in a sibling staging directory
    before a same-filesystem rename publishes it.

    Existing destinations are immutable. ``overwrite=True`` is retained only for
    API compatibility and still fails closed rather than deleting prior evidence.

    The output is HF-*style* only. It does not claim Transformers architecture
    compatibility or runtime logit/generation parity. An optional external parity
    hook may attach evidence while those compatibility claims remain unchanged.
    """

    source = Path(checkpoint_dir)
    verified = prepare_checkpoint_load(source)
    source_manifest = verified.manifest
    source_manifest_bytes = verified._manifest_bytes
    source_weights_bytes = verified._artifacts[WEIGHTS_NAME]

    destination = Path(output_dir)
    if destination.exists():
        suffix = " (overwrite=True does not permit destructive replacement)" if overwrite else ""
        raise FileExistsError(f"export destination already exists: {destination}{suffix}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        config_bytes = canonical_json_bytes(dict(hf_config)) + b"\n"
        (staging / EXPORTED_WEIGHTS_NAME).write_bytes(source_weights_bytes)
        (staging / EXPORTED_CONFIG_NAME).write_bytes(config_bytes)
        (staging / EXPORTED_SOURCE_MANIFEST_NAME).write_bytes(source_manifest_bytes)

        weights_sha = sha256_bytes(source_weights_bytes)
        config_sha = sha256_bytes(config_bytes)
        parity_request: dict[str, Any] = {
            "schema": "12-6.export-parity-request.v2",
            "status": "NOT_TESTED",
            "checkpoint_id": source_manifest["checkpoint_id"],
            "reference_weights_sha256": weights_sha,
            "candidate_weights_sha256": weights_sha,
            "candidate_config_sha256": config_sha,
            "required_checks": list(_REQUIRED_PARITY_CHECKS),
            "authority": "D07_or_independent_parity_harness",
            "hook_result": None,
        }
        if parity_hook is not None:
            result = parity_hook(source, staging)
            if not isinstance(result, Mapping):
                raise TypeError("parity_hook must return a mapping")
            parity_request["hook_result"] = dict(result)
            parity_request["status"] = "EXTERNAL_EVIDENCE_ATTACHED"
        parity_bytes = canonical_json_bytes(parity_request) + b"\n"
        (staging / PARITY_REQUEST_NAME).write_bytes(parity_bytes)

        attestation = {
            "schema": "12-6.hf-style-export.v2",
            "checkpoint_id": source_manifest["checkpoint_id"],
            "source_manifest_sha256": sha256_bytes(source_manifest_bytes),
            "model_safetensors_sha256": weights_sha,
            "config_sha256": config_sha,
            "parity_request_sha256": sha256_bytes(parity_bytes),
            "compatibility": dict(_COMPATIBILITY),
        }
        attestation_bytes = canonical_json_bytes(attestation) + b"\n"
        (staging / EXPORT_ATTESTATION_NAME).write_bytes(attestation_bytes)
        (staging / EXPORT_CHECKSUM_NAME).write_text(
            f"{sha256_bytes(attestation_bytes)}  {EXPORT_ATTESTATION_NAME}\n",
            encoding="ascii",
        )

        verify_hf_directory(staging)
        if destination.exists():
            raise FileExistsError(
                f"export destination appeared during publish: {destination}"
            )
        os.rename(staging, destination)
        return destination
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
