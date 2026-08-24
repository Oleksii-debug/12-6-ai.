"""Conservative Hugging Face-style directory export for verified 12-6 checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import (
    MANIFEST_NAME,
    WEIGHTS_NAME,
    CheckpointIntegrityError,
    _read_regular_bytes,
    canonical_json_bytes,
    prepare_checkpoint_load,
    sha256_bytes,
)

EXPORT_ATTESTATION_NAME = "12-6-export.json"
PARITY_REQUEST_NAME = "12-6-parity-request.json"
EXPORTED_MANIFEST_NAME = "12-6-checkpoint-manifest.json"
EXPORTED_WEIGHTS_NAME = "model.safetensors"
EXPORTED_CONFIG_NAME = "config.json"
_EXPORT_NAMES = frozenset(
    {
        EXPORTED_WEIGHTS_NAME,
        EXPORTED_CONFIG_NAME,
        EXPORTED_MANIFEST_NAME,
        EXPORT_ATTESTATION_NAME,
        PARITY_REQUEST_NAME,
    }
)
ParityHook = Callable[[Path, Path], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class VerifiedHFStyleExport:
    """Immutable byte snapshot of one verified HF-style export."""

    source_manifest: Mapping[str, Any]
    config: Mapping[str, Any]
    attestation: Mapping[str, Any]
    parity_request: Mapping[str, Any]
    weights_bytes: bytes
    source_manifest_bytes: bytes


def _parse_json_object(data: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(f"{name} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"{name} must contain a JSON object")
    return value


def _require_export_directory(root: Path) -> None:
    try:
        root_stat = root.lstat()
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(f"HF-style export does not exist: {root}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise CheckpointIntegrityError("HF-style export root must be a real directory")
    names = {entry.name for entry in root.iterdir()}
    if names != _EXPORT_NAMES:
        missing = sorted(_EXPORT_NAMES - names)
        unexpected = sorted(names - _EXPORT_NAMES)
        raise CheckpointIntegrityError(
            f"HF-style export inventory mismatch: missing={missing}, unexpected={unexpected}"
        )


def verify_hf_directory(
    checkpoint_dir: str | Path,
    export_dir: str | Path,
) -> VerifiedHFStyleExport:
    """Verify an export against one immutable snapshot of its canonical checkpoint.

    The source checkpoint is read through ``prepare_checkpoint_load`` and the
    exported files are each read exactly once through the same regular-file,
    no-symlink primitive used by checkpoint-v1. The returned weight bytes are the
    exact bytes that were verified; runtime parity code can decode this snapshot
    without reopening a mutable pathname.
    """

    source = prepare_checkpoint_load(checkpoint_dir)
    source_manifest = source.manifest
    source_manifest_bytes = source._manifest_bytes
    source_weights_bytes = source._artifacts[WEIGHTS_NAME]

    root = Path(export_dir)
    _require_export_directory(root)
    files = {name: _read_regular_bytes(root, name) for name in sorted(_EXPORT_NAMES)}

    if files[EXPORTED_MANIFEST_NAME] != source_manifest_bytes:
        raise CheckpointIntegrityError(
            "exported checkpoint manifest is not the exact verified source manifest"
        )
    if files[EXPORTED_WEIGHTS_NAME] != source_weights_bytes:
        raise CheckpointIntegrityError(
            "exported model.safetensors is not the exact verified source weight bytes"
        )

    config = _parse_json_object(files[EXPORTED_CONFIG_NAME], name=EXPORTED_CONFIG_NAME)
    attestation = _parse_json_object(
        files[EXPORT_ATTESTATION_NAME], name=EXPORT_ATTESTATION_NAME
    )
    parity_request = _parse_json_object(files[PARITY_REQUEST_NAME], name=PARITY_REQUEST_NAME)

    source_manifest_sha = sha256_bytes(source_manifest_bytes)
    weights_sha = sha256_bytes(source_weights_bytes)
    config_sha = sha256_bytes(files[EXPORTED_CONFIG_NAME])

    expected_attestation = {
        "schema": "12-6.hf-style-export.v1",
        "checkpoint_id": source_manifest["checkpoint_id"],
        "source_manifest_sha256": source_manifest_sha,
        "model_safetensors_sha256": weights_sha,
        "config_sha256": config_sha,
        "compatibility": {
            "layout": "HF_STYLE_SAFETENSORS_DIRECTORY",
            "weights": "EXACT_CANONICAL_BYTE_COPY",
            "transformers_architecture": "NOT_CLAIMED",
            "runtime_logit_generation_parity": "NOT_TESTED",
        },
    }
    if attestation != expected_attestation:
        raise CheckpointIntegrityError("HF-style export attestation does not match verified bytes")

    if parity_request.get("schema") != "12-6.export-parity-request.v1":
        raise CheckpointIntegrityError("unsupported HF-style parity request schema")
    expected_fields = {
        "checkpoint_id": source_manifest["checkpoint_id"],
        "reference_weights_sha256": weights_sha,
        "candidate_weights_sha256": weights_sha,
        "candidate_config_sha256": config_sha,
        "required_checks": [
            "prompt_token_identity",
            "next_token_logit_parity",
            "greedy_generation_parity",
        ],
        "authority": "D07_or_independent_parity_harness",
    }
    for field, expected in expected_fields.items():
        if parity_request.get(field) != expected:
            raise CheckpointIntegrityError(f"HF-style parity request field mismatch: {field}")
    status = parity_request.get("status")
    hook_result = parity_request.get("hook_result")
    if status == "NOT_TESTED":
        if hook_result is not None:
            raise CheckpointIntegrityError("NOT_TESTED parity request cannot contain hook evidence")
    elif status == "EXTERNAL_EVIDENCE_ATTACHED":
        if not isinstance(hook_result, Mapping):
            raise CheckpointIntegrityError("attached parity hook evidence must be an object")
    else:
        raise CheckpointIntegrityError("unsupported HF-style parity request status")

    return VerifiedHFStyleExport(
        source_manifest=source_manifest,
        config=config,
        attestation=attestation,
        parity_request=parity_request,
        weights_bytes=files[EXPORTED_WEIGHTS_NAME],
        source_manifest_bytes=files[EXPORTED_MANIFEST_NAME],
    )


def export_hf_directory(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    hf_config: Mapping[str, Any],
    overwrite: bool = False,
    parity_hook: ParityHook | None = None,
) -> Path:
    """Create an immutable, verified HF-style SafeTensors/config layout.

    Guarantees:
    - one verified checkpoint byte snapshot is the sole export source;
    - ``model.safetensors`` and copied manifest are written from those verified bytes;
    - the full directory is staged and verified before an atomic publish;
    - an existing export is never destructively removed, including with
      ``overwrite=True``. Callers must publish a new content/version path instead.

    Non-guarantees are equally explicit: an HF-style directory is *not* a claim
    that ``transformers.AutoModel`` can instantiate the 12-6 architecture. Runtime
    logit/generation parity remains ``NOT_TESTED`` in the export attestation; a
    D07/independent harness may attach separate parity evidence.
    """

    source_path = Path(checkpoint_dir)
    verified = prepare_checkpoint_load(source_path)
    source_manifest = verified.manifest
    source_manifest_bytes = verified._manifest_bytes
    source_weights_bytes = verified._artifacts[WEIGHTS_NAME]

    destination = Path(output_dir)
    if destination.exists():
        mode = "overwrite=True" if overwrite else "overwrite=False"
        raise FileExistsError(
            "HF-style export destinations are immutable; existing destination "
            f"cannot be replaced ({mode}): {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )

    try:
        exported_weights = staging / EXPORTED_WEIGHTS_NAME
        exported_config = staging / EXPORTED_CONFIG_NAME
        exported_source_manifest = staging / EXPORTED_MANIFEST_NAME

        exported_weights.write_bytes(source_weights_bytes)
        exported_config.write_bytes(canonical_json_bytes(dict(hf_config)) + b"\n")
        exported_source_manifest.write_bytes(source_manifest_bytes)

        source_weights_sha = sha256_bytes(source_weights_bytes)
        exported_weights_sha = sha256_bytes(exported_weights.read_bytes())
        if exported_weights_sha != source_weights_sha:
            raise RuntimeError("HF-style export weight bytes changed during staging")

        attestation = {
            "schema": "12-6.hf-style-export.v1",
            "checkpoint_id": source_manifest["checkpoint_id"],
            "source_manifest_sha256": sha256_bytes(source_manifest_bytes),
            "model_safetensors_sha256": exported_weights_sha,
            "config_sha256": sha256_bytes(exported_config.read_bytes()),
            "compatibility": {
                "layout": "HF_STYLE_SAFETENSORS_DIRECTORY",
                "weights": "EXACT_CANONICAL_BYTE_COPY",
                "transformers_architecture": "NOT_CLAIMED",
                "runtime_logit_generation_parity": "NOT_TESTED",
            },
        }
        (staging / EXPORT_ATTESTATION_NAME).write_bytes(
            canonical_json_bytes(attestation) + b"\n"
        )

        parity_request: dict[str, Any] = {
            "schema": "12-6.export-parity-request.v1",
            "status": "NOT_TESTED",
            "checkpoint_id": source_manifest["checkpoint_id"],
            "reference_weights_sha256": source_weights_sha,
            "candidate_weights_sha256": exported_weights_sha,
            "candidate_config_sha256": attestation["config_sha256"],
            "required_checks": [
                "prompt_token_identity",
                "next_token_logit_parity",
                "greedy_generation_parity",
            ],
            "authority": "D07_or_independent_parity_harness",
            "hook_result": None,
        }
        if parity_hook is not None:
            result = parity_hook(source_path, staging)
            if not isinstance(result, Mapping):
                raise TypeError("parity_hook must return a mapping")
            parity_request["hook_result"] = dict(result)
            parity_request["status"] = "EXTERNAL_EVIDENCE_ATTACHED"

        (staging / PARITY_REQUEST_NAME).write_bytes(
            canonical_json_bytes(parity_request) + b"\n"
        )

        verify_hf_directory(source_path, staging)
        os.replace(staging, destination)
        return destination
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise
