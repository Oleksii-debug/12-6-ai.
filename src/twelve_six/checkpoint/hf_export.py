"""Conservative Hugging Face-style directory export for verified 12-6 checkpoints."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .core import (
    MANIFEST_NAME,
    WEIGHTS_NAME,
    canonical_json_bytes,
    sha256_file,
    verify_checkpoint,
)

EXPORT_ATTESTATION_NAME = "12-6-export.json"
PARITY_REQUEST_NAME = "12-6-parity-request.json"
ParityHook = Callable[[Path, Path], Mapping[str, Any]]


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
    - the source checkpoint is verified before export;
    - ``model.safetensors`` is an exact byte copy of canonical checkpoint weights;
    - config and provenance hashes are emitted in a machine-readable attestation.

    Non-guarantees are equally explicit: an HF-style directory is *not* a claim
    that ``transformers.AutoModel`` can instantiate the 12-6 architecture. Runtime
    logit/generation parity remains ``NOT_TESTED`` unless an external D07-owned
    parity hook is supplied. D05 records that hook result but does not promote it
    into architecture compatibility authority.
    """

    source = Path(checkpoint_dir)
    source_manifest = verify_checkpoint(source)
    destination = Path(output_dir)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"export destination already exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    exported_weights = destination / "model.safetensors"
    exported_config = destination / "config.json"
    exported_source_manifest = destination / "12-6-checkpoint-manifest.json"

    shutil.copy2(source / WEIGHTS_NAME, exported_weights)
    exported_config.write_bytes(canonical_json_bytes(dict(hf_config)) + b"\n")
    shutil.copy2(source / MANIFEST_NAME, exported_source_manifest)

    source_weights_sha = sha256_file(source / WEIGHTS_NAME)
    exported_weights_sha = sha256_file(exported_weights)
    if exported_weights_sha != source_weights_sha:
        raise RuntimeError("HF-style export weight copy changed canonical SafeTensors bytes")

    attestation = {
        "schema": "12-6.hf-style-export.v1",
        "checkpoint_id": source_manifest["checkpoint_id"],
        "source_manifest_sha256": sha256_file(source / MANIFEST_NAME),
        "model_safetensors_sha256": exported_weights_sha,
        "config_sha256": sha256_file(exported_config),
        "compatibility": {
            "layout": "HF_STYLE_SAFETENSORS_DIRECTORY",
            "weights": "EXACT_CANONICAL_BYTE_COPY",
            "transformers_architecture": "NOT_CLAIMED",
            "runtime_logit_generation_parity": "NOT_TESTED",
        },
    }
    (destination / EXPORT_ATTESTATION_NAME).write_bytes(
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
        result = parity_hook(source, destination)
        if not isinstance(result, Mapping):
            raise TypeError("parity_hook must return a mapping")
        parity_request["hook_result"] = dict(result)
        parity_request["status"] = "EXTERNAL_EVIDENCE_ATTACHED"

    (destination / PARITY_REQUEST_NAME).write_bytes(
        canonical_json_bytes(parity_request) + b"\n"
    )
    return destination
