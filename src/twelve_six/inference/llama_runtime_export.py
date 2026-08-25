"""Materialize a verified 12-6 export as a standard local Llama model directory.

The D05 HF-style export deliberately preserves canonical 12-6 tensor names/bytes.
This D07 layer consumes that verified source, applies the proven tensor-name/RoPE
basis conversion, and writes a separate standard Llama directory suitable for
maintained Transformers and, after independent runtime acceptance, vLLM.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from safetensors.torch import load as load_safetensors_bytes
from safetensors.torch import save_file

from twelve_six.checkpoint import verify_hf_directory
from twelve_six.checkpoint.hf_export import (
    EXPORTED_CONFIG_NAME,
    EXPORTED_SOURCE_MANIFEST_NAME,
    EXPORTED_WEIGHTS_NAME,
)
from twelve_six.model import ModelSpec

from .transformers_llama import convert_state_dict_to_llama, llama_config_dict

RUNTIME_EXPORT_SCHEMA = "12-6.standard-llama-runtime-export.v1"
RUNTIME_WEIGHTS_NAME = "model.safetensors"
RUNTIME_CONFIG_NAME = "config.json"
RUNTIME_PROVENANCE_NAME = "12-6-runtime-provenance.json"
_RUNTIME_FILES = frozenset(
    {RUNTIME_WEIGHTS_NAME, RUNTIME_CONFIG_NAME, RUNTIME_PROVENANCE_NAME}
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_object(data: bytes, *, artifact: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{artifact} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{artifact} must contain a JSON object")
    return value


def _read_exact_files(directory: Path) -> dict[str, bytes]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("standard Llama runtime export must be a real directory")
    names = {entry.name for entry in directory.iterdir()}
    if names != _RUNTIME_FILES:
        raise ValueError(
            "standard Llama runtime export inventory mismatch: "
            f"missing={sorted(_RUNTIME_FILES - names)}, "
            f"unexpected={sorted(names - _RUNTIME_FILES)}"
        )
    payloads: dict[str, bytes] = {}
    for name in sorted(_RUNTIME_FILES):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"runtime export payload must be regular non-symlink file: {name}")
        payloads[name] = path.read_bytes()
    return payloads


def materialize_standard_llama_directory(
    source_export_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Convert verified canonical exported bytes into standard Llama tensor layout."""

    source = Path(source_export_dir)
    source_attestation = verify_hf_directory(source)
    source_weights = (source / EXPORTED_WEIGHTS_NAME).read_bytes()
    source_config = (source / EXPORTED_CONFIG_NAME).read_bytes()
    source_manifest_bytes = (source / EXPORTED_SOURCE_MANIFEST_NAME).read_bytes()

    if _sha256(source_weights) != source_attestation.get("model_safetensors_sha256"):
        raise ValueError("source export weights changed after verification")
    if _sha256(source_config) != source_attestation.get("config_sha256"):
        raise ValueError("source export config changed after verification")
    if _sha256(source_manifest_bytes) != source_attestation.get("source_manifest_sha256"):
        raise ValueError("source export manifest changed after verification")

    source_manifest = _json_object(
        source_manifest_bytes,
        artifact=EXPORTED_SOURCE_MANIFEST_NAME,
    )
    identity = source_manifest.get("identity")
    if not isinstance(identity, dict) or not isinstance(identity.get("model_spec"), dict):
        raise ValueError("source export manifest is missing ModelSpec")
    spec = ModelSpec.from_dict(identity["model_spec"])
    if spec.identity_sha256() != identity.get("model_spec_hash"):
        raise ValueError("source export ModelSpec hash mismatch")

    expected_config = llama_config_dict(spec)
    actual_config = _json_object(source_config, artifact=EXPORTED_CONFIG_NAME)
    if actual_config != expected_config:
        raise ValueError("source export config is not the exact D07 Llama mapping")

    try:
        source_state = load_safetensors_bytes(source_weights)
    except Exception as exc:
        raise ValueError("source exported safetensors bytes cannot be decoded") from exc
    converted = convert_state_dict_to_llama(spec, source_state)

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"runtime export destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        config_bytes = _canonical_json_bytes(expected_config)
        (staging / RUNTIME_CONFIG_NAME).write_bytes(config_bytes)
        save_file(converted, staging / RUNTIME_WEIGHTS_NAME)
        runtime_weights = (staging / RUNTIME_WEIGHTS_NAME).read_bytes()
        provenance: dict[str, Any] = {
            "schema": RUNTIME_EXPORT_SCHEMA,
            "source_checkpoint_id": source_attestation["checkpoint_id"],
            "source_manifest_sha256": _sha256(source_manifest_bytes),
            "source_weights_sha256": _sha256(source_weights),
            "source_config_sha256": _sha256(source_config),
            "model_spec_sha256": spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "target_architecture": "LlamaForCausalLM",
            "runtime_weights_sha256": _sha256(runtime_weights),
            "runtime_config_sha256": _sha256(config_bytes),
            "rope_transform": "PAIRWISE_INTERLEAVED_TO_LLAMA_HALF_SPLIT",
            "foreign_pretrained_weights": False,
            "model_downloaded": False,
            "raw_base_only": True,
            "runtime_compatibility_requires_execution_evidence": True,
            "promotion_authority": False,
        }
        provenance_bytes = _canonical_json_bytes(provenance)
        (staging / RUNTIME_PROVENANCE_NAME).write_bytes(provenance_bytes)
        verify_standard_llama_directory(staging)
        staging.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_standard_llama_directory(directory: str | Path) -> dict[str, Any]:
    """Verify local standard-Llama bytes and their source-bound provenance."""

    payloads = _read_exact_files(Path(directory))
    config = _json_object(payloads[RUNTIME_CONFIG_NAME], artifact=RUNTIME_CONFIG_NAME)
    provenance = _json_object(
        payloads[RUNTIME_PROVENANCE_NAME],
        artifact=RUNTIME_PROVENANCE_NAME,
    )
    if provenance.get("schema") != RUNTIME_EXPORT_SCHEMA:
        raise ValueError("unsupported standard Llama runtime export schema")
    required_false = ("foreign_pretrained_weights", "model_downloaded", "promotion_authority")
    if any(provenance.get(field) is not False for field in required_false):
        raise ValueError("runtime export provenance contains a prohibited authority/weight claim")
    if provenance.get("raw_base_only") is not True:
        raise ValueError("runtime export provenance weakened raw-Base boundary")
    if provenance.get("target_architecture") != "LlamaForCausalLM":
        raise ValueError("runtime export target architecture mismatch")
    if provenance.get("runtime_weights_sha256") != _sha256(payloads[RUNTIME_WEIGHTS_NAME]):
        raise ValueError("runtime export weights hash mismatch")
    if provenance.get("runtime_config_sha256") != _sha256(payloads[RUNTIME_CONFIG_NAME]):
        raise ValueError("runtime export config hash mismatch")
    if config.get("architectures") != ["LlamaForCausalLM"]:
        raise ValueError("runtime export config architecture mismatch")
    if any(config.get(field) is not None for field in ("bos_token_id", "eos_token_id", "pad_token_id")):
        raise ValueError("runtime export config invented special-token semantics")
    return provenance
