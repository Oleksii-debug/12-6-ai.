"""Plan-4 immutable SafeTensors + canonical S0 tokenizer transfer bundle.

This is a packaging adapter over the incumbent D05 checkpoint/HF exporter,
not a second weight format, model loader, or Transformers compatibility claim.
"""
from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from twelve_six.checkpoint import CheckpointIntegrityError, hf_export as hf
from twelve_six.tokenization.byte import (
    BYTE_TOKENIZER_HASH,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
    canonical_config_json,
    canonical_vocab_json,
)

SCHEMA = "12-6.plan4-safe-export.v1"
MANIFEST = "12-6-safe-bundle.json"
CHECKSUM = "12-6-safe-bundle.sha256"
TOKENIZER = "tokenizer.json"
VOCAB = "tokenizer-vocab.json"
HF_SUBDIR = "hf"
INVENTORY = frozenset({MANIFEST, CHECKSUM, TOKENIZER, VOCAB, HF_SUBDIR})


def _strict_root(root: Path) -> None:
    try:
        mode = root.lstat().st_mode
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError("safe bundle root missing") from exc
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise CheckpointIntegrityError("safe bundle root must be a real directory")
    actual = {item.name for item in root.iterdir()}
    if actual != INVENTORY:
        raise CheckpointIntegrityError("safe bundle unexpected or missing artifacts")
    hf_mode = (root / HF_SUBDIR).lstat().st_mode
    if not stat.S_ISDIR(hf_mode) or stat.S_ISLNK(hf_mode):
        raise CheckpointIntegrityError("safe bundle HF subtree must be a real directory")


def _tokenizer_files() -> tuple[bytes, bytes]:
    # Fails closed if current S0 implementation/config hashes have drifted.
    ByteTokenizer()
    return (
        canonical_config_json().encode("utf-8") + b"\n",
        canonical_vocab_json().encode("utf-8") + b"\n",
    )


def _bundle_claim(root: Path) -> dict[str, Any]:
    hf.verify_hf_directory(root / HF_SUBDIR)
    manifest = hf._json_object(
        hf._read_regular_bytes(root / HF_SUBDIR, hf.EXPORTED_SOURCE_MANIFEST_NAME),
        artifact="verified source checkpoint manifest",
    )
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        raise CheckpointIntegrityError("export source identity missing")
    if (identity.get("tokenizer_hash") != BYTE_TOKENIZER_HASH
            or identity.get("tokenizer_vocab_hash") != BYTE_VOCAB_HASH):
        raise CheckpointIntegrityError(
            "checkpoint tokenizer does not match canonical S0 byte tokenizer"
        )
    spec = identity.get("model_spec")
    if not isinstance(spec, dict):
        raise CheckpointIntegrityError("checkpoint model specification missing")

    token_config, token_vocab = _tokenizer_files()
    if (hf._read_regular_bytes(root, TOKENIZER) != token_config
            or hf._read_regular_bytes(root, VOCAB) != token_vocab):
        raise CheckpointIntegrityError("export tokenizer bytes differ from canonical tokenizer")
    config = hf._json_object(
        hf._read_regular_bytes(root / HF_SUBDIR, hf.EXPORTED_CONFIG_NAME),
        artifact="export model config",
    )
    required_config = {
        "schema": SCHEMA,
        "checkpoint_id": manifest["checkpoint_id"],
        "model_spec": spec,
        "tokenizer_hash": BYTE_TOKENIZER_HASH,
        "tokenizer_vocab_hash": BYTE_VOCAB_HASH,
    }
    if config != required_config:
        raise CheckpointIntegrityError("export model config differs from checkpoint authority")
    return {
        "schema": SCHEMA,
        "checkpoint_id": manifest["checkpoint_id"],
        "tokenizer_hash": BYTE_TOKENIZER_HASH,
        "tokenizer_vocab_hash": BYTE_VOCAB_HASH,
        "tokenizer_config_sha256": hf.sha256_bytes(token_config),
        "tokenizer_vocab_sha256": hf.sha256_bytes(token_vocab),
        "model_config_sha256": hf.sha256_bytes(
            hf._read_regular_bytes(root / HF_SUBDIR, hf.EXPORTED_CONFIG_NAME)
        ),
        "weights_sha256": hf.sha256_bytes(
            hf._read_regular_bytes(root / HF_SUBDIR, hf.EXPORTED_WEIGHTS_NAME)
        ),
        "hf_attestation_sha256": hf.sha256_bytes(
            hf._read_regular_bytes(root / HF_SUBDIR, hf.EXPORT_ATTESTATION_NAME)
        ),
        "interoperability": "HF_STYLE_ONLY_NO_TRANSFORMERS_PARITY_CLAIM",
    }


def verify_safe_export(
    directory: str | Path,
    *,
    expected_checkpoint_id: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Read and verify exact export bytes, optionally pinned by independent digest."""
    root = Path(directory)
    _strict_root(root)
    expected = _bundle_claim(root)
    blob = hf._read_regular_bytes(root, MANIFEST)
    checksum_bytes = hf._read_regular_bytes(root, CHECKSUM)
    try:
        checksum = checksum_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise CheckpointIntegrityError("bundle checksum must be ASCII") from exc
    digest = hf.sha256_bytes(blob)
    if checksum != f"{digest}  {MANIFEST}\n":
        raise CheckpointIntegrityError("safe bundle checksum mismatch")
    actual = hf._json_object(blob, artifact=MANIFEST)
    if actual != expected or blob != hf._strict_json_bytes(expected, artifact=MANIFEST) + b"\n":
        raise CheckpointIntegrityError("safe bundle claims mismatch verified bytes")
    if expected_checkpoint_id is not None and expected["checkpoint_id"] != expected_checkpoint_id:
        raise CheckpointIntegrityError("safe bundle checkpoint pin mismatch")
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise CheckpointIntegrityError("safe bundle independent digest pin mismatch")
    return dict(expected, manifest_sha256=digest)


def export_safe_bundle(checkpoint_dir: str | Path, destination: str | Path) -> Path:
    """Transactionally publish a create-only bundle without touching source weights."""
    final = Path(destination)
    if os.path.lexists(final):
        raise FileExistsError("safe export destination exists; existing evidence is immutable")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.plan4-stage-", dir=final.parent))
    identity = hf._temporary_directory_identity(staging)
    try:
        # Reuse incumbent D05 verified loader and byte-identical HF-style export.
        # Only canonical S0 byte tokenizer is supported by this v1 adapter.
        checkpoint = hf.prepare_checkpoint_load(checkpoint_dir)
        source = checkpoint.manifest
        binding = source.get("identity")
        if not isinstance(binding, dict):
            raise CheckpointIntegrityError("checkpoint identity missing")
        if (binding.get("tokenizer_hash") != BYTE_TOKENIZER_HASH
                or binding.get("tokenizer_vocab_hash") != BYTE_VOCAB_HASH):
            raise CheckpointIntegrityError("checkpoint tokenizer incompatible with S0 bundle")
        spec = binding.get("model_spec")
        if not isinstance(spec, dict):
            raise CheckpointIntegrityError("checkpoint model specification missing")
        token_config, token_vocab = _tokenizer_files()
        (staging / TOKENIZER).write_bytes(token_config)
        (staging / VOCAB).write_bytes(token_vocab)
        hf.export_hf_directory(
            checkpoint_dir,
            staging / HF_SUBDIR,
            hf_config={
                "schema": SCHEMA,
                "checkpoint_id": source["checkpoint_id"],
                "model_spec": spec,
                "tokenizer_hash": BYTE_TOKENIZER_HASH,
                "tokenizer_vocab_hash": BYTE_VOCAB_HASH,
            },
        )
        # Detect concurrent modification between our identity read and incumbent
        # export: only the same verified checkpoint is admitted.
        exported_manifest = hf._json_object(
            hf._read_regular_bytes(staging / HF_SUBDIR, hf.EXPORTED_SOURCE_MANIFEST_NAME),
            artifact="exported source manifest",
        )
        if exported_manifest["checkpoint_id"] != source["checkpoint_id"]:
            raise CheckpointIntegrityError("checkpoint changed during transfer export")
        claim = _bundle_claim(staging)
        payload = hf._strict_json_bytes(claim, artifact=MANIFEST) + b"\n"
        (staging / MANIFEST).write_bytes(payload)
        (staging / CHECKSUM).write_text(
            f"{hf.sha256_bytes(payload)}  {MANIFEST}\n", encoding="ascii"
        )
        verify_safe_export(staging, expected_checkpoint_id=source["checkpoint_id"])
        hf._publish_directory_noreplace(staging, final)
        staging = None
        return final
    finally:
        if staging is not None:
            hf._cleanup_temp_paths_strict(((staging, "Plan4 safe export staging", identity),))
