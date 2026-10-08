"""Plan 4 Section 4 acceptance: reuse canonical D05 export, never deserialize code."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import CheckpointIdentity, CheckpointIntegrityError, save_checkpoint
from twelve_six.tokenization.byte import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH
from tools.plan4_safe_export import (
    CHECKSUM,
    HF_SUBDIR,
    MANIFEST,
    TOKENIZER,
    VOCAB,
    export_safe_bundle,
    verify_safe_export,
)


class TinyModel:
    def state_dict(self):
        return {"weight": np.array([1.25], dtype=np.float64)}


def _identity(*, tokenizer_hash: str = BYTE_TOKENIZER_HASH) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="b" * 40,
        model_spec={"model_type": "safe-plan4-fixture", "vocab_size": 256},
        parameter_count=1,
        tokenizer_hash=tokenizer_hash,
        tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        dataset_manifest_hash="3" * 64,
        run_manifest_hash="4" * 64,
        training_config={"steps": 1},
        seed=7,
        precision="float64",
        step=1,
        tokens_seen=1,
        optimizer={"name": "none"},
        scheduler=None,
    )


def _source(tmp_path: Path, *, tokenizer_hash: str = BYTE_TOKENIZER_HASH):
    checkpoint = tmp_path / "checkpoint"
    source = save_checkpoint(
        checkpoint, model=TinyModel(), identity=_identity(tokenizer_hash=tokenizer_hash)
    )
    return checkpoint, source


def _bundle(tmp_path: Path) -> tuple[Path, dict]:
    checkpoint, source = _source(tmp_path)
    output = tmp_path / "portable"
    export_safe_bundle(checkpoint, output)
    return output, source


def test_byte_identical_verified_safe_export_and_pinned_restart(tmp_path: Path):
    out, source = _bundle(tmp_path)
    a = verify_safe_export(out, expected_checkpoint_id=source["checkpoint_id"])
    b = verify_safe_export(
        out,
        expected_checkpoint_id=source["checkpoint_id"],
        expected_manifest_sha256=a["manifest_sha256"],
    )
    assert a == b
    assert (out / HF_SUBDIR / "model.safetensors").is_file()
    assert a["tokenizer_hash"] == BYTE_TOKENIZER_HASH
    assert a["tokenizer_vocab_hash"] == BYTE_VOCAB_HASH
    assert a["interoperability"] == "HF_STYLE_ONLY_NO_TRANSFORMERS_PARITY_CLAIM"
    assert json.loads((out / HF_SUBDIR / "config.json").read_text())["checkpoint_id"] == (
        source["checkpoint_id"]
    )
    assert len(a["manifest_sha256"]) == 64


def test_source_tokenizer_mismatch_blocks_publication(tmp_path: Path):
    checkpoint, _ = _source(tmp_path, tokenizer_hash="9" * 64)
    with pytest.raises(CheckpointIntegrityError, match="tokenizer incompatible"):
        export_safe_bundle(checkpoint, tmp_path / "portable")
    assert not (tmp_path / "portable").exists()
    assert not list(tmp_path.glob(".portable.plan4-stage-*"))


def test_corrupt_tokenizer_cannot_be_accepted_by_resigning_manifest(tmp_path: Path):
    out, _ = _bundle(tmp_path)
    (out / TOKENIZER).write_bytes(b'{"encoding":"unexpected"}\n')
    with pytest.raises(CheckpointIntegrityError, match="tokenizer bytes"):
        verify_safe_export(out)


def test_corrupt_weights_fail_verified_incumbent_integrity(tmp_path: Path):
    out, _ = _bundle(tmp_path)
    weight = out / HF_SUBDIR / "model.safetensors"
    weight.write_bytes(weight.read_bytes() + b"corruption")
    with pytest.raises(CheckpointIntegrityError):
        verify_safe_export(out)


def test_corrupt_manifest_and_external_checkpoint_pin_fail_closed(tmp_path: Path):
    out, source = _bundle(tmp_path)
    with pytest.raises(CheckpointIntegrityError, match="checkpoint pin"):
        verify_safe_export(out, expected_checkpoint_id="f" * 64)
    with pytest.raises(CheckpointIntegrityError, match="digest pin"):
        verify_safe_export(out, expected_manifest_sha256="0" * 64)
    (out / MANIFEST).write_bytes(b"{}\n")
    with pytest.raises(CheckpointIntegrityError, match="checksum mismatch"):
        verify_safe_export(out)


def test_unknown_artifact_and_symlink_are_rejected(tmp_path: Path):
    out, _ = _bundle(tmp_path)
    (out / "unexpected.pkl").write_bytes(b"never load")
    with pytest.raises(CheckpointIntegrityError, match="unexpected"):
        verify_safe_export(out)
    (out / "unexpected.pkl").unlink()
    if hasattr(os, "symlink"):
        vocab = out / VOCAB
        copy = tmp_path / "vocab-copy"
        copy.write_bytes(vocab.read_bytes())
        vocab.unlink()
        try:
            vocab.symlink_to(copy)
        except OSError:
            pytest.skip("symlink privilege unavailable")
        with pytest.raises(CheckpointIntegrityError, match="non-symlink"):
            verify_safe_export(out)


def test_create_only_existing_bundle_never_overwritten(tmp_path: Path):
    out, _ = _bundle(tmp_path)
    original = (out / CHECKSUM).read_bytes()
    checkpoint = tmp_path / "checkpoint"
    with pytest.raises(FileExistsError, match="immutable"):
        export_safe_bundle(checkpoint, out)
    assert (out / CHECKSUM).read_bytes() == original
    verify_safe_export(out)


def test_interrupted_atomic_publication_leaves_no_output_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint, _ = _source(tmp_path)
    from tools import plan4_safe_export as bundle

    def interrupted(*_args):
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(bundle.hf, "_publish_directory_noreplace", interrupted)
    with pytest.raises(OSError, match="interruption"):
        export_safe_bundle(checkpoint, tmp_path / "portable")
    assert not (tmp_path / "portable").exists()
    assert not list(tmp_path.glob(".portable.plan4-stage-*"))
