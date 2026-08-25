import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import (
    CheckpointIdentity,
    CheckpointIntegrityError,
    export_hf_directory,
    hf_export,
    save_checkpoint,
    verify_checkpoint,
    verify_hf_directory,
)


class Model:
    def __init__(self, value: float):
        self.value = np.array([value], dtype=np.float64)

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state, strict=True):
        self.value = state["value"]


def identity(fill: str = "a") -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha=fill * 40,
        model_spec={"model_type": "twelve_six_export_transactional"},
        parameter_count=1,
        tokenizer_hash="1" * 64,
        tokenizer_vocab_hash="2" * 64,
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


def snapshot_tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_existing_export_is_immutable_even_with_overwrite_true(tmp_path: Path):
    first_checkpoint = tmp_path / "checkpoint-a"
    second_checkpoint = tmp_path / "checkpoint-b"
    output = tmp_path / "hf"
    save_checkpoint(first_checkpoint, model=Model(1.0), identity=identity("a"))
    save_checkpoint(second_checkpoint, model=Model(2.0), identity=identity("b"))
    export_hf_directory(
        first_checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    before = snapshot_tree(output)

    with pytest.raises(FileExistsError, match="destructive replacement"):
        export_hf_directory(
            second_checkpoint,
            output,
            hf_config={"model_type": "twelve_six_export_transactional"},
            overwrite=True,
        )

    assert snapshot_tree(output) == before
    verify_hf_directory(output)


def test_export_consumes_verified_checkpoint_snapshot_without_reopening_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(3.0), identity=identity("c"))
    original_weights = (checkpoint / "weights.safetensors").read_bytes()
    real_prepare = hf_export.prepare_checkpoint_load

    def prepare_then_tamper(path: Path):
        verified = real_prepare(path)
        (checkpoint / "weights.safetensors").write_bytes(b"tampered-after-snapshot")
        return verified

    monkeypatch.setattr(hf_export, "prepare_checkpoint_load", prepare_then_tamper)
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    assert (output / "model.safetensors").read_bytes() == original_weights
    verify_hf_directory(output)


def test_parity_hook_reads_verified_reference_after_source_path_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    manifest = save_checkpoint(checkpoint, model=Model(3.5), identity=identity("c"))
    original_weights = (checkpoint / "weights.safetensors").read_bytes()
    real_prepare = hf_export.prepare_checkpoint_load
    first_call = True

    def prepare_then_tamper(path: Path):
        nonlocal first_call
        verified = real_prepare(path)
        if first_call:
            first_call = False
            (checkpoint / "weights.safetensors").write_bytes(b"source-path-now-corrupt")
        return verified

    def parity_hook(reference: Path, staging: Path):
        assert verify_checkpoint(reference)["checkpoint_id"] == manifest["checkpoint_id"]
        assert (reference / "weights.safetensors").read_bytes() == original_weights
        assert (staging / "model.safetensors").read_bytes() == original_weights
        return {"status": "PASS", "evidence_ref": "verified-reference-test"}

    monkeypatch.setattr(hf_export, "prepare_checkpoint_load", prepare_then_tamper)
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
        parity_hook=parity_hook,
    )

    verify_hf_directory(output)
    assert not list(tmp_path.glob(".hf.reference-*"))


def test_hook_failure_leaves_no_published_or_staging_export(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(4.0), identity=identity("d"))

    def broken_hook(_reference: Path, _staging: Path):
        raise RuntimeError("parity failed")

    with pytest.raises(RuntimeError, match="parity failed"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_export_transactional"},
            parity_hook=broken_hook,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".hf.staging-*"))
    assert not list(tmp_path.glob(".hf.reference-*"))


@pytest.mark.skipif(
    os.name != "nt" and not sys.platform.startswith("linux"),
    reason="atomic no-replace publish is only implemented on Windows/Linux",
)
def test_concurrent_destination_creation_is_not_replaced(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(4.5), identity=identity("d"))

    def racing_hook(_reference: Path, _staging: Path):
        output.mkdir()
        (output / "owner-evidence.txt").write_text("preserve me", encoding="utf-8")
        return {"status": "PASS", "evidence_ref": "racing-destination-test"}

    with pytest.raises(FileExistsError, match="appeared during publish"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_export_transactional"},
            parity_hook=racing_hook,
        )

    assert (output / "owner-evidence.txt").read_text(encoding="utf-8") == "preserve me"
    assert not list(tmp_path.glob(".hf.staging-*"))
    assert not list(tmp_path.glob(".hf.reference-*"))


def test_export_verifier_rejects_payload_tamper(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(5.0), identity=identity("e"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    weights = output / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"x")
    with pytest.raises(CheckpointIntegrityError, match="canonical"):
        verify_hf_directory(output)


def test_export_verifier_rejects_attestation_tamper(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(6.0), identity=identity("f"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    attestation_path = output / "12-6-export.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["compatibility"]["transformers_architecture"] = "CLAIMED"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="attestation checksum"):
        verify_hf_directory(output)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support unavailable")
def test_export_verifier_rejects_symlink_payload(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(7.0), identity=identity("0"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    weights = output / "model.safetensors"
    target = tmp_path / "weights-copy.safetensors"
    target.write_bytes(weights.read_bytes())
    weights.unlink()
    try:
        weights.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable in this environment")

    with pytest.raises(CheckpointIntegrityError, match="non-symlink"):
        verify_hf_directory(output)
