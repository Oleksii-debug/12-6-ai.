import json
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import (
    CheckpointIdentity,
    CheckpointIntegrityError,
    export_hf_directory,
    save_checkpoint,
    verify_hf_export,
)


class Model:
    def __init__(self):
        self.value = np.array([1.0, 2.0])

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state, strict=True):
        self.value = state["value"]


def checkpoint_identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="d" * 40,
        model_spec={"model_type": "twelve_six_test"},
        parameter_count=2,
        tokenizer_hash="1" * 64,
        tokenizer_vocab_hash="2" * 64,
        dataset_manifest_hash="3" * 64,
        run_manifest_hash="4" * 64,
        training_config={"steps": 0},
        seed=1,
        precision="float64",
        step=0,
        tokens_seen=0,
        optimizer={"name": "none"},
        scheduler=None,
    )


def test_hf_export_preserves_verified_weights_and_disclaims_runtime_compatibility(
    tmp_path: Path,
):
    model = Model()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=model, identity=checkpoint_identity())
    output = export_hf_directory(
        checkpoint,
        tmp_path / "hf",
        hf_config={"model_type": "twelve_six_test"},
    )

    assert (output / "model.safetensors").read_bytes() == (
        checkpoint / "weights.safetensors"
    ).read_bytes()
    assert (output / "config.json").is_file()
    assert (output / "12-6-checkpoint-manifest.json").is_file()

    attestation = verify_hf_export(checkpoint, output)
    assert attestation["compatibility"] == {
        "layout": "HF_STYLE_SAFETENSORS_DIRECTORY",
        "runtime_logit_generation_parity": "NOT_TESTED",
        "transformers_architecture": "NOT_CLAIMED",
        "weights": "EXACT_CANONICAL_BYTE_COPY",
    }
    assert set(attestation["files"]) == {
        "model.safetensors",
        "config.json",
        "12-6-checkpoint-manifest.json",
        "12-6-parity-request.json",
    }
    assert len(attestation["attestation_sha256"]) == 64

    parity = json.loads((output / "12-6-parity-request.json").read_text(encoding="utf-8"))
    assert parity["status"] == "NOT_TESTED"
    assert parity["hook_result"] is None
    assert parity["reference_weights_sha256"] == parity["candidate_weights_sha256"]
    assert parity["required_checks"] == [
        "prompt_token_identity",
        "next_token_logit_parity",
        "greedy_generation_parity",
    ]


def test_hf_export_invokes_external_parity_hook_without_overclaiming(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(), identity=checkpoint_identity())
    final_output = tmp_path / "hf"
    calls = []

    def parity_hook(source: Path, staging: Path):
        calls.append((source, staging, final_output.exists()))
        return {"status": "PASS", "evidence_ref": "test-only-d07-parity"}

    output = export_hf_directory(
        checkpoint,
        final_output,
        hf_config={"model_type": "twelve_six_test"},
        parity_hook=parity_hook,
    )

    assert len(calls) == 1
    source, staging, final_existed_during_hook = calls[0]
    assert source == checkpoint
    assert staging != output
    assert not final_existed_during_hook
    parity = json.loads((output / "12-6-parity-request.json").read_text(encoding="utf-8"))
    assert parity["status"] == "EXTERNAL_EVIDENCE_ATTACHED"
    assert parity["hook_result"] == {
        "status": "PASS",
        "evidence_ref": "test-only-d07-parity",
    }
    attestation = verify_hf_export(checkpoint, output)
    assert attestation["compatibility"]["transformers_architecture"] == "NOT_CLAIMED"
    assert attestation["compatibility"]["runtime_logit_generation_parity"] == "NOT_TESTED"


def test_hf_export_existing_destination_is_immutable_even_with_overwrite(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(), identity=checkpoint_identity())
    output = tmp_path / "hf"
    output.mkdir()
    marker = output / "prior-export-marker.txt"
    marker.write_text("must-survive", encoding="utf-8")

    with pytest.raises(FileExistsError, match="immutable"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_test"},
            overwrite=True,
        )

    assert marker.read_text(encoding="utf-8") == "must-survive"


def test_hf_export_rejects_parity_hook_payload_mutation_before_publish(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(), identity=checkpoint_identity())
    output = tmp_path / "hf"

    def parity_hook(_source: Path, staging: Path):
        (staging / "model.safetensors").write_bytes(b"tampered")
        return {"status": "PASS"}

    with pytest.raises(CheckpointIntegrityError, match="SafeTensors"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_test"},
            parity_hook=parity_hook,
        )

    assert not output.exists()


def test_hf_export_rejects_parity_hook_extra_artifact_before_publish(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(), identity=checkpoint_identity())
    output = tmp_path / "hf"

    def parity_hook(_source: Path, staging: Path):
        (staging / "untracked.bin").write_bytes(b"not-attested")
        return {"status": "PASS"}

    with pytest.raises(CheckpointIntegrityError, match="inventory mismatch"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_test"},
            parity_hook=parity_hook,
        )

    assert not output.exists()


def test_hf_export_hook_failure_does_not_publish_partial_directory(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(), identity=checkpoint_identity())
    output = tmp_path / "hf"

    def parity_hook(_source: Path, _staging: Path):
        raise RuntimeError("synthetic external parity failure")

    with pytest.raises(RuntimeError, match="synthetic external parity failure"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_test"},
            parity_hook=parity_hook,
        )

    assert not output.exists()


def test_verify_hf_export_fails_closed_on_post_publish_tamper(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(), identity=checkpoint_identity())
    output = export_hf_directory(
        checkpoint,
        tmp_path / "hf",
        hf_config={"model_type": "twelve_six_test"},
    )
    (output / "config.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="config hash"):
        verify_hf_export(checkpoint, output)


def test_verify_hf_export_rejects_untracked_file(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(), identity=checkpoint_identity())
    output = export_hf_directory(
        checkpoint,
        tmp_path / "hf",
        hf_config={"model_type": "twelve_six_test"},
    )
    (output / "untracked.txt").write_text("drift", encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="inventory mismatch"):
        verify_hf_export(checkpoint, output)
