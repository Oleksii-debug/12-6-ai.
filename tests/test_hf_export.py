import json
from pathlib import Path

import numpy as np

from twelve_six.checkpoint import (
    CheckpointIdentity,
    export_hf_directory,
    save_checkpoint,
    verify_checkpoint,
    verify_hf_directory,
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
    assert (output / "12-6-export.sha256").is_file()

    attestation = verify_hf_directory(output)
    assert attestation["schema"] == "12-6.hf-style-export.v2"
    assert attestation["compatibility"] == {
        "layout": "HF_STYLE_SAFETENSORS_DIRECTORY",
        "runtime_logit_generation_parity": "NOT_TESTED",
        "transformers_architecture": "NOT_CLAIMED",
        "weights": "EXACT_CANONICAL_BYTE_COPY",
    }

    parity = json.loads((output / "12-6-parity-request.json").read_text(encoding="utf-8"))
    assert parity["schema"] == "12-6.export-parity-request.v2"
    assert parity["status"] == "NOT_TESTED"
    assert parity["hook_result"] is None
    assert parity["reference_weights_sha256"] == parity["candidate_weights_sha256"]
    assert parity["required_checks"] == [
        "prompt_token_identity",
        "next_token_logit_parity",
        "greedy_generation_parity",
    ]


def test_hf_export_invokes_external_parity_hook_on_verified_reference_snapshot(
    tmp_path: Path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint_manifest = save_checkpoint(
        checkpoint,
        model=Model(),
        identity=checkpoint_identity(),
    )
    calls = []

    def parity_hook(reference: Path, destination: Path):
        assert reference != checkpoint
        assert reference.name.startswith(".hf.reference-")
        assert destination.name.startswith(".hf.staging-")
        assert verify_checkpoint(reference)["checkpoint_id"] == checkpoint_manifest["checkpoint_id"]
        assert (destination / "model.safetensors").is_file()
        calls.append((reference.name, destination.name))
        return {"status": "PASS", "evidence_ref": "test-only-d07-parity"}

    output = export_hf_directory(
        checkpoint,
        tmp_path / "hf",
        hf_config={"model_type": "twelve_six_test"},
        parity_hook=parity_hook,
    )

    assert len(calls) == 1
    assert not list(tmp_path.glob(".hf.reference-*"))
    parity = json.loads((output / "12-6-parity-request.json").read_text(encoding="utf-8"))
    assert parity["status"] == "EXTERNAL_EVIDENCE_ATTACHED"
    assert parity["hook_result"] == {
        "status": "PASS",
        "evidence_ref": "test-only-d07-parity",
    }
    attestation = verify_hf_directory(output)
    assert attestation["compatibility"]["transformers_architecture"] == "NOT_CLAIMED"
    assert attestation["compatibility"]["runtime_logit_generation_parity"] == "NOT_TESTED"
