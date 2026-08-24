import json
from pathlib import Path

import numpy as np
import pytest

from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint
from twelve_six.checkpoint.hf_export import export_hf_directory, verify_hf_directory


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

    verified = verify_hf_directory(checkpoint, output)
    assert verified.weights_bytes == (checkpoint / "weights.safetensors").read_bytes()
    assert (output / "config.json").is_file()
    assert (output / "12-6-checkpoint-manifest.json").is_file()

    attestation = json.loads((output / "12-6-export.json").read_text(encoding="utf-8"))
    assert attestation["compatibility"] == {
        "layout": "HF_STYLE_SAFETENSORS_DIRECTORY",
        "runtime_logit_generation_parity": "NOT_TESTED",
        "transformers_architecture": "NOT_CLAIMED",
        "weights": "EXACT_CANONICAL_BYTE_COPY",
    }

    parity = json.loads((output / "12-6-parity-request.json").read_text(encoding="utf-8"))
    assert parity["status"] == "NOT_TESTED"
    assert parity["hook_result"] is None
    assert parity["reference_weights_sha256"] == parity["candidate_weights_sha256"]
    assert parity["required_checks"] == [
        "prompt_token_identity",
        "next_token_logit_parity",
        "greedy_generation_parity",
    ]


def test_hf_export_invokes_external_parity_hook_on_staging_before_publish(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(), identity=checkpoint_identity())
    calls = []

    def parity_hook(source: Path, staging: Path):
        assert staging.exists()
        assert staging.name.startswith(".hf.staging-")
        calls.append((source, staging.name))
        return {"status": "PASS", "evidence_ref": "test-only-d07-parity"}

    output = export_hf_directory(
        checkpoint,
        tmp_path / "hf",
        hf_config={"model_type": "twelve_six_test"},
        parity_hook=parity_hook,
    )

    assert calls == [(checkpoint, calls[0][1])]
    assert output == tmp_path / "hf"
    parity = json.loads((output / "12-6-parity-request.json").read_text(encoding="utf-8"))
    assert parity["status"] == "EXTERNAL_EVIDENCE_ATTACHED"
    assert parity["hook_result"] == {
        "status": "PASS",
        "evidence_ref": "test-only-d07-parity",
    }
    attestation = json.loads((output / "12-6-export.json").read_text(encoding="utf-8"))
    assert attestation["compatibility"]["transformers_architecture"] == "NOT_CLAIMED"
    assert attestation["compatibility"]["runtime_logit_generation_parity"] == "NOT_TESTED"
    verify_hf_directory(checkpoint, output)


def test_hf_export_existing_destination_is_immutable_even_with_overwrite(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(), identity=checkpoint_identity())
    output = export_hf_directory(
        checkpoint,
        tmp_path / "hf",
        hf_config={"model_type": "first"},
    )
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(FileExistsError, match="immutable"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "second"},
            overwrite=True,
        )

    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert after == before


def test_hf_export_hook_failure_never_publishes_partial_directory(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(), identity=checkpoint_identity())
    output = tmp_path / "hf"

    def parity_hook(_source: Path, _staging: Path):
        raise RuntimeError("injected parity failure")

    with pytest.raises(RuntimeError, match="injected parity failure"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_test"},
            parity_hook=parity_hook,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".hf.staging-*"))


def test_hf_export_rejects_tampered_or_extended_published_directory(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(), identity=checkpoint_identity())
    output = export_hf_directory(
        checkpoint,
        tmp_path / "hf",
        hf_config={"model_type": "twelve_six_test"},
    )

    weights = output / "model.safetensors"
    original = weights.read_bytes()
    weights.write_bytes(original + b"tamper")
    with pytest.raises(RuntimeError, match="exact verified source weight bytes"):
        verify_hf_directory(checkpoint, output)
    weights.write_bytes(original)

    (output / "unexpected.bin").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        verify_hf_directory(checkpoint, output)


def test_hf_export_source_mutation_after_staging_fails_before_publish(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(), identity=checkpoint_identity())
    output = tmp_path / "hf"

    def mutate_source(_source: Path, _staging: Path):
        (checkpoint / "weights.safetensors").write_bytes(b"corrupt-after-snapshot")
        return {"status": "PASS", "evidence_ref": "injected-source-mutation"}

    with pytest.raises(RuntimeError):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_test"},
            parity_hook=mutate_source,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".hf.staging-*"))
