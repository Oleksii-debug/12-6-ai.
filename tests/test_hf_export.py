from pathlib import Path

import numpy as np

from twelve_six.checkpoint import CheckpointIdentity, export_hf_directory, save_checkpoint


class Model:
    def __init__(self):
        self.value = np.array([1.0, 2.0])

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state, strict=True):
        self.value = state["value"]


def test_hf_export_preserves_verified_weights_and_provenance(tmp_path: Path):
    model = Model()
    identity = CheckpointIdentity(
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
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=model, identity=identity)
    output = export_hf_directory(
        checkpoint,
        tmp_path / "hf",
        hf_config={"model_type": "twelve_six_test"},
    )
    assert (output / "model.safetensors").read_bytes() == (checkpoint / "weights.safetensors").read_bytes()
    assert (output / "config.json").is_file()
    assert (output / "12-6-checkpoint-manifest.json").is_file()
