from __future__ import annotations

import shutil
from pathlib import Path

import torch

from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint
from twelve_six.inference import first_party
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


def _save_s0_checkpoint(path: Path) -> tuple[TwelveSixDecoder, dict[str, object]]:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    torch.manual_seed(20260825)
    model = TwelveSixDecoder(stage.model, stage.init)
    identity = CheckpointIdentity(
        git_sha="a" * 40,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.model.parameter_count(),
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={
            "training": {"context_length": stage.model.max_seq_len},
            "data": {"tokenizer_version": tokenizer.identity.version},
        },
        seed=20260825,
        precision="fp32",
        step=3,
        tokens_seen=384,
        optimizer={"name": "test-only"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )
    manifest = save_checkpoint(path, model=model, identity=identity)
    return model, manifest


def test_first_party_load_uses_one_verified_snapshot_after_source_removal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    source_model, manifest = _save_s0_checkpoint(checkpoint)
    expected_state = {
        name: tensor.detach().clone() for name, tensor in source_model.state_dict().items()
    }
    real_prepare = first_party.prepare_checkpoint_load
    prepared = []

    def prepare_then_remove(path: Path):
        verified = real_prepare(path)
        prepared.append(verified)
        shutil.rmtree(path)
        return verified

    monkeypatch.setattr(first_party, "prepare_checkpoint_load", prepare_then_remove)
    backend = first_party.load_first_party_backend(checkpoint)

    assert len(prepared) == 1
    assert not checkpoint.exists()
    assert backend.diagnostics()["checkpoint_id"] == manifest["checkpoint_id"]
    assert backend.diagnostics()["git_sha"] == "a" * 40
    for name, expected in expected_state.items():
        torch.testing.assert_close(
            backend.model.state_dict()[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_first_party_diagnostics_are_anchored_to_private_verified_manifest(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    _, manifest = _save_s0_checkpoint(checkpoint)
    backend = first_party.load_first_party_backend(checkpoint)
    before = backend.diagnostics()

    exposed = backend.manifest
    exposed["checkpoint_id"] = "0" * 64
    exposed["identity"]["git_sha"] = "b" * 40
    exposed["identity"]["dataset_manifest_hash"] = "c" * 64

    assert backend.diagnostics() == before
    assert backend.manifest["checkpoint_id"] == manifest["checkpoint_id"]
    assert backend.manifest["identity"]["git_sha"] == "a" * 40
    assert backend.manifest is not exposed
