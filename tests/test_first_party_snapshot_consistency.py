from __future__ import annotations

from pathlib import Path

import torch

import twelve_six.inference.first_party as first_party
from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint, verify_checkpoint
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


def _identity(*, step: int) -> CheckpointIdentity:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.model.parameter_count(),
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash="b" * 64,
        run_manifest_hash="c" * 64,
        training_config={
            "training": {"context_length": stage.model.max_seq_len},
            "data": {"tokenizer_version": tokenizer.identity.version},
        },
        seed=1337,
        precision="fp32",
        step=step,
        tokens_seen=step * 128,
        optimizer={"name": "test-only"},
        scheduler=None,
        environment_lock_hash="d" * 64,
    )


def _state_snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def test_first_party_loader_uses_one_verified_snapshot_across_source_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")

    source_model = TwelveSixDecoder(stage.model, stage.init)
    source_state = _state_snapshot(source_model)
    source = tmp_path / "checkpoint"
    source_manifest = save_checkpoint(source, model=source_model, identity=_identity(step=1))

    replacement_model = TwelveSixDecoder(stage.model, stage.init)
    with torch.no_grad():
        for parameter in replacement_model.parameters():
            parameter.add_(0.25)
    replacement_state = _state_snapshot(replacement_model)
    assert any(
        not torch.equal(source_state[name], replacement_state[name]) for name in source_state
    )
    replacement = tmp_path / "replacement"
    replacement_manifest = save_checkpoint(
        replacement,
        model=replacement_model,
        identity=_identity(step=2),
    )
    assert replacement_manifest["checkpoint_id"] != source_manifest["checkpoint_id"]

    real_prepare = first_party.prepare_checkpoint_load
    calls = 0

    def prepare_then_replace(path: Path):
        nonlocal calls
        calls += 1
        verified = real_prepare(path)
        archived = tmp_path / "archived-source"
        Path(path).rename(archived)
        replacement.rename(path)
        return verified

    monkeypatch.setattr(first_party, "prepare_checkpoint_load", prepare_then_replace)

    backend = first_party.load_first_party_backend(source)

    assert calls == 1
    assert verify_checkpoint(source)["checkpoint_id"] == replacement_manifest["checkpoint_id"]
    assert backend.diagnostics()["checkpoint_id"] == source_manifest["checkpoint_id"]
    assert backend.diagnostics()["step"] == 1
    assert backend.diagnostics()["tokens_seen"] == 128
    for name, tensor in backend.model.state_dict().items():
        assert torch.equal(tensor.detach().cpu(), source_state[name])
        assert not torch.equal(tensor.detach().cpu(), replacement_state[name])
