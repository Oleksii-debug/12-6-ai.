from __future__ import annotations

import shutil
from pathlib import Path

import torch

import twelve_six.inference.first_party as first_party
from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint, verify_checkpoint
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


def _identity(
    *,
    git_sha: str,
    model: TwelveSixDecoder,
    tokenizer: ByteTokenizer,
    step: int,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha=git_sha,
        model_spec=model.spec.to_dict(),
        parameter_count=model.spec.parameter_count(),
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={
            "training": {"context_length": model.spec.max_seq_len},
            "data": {"tokenizer_version": tokenizer.identity.version},
        },
        seed=1337,
        precision="fp32",
        step=step,
        tokens_seen=step * 128,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def _state_copy(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def test_first_party_loader_uses_one_snapshot_for_identity_and_weights(
    tmp_path: Path, monkeypatch
) -> None:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()

    torch.manual_seed(11)
    original_model = TwelveSixDecoder(stage.model, stage.init)
    original_state = _state_copy(original_model)
    checkpoint = tmp_path / "checkpoint"
    original_manifest = save_checkpoint(
        checkpoint,
        model=original_model,
        identity=_identity(
            git_sha="a" * 40,
            model=original_model,
            tokenizer=tokenizer,
            step=7,
        ),
    )

    torch.manual_seed(22)
    replacement_model = TwelveSixDecoder(stage.model, stage.init)
    replacement_state = _state_copy(replacement_model)
    replacement = tmp_path / "replacement"
    replacement_manifest = save_checkpoint(
        replacement,
        model=replacement_model,
        identity=_identity(
            git_sha="b" * 40,
            model=replacement_model,
            tokenizer=tokenizer,
            step=9,
        ),
    )
    assert original_manifest["checkpoint_id"] != replacement_manifest["checkpoint_id"]
    assert any(
        not torch.equal(original_state[name], replacement_state[name])
        for name in original_state
    )

    real_prepare = first_party.prepare_checkpoint_load
    prepare_calls = 0

    def prepare_then_replace(path: Path):
        nonlocal prepare_calls
        prepare_calls += 1
        verified = real_prepare(path)
        shutil.rmtree(path)
        shutil.copytree(replacement, path)
        return verified

    monkeypatch.setattr(first_party, "prepare_checkpoint_load", prepare_then_replace)
    backend = first_party.load_first_party_backend(checkpoint)

    assert prepare_calls == 1
    diagnostics = backend.diagnostics()
    assert diagnostics["checkpoint_id"] == original_manifest["checkpoint_id"]
    assert diagnostics["git_sha"] == "a" * 40
    assert diagnostics["step"] == 7

    loaded_state = backend.model.state_dict()
    assert loaded_state.keys() == original_state.keys()
    assert all(
        torch.equal(loaded_state[name].detach().cpu(), original_state[name])
        for name in original_state
    )

    # The path now names a different valid checkpoint. It is intentionally not
    # authority for the already-created backend: identities and weights both
    # came from the same immutable D05 snapshot captured before this swap.
    live_path_manifest = verify_checkpoint(checkpoint)
    assert live_path_manifest["checkpoint_id"] == replacement_manifest["checkpoint_id"]
    assert live_path_manifest["identity"]["git_sha"] == "b" * 40
