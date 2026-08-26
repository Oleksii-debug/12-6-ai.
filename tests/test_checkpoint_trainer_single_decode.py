from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.checkpoint import CheckpointIdentity
from twelve_six.checkpoint import core as checkpoint_core
from twelve_six.checkpoint import trainer_adapter
from twelve_six.training import Trainer, TrainerConfig


def _identity(parameter_count: int) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "trainer-single-decode-probe", "parameters": parameter_count},
        parameter_count=parameter_count,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"probe": True},
        seed=23,
        precision="float32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "AdamW"},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def test_production_trainer_resume_decodes_verified_payload_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    config = TrainerConfig(max_steps=1, seed=23)

    source_model = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        source_model.weight.fill_(3.0)
    source_trainer = Trainer(source_model, config)

    checkpoint = tmp_path / "trainer-single-decode"
    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source_trainer,
        identity=_identity(sum(parameter.numel() for parameter in source_model.parameters())),
    )

    target_model = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        target_model.weight.fill_(-2.0)
    target_trainer = Trainer(target_model, config)

    original_decode = checkpoint_core._decode_verified_state
    decode_calls = 0

    def counted_decode(verified: checkpoint_core.VerifiedCheckpoint):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(verified)

    # The historical path decoded once through trainer_adapter and then again
    # through core.load_verified_checkpoint(). Count either call site so this
    # regression would fail against that implementation with decode_calls == 2.
    monkeypatch.setattr(trainer_adapter, "_decode_verified_state", counted_decode)
    monkeypatch.setattr(checkpoint_core, "_decode_verified_state", counted_decode)

    result = trainer_adapter.load_trainer_checkpoint(
        checkpoint,
        model=target_model,
        trainer=target_trainer,
        restore_rng=False,
    )

    assert decode_calls == 1
    assert result.manifest["checkpoint_id"]
    assert result.trainer_state["optimizer_step"] == 0
    assert result.trainer_state["tokens_seen"] == 0
    torch.testing.assert_close(
        target_model.weight,
        torch.full_like(target_model.weight, 3.0),
        rtol=0,
        atol=0,
    )
