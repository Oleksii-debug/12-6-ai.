from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

import twelve_six.checkpoint.core as checkpoint_core
import twelve_six.checkpoint.trainer_adapter as trainer_adapter
from twelve_six.checkpoint import CheckpointIdentity


class _TrainerProbe:
    def __init__(self, model: Any, *, populated: bool) -> None:
        torch = pytest.importorskip("torch")
        self.config = {"gradient_accumulation_steps": 1, "max_steps": 4}
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        self.scheduler = None
        self.scaler = None
        self.loads = 0
        if populated:
            loss = model(torch.ones(1, 3)).sum()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self) -> dict[str, Any]:
        return {
            "micro_step": 1,
            "optimizer_step": 1,
            "tokens_seen": 3,
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": None,
            "scaler": None,
            "config": copy.deepcopy(self.config),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.loads += 1
        self.optimizer.load_state_dict(state["optimizer"])


def _identity(parameter_count: int) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "single-decode-probe", "parameters": parameter_count},
        parameter_count=parameter_count,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"probe": True},
        seed=17,
        precision="float32",
        step=1,
        tokens_seen=3,
        optimizer={"name": "SGD", "lr": 0.05, "momentum": 0.9},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )


def test_trainer_checkpoint_restore_decodes_verified_payload_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    source_model = torch.nn.Linear(3, 2, bias=False)
    source_trainer = _TrainerProbe(source_model, populated=True)
    checkpoint = tmp_path / "single-decode"
    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source_trainer,
        identity=_identity(sum(parameter.numel() for parameter in source_model.parameters())),
    )

    original_decode = checkpoint_core._decode_verified_state
    decode_calls = 0

    def counted_decode(verified: checkpoint_core.VerifiedCheckpoint):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(verified)

    monkeypatch.setattr(checkpoint_core, "_decode_verified_state", counted_decode)
    monkeypatch.setattr(trainer_adapter, "_decode_verified_state", counted_decode)

    torch.manual_seed(99)
    target_model = torch.nn.Linear(3, 2, bias=False)
    target_trainer = _TrainerProbe(target_model, populated=False)
    trainer_adapter.load_trainer_checkpoint(
        checkpoint,
        model=target_model,
        trainer=target_trainer,
        restore_rng=False,
    )

    assert decode_calls == 1
    assert target_trainer.loads == 1
    for name, tensor in target_model.state_dict().items():
        torch.testing.assert_close(tensor, source_model.state_dict()[name], rtol=0, atol=0)
