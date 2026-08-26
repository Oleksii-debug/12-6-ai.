from __future__ import annotations

import copy

import pytest

import twelve_six.checkpoint.core as checkpoint_core
import twelve_six.checkpoint.trainer_adapter as trainer_adapter
from twelve_six.checkpoint import CheckpointIdentity


class _Trainer:
    def __init__(self, model):
        torch = pytest.importorskip("torch")
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        self.scheduler = None
        self.scaler = None
        self.config = {"gradient_accumulation_steps": 1, "max_steps": 10}
        self.micro_step = 0
        self.optimizer_step = 0
        self.tokens_seen = 0
        self.loads = 0

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None,
            "scaler": None,
            "config": copy.deepcopy(self.config),
            "micro_step": self.micro_step,
            "optimizer_step": self.optimizer_step,
            "tokens_seen": self.tokens_seen,
        }

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.micro_step = state["micro_step"]
        self.optimizer_step = state["optimizer_step"]
        self.tokens_seen = state["tokens_seen"]
        self.loads += 1


def _identity(parameter_count: int) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "single-decode-resume-probe", "parameters": parameter_count},
        parameter_count=parameter_count,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"kind": "single-decode-resume-probe"},
        seed=7,
        precision="fp32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "SGD", "lr": 0.05, "momentum": 0.9},
        scheduler=None,
    )


def test_trainer_resume_decodes_verified_snapshot_once(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    source_model = torch.nn.Linear(4, 3, bias=False)
    source_trainer = _Trainer(source_model)
    checkpoint = tmp_path / "single-decode"
    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source_trainer,
        identity=_identity(sum(parameter.numel() for parameter in source_model.parameters())),
    )

    original_decode = checkpoint_core._decode_verified_state
    calls = 0

    def counted_decode(verified):
        nonlocal calls
        calls += 1
        return original_decode(verified)

    # The historical adapter decoded through its local alias for preflight and
    # then core.load_verified_checkpoint decoded again. Count both call surfaces
    # so this regression fails if that two-decode path returns.
    monkeypatch.setattr(checkpoint_core, "_decode_verified_state", counted_decode)
    monkeypatch.setattr(trainer_adapter, "_decode_verified_state", counted_decode)

    torch.manual_seed(99)
    target_model = torch.nn.Linear(4, 3, bias=False)
    target_trainer = _Trainer(target_model)
    trainer_adapter.load_trainer_checkpoint(
        checkpoint,
        model=target_model,
        trainer=target_trainer,
        restore_rng=False,
    )

    assert calls == 1
    assert target_trainer.loads == 1
    for name, expected in source_model.state_dict().items():
        torch.testing.assert_close(target_model.state_dict()[name], expected, rtol=0, atol=0)
