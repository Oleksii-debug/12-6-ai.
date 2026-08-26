from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

import twelve_six.checkpoint.trainer_adapter as trainer_adapter
from twelve_six.checkpoint import CheckpointIdentity


class _TrainerProbe:
    def __init__(self, model: Any) -> None:
        torch = pytest.importorskip("torch")
        self.config = {"gradient_accumulation_steps": 1, "max_steps": 2}
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        self.scheduler = None
        self.scaler = None
        self.loads = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "micro_step": 0,
            "optimizer_step": 0,
            "tokens_seen": 0,
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
        step=0,
        tokens_seen=0,
        optimizer={"name": "SGD", "lr": 0.1},
        scheduler=None,
    )


def test_trainer_resume_decodes_verified_checkpoint_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(11)
    source_model = torch.nn.Linear(3, 2, bias=False)
    source_trainer = _TrainerProbe(source_model)
    checkpoint = tmp_path / "single-decode"
    trainer_adapter.save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source_trainer,
        identity=_identity(sum(parameter.numel() for parameter in source_model.parameters())),
    )

    torch.manual_seed(99)
    target_model = torch.nn.Linear(3, 2, bias=False)
    target_trainer = _TrainerProbe(target_model)

    real_decode = trainer_adapter._decode_verified_state
    decode_calls = 0

    def counting_decode(verified: Any) -> Any:
        nonlocal decode_calls
        decode_calls += 1
        return real_decode(verified)

    monkeypatch.setattr(trainer_adapter, "_decode_verified_state", counting_decode)

    result = trainer_adapter.load_trainer_checkpoint(
        checkpoint,
        model=target_model,
        trainer=target_trainer,
        restore_rng=False,
    )

    assert decode_calls == 1
    assert target_trainer.loads == 1
    assert result.manifest["identity"]["step"] == 0
    for name, tensor in source_model.state_dict().items():
        torch.testing.assert_close(target_model.state_dict()[name], tensor, rtol=0, atol=0)
