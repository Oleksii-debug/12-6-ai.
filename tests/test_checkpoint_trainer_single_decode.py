from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from twelve_six.checkpoint import CheckpointIdentity
from twelve_six.checkpoint import trainer_adapter
from twelve_six.checkpoint.trainer_adapter import (
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)


class _GenericTrainer:
    def __init__(self) -> None:
        self.config = {"gradient_accumulation_steps": 1, "max_steps": 4}
        self.velocity = [1.0, 2.0]
        self.loads = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "micro_step": 0,
            "optimizer_step": 0,
            "tokens_seen": 0,
            "optimizer": {"velocity": list(self.velocity)},
            "scheduler": None,
            "scaler": None,
            "config": copy.deepcopy(self.config),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["config"] != self.config:
            raise ValueError("config mismatch")
        velocity = state["optimizer"]["velocity"]
        if not isinstance(velocity, list) or len(velocity) != 2:
            raise ValueError("velocity geometry mismatch")
        self.velocity = list(velocity)
        self.loads += 1


def _identity(parameter_count: int) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec={"kind": "single-decode-probe", "parameters": parameter_count},
        parameter_count=parameter_count,
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"probe": "single-decode"},
        seed=17,
        precision="float32",
        step=0,
        tokens_seen=0,
        optimizer={"name": "generic-probe"},
        scheduler=None,
    )


def test_trainer_resume_decodes_verified_snapshot_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    source_model = torch.nn.Linear(3, 2, bias=False)
    source_trainer = _GenericTrainer()
    source_trainer.velocity = [3.0, 4.0]
    checkpoint = tmp_path / "single-decode"
    save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source_trainer,
        identity=_identity(sum(parameter.numel() for parameter in source_model.parameters())),
    )

    torch.manual_seed(99)
    target_model = torch.nn.Linear(3, 2, bias=False)
    target_trainer = _GenericTrainer()

    original_decode = trainer_adapter._decode_verified_state
    decode_calls = 0

    def counted_decode(verified: Any) -> Any:
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(verified)

    monkeypatch.setattr(trainer_adapter, "_decode_verified_state", counted_decode)

    result = load_trainer_checkpoint(
        checkpoint,
        model=target_model,
        trainer=target_trainer,
        restore_rng=False,
    )

    assert decode_calls == 1
    assert result.manifest["identity"]["step"] == 0
    assert target_trainer.loads == 1
    assert target_trainer.velocity == [3.0, 4.0]
    for name, source_tensor in source_model.state_dict().items():
        torch.testing.assert_close(
            target_model.state_dict()[name],
            source_tensor,
            rtol=0,
            atol=0,
        )
