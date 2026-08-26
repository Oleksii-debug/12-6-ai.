from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

import twelve_six.checkpoint.core as checkpoint_core
from twelve_six.checkpoint import CheckpointIdentity
from twelve_six.checkpoint.trainer_adapter import (
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)


class _TrainerProbe:
    def __init__(self, model: Any, *, populated: bool) -> None:
        torch = pytest.importorskip("torch")
        self.config = {"gradient_accumulation_steps": 1, "max_steps": 10}
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


def test_trainer_resume_decodes_verified_payload_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    source_model = torch.nn.Linear(3, 2, bias=False)
    source_trainer = _TrainerProbe(source_model, populated=True)
    checkpoint = tmp_path / "single-decode"
    save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source_trainer,
        identity=_identity(sum(parameter.numel() for parameter in source_model.parameters())),
    )

    torch.manual_seed(99)
    target_model = torch.nn.Linear(3, 2, bias=False)
    target_trainer = _TrainerProbe(target_model, populated=False)

    real_load = checkpoint_core.load_safetensors_bytes
    decode_calls = 0

    def counting_load(payload: bytes) -> dict[str, Any]:
        nonlocal decode_calls
        decode_calls += 1
        return real_load(payload)

    monkeypatch.setattr(checkpoint_core, "load_safetensors_bytes", counting_load)

    result = load_trainer_checkpoint(
        checkpoint,
        model=target_model,
        trainer=target_trainer,
        restore_rng=False,
    )

    # One verified checkpoint decode reads exactly two SafeTensors payloads:
    # model weights and state tensors. The previous trainer path read both twice.
    assert decode_calls == 2
    assert target_trainer.loads == 1
    assert result.manifest["identity"]["step"] == 1
    assert result.trainer_state["optimizer_step"] == 1
    for name, tensor in source_model.state_dict().items():
        torch.testing.assert_close(target_model.state_dict()[name], tensor, rtol=0, atol=0)


def test_identity_mismatch_fails_before_payload_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    source_model = torch.nn.Linear(3, 2, bias=False)
    source_trainer = _TrainerProbe(source_model, populated=True)
    checkpoint = tmp_path / "identity-first"
    save_trainer_checkpoint(
        checkpoint,
        model=source_model,
        trainer=source_trainer,
        identity=_identity(sum(parameter.numel() for parameter in source_model.parameters())),
    )

    target_model = torch.nn.Linear(3, 2, bias=False)
    target_trainer = _TrainerProbe(target_model, populated=False)
    before = {name: tensor.detach().clone() for name, tensor in target_model.state_dict().items()}

    def forbidden_decode(_: bytes) -> dict[str, Any]:
        raise AssertionError("payload decoding must not start after identity mismatch")

    monkeypatch.setattr(checkpoint_core, "load_safetensors_bytes", forbidden_decode)

    with pytest.raises(Exception, match="checkpoint identity mismatch"):
        load_trainer_checkpoint(
            checkpoint,
            model=target_model,
            trainer=target_trainer,
            restore_rng=False,
            expected_git_sha="1" * 40,
        )

    for name, tensor in target_model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    assert target_trainer.loads == 0
