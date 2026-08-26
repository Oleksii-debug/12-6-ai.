from __future__ import annotations

from types import SimpleNamespace

import pytest

from twelve_six.checkpoint import CheckpointCompatibilityError
from twelve_six.checkpoint import trainer_adapter


def _live_trainer_with_sgd():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss = model(torch.ones(4, 3)).sum()
    loss.backward()
    optimizer.step()
    config = {
        "gradient_accumulation_steps": 1,
        "max_steps": 8,
        "precision": "fp32",
    }
    trainer = SimpleNamespace(
        config=config,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        load_state_dict=lambda _state: None,
    )
    return torch, model, trainer


def _valid_state(trainer):
    return {
        "micro_step": 1,
        "optimizer_step": 1,
        "tokens_seen": 32,
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": None,
        "scaler": None,
        "config": dict(trainer.config),
    }


def test_nested_wrong_shaped_momentum_rejects_before_production_model_load(monkeypatch) -> None:
    torch, model, trainer = _live_trainer_with_sgd()
    state = _valid_state(trainer)
    first_id = next(iter(state["optimizer"]["state"]))
    state["optimizer"]["state"][first_id]["momentum_buffer"] = torch.zeros(1)

    verified = SimpleNamespace(manifest={"identity": {}})
    monkeypatch.setattr(trainer_adapter, "prepare_checkpoint_load", lambda _path: verified)
    monkeypatch.setattr(
        trainer_adapter,
        "_decode_verified_state",
        lambda _verified: ({}, {"trainer": state}),
    )

    model_load_called = False

    def forbidden_model_load(*args, **kwargs):
        nonlocal model_load_called
        model_load_called = True
        raise AssertionError("model load must not run after failed trainer preflight")

    monkeypatch.setattr(trainer_adapter, "load_verified_checkpoint", forbidden_model_load)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="optimizer state shape mismatch"):
        trainer_adapter.load_trainer_checkpoint("unused", model=model, trainer=trainer)

    assert model_load_called is False
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [("micro_step", -1), ("optimizer_step", -1), ("tokens_seen", -1)],
)
def test_nested_negative_trainer_counters_fail_closed(field: str, value: int) -> None:
    _torch, _model, trainer = _live_trainer_with_sgd()
    state = _valid_state(trainer)
    state[field] = value

    with pytest.raises(CheckpointCompatibilityError, match=field):
        trainer_adapter._preflight_trainer_state(trainer, state)


def test_nested_trainer_accumulation_boundary_is_validated() -> None:
    _torch, _model, trainer = _live_trainer_with_sgd()
    trainer.config["gradient_accumulation_steps"] = 2
    state = _valid_state(trainer)
    state["config"] = dict(trainer.config)
    state["micro_step"] = 1

    with pytest.raises(CheckpointCompatibilityError, match="accumulation boundary"):
        trainer_adapter._preflight_trainer_state(trainer, state)


def test_nested_scaler_presence_mismatch_fails_closed() -> None:
    _torch, _model, trainer = _live_trainer_with_sgd()
    state = _valid_state(trainer)
    state["scaler"] = {"scale": 65536.0}

    with pytest.raises(CheckpointCompatibilityError, match="scaler state/config mismatch"):
        trainer_adapter._preflight_trainer_state(trainer, state)


def test_valid_nested_sgd_trainer_state_passes_preflight() -> None:
    _torch, _model, trainer = _live_trainer_with_sgd()
    trainer_adapter._preflight_trainer_state(trainer, _valid_state(trainer))
