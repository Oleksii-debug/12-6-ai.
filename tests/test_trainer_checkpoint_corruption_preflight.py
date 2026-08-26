from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from twelve_six.checkpoint import CheckpointCompatibilityError
from twelve_six.checkpoint import trainer_adapter


@dataclass(frozen=True)
class _Config:
    gradient_accumulation_steps: int = 2
    max_steps: int = 8
    learning_rate: float = 0.1


def _trainer_and_state():
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    config = _Config()
    trainer = SimpleNamespace(
        config=config,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
    )
    state = {
        "micro_step": 2,
        "optimizer_step": 1,
        "tokens_seen": 7,
        "optimizer": optimizer.state_dict(),
        "scheduler": None,
        "scaler": None,
        "config": {
            "gradient_accumulation_steps": 2,
            "max_steps": 8,
            "learning_rate": 0.1,
        },
    }
    return torch, trainer, state


def _manifest(*, step: int = 1, tokens_seen: int = 7):
    return {"identity": {"step": step, "tokens_seen": tokens_seen}}


def test_valid_trainer_owned_state_passes_preflight() -> None:
    _torch, trainer, state = _trainer_and_state()
    trainer_adapter._preflight_trainer_state(
        trainer,
        state,
        manifest=_manifest(),
    )


def test_wrong_shaped_nested_momentum_is_rejected() -> None:
    torch, trainer, state = _trainer_and_state()
    parameter_id = next(iter(state["optimizer"]["state"]))
    state["optimizer"]["state"][parameter_id]["momentum_buffer"] = torch.zeros(1)

    with pytest.raises(
        CheckpointCompatibilityError,
        match="optimizer state shape mismatch",
    ):
        trainer_adapter._preflight_trainer_state(
            trainer,
            state,
            manifest=_manifest(),
        )


def test_invalid_nested_counters_are_rejected() -> None:
    _torch, trainer, state = _trainer_and_state()
    state["optimizer_step"] = -1

    with pytest.raises(
        CheckpointCompatibilityError,
        match="optimizer_step must be a non-negative integer",
    ):
        trainer_adapter._preflight_trainer_state(
            trainer,
            state,
            manifest=_manifest(),
        )


def test_accumulation_boundary_is_rejected_before_resume() -> None:
    _torch, trainer, state = _trainer_and_state()
    state["micro_step"] = 3

    with pytest.raises(
        CheckpointCompatibilityError,
        match="complete committed accumulation boundary",
    ):
        trainer_adapter._preflight_trainer_state(
            trainer,
            state,
            manifest=_manifest(),
        )


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (_manifest(step=2), "optimizer_step disagrees"),
        (_manifest(tokens_seen=8), "tokens_seen disagrees"),
    ],
)
def test_trainer_progress_must_match_checkpoint_identity(manifest, message: str) -> None:
    _torch, trainer, state = _trainer_and_state()

    with pytest.raises(CheckpointCompatibilityError, match=message):
        trainer_adapter._preflight_trainer_state(
            trainer,
            state,
            manifest=manifest,
        )


def test_public_trainer_load_rejects_corruption_before_model_loader(monkeypatch) -> None:
    torch, trainer, state = _trainer_and_state()
    parameter_id = next(iter(state["optimizer"]["state"]))
    state["optimizer"]["state"][parameter_id]["momentum_buffer"] = torch.zeros(1)

    verified = SimpleNamespace(manifest=_manifest())
    called = {"model_loader": False, "trainer_loader": False}

    monkeypatch.setattr(
        trainer_adapter,
        "prepare_checkpoint_load",
        lambda _directory: verified,
    )
    monkeypatch.setattr(
        trainer_adapter,
        "_decode_verified_state",
        lambda _verified: ({}, {"trainer": state}),
    )

    def _model_loader(*_args, **_kwargs):
        called["model_loader"] = True
        raise AssertionError("model loader must not run after failed trainer preflight")

    monkeypatch.setattr(trainer_adapter, "load_verified_checkpoint", _model_loader)
    trainer.load_state_dict = lambda _state: called.__setitem__("trainer_loader", True)

    with pytest.raises(
        CheckpointCompatibilityError,
        match="optimizer state shape mismatch",
    ):
        trainer_adapter.load_trainer_checkpoint(
            "unused-checkpoint-path",
            model=object(),
            trainer=trainer,
            restore_rng=False,
        )

    assert called == {"model_loader": False, "trainer_loader": False}
