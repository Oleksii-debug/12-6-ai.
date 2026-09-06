from __future__ import annotations

import copy
import sys
import types
from importlib import metadata

import pytest
import torch

from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.training import TrainerConfig
from twelve_six.training.schedulefree import (
    SCHEDULEFREE_BINDING,
    ScheduleFreeTrainer,
)


class FakeScheduleFree(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        *,
        lr,
        betas,
        eps,
        weight_decay,
        foreach,
        warmup_steps,
        r,
        weight_lr_power,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "foreach": foreach,
            "warmup_steps": warmup_steps,
            "r": r,
            "weight_lr_power": weight_lr_power,
            "train_mode": False,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            for parameter in group["params"]:
                state = self.state[parameter]
                if "z" in state:
                    parameter.lerp_(state["z"], 1 - beta1)
            group["train_mode"] = True

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            for parameter in group["params"]:
                state = self.state[parameter]
                if "z" in state:
                    parameter.lerp_(state["z"], 1 - 1 / beta1)
            group["train_mode"] = False

    @torch.no_grad()
    def step(self, closure=None):
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError("fake optimizer not in train mode")
        for group in self.param_groups:
            lr = group["lr"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                parameter.add_(parameter.grad, alpha=-lr)
                state["z"] = parameter.detach().clone().add_(0.25)


def repeating_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3], [1, 2, 3, 0, 1, 2, 3, 0]],
            dtype=torch.long,
        )
    }


def tiny_model() -> TwelveSixDecoder:
    torch.manual_seed(17)
    return TwelveSixDecoder(
        ModelSpec(
            vocab_size=4,
            d_model=16,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            ffn_hidden=32,
            max_seq_len=16,
        )
    )


def install_fake_schedulefree(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("schedulefree")
    module.AdamWScheduleFree = FakeScheduleFree
    monkeypatch.setitem(sys.modules, "schedulefree", module)
    real_version = metadata.version

    def fake_version(name: str) -> str:
        if name == "schedulefree":
            return SCHEDULEFREE_BINDING["package_version"]
        return real_version(name)

    monkeypatch.setattr(metadata, "version", fake_version)


def config(**kwargs) -> TrainerConfig:
    defaults = dict(
        learning_rate=0.01,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        eps=1e-8,
        scheduler="constant",
        warmup_steps=0,
        precision="fp32",
        grad_clip_norm=1.0,
        gradient_accumulation_steps=1,
        deterministic_algorithms=True,
    )
    defaults.update(kwargs)
    return TrainerConfig(**defaults)


def test_missing_schedulefree_dependency_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "schedulefree", None)
    with pytest.raises(RuntimeError, match="schedulefree==1.4.1 is required"):
        ScheduleFreeTrainer(tiny_model(), config())


def test_wrong_schedulefree_version_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("schedulefree")
    module.AdamWScheduleFree = FakeScheduleFree
    monkeypatch.setitem(sys.modules, "schedulefree", module)
    real_version = metadata.version

    def fake_version(name: str) -> str:
        if name == "schedulefree":
            return "9.9.9"
        return real_version(name)

    monkeypatch.setattr(metadata, "version", fake_version)
    with pytest.raises(RuntimeError, match="exactly 1.4.1"):
        ScheduleFreeTrainer(tiny_model(), config())


def test_scheduler_and_warmup_must_remain_external_noops(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_schedulefree(monkeypatch)
    with pytest.raises(ValueError, match="constant external scheduler"):
        ScheduleFreeTrainer(tiny_model(), config(scheduler="cosine"))
    with pytest.raises(ValueError, match="warmup_steps=0"):
        ScheduleFreeTrainer(tiny_model(), config(warmup_steps=1))


def test_train_step_enters_train_mode_and_state_dict_returns_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_schedulefree(monkeypatch)
    trainer = ScheduleFreeTrainer(tiny_model(), config())
    assert all(group["train_mode"] is False for group in trainer.optimizer.param_groups)

    metrics = trainer.train_batch(repeating_batch())
    assert metrics.optimizer_stepped is True
    assert all(group["train_mode"] is True for group in trainer.optimizer.param_groups)

    state = trainer.state_dict()
    assert all(group["train_mode"] is False for group in trainer.optimizer.param_groups)
    assert all(group["train_mode"] is False for group in state["optimizer_state"]["param_groups"])
    assert state["config"]["optimizer_binding"] == SCHEDULEFREE_BINDING
    assert trainer.model.training is False


def test_state_dict_rejects_mid_accumulation_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_schedulefree(monkeypatch)
    trainer = ScheduleFreeTrainer(tiny_model(), config(gradient_accumulation_steps=2))
    metrics = trainer.train_batch(repeating_batch())
    assert metrics.optimizer_stepped is False
    with pytest.raises(RuntimeError, match="checkpoint-safe"):
        trainer.state_dict()


def test_load_requires_exact_binding_and_eval_mode_state(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_schedulefree(monkeypatch)
    trainer = ScheduleFreeTrainer(tiny_model(), config())
    trainer.train_batch(repeating_batch())
    state = trainer.state_dict()

    bad_binding = copy.deepcopy(state)
    bad_binding["config"]["optimizer_binding"]["package_version"] = "9.9.9"
    fresh = ScheduleFreeTrainer(tiny_model(), config())
    with pytest.raises(ValueError, match="optimizer binding mismatch"):
        fresh.load_state_dict(bad_binding)

    bad_mode = copy.deepcopy(state)
    bad_mode["optimizer_state"]["param_groups"][0]["train_mode"] = True
    with pytest.raises(ValueError, match="eval-mode"):
        fresh.load_state_dict(bad_mode)


def test_fresh_object_resume_matches_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_schedulefree(monkeypatch)
    base = tiny_model()
    model_a = tiny_model()
    model_b = tiny_model()
    model_a.load_state_dict(base.state_dict())
    model_b.load_state_dict(base.state_dict())

    trainer_a = ScheduleFreeTrainer(model_a, config())
    trainer_a.train_batch(repeating_batch())
    checkpoint = trainer_a.state_dict()
    model_checkpoint = copy.deepcopy(model_a.state_dict())

    trainer_a.train_batch(repeating_batch())
    final_a = copy.deepcopy(model_a.state_dict())

    model_b.load_state_dict(model_checkpoint)
    trainer_b = ScheduleFreeTrainer(model_b, config())
    trainer_b.load_state_dict(checkpoint)
    assert all(group["train_mode"] is False for group in trainer_b.optimizer.param_groups)
    trainer_b.train_batch(repeating_batch())

    for name, tensor in final_a.items():
        assert torch.equal(tensor, model_b.state_dict()[name]), name
