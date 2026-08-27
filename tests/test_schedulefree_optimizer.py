from __future__ import annotations

import copy
from importlib import metadata as importlib_metadata

import pytest
import torch
from torch import nn
from torch.optim import Optimizer

from twelve_six.training import Trainer, TrainerConfig
from twelve_six.training import schedulefree as sf


class ToyBigramLM(nn.Module):
    def __init__(self, vocab_size: int = 4) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.table(input_ids)


class FakeScheduleFreeAdamW(Optimizer):
    """Small mode-faithful test double; not a numerical Schedule-Free implementation."""

    def __init__(
        self,
        params,
        *,
        lr,
        betas,
        eps,
        weight_decay,
        warmup_steps,
        inner_momentum,
        foreach,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps,
            "inner_momentum": inner_momentum,
            "foreach": foreach,
            "train_mode": False,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def train(self) -> None:
        for group in self.param_groups:
            if not group["train_mode"]:
                beta1, _ = group["betas"]
                for parameter in group["params"]:
                    state = self.state[parameter]
                    if "z" in state:
                        parameter.lerp_(state["z"], 1 - beta1)
                group["train_mode"] = True

    @torch.no_grad()
    def eval(self) -> None:
        for group in self.param_groups:
            if group["train_mode"]:
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
        return None


def repeating_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3], [1, 2, 3, 0, 1, 2, 3, 0]],
            dtype=torch.long,
        )
    }


def patch_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sf, "_load_schedulefree_optimizer_class", lambda: FakeScheduleFreeAdamW)


def test_exact_dependency_missing_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> str:
        raise importlib_metadata.PackageNotFoundError("schedulefree")

    monkeypatch.setattr(sf.importlib_metadata, "version", missing)
    with pytest.raises(sf.ScheduleFreeDependencyError, match="schedulefree==1.4.1"):
        sf._load_schedulefree_optimizer_class()


def test_wrong_dependency_version_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sf.importlib_metadata, "version", lambda _name: "1.4.0")
    with pytest.raises(sf.ScheduleFreeDependencyError, match="version mismatch"):
        sf._load_schedulefree_optimizer_class()


@pytest.mark.parametrize(
    "config",
    [
        TrainerConfig(max_steps=1, scheduler="cosine"),
        TrainerConfig(max_steps=1, warmup_steps=1, scheduler="linear_warmup"),
    ],
)
def test_nonmatched_scheduler_or_warmup_rejected_before_import(
    monkeypatch: pytest.MonkeyPatch,
    config: TrainerConfig,
) -> None:
    monkeypatch.setattr(
        sf,
        "_load_schedulefree_optimizer_class",
        lambda: pytest.fail("dependency import should not run"),
    )
    with pytest.raises(sf.ScheduleFreeConfigError):
        sf.build_schedulefree_adamw(ToyBigramLM(), config)


def test_builder_freezes_candidate_specific_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fake(monkeypatch)
    config = TrainerConfig(
        learning_rate=2.2e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        max_steps=1,
        gradient_clip_norm=1.0,
        scheduler="constant",
        warmup_steps=0,
    )
    optimizer = sf.build_schedulefree_adamw(ToyBigramLM(), config)
    group = optimizer.param_groups[0]
    assert group["lr"] == 2.2e-4
    assert group["betas"] == (0.9, 0.95)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0.1
    assert group["warmup_steps"] == 0
    assert group["inner_momentum"] == 0.0
    assert group["foreach"] is False
    assert group["train_mode"] is False


def test_trainer_enters_optimizer_train_mode_before_step(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fake(monkeypatch)
    trainer = sf.ScheduleFreeTrainer(ToyBigramLM(), TrainerConfig(max_steps=1))
    with pytest.raises(RuntimeError, match="not in train mode"):
        trainer.optimizer.step()
    metrics = trainer.train_microbatch(repeating_batch())
    assert metrics.optimizer_stepped is True
    assert trainer.optimizer.param_groups[0]["train_mode"] is True


def test_state_dict_switches_to_eval_and_binds_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fake(monkeypatch)
    trainer = sf.ScheduleFreeTrainer(ToyBigramLM(), TrainerConfig(max_steps=2, seed=19))
    trainer.train_microbatch(repeating_batch())
    state = trainer.state_dict()
    assert trainer.optimizer.param_groups[0]["train_mode"] is False
    assert trainer.model.training is False
    assert state.optimizer["param_groups"][0]["train_mode"] is False
    assert state.config["_optimizer_binding"] == sf.schedulefree_optimizer_binding()


def test_fresh_object_resume_preserves_next_update(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fake(monkeypatch)
    torch.manual_seed(23)
    config = TrainerConfig(max_steps=3, seed=23, learning_rate=0.02, gradient_clip_norm=None)
    source_model = ToyBigramLM()
    source = sf.ScheduleFreeTrainer(source_model, config)
    source.train_microbatch(repeating_batch())
    source_state = source.state_dict()
    checkpoint_model_state = copy.deepcopy(source_model.state_dict())

    target_model = ToyBigramLM()
    target_model.load_state_dict(checkpoint_model_state)
    target = sf.ScheduleFreeTrainer(target_model, config)
    target.load_state_dict(source_state)

    source.train_microbatch(repeating_batch())
    target.train_microbatch(repeating_batch())
    assert source.optimizer_step == target.optimizer_step == 2
    for source_parameter, target_parameter in zip(
        source_model.parameters(), target_model.parameters(), strict=True
    ):
        torch.testing.assert_close(source_parameter, target_parameter)


def test_malformed_train_mode_rejected_before_load(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_fake(monkeypatch)
    config = TrainerConfig(max_steps=2)
    source = sf.ScheduleFreeTrainer(ToyBigramLM(), config)
    source.train_microbatch(repeating_batch())
    state = source.state_dict()
    malformed = copy.deepcopy(state)
    malformed.optimizer["param_groups"][0]["train_mode"] = True

    target = sf.ScheduleFreeTrainer(ToyBigramLM(), config)
    before = copy.deepcopy(target.optimizer.state_dict())
    with pytest.raises(sf.ScheduleFreeStateError, match="not in eval mode"):
        target.load_state_dict(malformed)
    assert target.optimizer.state_dict() == before


def test_wrong_binding_rejected_and_plain_trainer_rejects_bound_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_fake(monkeypatch)
    config = TrainerConfig(max_steps=2)
    source = sf.ScheduleFreeTrainer(ToyBigramLM(), config)
    source.train_microbatch(repeating_batch())
    state = source.state_dict()

    wrong = copy.deepcopy(state)
    wrong.config["_optimizer_binding"]["package_version"] = "9.9.9"
    target = sf.ScheduleFreeTrainer(ToyBigramLM(), config)
    with pytest.raises(sf.ScheduleFreeStateError, match="binding mismatch"):
        target.load_state_dict(wrong)

    plain = Trainer(ToyBigramLM(), config)
    with pytest.raises(ValueError, match="config mismatch"):
        plain.load_state_dict(state)
