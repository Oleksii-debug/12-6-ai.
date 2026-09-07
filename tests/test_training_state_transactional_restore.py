from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.optim import AdamW

from twelve_six.training import Trainer, TrainerConfig


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = nn.Embedding(4, 4)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.table(input_ids)


class FailOnceAdamW(AdamW):
    def __init__(self, params, *, lr: float) -> None:
        super().__init__(params, lr=lr)
        self.fail_next_load = False

    def load_state_dict(self, state_dict):
        if self.fail_next_load:
            self.param_groups[0]["lr"] = 99.0
            self.fail_next_load = False
            raise RuntimeError("injected optimizer restore failure")
        return super().load_state_dict(state_dict)


def test_scheduler_shape_mismatch_is_rejected_before_optimizer_mutation() -> None:
    config = TrainerConfig(
        learning_rate=0.1,
        max_steps=4,
        warmup_steps=1,
        scheduler="cosine",
        seed=17,
    )
    source = Trainer(TinyLM(), config)
    incoming = source.state_dict()
    incoming.optimizer["param_groups"][0]["lr"] = 0.777
    incoming = replace(incoming, scheduler=None)

    target = Trainer(TinyLM(), config)
    before_lr = target.optimizer.param_groups[0]["lr"]

    with pytest.raises(ValueError, match="scheduler state/config mismatch"):
        target.load_state_dict(incoming)

    assert target.optimizer.param_groups[0]["lr"] == before_lr
    assert target.micro_step == 0
    assert target.optimizer_step == 0
    assert target.tokens_seen == 0


def test_optimizer_restore_exception_rolls_back_partial_optimizer_mutation() -> None:
    config = TrainerConfig(
        learning_rate=0.1,
        max_steps=4,
        scheduler="constant",
        seed=23,
    )
    source = Trainer(TinyLM(), config)
    incoming = source.state_dict()
    incoming.optimizer["param_groups"][0]["lr"] = 0.321

    target_model = TinyLM()
    optimizer = FailOnceAdamW(target_model.parameters(), lr=0.1)
    target = Trainer(target_model, config, optimizer=optimizer)
    before_lr = target.optimizer.param_groups[0]["lr"]
    optimizer.fail_next_load = True

    with pytest.raises(RuntimeError, match="injected optimizer restore failure"):
        target.load_state_dict(incoming)

    assert target.optimizer.param_groups[0]["lr"] == before_lr
    assert target.micro_step == 0
    assert target.optimizer_step == 0
    assert target.tokens_seen == 0
    target.assert_checkpoint_safe()
