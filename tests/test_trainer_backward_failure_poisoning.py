"""Adversarial D02 backward interruption tests (synthetic inputs, no training credit)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from twelve_six.training import Trainer, TrainerConfig, TrainingStateInvalidError


class _FaultDuringBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, failure_type):
        ctx.failure_type = failure_type
        return values.clone()

    @staticmethod
    def backward(ctx, incoming):
        raise ctx.failure_type("synthetic backward interruption")


class _TinyLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
        self.failure_type = None

    def forward(self, input_ids):
        values = self.weight
        if self.failure_type is not None:
            values = _FaultDuringBackward.apply(values, self.failure_type)
        return values.reshape(1, 1, 3).expand(*input_ids.shape, 3)


_BATCH = {
    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
    "target_ids": torch.tensor([[1, 2]], dtype=torch.long),
}


@pytest.mark.parametrize("failure_type", [ValueError, KeyboardInterrupt])
def test_non_runtime_backward_failure_poisoned_and_cannot_replay(failure_type):
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )

    first = trainer.train_microbatch(_BATCH)
    assert first.micro_step == 1
    assert first.optimizer_step == 0
    assert model.weight.grad is not None  # One accumulation microbatch is pending.
    before_weights = model.weight.detach().clone()

    model.failure_type = failure_type
    with pytest.raises(failure_type, match="synthetic backward interruption"):
        trainer.train_microbatch(_BATCH)

    assert trainer.micro_step == 1
    assert trainer.optimizer_step == 0
    assert trainer.tokens_seen == 2
    torch.testing.assert_close(model.weight, before_weights, rtol=0, atol=0)
    assert model.weight.grad is None  # Partial and prior gradients were discarded.

    model.failure_type = None
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.train_microbatch(_BATCH)
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.assert_checkpoint_safe()
    with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
        trainer.state_dict()


def test_clean_backward_still_updates_after_exact_accumulation_boundary():
    model = _TinyLogitModel()
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, gradient_accumulation_steps=2, seed=17),
    )
    before = model.weight.detach().clone()

    first = trainer.train_microbatch(_BATCH)
    second = trainer.train_microbatch(_BATCH)

    assert first.optimizer_stepped is False
    assert second.optimizer_stepped is True
    assert trainer.micro_step == 2
    assert trainer.optimizer_step == 1
    assert trainer.tokens_seen == 4
    assert not torch.equal(before, model.weight)
    assert trainer.state_dict().optimizer_step == 1
