"""Named AdamW state and pre-backward torch-policy fences must coexist.

Synthetic CPU-only integration; no lawful corpus, trained weights, or scale credit.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from twelve_six.training import Trainer, TrainerConfig, TrainingStateInvalidError


class _NamedLogits(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor([0.1, -0.2, 0.3]))
        self.right = torch.nn.Parameter(torch.tensor([-0.1, 0.2, -0.3]))
        self.drift = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.drift:
            torch.use_deterministic_algorithms(False, warn_only=False)
        logits = self.left + 0.5 * self.right
        return logits.reshape(1, 1, 3).expand(*input_ids.shape, 3)


_BATCH = {
    "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
    "target_ids": torch.tensor([[1, 2]], dtype=torch.long),
}


@pytest.mark.parametrize("drift_stage", ["forward", "scale"])
def test_named_adamw_step_then_policy_drift_preserves_committed_state(
    monkeypatch: pytest.MonkeyPatch, drift_stage: str,
) -> None:
    py_before = random.getstate()
    np_before = np.random.get_state()
    cpu_before = torch.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    policy_before = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        model = _NamedLogits()
        config = TrainerConfig(
            seed=703, max_steps=2,
            deterministic_algorithms=True, deterministic_warn_only=True,
        )
        optimizer = torch.optim.AdamW(
            [{"params": [model.left]}, {"params": [model.right]}],
            lr=config.learning_rate,
        )
        trainer = Trainer(model, config, device="cpu", optimizer=optimizer)
        first = trainer.train_microbatch(_BATCH)
        assert first.optimizer_stepped
        assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
        trainer.assert_checkpoint_safe()
        state = trainer.state_dict()
        assert [group["param_names"] for group in state.optimizer["param_groups"]] == [
            ["left"], ["right"],
        ]
        assert all(trainer.optimizer.state[p]["exp_avg"].numel() == 3 for p in (
            model.left, model.right,
        ))

        before = {name: parameter.detach().clone() for name, parameter in (
            ("left", model.left), ("right", model.right),
        )}
        moments = {name: trainer.optimizer.state[parameter]["exp_avg"].clone() for (
            name, parameter,
        ) in (("left", model.left), ("right", model.right))}
        backward_calls: list[str] = []
        model.left.register_hook(lambda grad: backward_calls.append("left") or grad)
        model.right.register_hook(lambda grad: backward_calls.append("right") or grad)
        if drift_stage == "forward":
            model.drift = True
        else:
            native_scale = trainer.scaler.scale

            def drift_scale(value: torch.Tensor) -> torch.Tensor:
                result = native_scale(value)
                torch.use_deterministic_algorithms(False, warn_only=False)
                return result

            monkeypatch.setattr(trainer.scaler, "scale", drift_scale)

        with pytest.raises(TrainingStateInvalidError, match="deterministic"):
            trainer.train_microbatch(_BATCH)

        assert backward_calls == [], "either named parameter received a backward hook"
        assert model.left.grad is None and model.right.grad is None
        assert (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen) == (1, 1, 2)
        assert trainer._failure_reason is not None
        for name, parameter in (("left", model.left), ("right", model.right)):
            torch.testing.assert_close(parameter.detach(), before[name], rtol=0, atol=0)
            torch.testing.assert_close(
                trainer.optimizer.state[parameter]["exp_avg"], moments[name],
                rtol=0, atol=0,
            )
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.raises(TrainingStateInvalidError, match="verified checkpoint"):
            trainer.train_microbatch(_BATCH)
    finally:
        random.setstate(py_before)
        np.random.set_state(np_before)
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state_all(cuda_before)
        torch.use_deterministic_algorithms(
            policy_before[0], warn_only=policy_before[1],
        )
