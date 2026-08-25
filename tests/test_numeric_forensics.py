from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from twelve_six.checkpoint.trainer_adapter import save_trainer_checkpoint
from twelve_six.training import (
    NonFiniteTrainingError,
    Trainer,
    TrainerConfig,
    TrainingStateInvalidError,
    batch_identity_sha256,
)


class ToyBigramLM(nn.Module):
    def __init__(self, vocab_size: int = 4) -> None:
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.table(input_ids)


def repeating_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(
            [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [1, 2, 3, 0, 1, 2, 3, 0],
            ],
            dtype=torch.long,
        )
    }


class InjectNonFiniteAdamW(torch.optim.AdamW):
    """Test-only optimizer that poisons one parameter after a real AdamW update."""

    def __init__(self, params, *, poison: str) -> None:
        super().__init__(params, lr=0.05)
        self.poison = poison

    def step(self, closure=None):
        result = super().step(closure)
        value = float("nan") if self.poison == "nan" else float("inf")
        with torch.no_grad():
            parameter = self.param_groups[0]["params"][0]
            parameter.view(-1)[0] = value
        return result


def test_batch_identity_hash_ignores_non_tensor_metadata_and_raw_text() -> None:
    tokens = repeating_batch()["input_ids"]
    first = {"input_ids": tokens, "raw_text": "private training sentence"}
    second = {"input_ids": tokens, "raw_text": "different private sentence"}

    first_hash = batch_identity_sha256(first)
    second_hash = batch_identity_sha256(second)

    assert first_hash == second_hash
    assert len(first_hash) == 64
    assert "private" not in first_hash


def test_nonfinite_gradient_captures_bounded_parameter_identity() -> None:
    model = ToyBigramLM()
    model.table.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    trainer = Trainer(model, TrainerConfig(max_steps=1, precision="fp32"))

    with pytest.raises(NonFiniteTrainingError, match="gradient") as caught:
        trainer.train_microbatch(repeating_batch())

    diagnostics = caught.value.diagnostics
    assert diagnostics is not None
    assert diagnostics.kind == "gradient"
    assert diagnostics.optimizer_step == 0
    assert diagnostics.micro_step == 1
    assert diagnostics.tokens_seen == 14
    assert diagnostics.batch_tokens == 14
    assert diagnostics.precision == "fp32"
    assert diagnostics.gradient_norm is None
    assert diagnostics.gradient_norm_finite is False
    assert diagnostics.affected_parameter_names == ("table.weight",)
    assert diagnostics.affected_module_names == ("table",)
    assert diagnostics.affected_names_truncated is False
    assert diagnostics.raw_training_text_logged is False
    assert trainer.failure_diagnostics == diagnostics


@pytest.mark.parametrize("poison", ["nan", "inf"])
def test_nonfinite_update_is_not_logically_committed_and_blocks_checkpoint(
    poison: str, tmp_path
) -> None:
    torch.manual_seed(31)
    model = ToyBigramLM()
    optimizer = InjectNonFiniteAdamW(model.parameters(), poison=poison)
    trainer = Trainer(
        model,
        TrainerConfig(max_steps=1, precision="fp32", gradient_clip_norm=1.0, seed=31),
        optimizer=optimizer,
    )

    with pytest.raises(NonFiniteTrainingError, match="update") as caught:
        trainer.train_microbatch(repeating_batch())

    diagnostics = caught.value.diagnostics
    assert diagnostics is not None
    assert diagnostics.kind == "update"
    assert trainer.optimizer_step == 0
    assert diagnostics.optimizer_step == 0
    assert diagnostics.micro_step == 1
    assert diagnostics.tokens_seen == 14
    assert diagnostics.gradient_norm is not None
    assert math.isfinite(diagnostics.gradient_norm)
    assert diagnostics.gradient_norm_finite is True
    assert diagnostics.affected_parameter_names == ("table.weight",)
    assert diagnostics.affected_module_names == ("table",)
    assert len(diagnostics.batch_identity_sha256) == 64
    assert diagnostics.raw_training_text_logged is False

    with pytest.raises(TrainingStateInvalidError, match="fresh trainer"):
        trainer.train_microbatch(repeating_batch())
    with pytest.raises(TrainingStateInvalidError):
        trainer.state_dict()

    checkpoint_directory = tmp_path / "poisoned-checkpoint"
    with pytest.raises(TrainingStateInvalidError):
        save_trainer_checkpoint(
            checkpoint_directory,
            model=model,
            trainer=trainer,
            identity=None,  # state_dict must reject poison before publication uses identity
        )
    assert not checkpoint_directory.exists()


def test_activation_health_provider_is_failure_only_and_privacy_bounded() -> None:
    calls = 0

    def activation_health():
        nonlocal calls
        calls += 1
        return {
            "blocks.0.residual_rms": 1.25,
            "blocks.0.nonfinite": float("inf"),
            "unsafe_free_text": "do not copy this",
        }

    class NaNModel(ToyBigramLM):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return super().forward(input_ids) / torch.tensor(0.0)

    trainer = Trainer(
        NaNModel(),
        TrainerConfig(max_steps=1),
        activation_health_provider=activation_health,
    )
    assert calls == 0

    with pytest.raises(NonFiniteTrainingError) as caught:
        trainer.train_microbatch(repeating_batch())

    diagnostics = caught.value.diagnostics
    assert diagnostics is not None
    assert calls == 1
    assert diagnostics.activation_health == {
        "blocks.0.nonfinite": "nonfinite",
        "blocks.0.residual_rms": 1.25,
    }
