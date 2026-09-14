from __future__ import annotations

import pytest
import torch
from torch import nn

from twelve_six.training import Trainer, TrainerConfig


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = nn.Embedding(4, 4)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.table(input_ids)


def _batch() -> dict[str, torch.Tensor]:
    return {"input_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long)}


def test_direct_train_microbatch_refuses_update_past_max_steps() -> None:
    trainer = Trainer(TinyLM(), TrainerConfig(max_steps=1, learning_rate=0.05))

    trainer.train_microbatch(_batch())

    assert trainer.optimizer_step == 1
    trainer.assert_checkpoint_safe()
    with pytest.raises(RuntimeError, match="max_steps already reached"):
        trainer.train_microbatch(_batch())
    assert trainer.optimizer_step == 1
