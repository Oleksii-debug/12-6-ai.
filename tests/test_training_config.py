from __future__ import annotations

import pytest

from twelve_six.training import TrainerConfig


@pytest.mark.parametrize(
    "override",
    [
        {"learning_rate": float("nan")},
        {"weight_decay": float("inf")},
        {"eps": float("nan")},
        {"betas": (0.9, float("nan"))},
        {"max_steps": 1.5},
        {"warmup_steps": True},
        {"gradient_accumulation_steps": 0},
        {"gradient_clip_norm": float("inf")},
        {"scheduler": "polynomial"},
        {"precision": "tf32"},
        {"seed": -1},
        {"deterministic_algorithms": 1},
    ],
)
def test_invalid_training_config_fails_closed(override: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrainerConfig(**override)


def test_warmup_may_not_exceed_max_steps() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainerConfig(max_steps=2, warmup_steps=3)
