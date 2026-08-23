from __future__ import annotations

import pytest

from twelve_six.training import TrainerConfig


@pytest.mark.parametrize(
    ("override", "error_type"),
    [
        ({"learning_rate": float("nan")}, ValueError),
        ({"weight_decay": float("inf")}, ValueError),
        ({"eps": float("nan")}, ValueError),
        ({"betas": (0.9, float("nan"))}, ValueError),
        ({"max_steps": 1.5}, TypeError),
        ({"warmup_steps": True}, TypeError),
        ({"gradient_accumulation_steps": 0}, ValueError),
        ({"gradient_clip_norm": float("inf")}, ValueError),
        ({"scheduler": "polynomial"}, ValueError),
        ({"precision": "tf32"}, ValueError),
        ({"seed": -1}, ValueError),
        ({"deterministic_algorithms": 1}, TypeError),
    ],
)
def test_invalid_training_config_fails_closed(
    override: dict[str, object], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        TrainerConfig(**override)


def test_warmup_may_not_exceed_max_steps() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainerConfig(max_steps=2, warmup_steps=3)
