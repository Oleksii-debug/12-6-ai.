from __future__ import annotations

from twelve_six.long_run_100k import (
    CAPACITY,
    EVALUATION_BUDGETS,
    FINAL_TOKENS,
    NO_IMPROVEMENT_MIN_TOKENS,
    NO_IMPROVEMENT_WINDOW,
    RESUME_TOKENS,
    RETAINED_CHECKPOINT_BUDGETS,
    _no_improvement,
    _should_sample_step,
    _trainer_config,
)
from twelve_six.scaling_experiment import controlled_specs


def test_long_run_preserves_exact_100k_control_geometry_and_recipe() -> None:
    spec = controlled_specs()[0]
    config = _trainer_config()
    assert spec.parameter_count() == 95_568
    assert spec.vocab_size == 256
    assert spec.max_seq_len == 256
    assert config.learning_rate == 3e-4
    assert config.betas == (0.9, 0.95)
    assert config.weight_decay == 0.0
    assert config.gradient_clip_norm == 1.0
    assert config.scheduler == "constant"
    assert config.warmup_steps == 0
    assert config.max_steps == FINAL_TOKENS // CAPACITY


def test_long_run_budgets_are_exact_and_resume_is_retained() -> None:
    assert FINAL_TOKENS % CAPACITY == 0
    assert RESUME_TOKENS % CAPACITY == 0
    assert RESUME_TOKENS in EVALUATION_BUDGETS
    assert RESUME_TOKENS in RETAINED_CHECKPOINT_BUDGETS
    assert FINAL_TOKENS in EVALUATION_BUDGETS
    assert FINAL_TOKENS in RETAINED_CHECKPOINT_BUDGETS
    assert list(EVALUATION_BUDGETS) == sorted(set(EVALUATION_BUDGETS))
    assert list(RETAINED_CHECKPOINT_BUDGETS) == sorted(set(RETAINED_CHECKPOINT_BUDGETS))


def test_step_telemetry_is_dense_early_and_progressively_spaced() -> None:
    assert all(_should_sample_step(step, step * CAPACITY) for step in range(1, 65))
    assert _should_sample_step(72, 72 * CAPACITY)
    assert not _should_sample_step(73, 73 * CAPACITY)
    assert _should_sample_step(544, 544 * CAPACITY)
    assert not _should_sample_step(545, 545 * CAPACITY)
    assert _should_sample_step(2_176, 2_176 * CAPACITY)
    assert not _should_sample_step(2_177, 2_177 * CAPACITY)


def _point(tokens: int, loss: float) -> dict[str, float | int]:
    return {"optimized_tokens": tokens, "validation_loss": loss}


def test_no_improvement_is_fail_closed_until_late_window() -> None:
    points = [
        _point(0, 5.0),
        _point(262_144, 4.0),
        _point(524_288, 3.5),
        _point(1_048_576, 3.4),
    ]
    assert not _no_improvement(points, 1_048_576)
    assert NO_IMPROVEMENT_WINDOW == 4
    assert NO_IMPROVEMENT_MIN_TOKENS >= RESUME_TOKENS


def test_no_improvement_requires_less_than_threshold_progress() -> None:
    plateau = [
        _point(0, 5.0),
        _point(524_288, 3.00000),
        _point(1_048_576, 2.99998),
        _point(1_310_720, 3.00002),
        _point(NO_IMPROVEMENT_MIN_TOKENS, 2.99999),
    ]
    assert _no_improvement(plateau, NO_IMPROVEMENT_MIN_TOKENS)

    improving = [
        _point(0, 5.0),
        _point(524_288, 3.0),
        _point(1_048_576, 2.9),
        _point(1_310_720, 2.8),
        _point(NO_IMPROVEMENT_MIN_TOKENS, 2.7),
    ]
    assert not _no_improvement(improving, NO_IMPROVEMENT_MIN_TOKENS)
