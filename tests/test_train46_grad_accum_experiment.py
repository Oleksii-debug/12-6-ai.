from __future__ import annotations

import torch

from twelve_six.train46_grad_accum_experiment import (
    ACCUM_MICROBATCH_SIZE,
    BETA2,
    EFFECTIVE_BATCH_SIZE,
    FULL_MICROBATCH_SIZE,
    LOSS_ABS_TOL,
    OPTIMIZER_STATE_MAX_ABS_TOL,
    PARAMETER_MAX_ABS_TOL,
    PARAMETER_REL_L2_TOL,
    _aligned_effective_batch,
    _config,
    _equivalence_pass,
)


def test_train46_keeps_incumbent_optimizer_identity() -> None:
    config = _config(accumulation=4)
    assert config.learning_rate == 3e-4
    assert config.betas == (0.9, BETA2) == (0.9, 0.95)
    assert config.weight_decay == 0.0
    assert config.gradient_clip_norm == 1.0
    assert config.warmup_steps == 0
    assert config.scheduler == "constant"
    assert config.precision == "fp32"
    assert config.gradient_accumulation_steps == 4


def test_train46_effective_batch_has_deliberately_unequal_valid_token_counts() -> None:
    stream = bytes(range(256)) * 8
    batch = _aligned_effective_batch(stream, update=0)
    assert batch["input_ids"].shape[0] == EFFECTIVE_BATCH_SIZE == 4
    assert batch["loss_mask"].dtype == torch.bool
    counts = [int(batch["loss_mask"][row].sum().item()) for row in range(4)]
    assert counts == [63, 62, 61, 60]
    assert sum(counts) == 246


def test_train46_compares_mathematically_equal_effective_batch_shapes() -> None:
    assert FULL_MICROBATCH_SIZE == EFFECTIVE_BATCH_SIZE == 4
    assert ACCUM_MICROBATCH_SIZE == 1
    assert EFFECTIVE_BATCH_SIZE // FULL_MICROBATCH_SIZE == 1
    assert EFFECTIVE_BATCH_SIZE // ACCUM_MICROBATCH_SIZE == 4


def test_train46_equivalence_gate_is_fail_closed() -> None:
    assert _equivalence_pass(
        token_match=True,
        parameter_max_abs=PARAMETER_MAX_ABS_TOL,
        parameter_relative_l2=PARAMETER_REL_L2_TOL,
        max_loss_diff=LOSS_ABS_TOL,
        optimizer_max_abs=OPTIMIZER_STATE_MAX_ABS_TOL,
    )
    assert not _equivalence_pass(
        token_match=False,
        parameter_max_abs=0.0,
        parameter_relative_l2=0.0,
        max_loss_diff=0.0,
        optimizer_max_abs=0.0,
    )
    assert not _equivalence_pass(
        token_match=True,
        parameter_max_abs=PARAMETER_MAX_ABS_TOL * 2,
        parameter_relative_l2=0.0,
        max_loss_diff=0.0,
        optimizer_max_abs=0.0,
    )
