from __future__ import annotations

import torch

from tools.run_milestone100_first_learned_base import (
    BATCH_SIZE,
    FINAL_STEPS,
    RESUME_STEP,
    SEQUENCE_LENGTH,
    _model_spec,
    _state_hash,
    _trainer_config,
)
from twelve_six.model import InitSpec, TwelveSixDecoder, count_trainable_parameters


def test_exact_100k_model_identity() -> None:
    spec = _model_spec()
    assert spec.vocab_size == 512
    assert spec.parameter_count() == 107_856
    model = TwelveSixDecoder(spec, InitSpec())
    assert count_trainable_parameters(model) == 107_856


def test_training_plan_has_real_resume_boundary() -> None:
    config = _trainer_config()
    assert config.max_steps == FINAL_STEPS == 1536
    assert 0 < RESUME_STEP < FINAL_STEPS
    assert RESUME_STEP == 768
    assert config.precision == "fp32"
    assert config.gradient_accumulation_steps == 1
    assert BATCH_SIZE == 8
    assert SEQUENCE_LENGTH == 128


def test_scratch_initialization_is_seed_reproducible_and_not_constant() -> None:
    spec = _model_spec()
    init = InitSpec()
    torch.manual_seed(1337)
    first = TwelveSixDecoder(spec, init)
    first_hash = _state_hash(first)
    torch.manual_seed(1337)
    second = TwelveSixDecoder(spec, init)
    second_hash = _state_hash(second)
    torch.manual_seed(1338)
    third = TwelveSixDecoder(spec, init)
    third_hash = _state_hash(third)
    assert first_hash == second_hash
    assert third_hash != first_hash
