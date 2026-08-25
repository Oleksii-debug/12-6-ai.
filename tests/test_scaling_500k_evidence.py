from __future__ import annotations

import math

import pytest
import torch

from twelve_six.model import InitSpec, TwelveSixDecoder
from twelve_six.scaling_500k_evidence import (
    DEFAULT_SEEDS,
    DEFAULT_TOKEN_BUDGETS,
    SHARED_COMPARISON_BUDGET,
    TARGET_PARAMETERS,
    _bpb,
    _generation_snapshot,
    _model_state_sha256,
    _target_spec,
)
from twelve_six.tokenization import ByteTokenizer


def test_target_is_exact_research41_467808_spec() -> None:
    spec = _target_spec()
    assert spec.parameter_count() == TARGET_PARAMETERS == 467_808
    assert spec.vocab_size == 256
    assert spec.max_seq_len == 256
    assert (spec.d_model, spec.n_layers, spec.n_heads, spec.head_dim, spec.d_ff) == (
        96,
        4,
        6,
        16,
        256,
    )


def test_default_run_preserves_shared_budget_and_two_seeds() -> None:
    assert DEFAULT_SEEDS == (1337, 1338)
    assert SHARED_COMPARISON_BUDGET == 65_536
    assert SHARED_COMPARISON_BUDGET in DEFAULT_TOKEN_BUDGETS
    assert DEFAULT_TOKEN_BUDGETS[-1] == 262_144


def test_byte_tokenizer_bpb_conversion_is_exact_definition() -> None:
    assert _bpb(math.log(2.0)) == pytest.approx(1.0)
    assert _bpb(2.1190375715199083) == pytest.approx(3.057124995889164)


def test_generation_snapshot_is_greedy_and_rng_free() -> None:
    torch.manual_seed(1337)
    model = TwelveSixDecoder(_target_spec(), InitSpec())
    tokenizer = ByteTokenizer()
    before = torch.random.get_rng_state().clone()
    first = _generation_snapshot(model, tokenizer, prompt="The ", max_new_tokens=4)
    after = torch.random.get_rng_state().clone()
    second = _generation_snapshot(model, tokenizer, prompt="The ", max_new_tokens=4)
    assert torch.equal(before, after)
    assert first == second
    assert first["decoding"] == "greedy_argmax"
    assert len(first["generated_token_ids"]) == 4


def test_model_state_digest_is_exact_and_seed_sensitive() -> None:
    torch.manual_seed(1337)
    left = TwelveSixDecoder(_target_spec(), InitSpec())
    torch.manual_seed(1337)
    same = TwelveSixDecoder(_target_spec(), InitSpec())
    torch.manual_seed(1338)
    different = TwelveSixDecoder(_target_spec(), InitSpec())
    assert _model_state_sha256(left) == _model_state_sha256(same)
    assert _model_state_sha256(left) != _model_state_sha256(different)
