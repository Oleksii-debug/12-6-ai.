from __future__ import annotations

import math

import pytest
import torch

from twelve_six.context_scaling import (
    ContextPackingSpec,
    context_probe_spec,
    estimate_context_cost,
    isolated_document_packing_estimate,
    measure_context_candidate_packing,
)
from twelve_six.distributed import ModelScaleSpec, ParallelPlan, estimate_training_memory
from twelve_six.model import ModelSpec, RotaryEmbedding, TwelveSixDecoder, apply_rope
from twelve_six.packing import TextRecord, measure_packed_split
from twelve_six.tokenization import ByteTokenizer


def _s0_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=128,
        d_model=20,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        head_dim=10,
        d_ff=56,
        rope_rotary_dim=10,
    )


def _s2_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=2048,
        max_seq_len=512,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        head_dim=32,
        d_ff=352,
        rope_rotary_dim=32,
    )


def test_kv_cache_formula_matches_model_native_unexpanded_layout() -> None:
    estimate = estimate_context_cost(
        _s2_spec(),
        sequence_length=512,
        batch_size=1,
        activation_element_bytes=2,
        kv_element_bytes=2,
    )
    assert estimate.kv_cache_bytes == 1_048_576
    assert estimate.kv_cache_elements == 2 * 4 * 4 * 512 * 32


def test_dense_attention_score_equivalent_is_quadratic_in_context() -> None:
    spec = _s2_spec()
    c512 = estimate_context_cost(spec, sequence_length=512)
    c1024 = estimate_context_cost(
        spec,
        sequence_length=1024,
        enforce_model_limit=False,
    )
    assert c1024.attention_score_equivalent_bytes == 4 * c512.attention_score_equivalent_bytes
    assert c1024.kv_cache_bytes == 2 * c512.kv_cache_bytes


def test_distributed_memory_planner_exposes_quadratic_attention_term() -> None:
    c256 = ModelScaleSpec(1_000_000, 128, 4, 4, 256)
    c512 = ModelScaleSpec(1_000_000, 128, 4, 4, 512)
    m256 = estimate_training_memory(c256, ParallelPlan())
    m512 = estimate_training_memory(c512, ParallelPlan())
    assert m256.attention_score_equivalent_bytes_per_rank > 0
    assert m512.attention_score_equivalent_bytes_per_rank == (
        4 * m256.attention_score_equivalent_bytes_per_rank
    )
    historical = estimate_training_memory(
        c512,
        ParallelPlan(),
        attention_memory_mode="linear_only",
    )
    assert historical.attention_score_equivalent_bytes_per_rank == 0
    assert historical.activation_bytes_per_rank == historical.linear_activation_bytes_per_rank


def test_probe_spec_changes_identity_without_mutating_s0() -> None:
    canonical = _s0_spec()
    probe = context_probe_spec(canonical, max_seq_len=512)
    assert canonical.max_seq_len == 128
    assert probe.max_seq_len == 512
    assert probe.identity_sha256() != canonical.identity_sha256()
    assert probe.parameter_count() == canonical.parameter_count()


def test_context_cost_enforces_checkpoint_limit_by_default() -> None:
    with pytest.raises(ValueError, match="exceeds ModelSpec max_seq_len"):
        estimate_context_cost(_s0_spec(), sequence_length=256)


def test_current_decoder_mechanically_executes_identity_distinct_256_probe() -> None:
    probe = context_probe_spec(_s0_spec(), max_seq_len=256)
    torch.manual_seed(126)
    model = TwelveSixDecoder(probe)
    ids = torch.randint(0, probe.vocab_size, (1, probe.max_seq_len), dtype=torch.long)
    output = model(ids)
    assert output.logits.shape == (1, 256, probe.vocab_size)


def test_rope_absolute_offset_matches_slice_of_full_positions() -> None:
    rope = RotaryEmbedding(rotary_dim=10, theta=10_000.0)
    full_cos, full_sin = rope.cos_sin(
        8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    tail_cos, tail_sin = rope.cos_sin(
        3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        position_offset=5,
    )
    assert torch.equal(tail_cos, full_cos[5:8])
    assert torch.equal(tail_sin, full_sin[5:8])


def test_partial_rotary_leaves_unrotated_head_tail_unchanged() -> None:
    rope = RotaryEmbedding(rotary_dim=6, theta=10_000.0)
    torch.manual_seed(126)
    x = torch.randn(1, 2, 4, 10)
    cos, sin = rope.cos_sin(4, device=x.device, dtype=x.dtype)
    rotated = apply_rope(x, cos, sin, rotary_dim=6)
    assert torch.equal(rotated[..., 6:], x[..., 6:])
    assert not torch.equal(rotated[..., :6], x[..., :6])


def test_s0_fixture_isolated_packing_utilization_falls_with_longer_blocks() -> None:
    # Exact UTF-8 byte-token lengths of the 12-document controlled S0 fixture.
    lengths = (141, 134, 110, 138, 145, 258, 237, 237, 269, 251, 143, 263)
    c128 = isolated_document_packing_estimate(lengths, sequence_length=128)
    c256 = isolated_document_packing_estimate(lengths, sequence_length=256)
    c512 = isolated_document_packing_estimate(lengths, sequence_length=512)

    assert c128.unique_next_token_pairs == 2314
    assert c128.emitted_blocks == 26
    assert c256.emitted_blocks == 15
    assert c512.emitted_blocks == 12
    assert c128.pair_utilization == pytest.approx(2314 / (26 * 127))
    assert c256.pair_utilization == pytest.approx(2314 / (15 * 255))
    assert c512.pair_utilization == pytest.approx(2314 / (12 * 511))
    assert c128.pair_utilization > c256.pair_utilization > c512.pair_utilization


def test_context_candidate_packing_has_distinct_identity_without_weakening_s0() -> None:
    tokenizer = ByteTokenizer()
    records = (
        TextRecord("a", "a" * 300, "train"),
        TextRecord("b", "b" * 90, "train"),
    )
    packing256 = ContextPackingSpec(sequence_length=256)
    packing512 = ContextPackingSpec(sequence_length=512)
    assert packing256.identity_sha256(
        tokenizer_config_sha256=tokenizer.identity.config_sha256
    ) != packing512.identity_sha256(
        tokenizer_config_sha256=tokenizer.identity.config_sha256
    )

    measurement = measure_context_candidate_packing(
        records,
        tokenizer,
        packing_spec=packing256,
        dataset_id="context-fixture-v1",
        dataset_identity_sha256="a" * 64,
        source_jsonl_sha256="b" * 64,
        split="train",
    )
    assert measurement.sequence_length == 256
    assert measurement.document_count == 2
    assert measurement.token_count == 390
    assert measurement.causal_loss_token_count == 388
    assert measurement.packed_example_count == 3
    assert measurement.token_length_p50 == 90
    assert measurement.token_length_p90 == 300
    assert measurement.causal_pair_utilization == pytest.approx(388 / (3 * 255))
    assert len(measurement.identity_sha256()) == 64

    with pytest.raises(ValueError, match="sequence_length must be 128"):
        measure_packed_split(
            records,
            tokenizer,
            dataset_id="context-fixture-v1",
            dataset_identity_sha256="a" * 64,
            source_jsonl_sha256="b" * 64,
            split="train",
            sequence_length=256,
        )


def test_short_documents_do_not_invent_loss_pairs() -> None:
    estimate = isolated_document_packing_estimate((1, 2, 3), sequence_length=8)
    assert estimate.document_count == 3
    assert estimate.unique_next_token_pairs == 3
    assert estimate.emitted_blocks == 2
    assert math.isclose(estimate.pair_utilization, 3 / 14)
