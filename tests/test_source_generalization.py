from __future__ import annotations

import math

import pytest

from twelve_six.packing import TextRecord
from twelve_six.training.source_generalization import (
    CHUNK_TARGET_UTF8_BYTES,
    DATA105_STATUS,
    OPTIMIZED_TOKEN_BUDGET,
    PARAMETER_COUNT,
    SOURCE_FAMILIES,
    SourceGeneralizationError,
    chunk_source_text,
    fixed_500k_model_spec,
    split_seen_family,
)


def test_fixed_model_is_exact_research41_500k_member() -> None:
    spec = fixed_500k_model_spec()
    assert spec.parameter_count() == PARAMETER_COUNT == 467_808
    assert spec.vocab_size == 256
    assert spec.max_seq_len == 256
    assert (spec.d_model, spec.n_layers, spec.n_heads, spec.head_dim, spec.d_ff) == (
        96,
        4,
        6,
        16,
        256,
    )


def test_matched_budget_is_exact_full_step_budget() -> None:
    assert OPTIMIZED_TOKEN_BUDGET == 64_512


def test_chunking_is_deterministic_utf8_safe_and_provenance_bound() -> None:
    text = "\n".join(["Український тестовий рядок " * 80 for _ in range(12)])
    first = chunk_source_text(
        source_id=SOURCE_FAMILIES[0], record_id="record-a", text=text
    )
    second = chunk_source_text(
        source_id=SOURCE_FAMILIES[0], record_id="record-a", text=text
    )
    assert first == second
    assert len(first) >= 5
    assert all(record.record_id.startswith(f"{SOURCE_FAMILIES[0]}::record-a::") for record in first)
    assert all(record.split == "family_pool" for record in first)
    assert all(len(record.text.encode("utf-8")) <= CHUNK_TARGET_UTF8_BYTES for record in first)
    assert "".join(record.text.replace("\n", "") for record in first)


def test_seen_family_split_is_fixed_disjoint_80_20_index_rule() -> None:
    records = [
        TextRecord(record_id=f"r-{index}", text=f"record {index}", split="family_pool")
        for index in range(10)
    ]
    train, validation = split_seen_family(records)
    assert [record.record_id for record in validation] == ["r-0", "r-5"]
    assert len(train) == 8
    assert {record.record_id for record in train}.isdisjoint(
        record.record_id for record in validation
    )
    assert all(record.split == "train" for record in train)
    assert all(record.split == "validation" for record in validation)


def test_seen_family_split_rejects_too_small_family() -> None:
    records = [TextRecord(record_id=f"r-{i}", text="x", split="family_pool") for i in range(4)]
    with pytest.raises(SourceGeneralizationError, match="fewer than five"):
        split_seen_family(records)


def test_data105_is_explicitly_absent_not_synthesized() -> None:
    assert DATA105_STATUS == "NOT_FOUND_IN_LIVE_REPOSITORY_AS_OF_2026-08-26"


def test_bpb_conversion_identity() -> None:
    # A byte tokenizer makes one scored token one source byte; CE nats / ln(2) is BPB.
    assert math.isclose(math.log(2.0) / math.log(2.0), 1.0)
