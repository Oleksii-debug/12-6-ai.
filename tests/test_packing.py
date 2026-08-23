from __future__ import annotations

import json
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest

from twelve_six.packing import (
    DEFAULT_SEQUENCE_LENGTH,
    PACKING_CONFIG_HASH,
    DeterministicMixtureSampler,
    JsonlRecordError,
    SplitMixError,
    TextRecord,
    batch_examples,
    canonical_packing_config_json,
    collate_rows,
    deterministic_shard,
    iter_packed_examples,
    packing_config_hash,
    records_from_jsonl_lines,
)
from twelve_six.tokenization import ByteTokenizer


def _training_pairs(examples):
    pairs = []
    for example in examples:
        for index, keep in enumerate(example.loss_mask):
            if keep:
                pairs.append((example.input_ids[index], example.labels[index + 1]))
    return pairs


def test_packing_is_deterministic_and_preserves_all_within_document_pairs() -> None:
    tokenizer = ByteTokenizer()
    records = [
        TextRecord("r1", "abcdef", "train"),
        TextRecord("r2", "Ж🙂", "train"),
    ]
    kwargs = {
        "tokenizer": tokenizer,
        "expected_split": "train",
        "sequence_length": 4,
    }
    first = list(iter_packed_examples(records, **kwargs))
    second = list(iter_packed_examples(records, **kwargs))
    assert first == second

    expected_pairs = []
    for record in records:
        stream = tokenizer.encode(record.text)
        expected_pairs.extend(pairwise(stream))
    assert _training_pairs(first) == expected_pairs


def test_final_partial_block_is_masked_not_dropped() -> None:
    tokenizer = ByteTokenizer()
    records = [TextRecord("r1", "xyz", "train")]
    [example] = list(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split="train",
            sequence_length=8,
        )
    )
    assert example.input_ids[:3] == tuple(tokenizer.encode("xyz"))
    assert example.labels[:3] == tuple(tokenizer.encode("xyz"))
    assert example.labels[3:] == (-100, -100, -100, -100, -100)
    assert example.loss_mask == (1, 1, 0, 0, 0, 0, 0, 0)
    assert example.num_loss_tokens == 2


def test_full_blocks_overlap_one_token_for_d02_shifted_loss() -> None:
    tokenizer = ByteTokenizer()
    [first, second] = list(
        iter_packed_examples(
            [TextRecord("r1", "abcde", "train")],
            tokenizer,
            expected_split="train",
            sequence_length=4,
        )
    )
    assert first.input_ids == tuple(tokenizer.encode("abcd"))
    assert second.input_ids[:2] == tuple(tokenizer.encode("de"))
    encoded = tokenizer.encode("abcde")
    assert _training_pairs([first, second]) == list(pairwise(encoded))


def test_cross_document_packing_fails_without_semantic_eos() -> None:
    tokenizer = ByteTokenizer()
    with pytest.raises(ValueError, match="EOS"):
        list(
            iter_packed_examples(
                [TextRecord("r1", "abc", "train"), TextRecord("r2", "def", "train")],
                tokenizer,
                expected_split="train",
                cross_document=True,
            )
        )


def test_split_mixing_fails_closed() -> None:
    tokenizer = ByteTokenizer()
    records = [
        TextRecord("r1", "train", "train"),
        TextRecord("r2", "validation", "validation"),
    ]
    with pytest.raises(SplitMixError):
        list(iter_packed_examples(records, tokenizer, expected_split="train"))


def test_batching_and_collation_keep_tail_by_default() -> None:
    tokenizer = ByteTokenizer()
    records = [TextRecord(str(i), "abcd", "train") for i in range(5)]
    examples = list(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split="train",
            sequence_length=4,
        )
    )
    batches = list(batch_examples(examples, batch_size=4))
    assert sum(len(batch) for batch in batches) == len(examples)

    rows = collate_rows(batches[0])
    assert set(rows) == {"input_ids", "labels", "attention_mask", "loss_mask"}
    assert len(rows["input_ids"]) == len(batches[0])
    assert len(rows["input_ids"][0]) == 4


def test_deterministic_sharding_is_disjoint_and_complete() -> None:
    records = [TextRecord(str(i), f"text-{i}", "train") for i in range(10)]
    shards = [list(deterministic_shard(records, shard_index=i, num_shards=3)) for i in range(3)]
    ids = [record.record_id for shard in shards for record in shard]
    assert Counter(ids) == Counter(record.record_id for record in records)
    assert all(
        {a.record_id for a in shards[i]}.isdisjoint(b.record_id for b in shards[j])
        for i in range(3)
        for j in range(i + 1, 3)
    )


def test_d03_jsonl_adapter_preserves_order_and_explicit_split() -> None:
    lines = [
        json.dumps({"id": "d1", "text": "first", "source_id": "s"}),
        json.dumps({"id": "d2", "text": "second", "source_id": "s"}),
    ]
    records = list(records_from_jsonl_lines(lines, split="validation"))
    assert [record.record_id for record in records] == ["d1", "d2"]
    assert [record.split for record in records] == ["validation", "validation"]


def test_d03_jsonl_adapter_rejects_duplicate_ids() -> None:
    lines = [
        json.dumps({"id": "d1", "text": "first"}),
        json.dumps({"id": "d1", "text": "again"}),
    ]
    with pytest.raises(JsonlRecordError, match="duplicate"):
        list(records_from_jsonl_lines(lines, split="train"))


def test_packing_identity_matches_repository_config_and_d01_context() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "s0" / "packing_byte_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert canonical_packing_config_json() == canonical
    assert packing_config_hash() == PACKING_CONFIG_HASH
    assert PACKING_CONFIG_HASH == (
        "23a695b807f3e3f5c61d19c34968bcd88fafc6a45346dc08673d7a494219f285"
    )
    assert DEFAULT_SEQUENCE_LENGTH == 128


def test_mixture_sampler_is_repeatable_and_seeded() -> None:
    a = DeterministicMixtureSampler({"code": 1.0, "text": 3.0}, seed=42)
    b = DeterministicMixtureSampler({"text": 3.0, "code": 1.0}, seed=42)
    c = DeterministicMixtureSampler({"code": 1.0, "text": 3.0}, seed=43)

    seq_a = [a.source_for_step(i) for i in range(64)]
    seq_b = [b.source_for_step(i) for i in range(64)]
    seq_c = [c.source_for_step(i) for i in range(64)]

    assert seq_a == seq_b
    assert seq_a != seq_c
    assert set(seq_a) == {"code", "text"}


@pytest.mark.parametrize(
    "weights",
    [{}, {"a": 0.0}, {"a": -1.0}, {"a": float("inf")}, {"": 1.0}],
)
def test_mixture_sampler_rejects_invalid_weights(weights) -> None:
    with pytest.raises(ValueError):
        DeterministicMixtureSampler(weights)
