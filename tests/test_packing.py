from __future__ import annotations

from collections import Counter

import pytest

from twelve_six.data import (
    DeterministicMixtureSampler,
    SplitMixError,
    TextRecord,
    batch_examples,
    deterministic_shard,
    iter_packed_examples,
)
from twelve_six.tokenization import ByteTokenizer


def _training_pairs(examples):
    pairs = []
    for example in examples:
        for input_id, target_id, keep in zip(
            example.input_ids, example.target_ids, example.loss_mask, strict=True
        ):
            if keep:
                pairs.append((input_id, target_id))
    return pairs


def test_packing_is_deterministic_and_preserves_all_adjacent_pairs() -> None:
    tokenizer = ByteTokenizer()
    records = [
        TextRecord("r1", "ab", "train"),
        TextRecord("r2", "Ж🙂", "train"),
    ]
    kwargs = dict(
        tokenizer=tokenizer,
        expected_split="train",
        sequence_length=4,
        add_bos=True,
        add_eos=True,
    )
    first = list(iter_packed_examples(records, **kwargs))
    second = list(iter_packed_examples(records, **kwargs))
    assert first == second

    stream = []
    for record in records:
        stream.extend(tokenizer.encode(record.text, add_bos=True, add_eos=True))
    expected_pairs = list(zip(stream, stream[1:]))
    assert _training_pairs(first) == expected_pairs


def test_final_partial_block_is_padded_and_masked_not_dropped() -> None:
    tokenizer = ByteTokenizer()
    records = [TextRecord("r1", "x", "train")]
    examples = list(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split="train",
            sequence_length=8,
        )
    )
    assert len(examples) == 1
    example = examples[0]
    assert sum(example.loss_mask) == 2
    assert example.loss_mask == (1, 1, 0, 0, 0, 0, 0, 0)
    assert example.record_ids == ("r1",)


def test_split_mixing_fails_closed() -> None:
    tokenizer = ByteTokenizer()
    records = [
        TextRecord("r1", "train", "train"),
        TextRecord("r2", "validation", "validation"),
    ]
    with pytest.raises(SplitMixError):
        list(
            iter_packed_examples(
                records,
                tokenizer,
                expected_split="train",
                sequence_length=8,
            )
        )


def test_batching_keeps_tail_by_default() -> None:
    tokenizer = ByteTokenizer()
    records = [TextRecord(str(i), "abc", "train") for i in range(5)]
    examples = list(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split="train",
            sequence_length=3,
        )
    )
    batches = list(batch_examples(examples, batch_size=4))
    assert sum(len(batch) for batch in batches) == len(examples)
    assert 1 <= len(batches[-1]) <= 4


def test_deterministic_sharding_is_disjoint_and_complete() -> None:
    records = [TextRecord(str(i), f"text-{i}", "train") for i in range(10)]
    shards = [list(deterministic_shard(records, shard_index=i, num_shards=3)) for i in range(3)]
    ids = [record.record_id for shard in shards for record in shard]
    assert Counter(ids) == Counter(record.record_id for record in records)
    assert all(
        set(a.record_id for a in shards[i]).isdisjoint(b.record_id for b in shards[j])
        for i in range(3)
        for j in range(i + 1, 3)
    )


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
