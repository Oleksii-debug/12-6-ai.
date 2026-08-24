from __future__ import annotations

from dataclasses import replace

import pytest

from twelve_six.packing.core import (
    PACKING_CONFIG_HASH,
    TextRecord,
    collate_rows,
    iter_packed_examples,
)
from twelve_six.packing.scale import (
    DeterministicShardPlan,
    IntegerMixturePlan,
    MixtureComponent,
    PackingRestartCursor,
    audit_packed_examples,
)
from twelve_six.tokenization.byte import BYTE_CONFIG_HASH, BYTE_VOCAB_HASH, ByteTokenizer

DATASET_A = "a" * 64
DATASET_B = "b" * 64


def _mixture() -> IntegerMixturePlan:
    return IntegerMixturePlan(
        (
            MixtureComponent("uk", 3, DATASET_A),
            MixtureComponent("en", 7, DATASET_B),
        ),
        seed=17,
    )


def _shards() -> DeterministicShardPlan:
    return DeterministicShardPlan(DATASET_A, "train", 8)


def _cursor() -> PackingRestartCursor:
    mixture = _mixture()
    shards = _shards()
    return PackingRestartCursor(
        mixture_plan_sha256=mixture.identity_sha256,
        shard_plan_sha256=shards.identity_sha256,
        dataset_manifest_sha256=DATASET_A,
        tokenizer_config_sha256=BYTE_CONFIG_HASH,
        tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
        packing_config_sha256=PACKING_CONFIG_HASH,
        split="train",
        global_sample_index=123,
        shard_epoch=2,
        shard_index=5,
        document_index=41,
        token_offset=17,
        rank=1,
        world_size=4,
    )


def _require_cursor(cursor: PackingRestartCursor, *, world_size: int = 4) -> None:
    mixture = _mixture()
    shards = _shards()
    cursor.require_compatible(
        mixture_plan_sha256=mixture.identity_sha256,
        shard_plan_sha256=shards.identity_sha256,
        dataset_manifest_sha256=DATASET_A,
        tokenizer_config_sha256=BYTE_CONFIG_HASH,
        tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
        packing_config_sha256=PACKING_CONFIG_HASH,
        split="train",
        rank=1,
        world_size=world_size,
    )


def test_integer_mixture_is_order_independent_and_restart_addressable() -> None:
    forward = _mixture()
    reverse = IntegerMixturePlan(tuple(reversed(forward.components)), seed=17)
    assert reverse.identity_sha256 == forward.identity_sha256
    expected = [forward.source_for_sample(index) for index in range(100)]
    assert expected == [reverse.source_for_sample(index) for index in range(100)]
    assert set(expected) == {"en", "uk"}


def test_content_addressed_sharding_is_independent_of_input_order() -> None:
    plan = _shards()
    ids = ("doc-3", "doc-1", "doc-2", "doc-4")
    first = {record_id: plan.shard_for_record(record_id) for record_id in ids}
    second = {record_id: plan.shard_for_record(record_id) for record_id in reversed(ids)}
    assert first == second
    assert all(0 <= shard < 8 for shard in first.values())


def test_restart_cursor_fails_closed_on_identity_or_topology_drift() -> None:
    cursor = _cursor()
    _require_cursor(cursor)
    assert _mixture().source_for_sample(cursor.global_sample_index) == _mixture().source_for_sample(123)

    with pytest.raises(ValueError, match="tokenizer_config_sha256"):
        cursor.require_compatible(
            mixture_plan_sha256=cursor.mixture_plan_sha256,
            shard_plan_sha256=cursor.shard_plan_sha256,
            dataset_manifest_sha256=DATASET_A,
            tokenizer_config_sha256="d" * 64,
            tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
            packing_config_sha256=PACKING_CONFIG_HASH,
            split="train",
            rank=1,
            world_size=4,
        )
    with pytest.raises(ValueError, match="tokenizer_vocab_sha256"):
        cursor.require_compatible(
            mixture_plan_sha256=cursor.mixture_plan_sha256,
            shard_plan_sha256=cursor.shard_plan_sha256,
            dataset_manifest_sha256=DATASET_A,
            tokenizer_config_sha256=BYTE_CONFIG_HASH,
            tokenizer_vocab_sha256="c" * 64,
            packing_config_sha256=PACKING_CONFIG_HASH,
            split="train",
            rank=1,
            world_size=4,
        )
    with pytest.raises(ValueError, match="shard_plan_sha256"):
        cursor.require_compatible(
            mixture_plan_sha256=cursor.mixture_plan_sha256,
            shard_plan_sha256="e" * 64,
            dataset_manifest_sha256=DATASET_A,
            tokenizer_config_sha256=BYTE_CONFIG_HASH,
            tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
            packing_config_sha256=PACKING_CONFIG_HASH,
            split="train",
            rank=1,
            world_size=4,
        )
    with pytest.raises(ValueError, match="world_size"):
        _require_cursor(replace(cursor, world_size=8), world_size=4)


def test_document_isolation_masked_accounting_and_no_double_shift() -> None:
    tokenizer = ByteTokenizer()
    records = (
        TextRecord("a", "ABCDE", "train"),
        TextRecord("b", "xyz", "train"),
    )
    examples = tuple(
        iter_packed_examples(records, tokenizer, expected_split="train", sequence_length=4)
    )
    accounting = audit_packed_examples(examples, vocab_size=tokenizer.vocab_size)
    assert accounting.loss_tokens == (len("ABCDE") - 1) + (len("xyz") - 1)
    assert all(example.record_ids in {("a",), ("b",)} for example in examples)

    labels = collate_rows(examples[:1], target_mode="labels")
    aligned = collate_rows(examples[:1], target_mode="target_ids")
    example = examples[0]
    assert labels["labels"][0] == example.labels
    assert "loss_mask" not in labels
    for index, keep in enumerate(example.loss_mask):
        if keep:
            assert aligned["target_ids"][0][index] == example.labels[index + 1]
        else:
            assert aligned["target_ids"][0][index] == -100


def test_accounting_rejects_loss_mask_under_count() -> None:
    tokenizer = ByteTokenizer()
    example = next(
        iter_packed_examples(
            (TextRecord("a", "ABCD", "train"),),
            tokenizer,
            expected_split="train",
            sequence_length=4,
        )
    )
    bad = replace(example, loss_mask=(1, 0, 0, 0))
    with pytest.raises(ValueError, match="loss-token accounting drifted"):
        audit_packed_examples((bad,), vocab_size=tokenizer.vocab_size)


def test_accounting_rejects_cross_document_provenance() -> None:
    tokenizer = ByteTokenizer()
    records = (
        TextRecord("a", "ab", "train"),
        TextRecord("b", "cd", "train"),
    )
    with pytest.raises(ValueError, match="explicit EOS"):
        tuple(
            iter_packed_examples(
                records,
                tokenizer,
                expected_split="train",
                sequence_length=4,
                cross_document=True,
                add_eos=True,
            )
        )
