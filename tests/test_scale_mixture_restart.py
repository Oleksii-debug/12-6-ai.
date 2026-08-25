from __future__ import annotations

import pytest
import torch

from twelve_six.packing.core import (
    SplitMixError,
    TextRecord,
    collate_rows,
    iter_packed_examples,
)
from twelve_six.packing.scale_contracts import (
    MixtureContractError,
    MixturePlan,
    MixtureSource,
    RestartCursor,
)
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training.loss import causal_lm_loss, causal_pair_loss

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _plan(
    *,
    sources: tuple[MixtureSource, ...] | None = None,
    tokenizer_vocab_sha256: str = SHA_B,
) -> MixturePlan:
    return MixturePlan(
        plan_id="s1-mixture-fixture-v1",
        tokenizer_config_sha256=SHA_A,
        tokenizer_vocab_sha256=tokenizer_vocab_sha256,
        packing_config_sha256=SHA_C,
        sources=sources
        or (
            MixtureSource("en", SHA_D, 3),
            MixtureSource("uk", SHA_E, 2),
        ),
        seed=17,
        num_shards=4,
        shard_seed=91,
    )


def test_integer_mixture_is_order_stable_and_identity_bound() -> None:
    plan = _plan()
    reversed_plan = _plan(sources=tuple(reversed(plan.sources)))
    assert plan.sha256 == reversed_plan.sha256
    assert [plan.source_for_sample(i) for i in range(200)] == [
        reversed_plan.source_for_sample(i) for i in range(200)
    ]

    changed_vocab = _plan(tokenizer_vocab_sha256="f" * 64)
    assert changed_vocab.sha256 != plan.sha256
    assert [changed_vocab.source_for_sample(i) for i in range(20)] != [
        plan.source_for_sample(i) for i in range(20)
    ]


def test_record_sharding_is_reorder_independent_and_total() -> None:
    plan = _plan()
    record_ids = [f"record-{index:03d}" for index in range(100)]
    forward = {record_id: plan.shard_for_record(record_id) for record_id in record_ids}
    reverse = {record_id: plan.shard_for_record(record_id) for record_id in reversed(record_ids)}
    assert forward == reverse
    assert all(0 <= shard < plan.num_shards for shard in forward.values())


def test_restart_cursor_reproduces_exact_schedule_suffix() -> None:
    plan = _plan()
    uninterrupted = [plan.source_for_sample(index) for index in range(50)]

    cursor = RestartCursor.initial(plan)
    consumed: list[str] = []
    for _ in range(19):
        source, offset = cursor.next_source_and_offset(plan)
        assert offset == consumed.count(source)
        consumed.append(source)
        cursor = cursor.advance(
            plan,
            source_name=source,
            emitted_sequences=1,
            emitted_loss_tokens=128,
        )

    resumed = consumed[:]
    while cursor.next_sample_index < 50:
        source, _ = cursor.next_source_and_offset(plan)
        resumed.append(source)
        cursor = cursor.advance(
            plan,
            source_name=source,
            emitted_sequences=1,
            emitted_loss_tokens=128,
        )

    assert resumed == uninterrupted
    assert cursor.emitted_sequences == 50
    assert cursor.emitted_loss_tokens == 6400
    assert sum(dict(cursor.source_offsets).values()) == 50


def test_restart_cursor_rejects_plan_drift_and_wrong_source() -> None:
    plan = _plan()
    cursor = RestartCursor.initial(plan)
    changed = _plan(tokenizer_vocab_sha256="f" * 64)
    with pytest.raises(MixtureContractError, match="different mixture plan"):
        cursor.require_compatible(changed)

    expected, _ = cursor.next_source_and_offset(plan)
    wrong = "uk" if expected == "en" else "en"
    with pytest.raises(MixtureContractError, match="deterministic schedule"):
        cursor.advance(plan, source_name=wrong, emitted_sequences=1, emitted_loss_tokens=5)


def test_s0_default_packing_prevents_cross_document_leakage_and_counts_loss_tokens() -> None:
    tokenizer = ByteTokenizer()
    records = [
        TextRecord(record_id="doc-a", text="abcd", split="train"),
        TextRecord(record_id="doc-b", text="xyz", split="train"),
    ]
    examples = list(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split="train",
            sequence_length=4,
        )
    )
    assert examples
    assert all(len(example.record_ids) == 1 for example in examples)
    assert sum(example.num_loss_tokens for example in examples) == (4 - 1) + (3 - 1)

    with pytest.raises(ValueError, match="explicit EOS"):
        list(
            iter_packed_examples(
                records,
                tokenizer,
                expected_split="train",
                sequence_length=4,
                cross_document=True,
            )
        )


def test_collate_contract_prevents_double_shift_and_matches_d02_losses() -> None:
    tokenizer = ByteTokenizer()
    example = next(
        iter_packed_examples(
            [TextRecord(record_id="doc", text="abcd", split="train")],
            tokenizer,
            expected_split="train",
            sequence_length=4,
        )
    )
    labels_rows = collate_rows([example], target_mode="labels")
    aligned_rows = collate_rows([example], target_mode="target_ids")

    assert labels_rows["labels"][0] == (97, 98, 99, 100)
    assert aligned_rows["target_ids"][0] == (98, 99, 100, -100)
    assert aligned_rows["loss_mask"][0] == (1, 1, 1, 0)

    generator = torch.Generator().manual_seed(7)
    logits = torch.randn((1, 4, 256), generator=generator)
    labels = torch.tensor(labels_rows["labels"])
    target_ids = torch.tensor(aligned_rows["target_ids"])
    loss_mask = torch.tensor(aligned_rows["loss_mask"])
    shifted_loss = causal_lm_loss(logits, labels)
    aligned_loss = causal_pair_loss(logits, target_ids, loss_mask=loss_mask)
    assert torch.allclose(shifted_loss, aligned_loss)


def test_split_mixing_still_fails_closed() -> None:
    tokenizer = ByteTokenizer()
    with pytest.raises(SplitMixError):
        list(
            iter_packed_examples(
                [TextRecord(record_id="validation-doc", text="abc", split="validation")],
                tokenizer,
                expected_split="train",
            )
        )
