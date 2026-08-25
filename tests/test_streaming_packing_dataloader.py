from __future__ import annotations

import hashlib
from collections.abc import Iterable

import pytest
import torch
from torch import nn

from twelve_six.packing.core import TextRecord
from twelve_six.packing.scale_contracts import MixturePlan, MixtureSource
from twelve_six.packing.streaming import (
    CursorAwareIterableDataset,
    RuntimeTopology,
    StreamCursor,
    StreamingDataError,
    build_dataloader,
    iter_eos_segmented_examples,
    iter_packed_stream,
    iter_trainer_batches,
    merge_stream_cursors,
    project_stream_cursor,
    segmented_to_current_trainer_batch,
)
from twelve_six.tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
    TokenizerIdentity,
)
from twelve_six.training import Trainer, TrainerConfig


def _plan(*, shards: int = 8) -> MixturePlan:
    return MixturePlan(
        plan_id="streaming-test",
        tokenizer_config_sha256=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
        packing_config_sha256="3" * 64,
        sources=(MixtureSource("synthetic", "4" * 64, 1),),
        seed=17,
        num_shards=shards,
        shard_seed=29,
    )


def _records(count: int = 96) -> tuple[TextRecord, ...]:
    values = []
    for index in range(count):
        repeat = 1 + (index % 17)
        text = (f"doc-{index:04d} українська English code() " * repeat).strip()
        values.append(TextRecord(record_id=f"r-{index:04d}", text=text, split="train"))
    return tuple(values)


def _record_factory() -> Iterable[TextRecord]:
    return iter(_records(80))


def _item_key(item) -> tuple[object, ...]:
    return (
        item.logical_shard,
        item.record_ordinal,
        item.window_index,
        item.example.input_ids,
        item.example.loss_mask,
        item.example.record_ids,
    )


def test_streaming_is_exactly_restartable_and_preserves_document_loss_tokens() -> None:
    tokenizer = ByteTokenizer()
    plan = _plan()
    records = _records()
    items = list(
        iter_packed_stream(
            records,
            tokenizer,
            plan,
            source_name="synthetic",
            split="train",
            sequence_length=64,
        )
    )
    expected_loss_tokens = sum(max(len(record.text.encode("utf-8")) - 1, 0) for record in records)
    assert sum(item.example.num_loss_tokens for item in items) == expected_loss_tokens
    assert all(len(item.example.record_ids) == 1 for item in items)

    cut = len(items) // 3
    resumed = list(
        iter_packed_stream(
            records,
            tokenizer,
            plan,
            source_name="synthetic",
            split="train",
            sequence_length=64,
            cursor=items[cut - 1].cursor_after,
        )
    )
    assert [_item_key(item) for item in resumed] == [
        _item_key(item) for item in items[cut:]
    ]


def test_logical_shards_cover_ranks_workers_without_duplicates() -> None:
    tokenizer = ByteTokenizer()
    plan = _plan(shards=16)
    records = _records(120)
    seen: list[tuple[object, ...]] = []
    loss_tokens = 0
    cursors: list[StreamCursor] = []

    for rank in range(2):
        for worker_id in range(2):
            topology = RuntimeTopology(
                rank=rank,
                world_size=2,
                worker_id=worker_id,
                num_workers=2,
            )
            part = list(
                iter_packed_stream(
                    records,
                    tokenizer,
                    plan,
                    source_name="synthetic",
                    split="train",
                    topology=topology,
                    sequence_length=48,
                )
            )
            seen.extend(_item_key(item) for item in part)
            loss_tokens += sum(item.example.num_loss_tokens for item in part)
            cursors.append(
                part[-1].cursor_after
                if part
                else StreamCursor.initial(
                    plan,
                    source_name="synthetic",
                    split="train",
                    topology=topology,
                )
            )

    single = list(
        iter_packed_stream(
            records,
            tokenizer,
            plan,
            source_name="synthetic",
            split="train",
            sequence_length=48,
        )
    )
    assert sorted(seen, key=repr) == sorted((_item_key(item) for item in single), key=repr)
    assert loss_tokens == sum(item.example.num_loss_tokens for item in single)

    merged = merge_stream_cursors(cursors, plan)
    for rank in range(3):
        topology = RuntimeTopology(rank=rank, world_size=3)
        projected = project_stream_cursor(merged, plan, topology=topology)
        assert list(
            iter_packed_stream(
                records,
                tokenizer,
                plan,
                source_name="synthetic",
                split="train",
                topology=topology,
                cursor=projected,
                sequence_length=48,
            )
        ) == []


def test_world_size_change_requires_complete_logical_shard_checkpoint() -> None:
    plan = _plan(shards=8)
    rank_zero = StreamCursor.initial(
        plan,
        source_name="synthetic",
        split="train",
        topology=RuntimeTopology(rank=0, world_size=2),
    )
    with pytest.raises(StreamingDataError, match="coverage of every logical shard"):
        merge_stream_cursors((rank_zero,), plan)


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(256, 8)
        self.output = nn.Linear(8, 256, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.output(self.embedding(input_ids))


def test_tensor_batches_are_consumed_by_actual_trainer_with_exact_token_accounting() -> None:
    tokenizer = ByteTokenizer()
    plan = _plan()
    items = iter_packed_stream(
        _records(10),
        tokenizer,
        plan,
        source_name="synthetic",
        split="train",
        sequence_length=32,
    )
    envelope = next(iter_trainer_batches(items, batch_size=4, target_mode="labels"))
    trainer = Trainer(_TinyLM(), TrainerConfig(max_steps=2, learning_rate=1e-3))
    metrics = trainer.train_microbatch(envelope.batch)
    assert metrics.tokens == envelope.loss_tokens
    assert trainer.tokens_seen == envelope.loss_tokens


def test_torch_dataloader_workers_preserve_membership_and_loss_accounting() -> None:
    tokenizer = ByteTokenizer()
    plan = _plan(shards=16)
    dataset = CursorAwareIterableDataset(
        _record_factory,
        tokenizer,
        plan,
        source_name="synthetic",
        split="train",
        sequence_length=48,
    )
    direct = list(build_dataloader(dataset, batch_size=5, num_workers=0))
    workers = list(
        build_dataloader(
            dataset,
            batch_size=5,
            num_workers=2,
            prefetch_factor=2,
        )
    )
    assert sum(batch.loss_tokens for batch in workers) == sum(
        batch.loss_tokens for batch in direct
    )
    assert sum(batch.examples for batch in workers) == sum(batch.examples for batch in direct)


class _EosTokenizer:
    pad_id = None
    bos_id = None
    eos_id = 256
    vocab_size = 257
    version = "controlled-eos-v1"

    @property
    def identity(self) -> TokenizerIdentity:
        return TokenizerIdentity(
            version=self.version,
            config_sha256=hashlib.sha256(b"eos-config").hexdigest(),
            vocab_sha256=hashlib.sha256(b"eos-vocab").hexdigest(),
            vocab_size=self.vocab_size,
            normalization="none",
            encoding="utf-8",
            special_tokens={"eos": self.eos_id},
        )

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if add_bos:
            raise ValueError("no bos")
        values = list(text.encode("utf-8"))
        if add_eos:
            values.append(self.eos_id)
        return values

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
        errors: str = "strict",
    ) -> str:
        values = [value for value in token_ids if not skip_special_tokens or value != self.eos_id]
        return bytes(values).decode("utf-8", errors=errors)


def test_eos_cross_document_layout_keeps_segments_and_current_trainer_fails_closed() -> None:
    records = (
        TextRecord("a", "abcdef", "train"),
        TextRecord("b", "ghijkl", "train"),
        TextRecord("c", "mnopqr", "train"),
    )
    examples = list(
        iter_eos_segmented_examples(
            records,
            _EosTokenizer(),
            split="train",
            sequence_length=10,
        )
    )
    assert any(len(example.record_ids) > 1 for example in examples)
    for example in examples:
        for index, keep in enumerate(example.loss_mask[:-1]):
            if example.segment_ids[index] != example.segment_ids[index + 1]:
                assert keep == 0
    with pytest.raises(StreamingDataError, match="block-causal segment_ids"):
        segmented_to_current_trainer_batch(examples)
