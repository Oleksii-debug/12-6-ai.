from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.packing.core import TextRecord
from twelve_six.packing.scale_contracts import MixturePlan, MixtureSource
from twelve_six.packing.sharded_storage import (
    LogicalShardFile,
    PhysicalShardRecordFactory,
    build_physical_shard_dataset,
    write_logical_jsonl_shards,
)
from twelve_six.packing.streaming import StreamingDataError, build_dataloader
from twelve_six.tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer


def _plan(shards: int = 16) -> MixturePlan:
    return MixturePlan(
        plan_id="physical-shard-test",
        tokenizer_config_sha256=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
        packing_config_sha256="9" * 64,
        sources=(MixtureSource("synthetic", "a" * 64, 1),),
        seed=77,
        num_shards=shards,
        shard_seed=88,
    )


def _records(count: int = 160) -> tuple[TextRecord, ...]:
    return tuple(
        TextRecord(
            f"physical-{index:05d}",
            (f"English Українська code {index} " * (2 + index % 11)).strip(),
            "train",
        )
        for index in range(count)
    )


def test_physical_factory_opens_only_rank_owned_files(tmp_path: Path) -> None:
    plan = _plan()
    files = write_logical_jsonl_shards(_records(), plan, tmp_path)
    rank0 = PhysicalShardRecordFactory(plan, files, split="train", rank=0, world_size=2)
    rank1 = PhysicalShardRecordFactory(plan, files, split="train", rank=1, world_size=2)
    opened0 = {file.logical_shard for file in rank0.opened_files_for_current_worker()}
    opened1 = {file.logical_shard for file in rank1.opened_files_for_current_worker()}
    present = {file.logical_shard for file in files}
    assert opened0.isdisjoint(opened1)
    assert opened0 | opened1 == present
    assert all(shard % 2 == 0 for shard in opened0)
    assert all(shard % 2 == 1 for shard in opened1)


def test_two_rank_physical_datasets_equal_single_rank_membership_and_loss(tmp_path: Path) -> None:
    plan = _plan()
    tokenizer = ByteTokenizer()
    files = write_logical_jsonl_shards(_records(), plan, tmp_path)

    single = build_physical_shard_dataset(
        plan,
        tokenizer,
        files,
        source_name="synthetic",
        split="train",
        sequence_length=64,
    )
    single_batches = list(build_dataloader(single, batch_size=7, num_workers=0))

    distributed = []
    for rank in range(2):
        dataset = build_physical_shard_dataset(
            plan,
            tokenizer,
            files,
            source_name="synthetic",
            split="train",
            rank=rank,
            world_size=2,
            sequence_length=64,
        )
        distributed.extend(build_dataloader(dataset, batch_size=7, num_workers=0))

    assert sum(batch.examples for batch in distributed) == sum(
        batch.examples for batch in single_batches
    )
    assert sum(batch.loss_tokens for batch in distributed) == sum(
        batch.loss_tokens for batch in single_batches
    )


def test_multiprocess_dataloader_over_physical_shards_preserves_exact_counts(tmp_path: Path) -> None:
    plan = _plan(shards=32)
    tokenizer = ByteTokenizer()
    files = write_logical_jsonl_shards(_records(240), plan, tmp_path)
    dataset = build_physical_shard_dataset(
        plan,
        tokenizer,
        files,
        source_name="synthetic",
        split="train",
        sequence_length=64,
    )
    direct = list(build_dataloader(dataset, batch_size=9, num_workers=0))
    workers = list(
        build_dataloader(dataset, batch_size=9, num_workers=2, prefetch_factor=2)
    )
    assert sum(batch.examples for batch in workers) == sum(batch.examples for batch in direct)
    assert sum(batch.loss_tokens for batch in workers) == sum(
        batch.loss_tokens for batch in direct
    )


def test_misplaced_record_fails_closed_instead_of_becoming_invisible(tmp_path: Path) -> None:
    plan = _plan()
    record = _records(1)[0]
    actual = plan.shard_for_record(record.record_id)
    wrong = (actual + 1) % plan.num_shards
    path = tmp_path / "wrong.jsonl"
    path.write_text(
        json.dumps({"id": record.record_id, "text": record.text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    factory = PhysicalShardRecordFactory(
        plan,
        (LogicalShardFile(wrong, str(path), "jsonl"),),
        split="train",
    )
    with pytest.raises(StreamingDataError, match="stored in shard"):
        list(factory())
