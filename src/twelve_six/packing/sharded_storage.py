"""Physical logical-shard storage adapters for scalable rank/worker consumption."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from twelve_six.tokenization import TokenizerProtocol

from .core import TextRecord
from .scale_contracts import MixturePlan
from .streaming import (
    CursorAwareIterableDataset,
    RuntimeTopology,
    StreamCursor,
    StreamingDataError,
    iter_jsonl_stream,
    iter_parquet_stream,
)

ShardFormat = Literal["jsonl", "parquet"]


@dataclass(frozen=True, slots=True)
class LogicalShardFile:
    """One physical file whose records all belong to one stable logical shard."""

    logical_shard: int
    path: str
    format: ShardFormat

    def __post_init__(self) -> None:
        if self.logical_shard < 0:
            raise StreamingDataError("logical_shard must be non-negative")
        if not self.path:
            raise StreamingDataError("physical shard path must be non-empty")
        if self.format not in ("jsonl", "parquet"):
            raise StreamingDataError("physical shard format must be 'jsonl' or 'parquet'")


class PhysicalShardRecordFactory:
    """Open only the logical shard files assigned to the current rank/worker.

    The factory deliberately validates record->logical-shard identity while reading. A
    physically misplaced record must fail closed: if it were merely filtered, that record
    could disappear because the worker owning its true logical shard never opens this file.
    """

    def __init__(
        self,
        plan: MixturePlan,
        files: tuple[LogicalShardFile, ...],
        *,
        split: str,
        rank: int = 0,
        world_size: int = 1,
        parquet_batch_rows: int = 1024,
    ) -> None:
        RuntimeTopology(rank=rank, world_size=world_size)
        if not split:
            raise StreamingDataError("split must be non-empty")
        if parquet_batch_rows <= 0:
            raise StreamingDataError("parquet_batch_rows must be positive")
        if not files:
            raise StreamingDataError("physical shard file set must not be empty")
        for file in files:
            if file.logical_shard >= plan.num_shards:
                raise StreamingDataError("physical file logical shard is outside MixturePlan")
        paths = [file.path for file in files]
        if len(paths) != len(set(paths)):
            raise StreamingDataError("physical shard paths must be unique")
        self.plan = plan
        self.files = tuple(sorted(files, key=lambda item: (item.logical_shard, item.path)))
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.parquet_batch_rows = parquet_batch_rows

    def _topology(self) -> RuntimeTopology:
        worker = torch.utils.data.get_worker_info()
        return RuntimeTopology(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=0 if worker is None else worker.id,
            num_workers=1 if worker is None else worker.num_workers,
        )

    def opened_files_for_current_worker(self) -> tuple[LogicalShardFile, ...]:
        topology = self._topology()
        return tuple(file for file in self.files if topology.owns_logical_shard(file.logical_shard))

    def __call__(self):
        for file in self.opened_files_for_current_worker():
            if file.format == "jsonl":
                records = iter_jsonl_stream(file.path, split=self.split)
            else:
                records = iter_parquet_stream(
                    file.path,
                    split=self.split,
                    batch_rows=self.parquet_batch_rows,
                )
            for record in records:
                actual_shard = self.plan.shard_for_record(record.record_id)
                if actual_shard != file.logical_shard:
                    raise StreamingDataError(
                        f"record {record.record_id!r} hashes to logical shard {actual_shard} "
                        f"but is stored in shard {file.logical_shard}"
                    )
                yield record


def build_physical_shard_dataset(
    plan: MixturePlan,
    tokenizer: TokenizerProtocol,
    files: tuple[LogicalShardFile, ...],
    *,
    source_name: str,
    split: str,
    rank: int = 0,
    world_size: int = 1,
    merged_cursor: StreamCursor | None = None,
    sequence_length: int = 128,
    max_document_tokens: int | None = None,
    parquet_batch_rows: int = 1024,
) -> CursorAwareIterableDataset:
    """Build the existing cursor-aware dataset over physically partitioned shard files."""
    factory = PhysicalShardRecordFactory(
        plan,
        files,
        split=split,
        rank=rank,
        world_size=world_size,
        parquet_batch_rows=parquet_batch_rows,
    )
    return CursorAwareIterableDataset(
        factory,
        tokenizer,
        plan,
        source_name=source_name,
        split=split,
        rank=rank,
        world_size=world_size,
        merged_cursor=merged_cursor,
        sequence_length=sequence_length,
        max_document_tokens=max_document_tokens,
    )


def write_logical_jsonl_shards(
    records: tuple[TextRecord, ...],
    plan: MixturePlan,
    output_dir: str | Path,
    *,
    prefix: str = "part",
) -> tuple[LogicalShardFile, ...]:
    """LOCAL_FREE fixture helper; production corpus packaging remains D03-owned."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    handles: dict[int, object] = {}
    paths: dict[int, Path] = {}
    try:
        import json

        for record in records:
            logical_shard = plan.shard_for_record(record.record_id)
            if logical_shard not in handles:
                path = directory / f"{prefix}-{logical_shard:05d}.jsonl"
                paths[logical_shard] = path
                handles[logical_shard] = path.open("w", encoding="utf-8", newline="\n")
            handle = handles[logical_shard]
            handle.write(
                json.dumps(
                    {"id": record.record_id, "text": record.text},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    finally:
        for handle in handles.values():
            handle.close()
    return tuple(
        LogicalShardFile(shard, str(paths[shard]), "jsonl") for shard in sorted(paths)
    )
