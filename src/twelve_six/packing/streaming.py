"""Bounded-memory streaming, logical sharding, restart, and Trainer batch adapters."""

from __future__ import annotations

import json
import queue
import re
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import torch

from twelve_six.tokenization import TokenizerProtocol

from .core import (
    DEFAULT_FILL_TOKEN_ID,
    DEFAULT_IGNORE_INDEX,
    PackedCausalExample,
    TextRecord,
    collate_rows,
    iter_packed_examples,
)
from .scale_contracts import MixturePlan

STREAM_CURSOR_SCHEMA = "12-6.streaming-pack-cursor.v1"
STREAMING_READER_VERSION = "bounded-stream-reader-v1"
LOGICAL_SHARD_ASSIGNMENT_VERSION = "logical-shard-consumer-mod-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StreamingDataError(ValueError):
    """Raised when a streaming/restart contract fails closed."""


@dataclass(frozen=True, slots=True)
class RuntimeTopology:
    """One distributed rank and one local worker within a fixed logical-shard plan."""

    rank: int = 0
    world_size: int = 1
    worker_id: int = 0
    num_workers: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("rank", self.rank),
            ("world_size", self.world_size),
            ("worker_id", self.worker_id),
            ("num_workers", self.num_workers),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.world_size <= 0 or self.num_workers <= 0:
            raise StreamingDataError("world_size and num_workers must be positive")
        if not 0 <= self.rank < self.world_size:
            raise StreamingDataError("rank must satisfy 0 <= rank < world_size")
        if not 0 <= self.worker_id < self.num_workers:
            raise StreamingDataError("worker_id must satisfy 0 <= worker_id < num_workers")

    @property
    def consumer_count(self) -> int:
        return self.world_size * self.num_workers

    @property
    def consumer_index(self) -> int:
        return self.rank * self.num_workers + self.worker_id

    def owns_logical_shard(self, logical_shard: int) -> bool:
        if logical_shard < 0:
            raise StreamingDataError("logical_shard must be non-negative")
        return logical_shard % self.consumer_count == self.consumer_index


@dataclass(frozen=True, slots=True)
class ShardPosition:
    logical_shard: int
    next_record_ordinal: int
    next_window_index: int = 0

    def __post_init__(self) -> None:
        if self.logical_shard < 0:
            raise StreamingDataError("logical_shard must be non-negative")
        if self.next_record_ordinal < 0 or self.next_window_index < 0:
            raise StreamingDataError("restart positions must be non-negative")


@dataclass(frozen=True, slots=True)
class StreamCursor:
    """Committed progress keyed by stable logical shards, not physical workers."""

    plan_sha256: str
    source_name: str
    split: str
    positions: tuple[ShardPosition, ...]
    emitted_examples: int = 0
    emitted_loss_tokens: int = 0

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.plan_sha256) is None:
            raise StreamingDataError("plan_sha256 must be a lowercase SHA-256 digest")
        if not self.source_name or not self.split:
            raise StreamingDataError("source_name and split must be non-empty")
        shard_ids = [position.logical_shard for position in self.positions]
        if shard_ids != sorted(shard_ids) or len(shard_ids) != len(set(shard_ids)):
            raise StreamingDataError("cursor positions must contain sorted unique logical shards")
        if self.emitted_examples < 0 or self.emitted_loss_tokens < 0:
            raise StreamingDataError("emitted counters must be non-negative")

    @classmethod
    def initial(
        cls,
        plan: MixturePlan,
        *,
        source_name: str,
        split: str,
        topology: RuntimeTopology | None = None,
    ) -> StreamCursor:
        topology = topology or RuntimeTopology()
        positions = tuple(
            ShardPosition(shard, 0, 0)
            for shard in range(plan.num_shards)
            if topology.owns_logical_shard(shard)
        )
        return cls(plan.sha256, source_name, split, positions)

    def require_compatible(self, plan: MixturePlan, *, source_name: str, split: str) -> None:
        if self.plan_sha256 != plan.sha256:
            raise StreamingDataError("stream cursor belongs to a different MixturePlan")
        if self.source_name != source_name or self.split != split:
            raise StreamingDataError("stream cursor source/split identity mismatch")
        if any(position.logical_shard >= plan.num_shards for position in self.positions):
            raise StreamingDataError("stream cursor contains a shard outside the plan")

    def position_for(self, logical_shard: int) -> ShardPosition:
        for position in self.positions:
            if position.logical_shard == logical_shard:
                return position
        return ShardPosition(logical_shard, 0, 0)

    def with_position(
        self,
        position: ShardPosition,
        *,
        emitted_examples: int,
        emitted_loss_tokens: int,
    ) -> StreamCursor:
        positions = {item.logical_shard: item for item in self.positions}
        positions[position.logical_shard] = position
        return StreamCursor(
            plan_sha256=self.plan_sha256,
            source_name=self.source_name,
            split=self.split,
            positions=tuple(positions[key] for key in sorted(positions)),
            emitted_examples=self.emitted_examples + emitted_examples,
            emitted_loss_tokens=self.emitted_loss_tokens + emitted_loss_tokens,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": STREAM_CURSOR_SCHEMA,
            "plan_sha256": self.plan_sha256,
            "source_name": self.source_name,
            "split": self.split,
            "positions": [
                {
                    "logical_shard": item.logical_shard,
                    "next_record_ordinal": item.next_record_ordinal,
                    "next_window_index": item.next_window_index,
                }
                for item in self.positions
            ],
            "emitted_examples": self.emitted_examples,
            "emitted_loss_tokens": self.emitted_loss_tokens,
        }


def merge_stream_cursors(cursors: Sequence[StreamCursor], plan: MixturePlan) -> StreamCursor:
    """Merge rank/worker cursors at a checkpoint so logical shards can be reassigned."""
    if not cursors:
        raise StreamingDataError("at least one cursor is required")
    first = cursors[0]
    first.require_compatible(plan, source_name=first.source_name, split=first.split)
    merged: dict[int, ShardPosition] = {}
    emitted_examples = 0
    emitted_loss_tokens = 0
    for cursor in cursors:
        cursor.require_compatible(plan, source_name=first.source_name, split=first.split)
        emitted_examples += cursor.emitted_examples
        emitted_loss_tokens += cursor.emitted_loss_tokens
        for position in cursor.positions:
            if position.logical_shard in merged:
                raise StreamingDataError("logical shard appears in more than one committed cursor")
            merged[position.logical_shard] = position
    if set(merged) != set(range(plan.num_shards)):
        raise StreamingDataError("world-size change requires committed coverage of every logical shard")
    return StreamCursor(
        plan_sha256=plan.sha256,
        source_name=first.source_name,
        split=first.split,
        positions=tuple(merged[index] for index in range(plan.num_shards)),
        emitted_examples=emitted_examples,
        emitted_loss_tokens=emitted_loss_tokens,
    )


def project_stream_cursor(
    cursor: StreamCursor,
    plan: MixturePlan,
    *,
    topology: RuntimeTopology,
) -> StreamCursor:
    """Project a merged cursor onto a new topology without resetting logical shards."""
    cursor.require_compatible(plan, source_name=cursor.source_name, split=cursor.split)
    available = {item.logical_shard: item for item in cursor.positions}
    expected = [shard for shard in range(plan.num_shards) if topology.owns_logical_shard(shard)]
    if any(shard not in available for shard in expected):
        raise StreamingDataError("cursor lacks progress for a newly assigned logical shard")
    return StreamCursor(
        plan_sha256=cursor.plan_sha256,
        source_name=cursor.source_name,
        split=cursor.split,
        positions=tuple(available[shard] for shard in expected),
    )


def iter_jsonl_stream(path: str | Path, *, split: str) -> Iterator[TextRecord]:
    """Stream normalized D03-style JSONL in O(1) reader state.

    Global duplicate-ID checks belong upstream in D03. Retaining every seen ID in the
    training reader would make reader memory grow with corpus cardinality.
    """
    if not split:
        raise StreamingDataError("split must be non-empty")
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise StreamingDataError(f"invalid JSON at line {line_number}") from exc
            if not isinstance(payload, dict):
                raise StreamingDataError(f"line {line_number} must be a JSON object")
            record_id = payload.get("id")
            text = payload.get("text")
            if not isinstance(record_id, str) or not record_id:
                raise StreamingDataError(f"line {line_number} has invalid id")
            if not isinstance(text, str):
                raise StreamingDataError(f"line {line_number} has invalid text")
            yield TextRecord(record_id=record_id, text=text, split=split)


def iter_parquet_stream(
    path: str | Path,
    *,
    split: str,
    batch_rows: int = 1024,
    id_column: str = "id",
    text_column: str = "text",
) -> Iterator[TextRecord]:
    """Stream Parquet record batches through maintained PyArrow when available."""
    if batch_rows <= 0:
        raise StreamingDataError("batch_rows must be positive")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional runtime is dependency-owned
        raise RuntimeError(
            "Parquet streaming requires the optional maintained 'pyarrow' runtime"
        ) from exc
    parquet = pq.ParquetFile(str(path))
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=[id_column, text_column]):
        ids = batch.column(0).to_pylist()
        texts = batch.column(1).to_pylist()
        for record_id, text in zip(ids, texts, strict=True):
            if not isinstance(record_id, str) or not record_id or not isinstance(text, str):
                raise StreamingDataError("Parquet id/text columns must contain non-empty str values")
            yield TextRecord(record_id=record_id, text=text, split=split)


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class _Raised:
    error: BaseException


def prefetch_bounded(iterable: Iterable[T], *, max_items: int) -> Iterator[T]:
    """Prefetch in one daemon thread while bounding queued objects exactly."""
    if max_items <= 0:
        raise StreamingDataError("max_items must be positive")
    items: queue.Queue[object] = queue.Queue(maxsize=max_items)
    sentinel = object()

    def produce() -> None:
        try:
            for item in iterable:
                items.put(item)
        except BaseException as exc:
            items.put(_Raised(exc))
        finally:
            items.put(sentinel)

    thread = threading.Thread(target=produce, name="twelve-six-data-prefetch", daemon=True)
    thread.start()
    while True:
        item = items.get()
        if item is sentinel:
            break
        if isinstance(item, _Raised):
            raise item.error
        yield item  # type: ignore[misc]


@dataclass(frozen=True, slots=True)
class PackedStreamItem:
    example: PackedCausalExample
    logical_shard: int
    record_ordinal: int
    window_index: int
    cursor_after: StreamCursor


@dataclass(frozen=True, slots=True)
class TrainerBatchEnvelope:
    batch: Mapping[str, torch.Tensor]
    cursor_after: StreamCursor
    examples: int
    loss_tokens: int


def _tensorize_rows(rows: Mapping[str, Sequence[Sequence[int]]]) -> dict[str, torch.Tensor]:
    return {name: torch.tensor(values, dtype=torch.long) for name, values in rows.items()}


def _owned_shards(plan: MixturePlan, topology: RuntimeTopology) -> set[int]:
    return {shard for shard in range(plan.num_shards) if topology.owns_logical_shard(shard)}


def iter_packed_stream(
    records: Iterable[TextRecord],
    tokenizer: TokenizerProtocol,
    plan: MixturePlan,
    *,
    source_name: str,
    split: str,
    topology: RuntimeTopology | None = None,
    cursor: StreamCursor | None = None,
    sequence_length: int = 128,
    fill_token_id: int = DEFAULT_FILL_TOKEN_ID,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    add_bos: bool = False,
    add_eos: bool = False,
    max_document_tokens: int | None = None,
) -> Iterator[PackedStreamItem]:
    """Pack owned logical shards with exact per-window restart and bounded corpus memory."""
    topology = topology or RuntimeTopology()
    cursor = cursor or StreamCursor.initial(
        plan, source_name=source_name, split=split, topology=topology
    )
    cursor.require_compatible(plan, source_name=source_name, split=split)
    owned = _owned_shards(plan, topology)
    if {position.logical_shard for position in cursor.positions} != owned:
        raise StreamingDataError("cursor shard ownership does not match runtime topology")

    shard_ordinals = {shard: 0 for shard in owned}
    current_cursor = cursor
    for record in records:
        if record.split != split:
            raise StreamingDataError("record split does not match stream split")
        logical_shard = plan.shard_for_record(record.record_id)
        if logical_shard not in owned:
            continue
        ordinal = shard_ordinals[logical_shard]
        shard_ordinals[logical_shard] += 1
        position = current_cursor.position_for(logical_shard)
        if ordinal < position.next_record_ordinal:
            continue
        if ordinal > position.next_record_ordinal:
            raise StreamingDataError("source order/cursor drift: expected record ordinal was not observed")

        if max_document_tokens is not None:
            if max_document_tokens < 2:
                raise StreamingDataError("max_document_tokens must be at least two")
            token_count = len(tokenizer.encode(record.text, add_bos=add_bos, add_eos=add_eos))
            if token_count > max_document_tokens:
                raise StreamingDataError(
                    f"record {record.record_id!r} has {token_count} tokens; "
                    f"configured maximum is {max_document_tokens}"
                )

        examples = iter(
            iter_packed_examples(
                (record,),
                tokenizer,
                expected_split=split,
                sequence_length=sequence_length,
                fill_token_id=fill_token_id,
                ignore_index=ignore_index,
                add_bos=add_bos,
                add_eos=add_eos,
                cross_document=False,
            )
        )
        window_index = 0
        current_example = next(examples, None)
        while current_example is not None:
            following_example = next(examples, None)
            if window_index < position.next_window_index:
                window_index += 1
                current_example = following_example
                continue
            if window_index > position.next_window_index:
                raise StreamingDataError("window restart cursor drifted within a document")
            next_position = (
                ShardPosition(logical_shard, ordinal + 1, 0)
                if following_example is None
                else ShardPosition(logical_shard, ordinal, window_index + 1)
            )
            next_cursor = current_cursor.with_position(
                next_position,
                emitted_examples=1,
                emitted_loss_tokens=current_example.num_loss_tokens,
            )
            yield PackedStreamItem(
                example=current_example,
                logical_shard=logical_shard,
                record_ordinal=ordinal,
                window_index=window_index,
                cursor_after=next_cursor,
            )
            current_cursor = next_cursor
            position = next_position
            window_index += 1
            current_example = following_example

        if window_index == 0:
            final_position = ShardPosition(logical_shard, ordinal + 1, 0)
            current_cursor = StreamCursor(
                plan_sha256=current_cursor.plan_sha256,
                source_name=current_cursor.source_name,
                split=current_cursor.split,
                positions=tuple(
                    final_position if item.logical_shard == logical_shard else item
                    for item in current_cursor.positions
                ),
                emitted_examples=current_cursor.emitted_examples,
                emitted_loss_tokens=current_cursor.emitted_loss_tokens,
            )


def iter_trainer_batches(
    items: Iterable[PackedStreamItem],
    *,
    batch_size: int,
    target_mode: str = "labels",
    drop_last: bool = False,
) -> Iterator[TrainerBatchEnvelope]:
    """Collate streaming examples into Tensor batches accepted by D02 Trainer."""
    if batch_size <= 0:
        raise StreamingDataError("batch_size must be positive")
    pending: list[PackedStreamItem] = []
    for item in items:
        pending.append(item)
        if len(pending) == batch_size:
            examples = [entry.example for entry in pending]
            yield TrainerBatchEnvelope(
                batch=_tensorize_rows(collate_rows(examples, target_mode=target_mode)),
                cursor_after=pending[-1].cursor_after,
                examples=len(pending),
                loss_tokens=sum(example.num_loss_tokens for example in examples),
            )
            pending = []
    if pending and not drop_last:
        examples = [entry.example for entry in pending]
        yield TrainerBatchEnvelope(
            batch=_tensorize_rows(collate_rows(examples, target_mode=target_mode)),
            cursor_after=pending[-1].cursor_after,
            examples=len(pending),
            loss_tokens=sum(example.num_loss_tokens for example in examples),
        )


@dataclass(frozen=True, slots=True)
class SegmentedPackedExample:
    """Future EOS-aware cross-document layout requiring block-causal attention support."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[int, ...]
    segment_ids: tuple[int, ...]
    record_ids: tuple[str, ...]


def iter_eos_segmented_examples(
    records: Iterable[TextRecord],
    tokenizer: TokenizerProtocol,
    *,
    split: str,
    sequence_length: int,
    fill_token_id: int = DEFAULT_FILL_TOKEN_ID,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> Iterator[SegmentedPackedExample]:
    """Pack EOS-terminated docs while retaining boundaries needed for block attention."""
    if tokenizer.eos_id is None:
        raise StreamingDataError("EOS-segment packing requires a semantic tokenizer EOS id")
    if sequence_length < 2:
        raise StreamingDataError("sequence_length must be at least two")
    tokens: list[int] = []
    segments: list[int] = []
    sources: list[str] = []
    next_segment = 0

    def emit() -> SegmentedPackedExample:
        actual = len(tokens)
        pad = sequence_length - actual
        loss_mask = [0] * sequence_length
        for index in range(max(actual - 1, 0)):
            if segments[index] == segments[index + 1]:
                loss_mask[index] = 1
        return SegmentedPackedExample(
            input_ids=tuple(tokens + [fill_token_id] * pad),
            labels=tuple(tokens + [ignore_index] * pad),
            loss_mask=tuple(loss_mask),
            segment_ids=tuple(segments + [-1] * pad),
            record_ids=tuple(dict.fromkeys(sources)),
        )

    for record in records:
        if record.split != split:
            raise StreamingDataError("record split does not match segmented stream split")
        encoded = tokenizer.encode(record.text, add_eos=True)
        if not encoded or encoded[-1] != tokenizer.eos_id:
            raise StreamingDataError("tokenizer add_eos=True did not terminate with eos_id")
        offset = 0
        while offset < len(encoded):
            room = sequence_length - len(tokens)
            take = min(room, len(encoded) - offset)
            tokens.extend(encoded[offset : offset + take])
            segments.extend([next_segment] * take)
            sources.extend([record.record_id] * take)
            offset += take
            if len(tokens) == sequence_length:
                yield emit()
                tokens[:] = tokens[-1:]
                segments[:] = segments[-1:]
                sources[:] = sources[-1:]
        next_segment += 1
    if len(tokens) >= 2:
        yield emit()


def segmented_to_current_trainer_batch(
    _: Sequence[SegmentedPackedExample],
) -> Mapping[str, torch.Tensor]:
    """Prevent current D01/D02 from silently discarding cross-document attention boundaries."""
    raise StreamingDataError(
        "current Trainer/model does not consume block-causal segment_ids; "
        "cross-document EOS packing is not training-eligible yet"
    )


class CursorAwareIterableDataset(torch.utils.data.IterableDataset):
    """Torch IterableDataset deriving disjoint logical shards per rank and worker."""

    def __init__(
        self,
        record_factory: Callable[[], Iterable[TextRecord]],
        tokenizer: TokenizerProtocol,
        plan: MixturePlan,
        *,
        source_name: str,
        split: str,
        rank: int = 0,
        world_size: int = 1,
        merged_cursor: StreamCursor | None = None,
        sequence_length: int = 128,
        max_document_tokens: int | None = None,
    ) -> None:
        super().__init__()
        RuntimeTopology(rank=rank, world_size=world_size)
        self.record_factory = record_factory
        self.tokenizer = tokenizer
        self.plan = plan
        self.source_name = source_name
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.merged_cursor = merged_cursor
        self.sequence_length = sequence_length
        self.max_document_tokens = max_document_tokens

    def __iter__(self) -> Iterator[PackedStreamItem]:
        worker = torch.utils.data.get_worker_info()
        topology = RuntimeTopology(
            rank=self.rank,
            world_size=self.world_size,
            worker_id=0 if worker is None else worker.id,
            num_workers=1 if worker is None else worker.num_workers,
        )
        cursor = (
            StreamCursor.initial(
                self.plan,
                source_name=self.source_name,
                split=self.split,
                topology=topology,
            )
            if self.merged_cursor is None
            else project_stream_cursor(self.merged_cursor, self.plan, topology=topology)
        )
        yield from iter_packed_stream(
            self.record_factory(),
            self.tokenizer,
            self.plan,
            source_name=self.source_name,
            split=self.split,
            topology=topology,
            cursor=cursor,
            sequence_length=self.sequence_length,
            max_document_tokens=self.max_document_tokens,
        )


def collate_stream_items(
    items: Sequence[PackedStreamItem],
    *,
    target_mode: str = "labels",
) -> TrainerBatchEnvelope:
    """DataLoader collate function retaining only progress actually returned to the caller."""
    if not items:
        raise StreamingDataError("cannot collate an empty streaming batch")
    cursor = items[0].cursor_after
    shard_domain = tuple(position.logical_shard for position in cursor.positions)
    for item in items[1:]:
        candidate = item.cursor_after
        if (
            candidate.plan_sha256 != cursor.plan_sha256
            or candidate.source_name != cursor.source_name
            or candidate.split != cursor.split
            or tuple(position.logical_shard for position in candidate.positions) != shard_domain
        ):
            raise StreamingDataError("one DataLoader batch mixed independent worker cursor domains")
        cursor = candidate
    examples = [item.example for item in items]
    return TrainerBatchEnvelope(
        batch=_tensorize_rows(collate_rows(examples, target_mode=target_mode)),
        cursor_after=cursor,
        examples=len(items),
        loss_tokens=sum(example.num_loss_tokens for example in examples),
    )


def build_dataloader(
    dataset: CursorAwareIterableDataset,
    *,
    batch_size: int,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    persistent_workers: bool = False,
    target_mode: str = "labels",
) -> torch.utils.data.DataLoader:
    """Build a bounded torch DataLoader without hiding worker-prefetch semantics."""
    if batch_size <= 0 or num_workers < 0 or prefetch_factor <= 0:
        raise StreamingDataError("invalid DataLoader batch/worker/prefetch configuration")
    if persistent_workers and num_workers == 0:
        raise StreamingDataError("persistent_workers requires num_workers > 0")
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": partial(collate_stream_items, target_mode=target_mode),
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return torch.utils.data.DataLoader(**kwargs)
