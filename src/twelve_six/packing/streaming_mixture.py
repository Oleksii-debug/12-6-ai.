"""Execute the incumbent MixturePlan/RestartCursor over restartable streaming batches."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch

from .scale_contracts import MixturePlan, RestartCursor
from .streaming import StreamCursor, StreamingDataError, TrainerBatchEnvelope

MIXTURE_RUNTIME_CURSOR_SCHEMA = "12-6.streaming-mixture-runtime-cursor.v1"


@dataclass(frozen=True, slots=True)
class MixtureRuntimeCursor:
    """One deterministic schedule cursor plus committed per-source stream positions."""

    schedule: RestartCursor
    source_streams: tuple[tuple[str, StreamCursor], ...]

    @classmethod
    def initial(
        cls,
        plan: MixturePlan,
        source_streams: Mapping[str, StreamCursor],
    ) -> MixtureRuntimeCursor:
        expected = tuple(source.name for source in plan.ordered_sources)
        if tuple(sorted(source_streams)) != expected:
            raise StreamingDataError("mixture runtime cursor source set does not match MixturePlan")
        cls._validate_streams(plan, source_streams)
        return cls(
            schedule=RestartCursor.initial(plan),
            source_streams=tuple((name, source_streams[name]) for name in expected),
        )

    @staticmethod
    def _validate_streams(plan: MixturePlan, source_streams: Mapping[str, StreamCursor]) -> None:
        for source in plan.ordered_sources:
            cursor = source_streams.get(source.name)
            if cursor is None:
                raise StreamingDataError(f"missing stream cursor for source {source.name!r}")
            cursor.require_compatible(
                plan,
                source_name=source.name,
                split=cursor.split,
            )

    def require_compatible(self, plan: MixturePlan) -> None:
        self.schedule.require_compatible(plan)
        streams = dict(self.source_streams)
        expected = tuple(source.name for source in plan.ordered_sources)
        if tuple(name for name, _ in self.source_streams) != expected:
            raise StreamingDataError("mixture runtime stream order/source set drifted")
        self._validate_streams(plan, streams)

    def stream_for(self, source_name: str) -> StreamCursor:
        streams = dict(self.source_streams)
        try:
            return streams[source_name]
        except KeyError as exc:
            raise StreamingDataError(f"unknown mixture source {source_name!r}") from exc

    def with_advance(
        self,
        plan: MixturePlan,
        *,
        source_name: str,
        source_cursor: StreamCursor,
        emitted_sequences: int,
        emitted_loss_tokens: int,
    ) -> MixtureRuntimeCursor:
        self.require_compatible(plan)
        source_cursor.require_compatible(
            plan,
            source_name=source_name,
            split=source_cursor.split,
        )
        streams = dict(self.source_streams)
        streams[source_name] = source_cursor
        schedule = self.schedule.advance(
            plan,
            source_name=source_name,
            emitted_sequences=emitted_sequences,
            emitted_loss_tokens=emitted_loss_tokens,
        )
        return MixtureRuntimeCursor(
            schedule=schedule,
            source_streams=tuple(
                (source.name, streams[source.name]) for source in plan.ordered_sources
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": MIXTURE_RUNTIME_CURSOR_SCHEMA,
            "schedule": self.schedule.to_dict(),
            "source_streams": {
                name: cursor.to_dict() for name, cursor in self.source_streams
            },
        }


@dataclass(frozen=True, slots=True)
class MixtureTrainerBatchEnvelope:
    """One scheduled source batch ready for the existing D02 Trainer."""

    batch: Mapping[str, torch.Tensor]
    source_name: str
    examples: int
    loss_tokens: int
    cursor_after: MixtureRuntimeCursor


def iter_mixture_batches(
    plan: MixturePlan,
    source_batches: Mapping[str, Iterator[TrainerBatchEnvelope]],
    *,
    cursor: MixtureRuntimeCursor,
    max_batches: int | None = None,
) -> Iterator[MixtureTrainerBatchEnvelope]:
    """Follow MixturePlan exactly; never silently reweight an exhausted source.

    The incumbent `RestartCursor` remains the schedule authority. DATA-13 only binds each
    scheduled batch to the exact per-source streaming cursor that generated it.
    """
    if max_batches is not None and max_batches < 0:
        raise StreamingDataError("max_batches must be non-negative or None")
    cursor.require_compatible(plan)
    expected_names = tuple(source.name for source in plan.ordered_sources)
    if tuple(sorted(source_batches)) != expected_names:
        raise StreamingDataError("source batch iterator set does not match MixturePlan")

    current = cursor
    emitted = 0
    while max_batches is None or emitted < max_batches:
        source_name, _ = current.schedule.next_source_and_offset(plan)
        try:
            envelope = next(source_batches[source_name])
        except StopIteration as exc:
            raise StreamingDataError(
                f"mixture source {source_name!r} exhausted; explicit epoch/oversampling "
                "policy is required instead of silent reweighting"
            ) from exc
        if envelope.cursor_after.source_name != source_name:
            raise StreamingDataError("scheduled source returned a batch from another source")
        next_cursor = current.with_advance(
            plan,
            source_name=source_name,
            source_cursor=envelope.cursor_after,
            emitted_sequences=envelope.examples,
            emitted_loss_tokens=envelope.loss_tokens,
        )
        yield MixtureTrainerBatchEnvelope(
            batch=envelope.batch,
            source_name=source_name,
            examples=envelope.examples,
            loss_tokens=envelope.loss_tokens,
            cursor_after=next_cursor,
        )
        current = next_cursor
        emitted += 1
