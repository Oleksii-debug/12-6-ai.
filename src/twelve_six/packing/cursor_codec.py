"""Strict data-only codecs for durable D05-compatible streaming cursor payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .scale_contracts import RESTART_CURSOR_SCHEMA, MixturePlan, RestartCursor
from .streaming import STREAM_CURSOR_SCHEMA, ShardPosition, StreamCursor, StreamingDataError
from .streaming_mixture import MIXTURE_RUNTIME_CURSOR_SCHEMA, MixtureRuntimeCursor


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StreamingDataError(f"{field} must be a mapping")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StreamingDataError(f"{field} must be an integer")
    return value


def stream_cursor_from_dict(
    payload: Mapping[str, Any],
    plan: MixturePlan,
    *,
    source_name: str | None = None,
    split: str | None = None,
) -> StreamCursor:
    """Decode one logical-shard cursor and bind it to the current immutable plan."""
    data = _mapping(payload, field="stream cursor")
    if data.get("schema") != STREAM_CURSOR_SCHEMA:
        raise StreamingDataError("unsupported stream cursor schema")
    raw_positions = data.get("positions")
    if not isinstance(raw_positions, list):
        raise StreamingDataError("stream cursor positions must be a list")
    positions: list[ShardPosition] = []
    for index, raw in enumerate(raw_positions):
        item = _mapping(raw, field=f"positions[{index}]")
        positions.append(
            ShardPosition(
                logical_shard=_integer(
                    item.get("logical_shard"), field=f"positions[{index}].logical_shard"
                ),
                next_record_ordinal=_integer(
                    item.get("next_record_ordinal"),
                    field=f"positions[{index}].next_record_ordinal",
                ),
                next_window_index=_integer(
                    item.get("next_window_index", 0),
                    field=f"positions[{index}].next_window_index",
                ),
            )
        )
    actual_source = data.get("source_name")
    actual_split = data.get("split")
    if not isinstance(actual_source, str) or not isinstance(actual_split, str):
        raise StreamingDataError("stream cursor source_name/split must be strings")
    cursor = StreamCursor(
        plan_sha256=data.get("plan_sha256"),
        source_name=actual_source,
        split=actual_split,
        positions=tuple(positions),
        emitted_examples=_integer(data.get("emitted_examples", 0), field="emitted_examples"),
        emitted_loss_tokens=_integer(
            data.get("emitted_loss_tokens", 0), field="emitted_loss_tokens"
        ),
    )
    cursor.require_compatible(
        plan,
        source_name=actual_source if source_name is None else source_name,
        split=actual_split if split is None else split,
    )
    return cursor


def restart_cursor_from_dict(
    payload: Mapping[str, Any],
    plan: MixturePlan,
) -> RestartCursor:
    """Decode the incumbent PR #73 schedule cursor without modifying its ownership."""
    data = _mapping(payload, field="restart cursor")
    if data.get("schema") != RESTART_CURSOR_SCHEMA:
        raise StreamingDataError("unsupported mixture restart cursor schema")
    raw_offsets = _mapping(data.get("source_offsets"), field="source_offsets")
    expected_names = tuple(source.name for source in plan.ordered_sources)
    if tuple(sorted(raw_offsets)) != expected_names:
        raise StreamingDataError("restart cursor source set does not match MixturePlan")
    cursor = RestartCursor(
        plan_sha256=data.get("plan_sha256"),
        next_sample_index=_integer(data.get("next_sample_index"), field="next_sample_index"),
        source_offsets=tuple(
            (name, _integer(raw_offsets[name], field=f"source_offsets.{name}"))
            for name in expected_names
        ),
        emitted_sequences=_integer(data.get("emitted_sequences"), field="emitted_sequences"),
        emitted_loss_tokens=_integer(
            data.get("emitted_loss_tokens"), field="emitted_loss_tokens"
        ),
    )
    try:
        cursor.require_compatible(plan)
    except ValueError as exc:
        raise StreamingDataError(str(exc)) from exc
    return cursor


def mixture_runtime_cursor_from_dict(
    payload: Mapping[str, Any],
    plan: MixturePlan,
) -> MixtureRuntimeCursor:
    """Decode schedule + all per-source stream positions as one restart unit."""
    data = _mapping(payload, field="mixture runtime cursor")
    if data.get("schema") != MIXTURE_RUNTIME_CURSOR_SCHEMA:
        raise StreamingDataError("unsupported mixture runtime cursor schema")
    schedule = restart_cursor_from_dict(
        _mapping(data.get("schedule"), field="schedule"),
        plan,
    )
    raw_streams = _mapping(data.get("source_streams"), field="source_streams")
    expected_names = tuple(source.name for source in plan.ordered_sources)
    if tuple(sorted(raw_streams)) != expected_names:
        raise StreamingDataError("mixture stream source set does not match MixturePlan")
    streams = tuple(
        (
            name,
            stream_cursor_from_dict(
                _mapping(raw_streams[name], field=f"source_streams.{name}"),
                plan,
                source_name=name,
            ),
        )
        for name in expected_names
    )
    cursor = MixtureRuntimeCursor(schedule=schedule, source_streams=streams)
    cursor.require_compatible(plan)
    return cursor
