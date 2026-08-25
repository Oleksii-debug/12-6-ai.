from __future__ import annotations

import copy

import pytest

from twelve_six.packing.cursor_codec import (
    mixture_runtime_cursor_from_dict,
    restart_cursor_from_dict,
    stream_cursor_from_dict,
)
from twelve_six.packing.scale_contracts import MixturePlan, MixtureSource
from twelve_six.packing.streaming import StreamCursor, StreamingDataError
from twelve_six.packing.streaming_mixture import MixtureRuntimeCursor
from twelve_six.tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH


def _plan(*, seed: int = 1) -> MixturePlan:
    return MixturePlan(
        plan_id=f"cursor-codec-{seed}",
        tokenizer_config_sha256=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
        packing_config_sha256="b" * 64,
        sources=(
            MixtureSource("english", "c" * 64, 3),
            MixtureSource("ukrainian", "d" * 64, 2),
        ),
        seed=seed,
        num_shards=8,
        shard_seed=19,
    )


def test_stream_cursor_data_only_round_trip_is_exact() -> None:
    plan = _plan()
    cursor = StreamCursor.initial(plan, source_name="english", split="train")
    decoded = stream_cursor_from_dict(
        cursor.to_dict(),
        plan,
        source_name="english",
        split="train",
    )
    assert decoded == cursor


def test_mixture_runtime_cursor_data_only_round_trip_is_exact() -> None:
    plan = _plan()
    cursor = MixtureRuntimeCursor.initial(
        plan,
        {
            source.name: StreamCursor.initial(
                plan,
                source_name=source.name,
                split="train",
            )
            for source in plan.ordered_sources
        },
    )
    decoded = mixture_runtime_cursor_from_dict(cursor.to_dict(), plan)
    assert decoded == cursor
    assert restart_cursor_from_dict(cursor.schedule.to_dict(), plan) == cursor.schedule


def test_cursor_schema_source_and_plan_drift_fail_closed() -> None:
    plan = _plan()
    cursor = StreamCursor.initial(plan, source_name="english", split="train")
    bad_schema = copy.deepcopy(cursor.to_dict())
    bad_schema["schema"] = "unknown"
    with pytest.raises(StreamingDataError, match="schema"):
        stream_cursor_from_dict(bad_schema, plan)

    with pytest.raises(StreamingDataError, match="source/split identity"):
        stream_cursor_from_dict(
            cursor.to_dict(),
            plan,
            source_name="ukrainian",
            split="train",
        )

    with pytest.raises(StreamingDataError, match="different MixturePlan"):
        stream_cursor_from_dict(cursor.to_dict(), _plan(seed=2))
