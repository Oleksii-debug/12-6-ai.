from __future__ import annotations

from twelve_six.packing.core import TextRecord
from twelve_six.packing.scale_contracts import MixturePlan, MixtureSource
from twelve_six.packing.streaming import StreamCursor, iter_packed_stream, iter_trainer_batches
from twelve_six.packing.streaming_mixture import MixtureRuntimeCursor, iter_mixture_batches
from twelve_six.tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer


def _plan() -> MixturePlan:
    return MixturePlan(
        plan_id="streaming-mixture-test",
        tokenizer_config_sha256=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
        packing_config_sha256="5" * 64,
        sources=(
            MixtureSource("english", "6" * 64, 3),
            MixtureSource("ukrainian", "7" * 64, 2),
        ),
        seed=123,
        num_shards=8,
        shard_seed=456,
    )


def _records(prefix: str, text: str, count: int = 200):
    return tuple(
        TextRecord(f"{prefix}-{index:04d}", f"{text} {index} " * (2 + index % 7), "train")
        for index in range(count)
    )


def _source_batches(plan, source_name, records, cursor=None):
    tokenizer = ByteTokenizer()
    items = iter_packed_stream(
        records,
        tokenizer,
        plan,
        source_name=source_name,
        split="train",
        cursor=cursor,
        sequence_length=64,
    )
    return iter_trainer_batches(items, batch_size=4)


def _initial(plan: MixturePlan) -> MixtureRuntimeCursor:
    return MixtureRuntimeCursor.initial(
        plan,
        {
            source.name: StreamCursor.initial(plan, source_name=source.name, split="train")
            for source in plan.ordered_sources
        },
    )


def test_mixture_schedule_and_stream_cursors_resume_exact_suffix() -> None:
    plan = _plan()
    corpora = {
        "english": _records("en", "English general pretraining code"),
        "ukrainian": _records("uk", "Український загальний передтренувальний текст"),
    }
    initial = _initial(plan)
    uninterrupted = list(
        iter_mixture_batches(
            plan,
            {
                name: _source_batches(plan, name, records)
                for name, records in corpora.items()
            },
            cursor=initial,
            max_batches=30,
        )
    )
    cut = 11
    checkpoint = uninterrupted[cut - 1].cursor_after
    resumed = list(
        iter_mixture_batches(
            plan,
            {
                name: _source_batches(
                    plan,
                    name,
                    records,
                    cursor=checkpoint.stream_for(name),
                )
                for name, records in corpora.items()
            },
            cursor=checkpoint,
            max_batches=30 - cut,
        )
    )
    assert [batch.source_name for batch in resumed] == [
        batch.source_name for batch in uninterrupted[cut:]
    ]
    assert [batch.loss_tokens for batch in resumed] == [
        batch.loss_tokens for batch in uninterrupted[cut:]
    ]
    assert resumed[-1].cursor_after.schedule.next_sample_index == 30
    assert resumed[-1].cursor_after.schedule.emitted_loss_tokens == sum(
        batch.loss_tokens for batch in uninterrupted
    )
