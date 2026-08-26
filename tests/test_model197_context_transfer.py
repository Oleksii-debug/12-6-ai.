from __future__ import annotations

import math

from twelve_six.model197_context_transfer import (
    HORIZONS,
    _paired_bootstrap,
    _record_target_count,
)
from twelve_six.packing import TextRecord, collate_right_trimmed_rows, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer


def test_each_context_covers_same_within_document_targets_once() -> None:
    tok = ByteTokenizer()
    records = (
        TextRecord("a", "a" * 300, "train"),
        TextRecord("b", "b" * 700, "train"),
    )
    expected = sum(_record_target_count(tok, record) for record in records)
    counts = {}
    for horizon in HORIZONS:
        total = 0
        for record in records:
            for example in iter_packed_examples(
                (record,), tok, expected_split="train", sequence_length=horizon,
                cross_document=False,
            ):
                rows = collate_right_trimmed_rows((example,), target_mode="labels")
                total += sum(value != -100 for value in rows["labels"][0][1:])
        counts[horizon] = total
    assert counts == {256: expected, 512: expected, 1024: expected}


def test_byte_target_count_is_document_bytes_minus_one() -> None:
    tok = ByteTokenizer()
    record = TextRecord("uk", "Привіт", "train")
    assert _record_target_count(tok, record) == len("Привіт".encode("utf-8")) - 1


def test_paired_bootstrap_is_deterministic_and_directional() -> None:
    a = [
        {"record_id": "a", "byte_tokens": 600, "targets": 10, "nll": 20.0},
        {"record_id": "b", "byte_tokens": 700, "targets": 10, "nll": 22.0},
    ]
    b = [
        {"record_id": "a", "byte_tokens": 600, "targets": 10, "nll": 18.0},
        {"record_id": "b", "byte_tokens": 700, "targets": 10, "nll": 20.0},
    ]
    first = _paired_bootstrap(a, b, min_bytes=512)
    second = _paired_bootstrap(a, b, min_bytes=512)
    assert first == second
    assert first["eligible"] is True
    assert first["documents"] == 2
    assert first["delta_b_minus_a_bpb"] < 0
    assert math.isfinite(first["ci95"][0])
    assert first["ci95"][1] < 0
