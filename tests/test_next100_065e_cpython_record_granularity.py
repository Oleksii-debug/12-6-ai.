from __future__ import annotations

import pytest

from twelve_six.data import cross_source_capacity_audit_v5 as v5
from twelve_six.data import cross_source_capacity_audit_v7 as v7


def _spec(payloads: list[bytes]) -> dict[str, object]:
    return {
        "source_id": "en.python.docs.tutorial-introduction",
        "source_family": "github:python/cpython",
        "stable_origin_id": "github:python/cpython",
        "modality": "en",
        "head_sha": "a" * 40,
        "dedicated_workflow_run": 1,
        "upstream_commit": "b" * 40,
        "upstream_path": "Doc/tutorial/introduction.rst",
        "accepted_normalized_sha256": [v5._sha256(payload) for payload in payloads],
    }


def test_cpython_accepted_chunks_are_independent_dedup_objects() -> None:
    payloads = [f"accepted-{index}".encode() for index in range(14)]
    chunk_indexes = [index for index in range(16) if index not in {3, 11}]
    rows, mapped, evidence = v7._cpython_record_rows(
        _spec(payloads),
        payloads,
        chunk_indexes,
    )

    assert len(rows) == 14
    assert len(mapped) == 14
    assert len(evidence) == 14
    assert len({row["source_id"] for row in rows}) == 14
    assert {row["source_family"] for row in rows} == {"github:python/cpython"}
    assert sum(row["declared_capacity_bytes"] for row in rows) == sum(
        len(payload) for payload in payloads
    )
    assert all(b"\n\n" not in payload for payload in mapped.values())


def test_cpython_granularity_rejects_duplicate_chunk_indexes() -> None:
    payloads = [f"accepted-{index}".encode() for index in range(14)]
    with pytest.raises(
        v7.CrossSourceV7Error,
        match="chunk indexes are not unique",
    ):
        v7._cpython_record_rows(
            _spec(payloads),
            payloads,
            [0] * 14,
        )


def test_cpython_granularity_rejects_aggregate_or_missing_record() -> None:
    payloads = [f"accepted-{index}".encode() for index in range(13)]
    with pytest.raises(
        v7.CrossSourceV7Error,
        match="record-count drift",
    ):
        v7._cpython_record_rows(
            _spec(payloads),
            payloads,
            list(range(13)),
        )


def test_v7_source_object_count_matches_record_granular_composition() -> None:
    assert v7.EXPECTED_SOURCE_OBJECT_COUNT == 21 + 1 + 14 + 5 + 3
