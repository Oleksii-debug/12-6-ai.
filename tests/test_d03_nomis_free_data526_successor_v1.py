from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/materialize_d03_nomis_free_data526_successor_v1.py"
SPEC = importlib.util.spec_from_file_location("swarm2065_clean_data526", TOOL)
assert SPEC is not None and SPEC.loader is not None
data526 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data526)


def test_clean_historical_and_final_oracles_are_nomisonly_delta() -> None:
    assert data526.EXPECTED_HISTORICAL_SOURCES == 34
    assert data526.EXPECTED_HISTORICAL_RECORDS == 47
    assert data526.EXPECTED_HISTORICAL_BYTES == 2_213_956
    assert data526.EXPECTED_HISTORICAL_DIRECT_SOURCES == 24
    assert data526.EXPECTED_HISTORICAL_DIRECT_BYTES == 2_015_905
    assert data526.EXPECTED_HISTORICAL_MODALITY_BYTES == {
        "uk": 99_197,
        "en": 1_838_293,
        "code": 276_466,
    }
    assert data526.EXPECTED_SURVIVOR_SOURCES == 261
    assert data526.EXPECTED_FINAL_BYTES == 6_093_662
    assert data526.EXPECTED_FINAL_RECORDS == 274


def test_truth_boundary_never_promotes_clean_record_graph_to_training() -> None:
    boundary = data526.TRUTH_BOUNDARY
    assert boundary["clean_data526_record_graph_materialized"] is True
    assert boundary["corpus_released"] is False
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["authorized_training_exposure"] == 0
    assert boundary["model_training_executed"] is False
    assert boundary["optimizer_updates"] == 0
    assert boundary["learned_weights_created"] is False
    assert boundary["final_test_payload_read"] is False
    assert boundary["paid_compute_used"] is False


def test_historical_aggregate_keeps_multi_record_source_capacity() -> None:
    records = [
        {
            "record_id": "a#1",
            "source_id": "a",
            "family": "fa",
            "modality": "code",
            "normalized_payload": "abc",
        },
        {
            "record_id": "a#2",
            "source_id": "a",
            "family": "fa",
            "modality": "code",
            "normalized_payload": "de",
        },
        {
            "record_id": "b",
            "source_id": "b",
            "family": "fb",
            "modality": "uk",
            "normalized_payload": "ж",
        },
    ]
    aggregates = data526._historical_source_aggregates(records)
    assert aggregates["a"] == ("fa", "code", 5)
    assert aggregates["b"] == ("fb", "uk", len("ж".encode()))


def test_historical_aggregate_rejects_mixed_family_for_same_source() -> None:
    records = [
        {
            "record_id": "a#1",
            "source_id": "a",
            "family": "fa",
            "modality": "code",
            "normalized_payload": "a",
        },
        {
            "record_id": "a#2",
            "source_id": "a",
            "family": "fb",
            "modality": "code",
            "normalized_payload": "b",
        },
    ]
    with pytest.raises(data526.CleanData526Error, match="mixed historical family"):
        data526._historical_source_aggregates(records)


def test_composer_covers_exact_survivor_set_and_rejects_non_survivors() -> None:
    historical = [
        {
            "record_id": "hist",
            "source_id": "hist",
            "family": "family-hist",
            "modality": "uk",
            "normalized_payload": "abc",
        }
    ]
    bulk_raw = b"xyz"
    bulk_sha = hashlib.sha256(bulk_raw).hexdigest()
    bulk_rows = [
        {
            "source_id": "bulk",
            "source_family": "family-bulk",
            "modality": "code",
            "declared_capacity_bytes": len(bulk_raw),
            "expected_raw_sha256": bulk_sha,
        }
    ]
    survivor = {
        "survivors": [
            {
                "source_id": "hist",
                "source_family": "family-hist",
                "modality": "uk",
                "declared_capacity_bytes": 3,
                "verified_raw_sha256": "h" * 64,
            },
            {
                "source_id": "bulk",
                "source_family": "family-bulk",
                "modality": "code",
                "declared_capacity_bytes": 3,
                "verified_raw_sha256": bulk_sha,
            },
        ]
    }
    patches = (
        mock.patch.object(data526, "EXPECTED_HISTORICAL_SOURCES", 1),
        mock.patch.object(data526, "EXPECTED_BULK_SOURCES", 1),
        mock.patch.object(data526, "EXPECTED_COMPOSED_SOURCES", 2),
        mock.patch.object(data526, "EXPECTED_SURVIVOR_SOURCES", 2),
        mock.patch.object(data526, "EXPECTED_FINAL_BYTES", 6),
        mock.patch.object(data526, "EXPECTED_FINAL_RECORDS", 2),
    )
    for patch in patches:
        patch.start()
    try:
        records = data526.compose_clean_records(
            historical_records=historical,
            bulk_rows=bulk_rows,
            bulk_payloads={"bulk": bulk_raw},
            survivor_authority=survivor,
        )
    finally:
        for patch in reversed(patches):
            patch.stop()
    assert [record["record_id"] for record in records] == ["bulk", "hist"]


def test_execution_head_must_be_exact_lowercase_git_sha() -> None:
    assert data526._validated_git_sha("a" * 40) == "a" * 40
    with pytest.raises(data526.CleanData526Error, match="lowercase 40-hex"):
        data526._validated_git_sha("A" * 40)
