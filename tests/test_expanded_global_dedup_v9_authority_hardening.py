from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data.expanded_global_dedup_v9 import (
    RADA_DATASET,
    RADA_FAMILY,
    RADA_REVISION,
    V8_NESTED_V3_SHA256,
    ExpandedDedupError,
    _restrict_lineage_to_survivors,
    _validate_reconstructed_v8_against_preflight,
    validate_rada_rows,
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _rada_row(text: str = "alpha") -> dict[str, object]:
    raw = text.encode("utf-8")
    digest = _sha(raw)
    return {
        "record_id": "parent-a",
        "quality_parent_record_id": "parent-a",
        "source_dataset": RADA_DATASET,
        "source_revision": RADA_REVISION,
        "source_family": RADA_FAMILY,
        "source_path": "texts/parent-a.txt",
        "source_payload_sha256": "b" * 64,
        "source_payload_bytes": 100,
        "quality_privacy_complete": True,
        "language_quality_privacy_complete": False,
        "privacy_gate_action": "ALLOW",
        "training_eligible": False,
        "evaluation_eligible": False,
        "global_dedup_complete": False,
        "reserved_evaluation_decontamination_complete": False,
        "quality_gate_status": "RETAIN_ALL",
        "quality_unit_kind": "SOURCE_NATIVE_DOCUMENT",
        "quality_window_index": None,
        "quality_unit_utf8_sha256": digest,
        "decoded_text_utf8_sha256": digest,
        "privacy_scan_input_sha256": digest,
        "quality_unit_utf8_bytes": len(raw),
        "decoded_text_utf8_bytes": len(raw),
        "privacy_scan_input_bytes": len(raw),
        "text": text,
    }


def _authenticated_rada_fixture() -> tuple[list[dict[str, object]], bytes, dict[str, object]]:
    rows = [_rada_row()]
    raw = (
        json.dumps(rows[0], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    report: dict[str, object] = {
        "output_jsonl_sha256": _sha(raw),
        "output_records": 1,
        "output_text_utf8_bytes": len(b"alpha"),
        "quality": {"partial_units_materialized_before_privacy": 0},
        "privacy": {"held_units": 0},
    }
    return rows, raw, report


def test_authenticated_rada_rows_are_the_matcher_rows() -> None:
    rows, raw, report = _authenticated_rada_fixture()
    assert validate_rada_rows(rows, raw, report) == rows


def test_authenticated_rada_raw_rejects_same_size_substituted_rows() -> None:
    rows, raw, report = _authenticated_rada_fixture()
    substituted = [_rada_row("omega")]
    assert len(str(rows[0]["text"]).encode()) == len(str(substituted[0]["text"]).encode())
    with pytest.raises(ExpandedDedupError, match="do not match authenticated output JSONL"):
        validate_rada_rows(substituted, raw, report)


def _v8_semantic_fixture() -> tuple[
    dict[str, object], dict[str, bytes], dict[str, object], dict[str, object]
]:
    payload = b"base"
    raw_sha = _sha(payload)
    stable_origin = "origin:stable"
    stable_object = "object:stable"
    source = {
        "source_id": "base-1",
        "source_family": "family",
        "stable_origin_id": stable_origin,
        "stable_object_id": stable_object,
        "modality": "en",
        "evidence_status": "DEDICATED_TERMINAL",
        "declared_capacity_bytes": len(payload),
        "comparison_normalization": None,
    }
    projection = {
        "source_id": "base-1",
        "source_family": "family",
        "stable_origin_id_sha256": _sha(stable_origin.encode()),
        "stable_object_id_sha256": _sha(stable_object.encode()),
        "modality": "en",
        "evidence_status": "DEDICATED_TERMINAL",
        "declared_capacity_bytes": len(payload),
        "verified_raw_bytes": len(payload),
        "verified_raw_sha256": raw_sha,
        "comparison_policy": "DATA232_GENERIC_FROM_RAW",
        "comparison_payload_bytes": len(payload),
        "comparison_payload_sha256": raw_sha,
        "normalized_sha256": "c" * 64,
    }
    survivor = {
        "source_id": "base-1",
        **{
            key: projection[key]
            for key in (
                "source_family",
                "modality",
                "declared_capacity_bytes",
                "verified_raw_sha256",
                "normalized_sha256",
                "stable_origin_id_sha256",
                "stable_object_id_sha256",
            )
        },
    }
    inventory: dict[str, object] = {"sources": [source], "lineage_edges": []}
    payloads = {"base-1": payload}
    survivor_authority: dict[str, object] = {"survivors": [survivor]}
    preflight: dict[str, object] = {
        "report_sha256": V8_NESTED_V3_SHA256,
        "sources": [projection],
    }
    return inventory, payloads, survivor_authority, preflight


def test_v8_preflight_cross_binds_stable_and_normalized_identities() -> None:
    inventory, payloads, survivor_authority, preflight = _v8_semantic_fixture()
    _validate_reconstructed_v8_against_preflight(
        inventory,
        payloads,
        survivor_authority,
        preflight,
    )

    bad = copy.deepcopy(inventory)
    bad["sources"][0]["stable_origin_id"] = "substituted-origin"
    with pytest.raises(ExpandedDedupError, match="stable origin authority drift"):
        _validate_reconstructed_v8_against_preflight(
            bad,
            payloads,
            survivor_authority,
            preflight,
        )


def test_v8_preflight_cross_binds_comparison_metadata() -> None:
    inventory, payloads, survivor_authority, preflight = _v8_semantic_fixture()
    bad = copy.deepcopy(inventory)
    bad["sources"][0]["comparison_normalization"] = "RUST_BOOK_SOURCE_MARKDOWN_PROSE_ONLY_V1"
    with pytest.raises(ExpandedDedupError, match="generic comparison metadata substituted"):
        _validate_reconstructed_v8_against_preflight(
            bad,
            payloads,
            survivor_authority,
            preflight,
        )


def test_non_survivor_lineage_edges_are_removed_before_expanded_matching() -> None:
    inventory = {
        "sources": [
            {"source_id": "a"},
            {"source_id": "b"},
            {"source_id": "dropped"},
        ],
        "lineage_edges": [
            {"left_source_id": "a", "right_source_id": "b", "relation": "fork"},
            {"left_source_id": "a", "right_source_id": "dropped", "relation": "fork"},
        ],
    }
    survivors = {"survivors": [{"source_id": "a"}, {"source_id": "b"}]}
    prepared = _restrict_lineage_to_survivors(inventory, survivors)
    assert prepared["lineage_edges"] == [inventory["lineage_edges"][0]]
