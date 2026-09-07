from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/derive_next100_065f_v8_survivors.py"
SPEC = importlib.util.spec_from_file_location("derive_next100_065f_v8_survivors", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _source(source_id: str, capacity: int, modality: str = "code") -> dict[str, object]:
    return {
        "source_id": source_id,
        "source_family": f"family:{source_id}",
        "modality": modality,
        "declared_capacity_bytes": capacity,
        "verified_raw_sha256": (source_id[0] * 64),
        "normalized_sha256": (source_id[-1] * 64),
        "stable_origin_id_sha256": "1" * 64,
        "stable_object_id_sha256": "2" * 64,
    }


def _report() -> dict[str, object]:
    sources = [
        _source("a", 10, "uk"),
        _source("b", 15, "uk"),
        _source("c", 20, "en"),
        _source("d", 20, "en"),
        _source("e", 30, "code"),
    ]
    return {
        "report_sha256": "a" * 64,
        "source_vector": {
            "source_object_count": 5,
            "source_capacity_bytes_before_global_dedup": 95,
            "conservative_unique_capacity_bytes_after_global_dedup": 65,
            "duplicate_discount_bytes": 30,
            "duplicate_cluster_count": 2,
        },
        "dedup_v3": {
            "report_sha256": "b" * 64,
            "sources": sources,
            "terminal_candidates": {
                "duplicate_clusters": [["a", "b"], ["c", "d"]],
            },
        },
    }


def test_survivor_authority_uses_v3_capacity_rule_and_stable_tie_break() -> None:
    authority = MODULE.derive_survivor_authority(_report())
    assert [row["source_id"] for row in authority["survivors"]] == ["b", "c", "e"]
    assert authority["post_dedup_declared_capacity_bytes"] == 65
    assert authority["duplicate_discount_bytes"] == 30
    assert authority["duplicate_clusters"] == [
        {
            "member_source_ids": ["a", "b"],
            "selected_source_id": "b",
            "selected_declared_capacity_bytes": 15,
        },
        {
            "member_source_ids": ["c", "d"],
            "selected_source_id": "c",
            "selected_declared_capacity_bytes": 20,
        },
    ]
    assert authority["truth_boundary"]["training_record_inventory_materialized"] is False
    assert authority["truth_boundary"]["authorized_training_exposure"] == 0


def test_survivor_authority_is_deterministic_under_cluster_member_order() -> None:
    report = _report()
    first = MODULE.derive_survivor_authority(report)
    report["dedup_v3"]["terminal_candidates"]["duplicate_clusters"] = [["d", "c"], ["b", "a"]]
    second = MODULE.derive_survivor_authority(report)
    assert first == second


def test_survivor_authority_rejects_overlapping_clusters() -> None:
    report = _report()
    report["dedup_v3"]["terminal_candidates"]["duplicate_clusters"] = [["a", "b"], ["b", "c"]]
    with pytest.raises(MODULE.SurvivorAuthorityError, match="overlap"):
        MODULE.derive_survivor_authority(report)


def test_survivor_authority_rejects_capacity_not_reproducing_v3_summary() -> None:
    report = _report()
    report["source_vector"]["conservative_unique_capacity_bytes_after_global_dedup"] = 66
    with pytest.raises(MODULE.SurvivorAuthorityError, match="does not reproduce"):
        MODULE.derive_survivor_authority(report)


def test_verifier_rejects_mutated_selected_source() -> None:
    report = _report()
    authority = MODULE.derive_survivor_authority(report)
    mutated = copy.deepcopy(authority)
    mutated["duplicate_clusters"][0]["selected_source_id"] = "a"
    with pytest.raises(MODULE.SurvivorAuthorityError, match="exact deterministic derivation"):
        MODULE.verify_survivor_authority(report, mutated)
