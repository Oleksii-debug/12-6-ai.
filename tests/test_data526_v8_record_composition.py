import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

composer = importlib.import_module("compose_data526_records_from_v8")


CONFIG = ROOT / "configs" / "data" / "data526_v8_record_composition_v1.json"

EXPECTED_V8_REPORT = "942cc15af60ee36a79345beba77fec347e33ee919d8148fb319b19d9ee072e5a"
EXPECTED_V8_V3 = "e7244cb7f6df6062dc838112b1e3e6fe87e41644a9e8b0071e93c50f1166b67d"
EXPECTED_SURVIVORS = "8115b35662fa89900b37f54f7cffd5f2fa8b2f8c60a72bcb18b89c92aaade0cf"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _source_row(index: int, *, modality: str = "code") -> dict:
    digest = f"{index + 1:064x}"[-64:]
    return {
        "source_id": f"source-{index:03d}",
        "source_family": f"family-{index % 7}",
        "modality": modality,
        "declared_capacity_bytes": index + 10,
        "verified_raw_sha256": digest,
        "normalized_sha256": f"{index + 1001:064x}"[-64:],
        "stable_origin_id_sha256": f"{index + 2001:064x}"[-64:],
        "stable_object_id_sha256": f"{index + 3001:064x}"[-64:],
    }


def _synthetic_report_and_survivor() -> tuple[dict, dict]:
    sources = [_source_row(index) for index in range(composer.EXPECTED_COMPOSED_SOURCES)]
    survivor_rows = [copy.deepcopy(row) for row in sources[:-2]]
    source_bytes = sum(row["declared_capacity_bytes"] for row in sources)
    survivor_bytes = sum(row["declared_capacity_bytes"] for row in survivor_rows)
    by_modality = {
        modality: {
            "source_object_count": sum(1 for row in survivor_rows if row["modality"] == modality),
            "declared_capacity_bytes": sum(
                row["declared_capacity_bytes"]
                for row in survivor_rows
                if row["modality"] == modality
            ),
        }
        for modality in ("uk", "en", "code")
    }
    report = {"dedup_v3": {"sources": sources}}
    survivor = {
        "pre_dedup_source_object_count": len(sources),
        "post_dedup_survivor_source_object_count": len(survivor_rows),
        "pre_dedup_declared_capacity_bytes": source_bytes,
        "post_dedup_declared_capacity_bytes": survivor_bytes,
        "duplicate_discount_bytes": source_bytes - survivor_bytes,
        "duplicate_clusters": [
            {
                "member_source_ids": [
                    survivor_rows[0]["source_id"],
                    sources[-2]["source_id"],
                    sources[-1]["source_id"],
                ],
                "selected_source_id": survivor_rows[0]["source_id"],
                "selected_declared_capacity_bytes": survivor_rows[0]["declared_capacity_bytes"],
            }
        ],
        "duplicate_cluster_count": 1,
        "survivors": survivor_rows,
        "by_modality": by_modality,
    }
    return report, survivor


def test_exact_terminal_v8_authority_is_sealed_and_accepted():
    config = _config()
    composer.verify_config(config, require_terminal_v8=True)
    v8 = config["v8_terminal_input"]
    assert v8["status"] == composer.TERMINAL_V8_STATUS
    assert v8["report_sha256"] == EXPECTED_V8_REPORT
    assert v8["nested_v3_report_sha256"] == EXPECTED_V8_V3
    assert v8["survivor_authority_sha256"] == EXPECTED_SURVIVORS
    assert v8["workflow_run_id"] == 34159790818
    assert v8["workflow_conclusion"] == "success"
    assert v8["artifact_id"] == 10032510626
    assert v8["artifact_digest"] == composer.EXPECTED_V8_ARTIFACT_DIGEST


def test_terminal_v8_config_self_consistent_substitution_is_rejected():
    config = _config()
    config["v8_terminal_input"]["survivor_authority_sha256"] = "a" * 64
    with pytest.raises(composer.Data526V8Error, match="authority drift/substitution"):
        composer.verify_config(config, require_terminal_v8=True)

    config = _config()
    config["v8_terminal_input"]["workflow_run_id"] += 1
    with pytest.raises(composer.Data526V8Error, match="authority drift/substitution"):
        composer.verify_config(config, require_terminal_v8=True)


def test_v8_report_self_hash_is_recomputed_with_producer_canonicalization():
    report = {"schema_version": composer.V8_REPORT_SCHEMA, "unicode_probe": "Україна"}
    body = dict(report)
    report["report_sha256"] = composer._sha256(composer._canonical_utf8(body))
    composer._validate_v8_report_self_hash(report)

    report["unicode_probe"] = "tampered"
    with pytest.raises(composer.Data526V8Error, match="self-hash mismatch"):
        composer._validate_v8_report_self_hash(report)


def test_survivor_rows_must_match_nested_source_authority_metadata():
    report, survivor = _synthetic_report_and_survivor()
    composer._validate_survivor_rows_against_report(report, survivor)

    for key, replacement in (
        ("source_family", "substituted-family"),
        ("modality", "uk"),
        ("declared_capacity_bytes", 999999),
        ("verified_raw_sha256", "f" * 64),
        ("normalized_sha256", "e" * 64),
        ("stable_origin_id_sha256", "d" * 64),
        ("stable_object_id_sha256", "c" * 64),
    ):
        mutated = copy.deepcopy(survivor)
        mutated["survivors"][0][key] = replacement
        with pytest.raises(composer.Data526V8Error, match=f"{key} drift"):
            composer._validate_survivor_rows_against_report(report, mutated)


def test_survivor_row_summary_drift_is_rejected():
    report, survivor = _synthetic_report_and_survivor()
    mutated = copy.deepcopy(survivor)
    mutated["post_dedup_declared_capacity_bytes"] += 1
    with pytest.raises(composer.Data526V8Error, match="survivor row-byte summary drift"):
        composer._validate_survivor_rows_against_report(report, mutated)

    mutated = copy.deepcopy(survivor)
    mutated["by_modality"]["code"]["source_object_count"] -= 1
    with pytest.raises(composer.Data526V8Error, match="code summary drift"):
        composer._validate_survivor_rows_against_report(report, mutated)


def test_materialized_bulk_binding_rejects_metadata_and_hash_drift():
    row = _source_row(0)
    composer._validate_materialized_source_binding(
        row,
        source_id=row["source_id"],
        family=row["source_family"],
        modality=row["modality"],
        declared_capacity_bytes=row["declared_capacity_bytes"],
        verified_raw_sha256=row["verified_raw_sha256"],
    )
    cases = (
        {"family": "other-family"},
        {"modality": "uk"},
        {"declared_capacity_bytes": row["declared_capacity_bytes"] + 1},
        {"verified_raw_sha256": "0" * 64},
    )
    base = {
        "source_id": row["source_id"],
        "family": row["source_family"],
        "modality": row["modality"],
        "declared_capacity_bytes": row["declared_capacity_bytes"],
        "verified_raw_sha256": row["verified_raw_sha256"],
    }
    for overrides in cases:
        kwargs = {**base, **overrides}
        with pytest.raises(composer.Data526V8Error):
            composer._validate_materialized_source_binding(row, **kwargs)


def test_current_v8_record_graph_target_is_exact_and_fail_closed():
    config = _config()
    target = config["expected_terminal_composition"]

    assert target["pre_dedup_source_object_count"] == 264
    assert target["post_dedup_survivor_source_object_count"] == 262
    assert target["historical_record_count"] == composer.EXPECTED_HISTORICAL_RECORDS
    assert target["bulk_eligible_file_count"] == composer.EXPECTED_BULK_FILES
    assert target["bulk_removed_record_count"] == 2
    assert target["post_v8_record_count"] == 275
    assert target["post_v8_payload_bytes"] == 6_095_321
    assert len(set(target["removed_bulk_source_ids"])) == 2

    boundary = config["claim_boundary"]
    assert boundary["terminal_v8_consumed"] is False
    assert boundary["record_graph_materialized"] is False
    assert boundary["authorized_unique_optimized_targets"] == 0
    assert boundary["optimizer_updates"] == 0
    assert boundary["training_executed"] is False
    assert boundary["paid_compute_used"] is False


def test_record_count_math_tracks_only_the_two_terminal_bulk_eliminations():
    config = _config()
    target = config["expected_terminal_composition"]
    expected = (
        target["historical_record_count"]
        + target["bulk_eligible_file_count"]
        - target["bulk_removed_record_count"]
    )
    assert expected == target["post_v8_record_count"] == 275


def test_execution_head_sha_binding_is_fail_closed():
    exact = "a" * 40
    assert composer._validated_git_sha(exact) == exact
    for invalid in ("", "a" * 39, "a" * 41, "A" * 40, "g" * 40):
        with pytest.raises(composer.Data526V8Error):
            composer._validated_git_sha(invalid)
