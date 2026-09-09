import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import compose_data526_records_from_v8 as composer


CONFIG = ROOT / "configs" / "data" / "data526_v8_record_composition_v1.json"

EXPECTED_V8_REPORT = "942cc15af60ee36a79345beba77fec347e33ee919d8148fb319b19d9ee072e5a"
EXPECTED_V8_V3 = "e7244cb7f6df6062dc838112b1e3e6fe87e41644a9e8b0071e93c50f1166b67d"
EXPECTED_SURVIVORS = "8115b35662fa89900b37f54f7cffd5f2fa8b2f8c60a72bcb18b89c92aaade0cf"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


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
