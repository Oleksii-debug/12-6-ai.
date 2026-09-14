from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.learned20m_capacity_sufficiency_v1 import (
    CapacityReportError,
    load_and_validate,
    validate_report,
)

REPORT = Path(__file__).parents[1] / "reports" / "learned20m_capacity_sufficiency_v1.json"
MAIN_SHA = "fc57531f7f8e7ef03323df641d8196af6c573c8b"


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_checked_in_report_validates() -> None:
    document = load_and_validate(REPORT, expected_main_sha=MAIN_SHA)
    assert document["decision"]["routing"] == "CONTINUE_ONLY_NAMED_HIGH_YIELD_EXECUTIONS"
    assert document["capacity_math"]["authorized_unique_loss_positions_now"] == 0
    minimum_deficit = document["capacity_math"][
        "minimum_deficit_if_oracle_becomes_exact_clean_successor"
    ]
    assert minimum_deficit == 4_571_642


def test_rejects_stale_main_root() -> None:
    with pytest.raises(CapacityReportError, match="stale report root"):
        load_and_validate(REPORT, expected_main_sha="0" * 40)


def test_rejects_bool_as_integer() -> None:
    document = _report()
    document["floor"]["meaningful_unique_loss_positions"] = True
    with pytest.raises(CapacityReportError, match="not bool"):
        validate_report(document)


def test_rejects_oracle_authority_promotion() -> None:
    document = _report()
    document["clean_base_oracle"]["launch_authoritative"] = True
    with pytest.raises(CapacityReportError, match="cannot become launch-authoritative"):
        validate_report(document)


def test_rejects_candidate_credit_promotion() -> None:
    document = _report()
    document["candidates"][0]["canonical_unique_loss_positions_credited"] = 1
    with pytest.raises(CapacityReportError, match="cannot receive unique-loss credit"):
        validate_report(document)


def test_rejects_candidate_global_dedup_promotion() -> None:
    document = _report()
    document["candidates"][0]["global_dedup_executed"] = True
    with pytest.raises(CapacityReportError, match="cannot promote"):
        validate_report(document)


def test_rejects_arithmetic_drift() -> None:
    document = _report()
    document["capacity_math"]["named_candidate_raw_bytes"] += 1
    with pytest.raises(CapacityReportError, match="arithmetic mismatch"):
        validate_report(document)


def test_rejects_stop_verdict_without_terminal_floor() -> None:
    document = _report()
    document["decision"]["routing"] = "STOP_NEW_SOURCE_ACQUISITION_FINISH_PIPELINE"
    with pytest.raises(CapacityReportError, match="STOP decision is illegal"):
        validate_report(document)


def test_rejects_new_source_fanout() -> None:
    document = _report()
    document["decision"]["open_new_source_families"] = True
    with pytest.raises(CapacityReportError, match="must not fan out"):
        validate_report(document)


def test_rejects_numeric_expected_post_filter_range() -> None:
    document = _report()
    document["capacity_math"]["post_filter_unique_loss_expected_range_authorized"] = [1, 2]
    with pytest.raises(CapacityReportError, match="does not authorize"):
        validate_report(document)


def test_rejects_duplicate_json_key(tmp_path: Path) -> None:
    raw = REPORT.read_text(encoding="utf-8")
    raw = raw.replace(
        '"schema_version": "12-6.learned20m-capacity-sufficiency.v1",',
        '"schema_version": "12-6.learned20m-capacity-sufficiency.v1",\n'
        '  "schema_version": "12-6.learned20m-capacity-sufficiency.v1",',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(CapacityReportError, match="duplicate JSON key"):
        load_and_validate(path)


def test_raw_byte_ceiling_never_becomes_expected_unique_loss_range() -> None:
    document = _report()
    raw_ceiling = document["capacity_math"]["post_filter_mathematical_upper_bound"]
    assert raw_ceiling == 29_627_353
    assert document["capacity_math"]["post_filter_unique_loss_expected_range_authorized"] is None
    assert raw_ceiling > document["floor"]["meaningful_unique_loss_positions"]


def test_rejects_candidate_exact_head_drift() -> None:
    document = _report()
    document["candidates"][0]["exact_head"] = "0" * 40
    with pytest.raises(CapacityReportError, match="evidence root or physical accounting drifted"):
        validate_report(document)


def test_rejects_global_dedup_released_head_drift() -> None:
    document = _report()
    document["evidence_roots"]["global_dedup_released_head"] = "0" * 40
    with pytest.raises(CapacityReportError, match="released head drifted"):
        validate_report(document)


def test_rejects_global_dedup_ci_promotion_or_drift() -> None:
    document = _report()
    document["evidence_roots"]["global_dedup_exact_head_ci_conclusion"] = "FAILURE"
    with pytest.raises(CapacityReportError, match="CI is not the observed"):
        validate_report(document)


def test_rejects_immediate_trigger_reordering() -> None:
    document = _report()
    document["decision"]["immediate_triggers"] = list(
        reversed(document["decision"]["immediate_triggers"])
    )
    with pytest.raises(CapacityReportError, match="immediate_triggers drifted"):
        validate_report(document)
