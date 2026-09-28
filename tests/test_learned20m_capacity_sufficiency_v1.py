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
MAIN_SHA = "7b3df41c10a826183fab0b04ae85a90cdf0ce351"


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_checked_in_report_validates() -> None:
    document = load_and_validate(REPORT, expected_main_sha=MAIN_SHA)
    assert document["scientific_truth"]["authorized_postpack_unique_loss_positions"] == 0
    assert document["decision"]["stop_new_source_acquisition_finish_pipeline"] is False


def test_rejects_stale_main_root() -> None:
    with pytest.raises(CapacityReportError, match="stale report root"):
        load_and_validate(REPORT, expected_main_sha="0" * 40)


def test_rejects_bool_as_integer() -> None:
    document = _report()
    document["decision"]["meaningful_unique_loss_floor"] = True
    with pytest.raises(CapacityReportError, match="not bool"):
        validate_report(document)


def test_rejects_bool_integer_alias_in_scientific_truth() -> None:
    document = _report()
    document["scientific_truth"]["training_executed"] = 0
    with pytest.raises(CapacityReportError, match="must be a JSON boolean"):
        validate_report(document)


def test_rejects_integer_bool_alias_in_scientific_truth() -> None:
    document = _report()
    document["scientific_truth"]["authorized_optimized_target_exposure"] = False
    with pytest.raises(CapacityReportError, match="must be an integer, not bool"):
        validate_report(document)


def test_rejects_duplicate_pending_lane() -> None:
    document = _report()
    document["pending_existing_high_yield_work"].append(
        document["pending_existing_high_yield_work"][0].copy()
    )
    with pytest.raises(CapacityReportError, match="pending high-yield work set mismatch"):
        validate_report(document)


def test_rejects_indexed_executor_science_promotion() -> None:
    document = _report()
    document["indexed_executor"]["defines_new_matcher_science"] = True
    with pytest.raises(CapacityReportError, match="cannot claim new matcher science"):
        validate_report(document)


def test_rejects_clean_supply_credit_promotion() -> None:
    document = _report()
    document["terminal_physical_clean_supply"]["capacity_credit_bytes"] = 1
    with pytest.raises(CapacityReportError, match="cannot receive canonical capacity credit"):
        validate_report(document)


def test_rejects_clean_supply_launch_promotion() -> None:
    document = _report()
    document["terminal_physical_clean_supply"]["launch_authoritative"] = True
    with pytest.raises(CapacityReportError, match="not launch-authoritative"):
        validate_report(document)


def test_rejects_clean_materialization_identity_drift() -> None:
    document = _report()
    document["terminal_physical_clean_supply"]["materialization_identity_sha256"] = "0" * 64
    with pytest.raises(CapacityReportError, match="materialization_identity_sha256 drifted"):
        validate_report(document)


def test_rejects_balance_policy_identity_drift() -> None:
    document = _report()
    document["balance_policy"]["policy_identity_sha256"] = "0" * 64
    with pytest.raises(CapacityReportError, match="balance policy identity drifted"):
        validate_report(document)


def test_rejects_source_bytes_relabeling() -> None:
    document = _report()
    document["balance_policy"]["source_bytes_are_not_unique_loss_positions"] = False
    with pytest.raises(CapacityReportError, match="cannot be relabeled"):
        validate_report(document)


def test_rejects_indexed_executor_head_drift() -> None:
    document = _report()
    document["indexed_executor"]["head"] = "0" * 40
    with pytest.raises(CapacityReportError, match="indexed executor head drifted"):
        validate_report(document)


def test_rejects_pending_survivor_promotion() -> None:
    document = _report()
    document["pending_existing_high_yield_work"][0][
        "post_global_dedup_survivor_bytes"
    ] = 1
    with pytest.raises(CapacityReportError, match="cannot assert terminal"):
        validate_report(document)


def test_rejects_rada_cap_above_global_family_limit() -> None:
    document = _report()
    document["pending_existing_high_yield_work"][0]["balance_credit_bytes"] = 5_000_001
    with pytest.raises(CapacityReportError, match="one-family global cap"):
        validate_report(document)


def test_rejects_pending_capacity_credit_promotion() -> None:
    document = _report()
    document["pending_existing_high_yield_work"][1]["capacity_credit_bytes"] = 1
    with pytest.raises(CapacityReportError, match="cannot receive canonical capacity credit"):
        validate_report(document)


def test_rejects_independent_family_credit_promotion() -> None:
    document = _report()
    document["existing_independent_family_backlog"][0]["capacity_credit_bytes"] = 1
    with pytest.raises(CapacityReportError, match="cannot receive capacity credit"):
        validate_report(document)


def test_rejects_downstream_terminal_self_promotion() -> None:
    document = _report()
    document["downstream_authority_backlog"][0]["terminal_authority"] = True
    with pytest.raises(CapacityReportError, match="cannot self-promote"):
        validate_report(document)


def test_rejects_optimistic_arithmetic_drift() -> None:
    document = _report()
    document["optimistic_source_mixture_bound"][
        "optimistic_total_before_overlap_or_further_loss"
    ] += 1
    with pytest.raises(CapacityReportError, match="arithmetic drifted"):
        validate_report(document)


def test_rejects_optimistic_bound_as_authoritative_credit() -> None:
    document = _report()
    document["optimistic_source_mixture_bound"]["authoritative_capacity_credit_bytes"] = 1
    with pytest.raises(CapacityReportError, match="arithmetic drifted"):
        validate_report(document)


def test_rejects_stop_before_terminal_unique_loss_floor() -> None:
    document = _report()
    document["decision"]["stop_new_source_acquisition_finish_pipeline"] = True
    with pytest.raises(CapacityReportError, match="STOP is not authorized"):
        validate_report(document)


def test_rejects_new_distant_source_fanout() -> None:
    document = _report()
    document["decision"]["open_new_distant_source_families"] = True
    with pytest.raises(CapacityReportError, match="do not open distant source families"):
        validate_report(document)


def test_rejects_unique_loss_credit_without_ledger() -> None:
    document = _report()
    document["decision"]["authorized_postpack_unique_loss_positions"] = 1
    with pytest.raises(CapacityReportError, match="zero authorized postpack"):
        validate_report(document)


def test_rejects_duplicate_json_key(tmp_path: Path) -> None:
    raw = REPORT.read_text(encoding="utf-8")
    raw = raw.replace(
        '"schema_version": "12-6.learned20m-capacity-sufficiency.v2",',
        '"schema_version": "12-6.learned20m-capacity-sufficiency.v2",\n'
        '  "schema_version": "12-6.learned20m-capacity-sufficiency.v2",',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(CapacityReportError, match="duplicate JSON key"):
        load_and_validate(path)


@pytest.mark.parametrize(
    "number",
    ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"],
)
def test_rejects_nonfinite_json_numbers(tmp_path: Path, number: str) -> None:
    raw = REPORT.read_text(encoding="utf-8")
    raw = raw.replace('"claim_issue": 2277', f'"claim_issue": {number}', 1)
    path = tmp_path / "nonfinite.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(CapacityReportError, match="nonfinite JSON number"):
        load_and_validate(path)


def test_rejects_unknown_top_level_key() -> None:
    document = _report()
    document["future_credit"] = 1
    with pytest.raises(CapacityReportError, match="report keys mismatch"):
        validate_report(document)
