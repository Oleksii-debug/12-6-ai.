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
MAIN_SHA = "1b6bcb08486f1f4fc5c47a6419631b636837d50d"


def _report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


def test_checked_in_report_validates() -> None:
    document = load_and_validate(REPORT, expected_main_sha=MAIN_SHA)
    assert document["scientific_truth"]["authorized_postpack_unique_loss_positions"] == 0
    assert document["decision"]["stop_new_source_acquisition_finish_pipeline"] is False


def test_rejects_stale_main_root() -> None:
    with pytest.raises(CapacityReportError, match="stale report root"):
        load_and_validate(REPORT, expected_main_sha="0" * 40)


def test_rejects_previous_main_even_with_self_consistent_old_evidence() -> None:
    with pytest.raises(CapacityReportError, match="stale report root"):
        load_and_validate(REPORT, expected_main_sha="a1bc7430022d174d27ff57213212a0ec5cc7ed1e")


def test_rejects_old_report_main_after_current_main_rebind() -> None:
    with pytest.raises(CapacityReportError, match="stale report root"):
        load_and_validate(
            REPORT, expected_main_sha="903bbd642068cf00eab0f864a739b7f7e1d280ea"
        )


def test_rejects_immediately_previous_main_after_current_main_rebind() -> None:
    with pytest.raises(CapacityReportError, match="stale report root"):
        load_and_validate(
            REPORT, expected_main_sha="49bdba879e400e217a8e3b5c1b4015c51177ef58"
        )


def test_rejects_main31_after_current_main_rebind() -> None:
    with pytest.raises(CapacityReportError, match="stale report root"):
        load_and_validate(
            REPORT, expected_main_sha="31b9e030cb179e895fcbf3404b1d1fd045515867"
        )


def test_rejects_mainbcc8_after_current_main_rebind() -> None:
    with pytest.raises(CapacityReportError, match="stale report root"):
        load_and_validate(
            REPORT, expected_main_sha="bcc8f75090dd31a5e42a9045d96044d9721bebae"
        )


def test_rejects_control_issue_drift() -> None:
    document = _report()
    document["control_issues"][-1] = 9999
    with pytest.raises(CapacityReportError, match="control_issues drifted"):
        validate_report(document)


def test_rejects_policy_fraction_bool_alias() -> None:
    document = _report()
    document["balance_policy"]["max_family_fraction_total"]["numerator"] = True
    with pytest.raises(CapacityReportError, match="must be an integer, not bool"):
        validate_report(document)


def test_rejects_policy_strata_drift() -> None:
    document = _report()
    document["balance_policy"]["strata"]["ua"]["numerator"] = 8
    with pytest.raises(CapacityReportError, match="numerator drifted"):
        validate_report(document)


def test_rejects_pending_non_object_structure() -> None:
    document = _report()
    document["pending_existing_high_yield_work"] = ["rada_two_clean"]
    with pytest.raises(CapacityReportError, match="JSON array of objects"):
        validate_report(document)


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
    document["pending_existing_high_yield_work"][0][
        "optimistic_balance_upper_bound_bytes"
    ] = 5_000_001
    with pytest.raises(CapacityReportError, match="one-family global cap"):
        validate_report(document)


def test_rejects_legacy_balance_credit_field() -> None:
    document = _report()
    row = document["pending_existing_high_yield_work"][0]
    row["balance_credit_bytes"] = row["optimistic_balance_upper_bound_bytes"]
    with pytest.raises(CapacityReportError, match="legacy balance_credit_bytes field is forbidden"):
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
    document["downstream_authority_backlog"][1]["terminal_authority"] = True
    with pytest.raises(
        CapacityReportError,
        match="downstream terminal flag inconsistent with audited clean composition",
    ):
        validate_report(document)


@pytest.mark.parametrize(
    ("target_id", "forged_terminal"),
    [
        ("clean_current_composition", False),
        ("balance_execution", True),
        ("packing", True),
        ("postpack_unique_loss", True),
    ],
)
def test_downstream_terminal_flags_require_exact_audited_stage(
    target_id: str, forged_terminal: bool
) -> None:
    document = _report()
    for row in document["downstream_authority_backlog"]:
        if row["id"] == target_id:
            row["terminal_authority"] = forged_terminal
            break
    else:
        pytest.fail(f"missing downstream stage fixture: {target_id}")
    with pytest.raises(
        CapacityReportError,
        match="downstream terminal flag inconsistent with audited clean composition",
    ):
        validate_report(document)


@pytest.mark.parametrize("forged_terminal", [1, "true", None])
def test_downstream_terminal_flags_reject_non_boolean_values(
    forged_terminal: object,
) -> None:
    document = _report()
    document["downstream_authority_backlog"][1]["terminal_authority"] = forged_terminal
    with pytest.raises(CapacityReportError, match="must be a JSON boolean"):
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
        '"schema_version": "12-6.learned20m-capacity-sufficiency.v3",',
        '"schema_version": "12-6.learned20m-capacity-sufficiency.v3",\n'
        '  "schema_version": "12-6.learned20m-capacity-sufficiency.v3",',
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
    marker = '"claim_issue": 2570'
    assert raw.count(marker) == 1
    raw = raw.replace(marker, f'"claim_issue": {number}', 1)
    path = tmp_path / "nonfinite.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(CapacityReportError, match="nonfinite JSON number"):
        load_and_validate(path)


def test_rejects_unknown_top_level_key() -> None:
    document = _report()
    document["future_credit"] = 1
    with pytest.raises(CapacityReportError, match="report keys mismatch"):
        validate_report(document)

def test_post_qp_supply_is_separate_and_zero_credit() -> None:
    document = _report()
    assert document["terminal_physical_clean_supply"]["records"] == 257
    assert document["terminal_clean_post_qp_supply"]["records"] == 254
    assert document["terminal_clean_post_qp_supply"]["payload_utf8_bytes"] == 5_428_358
    assert document["terminal_clean_post_qp_supply"]["capacity_credit_bytes"] == 0
    assert document["optimistic_source_mixture_bound"]["terminal_clean_payload_bytes"] == 5_601_716
    validate_report(document)


@pytest.mark.parametrize("field,value", [
    ("records", 257),
    ("payload_utf8_bytes", 5_601_716),
    ("run_id", 36893337850),
    ("artifact_zip_sha256", "0" * 64),
    ("records_jsonl_sha256", "0" * 64),
    ("upstream_pre_qp_materialization_identity_sha256", "0" * 64),
])
def test_rejects_post_qp_stage_or_physical_root_substitution(
    field: str, value: object
) -> None:
    document = _report()
    document["terminal_clean_post_qp_supply"][field] = value
    with pytest.raises(CapacityReportError, match="post-QP"):
        validate_report(document)


@pytest.mark.parametrize("stratum,value", [
    ("ua", 5_601_716),
    ("en", True),
    ("code", -1),
])
def test_rejects_post_qp_stratum_substitution(stratum: str, value: object) -> None:
    document = _report()
    document["terminal_clean_post_qp_supply"]["by_stratum_bytes"][stratum] = value
    with pytest.raises(CapacityReportError, match="post-QP"):
        validate_report(document)


def test_rejects_post_qp_source_credit_promotion() -> None:
    document = _report()
    document["terminal_clean_post_qp_supply"]["capacity_credit_bytes"] = 1
    with pytest.raises(CapacityReportError, match="cannot grant canonical capacity"):
        validate_report(document)


def test_rejects_missing_post_qp_provenance_field() -> None:
    document = _report()
    del document["terminal_clean_post_qp_supply"]["artifact_id"]
    with pytest.raises(CapacityReportError, match="post-QP evidence keys mismatch"):
        validate_report(document)


def test_rejects_extra_post_qp_provenance_field() -> None:
    document = _report()
    document["terminal_clean_post_qp_supply"]["foreign_root"] = "0" * 64
    with pytest.raises(CapacityReportError, match="post-QP evidence keys mismatch"):
        validate_report(document)


@pytest.mark.parametrize("field,value", [
    ("en_clean_baseline_bytes", 2_977_845),
    ("en_gap_before_new_candidates_bytes", 4_022_155),
    ("caselaw_max_optimistic_new_clean_increment_bytes", 5_000_000),
    ("minimum_other_independent_clean_en_bytes", 0),
    ("code_clean_gap_before_new_candidates_bytes", True),
])
def test_rejects_en_early_stage_or_family_cap_miscount(
    field: str, value: object
) -> None:
    document = _report()
    document["en_family_cap_necessary_condition"][field] = value
    with pytest.raises(CapacityReportError, match="EN condition|EN family-cap"):
        validate_report(document)


def test_rejects_theoretical_caselaw_as_observed_clean_capacity() -> None:
    document = _report()
    document["en_family_cap_necessary_condition"][
        "candidate_bytes_are_not_observed_clean_survivors"
    ] = False
    with pytest.raises(CapacityReportError, match="candidate bytes"):
        validate_report(document)


def test_rejects_optimistic_earlier_stage_double_counting() -> None:
    document = _report()
    document["optimistic_source_mixture_bound"]["terminal_clean_payload_bytes"] = 5_428_358
    with pytest.raises(CapacityReportError, match="mixes clean stages"):
        validate_report(document)


def test_rejects_unknown_en_condition_key() -> None:
    document = _report()
    document["en_family_cap_necessary_condition"]["physical_credit"] = 1
    with pytest.raises(CapacityReportError, match="EN necessary-condition keys mismatch"):
        validate_report(document)


def test_rejects_downstream_clean_qp_terminal_reversal() -> None:
    document = _report()
    document["downstream_authority_backlog"][0]["terminal_authority"] = False
    with pytest.raises(CapacityReportError, match="downstream terminal flag"):
        validate_report(document)


def test_rejects_en_condition_credit_promotion() -> None:
    document = _report()
    document["en_family_cap_necessary_condition"]["authoritative_capacity_credit_bytes"] = 1
    with pytest.raises(CapacityReportError, match="cannot grant capacity credit"):
        validate_report(document)


@pytest.mark.parametrize("depth", [80, 1_300])
def test_rejects_excessively_nested_json_without_traceback(
    tmp_path: Path, depth: int
) -> None:
    raw = REPORT.read_text(encoding="utf-8")
    marker = '"claim_issue": 2570'
    assert raw.count(marker) == 1
    value = "[" * depth + "2570" + "]" * depth
    path = tmp_path / "nested.json"
    path.write_text(
        raw.replace(marker, f'"claim_issue": {value}', 1), encoding="utf-8"
    )
    with pytest.raises(
        CapacityReportError, match="depth or node limit|invalid capacity report JSON"
    ):
        load_and_validate(path)


def test_rejects_deep_direct_report_before_canonical_serialization() -> None:
    document = _report()
    value: object = "leaf"
    for _ in range(80):
        value = [value]
    document["decision"]["untrusted_metadata"] = value
    with pytest.raises(CapacityReportError, match="depth or node limit"):
        validate_report(document)


def test_rejects_oversized_report_before_json_parse(tmp_path: Path) -> None:
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (1_048_576 + 1))
    with pytest.raises(CapacityReportError, match="byte limit"):
        load_and_validate(path)


def test_rejects_invalid_utf8_without_traceback(tmp_path: Path) -> None:
    path = tmp_path / "invalid-utf8.json"
    path.write_bytes(bytes([0xFF]))
    with pytest.raises(CapacityReportError, match="invalid capacity report JSON"):
        load_and_validate(path)


def test_rejects_missing_report_with_controlled_error(tmp_path: Path) -> None:
    with pytest.raises(CapacityReportError, match="cannot read capacity report"):
        load_and_validate(tmp_path / "missing.json")


def test_rejects_non_json_direct_value() -> None:
    document = _report()
    document["decision"]["untrusted_metadata"] = object()
    with pytest.raises(CapacityReportError, match="non-JSON value"):
        validate_report(document)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_direct_report_values(value: float) -> None:
    document = _report()
    document["decision"]["untrusted_metadata"] = value
    with pytest.raises(CapacityReportError, match="nonfinite JSON number"):
        validate_report(document)


@pytest.mark.parametrize("surrogate", [0xD800, 0xDFFF])
def test_rejects_unpaired_unicode_direct_report(surrogate: int) -> None:
    document = _report()
    document["decision"]["untrusted_metadata"] = chr(surrogate)
    with pytest.raises(CapacityReportError, match="invalid Unicode"):
        validate_report(document)


def test_rejects_escaped_unpaired_unicode_in_file(tmp_path: Path) -> None:
    raw = REPORT.read_text(encoding="utf-8")
    marker = '"claim_issue": 2570'
    assert raw.count(marker) == 1
    path = tmp_path / "surrogate.json"
    replacement = '"claim_issue": ' + json.dumps(chr(0xD800))
    path.write_text(raw.replace(marker, replacement, 1), encoding="utf-8")
    with pytest.raises(CapacityReportError, match="invalid Unicode"):
        load_and_validate(path)


def test_rejects_overlong_integer_literal_without_traceback(tmp_path: Path) -> None:
    raw = REPORT.read_text(encoding="utf-8")
    marker = '"claim_issue": 2570'
    assert raw.count(marker) == 1
    replacement = '"claim_issue": ' + "9" * 5_000
    path = tmp_path / "overlong-integer.json"
    path.write_text(raw.replace(marker, replacement, 1), encoding="utf-8")
    with pytest.raises(
        CapacityReportError, match="integer exceeds limit|invalid capacity report JSON"
    ):
        load_and_validate(path)


def test_rejects_huge_direct_integer_without_serialization() -> None:
    document = _report()
    document["decision"]["untrusted_metadata"] = 10**5_000
    with pytest.raises(CapacityReportError, match="integer exceeds limit"):
        validate_report(document)


@pytest.mark.parametrize(
    "path,bad",
    [
        (("terminal_physical_clean_supply",), None),
        (("terminal_physical_clean_supply",), []),
        (("balance_policy",), None),
        (("indexed_executor",), None),
        (("optimistic_source_mixture_bound",), None),
        (("decision",), None),
        (("pending_existing_high_yield_work", 0, "id"), []),
        (("existing_independent_family_backlog", 0, "id"), {}),
        (("downstream_authority_backlog", 0, "id"), []),
        (("pending_existing_high_yield_work", 0), None),
        (("existing_independent_family_backlog", 0), []),
        (("downstream_authority_backlog", 0), None),
    ],
)
@pytest.mark.parametrize("via_file", (False, True))
def test_malformed_capacity_nested_values_have_controlled_errors(
    tmp_path: Path, path: tuple[str | int, ...], bad: object, via_file: bool
) -> None:
    document = _report()
    parent = document
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = bad
    with pytest.raises(
        CapacityReportError,
        match="must be a JSON object|must be a string|JSON array of objects",
    ):
        if via_file:
            source = tmp_path / "malformed-capacity.json"
            source.write_text(json.dumps(document), encoding="utf-8")
            load_and_validate(source)
        else:
            validate_report(document)


@pytest.mark.parametrize(
    "path",
    [
        ("terminal_physical_clean_supply", "classification"),
        ("terminal_physical_clean_supply", "materialization_identity_sha256"),
        ("balance_policy", "target_total_source_bytes"),
        ("indexed_executor", "integrated"),
        ("optimistic_source_mixture_bound", "classification"),
        ("decision", "authorized_postpack_unique_loss_positions"),
        ("pending_existing_high_yield_work", 0, "id"),
        ("pending_existing_high_yield_work", 0, "capacity_credit_bytes"),
        ("pending_existing_high_yield_work", 1, "candidate_utf8_bytes"),
        ("existing_independent_family_backlog", 0, "observed_head"),
        ("downstream_authority_backlog", 0, "terminal_authority"),
    ],
)
@pytest.mark.parametrize("via_file", (False, True))
def test_missing_capacity_nested_fields_have_controlled_errors(
    tmp_path: Path, path: tuple[str | int, ...], via_file: bool
) -> None:
    document = _report()
    parent = document
    for part in path[:-1]:
        parent = parent[part]
    del parent[path[-1]]
    with pytest.raises(CapacityReportError, match="missing required fields"):
        if via_file:
            source = tmp_path / "missing-capacity-field.json"
            source.write_text(json.dumps(document), encoding="utf-8")
            load_and_validate(source)
        else:
            validate_report(document)
