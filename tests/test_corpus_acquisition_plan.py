from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data.corpus_acquisition_plan import (
    CorpusAcquisitionPlanError,
    validate_acquisition_plan,
)

PLAN_PATH = Path("configs/data/research_corpus_v1_scalable_acquisition_plan_v1.json")


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _validate_for_planning(plan: dict) -> dict:
    return validate_acquisition_plan(plan, allow_stale_planning=True)


def test_current_plan_is_structurally_valid_for_read_only_planning_only() -> None:
    summary = _validate_for_planning(_plan())
    assert summary["status"] == "VALID_PLANNING_ARTIFACT_STALE_BASELINE_CAPACITY_ZERO"
    assert summary["baseline_refresh_required"] is True
    assert summary["execution_ready"] is False
    assert summary["blocking_reason"] == "PR527_CLOSED_UNMERGED_REBIND_TO_SURVIVING_NEXT100_063"
    assert summary["training_authorized_bytes"] == 0
    assert summary["long_training_authorized"] is False
    assert summary["planning_gap_bytes"]["total"] == 19_685_860
    assert summary["family_planning_caps"] == {
        "ua": 5_000_000,
        "en": 4_200_000,
        "code": 2_400_000,
    }
    assert summary["candidate_family_counts"]["ua"] >= 3
    assert summary["candidate_family_counts"]["en"] >= 3
    assert summary["candidate_family_counts"]["code"] >= 3


def test_default_validation_blocks_stale_baseline_before_execution_or_materialization() -> None:
    with pytest.raises(CorpusAcquisitionPlanError, match="PR #527 is closed unmerged"):
        validate_acquisition_plan(_plan())


def test_candidate_bytes_cannot_be_relabelled_as_capacity() -> None:
    plan = _plan()
    plan["candidate_streams"][0]["credited_bytes"] = 1
    with pytest.raises(CorpusAcquisitionPlanError, match="cannot credit corpus capacity"):
        _validate_for_planning(plan)


def test_pre_dedup_baseline_cannot_authorize_training() -> None:
    plan = _plan()
    plan["baseline"]["training_authorized_bytes"] = 314_140
    with pytest.raises(CorpusAcquisitionPlanError, match="cannot authorize training bytes"):
        _validate_for_planning(plan)


def test_long_training_cannot_be_enabled_by_planning_artifact() -> None:
    plan = _plan()
    plan["capacity_firewall"]["long_training_authorized"] = True
    with pytest.raises(CorpusAcquisitionPlanError, match="long training must remain blocked"):
        _validate_for_planning(plan)


def test_planning_gap_must_match_target_minus_observed() -> None:
    plan = _plan()
    plan["target"]["planning_gap_from_observed_pre_dedup"]["ua"] += 1
    plan["target"]["planning_gap_from_observed_pre_dedup"]["total"] += 1
    with pytest.raises(CorpusAcquisitionPlanError, match="target minus observed"):
        _validate_for_planning(plan)


def test_each_stratum_plan_must_cover_current_gap() -> None:
    plan = _plan()
    for row in plan["candidate_streams"]:
        if row["stratum"] == "code":
            row["planning_request_bytes"] = 1
    with pytest.raises(CorpusAcquisitionPlanError, match="code: planning requests"):
        _validate_for_planning(plan)


def test_rights_retests_cannot_be_removed() -> None:
    plan = _plan()
    plan["candidate_streams"][0]["required_retests"] = ["rights"]
    with pytest.raises(CorpusAcquisitionPlanError, match="three explicit admission retests"):
        _validate_for_planning(plan)


def test_candidate_status_cannot_be_promoted_to_pass() -> None:
    plan = copy.deepcopy(_plan())
    plan["candidate_streams"][0]["status"] = "PASS"
    with pytest.raises(CorpusAcquisitionPlanError, match="unsupported nonterminal status"):
        _validate_for_planning(plan)


def test_source_convergence_pr_and_head_are_hard_bound() -> None:
    plan = _plan()
    plan["baseline"]["source_convergence_pr"] = 538
    with pytest.raises(CorpusAcquisitionPlanError, match="PR binding changed"):
        _validate_for_planning(plan)

    plan = _plan()
    plan["baseline"]["source_convergence_head"] = "0" * 40
    with pytest.raises(CorpusAcquisitionPlanError, match="head binding changed"):
        _validate_for_planning(plan)


def test_baseline_status_cannot_be_silently_promoted() -> None:
    plan = _plan()
    plan["baseline"]["status"] = "TERMINAL"
    with pytest.raises(CorpusAcquisitionPlanError, match="baseline status changed"):
        _validate_for_planning(plan)


def test_observed_authority_vectors_are_hard_bound() -> None:
    plan = _plan()
    plan["baseline"]["observed_pre_dedup_bytes"]["en"] += 1
    plan["baseline"]["observed_pre_dedup_bytes"]["total"] += 1
    with pytest.raises(CorpusAcquisitionPlanError, match="authority vector changed"):
        _validate_for_planning(plan)

    plan = _plan()
    plan["baseline"]["observed_independent_families"]["en"] += 1
    plan["baseline"]["observed_independent_families"]["total"] += 1
    with pytest.raises(CorpusAcquisitionPlanError, match="family vector changed"):
        _validate_for_planning(plan)


def test_data295_policy_and_20m_target_are_hard_bound() -> None:
    plan = _plan()
    plan["target"]["policy"] = "UNREGISTERED_TARGET"
    with pytest.raises(CorpusAcquisitionPlanError, match="target policy changed"):
        _validate_for_planning(plan)

    plan = _plan()
    plan["target"]["bytes"]["ua"] -= 1_000_000
    plan["target"]["bytes"]["total"] -= 1_000_000
    plan["target"]["planning_gap_from_observed_pre_dedup"]["ua"] -= 1_000_000
    plan["target"]["planning_gap_from_observed_pre_dedup"]["total"] -= 1_000_000
    with pytest.raises(CorpusAcquisitionPlanError, match="20 MB target vector changed"):
        _validate_for_planning(plan)


def test_single_family_planning_budget_cannot_exceed_data295_cap() -> None:
    plan = _plan()
    row = next(
        item
        for item in plan["candidate_streams"]
        if item["family_candidate"] == "ua.rada.official-acts"
    )
    row["planning_request_bytes"] = 5_000_001
    with pytest.raises(CorpusAcquisitionPlanError, match="exceeds ua family cap"):
        _validate_for_planning(plan)


def test_same_family_cannot_cross_strata() -> None:
    plan = _plan()
    code_row = next(item for item in plan["candidate_streams"] if item["stratum"] == "code")
    code_row["family_candidate"] = "ua.rada.official-acts"
    with pytest.raises(CorpusAcquisitionPlanError, match="multiple strata"):
        _validate_for_planning(plan)


def test_capacity_credit_status_cannot_be_weakened() -> None:
    plan = _plan()
    plan["capacity_firewall"]["minimum_status_for_capacity_credit"] = "SOURCE_URL_FOUND"
    with pytest.raises(CorpusAcquisitionPlanError, match="capacity-credit status changed"):
        _validate_for_planning(plan)


def test_stale_execution_marker_requires_intentional_rebind_audit() -> None:
    plan = _plan()
    plan["next_execution_order"][0] = "execute_source_batches_now"
    with pytest.raises(CorpusAcquisitionPlanError, match="stale baseline execution marker"):
        _validate_for_planning(plan)
