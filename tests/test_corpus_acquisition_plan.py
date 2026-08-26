from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data.corpus_acquisition_plan import (
    BASELINE_REGISTRY_IDENTITY,
    CorpusAcquisitionPlanError,
    validate_acquisition_plan,
)

PLAN_PATH = Path("configs/data/research_corpus_v1_scalable_acquisition_plan_v1.json")


def _plan() -> dict:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def test_current_plan_is_valid_and_authorizes_zero_training_bytes() -> None:
    summary = validate_acquisition_plan(_plan())
    assert summary["status"] == "VALID_PLANNING_ARTIFACT_CAPACITY_ZERO"
    assert summary["baseline_source_registry_identity_sha256"] == BASELINE_REGISTRY_IDENTITY
    assert summary["observed_pre_dedup_bytes"] == {
        "ua": 100_856,
        "en": 1_838_293,
        "code": 106_031,
        "total": 2_045_180,
    }
    assert summary["observed_independent_families"] == {
        "ua": 4,
        "en": 5,
        "code": 5,
        "total": 14,
    }
    assert summary["training_authorized_bytes"] == 0
    assert summary["long_training_authorized"] is False
    assert summary["planning_gap_bytes"] == {
        "ua": 8_899_144,
        "en": 5_161_707,
        "code": 3_893_969,
        "total": 17_954_820,
    }
    assert summary["candidate_family_counts"]["ua"] >= 3
    assert summary["candidate_family_counts"]["en"] >= 3
    assert summary["candidate_family_counts"]["code"] >= 3


def test_candidate_bytes_cannot_be_relabelled_as_capacity() -> None:
    plan = _plan()
    plan["candidate_streams"][0]["credited_bytes"] = 1
    with pytest.raises(CorpusAcquisitionPlanError, match="cannot credit corpus capacity"):
        validate_acquisition_plan(plan)


def test_pre_dedup_baseline_cannot_authorize_training() -> None:
    plan = _plan()
    plan["baseline"]["training_authorized_bytes"] = 2_045_180
    with pytest.raises(CorpusAcquisitionPlanError, match="cannot authorize training bytes"):
        validate_acquisition_plan(plan)


def test_long_training_cannot_be_enabled_by_planning_artifact() -> None:
    plan = _plan()
    plan["capacity_firewall"]["long_training_authorized"] = True
    with pytest.raises(CorpusAcquisitionPlanError, match="long training must remain blocked"):
        validate_acquisition_plan(plan)


def test_planning_gap_must_match_target_minus_observed() -> None:
    plan = _plan()
    plan["target"]["planning_gap_from_observed_pre_dedup"]["ua"] += 1
    plan["target"]["planning_gap_from_observed_pre_dedup"]["total"] += 1
    with pytest.raises(CorpusAcquisitionPlanError, match="target minus observed"):
        validate_acquisition_plan(plan)


def test_each_stratum_plan_must_cover_current_gap() -> None:
    plan = _plan()
    for row in plan["candidate_streams"]:
        if row["stratum"] == "code":
            row["planning_request_bytes"] = 1
    with pytest.raises(CorpusAcquisitionPlanError, match="code: planning requests"):
        validate_acquisition_plan(plan)


def test_rights_retests_cannot_be_removed() -> None:
    plan = _plan()
    plan["candidate_streams"][0]["required_retests"] = ["rights"]
    with pytest.raises(CorpusAcquisitionPlanError, match="three explicit admission retests"):
        validate_acquisition_plan(plan)


def test_candidate_status_cannot_be_promoted_to_pass() -> None:
    plan = copy.deepcopy(_plan())
    plan["candidate_streams"][0]["status"] = "PASS"
    with pytest.raises(CorpusAcquisitionPlanError, match="unsupported nonterminal status"):
        validate_acquisition_plan(plan)


def test_stale_source_registry_identity_is_rejected() -> None:
    plan = _plan()
    plan["baseline"]["source_registry_identity_sha256"] = "0" * 64
    with pytest.raises(CorpusAcquisitionPlanError, match="source-registry identity drift"):
        validate_acquisition_plan(plan)


def test_stale_v3_byte_vector_is_rejected_even_if_arithmetic_is_consistent() -> None:
    plan = _plan()
    plan["baseline"]["observed_pre_dedup_bytes"] = {
        "ua": 100_856,
        "en": 150_643,
        "code": 106_031,
        "total": 357_530,
    }
    plan["target"]["planning_gap_from_observed_pre_dedup"] = {
        "ua": 8_899_144,
        "en": 6_849_357,
        "code": 3_893_969,
        "total": 19_642_470,
    }
    with pytest.raises(CorpusAcquisitionPlanError, match="byte vector drift from canonical V4"):
        validate_acquisition_plan(plan)


def test_family_vector_must_match_canonical_v4() -> None:
    plan = _plan()
    plan["baseline"]["observed_independent_families"]["en"] = 4
    plan["baseline"]["observed_independent_families"]["total"] = 13
    with pytest.raises(CorpusAcquisitionPlanError, match="family vector drift from canonical V4"):
        validate_acquisition_plan(plan)


def test_pr_head_movement_is_allowed_when_immutable_registry_identity_is_unchanged() -> None:
    plan = _plan()
    plan["baseline"]["source_convergence_head"] = "1" * 40
    summary = validate_acquisition_plan(plan)
    assert summary["baseline_source_registry_identity_sha256"] == BASELINE_REGISTRY_IDENTITY
    assert summary["training_authorized_bytes"] == 0


def test_malformed_observed_pr_head_is_rejected() -> None:
    plan = _plan()
    plan["baseline"]["source_convergence_head"] = "not-a-sha"
    with pytest.raises(CorpusAcquisitionPlanError, match="40-char SHA"):
        validate_acquisition_plan(plan)
