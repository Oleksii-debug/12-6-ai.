from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.integration.late_wave_intake import (
    LateWaveIntakeError,
    validate_late_wave_snapshot,
    verify_base_ancestry,
)

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "evidence" / "swarm_exp_01" / "d01_late_wave_intake_snapshot_20260825.json"


def _document() -> dict[str, object]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_current_registry_is_fail_closed_and_bound_to_pr89() -> None:
    facts = validate_late_wave_snapshot(_document())
    assert facts["base_sha"] == "c631c024e641dac102036fafee6d78ba31c067cd"
    assert facts["minimum_required"] == (90, 91, 100)
    assert facts["collision_group_owners"]["repeatability-intake"] == 90
    assert facts["collision_group_owners"]["first-party-checkpoint-snapshot"] == 91
    assert facts["excluded_from_s0"] == (106,)
    assert facts["governance_held"] == (113,)
    assert facts["promotion_eligible"] is False


def test_registry_checkout_descends_from_exact_green_pr89() -> None:
    facts = verify_base_ancestry(_document(), ROOT)
    assert facts["base_sha"] == "c631c024e641dac102036fafee6d78ba31c067cd"
    assert facts["head_sha"] != facts["base_sha"]


def test_registry_rejects_nonterminal_base_workflow() -> None:
    document = copy.deepcopy(_document())
    document["base"]["required_workflows"]["CI"]["state"] = "queued"
    with pytest.raises(LateWaveIntakeError, match="not terminal success"):
        validate_late_wave_snapshot(document)


def test_registry_rejects_promotion_or_paid_compute_claim() -> None:
    document = copy.deepcopy(_document())
    document["truth_boundary"]["promotion_eligible"] = True
    with pytest.raises(LateWaveIntakeError, match="must remain false"):
        validate_late_wave_snapshot(document)
    document = copy.deepcopy(_document())
    document["truth_boundary"]["paid_compute_authorized"] = True
    with pytest.raises(LateWaveIntakeError, match="must remain false"):
        validate_late_wave_snapshot(document)


def test_registry_preserves_independent_audit_authority() -> None:
    document = copy.deepcopy(_document())
    document["truth_boundary"]["audits"]["AUDIT-A"] = "PASS"
    with pytest.raises(LateWaveIntakeError, match="audit authority"):
        validate_late_wave_snapshot(document)


def test_registry_rejects_collision_owner_outside_group() -> None:
    document = copy.deepcopy(_document())
    document["collision_groups"][0]["owner_pr"] = 999
    with pytest.raises(LateWaveIntakeError, match="owner must be a member"):
        validate_late_wave_snapshot(document)


def test_registry_rejects_one_pr_in_two_collision_groups() -> None:
    document = copy.deepcopy(_document())
    document["collision_groups"][2]["members"].append(91)
    with pytest.raises(LateWaveIntakeError, match="multiple collision groups"):
        validate_late_wave_snapshot(document)


def test_registry_rejects_s1_surface_entering_s0() -> None:
    document = copy.deepcopy(_document())
    s1 = next(item for item in document["items"] if item["pr"] == 106)
    s1["disposition"] = "PENDING_EXACT_HEAD_CI"
    with pytest.raises(LateWaveIntakeError, match="S1 numerical preflight"):
        validate_late_wave_snapshot(document)


def test_registry_rejects_d10_governance_takeover() -> None:
    document = copy.deepcopy(_document())
    d10 = next(item for item in document["items"] if item["pr"] == 113)
    d10["disposition"] = "PENDING_EXACT_HEAD_CI"
    with pytest.raises(LateWaveIntakeError, match="D10 governance work"):
        validate_late_wave_snapshot(document)


def test_registry_requires_exact_minimum_composition_set() -> None:
    document = copy.deepcopy(_document())
    document["next_composition_policy"]["minimum_required"] = [90, 91]
    with pytest.raises(LateWaveIntakeError, match="minimum late-wave composition set"):
        validate_late_wave_snapshot(document)


def test_registry_cannot_mark_recorded_all_green_required_head_as_pending() -> None:
    document = copy.deepcopy(_document())
    pr90 = next(item for item in document["items"] if item["pr"] == 90)
    pr90["workflow_state"] = {"CI": "success", "D02 Real S0 Training": "success", "D02 S0 Determinism Repeatability": "success"}
    with pytest.raises(LateWaveIntakeError, match="refresh snapshot"):
        validate_late_wave_snapshot(document)


def test_registry_requires_postcomposition_exact_head_rerun_and_both_audits() -> None:
    document = copy.deepcopy(_document())
    document["next_composition_policy"]["rerun_full_exact_head_workflows_after_composition"] = False
    with pytest.raises(LateWaveIntakeError, match="rerun is mandatory"):
        validate_late_wave_snapshot(document)
    document = copy.deepcopy(_document())
    document["next_composition_policy"]["request_both_independent_audits_on_final_exact_head"] = False
    with pytest.raises(LateWaveIntakeError, match="audit handoffs are mandatory"):
        validate_late_wave_snapshot(document)
