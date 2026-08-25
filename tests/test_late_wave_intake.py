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


def _item(document: dict[str, object], pr: int) -> dict[str, object]:
    return next(item for item in document["items"] if item["pr"] == pr)


def test_current_registry_is_state_aware_and_composition_ready() -> None:
    facts = validate_late_wave_snapshot(_document())
    assert facts["base_sha"] == "c631c024e641dac102036fafee6d78ba31c067cd"
    assert facts["minimum_required"] == (90, 91, 100)
    assert facts["composition_ready"] is True
    assert facts["composable_prs"] == (90, 91, 100)
    assert facts["green_prs"] == (90, 91, 92, 100)
    assert facts["red_prs"] == (95, 113)
    assert facts["held_prs"] == (95, 106, 113)
    assert facts["promotion_eligible"] is False


def test_registry_records_incumbents_and_closed_duplicate_pruning() -> None:
    facts = validate_late_wave_snapshot(_document())
    assert facts["collision_group_owners"] == {
        "repeatability-intake": 90,
        "first-party-checkpoint-snapshot": 91,
        "hf-export-transactionality": 95,
        "parity-oracle-hardening": 134,
    }
    assert facts["duplicate_closed_unmerged"] == (94, 97, 98, 102, 105, 111, 136)
    assert facts["superseded_closed_unmerged"] == ()


def test_registry_checkout_descends_from_exact_green_pr89() -> None:
    facts = verify_base_ancestry(_document(), ROOT)
    assert facts["base_sha"] == "c631c024e641dac102036fafee6d78ba31c067cd"
    assert facts["head_sha"] != facts["base_sha"]


def test_registry_rejects_old_schema() -> None:
    document = copy.deepcopy(_document())
    document["schema"] = "12-6.s0-late-wave-intake.v1"
    with pytest.raises(LateWaveIntakeError, match="unsupported schema"):
        validate_late_wave_snapshot(document)


def test_registry_rejects_nonterminal_or_incomplete_base() -> None:
    document = copy.deepcopy(_document())
    document["base"]["required_workflows"]["CI"] = {
        "run_id": 32778688850,
        "state": "queued",
        "conclusion": None,
    }
    with pytest.raises(LateWaveIntakeError, match="incomplete or non-green"):
        validate_late_wave_snapshot(document)


def test_registry_rejects_false_green_or_composable_claim() -> None:
    document = copy.deepcopy(_document())
    pr90 = _item(document, 90)
    pr90["workflow_evidence"]["CI"]["conclusion"] = "failure"
    with pytest.raises(LateWaveIntakeError, match="green/composable claim"):
        validate_late_wave_snapshot(document)


def test_registry_rejects_red_claim_without_terminal_failure() -> None:
    document = copy.deepcopy(_document())
    pr95 = _item(document, 95)
    for workflow in pr95["workflow_evidence"].values():
        workflow["conclusion"] = "success"
    with pytest.raises(LateWaveIntakeError, match="RED claim lacks terminal failure"):
        validate_late_wave_snapshot(document)


def test_registry_rejects_queued_claim_without_pending_workflow() -> None:
    document = copy.deepcopy(_document())
    pr95 = _item(document, 95)
    pr95["classifications"] = ["INCUMBENT", "QUEUED", "HOLD"]
    with pytest.raises(LateWaveIntakeError, match="QUEUED claim lacks"):
        validate_late_wave_snapshot(document)


def test_registry_rejects_composable_blocked_head() -> None:
    document = copy.deepcopy(_document())
    pr95 = _item(document, 95)
    pr95["classifications"] = ["INCUMBENT", "RED", "HOLD", "COMPOSABLE"]
    with pytest.raises(LateWaveIntakeError, match="COMPOSABLE conflicts"):
        validate_late_wave_snapshot(document)


def test_registry_requires_every_minimum_pr_to_be_composable() -> None:
    document = copy.deepcopy(_document())
    pr100 = _item(document, 100)
    pr100["classifications"] = ["GREEN"]
    with pytest.raises(LateWaveIntakeError, match="required PR #100 must be COMPOSABLE"):
        validate_late_wave_snapshot(document)


def test_registry_requires_collision_owner_to_be_incumbent() -> None:
    document = copy.deepcopy(_document())
    pr95 = _item(document, 95)
    pr95["classifications"] = ["RED", "HOLD"]
    with pytest.raises(LateWaveIntakeError, match="must be classified INCUMBENT"):
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


def test_registry_keeps_s1_out_of_s0() -> None:
    document = copy.deepcopy(_document())
    pr106 = _item(document, 106)
    pr106["classifications"] = ["GREEN"]
    pr106["workflow_evidence"] = {
        "CI": {"run_id": 1, "state": "completed", "conclusion": "success"}
    }
    with pytest.raises(LateWaveIntakeError, match="S1 work must remain HOLD"):
        validate_late_wave_snapshot(document)


def test_registry_keeps_d10_governance_separate() -> None:
    document = copy.deepcopy(_document())
    pr113 = _item(document, 113)
    pr113["classifications"] = ["RED"]
    with pytest.raises(LateWaveIntakeError, match="D10 governance work must remain HOLD"):
        validate_late_wave_snapshot(document)


def test_registry_preserves_independent_audit_and_truth_boundaries() -> None:
    document = copy.deepcopy(_document())
    document["truth_boundary"]["audits"]["AUDIT-A"] = "PASS"
    with pytest.raises(LateWaveIntakeError, match="audit authority"):
        validate_late_wave_snapshot(document)

    document = copy.deepcopy(_document())
    document["truth_boundary"]["promotion_eligible"] = True
    with pytest.raises(LateWaveIntakeError, match="must remain false"):
        validate_late_wave_snapshot(document)

    document = copy.deepcopy(_document())
    document["truth_boundary"]["paid_compute_authorized"] = True
    with pytest.raises(LateWaveIntakeError, match="must remain false"):
        validate_late_wave_snapshot(document)


def test_registry_requires_postcomposition_rerun_and_both_audits() -> None:
    document = copy.deepcopy(_document())
    document["next_composition_policy"]["rerun_full_exact_head_workflows_after_composition"] = False
    with pytest.raises(LateWaveIntakeError, match="rerun is mandatory"):
        validate_late_wave_snapshot(document)

    document = copy.deepcopy(_document())
    policy = document["next_composition_policy"]
    policy["request_both_independent_audits_on_final_exact_head"] = False
    with pytest.raises(LateWaveIntakeError, match="audit handoffs are mandatory"):
        validate_late_wave_snapshot(document)
