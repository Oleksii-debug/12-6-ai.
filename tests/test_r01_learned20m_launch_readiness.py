from __future__ import annotations

import copy
import json
from pathlib import Path

from twelve_six.learned20m_readiness import assess_learned20m_readiness

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/r01_learned20m_launch_readiness_v1.json"
SHA40 = "a" * 40
SHA64 = "b" * 64


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _authority(*, workflow: bool = True) -> dict:
    authority = {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": SHA40,
        "evidence_sha256": SHA64,
        "terminal": True,
    }
    if workflow:
        authority.update({"workflow_run_id": 123, "workflow_conclusion": "success"})
    return authority


def _make_local_pilot_ready() -> dict:
    data = _load()
    evidence = data["evidence"]
    evidence["code"]["git_sha"] = SHA40
    evidence["corpus"].update(
        {
            "manifest_sha256": SHA64,
            "split_sha256": SHA64,
            "packing_sha256": SHA64,
            "two_clean_builds_identical": True,
            "authority": _authority(),
        }
    )
    evidence["tokenizer"].update(
        {
            "identity_sha256": SHA64,
            "decision": "BYTE_BASELINE_RETAINED",
            "authority": _authority(),
        }
    )
    evidence["loss_ledger"].update(
        {
            "identity_sha256": SHA64,
            "unique_causal_loss_positions": 412_268_800,
            "authority": _authority(),
            "data_budget_authority": _authority(),
            "data_budget_status": "QUALIFIED",
        }
    )
    evidence["checkpoint_integrity"].update(
        {"authority": _authority(), "status": "PASS"}
    )
    evidence["evaluation"].update(
        {
            "firewall_authority": _authority(),
            "selection_validation_authority": _authority(),
            "status": "PASS",
        }
    )
    evidence["training_recipe"].update(
        {
            "authority": _authority(),
            "status": "QUALIFIED",
            "seed_count": 2,
            "config_sha256": SHA64,
            "stopping_policy_sha256": SHA64,
            "requested_unique_loss_positions": 412_268_800,
        }
    )
    return data


def _make_compute_request_ready() -> dict:
    data = _make_local_pilot_ready()
    evidence = data["evidence"]
    evidence["bounded_pilot"].update(
        {
            "authority": _authority(),
            "status": "PASS",
            "numerics_finite": True,
            "resume_equivalent": True,
            "loss_trajectory_acceptable": True,
        }
    )
    evidence["cost_envelope"].update(
        {
            "authority": _authority(workflow=False),
            "status": "ESTIMATED",
            "maximum_cost_usd": 50.0,
        }
    )
    evidence["independent_audit"].update(
        {"authority": _authority(), "status": "PASS"}
    )
    return data


def test_current_packet_is_blocked_at_all_three_phases() -> None:
    result = assess_learned20m_readiness(_load())
    assert not result.ready_for_local_free_pilot
    assert not result.ready_for_compute_authorization_request
    assert not result.material_training_authorized
    assert "data_budget_not_qualified" in result.local_free_pilot_blockers
    assert "checkpoint_integrity_not_terminal_pass" in result.local_free_pilot_blockers
    assert "requested_unique_loss_positions_not_positive" in result.local_free_pilot_blockers
    assert "compute_not_explicitly_authorized" in result.material_training_blockers


def test_local_pilot_ready_does_not_imply_compute_or_training_authority() -> None:
    result = assess_learned20m_readiness(_make_local_pilot_ready())
    assert result.ready_for_local_free_pilot
    assert not result.ready_for_compute_authorization_request
    assert not result.material_training_authorized
    assert "bounded_pilot_not_terminal_pass" in result.compute_request_blockers


def test_requested_unique_loss_budget_must_fit_terminal_ledger() -> None:
    data = _make_local_pilot_ready()
    ledger_positions = data["evidence"]["loss_ledger"]["unique_causal_loss_positions"]

    data["evidence"]["training_recipe"]["requested_unique_loss_positions"] = (
        ledger_positions + 1
    )
    result = assess_learned20m_readiness(data)
    assert not result.ready_for_local_free_pilot
    assert "requested_unique_loss_positions_exceed_ledger" in result.local_free_pilot_blockers

    data["evidence"]["training_recipe"]["requested_unique_loss_positions"] = ledger_positions
    result = assess_learned20m_readiness(data)
    assert result.ready_for_local_free_pilot


def test_requested_unique_loss_budget_rejects_malformed_values() -> None:
    for bad in (0, -1, True, 1.5, "100"):
        data = _make_local_pilot_ready()
        data["evidence"]["training_recipe"]["requested_unique_loss_positions"] = bad
        result = assess_learned20m_readiness(data)
        assert not result.ready_for_local_free_pilot
        assert (
            "requested_unique_loss_positions_not_positive"
            in result.local_free_pilot_blockers
        )


def test_compute_request_ready_does_not_imply_paid_training_authority() -> None:
    result = assess_learned20m_readiness(_make_compute_request_ready())
    assert result.ready_for_local_free_pilot
    assert result.ready_for_compute_authorization_request
    assert not result.material_training_authorized
    assert "compute_not_explicitly_authorized" in result.material_training_blockers


def test_explicit_compute_authority_must_cover_estimated_maximum() -> None:
    data = _make_compute_request_ready()
    data["evidence"]["compute_authorization"].update(
        {
            "authority": _authority(workflow=False),
            "status": "COMPUTE_AUTHORIZED",
            "maximum_cost_usd": 49.0,
        }
    )
    result = assess_learned20m_readiness(data)
    assert not result.material_training_authorized
    assert "authorized_cost_below_estimated_maximum" in result.material_training_blockers

    data["evidence"]["compute_authorization"]["maximum_cost_usd"] = 50.0
    result = assess_learned20m_readiness(data)
    assert result.material_training_authorized


def test_nonterminal_or_failed_workflow_reference_fails_closed() -> None:
    data = _make_local_pilot_ready()
    bad = copy.deepcopy(data["evidence"]["checkpoint_integrity"]["authority"])
    bad["workflow_conclusion"] = "failure"
    data["evidence"]["checkpoint_integrity"]["authority"] = bad
    result = assess_learned20m_readiness(data)
    assert not result.ready_for_local_free_pilot
    assert "checkpoint_integrity_authority_missing" in result.local_free_pilot_blockers


def test_model_authority_drift_blocks_every_phase() -> None:
    data = _make_compute_request_ready()
    data["model_authority"]["parameter_count"] += 1
    result = assess_learned20m_readiness(data)
    assert not result.ready_for_local_free_pilot
    assert "model_authority_parameter_count_mismatch" in result.local_free_pilot_blockers


def test_truth_boundary_cannot_be_promoted_by_packet_mutation() -> None:
    data = _make_compute_request_ready()
    data["truth_boundary"]["paid_compute_executed_by_this_package"] = True
    result = assess_learned20m_readiness(data)
    assert not result.ready_for_local_free_pilot
    assert (
        "truth_boundary_paid_compute_executed_by_this_package_must_be_false"
        in result.local_free_pilot_blockers
    )
