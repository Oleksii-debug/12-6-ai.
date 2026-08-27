from __future__ import annotations

import copy
import json
from pathlib import Path

from twelve_six.learned20m_readiness import assess_learned20m_readiness

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/r01_learned20m_launch_readiness_v1.json"
SHA40 = "a" * 40
SHA64 = "b" * 64
COMPUTE_REF = "issue:1#compute-authorized-example"
TRAINING_REF = "issue:1#training-authorized-example"


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
            "requested_total_training_exposures": 412_268_800,
            "max_exposures_per_unique_position": 1,
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
    for label in ("learned_3m", "learned_10m"):
        evidence["learned_scale_evidence"][label].update(
            {"authority": _authority(), "status": "PASS"}
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


def _add_material_authorizations(
    data: dict,
    *,
    compute_ref: str = COMPUTE_REF,
    training_ref: str = TRAINING_REF,
    maximum_cost_usd: float = 50.0,
) -> None:
    evidence = data["evidence"]
    evidence["compute_authorization"].update(
        {
            "authority": _authority(workflow=False),
            "status": "COMPUTE_AUTHORIZED",
            "decision_ref": compute_ref,
            "maximum_cost_usd": maximum_cost_usd,
        }
    )
    evidence["training_authorization"].update(
        {
            "authority": _authority(workflow=False),
            "status": "TRAINING_AUTHORIZED",
            "decision_ref": training_ref,
        }
    )


def test_current_packet_is_blocked_at_all_three_phases() -> None:
    result = assess_learned20m_readiness(_load())
    assert not result.ready_for_local_free_pilot
    assert not result.ready_for_compute_authorization_request
    assert not result.material_training_authorized
    assert "data_budget_not_qualified" in result.local_free_pilot_blockers
    assert "checkpoint_integrity_not_terminal_pass" in result.local_free_pilot_blockers
    assert "requested_unique_loss_positions_not_positive" in result.local_free_pilot_blockers
    assert "learned_3m_not_terminal_pass" not in result.compute_request_blockers
    assert "learned_10m_not_terminal_pass" not in result.compute_request_blockers
    assert "bounded_pilot_not_terminal_pass" in result.compute_request_blockers
    assert "cost_envelope_not_estimated" in result.compute_request_blockers
    assert "independent_audit_not_terminal_pass" in result.compute_request_blockers
    assert "compute_not_explicitly_authorized" in result.material_training_blockers
    assert "training_not_explicitly_authorized" in result.material_training_blockers


def test_local_pilot_ready_does_not_imply_compute_or_training_authority() -> None:
    result = assess_learned20m_readiness(_make_local_pilot_ready())
    assert result.ready_for_local_free_pilot
    assert not result.ready_for_compute_authorization_request
    assert not result.material_training_authorized
    assert "bounded_pilot_not_terminal_pass" in result.compute_request_blockers
    assert "learned_3m_not_terminal_pass" not in result.compute_request_blockers
    assert "learned_10m_not_terminal_pass" not in result.compute_request_blockers


def test_learned_scale_evidence_is_required_before_compute_request() -> None:
    data = _make_compute_request_ready()
    data["evidence"]["learned_scale_evidence"]["learned_10m"] = {
        "authority": None,
        "status": "NOT_RUN",
    }
    result = assess_learned20m_readiness(data)
    assert result.ready_for_local_free_pilot
    assert not result.ready_for_compute_authorization_request
    assert "learned_10m_authority_missing" in result.compute_request_blockers
    assert "learned_10m_not_terminal_pass" in result.compute_request_blockers


def test_requested_unique_loss_budget_must_fit_terminal_ledger() -> None:
    data = _make_local_pilot_ready()
    ledger_positions = data["evidence"]["loss_ledger"]["unique_causal_loss_positions"]
    data["evidence"]["training_recipe"]["requested_unique_loss_positions"] = ledger_positions + 1
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
        assert "requested_unique_loss_positions_not_positive" in result.local_free_pilot_blockers


def test_total_training_exposure_cannot_be_below_unique_requirement() -> None:
    data = _make_local_pilot_ready()
    recipe = data["evidence"]["training_recipe"]
    recipe["requested_total_training_exposures"] = recipe["requested_unique_loss_positions"] - 1
    result = assess_learned20m_readiness(data)
    assert not result.ready_for_local_free_pilot
    assert "total_training_exposures_below_unique_requirement" in result.local_free_pilot_blockers
    recipe["requested_total_training_exposures"] = recipe["requested_unique_loss_positions"]
    result = assess_learned20m_readiness(data)
    assert result.ready_for_local_free_pilot


def test_total_training_exposure_must_respect_replay_cap() -> None:
    data = _make_local_pilot_ready()
    recipe = data["evidence"]["training_recipe"]
    recipe["requested_unique_loss_positions"] = 100
    recipe["max_exposures_per_unique_position"] = 2
    recipe["requested_total_training_exposures"] = 201
    result = assess_learned20m_readiness(data)
    assert not result.ready_for_local_free_pilot
    assert "total_training_exposures_exceed_replay_cap" in result.local_free_pilot_blockers
    recipe["requested_total_training_exposures"] = 200
    result = assess_learned20m_readiness(data)
    assert result.ready_for_local_free_pilot


def test_exposure_controls_reject_malformed_values() -> None:
    for field in ("requested_total_training_exposures", "max_exposures_per_unique_position"):
        for bad in (0, -1, True, 1.5, "100"):
            data = _make_local_pilot_ready()
            data["evidence"]["training_recipe"][field] = bad
            result = assess_learned20m_readiness(data)
            assert not result.ready_for_local_free_pilot
            assert f"{field}_not_positive" in result.local_free_pilot_blockers


def test_compute_request_ready_does_not_imply_paid_training_authority() -> None:
    result = assess_learned20m_readiness(_make_compute_request_ready())
    assert result.ready_for_local_free_pilot
    assert result.ready_for_compute_authorization_request
    assert not result.material_training_authorized
    assert "compute_not_explicitly_authorized" in result.material_training_blockers
    assert "training_not_explicitly_authorized" in result.material_training_blockers


def test_packet_authored_authorizations_cannot_self_authorize() -> None:
    data = _make_compute_request_ready()
    _add_material_authorizations(data)
    result = assess_learned20m_readiness(data)
    assert not result.material_training_authorized
    assert "compute_authorization_ref_unverified" in result.material_training_blockers
    assert "training_authorization_ref_unverified" in result.material_training_blockers


def test_compute_and_training_authorization_refs_must_be_distinct() -> None:
    data = _make_compute_request_ready()
    _add_material_authorizations(data, compute_ref=COMPUTE_REF, training_ref=COMPUTE_REF)
    result = assess_learned20m_readiness(data, verified_authorization_refs={COMPUTE_REF})
    assert not result.material_training_authorized
    assert "compute_and_training_authorization_refs_must_be_distinct" in result.material_training_blockers


def test_explicit_material_authority_must_cover_estimated_maximum() -> None:
    data = _make_compute_request_ready()
    _add_material_authorizations(data, maximum_cost_usd=49.0)
    verified = {COMPUTE_REF, TRAINING_REF}
    result = assess_learned20m_readiness(data, verified_authorization_refs=verified)
    assert not result.material_training_authorized
    assert "authorized_cost_below_estimated_maximum" in result.material_training_blockers
    data["evidence"]["compute_authorization"]["maximum_cost_usd"] = 50.0
    result = assess_learned20m_readiness(data, verified_authorization_refs=verified)
    assert result.material_training_authorized


def test_training_authorization_is_independent_of_compute_authorization() -> None:
    data = _make_compute_request_ready()
    _add_material_authorizations(data)
    data["evidence"]["training_authorization"].update(
        {"authority": None, "status": "NOT_AUTHORIZED", "decision_ref": None}
    )
    result = assess_learned20m_readiness(data, verified_authorization_refs={COMPUTE_REF})
    assert not result.material_training_authorized
    assert "training_not_explicitly_authorized" in result.material_training_blockers
    assert "training_authorization_ref_missing" in result.material_training_blockers


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


def test_unit_truth_boundary_cannot_be_weakened() -> None:
    for key in (
        "source_bytes_are_not_loss_positions",
        "training_exposure_is_not_unique_data",
        "replay_cannot_inflate_unique_loss_ledger",
    ):
        data = _make_local_pilot_ready()
        data["truth_boundary"][key] = False
        result = assess_learned20m_readiness(data)
        assert not result.ready_for_local_free_pilot
        assert f"truth_boundary_{key}_must_be_true" in result.local_free_pilot_blockers
