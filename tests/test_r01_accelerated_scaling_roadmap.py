from __future__ import annotations

import copy
import json
from pathlib import Path

from twelve_six.accelerated_scaling import assess_roadmap, validate_roadmap

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/r01_accelerated_scaling_roadmap_v2.json"
SHA40 = "a" * 40
SHA64 = "b" * 64


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _authority() -> dict:
    return {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": SHA40,
        "evidence_sha256": SHA64,
        "workflow_run_id": 123,
        "workflow_conclusion": "success",
        "terminal": True,
    }


def _pass_learned(data: dict, key: str) -> None:
    requirements_key = (
        "terminal_20m_evidence_requirements"
        if key == "learned_20m"
        else "terminal_200m_evidence_requirements"
    )
    data["evidence_state"][key] = {
        "status": "PASS",
        "evidence_manifest_sha256": SHA64,
        "requirements_satisfied": list(data[requirements_key]),
        "terminal_authority": _authority(),
        "independent_audit_authority": _authority(),
    }


def _pass_feasibility(data: dict, key: str, decision: str = "GO") -> None:
    requirements_key = (
        "feasibility_200m_requirements"
        if key == "feasibility_200m"
        else "feasibility_1b_requirements"
    )
    data["evidence_state"][key] = {
        "status": "PASS",
        "packet_sha256": SHA64,
        "requirements_satisfied": list(data[requirements_key]),
        "terminal_authority": _authority(),
        "decision": decision,
    }


def test_current_roadmap_is_valid_but_blocks_at_terminal_20m() -> None:
    data = _load()
    assert validate_roadmap(data) == []
    result = assess_roadmap(data)
    assert result.contract_valid
    assert result.next_action == "FINISH_TERMINAL_LEARNED_20M_CRITICAL_PATH"
    assert not result.terminal_20m_proven
    assert not result.ready_for_200m_feasibility
    assert not result.ready_to_request_200m_authorization
    assert "learned_20m_not_terminal_pass" in result.blockers


def test_previous_100m_contract_is_superseded_only_for_post_20m_routing() -> None:
    previous = _load()["authority"]["previous_roadmap"]
    assert previous["disposition"] == "SUPERSEDED_FOR_POST_20M_ROUTING_ONLY"
    assert previous["scientific_gates_preserved"] is True
    assert len(previous["sha256"]) == 64


def test_strategy_and_previous_contract_authorities_are_exact() -> None:
    for section in ("source_strategy", "previous_roadmap"):
        data = _load()
        data["authority"][section]["sha256"] = "0" * 64
        errors = validate_roadmap(data)
        assert any(section in error or "roadmap_sha256" in error for error in errors)


def test_terminal_20m_cannot_be_made_optional_or_authorized() -> None:
    for field, value in (("mandatory", False), ("authorized_now", True)):
        data = _load()
        stage = next(item for item in data["scale_route"] if item["id"] == "TERMINAL_LEARNED_20M")
        stage[field] = value
        errors = validate_roadmap(data)
        assert any("learned_20m" in error for error in errors)


def test_50m_100m_probes_cannot_become_mandatory_full_campaigns() -> None:
    data = _load()
    stage = next(item for item in data["scale_route"] if item["id"] == "OPTIONAL_50M_100M_PROBES")
    stage["mandatory"] = True
    stage["full_campaign_required"] = True
    errors = validate_roadmap(data)
    assert "50m_100m_probes_must_remain_optional" in errors
    assert "50m_100m_full_campaign_must_not_be_required" in errors


def test_200m_cannot_skip_terminal_20m_or_freeze_modelspec() -> None:
    data = _load()
    stage = next(item for item in data["scale_route"] if item["id"] == "PRODUCT_200M")
    stage["requires_terminal_stage"] = "LAB_3M_10M"
    stage["modelspec_frozen"] = True
    errors = validate_roadmap(data)
    assert "product_200m_must_require_terminal_20m" in errors
    assert "product_200m_modelspec_cannot_be_frozen" in errors


def test_1b_cannot_skip_terminal_200m() -> None:
    data = _load()
    stage = next(item for item in data["scale_route"] if item["id"] == "PRODUCT_1B")
    stage["requires_terminal_stage"] = "TERMINAL_LEARNED_20M"
    assert "product_1b_must_require_terminal_200m" in validate_roadmap(data)


def test_paid_compute_and_foreign_weights_cannot_be_silently_authorized() -> None:
    for field in ("paid_compute_authorized", "foreign_pretrained_or_aligned_weights_allowed"):
        data = _load()
        data["hard_boundaries"][field] = True
        assert any(field in error for error in validate_roadmap(data))


def test_required_run_packet_identity_cannot_be_removed() -> None:
    data = _load()
    data["portable_training_runner"]["required_run_packet_fields"].remove(
        "unique_loss_ledger_sha256"
    )
    assert "required_run_packet_fields_incomplete" in validate_roadmap(data)


def test_backend_cannot_be_silently_selected_or_promoted() -> None:
    data = _load()
    portable = data["portable_training_runner"]
    portable["selected_backend"] = "LITGPT"
    portable["backend_candidates"][1]["canonical_now"] = True
    errors = validate_roadmap(data)
    assert "backend_cannot_be_selected_without_evidence" in errors
    assert "backend_LITGPT_cannot_be_canonical_now" in errors


def test_terminal_20m_opens_feasibility_not_200m_training() -> None:
    data = _load()
    _pass_learned(data, "learned_20m")
    result = assess_roadmap(data)
    assert result.contract_valid
    assert result.terminal_20m_proven
    assert result.ready_for_200m_feasibility
    assert result.next_action == "PREPARE_200M_FEASIBILITY_PACKET"
    assert not result.ready_to_request_200m_authorization


def test_go_200m_feasibility_only_opens_explicit_authorization_request() -> None:
    data = _load()
    _pass_learned(data, "learned_20m")
    _pass_feasibility(data, "feasibility_200m")
    result = assess_roadmap(data)
    assert result.contract_valid
    assert result.next_action == "REQUEST_EXPLICIT_200M_COMPUTE_AND_TRAINING_AUTHORIZATION"
    assert result.ready_to_request_200m_authorization
    assert not result.terminal_200m_proven
    assert data["hard_boundaries"]["material_training_authorized"] is False


def test_non_go_200m_feasibility_holds_scale_up() -> None:
    data = _load()
    _pass_learned(data, "learned_20m")
    _pass_feasibility(data, "feasibility_200m", decision="HOLD")
    result = assess_roadmap(data)
    assert result.contract_valid
    assert result.next_action == "HOLD_200M_AND_REASSESS_EVIDENCE"
    assert not result.ready_to_request_200m_authorization


def test_terminal_200m_opens_1b_feasibility_only() -> None:
    data = _load()
    _pass_learned(data, "learned_20m")
    _pass_feasibility(data, "feasibility_200m")
    _pass_learned(data, "learned_200m")
    result = assess_roadmap(data)
    assert result.contract_valid
    assert result.terminal_200m_proven
    assert result.ready_for_1b_feasibility
    assert result.next_action == "PREPARE_1B_FEASIBILITY_PACKET"
    assert not result.ready_to_request_1b_authorization


def test_later_evidence_cannot_precede_required_terminal_stage() -> None:
    data = _load()
    _pass_feasibility(data, "feasibility_200m")
    errors = validate_roadmap(data)
    assert "200m_feasibility_cannot_precede_terminal_20m" in errors


def test_invalid_contract_fails_closed() -> None:
    data = copy.deepcopy(_load())
    data["scale_route"].reverse()
    result = assess_roadmap(data)
    assert not result.contract_valid
    assert result.next_action == "REPAIR_INVALID_ROADMAP_CONTRACT"
    assert not result.ready_for_200m_feasibility
