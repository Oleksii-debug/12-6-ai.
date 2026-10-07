from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.learned20m_evaluation_firewall import (
    EVAL233_FINAL_TEST_RESERVATION_AUTHORITY,
    EVAL303_SELECTION_AUTHORITY,
    SELECTION_LOCK_AUTHORITY_ROLE,
    SELECTION_LOCK_AUTHORITY_SCHEMA_VERSION,
    EvaluationFirewallError,
    authorize_final_test_reporting,
    preselection_binding,
    readiness_evaluation_evidence,
    selection_lock_identity,
    self_identity,
    trusted_selection_lock_evidence_identity,
    validate_policy,
)
from twelve_six.learned20m_readiness import scientific_authority_token

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs/evaluation/learned20m_evaluation_firewall_v1.json"


def policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def lock() -> dict:
    item = {
        "selection_complete": True,
        "selection_lock_identity_sha256": "0" * 64,
        "selected_checkpoint_sha256": "2" * 64,
        "selection_validation_evidence_sha256": "3" * 64,
        "recipe_identity_sha256": "4" * 64,
        "train_trace_identity_sha256": "5" * 64,
        "created_from_final_test": False,
        "reselection_allowed": False,
        "optimizer_updates_after_lock_allowed": False,
        "final_test_payload_consumed_before_lock": False,
    }
    item["selection_lock_identity_sha256"] = selection_lock_identity(item)
    return item


def historical_forged_lock() -> dict:
    return {
        "selection_complete": True,
        "selection_lock_identity_sha256": "1" * 64,
        "selected_checkpoint_sha256": "2" * 64,
        "selection_validation_evidence_sha256": "3" * 64,
        "recipe_identity_sha256": "4" * 64,
        "train_trace_identity_sha256": "5" * 64,
        "created_from_final_test": False,
        "reselection_allowed": False,
        "optimizer_updates_after_lock_allowed": False,
        "final_test_payload_consumed_before_lock": False,
    }


def trusted_lock_authority(item: dict | None = None) -> dict:
    item = copy.deepcopy(lock() if item is None else item)
    trusted = {
        "schema_version": SELECTION_LOCK_AUTHORITY_SCHEMA_VERSION,
        "authority": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": "6" * 40,
            "evidence_sha256": "0" * 64,
            "workflow_run_id": 123,
            "workflow_conclusion": "success",
            "terminal": True,
        },
        "selection_lock_identity_sha256": item["selection_lock_identity_sha256"],
        "selected_checkpoint_sha256": item["selected_checkpoint_sha256"],
        "selection_validation_evidence_sha256": item[
            "selection_validation_evidence_sha256"
        ],
        "recipe_identity_sha256": item["recipe_identity_sha256"],
        "train_trace_identity_sha256": item["train_trace_identity_sha256"],
    }
    trusted["authority"]["evidence_sha256"] = trusted_selection_lock_evidence_identity(
        trusted
    )
    return trusted


def verified_lock_tokens(trusted: dict) -> set[str]:
    token = scientific_authority_token(
        SELECTION_LOCK_AUTHORITY_ROLE,
        trusted["authority"],
        require_workflow=True,
    )
    assert token is not None
    return {token}


def authorize(item: dict | None = None, trusted: dict | None = None) -> dict:
    item = lock() if item is None else item
    trusted = trusted_lock_authority(item) if trusted is None else trusted
    return authorize_final_test_reporting(
        policy(),
        item,
        trusted,
        verified_scientific_authorities=verified_lock_tokens(trusted),
    )


def test_checked_in_policy_is_terminal_boundary_but_non_authorizing() -> None:
    result = validate_policy(policy())
    assert result["status"] == "PASS"
    assert result["final_test_access_authorized"] is False
    assert result["training_authorized"] is False
    assert result["compute_authorized"] is False
    assert result["optimizer_updates_executed"] == 0


def test_policy_self_identity_is_deterministic() -> None:
    value = policy()
    assert value["policy_identity_sha256"] == self_identity(value)
    assert self_identity(copy.deepcopy(value)) == self_identity(value)


def test_preselection_binding_keeps_final_test_sealed() -> None:
    receipt = preselection_binding(policy())
    assert receipt["selection_validation_access_authorized"] is True
    assert receipt["selection_lock_bound"] is False
    assert receipt["final_test_access_authorized"] is False
    assert receipt["final_test_outcomes_reporting_authorized"] is False
    assert receipt["optimizer_updates_executed"] == 0
    assert receipt["optimized_target_delta"] == 0


def test_final_test_reporting_opens_only_after_verified_trusted_selection_lock() -> None:
    item = lock()
    trusted = trusted_lock_authority(item)
    receipt = authorize(item, trusted)
    assert receipt["selection_lock_identity_sha256"] == item["selection_lock_identity_sha256"]
    assert receipt["selection_lock_authority"] == trusted["authority"]
    assert receipt["selection_lock_authority_token"] in verified_lock_tokens(trusted)
    assert receipt["final_test_access_authorized"] is True
    assert receipt["final_test_outcomes_reporting_authorized"] is True
    assert receipt["reselection_authorized"] is False
    assert receipt["recipe_mutation_authorized"] is False
    assert receipt["training_data_mutation_authorized"] is False
    assert receipt["optimizer_updates_authorized"] is False


def test_historical_arbitrary_sha_fixture_no_longer_authorizes() -> None:
    item = historical_forged_lock()
    trusted = trusted_lock_authority(item)
    with pytest.raises(EvaluationFirewallError, match="self-identity mismatch"):
        authorize_final_test_reporting(
            policy(),
            item,
            trusted,
            verified_scientific_authorities=verified_lock_tokens(trusted),
        )


def test_canonical_candidate_without_trusted_authority_fails_closed() -> None:
    with pytest.raises(EvaluationFirewallError, match="trusted selection-lock authority required"):
        authorize_final_test_reporting(policy(), lock())


def test_matching_packet_without_out_of_packet_verified_token_fails_closed() -> None:
    item = lock()
    trusted = trusted_lock_authority(item)
    with pytest.raises(EvaluationFirewallError, match="authority unverified"):
        authorize_final_test_reporting(policy(), item, trusted)


def test_coherent_candidate_reseal_cannot_reuse_old_verified_authority() -> None:
    item = lock()
    original_trusted = trusted_lock_authority(item)
    original_tokens = verified_lock_tokens(original_trusted)

    item["selected_checkpoint_sha256"] = "9" * 64
    item["selection_lock_identity_sha256"] = selection_lock_identity(item)
    forged_trusted = trusted_lock_authority(item)

    with pytest.raises(EvaluationFirewallError, match="authority unverified"):
        authorize_final_test_reporting(
            policy(),
            item,
            forged_trusted,
            verified_scientific_authorities=original_tokens,
        )


def test_trusted_semantic_substitution_breaks_evidence_identity() -> None:
    item = lock()
    trusted = trusted_lock_authority(item)
    tokens = verified_lock_tokens(trusted)
    trusted["selected_checkpoint_sha256"] = "9" * 64
    with pytest.raises(EvaluationFirewallError, match="evidence identity mismatch"):
        authorize_final_test_reporting(
            policy(),
            item,
            trusted,
            verified_scientific_authorities=tokens,
        )


@pytest.mark.parametrize(
    "field",
    [
        "selection_lock_identity_sha256",
        "selected_checkpoint_sha256",
        "selection_validation_evidence_sha256",
        "recipe_identity_sha256",
        "train_trace_identity_sha256",
    ],
)
def test_candidate_trusted_identity_divergence_fails_closed(field: str) -> None:
    item = lock()
    trusted = trusted_lock_authority(item)
    tokens = verified_lock_tokens(trusted)
    if field == "selection_lock_identity_sha256":
        trusted[field] = "a" * 64
    else:
        item[field] = "a" * 64
        item["selection_lock_identity_sha256"] = selection_lock_identity(item)
    if field == "selection_lock_identity_sha256":
        trusted["authority"]["evidence_sha256"] = trusted_selection_lock_evidence_identity(
            trusted
        )
        tokens = verified_lock_tokens(trusted)
    with pytest.raises(EvaluationFirewallError, match="trusted selection-lock mismatch"):
        authorize_final_test_reporting(
            policy(),
            item,
            trusted,
            verified_scientific_authorities=tokens,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selection_complete", False),
        ("selection_complete", 1),
        ("created_from_final_test", True),
        ("created_from_final_test", 0),
        ("reselection_allowed", True),
        ("reselection_allowed", 0),
        ("optimizer_updates_after_lock_allowed", True),
        ("final_test_payload_consumed_before_lock", True),
    ],
)
def test_bad_selection_lock_fails_closed(field: str, value: object) -> None:
    item = lock()
    item[field] = value
    with pytest.raises(EvaluationFirewallError):
        authorize_final_test_reporting(policy(), item)


def test_unknown_selection_lock_field_fails_closed() -> None:
    item = lock()
    item["final_test_score"] = 0.1
    with pytest.raises(EvaluationFirewallError, match="keys drift"):
        authorize_final_test_reporting(policy(), item)


def test_unknown_trusted_selection_lock_field_fails_closed() -> None:
    item = lock()
    trusted = trusted_lock_authority(item)
    trusted["note"] = "caller controlled"
    with pytest.raises(EvaluationFirewallError, match="keys drift"):
        authorize_final_test_reporting(
            policy(),
            item,
            trusted,
            verified_scientific_authorities=verified_lock_tokens(
                trusted_lock_authority(item)
            ),
        )


def test_trusted_authority_bool_workflow_id_fails_closed() -> None:
    item = lock()
    trusted = trusted_lock_authority(item)
    trusted["authority"]["workflow_run_id"] = True
    with pytest.raises(EvaluationFirewallError, match="workflow run invalid"):
        authorize_final_test_reporting(policy(), item, trusted)


def test_bool_alias_is_rejected_for_zero_mutation_counter() -> None:
    item = policy()
    item["evaluation_protocol"]["optimizer_updates_during_evaluation"] = False
    item["policy_identity_sha256"] = self_identity(item)
    with pytest.raises(EvaluationFirewallError, match="mutation counter"):
        validate_policy(item)


def test_selection_authority_substitution_fails_closed() -> None:
    item = policy()
    item["selection_validation"]["authority"]["git_sha"] = "a" * 40
    item["policy_identity_sha256"] = self_identity(item)
    with pytest.raises(EvaluationFirewallError, match="exact authority drift"):
        validate_policy(item)


def test_final_test_authority_substitution_fails_closed() -> None:
    item = policy()
    item["final_test_reservation"]["authority"]["evidence_sha256"] = "a" * 64
    item["policy_identity_sha256"] = self_identity(item)
    with pytest.raises(EvaluationFirewallError, match="exact authority drift"):
        validate_policy(item)


def test_nonterminal_authority_fails_closed() -> None:
    item = policy()
    item["selection_validation"]["authority"]["terminal"] = False
    item["policy_identity_sha256"] = self_identity(item)
    with pytest.raises(EvaluationFirewallError, match="not terminal"):
        validate_policy(item)


def test_stale_or_red_workflow_fails_closed() -> None:
    item = policy()
    item["selection_validation"]["authority"]["workflow_conclusion"] = "failure"
    item["policy_identity_sha256"] = self_identity(item)
    with pytest.raises(EvaluationFirewallError, match="workflow not success"):
        validate_policy(item)


def test_recipe_mutation_from_selection_is_forbidden() -> None:
    item = policy()
    item["selection_validation"]["may_mutate_recipe"] = True
    item["policy_identity_sha256"] = self_identity(item)
    with pytest.raises(EvaluationFirewallError, match="selection boundary weakened"):
        validate_policy(item)


def test_final_test_cannot_trigger_reselection() -> None:
    item = policy()
    item["final_test_reservation"]["may_trigger_reselection"] = True
    item["policy_identity_sha256"] = self_identity(item)
    with pytest.raises(EvaluationFirewallError, match="final-test boundary weakened"):
        validate_policy(item)


def test_cadence_drift_fails_closed() -> None:
    item = policy()
    item["evaluation_protocol"]["checkpoint_boundaries_percent"] = [0, 50, 100]
    item["policy_identity_sha256"] = self_identity(item)
    with pytest.raises(EvaluationFirewallError, match="cadence drift"):
        validate_policy(item)


def test_policy_unknown_field_fails_closed() -> None:
    item = policy()
    item["helpful_note"] = "ignored?"
    with pytest.raises(EvaluationFirewallError, match="keys drift"):
        validate_policy(item)


def test_policy_identity_tamper_fails_closed() -> None:
    item = policy()
    item["policy_identity_sha256"] = "0" * 64
    with pytest.raises(EvaluationFirewallError, match="self-identity mismatch"):
        validate_policy(item)


def test_readiness_shape_uses_terminal_external_authorities() -> None:
    firewall = {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": "6" * 40,
        "evidence_sha256": policy()["policy_identity_sha256"],
        "workflow_run_id": 123,
        "workflow_conclusion": "success",
        "terminal": True,
    }
    evidence = readiness_evaluation_evidence(policy(), firewall)
    assert evidence["status"] == "PASS"
    assert evidence["firewall_authority"] == firewall
    assert evidence["selection_validation_authority"] == EVAL303_SELECTION_AUTHORITY


def test_readiness_shape_rejects_bool_workflow_id() -> None:
    firewall = copy.deepcopy(EVAL233_FINAL_TEST_RESERVATION_AUTHORITY)
    firewall["workflow_run_id"] = True
    with pytest.raises(EvaluationFirewallError, match="workflow run invalid"):
        readiness_evaluation_evidence(policy(), firewall)


def test_authority_constants_are_not_alias_mutated_by_receipts() -> None:
    receipt = preselection_binding(policy())
    receipt["selection_authority"]["git_sha"] = "f" * 40
    assert EVAL303_SELECTION_AUTHORITY["git_sha"] != "f" * 40
