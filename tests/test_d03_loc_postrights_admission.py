from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.d03_loc_postrights_admission import (
    LocPostRightsAdmissionError,
    load_and_validate,
    validate_admission,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ADMISSION_PATH = REPO_ROOT / "configs/data/d03_common_pile_loc_source_admission_v1.json"


def _payload() -> dict:
    return json.loads(ADMISSION_PATH.read_text(encoding="utf-8"))


def test_committed_admission_executes_exact_repository_and_history_proof() -> None:
    payload = load_and_validate(ADMISSION_PATH, repo_root=REPO_ROOT)
    assert payload["status"] == "PAYLOAD_SOURCE_ADMISSION_EXECUTED_ZERO_CREDIT"
    assert payload["truth_boundary"]["payload_source_admission_executed"] is True
    assert payload["truth_boundary"]["source_admitted_candidate_records"] == 28
    assert payload["truth_boundary"]["source_admitted_candidate_bytes"] == 4499908


def test_source_admission_does_not_grant_training_or_canonical_credit() -> None:
    boundary = _payload()["truth_boundary"]
    assert boundary["canonical_capacity_credited"] == 0
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["family_credit_added"] == 0
    assert boundary["authorized_unique_loss_positions"] == 0
    assert boundary["authorized_optimized_target_exposure"] == 0
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["model_training_executed"] is False
    assert boundary["optimizer_updates"] == 0
    assert boundary["learned_weights_created"] is False
    assert boundary["final_test_payload_accessed"] is False


def test_rejects_bool_alias_for_zero_training_authority() -> None:
    payload = _payload()
    payload["truth_boundary"]["training_authorized_bytes"] = False
    with pytest.raises(LocPostRightsAdmissionError, match="training_authorized_bytes drift"):
        validate_admission(payload)


def test_rejects_bool_alias_for_candidate_source_bytes() -> None:
    payload = _payload()
    payload["admitted_candidate"]["source_bytes"] = True
    with pytest.raises(LocPostRightsAdmissionError, match="source_bytes drift"):
        validate_admission(payload)


def test_rejects_candidate_identity_drift() -> None:
    payload = _payload()
    payload["admitted_candidate"]["candidate_sha256"] = "0" * 64
    with pytest.raises(LocPostRightsAdmissionError, match="candidate_sha256 drift"):
        validate_admission(payload)


def test_rejects_execution_head_substitution() -> None:
    payload = _payload()
    payload["historical_execution"]["execution_head_sha"] = "0" * 40
    with pytest.raises(LocPostRightsAdmissionError, match="execution_head_sha drift"):
        validate_admission(payload)


def test_rejects_record_contract_drift() -> None:
    payload = _payload()
    payload["record_contract"]["expected_license"] = "unknown"
    with pytest.raises(LocPostRightsAdmissionError, match="expected_license drift"):
        validate_admission(payload)


def test_rejects_rights_policy_identity_drift() -> None:
    payload = _payload()
    payload["rights_authority"]["policy_identity_sha256"] = "0" * 64
    with pytest.raises(LocPostRightsAdmissionError, match="policy_identity_sha256 drift"):
        validate_admission(payload)


def test_rejects_attempt_to_claim_payload_rematerialization() -> None:
    payload = _payload()
    payload["admitted_candidate"]["payload_rematerialized_in_this_package"] = True
    with pytest.raises(
        LocPostRightsAdmissionError,
        match="payload_rematerialized_in_this_package drift",
    ):
        validate_admission(payload)


def test_rejects_authority_widening_to_model_training() -> None:
    payload = copy.deepcopy(_payload())
    payload["truth_boundary"]["model_training_executed"] = True
    with pytest.raises(LocPostRightsAdmissionError, match="model_training_executed drift"):
        validate_admission(payload)


def test_rejects_unknown_root_field() -> None:
    payload = _payload()
    payload["training_ready"] = True
    with pytest.raises(LocPostRightsAdmissionError, match="admission schema drift"):
        validate_admission(payload)
