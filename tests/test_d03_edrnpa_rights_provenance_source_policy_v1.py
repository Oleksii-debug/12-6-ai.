from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).parents[1]
    / "tools"
    / "verify_d03_edrnpa_rights_provenance_source_policy_v1.py"
)
spec = importlib.util.spec_from_file_location("edrnpa_source_policy", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

ROOT = Path(__file__).resolve().parents[1]


def _base() -> dict:
    return json.loads(
        (ROOT / "configs/data/d03_edrnpa_rights_provenance_source_policy_v1.json")
        .read_text(encoding="utf-8")
    )


def _reject(payload: dict) -> None:
    with pytest.raises(mod.EdrnpaSourcePolicyError):
        mod.validate_policy(payload, root=ROOT)


def test_exact_source_policy_validates_and_keeps_zero_credit() -> None:
    result = mod.validate_policy(_base(), root=ROOT)
    assert result["admission_policy"]["decision_class"] == (
        "CONDITIONAL_UA_EDRNPA_NORMATIVE_TEXT_SOURCE_POLICY"
    )
    assert result["truth_boundary"]["source_policy_review_complete"] is True
    assert result["truth_boundary"]["parent_materialization_reference_bound"] is True
    assert result["truth_boundary"]["record_level_payload_provenance_executed"] is False
    assert result["admission_policy"]["payload_source_admission_executed"] is False
    assert result["admission_policy"]["admitted_payload_records"] == 0
    assert result["admission_policy"]["admitted_payload_bytes"] == 0
    assert result["truth_boundary"]["training_authorized_bytes"] == 0
    assert result["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert result["truth_boundary"]["tokenizer_fit_authorized"] is False
    assert result["truth_boundary"]["model_training_executed"] is False
    assert result["truth_boundary"]["learned_weights_created"] is False


def test_parent_product_blob_and_zero_credit_contract_are_physically_bound() -> None:
    mod._validate_parent_product(ROOT)


def test_unknown_root_key_fails_closed() -> None:
    payload = _base()
    payload["self_consistent_reseal"] = "forbidden"
    _reject(payload)


def test_unknown_nested_key_fails_closed() -> None:
    payload = _base()
    payload["admission_policy"]["invented_admission"] = True
    _reject(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("parent_product_head_sha", "0" * 40),
        ("parent_product_git_blob_sha1", "1" * 40),
        ("physical_execution_head_sha", "2" * 40),
        ("physical_execution_workflow_run", 1),
        ("physical_execution_workflow_job", 1),
    ],
)
def test_product_and_execution_authority_cannot_be_resealed(
    key: str, value: object
) -> None:
    payload = _base()
    payload["project_authority"][key] = value
    _reject(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("resource_updated", "2026-09-22T11:07:00+03:00"),
        ("resource_md5", "61097fcf212b47e06bcc7c9b23a11c7d"),
        ("source_sha256", "3" * 64),
        ("selected_objects", 177),
        ("selected_bytes", 4000001),
        ("inventory_identity_sha256", "4" * 64),
        ("selected_payload_identity_sha256", "5" * 64),
        ("report_identity_sha256", "6" * 64),
    ],
)
def test_pinned_candidate_identity_cannot_drift(key: str, value: object) -> None:
    payload = _base()
    payload["candidate_binding"][key] = value
    _reject(payload)


def test_newer_live_resource_cannot_silently_replace_executed_version() -> None:
    payload = _base()
    payload["candidate_binding"]["resource_updated"] = "2026-09-22T11:07:00+03:00"
    payload["candidate_binding"]["resource_md5"] = "61097fcf212b47e06bcc7c9b23a11c7d"
    _reject(payload)


def test_primary_public_evidence_cannot_be_rewritten() -> None:
    payload = _base()
    payload["primary_public_evidence"][0]["supported_fact"] = (
        "Public hosting alone authorizes every object for training."
    )
    _reject(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("dataset_license_alone_sufficient_for_payload_admission", True),
        ("requires_record_level_act_card_binding", False),
        ("requires_official_normative_act_classification", False),
        ("blanket_official_document_status_for_selected_objects", True),
        ("embedded_third_party_content_source_admitted", True),
        ("attachment_or_media_payload_source_admitted", True),
        ("ambiguous_provenance_source_admitted", True),
        ("unknown_act_class_source_admitted", True),
        ("record_level_provenance_classification_required", False),
        ("legal_conclusion_claimed", True),
        ("payload_source_admission_executed", True),
        ("admitted_payload_records", 1),
        ("admitted_payload_bytes", 1),
    ],
)
def test_source_policy_cannot_widen_into_payload_admission(
    key: str, value: object
) -> None:
    payload = _base()
    payload["admission_policy"][key] = value
    _reject(payload)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("record_level_payload_provenance_executed", True),
        ("canonical_corpus_admitted", True),
        ("canonical_capacity_credit_bytes", 1),
        ("family_count_credit_added", 1),
        ("training_authorized_bytes", 1),
        ("authorized_unique_loss_positions", 1),
        ("authorized_optimized_target_exposure", 1),
        ("tokenizer_fit_authorized", True),
        ("training_eligible", True),
        ("evaluation_eligible", True),
        ("optimizer_updates", 1),
        ("model_training_executed", True),
        ("learned_weights_created", True),
        ("final_test_accessed", True),
        ("paid_compute_used", True),
        ("foreign_pretrained_weights_used", True),
    ],
)
def test_scientific_truth_cannot_be_widened(key: str, value: object) -> None:
    payload = _base()
    payload["truth_boundary"][key] = value
    _reject(payload)


def test_bool_int_aliases_fail_closed() -> None:
    payload = _base()
    payload["truth_boundary"]["training_authorized_bytes"] = False
    _reject(payload)

    payload = _base()
    payload["truth_boundary"]["model_training_executed"] = 0
    _reject(payload)

    payload = _base()
    payload["admission_policy"]["admitted_payload_records"] = False
    _reject(payload)

    payload = _base()
    payload["project_authority"]["parent_product_pr"] = True
    _reject(payload)


def test_downstream_gates_are_exact_and_cannot_be_dropped() -> None:
    payload = _base()
    payload["downstream_required"] = payload["downstream_required"][:-1]
    _reject(payload)


def test_evidence_order_is_authoritative() -> None:
    payload = _base()
    payload["primary_public_evidence"] = list(
        reversed(payload["primary_public_evidence"])
    )
    _reject(payload)


def test_validate_policy_does_not_mutate_input() -> None:
    payload = _base()
    original = copy.deepcopy(payload)
    mod.validate_policy(payload, root=ROOT)
    assert payload == original
