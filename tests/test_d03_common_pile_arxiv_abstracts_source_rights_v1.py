from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.common_pile_arxiv_abstracts_rights import (
    POLICY_IDENTITY,
    ArxivAbstractsSourceRightsError,
    git_blob_sha1,
    policy_identity,
    validate_policy,
    validate_repository_bindings,
)

POLICY_PATH = Path("configs/data/d03_common_pile_arxiv_abstracts_source_rights_v1.json")


def load_policy() -> dict[str, object]:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def reseal(payload: dict[str, object]) -> None:
    payload["policy_identity_sha256"] = policy_identity(payload)


def test_policy_identity_and_zero_credit_are_frozen() -> None:
    payload = load_policy()
    assert validate_policy(payload) == POLICY_IDENTITY
    truth = payload["truth_boundary"]
    assert isinstance(truth, dict)
    assert truth["source_capacity_bytes_credited"] == 0
    assert truth["training_authorized_bytes"] == 0
    assert truth["authorized_unique_loss_positions"] == 0
    assert truth["authorized_optimized_target_exposure"] == 0
    assert truth["tokenizer_fit_authorized"] is False
    assert truth["model_training_executed"] is False
    assert truth["optimizer_updates"] == 0


def test_repository_bindings_match_current_main_contracts() -> None:
    validate_repository_bindings(load_policy(), ".")


def test_git_blob_sha1_matches_known_empty_blob() -> None:
    assert git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("generic_registry", "source_key", "arxiv_papers"),
        ("generic_registry", "expected_rights_basis_class", "PER_DOCUMENT_OPEN_LICENSE_FILTER"),
        ("upstream_common_pile", "revision", "0" * 40),
        ("dataset_contract", "shard_lfs_sha256", "0" * 64),
        ("dataset_contract", "shard_compressed_bytes", True),
        ("record_contract", "source_value", "arxiv-papers"),
        ("record_contract", "expected_metadata_license", "CC-BY"),
        ("project_decision", "full_eprint_rights_inferred", True),
        ("truth_boundary", "source_capacity_bytes_credited", 1),
        ("truth_boundary", "authorized_optimized_target_exposure", 1),
    ],
)
def test_contract_substitution_fails_closed(
    section: str,
    field: str,
    value: object,
) -> None:
    payload = copy.deepcopy(load_policy())
    section_value = payload[section]
    assert isinstance(section_value, dict)
    section_value[field] = value
    reseal(payload)
    with pytest.raises(ArxivAbstractsSourceRightsError):
        validate_policy(payload)


def test_primary_evidence_substitution_fails_closed() -> None:
    payload = copy.deepcopy(load_policy())
    evidence = payload["primary_evidence"]
    assert isinstance(evidence, list)
    assert isinstance(evidence[0], dict)
    evidence[0]["url"] = "https://example.com/arxiv-terms"
    reseal(payload)
    with pytest.raises(ArxivAbstractsSourceRightsError):
        validate_policy(payload)


def test_unknown_primary_evidence_field_fails_closed() -> None:
    payload = copy.deepcopy(load_policy())
    evidence = payload["primary_evidence"]
    assert isinstance(evidence, list)
    assert isinstance(evidence[0], dict)
    evidence[0]["unreviewed_note"] = "widen rights"
    reseal(payload)
    with pytest.raises(ArxivAbstractsSourceRightsError):
        validate_policy(payload)


def test_unknown_top_level_field_fails_closed() -> None:
    payload = copy.deepcopy(load_policy())
    payload["training_authorized"] = True
    reseal(payload)
    with pytest.raises(ArxivAbstractsSourceRightsError):
        validate_policy(payload)


def test_arxiv_metadata_scope_is_narrow_and_explicit() -> None:
    payload = load_policy()
    decision = payload["project_decision"]
    record = payload["record_contract"]
    assert isinstance(decision, dict)
    assert isinstance(record, dict)
    assert decision["decision"] == "CONDITIONAL_SOURCE_ADMISSION"
    assert decision["scope"] == "SOURCE_POLICY_ONLY"
    assert decision["cc0_scope_limited_to_descriptive_metadata_including_abstract"] is True
    assert decision["full_eprint_rights_inferred"] is False
    assert decision["conditional_admission_requires_exact_record_cc0"] is True
    assert record["rights_scope"] == "DESCRIPTIVE_METADATA_INCLUDING_ABSTRACT"


def test_unsealed_identity_drift_fails_closed() -> None:
    payload = copy.deepcopy(load_policy())
    decision = payload["project_decision"]
    assert isinstance(decision, dict)
    decision["full_eprint_rights_inferred"] = True
    with pytest.raises(ArxivAbstractsSourceRightsError, match="policy identity"):
        validate_policy(payload)
