from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.common_pile_loc_rights import (
    DATASET_CONTRACT,
    POLICY_IDENTITY,
    LocSourceRightsError,
    git_blob_sha1,
    policy_identity,
    validate_policy,
    validate_repository_bindings,
)

POLICY = Path("configs/data/d03_common_pile_loc_source_rights_v1.json")


@pytest.fixture
def payload() -> dict:
    return json.loads(POLICY.read_text(encoding="utf-8"))


def reseal(value: dict) -> dict:
    value["policy_identity_sha256"] = policy_identity(value)
    return value


def test_policy_is_exact_and_zero_credit(payload: dict) -> None:
    assert validate_policy(payload) == POLICY_IDENTITY
    truth = payload["truth_boundary"]
    assert truth["payload_source_admission_executed"] is False
    assert truth["source_capacity_bytes_credited"] == 0
    assert truth["training_authorized_bytes"] == 0
    assert truth["authorized_optimized_target_exposure"] == 0
    assert truth["model_training_executed"] is False
    assert truth["optimizer_updates"] == 0
    assert truth["final_test_payload_accessed"] is False


def test_repository_binding_matches_live_main_registry(payload: dict) -> None:
    validate_repository_bindings(payload, Path("."))


def test_git_blob_sha1_matches_known_empty_blob() -> None:
    assert git_blob_sha1(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("generic_registry", "blob_sha1", "0" * 40, "generic_registry.blob_sha1"),
        ("upstream_code", "revision", "0" * 40, "upstream_code.revision"),
        ("upstream_code", "collector_blob_sha1", "0" * 40, "collector_blob_sha1"),
        ("upstream_code", "metadata_blob_sha1", "0" * 40, "metadata_blob_sha1"),
        ("upstream_code", "readme_blob_sha1", "0" * 40, "readme_blob_sha1"),
        ("dataset_contract", "revision", "0" * 40, "dataset_contract.revision"),
        ("dataset_contract", "shard_lfs_sha256", "0" * 64, "shard_lfs_sha256"),
        ("record_contract", "expected_license", "CC-BY-4.0", "expected_license"),
        ("record_contract", "expected_language", "English", "expected_language"),
        ("record_contract", "source_value", "library_of_congress", "source_value"),
        ("project_decision", "decision", "TRAINING_AUTHORIZED", "decision"),
        ("truth_boundary", "training_authorized_bytes", 1, "training_authorized_bytes"),
        (
            "truth_boundary",
            "authorized_optimized_target_exposure",
            1,
            "authorized_optimized_target_exposure",
        ),
    ],
)
def test_resealed_semantic_substitution_fails(
    payload: dict,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    mutated = copy.deepcopy(payload)
    mutated[section][field] = value
    reseal(mutated)
    with pytest.raises(LocSourceRightsError, match=message):
        validate_policy(mutated)


def test_primary_evidence_fact_substitution_fails_even_when_resealed(payload: dict) -> None:
    mutated = copy.deepcopy(payload)
    mutated["primary_evidence"][0]["supported_fact"] = "Public domain, trust me."
    reseal(mutated)
    with pytest.raises(LocSourceRightsError, match="supported_fact"):
        validate_policy(mutated)


def test_primary_evidence_unknown_field_fails(payload: dict) -> None:
    mutated = copy.deepcopy(payload)
    mutated["primary_evidence"][0]["observed_fact"] = "self-authored promotion"
    reseal(mutated)
    with pytest.raises(LocSourceRightsError, match="schema drift"):
        validate_policy(mutated)


def test_unknown_policy_field_fails(payload: dict) -> None:
    mutated = copy.deepcopy(payload)
    mutated["training_authorized"] = True
    reseal(mutated)
    with pytest.raises(LocSourceRightsError, match="policy schema drift"):
        validate_policy(mutated)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("dataset_contract", "shard_compressed_bytes"),
        ("truth_boundary", "source_capacity_bytes_credited"),
        ("truth_boundary", "training_authorized_bytes"),
        ("truth_boundary", "authorized_optimized_target_exposure"),
        ("truth_boundary", "evaluation_authorized_bytes"),
        ("truth_boundary", "optimizer_updates"),
    ],
)
def test_bool_cannot_alias_integer_fields(payload: dict, section: str, field: str) -> None:
    mutated = copy.deepcopy(payload)
    mutated[section][field] = False
    reseal(mutated)
    with pytest.raises(LocSourceRightsError, match=field):
        validate_policy(mutated)


def test_incumbent_contract_is_exactly_frozen(payload: dict) -> None:
    assert payload["dataset_contract"] == DATASET_CONTRACT
    assert payload["dataset_contract"]["origin_pr"] == 1135
    assert payload["dataset_contract"]["origin_head"] == (
        "c72bef68127c82abd70016eeaf6070ff4fe096aa"
    )


def test_record_url_lineage_is_exact(payload: dict) -> None:
    record = payload["record_contract"]
    assert record["item_url_template"] == "https://www.loc.gov/item/{lccn}"
    assert record["text_file_url_allowed_hosts"] == ["tile.loc.gov", "tiles.loc.gov"]


def test_policy_identity_detects_unsealed_drift(payload: dict) -> None:
    mutated = copy.deepcopy(payload)
    mutated["primary_evidence"][1]["title"] += " drift"
    with pytest.raises(LocSourceRightsError, match="primary_evidence"):
        validate_policy(mutated)
