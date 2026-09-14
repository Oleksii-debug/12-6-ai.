from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.common_pile_rights import (
    CommonPileRightsError,
    registry_identity,
    validate_registry,
)

REGISTRY = Path("configs/data/common_pile_source_rights_v1.json")


@pytest.fixture
def payload() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def reseal(payload: dict) -> dict:
    payload["registry_identity_sha256"] = registry_identity(payload)
    return payload


def test_valid_registry(payload: dict) -> None:
    assert validate_registry(payload) == payload["registry_identity_sha256"]
    assert len(payload["sources"]) == 30


def test_identity_is_deterministic(payload: dict) -> None:
    assert registry_identity(payload) == registry_identity(copy.deepcopy(payload))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda p: p["global_policy"].__setitem__("canonical_training_authorized", True),
            "canonical_training_authorized",
        ),
        (lambda p: p["global_policy"].__setitem__("corpus_credit_bytes", 1), "corpus credit"),
        (
            lambda p: p["global_policy"].__setitem__("authorized_loss_positions", 1),
            "loss-position credit",
        ),
        (
            lambda p: p["global_policy"].__setitem__(
                "dataset_package_license_is_training_authority", True
            ),
            "dataset_package_license_is_training_authority",
        ),
        (
            lambda p: p["global_policy"].__setitem__(
                "repository_code_license_is_dataset_license", True
            ),
            "repository_code_license_is_dataset_license",
        ),
        (
            lambda p: p["global_policy"].__setitem__("final_test_payload_accessed", True),
            "final_test_payload_accessed",
        ),
        (lambda p: p.__setitem__("status", "ADOPTED"), "terminality"),
        (
            lambda p: p["upstream_authority"].__setitem__("audited_code_commit", "main"),
            "immutable commit",
        ),
    ],
)
def test_global_fail_closed_mutations(payload: dict, mutator, message: str) -> None:
    mutated = copy.deepcopy(payload)
    mutator(mutated)
    reseal(mutated)
    with pytest.raises(CommonPileRightsError, match=message):
        validate_registry(mutated)


def test_missing_source_is_rejected(payload: dict) -> None:
    mutated = copy.deepcopy(payload)
    mutated["sources"].pop()
    reseal(mutated)
    with pytest.raises(CommonPileRightsError, match="exactly 30"):
        validate_registry(mutated)


def test_duplicate_source_is_rejected(payload: dict) -> None:
    mutated = copy.deepcopy(payload)
    mutated["sources"][-1] = copy.deepcopy(mutated["sources"][0])
    reseal(mutated)
    with pytest.raises(CommonPileRightsError, match="source keys must be unique"):
        validate_registry(mutated)


def test_source_cannot_self_authorize(payload: dict) -> None:
    mutated = copy.deepcopy(payload)
    mutated["sources"][0]["canonical_training_authorized"] = True
    reseal(mutated)
    with pytest.raises(CommonPileRightsError, match="training authorization"):
        validate_registry(mutated)


def test_source_review_cannot_be_skipped(payload: dict) -> None:
    mutated = copy.deepcopy(payload)
    mutated["sources"][0]["project_review_status"] = "APPROVED"
    reseal(mutated)
    with pytest.raises(CommonPileRightsError, match="source review"):
        validate_registry(mutated)


def test_source_rights_and_provenance_are_required(payload: dict) -> None:
    mutated = copy.deepcopy(payload)
    mutated["sources"][0]["upstream_rights_claim"] = ""
    reseal(mutated)
    with pytest.raises(CommonPileRightsError, match="upstream_rights_claim"):
        validate_registry(mutated)

    mutated = copy.deepcopy(payload)
    mutated["sources"][0]["provenance_summary"] = ""
    reseal(mutated)
    with pytest.raises(CommonPileRightsError, match="provenance_summary"):
        validate_registry(mutated)


def test_final_test_firewall_is_source_local(payload: dict) -> None:
    mutated = copy.deepcopy(payload)
    mutated["sources"][0]["final_test_excluded"] = False
    reseal(mutated)
    with pytest.raises(CommonPileRightsError, match="final-test firewall"):
        validate_registry(mutated)


def test_identity_detects_unsealed_semantic_drift(payload: dict) -> None:
    mutated = copy.deepcopy(payload)
    mutated["sources"][0]["provenance_summary"] += " drift"
    with pytest.raises(CommonPileRightsError, match="registry identity mismatch"):
        validate_registry(mutated)
