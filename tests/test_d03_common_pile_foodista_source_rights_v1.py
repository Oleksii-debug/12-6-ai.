from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
MODULE = ROOT / "src/twelve_six/common_pile_foodista_rights.py"
spec = importlib.util.spec_from_file_location("foodista_rights", MODULE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

POLICY_PATH = ROOT / "configs/data/d03_common_pile_foodista_source_rights_v1.json"
REGISTRY_PATH = ROOT / "configs/data/common_pile_source_rights_v1.json"


def load_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def load_registry_bytes() -> bytes:
    return REGISTRY_PATH.read_bytes()


def test_foodista_rights_authority_is_exact_and_blocked_zero_credit() -> None:
    policy = load_policy()
    mod.validate_policy(policy, load_registry_bytes())

    assert policy["status"] == "SOURCE_POLICY_REVIEW_BLOCKED_ZERO_CREDIT"
    assert policy["decision"]["source_scope_qualified"] is False
    assert policy["decision"]["canonical_payload_admission_permitted"] is False
    assert policy["decision"]["zero_credit_mechanics_may_continue"] is True
    assert policy["risk_assessment"]["automated_web_crawl_declared_by_primary_source"] is True
    assert policy["risk_assessment"]["record_level_original_rights_provenance_available"] is False
    assert type(policy["truth_boundary"]["training_authorized_bytes"]) is int
    assert policy["truth_boundary"]["training_authorized_bytes"] == 0


def test_registry_binding_rejects_credit_or_review_drift() -> None:
    registry = json.loads(load_registry_bytes().decode("utf-8"))
    row = next(row for row in registry["sources"] if row["key"] == "foodista")
    row["credited_bytes"] = 1
    mutated = json.dumps(registry, separators=(",", ":")).encode("utf-8")

    with pytest.raises(ValueError):
        mod.validate_policy(load_policy(), mutated)


def test_policy_rejects_qualification_without_new_authority_version() -> None:
    policy = load_policy()
    policy["decision"]["source_scope_qualified"] = True
    with pytest.raises(ValueError):
        mod.validate_policy(policy, load_registry_bytes())


def test_policy_rejects_bool_alias_for_zero_credit() -> None:
    policy = load_policy()
    policy["truth_boundary"]["training_authorized_bytes"] = False
    with pytest.raises(ValueError):
        mod.validate_policy(policy, load_registry_bytes())


def test_policy_rejects_fabricated_record_level_provenance() -> None:
    policy = load_policy()
    policy["risk_assessment"]["record_level_original_rights_provenance_available"] = True
    with pytest.raises(ValueError):
        mod.validate_policy(policy, load_registry_bytes())


def test_policy_identity_is_content_bound() -> None:
    policy = load_policy()
    assert policy["policy_identity_sha256"] == mod.POLICY_IDENTITY
    assert mod._canonical_policy_hash(policy) == mod.POLICY_IDENTITY

    drift = deepcopy(policy)
    drift["primary_evidence"][0]["supported_fact"] += " drift"
    assert mod._canonical_policy_hash(drift) != mod.POLICY_IDENTITY
    with pytest.raises(ValueError):
        mod.validate_policy(drift, load_registry_bytes())
