from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/research/r01_20m_composition_gate_v1.json"
VALIDATOR_PATH = ROOT / "tools/validate_r01_20m_composition_gate.py"

spec = importlib.util.spec_from_file_location("r01_composition_validator", VALIDATOR_PATH)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def _load() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _fully_bound() -> dict:
    data = _load()
    data["status"] = "READY_FOR_ANCESTRY_CHECK"
    data["candidate_head_sha"] = "f" * 40
    for index, component in enumerate(data["components"], start=1):
        component["authority_sha"] = f"{index:040x}"
        component["state"] = "BOUND_EXACT_AUTHORITY"
    return data


def test_committed_manifest_is_structurally_valid_and_blocked() -> None:
    data = _load()
    assert validator.validate_manifest(data) == []
    assessment = validator.assess_composition(data)
    assert assessment["decision"] == "BLOCKED_MISSING_AUTHORITIES"
    assert assessment["ready"] is False


def test_required_role_cannot_be_removed() -> None:
    data = _load()
    data["components"] = data["components"][:-1]
    errors = validator.validate_manifest(data)
    assert any("role set mismatch" in error for error in errors)


def test_ready_status_requires_candidate_head() -> None:
    data = _load()
    data["status"] = "READY_FOR_ANCESTRY_CHECK"
    errors = validator.validate_manifest(data)
    assert any("candidate_head_sha" in error for error in errors)


def test_ready_status_requires_every_component_authority() -> None:
    data = _load()
    data["status"] = "READY_FOR_ANCESTRY_CHECK"
    data["candidate_head_sha"] = "f" * 40
    errors = validator.validate_manifest(data)
    assert any("every authority_sha" in error for error in errors)


def test_component_green_without_candidate_ancestry_fails_closed() -> None:
    data = _fully_bound()
    first_role = data["components"][0]["role"]
    first_sha = data["components"][0]["authority_sha"]

    assessment = validator.assess_composition(
        data,
        commit_exists=lambda _sha: True,
        is_ancestor=lambda component_sha, _candidate: component_sha != first_sha,
    )

    assert assessment["decision"] == "BLOCKED_COMPONENT_NOT_ANCESTOR"
    assert assessment["non_ancestor_roles"] == [first_role]
    assert assessment["ready"] is False


def test_all_required_authorities_on_one_candidate_can_pass_runtime_check() -> None:
    data = _fully_bound()
    assessment = validator.assess_composition(
        data,
        commit_exists=lambda _sha: True,
        is_ancestor=lambda _component, _candidate: True,
    )
    assert assessment["decision"] == "PASS_SINGLE_COMPOSED_CANDIDATE"
    assert assessment["ready"] is True


def test_committed_manifest_cannot_self_assert_pass() -> None:
    data = copy.deepcopy(_fully_bound())
    data["status"] = "PASS"
    errors = validator.validate_manifest(data)
    assert any("must not self-assert PASS" in error for error in errors)
