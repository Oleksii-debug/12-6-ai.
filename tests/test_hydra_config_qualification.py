from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "validate_hydra_config_qualification.py"
CONTRACT_PATH = ROOT / "configs" / "research" / "hydra_config_qualification_v1.json"

_spec = importlib.util.spec_from_file_location("hydra_qualification", TOOL_PATH)
assert _spec is not None and _spec.loader is not None
hydra_qualification = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hydra_qualification)

QualificationError = hydra_qualification.QualificationError
build_evidence = hydra_qualification.build_evidence
canonical_sha256 = hydra_qualification.canonical_sha256
portable_export_payload = hydra_qualification.portable_export_payload
validate_contract = hydra_qualification.validate_contract
validate_observation = hydra_qualification.validate_observation


@pytest.fixture()
def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _fixture(contract: dict) -> dict:
    return copy.deepcopy(contract["project_owned_fixture"])


def _rehash_observation(observation: dict) -> None:
    observation["resolved_config_sha256"] = canonical_sha256(observation["resolved_config"])
    observation["clean_rebuild_resolved_config_sha256"] = observation[
        "resolved_config_sha256"
    ]
    observation["portable_export_sha256"] = canonical_sha256(
        portable_export_payload(observation)
    )


def test_contract_and_project_owned_fixture_pass(contract: dict) -> None:
    summary = validate_contract(contract)
    assert summary["current_state"] == "CANDIDATE"
    evidence = build_evidence(contract, _fixture(contract))
    assert evidence["verdict"] == "PASS_CONTRACT_MECHANICS_CANDIDATE_NOT_ADOPTED"
    assert evidence["hydra_executed"] is False
    assert evidence["hydra_adopted"] is False
    assert evidence["stage_promotion_granted"] is False
    assert all(evidence["gates"].values())


def test_evidence_is_byte_deterministic(contract: dict) -> None:
    observation = _fixture(contract)
    first = build_evidence(contract, observation)
    second = build_evidence(contract, copy.deepcopy(observation))
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    assert first["evidence_sha256"] == second["evidence_sha256"]


def test_declared_override_changes_experiment_identity(contract: dict) -> None:
    baseline = build_evidence(contract, _fixture(contract))
    changed = _fixture(contract)
    changed["override_ledger"][0]["value"] = 32
    changed["resolved_config"]["training"]["steps"] = 32
    _rehash_observation(changed)

    changed_evidence = build_evidence(contract, changed)
    assert changed_evidence["experiment_identity_sha256"] != baseline[
        "experiment_identity_sha256"
    ]
    assert changed_evidence["resolved_config_sha256"] != baseline[
        "resolved_config_sha256"
    ]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("authority", "hydra_is_canonical_lineage_authority"), True),
        (("authority", "hydra_can_gate_stage_promotion"), True),
        (("execution_policy", "paid_compute_authorized"), True),
        (("execution_policy", "model_training_authorized"), True),
        (("promotion", "this_contract_grants_adoption"), True),
        (("promotion", "this_contract_grants_stage_promotion"), True),
        (("truth_boundary", "hydra_executed"), True),
        (("truth_boundary", "foreign_pretrained_weights_used"), True),
    ],
)
def test_contract_rejects_authority_or_truth_escalation(
    contract: dict, path: tuple[str, str], value: object
) -> None:
    bad = copy.deepcopy(contract)
    bad[path[0]][path[1]] = value
    with pytest.raises(QualificationError):
        validate_contract(bad)


def test_contract_rejects_mutable_or_drifted_upstream_identity(contract: dict) -> None:
    bad = copy.deepcopy(contract)
    bad["upstream"]["release_tag"] = "main"
    with pytest.raises(QualificationError):
        validate_contract(bad)

    bad = copy.deepcopy(contract)
    bad["upstream"]["commit_sha"] = "0" * 40
    with pytest.raises(QualificationError):
        validate_contract(bad)


def test_contract_rejects_registry_or_authority_drift(contract: dict) -> None:
    bad = copy.deepcopy(contract)
    bad["authority"]["registry"]["blob_sha"] = "not-a-sha"
    with pytest.raises(QualificationError):
        validate_contract(bad)

    bad = copy.deepcopy(contract)
    bad["authority"]["canonical_lineage_authorities"].append("HYDRA_OUTPUT_DIR")
    with pytest.raises(QualificationError):
        validate_contract(bad)


@pytest.mark.parametrize(
    "field",
    [
        "hidden_overrides_detected",
        "runtime_environment_interpolation_detected",
        "secret_bearing_fields_present",
        "hydra_runtime_output_is_lineage_authority",
    ],
)
def test_observation_rejects_hidden_or_nonportable_state(contract: dict, field: str) -> None:
    bad = _fixture(contract)
    bad[field] = True
    with pytest.raises(QualificationError):
        validate_observation(contract, bad)


def test_observation_rejects_unapproved_or_duplicate_override_source(contract: dict) -> None:
    bad = _fixture(contract)
    bad["override_ledger"][0]["source"] = "ENVIRONMENT_IMPLICIT"
    with pytest.raises(QualificationError):
        validate_observation(contract, bad)

    bad = _fixture(contract)
    bad["override_ledger"].append(copy.deepcopy(bad["override_ledger"][0]))
    with pytest.raises(QualificationError):
        validate_observation(contract, bad)


def test_observation_rejects_missing_default_hash(contract: dict) -> None:
    bad = _fixture(contract)
    bad["defaults_trace"][0]["sha256"] = ""
    with pytest.raises(QualificationError):
        validate_observation(contract, bad)


def test_observation_rejects_resolved_config_hash_drift(contract: dict) -> None:
    bad = _fixture(contract)
    bad["resolved_config"]["training"]["steps"] = 99
    with pytest.raises(QualificationError):
        validate_observation(contract, bad)


def test_observation_rejects_clean_rebuild_mismatch(contract: dict) -> None:
    bad = _fixture(contract)
    bad["clean_rebuild_resolved_config_sha256"] = "0" * 64
    with pytest.raises(QualificationError):
        validate_observation(contract, bad)


def test_observation_rejects_portable_export_mismatch(contract: dict) -> None:
    bad = _fixture(contract)
    bad["portable_export_sha256"] = "0" * 64
    with pytest.raises(QualificationError):
        validate_observation(contract, bad)


def test_observation_rejects_wrong_base_git_sha(contract: dict) -> None:
    bad = _fixture(contract)
    bad["base_git_sha"] = "f" * 40
    _rehash_observation(bad)
    with pytest.raises(QualificationError):
        validate_observation(contract, bad)


def test_evidence_hash_is_self_consistent(contract: dict) -> None:
    evidence = build_evidence(contract, _fixture(contract))
    core = dict(evidence)
    digest = core.pop("evidence_sha256")
    assert digest == canonical_sha256(core)
