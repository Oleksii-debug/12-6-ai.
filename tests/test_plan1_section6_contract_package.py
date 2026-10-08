"""Plan 1 S6: canonical baseline preservation and fail-closed schema evolution."""
from __future__ import annotations

import copy
import json

import pytest

import twelve_six.artifact_identity as original_artifacts
import twelve_six.system_architecture as original_architecture
from twelve_six.contracts import (
    ArtifactKind, ArtifactManifest, ArtifactRef, ContractEvolutionError,
    ContractPackageError, ErrorRecord, EvidenceRef, LifecycleObservation,
    baseline_manifest_v1, canonical_runtime_shell_v1, canonical_system_architecture_v1,
    decode_v1, encode_v1, propose_version_change, validate_version_change,
)


def _artifact() -> ArtifactRef:
    return ArtifactRef(ArtifactKind.CORPUS, 1, "a" * 64)


def _evolution(**overrides):
    params = {
        "contract_kind": "artifact_ref",
        "previous_version": 1,
        "next_version": 2,
        "previous_fields": {"kind": "string", "sha256": "string"},
        "next_fields": {"kind": "string", "sha256": "bytes"},
        "migration_fixture_sha256": "b" * 64,
        "deprecation_notice": "Consumers must migrate on explicit opt-in",
        "affected_plans": (1, 2, 3, 8),
        "compatibility_test_sha256": "c" * 64,
    }
    params.update(overrides)
    return propose_version_change(**params)


def test_contract_package_reexports_original_authorities_without_reimplementing() -> None:
    assert ArtifactRef is original_artifacts.ArtifactRef
    assert ArtifactManifest is original_artifacts.ArtifactManifest
    assert canonical_system_architecture_v1 is original_architecture.canonical_system_architecture_v1
    assert canonical_runtime_shell_v1 is original_architecture.canonical_runtime_shell_v1
    baseline = baseline_manifest_v1()
    assert baseline == baseline_manifest_v1()
    assert baseline["architecture_sha256"] == canonical_system_architecture_v1().identity_sha256()
    assert baseline["runtime_shell_sha256"] == canonical_runtime_shell_v1().identity_sha256()
    assert baseline["contains_execution_authority"] is False
    assert baseline["plans_2_to_8_nonblocking"] is True


@pytest.mark.parametrize("kind,payload", [
    ("artifact_ref", _artifact()),
    ("artifact_manifest", ArtifactManifest(schema_version=1, artifact=_artifact(), parents=())),
    ("evidence_ref", EvidenceRef(artifact=_artifact(), evidence_sha256="b" * 64, source_ref="fixture")),
    ("error_record", ErrorRecord("unavailable", "external service unavailable", True)),
    ("lifecycle_observation", LifecycleObservation("event.1", "recovered", 2, "a" * 64)),
])
def test_wire_roundtrip_is_deterministic_and_restart_stable(kind, payload) -> None:
    wire = encode_v1(kind, payload)
    assert encode_v1(kind, decode_v1(wire)) == wire
    assert decode_v1(wire) == payload
    assert b'"version":1' in wire
    assert b'"baseline_sha256":' in wire


def test_unknown_schema_kind_and_foreign_version_rejected() -> None:
    payload = json.loads(encode_v1("artifact_ref", _artifact()))
    for mutation in (
        {"schema": "foreign"},
        {"version": 2},
        {"version": True},
        {"kind": "training_runtime"},
        {"baseline_sha256": "f" * 64},
    ):
        with pytest.raises(ContractPackageError):
            decode_v1(json.dumps({**payload, **mutation}, sort_keys=True,
                                 separators=(",", ":")).encode())


def test_ambiguous_json_duplicate_and_noncanonical_bytes_rejected() -> None:
    wire = encode_v1("artifact_ref", _artifact())
    with pytest.raises(ContractPackageError, match="duplicate"):
        decode_v1(wire.replace(b'"kind":"artifact_ref"', b'"kind":"artifact_ref","kind":"artifact_ref"'))
    with pytest.raises(ContractPackageError, match="noncanonical"):
        decode_v1(json.dumps(json.loads(wire), indent=2).encode())
    with pytest.raises(ContractPackageError):
        decode_v1(b"\xff")
    with pytest.raises(ContractPackageError):
        decode_v1(wire + b" ")


def test_evidence_and_effect_authority_forgery_fail_closed() -> None:
    receipt = json.loads(encode_v1(
        "evidence_ref", EvidenceRef(_artifact(), "b" * 64, "fixture")
    ))
    receipt["payload"]["admitted"] = True
    with pytest.raises(ContractPackageError, match="admission"):
        decode_v1(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
    state = json.loads(encode_v1(
        "lifecycle_observation", LifecycleObservation("event.1", "queued", 1, "a" * 64)
    ))
    state["payload"]["effect_authority"] = True
    with pytest.raises(ContractPackageError, match="effect authority"):
        decode_v1(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())


def test_breaking_contract_migration_receipt_requires_evidence_and_impact() -> None:
    receipt = _evolution()
    assert receipt["breaking"] is True
    assert receipt["decision"] == "REVIEW_REQUIRED"
    assert receipt["auto_migrate_existing_artifacts"] is False
    assert receipt["reopen_terminal_plans"] is False
    assert validate_version_change(receipt) == receipt
    assert _evolution() == receipt


@pytest.mark.parametrize("mutation", [
    {"next_version": 1}, {"next_version": True}, {"migration_fixture_sha256": None},
    {"deprecation_notice": None}, {"affected_plans": ()},
    {"affected_plans": (3, 2)}, {"affected_plans": (1, True)},
    {"compatibility_test_sha256": "fake"},
])
def test_breaking_schema_changes_fail_without_migration_or_impact(mutation) -> None:
    with pytest.raises(ContractEvolutionError):
        _evolution(**mutation)


def test_additive_schema_still_requires_explicit_version_and_tests() -> None:
    declaration = _evolution(
        previous_fields={"kind": "string"}, next_fields={"kind": "string", "hash": "string"},
        migration_fixture_sha256=None, deprecation_notice=None, affected_plans=(1, 2),
    )
    assert declaration["breaking"] is False
    assert declaration["decision"] == "REVIEW_REQUIRED"
    validate_version_change(declaration)


def test_evolution_forged_approval_or_fixture_sha_refused() -> None:
    receipt = _evolution()
    for mutation in ({"decision": "APPROVED"}, {"declaration_sha256": "0" * 64},
                     {"migration_fixture_sha256": "d" * 64},
                     {"reopen_terminal_plans": True}):
        forged = copy.deepcopy(receipt)
        forged.update(mutation)
        with pytest.raises(ContractEvolutionError):
            validate_version_change(forged)
