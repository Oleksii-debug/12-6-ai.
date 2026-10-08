"""Plan-1 S6: contract packaging, fail-closed evolution and restart fixtures."""
from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

import twelve_six.artifact_identity as old_ids
import twelve_six.system_architecture as old_arch
from twelve_six.contracts import (
    BASELINE_SCHEMA_VERSION,
    ArtifactKind,
    ArtifactRef,
    ContractCompatibilityError,
    ContractEvidenceRef,
    ContractEvolutionReceipt,
    ContractSignal,
    build_generation_identity_manifest,
    canonical_system_architecture_v1,
    parse_generation_identity_manifest,
)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def ref() -> ArtifactRef:
    return ArtifactRef(ArtifactKind.CORPUS, 1, sha("corpus"))


def receipt(**overrides: object) -> ContractEvolutionReceipt:
    args = dict(
        contract_name="twelve_six.identity",
        source_version=1,
        target_version=1,
        source_semantics_sha256=sha("baseline"),
        target_semantics_sha256=sha("baseline"),
    )
    args.update(overrides)
    return ContractEvolutionReceipt(**args)


def test_facade_reuses_existing_v1_authority_without_identity_fork() -> None:
    assert BASELINE_SCHEMA_VERSION == 1
    assert ArtifactRef is old_ids.ArtifactRef
    assert canonical_system_architecture_v1 is old_arch.canonical_system_architecture_v1
    assert canonical_system_architecture_v1().identity_sha256() == (
        old_arch.canonical_system_architecture_v1().identity_sha256()
    )


def test_v1_generation_bytes_survive_packaging_restart() -> None:
    refs = {
        kind: ArtifactRef(kind, 1, sha(kind.value))
        for kind in old_ids.CANONICAL_ARTIFACT_KINDS
    }
    old = old_ids.build_generation_identity_manifest(refs)
    new = build_generation_identity_manifest(refs)
    assert new.canonical_json_bytes() == old.canonical_json_bytes()
    loaded = parse_generation_identity_manifest(new.canonical_json_bytes())
    assert loaded.identity_sha256() == old.identity_sha256()


def test_signal_inert_roundtrip_with_exact_artifact_and_evidence() -> None:
    evidence = ContractEvidenceRef(sha("evidence"), "a" * 40)
    signal = ContractSignal(1, "REJECTED", "POLICY_DENIED", ref(), evidence)
    loaded = ContractSignal.from_dict(signal.to_dict())
    assert loaded == signal
    assert loaded.to_dict() == signal.to_dict()


@pytest.mark.parametrize("bad", [0, True, 1.0, "1", 2])
def test_unknown_or_ambiguous_signal_version_is_denied(bad: object) -> None:
    with pytest.raises(ContractCompatibilityError):
        ContractSignal(bad, "CREATED", "OK", ref(), ContractEvidenceRef(sha("e"), "a" * 40))


@pytest.mark.parametrize("bad", ["MOVED", "done", "", False])
def test_signal_cannot_promote_unlisted_lifecycle(bad: object) -> None:
    with pytest.raises(ContractCompatibilityError):
        ContractSignal(1, bad, "OK", ref(), ContractEvidenceRef(sha("e"), "a" * 40))


def test_signal_forbids_mutated_identity_evidence_and_extra_wire_fields() -> None:
    signal = ContractSignal(1, "VERIFIED", "OK", ref(), ContractEvidenceRef(sha("e"), "a" * 40))
    object.__setattr__(signal.evidence, "sha256", "NOT_DIGEST")
    with pytest.raises(ContractCompatibilityError):
        signal.to_dict()
    ok = ContractSignal(1, "CREATED", "OK", ref(), ContractEvidenceRef(sha("e"), "a" * 40))
    wire = ok.to_dict()
    wire["unknown"] = "allowed?"
    with pytest.raises(ContractCompatibilityError):
        ContractSignal.from_dict(wire)


def test_same_version_semantics_replay_is_nonbreaking() -> None:
    assert not receipt().needs_requalification


def test_mutated_same_version_semantics_requires_new_version() -> None:
    with pytest.raises(ContractCompatibilityError):
        receipt(target_semantics_sha256=sha("different"))


def test_breaking_schema_requires_explicit_fixture_and_impact() -> None:
    with pytest.raises(ContractCompatibilityError):
        receipt(target_version=2)
    with pytest.raises(ContractCompatibilityError):
        receipt(target_version=2, migration_fixture_sha256=sha("fixture"))
    versioned = receipt(
        target_version=2,
        target_semantics_sha256=sha("different"),
        migration_fixture_sha256=sha("fixture"),
        affected_plans=(2, 8),
    )
    assert versioned.needs_requalification


def test_downgrade_duplicate_impact_bool_impact_and_tamper_fail_closed() -> None:
    with pytest.raises(ContractCompatibilityError):
        receipt(target_version=0)
    with pytest.raises(ContractCompatibilityError):
        receipt(
            target_version=2,
            migration_fixture_sha256=sha("fixture"),
            affected_plans=(2, 2),
        )
    with pytest.raises(ContractCompatibilityError):
        receipt(
            target_version=2,
            migration_fixture_sha256=sha("fixture"),
            affected_plans=(True,),
        )
    baseline = receipt()
    object.__setattr__(baseline, "target_semantics_sha256", sha("hidden"))
    with pytest.raises(ContractCompatibilityError):
        baseline.needs_requalification


def test_contract_evidence_exact_git_sha_and_digest() -> None:
    with pytest.raises(ContractCompatibilityError):
        ContractEvidenceRef(sha("e"), "a" * 7)
    with pytest.raises(ContractCompatibilityError):
        ContractEvidenceRef("A" * 64, "a" * 40)
    with pytest.raises(ContractCompatibilityError):
        ContractEvidenceRef(sha("e"), "a" * 40 + "1")
    with pytest.raises(ContractCompatibilityError):
        replace(ContractEvidenceRef(sha("e"), "a" * 40), source_commit_sha="bad")
