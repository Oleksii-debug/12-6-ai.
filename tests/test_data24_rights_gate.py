from __future__ import annotations

import hashlib

import pytest

from twelve_six.data.external_sources import (
    PROJECT_RIGHTS_POLICY_REF,
    RIGHTS_APPROVED,
    RIGHTS_REVIEW_REQUIRED,
    USE_ALLOWED,
    USE_DENIED,
    USE_REVIEW_REQUIRED,
    USE_UNKNOWN,
    EligibilityResolver,
    ExternalDataContractError,
    ExternalSourceSpec,
    RightsDecision,
    RightsEvidenceRef,
    SnapshotSpec,
    UsePermissions,
    build_eligibility_inventory,
    build_external_source_registry,
    validate_external_source_registry,
)
from twelve_six.data.multilingual_pretraining import (
    MultilingualDataError,
    PretrainingRecord,
    admit_for_pretraining,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evidence(source_id: str, version: str, kind: str, *, bound_version: str | None = None) -> RightsEvidenceRef:
    return RightsEvidenceRef(
        evidence_id=f"{source_id}-{version}-{kind}",
        evidence_kind=kind,
        uri=f"file:///rights/{source_id}/{version}/{kind}.txt",
        sha256=_sha(f"evidence:{source_id}:{version}:{kind}"),
        captured_at="2026-08-25T12:00:00Z",
        source_id=source_id,
        source_version=bound_version or version,
    )


def _source(
    source_id: str = "candidate",
    version: str = "v1",
    *,
    synthetic: bool = False,
    uses: UsePermissions | None = None,
    status: str = RIGHTS_APPROVED,
    allows_model_training: bool | None = None,
    evidence: tuple[RightsEvidenceRef, ...] | None = None,
    policy_ref: str = PROJECT_RIGHTS_POLICY_REF,
) -> ExternalSourceSpec:
    if uses is None:
        uses = UsePermissions(USE_ALLOWED, USE_ALLOWED, USE_ALLOWED, USE_ALLOWED, USE_DENIED)
    if allows_model_training is None:
        allows_model_training = status == RIGHTS_APPROVED
    if evidence is None:
        evidence = (
            _evidence(source_id, version, "project_authorship" if synthetic else "license_text"),
            _evidence(source_id, version, "policy_decision"),
        )
    return ExternalSourceSpec(
        source_id=source_id,
        source_version=version,
        provider="fixture-provider",
        source_url=f"https://example.invalid/{source_id}",
        source_kind="text",
        purpose="pretraining",
        synthetic=synthetic,
        benchmark_material=False,
        held_out=False,
        snapshot=SnapshotSpec(
            uri=f"file:///snapshots/{source_id}/{version}/payload.txt",
            sha256=_sha(f"snapshot:{source_id}:{version}"),
            size_bytes=1,
            retrieved_at="2026-08-25T12:00:00Z",
            upstream_version=version,
            retrieval_method="fixture",
        ),
        rights=RightsDecision(
            status=status,
            license_id="PROJECT-AUTHORED" if synthetic else "EXPLICIT-EVIDENCE",
            terms_url="https://example.invalid/terms",
            allows_model_training=allows_model_training,
            allows_derivatives=True,
            allows_redistribution=uses.redistribution == USE_ALLOWED,
            policy_ref=policy_ref,
            reviewed_at="2026-08-25T12:00:00Z",
            reviewer_ref="role://data-rights-owner",
            uses=uses,
            evidence_refs=evidence,
        ),
    )


def _record(source: ExternalSourceSpec, *, text: str | None = None) -> PretrainingRecord:
    return PretrainingRecord(
        record_id="record-1",
        source_id=source.source_id,
        source_version=source.source_version,
        source_manifest_sha256=source.source_manifest_sha256,
        split="train",
        source_purpose="pretraining",
        modality="natural",
        text=text or "The training data contains enough English words for deterministic admission testing.",
        language_hint="en",
        external=not source.synthetic,
        project_authored_synthetic=source.synthetic,
    )


def test_missing_machine_readable_rights_and_evidence_fail_closed() -> None:
    source = _source("legacy-like")
    source = ExternalSourceSpec(
        source.source_id, source.source_version, source.provider, source.source_url,
        source.source_kind, source.purpose, source.synthetic, source.benchmark_material,
        source.held_out, source.snapshot,
        RightsDecision(
            status=RIGHTS_APPROVED,
            license_id="EXPLICIT-EVIDENCE",
            terms_url="https://example.invalid/terms",
            allows_model_training=True,
            allows_derivatives=True,
            allows_redistribution=False,
            policy_ref=PROJECT_RIGHTS_POLICY_REF,
            reviewed_at="2026-08-25T12:00:00Z",
            reviewer_ref="role://data-rights-owner",
        ),
    )
    decision = EligibilityResolver(build_external_source_registry([source])).resolve("legacy-like", "v1")
    assert not decision.model_training_eligible
    assert "MISSING_MACHINE_READABLE_USE_DECISIONS" in decision.reasons
    assert "MISSING_IMMUTABLE_RIGHTS_EVIDENCE" in decision.reasons


def test_ambiguous_or_review_required_training_terms_are_not_promoted_to_allowed() -> None:
    uses = UsePermissions(USE_ALLOWED, USE_ALLOWED, USE_ALLOWED, USE_REVIEW_REQUIRED, USE_UNKNOWN)
    source = _source(
        "ambiguous",
        uses=uses,
        status=RIGHTS_REVIEW_REQUIRED,
        allows_model_training=False,
    )
    decision = EligibilityResolver(build_external_source_registry([source])).resolve("ambiguous", "v1")
    assert decision.model_training == USE_REVIEW_REQUIRED
    assert not decision.model_training_eligible
    assert "MODEL_TRAINING_NOT_EXPLICITLY_ALLOWED" in decision.reasons


def test_conflicting_legacy_and_machine_readable_metadata_fails_closed() -> None:
    uses = UsePermissions(USE_ALLOWED, USE_ALLOWED, USE_ALLOWED, USE_DENIED, USE_DENIED)
    source = _source("conflict", uses=uses, status=RIGHTS_APPROVED, allows_model_training=True)
    decision = EligibilityResolver(build_external_source_registry([source])).resolve("conflict", "v1")
    assert not decision.model_training_eligible
    assert "CONFLICTING_MODEL_TRAINING_METADATA" in decision.reasons
    assert "MODEL_TRAINING_NOT_EXPLICITLY_ALLOWED" in decision.reasons


def test_source_version_and_manifest_drift_are_rejected() -> None:
    source = _source("versioned", "v1")
    resolver = EligibilityResolver(build_external_source_registry([source]))
    with pytest.raises(ExternalDataContractError, match="unregistered source identity"):
        resolver.assert_model_training_eligible("versioned", "v2", source.source_manifest_sha256)
    with pytest.raises(ExternalDataContractError, match="SOURCE_VERSION_OR_MANIFEST_DRIFT"):
        resolver.assert_model_training_eligible("versioned", "v1", "0" * 64)


def test_public_access_only_does_not_imply_model_training_rights() -> None:
    uses = UsePermissions(USE_ALLOWED, USE_ALLOWED, USE_ALLOWED, USE_UNKNOWN, USE_UNKNOWN)
    source = _source(
        "public-only",
        uses=uses,
        status=RIGHTS_REVIEW_REQUIRED,
        allows_model_training=False,
    )
    decision = EligibilityResolver(build_external_source_registry([source])).resolve("public-only", "v1")
    assert source.source_url.startswith("https://")
    assert decision.acquisition == USE_ALLOWED
    assert decision.model_training == USE_UNKNOWN
    assert not decision.model_training_eligible


def test_evidence_must_be_immutable_and_bound_to_exact_source_version() -> None:
    bad = (
        _evidence("drift-evidence", "v1", "license_text", bound_version="v0"),
        _evidence("drift-evidence", "v1", "policy_decision"),
    )
    source = _source("drift-evidence", evidence=bad)
    decision = EligibilityResolver(build_external_source_registry([source])).resolve("drift-evidence", "v1")
    assert not decision.model_training_eligible
    assert "EVIDENCE_SOURCE_VERSION_MISMATCH" in decision.reasons
    with pytest.raises(ExternalDataContractError, match="stable"):
        RightsEvidenceRef(
            evidence_id="unstable", evidence_kind="license_text",
            uri="https://example.invalid/terms?latest=1", sha256="1" * 64,
            captured_at="2026-08-25T12:00:00Z", source_id="x", source_version="v1",
        )


def test_synthetic_project_authored_data_requires_registry_evidence_too() -> None:
    source = _source("project-synthetic", synthetic=True)
    resolver = EligibilityResolver(build_external_source_registry([source]))
    admitted = admit_for_pretraining(_record(source), eligibility_resolver=resolver)
    assert admitted.source_id == "project-synthetic"

    fake = PretrainingRecord(
        record_id="fake", source_id="not-registered", source_version="v1",
        source_manifest_sha256="2" * 64, split="train", source_purpose="pretraining",
        modality="natural",
        text="The project authored training data looks valid but has no registered source evidence.",
        language_hint="en", project_authored_synthetic=True,
        rights_status=RIGHTS_APPROVED, allows_model_training=True,
    )
    with pytest.raises(MultilingualDataError, match="unregistered source identity"):
        admit_for_pretraining(fake, eligibility_resolver=resolver)


def test_training_entry_requires_resolver_even_for_project_authored_record() -> None:
    source = _source("project-synthetic", synthetic=True)
    with pytest.raises(MultilingualDataError, match="eligibility resolver is required"):
        admit_for_pretraining(_record(source))


def test_inventory_is_registry_bound_and_separates_allowed_from_review() -> None:
    allowed = _source("allowed", synthetic=True)
    review = _source(
        "review",
        uses=UsePermissions(USE_ALLOWED, USE_ALLOWED, USE_ALLOWED, USE_REVIEW_REQUIRED, USE_UNKNOWN),
        status=RIGHTS_REVIEW_REQUIRED,
        allows_model_training=False,
    )
    inventory = build_eligibility_inventory(build_external_source_registry([review, allowed]))
    assert inventory["candidate_sources"] == 2
    assert inventory["model_training_allowed"] == 1
    assert inventory["model_training_blocked"] == 1
    assert inventory["unknown_or_review_required"] == 1
    assert len(inventory["inventory_sha256"]) == 64


def test_registry_rejects_missing_rights_object() -> None:
    registry = build_external_source_registry([])
    registry["sources"] = [{"source_id": "bad"}]
    with pytest.raises(ExternalDataContractError):
        validate_external_source_registry(registry)
