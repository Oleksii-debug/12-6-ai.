"""Fail-closed provenance, rights evidence, eligibility, and contamination contracts."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

REGISTRY_SCHEMA = "12-6.external-source-registry.v2"
LEGACY_REGISTRY_SCHEMA = "12-6.external-source-registry.v1"
RESERVED_SCHEMA = "12-6.reserved-fingerprints.v1"
ELIGIBILITY_INVENTORY_SCHEMA = "12-6.training-eligibility-inventory.v1"
PROJECT_RIGHTS_POLICY_REF = "policy://12-6/data/explicit-model-training-evidence-v1"

RIGHTS_APPROVED = "APPROVED_FOR_TRAINING"
RIGHTS_REJECTED = "REJECTED"
RIGHTS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
_ALLOWED_RIGHTS = frozenset({RIGHTS_APPROVED, RIGHTS_REJECTED, RIGHTS_REVIEW_REQUIRED})

USE_ALLOWED = "ALLOWED"
USE_DENIED = "DENIED"
USE_UNKNOWN = "UNKNOWN"
USE_REVIEW_REQUIRED = "REVIEW_REQUIRED"
_ALLOWED_USE_STATES = frozenset({USE_ALLOWED, USE_DENIED, USE_UNKNOWN, USE_REVIEW_REQUIRED})
_USE_FIELDS = ("acquisition", "storage", "analysis", "model_training", "redistribution")
_TRAIN_PURPOSES = frozenset({"pretraining", "training"})
_RESERVED_PURPOSES = frozenset({"benchmark", "evaluation", "validation", "test", "heldout_test"})
_SNAPSHOT_SCHEMES = frozenset({"file", "s3", "gs", "hf", "az", "r2"})
_EVIDENCE_SCHEMES = frozenset({"https", "file", "s3", "gs", "hf", "az", "r2"})
_RIGHTS_EVIDENCE_KINDS = frozenset({"license_text", "terms_snapshot", "explicit_permission", "project_authorship"})


class ExternalDataContractError(ValueError):
    """Raised when an external-source provenance invariant is violated."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalDataContractError(f"{field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if text != text.lower() or len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ExternalDataContractError(f"{field} must be a lowercase SHA-256 hex digest")
    return text


def _validate_source_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "hf"}:
        raise ExternalDataContractError("source_url must use https or hf")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ExternalDataContractError("source_url must contain no credentials/query/fragment")


def _validate_stable_uri(value: str, field: str, schemes: frozenset[str]) -> None:
    text = _require_text(value, field)
    if ".." in PurePosixPath(text.replace("\\", "/")).parts:
        raise ExternalDataContractError(f"{field} must not traverse parent directories")
    parsed = urlsplit(text)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ExternalDataContractError(f"{field} must be stable and contain no credentials/query/fragment")
    if parsed.scheme and parsed.scheme not in schemes:
        raise ExternalDataContractError(f"{field} scheme {parsed.scheme!r} is not approved")


@dataclass(frozen=True)
class UsePermissions:
    """Independent use states. Public accessibility never implies any state."""
    acquisition: str
    storage: str
    analysis: str
    model_training: str
    redistribution: str

    def __post_init__(self) -> None:
        for field in _USE_FIELDS:
            value = getattr(self, field)
            if value not in _ALLOWED_USE_STATES:
                raise ExternalDataContractError(f"rights.uses.{field} has unsupported state: {value!r}")

    def to_dict(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in _USE_FIELDS}


@dataclass(frozen=True)
class RightsEvidenceRef:
    """Content-addressed evidence bound to one exact source/version identity."""
    evidence_id: str
    evidence_kind: str
    uri: str
    sha256: str
    captured_at: str
    source_id: str
    source_version: str

    def __post_init__(self) -> None:
        for field in ("evidence_id", "evidence_kind", "uri", "captured_at", "source_id", "source_version"):
            _require_text(getattr(self, field), f"rights.evidence_refs.{field}")
        _require_sha256(self.sha256, "rights.evidence_refs.sha256")
        _validate_stable_uri(self.uri, "rights.evidence_refs.uri", _EVIDENCE_SCHEMES)


@dataclass(frozen=True)
class RightsDecision:
    """Recorded metadata; model-training entry is decided only by EligibilityResolver."""
    status: str
    license_id: str
    terms_url: str
    allows_model_training: bool
    allows_derivatives: bool
    allows_redistribution: bool
    policy_ref: str
    reviewed_at: str
    reviewer_ref: str
    uses: UsePermissions | None = None
    evidence_refs: tuple[RightsEvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _ALLOWED_RIGHTS:
            raise ExternalDataContractError(f"unsupported rights status: {self.status}")
        for field in ("license_id", "terms_url", "policy_ref", "reviewed_at", "reviewer_ref"):
            _require_text(getattr(self, field), f"rights.{field}")
        if self.license_id.upper() == "NOASSERTION" and self.status == RIGHTS_APPROVED:
            raise ExternalDataContractError("NOASSERTION cannot be approved for training")
        for field in ("allows_model_training", "allows_derivatives", "allows_redistribution"):
            if type(getattr(self, field)) is not bool:
                raise ExternalDataContractError(f"rights.{field} must be an exact boolean")
        if self.status == RIGHTS_APPROVED and not self.allows_model_training:
            raise ExternalDataContractError("APPROVED_FOR_TRAINING requires allows_model_training=true")
        if self.uses is not None and not isinstance(self.uses, UsePermissions):
            raise ExternalDataContractError("rights.uses must be UsePermissions or null")
        if not isinstance(self.evidence_refs, tuple) or any(not isinstance(x, RightsEvidenceRef) for x in self.evidence_refs):
            raise ExternalDataContractError("rights.evidence_refs must be a tuple of RightsEvidenceRef")
        ids = [item.evidence_id for item in self.evidence_refs]
        if len(ids) != len(set(ids)):
            raise ExternalDataContractError("rights evidence_id values must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "license_id": self.license_id,
            "terms_url": self.terms_url,
            "allows_model_training": self.allows_model_training,
            "allows_derivatives": self.allows_derivatives,
            "allows_redistribution": self.allows_redistribution,
            "policy_ref": self.policy_ref,
            "reviewed_at": self.reviewed_at,
            "reviewer_ref": self.reviewer_ref,
            "uses": None if self.uses is None else self.uses.to_dict(),
            "evidence_refs": [asdict(item) for item in self.evidence_refs],
        }


@dataclass(frozen=True)
class SnapshotSpec:
    uri: str
    sha256: str
    size_bytes: int
    retrieved_at: str
    upstream_version: str
    retrieval_method: str

    def __post_init__(self) -> None:
        _validate_stable_uri(self.uri, "snapshot.uri", _SNAPSHOT_SCHEMES)
        _require_sha256(self.sha256, "snapshot.sha256")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ExternalDataContractError("snapshot.size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ExternalDataContractError("snapshot.size_bytes must be non-negative")
        for field in ("retrieved_at", "upstream_version", "retrieval_method"):
            _require_text(getattr(self, field), f"snapshot.{field}")


@dataclass(frozen=True)
class ExternalSourceSpec:
    source_id: str
    source_version: str
    provider: str
    source_url: str
    source_kind: str
    purpose: str
    synthetic: bool
    benchmark_material: bool
    held_out: bool
    snapshot: SnapshotSpec
    rights: RightsDecision

    def __post_init__(self) -> None:
        for field in ("source_id", "source_version", "provider", "source_kind", "purpose"):
            _require_text(getattr(self, field), field)
        _require_text(self.source_url, "source_url")
        _validate_source_url(self.source_url)
        for field in ("synthetic", "benchmark_material", "held_out"):
            if type(getattr(self, field)) is not bool:
                raise ExternalDataContractError(f"{field} must be an exact boolean")
        purpose = self.purpose.strip().lower()
        if purpose in _TRAIN_PURPOSES and (self.benchmark_material or self.held_out):
            raise ExternalDataContractError("benchmark/held-out material cannot declare a training purpose")
        if purpose in _RESERVED_PURPOSES and not (self.benchmark_material or self.held_out):
            raise ExternalDataContractError("reserved evaluation/test purposes must be marked benchmark_material or held_out")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_version": self.source_version,
            "provider": self.provider,
            "source_url": self.source_url,
            "source_kind": self.source_kind,
            "purpose": self.purpose,
            "synthetic": self.synthetic,
            "benchmark_material": self.benchmark_material,
            "held_out": self.held_out,
            "snapshot": asdict(self.snapshot),
            "rights": self.rights.to_dict(),
        }

    @property
    def source_manifest_sha256(self) -> str:
        core = {"schema_version": "12-6.source-manifest.v1", "source": self.to_dict()}
        return _sha256_bytes(_canonical_json_bytes(core))

    def assert_training_eligible(self) -> None:
        purpose = self.purpose.strip().lower()
        if purpose not in _TRAIN_PURPOSES:
            raise ExternalDataContractError(f"{self.source_id}: source purpose is not training")
        if self.benchmark_material or self.held_out:
            raise ExternalDataContractError(f"{self.source_id}: benchmark/held-out source is reserved")
        if self.rights.status != RIGHTS_APPROVED or not self.rights.allows_model_training:
            raise ExternalDataContractError(f"{self.source_id}: rights are not approved for training")
        if self.rights.license_id.upper() == "NOASSERTION":
            raise ExternalDataContractError(f"{self.source_id}: license state is unresolved")
        EligibilityResolver(build_external_source_registry([self])).assert_model_training_eligible(
            self.source_id, self.source_version, self.source_manifest_sha256
        )


@dataclass(frozen=True)
class EligibilityDecision:
    source_id: str
    source_version: str
    source_manifest_sha256: str
    registry_identity_sha256: str
    acquisition: str
    storage: str
    analysis: str
    model_training: str
    redistribution: str
    model_training_eligible: bool
    source_purpose: str
    synthetic: bool
    evidence_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_ids"] = list(self.evidence_ids)
        value["reasons"] = list(self.reasons)
        return value


class EligibilityResolver:
    """Resolve use permissions from the canonical D03 source registry; never infer rights."""
    def __init__(self, registry: Mapping[str, Any], *, policy_ref: str = PROJECT_RIGHTS_POLICY_REF) -> None:
        self.registry = dict(registry)
        self.policy_ref = _require_text(policy_ref, "policy_ref")
        sources = validate_external_source_registry(registry)
        self.registry_identity_sha256 = _require_sha256(registry.get("registry_identity_sha256"), "registry_identity_sha256")
        self._sources = {(source.source_id, source.source_version): source for source in sources}

    def resolve(self, source_id: str, source_version: str, source_manifest_sha256: str | None = None) -> EligibilityDecision:
        key = (_require_text(source_id, "source_id"), _require_text(source_version, "source_version"))
        source = self._sources.get(key)
        if source is None:
            raise ExternalDataContractError(f"unregistered source identity: {key[0]}@{key[1]}")
        expected_manifest = source.source_manifest_sha256
        reasons: list[str] = []
        if source_manifest_sha256 is not None:
            supplied = _require_sha256(source_manifest_sha256, "source_manifest_sha256")
            if supplied != expected_manifest:
                reasons.append("SOURCE_VERSION_OR_MANIFEST_DRIFT")

        uses = source.rights.uses
        if uses is None:
            states = {field: USE_UNKNOWN for field in _USE_FIELDS}
            reasons.append("MISSING_MACHINE_READABLE_USE_DECISIONS")
        else:
            states = uses.to_dict()

        if source.purpose.strip().lower() not in _TRAIN_PURPOSES:
            reasons.append("NON_TRAINING_SOURCE_PURPOSE")
        if source.benchmark_material or source.held_out:
            reasons.append("RESERVED_OR_HELD_OUT")
        if source.rights.policy_ref != self.policy_ref:
            reasons.append("POLICY_REF_MISMATCH")
        if source.rights.status != RIGHTS_APPROVED:
            reasons.append("LEGACY_RIGHTS_STATUS_NOT_APPROVED")
        if source.rights.allows_model_training is not True:
            reasons.append("LEGACY_MODEL_TRAINING_FLAG_NOT_TRUE")
        if source.rights.license_id.upper() == "NOASSERTION":
            reasons.append("UNRESOLVED_LICENSE_ASSERTION")

        if uses is not None:
            if source.rights.allows_model_training != (uses.model_training == USE_ALLOWED):
                reasons.append("CONFLICTING_MODEL_TRAINING_METADATA")
            if source.rights.allows_redistribution != (uses.redistribution == USE_ALLOWED):
                reasons.append("CONFLICTING_REDISTRIBUTION_METADATA")
            for field in ("acquisition", "storage", "analysis", "model_training"):
                if states[field] != USE_ALLOWED:
                    reasons.append(f"{field.upper()}_NOT_EXPLICITLY_ALLOWED")

        evidence = source.rights.evidence_refs
        if not evidence:
            reasons.append("MISSING_IMMUTABLE_RIGHTS_EVIDENCE")
        else:
            for item in evidence:
                if item.source_id != source.source_id or item.source_version != source.source_version:
                    reasons.append("EVIDENCE_SOURCE_VERSION_MISMATCH")
                    break
            kinds = {item.evidence_kind for item in evidence}
            if "policy_decision" not in kinds:
                reasons.append("MISSING_POLICY_DECISION_EVIDENCE")
            if not (kinds & _RIGHTS_EVIDENCE_KINDS):
                reasons.append("MISSING_SOURCE_RIGHTS_EVIDENCE")

        reasons = sorted(set(reasons))
        eligible = not reasons and states["model_training"] == USE_ALLOWED
        return EligibilityDecision(
            source.source_id, source.source_version, expected_manifest,
            self.registry_identity_sha256,
            states["acquisition"], states["storage"], states["analysis"],
            states["model_training"], states["redistribution"], eligible,
            source.purpose, source.synthetic,
            tuple(sorted(item.evidence_id for item in evidence)), tuple(reasons),
        )

    def assert_model_training_eligible(self, source_id: str, source_version: str, source_manifest_sha256: str | None = None) -> EligibilityDecision:
        decision = self.resolve(source_id, source_version, source_manifest_sha256)
        if not decision.model_training_eligible:
            details = ", ".join(decision.reasons) or "model training not explicitly allowed"
            raise ExternalDataContractError(f"{source_id}@{source_version}: model-training eligibility denied: {details}")
        return decision

    def inventory(self) -> dict[str, Any]:
        decisions = [self.resolve(source.source_id, source.source_version) for source in self._sources.values()]
        decisions.sort(key=lambda x: (x.source_id, x.source_version))
        allowed = sum(item.model_training_eligible for item in decisions)
        review = sum(item.model_training in {USE_REVIEW_REQUIRED, USE_UNKNOWN} for item in decisions)
        core: dict[str, Any] = {
            "schema_version": ELIGIBILITY_INVENTORY_SCHEMA,
            "policy_ref": self.policy_ref,
            "registry_identity_sha256": self.registry_identity_sha256,
            "candidate_sources": len(decisions),
            "model_training_allowed": allowed,
            "model_training_blocked": len(decisions) - allowed,
            "unknown_or_review_required": review,
            "sources": [item.to_dict() for item in decisions],
        }
        return {**core, "inventory_sha256": _sha256_bytes(_canonical_json_bytes(core))}


@dataclass(frozen=True)
class ReservedSetSpec:
    set_id: str
    version: str
    source_id: str
    purpose: str
    normalized_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("set_id", "version", "source_id", "purpose"):
            _require_text(getattr(self, field), field)
        if self.purpose.strip().lower() not in _RESERVED_PURPOSES:
            raise ExternalDataContractError("reserved set purpose must be benchmark/evaluation/test")
        hashes = tuple(_require_sha256(value, "normalized_sha256") for value in self.normalized_sha256)
        if len(set(hashes)) != len(hashes):
            raise ExternalDataContractError("reserved set contains duplicate fingerprints")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["normalized_sha256"] = list(self.normalized_sha256)
        return value


def _rights_from_mapping(data: Mapping[str, Any]) -> RightsDecision:
    known = {"status", "license_id", "terms_url", "allows_model_training", "allows_derivatives", "allows_redistribution", "policy_ref", "reviewed_at", "reviewer_ref", "uses", "evidence_refs"}
    unknown = set(data) - known
    if unknown:
        raise ExternalDataContractError(f"unknown rights fields: {sorted(unknown)}")
    uses_raw = data.get("uses")
    if uses_raw is not None and not isinstance(uses_raw, Mapping):
        raise ExternalDataContractError("rights.uses must be an object or null")
    try:
        uses = None if uses_raw is None else UsePermissions(**dict(uses_raw))
    except TypeError as exc:
        raise ExternalDataContractError("rights.uses requires all five use dimensions") from exc
    evidence_raw = data.get("evidence_refs", [])
    if not isinstance(evidence_raw, list):
        raise ExternalDataContractError("rights.evidence_refs must be an array")
    evidence: list[RightsEvidenceRef] = []
    for item in evidence_raw:
        if not isinstance(item, Mapping):
            raise ExternalDataContractError("rights.evidence_refs entries must be objects")
        try:
            evidence.append(RightsEvidenceRef(**dict(item)))
        except TypeError as exc:
            raise ExternalDataContractError("rights evidence entry is incomplete or ambiguous") from exc
    return RightsDecision(
        status=data.get("status"), license_id=data.get("license_id"), terms_url=data.get("terms_url"),
        allows_model_training=data.get("allows_model_training"), allows_derivatives=data.get("allows_derivatives"),
        allows_redistribution=data.get("allows_redistribution"), policy_ref=data.get("policy_ref"),
        reviewed_at=data.get("reviewed_at"), reviewer_ref=data.get("reviewer_ref"),
        uses=uses, evidence_refs=tuple(evidence),
    )


def external_source_from_mapping(data: Mapping[str, Any]) -> ExternalSourceSpec:
    if not isinstance(data, Mapping):
        raise ExternalDataContractError("source entry must be an object")
    known = {"source_id", "source_version", "provider", "source_url", "source_kind", "purpose", "synthetic", "benchmark_material", "held_out", "snapshot", "rights"}
    unknown = set(data) - known
    if unknown:
        raise ExternalDataContractError(f"unknown source fields: {sorted(unknown)}")
    snapshot, rights = data.get("snapshot"), data.get("rights")
    if not isinstance(snapshot, Mapping) or not isinstance(rights, Mapping):
        raise ExternalDataContractError("source requires snapshot and rights objects")
    try:
        snapshot_spec = SnapshotSpec(**dict(snapshot))
    except TypeError as exc:
        raise ExternalDataContractError("snapshot entry is incomplete") from exc
    return ExternalSourceSpec(
        source_id=data.get("source_id"), source_version=data.get("source_version"), provider=data.get("provider"),
        source_url=data.get("source_url"), source_kind=data.get("source_kind"), purpose=data.get("purpose"),
        synthetic=data.get("synthetic"), benchmark_material=data.get("benchmark_material"), held_out=data.get("held_out"),
        snapshot=snapshot_spec, rights=_rights_from_mapping(rights),
    )


def build_external_source_registry(sources: Iterable[ExternalSourceSpec]) -> dict[str, Any]:
    entries = sorted((source.to_dict() for source in sources), key=lambda item: (item["source_id"], item["source_version"]))
    keys = [(item["source_id"], item["source_version"]) for item in entries]
    if len(keys) != len(set(keys)):
        raise ExternalDataContractError("duplicate source_id/source_version in registry")
    core = {"schema_version": REGISTRY_SCHEMA, "sources": entries}
    return {**core, "registry_identity_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def validate_external_source_registry(registry: Mapping[str, Any]) -> tuple[ExternalSourceSpec, ...]:
    schema = registry.get("schema_version")
    if schema not in {REGISTRY_SCHEMA, LEGACY_REGISTRY_SCHEMA}:
        raise ExternalDataContractError("unsupported external source registry schema")
    raw = registry.get("sources")
    if not isinstance(raw, list):
        raise ExternalDataContractError("external source registry sources must be an array")
    sources = tuple(external_source_from_mapping(item) for item in raw)
    ordered = tuple(sorted(sources, key=lambda x: (x.source_id, x.source_version)))
    if sources != ordered:
        raise ExternalDataContractError("external source registry is not canonical")
    if len({(x.source_id, x.source_version) for x in sources}) != len(sources):
        raise ExternalDataContractError("duplicate source_id/source_version in registry")
    if schema == LEGACY_REGISTRY_SCHEMA:
        core = {"schema_version": LEGACY_REGISTRY_SCHEMA, "sources": raw}
        expected = {**core, "registry_identity_sha256": _sha256_bytes(_canonical_json_bytes(core))}
    else:
        expected = build_external_source_registry(sources)
    if registry.get("registry_identity_sha256") != expected["registry_identity_sha256"]:
        raise ExternalDataContractError("external source registry identity mismatch")
    if dict(registry) != expected:
        raise ExternalDataContractError("external source registry is not canonical")
    return sources


def build_eligibility_inventory(registry: Mapping[str, Any]) -> dict[str, Any]:
    return EligibilityResolver(registry).inventory()


def build_reserved_fingerprint_registry(sets: Iterable[ReservedSetSpec]) -> dict[str, Any]:
    entries = sorted((item.to_dict() for item in sets), key=lambda item: (item["set_id"], item["version"]))
    keys = [(item["set_id"], item["version"]) for item in entries]
    if len(keys) != len(set(keys)):
        raise ExternalDataContractError("duplicate reserved set identity")
    core = {"schema_version": RESERVED_SCHEMA, "sets": entries}
    return {**core, "registry_identity_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def validate_reserved_fingerprint_registry(registry: Mapping[str, Any]) -> tuple[ReservedSetSpec, ...]:
    if registry.get("schema_version") != RESERVED_SCHEMA:
        raise ExternalDataContractError("unsupported reserved fingerprint registry schema")
    raw = registry.get("sets")
    if not isinstance(raw, list):
        raise ExternalDataContractError("reserved fingerprint registry sets must be an array")
    result: list[ReservedSetSpec] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ExternalDataContractError("reserved set entry must be an object")
        known = {"set_id", "version", "source_id", "purpose", "normalized_sha256"}
        if set(item) - known:
            raise ExternalDataContractError(f"unknown reserved set fields: {sorted(set(item)-known)}")
        hashes = item.get("normalized_sha256")
        if not isinstance(hashes, list):
            raise ExternalDataContractError("reserved normalized_sha256 must be an array")
        result.append(ReservedSetSpec(item.get("set_id"), item.get("version"), item.get("source_id"), item.get("purpose"), tuple(hashes)))
    expected = build_reserved_fingerprint_registry(result)
    if dict(registry) != expected:
        if registry.get("registry_identity_sha256") != expected["registry_identity_sha256"]:
            raise ExternalDataContractError("reserved fingerprint registry identity mismatch")
        raise ExternalDataContractError("reserved fingerprint registry is not canonical")
    return tuple(result)


def verify_local_snapshot(snapshot: SnapshotSpec, path: str | Path) -> None:
    payload = Path(path).read_bytes()
    if len(payload) != snapshot.size_bytes:
        raise ExternalDataContractError("snapshot size mismatch")
    if _sha256_bytes(payload) != snapshot.sha256:
        raise ExternalDataContractError("snapshot SHA-256 mismatch")


def contamination_report(training_records: Iterable[Mapping[str, Any]], reserved_registry: Mapping[str, Any]) -> dict[str, Any]:
    reserved_sets = validate_reserved_fingerprint_registry(reserved_registry)
    reserved_sources = {item.source_id for item in reserved_sets}
    reserved_hashes = {fingerprint for item in reserved_sets for fingerprint in item.normalized_sha256}
    source_collisions: list[str] = []
    content_collisions: list[str] = []
    checked = 0
    for record in training_records:
        if not isinstance(record, Mapping):
            raise ExternalDataContractError("training record must be an object")
        checked += 1
        source_id = _require_text(record.get("source_id"), "training.source_id")
        content_sha = _require_sha256(record.get("content_sha256"), "training.content_sha256")
        record_id = _require_text(record.get("id"), "training.id")
        if source_id in reserved_sources:
            source_collisions.append(record_id)
        if content_sha in reserved_hashes:
            content_collisions.append(record_id)
    core = {
        "schema_version": "12-6.contamination-report.v1",
        "checked_records": checked,
        "source_id_overlap_count": len(source_collisions),
        "content_sha256_overlap_count": len(content_collisions),
        "source_id_overlap_record_ids": sorted(source_collisions),
        "content_sha256_overlap_record_ids": sorted(content_collisions),
        "reserved_registry_identity_sha256": reserved_registry.get("registry_identity_sha256"),
    }
    return {**core, "report_sha256": _sha256_bytes(_canonical_json_bytes(core))}
