"""Fail-closed provenance and contamination contracts for external data sources."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

REGISTRY_SCHEMA = "12-6.external-source-registry.v1"
RESERVED_SCHEMA = "12-6.reserved-fingerprints.v1"
RIGHTS_APPROVED = "APPROVED_FOR_TRAINING"
RIGHTS_REJECTED = "REJECTED"
RIGHTS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
_ALLOWED_RIGHTS = frozenset({RIGHTS_APPROVED, RIGHTS_REJECTED, RIGHTS_REVIEW_REQUIRED})
_TRAIN_PURPOSES = frozenset({"pretraining", "training"})
_RESERVED_PURPOSES = frozenset({"benchmark", "evaluation", "validation", "test", "heldout_test"})
_SNAPSHOT_SCHEMES = frozenset({"file", "s3", "gs", "hf", "az", "r2"})


class ExternalDataContractError(ValueError):
    """Raised when an external-source provenance invariant is violated."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalDataContractError(f"{field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if text != text.lower():
        raise ExternalDataContractError(f"{field} must be lowercase SHA-256 hex")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ExternalDataContractError(f"{field} must be a lowercase SHA-256 hex digest")
    return text


def _validate_source_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "hf"}:
        raise ExternalDataContractError("source_url must use https or hf")
    if parsed.username or parsed.password:
        raise ExternalDataContractError("source_url must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ExternalDataContractError("source_url must not contain query parameters or fragments")


def _validate_snapshot_uri(value: str) -> None:
    if ".." in PurePosixPath(value.replace("\\", "/")).parts:
        raise ExternalDataContractError("snapshot.uri must not traverse parent directories")
    parsed = urlsplit(value)
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ExternalDataContractError(
            "snapshot.uri must be stable and contain no credentials/query"
        )
    if parsed.scheme and parsed.scheme not in _SNAPSHOT_SCHEMES:
        raise ExternalDataContractError(
            f"snapshot.uri scheme {parsed.scheme!r} is not an approved immutable-storage scheme"
        )


@dataclass(frozen=True)
class RightsDecision:
    """Human/policy decision about whether a source may enter model training."""

    status: str
    license_id: str
    terms_url: str
    allows_model_training: bool
    allows_derivatives: bool
    allows_redistribution: bool
    policy_ref: str
    reviewed_at: str
    reviewer_ref: str

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
            raise ExternalDataContractError(
                "APPROVED_FOR_TRAINING requires allows_model_training=true"
            )


@dataclass(frozen=True)
class SnapshotSpec:
    """Immutable byte snapshot reference stored outside Git."""

    uri: str
    sha256: str
    size_bytes: int
    retrieved_at: str
    upstream_version: str
    retrieval_method: str

    def __post_init__(self) -> None:
        _require_text(self.uri, "snapshot.uri")
        _validate_snapshot_uri(self.uri)
        _require_sha256(self.sha256, "snapshot.sha256")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ExternalDataContractError("snapshot.size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ExternalDataContractError("snapshot.size_bytes must be non-negative")
        for field in ("retrieved_at", "upstream_version", "retrieval_method"):
            _require_text(getattr(self, field), f"snapshot.{field}")


@dataclass(frozen=True)
class ExternalSourceSpec:
    """Versioned source record whose training eligibility is explicit and reviewable."""

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
            raise ExternalDataContractError(
                "benchmark/held-out material cannot declare a training purpose"
            )
        if purpose in _RESERVED_PURPOSES and not (self.benchmark_material or self.held_out):
            raise ExternalDataContractError(
                "reserved evaluation/test purposes must be marked benchmark_material or held_out"
            )

    def assert_training_eligible(self) -> None:
        """Fail closed unless the exact versioned source is approved for model training."""

        if self.purpose.strip().lower() not in _TRAIN_PURPOSES:
            raise ExternalDataContractError(f"{self.source_id}: source purpose is not training")
        if self.benchmark_material or self.held_out:
            raise ExternalDataContractError(
                f"{self.source_id}: benchmark/held-out source is reserved"
            )
        if self.rights.status != RIGHTS_APPROVED or not self.rights.allows_model_training:
            raise ExternalDataContractError(
                f"{self.source_id}: rights are not approved for training"
            )
        if self.rights.license_id.upper() == "NOASSERTION":
            raise ExternalDataContractError(f"{self.source_id}: license state is unresolved")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReservedSetSpec:
    """Versioned evaluation/test fingerprints that training data must not overlap."""

    set_id: str
    version: str
    source_id: str
    purpose: str
    normalized_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in ("set_id", "version", "source_id", "purpose"):
            _require_text(getattr(self, field), field)
        if self.purpose.strip().lower() not in _RESERVED_PURPOSES:
            raise ExternalDataContractError(
                "reserved set purpose must be benchmark/evaluation/test"
            )
        hashes = tuple(
            _require_sha256(value, "normalized_sha256")
            for value in self.normalized_sha256
        )
        if len(set(hashes)) != len(hashes):
            raise ExternalDataContractError("reserved set contains duplicate fingerprints")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["normalized_sha256"] = list(self.normalized_sha256)
        return result


def external_source_from_mapping(data: Mapping[str, Any]) -> ExternalSourceSpec:
    if not isinstance(data, Mapping):
        raise ExternalDataContractError("source entry must be an object")
    known = {
        "source_id",
        "source_version",
        "provider",
        "source_url",
        "source_kind",
        "purpose",
        "synthetic",
        "benchmark_material",
        "held_out",
        "snapshot",
        "rights",
    }
    unknown = set(data) - known
    if unknown:
        raise ExternalDataContractError(f"unknown source fields: {sorted(unknown)}")
    snapshot = data.get("snapshot")
    rights = data.get("rights")
    if not isinstance(snapshot, Mapping) or not isinstance(rights, Mapping):
        raise ExternalDataContractError("source requires snapshot and rights objects")
    return ExternalSourceSpec(
        source_id=data.get("source_id"),
        source_version=data.get("source_version"),
        provider=data.get("provider"),
        source_url=data.get("source_url"),
        source_kind=data.get("source_kind"),
        purpose=data.get("purpose"),
        synthetic=data.get("synthetic"),
        benchmark_material=data.get("benchmark_material"),
        held_out=data.get("held_out"),
        snapshot=SnapshotSpec(**dict(snapshot)),
        rights=RightsDecision(**dict(rights)),
    )


def build_external_source_registry(
    sources: Iterable[ExternalSourceSpec],
) -> dict[str, Any]:
    entries = sorted(
        (source.to_dict() for source in sources),
        key=lambda item: (item["source_id"], item["source_version"]),
    )
    keys = [(entry["source_id"], entry["source_version"]) for entry in entries]
    if len(set(keys)) != len(keys):
        raise ExternalDataContractError("duplicate source_id/source_version in registry")
    core = {"schema_version": REGISTRY_SCHEMA, "sources": entries}
    return {**core, "registry_identity_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def build_reserved_fingerprint_registry(
    sets: Iterable[ReservedSetSpec],
) -> dict[str, Any]:
    entries = sorted(
        (item.to_dict() for item in sets),
        key=lambda item: (item["set_id"], item["version"]),
    )
    keys = [(entry["set_id"], entry["version"]) for entry in entries]
    if len(set(keys)) != len(keys):
        raise ExternalDataContractError("duplicate reserved set identity")
    core = {"schema_version": RESERVED_SCHEMA, "sets": entries}
    return {**core, "registry_identity_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def validate_external_source_registry(
    registry: Mapping[str, Any],
) -> tuple[ExternalSourceSpec, ...]:
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise ExternalDataContractError("unsupported external source registry schema")
    raw_sources = registry.get("sources")
    if not isinstance(raw_sources, list):
        raise ExternalDataContractError("external source registry sources must be an array")
    sources = tuple(external_source_from_mapping(item) for item in raw_sources)
    expected = build_external_source_registry(sources)
    if registry.get("registry_identity_sha256") != expected["registry_identity_sha256"]:
        raise ExternalDataContractError("external source registry identity mismatch")
    if dict(registry) != expected:
        raise ExternalDataContractError("external source registry is not canonical")
    return sources


def validate_reserved_fingerprint_registry(
    registry: Mapping[str, Any],
) -> tuple[ReservedSetSpec, ...]:
    if registry.get("schema_version") != RESERVED_SCHEMA:
        raise ExternalDataContractError("unsupported reserved fingerprint registry schema")
    raw_sets = registry.get("sets")
    if not isinstance(raw_sets, list):
        raise ExternalDataContractError("reserved fingerprint registry sets must be an array")
    sets: list[ReservedSetSpec] = []
    for item in raw_sets:
        if not isinstance(item, Mapping):
            raise ExternalDataContractError("reserved set entry must be an object")
        known = {"set_id", "version", "source_id", "purpose", "normalized_sha256"}
        unknown = set(item) - known
        if unknown:
            raise ExternalDataContractError(f"unknown reserved set fields: {sorted(unknown)}")
        hashes = item.get("normalized_sha256")
        if not isinstance(hashes, list):
            raise ExternalDataContractError("reserved normalized_sha256 must be an array")
        sets.append(
            ReservedSetSpec(
                set_id=item.get("set_id"),
                version=item.get("version"),
                source_id=item.get("source_id"),
                purpose=item.get("purpose"),
                normalized_sha256=tuple(hashes),
            )
        )
    result = tuple(sets)
    expected = build_reserved_fingerprint_registry(result)
    if registry.get("registry_identity_sha256") != expected["registry_identity_sha256"]:
        raise ExternalDataContractError("reserved fingerprint registry identity mismatch")
    if dict(registry) != expected:
        raise ExternalDataContractError("reserved fingerprint registry is not canonical")
    return result


def verify_local_snapshot(snapshot: SnapshotSpec, path: str | Path) -> None:
    candidate = Path(path)
    payload = candidate.read_bytes()
    if len(payload) != snapshot.size_bytes:
        raise ExternalDataContractError("snapshot size mismatch")
    if _sha256_bytes(payload) != snapshot.sha256:
        raise ExternalDataContractError("snapshot SHA-256 mismatch")


def contamination_report(
    training_records: Iterable[Mapping[str, Any]],
    reserved_registry: Mapping[str, Any],
) -> dict[str, Any]:
    reserved_sets = validate_reserved_fingerprint_registry(reserved_registry)
    reserved_sources = {item.source_id for item in reserved_sets}
    reserved_hashes = {
        fingerprint
        for item in reserved_sets
        for fingerprint in item.normalized_sha256
    }

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

    result = {
        "schema_version": "12-6.contamination-report.v1",
        "checked_records": checked,
        "source_id_overlap_count": len(source_collisions),
        "content_sha256_overlap_count": len(content_collisions),
        "source_id_overlap_record_ids": sorted(source_collisions),
        "content_sha256_overlap_record_ids": sorted(content_collisions),
        "reserved_registry_identity_sha256": reserved_registry.get("registry_identity_sha256"),
    }
    core = dict(result)
    result["report_sha256"] = _sha256_bytes(_canonical_json_bytes(core))
    return result
