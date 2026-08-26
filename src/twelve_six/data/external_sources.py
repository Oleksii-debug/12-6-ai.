"""Fail-closed contracts for locally materialized external training sources.

The corpus builder never downloads network data. Registry entries are declarative evidence that a
specific local artifact was reviewed for training use, then bound by SHA-256 before ingestion.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTRY_SCHEMA = "12-6.external-source-registry.v1"
ALLOWED_STRATA = frozenset({"uk", "en", "code"})
HEX_DIGITS = frozenset("0123456789abcdef")


class ExternalSourceContractError(ValueError):
    """Raised when an external source is not explicitly safe to enter training."""


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in HEX_DIGITS for char in value)
    )


def _require_nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExternalSourceContractError(f"{label} must be a non-empty string")
    return value


def _repo_path(repo_root: Path, raw: str, *, label: str) -> Path:
    root = repo_root.resolve()
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ExternalSourceContractError(f"{label} escapes repository: {raw}") from exc
    return candidate


@dataclass(frozen=True)
class ExternalSource:
    source_id: str
    source_version: str
    stratum: str
    training_eligible: bool
    license: dict[str, Any]
    provenance: dict[str, Any]
    rights: dict[str, Any]
    materialization: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExternalSource":
        source_id = _require_nonempty_string(raw.get("source_id"), label="source_id")
        source_version = _require_nonempty_string(
            raw.get("source_version"), label=f"{source_id}: source_version"
        )
        stratum = _require_nonempty_string(raw.get("stratum"), label=f"{source_id}: stratum")
        if stratum not in ALLOWED_STRATA:
            raise ExternalSourceContractError(
                f"{source_id}: stratum must be one of {sorted(ALLOWED_STRATA)}"
            )
        training_eligible = raw.get("training_eligible")
        if not isinstance(training_eligible, bool):
            raise ExternalSourceContractError(
                f"{source_id}: training_eligible must be explicit boolean"
            )
        license_meta = raw.get("license")
        provenance = raw.get("provenance")
        rights = raw.get("rights")
        materialization = raw.get("materialization")
        for label, value in (
            ("license", license_meta),
            ("provenance", provenance),
            ("rights", rights),
            ("materialization", materialization),
        ):
            if not isinstance(value, dict):
                raise ExternalSourceContractError(f"{source_id}: {label} must be an object")
        return cls(
            source_id=source_id,
            source_version=source_version,
            stratum=stratum,
            training_eligible=training_eligible,
            license=dict(license_meta),
            provenance=dict(provenance),
            rights=dict(rights),
            materialization=dict(materialization),
        )

    def assert_training_eligible(self) -> None:
        if not self.training_eligible:
            raise ExternalSourceContractError(
                f"{self.source_id}: source is not marked training eligible"
            )
        if self.provenance.get("external_source") is not True:
            raise ExternalSourceContractError(
                f"{self.source_id}: provenance.external_source must be true"
            )
        if self.provenance.get("benchmark_material") is not False:
            raise ExternalSourceContractError(
                f"{self.source_id}: benchmark_material must be explicitly false"
            )
        _require_nonempty_string(
            self.provenance.get("source_url"),
            label=f"{self.source_id}: provenance.source_url",
        )
        _require_nonempty_string(
            self.license.get("identifier"),
            label=f"{self.source_id}: license.identifier",
        )
        if self.license.get("review_status") != "approved":
            raise ExternalSourceContractError(
                f"{self.source_id}: license.review_status must be approved"
            )
        if self.rights.get("training_use") != "allowed":
            raise ExternalSourceContractError(
                f"{self.source_id}: rights.training_use must be allowed"
            )
        if self.rights.get("review_status") != "approved":
            raise ExternalSourceContractError(
                f"{self.source_id}: rights.review_status must be approved"
            )
        _require_nonempty_string(
            self.rights.get("reviewed_by"),
            label=f"{self.source_id}: rights.reviewed_by",
        )
        if self.materialization.get("format") != "jsonl":
            raise ExternalSourceContractError(
                f"{self.source_id}: materialization.format must be jsonl"
            )
        _require_nonempty_string(
            self.materialization.get("path"),
            label=f"{self.source_id}: materialization.path",
        )
        if not _is_sha256(self.materialization.get("sha256")):
            raise ExternalSourceContractError(
                f"{self.source_id}: materialization.sha256 must be lowercase SHA-256"
            )
        _require_nonempty_string(
            self.materialization.get("text_field", "text"),
            label=f"{self.source_id}: materialization.text_field",
        )
        _require_nonempty_string(
            self.materialization.get("document_id_field", "document_id"),
            label=f"{self.source_id}: materialization.document_id_field",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_version": self.source_version,
            "stratum": self.stratum,
            "training_eligible": self.training_eligible,
            "license": dict(self.license),
            "provenance": dict(self.provenance),
            "rights": dict(self.rights),
            "materialization": dict(self.materialization),
        }


def validate_external_source_registry(
    registry: Mapping[str, Any],
) -> tuple[ExternalSource, ...]:
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise ExternalSourceContractError("unsupported external source registry schema")
    raw_sources = registry.get("sources")
    if not isinstance(raw_sources, list):
        raise ExternalSourceContractError("external source registry sources must be a list")

    sources: list[ExternalSource] = []
    identities: set[tuple[str, str]] = set()
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            raise ExternalSourceContractError("external source entry must be an object")
        source = ExternalSource.from_mapping(raw)
        identity = (source.source_id, source.source_version)
        if identity in identities:
            raise ExternalSourceContractError(
                f"duplicate external source identity: {source.source_id}@{source.source_version}"
            )
        identities.add(identity)
        sources.append(source)
    return tuple(sorted(sources, key=lambda item: (item.source_id, item.source_version)))


def iter_materialized_records(
    repo_root: Path,
    source: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """Yield hash-bound JSONL rows from one already-authorized external source."""
    parsed = ExternalSource.from_mapping(source)
    parsed.assert_training_eligible()
    materialization = parsed.materialization
    path = _repo_path(
        repo_root,
        str(materialization["path"]),
        label=f"{parsed.source_id}: materialization.path",
    )
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ExternalSourceContractError(
            f"{parsed.source_id}: materialized artifact is unavailable: {path}"
        ) from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != materialization["sha256"]:
        raise ExternalSourceContractError(
            f"{parsed.source_id}: materialized artifact hash mismatch"
        )
    try:
        text_payload = payload.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise ExternalSourceContractError(
            f"{parsed.source_id}: materialized artifact must be UTF-8"
        ) from exc

    text_field = str(materialization.get("text_field", "text"))
    document_id_field = str(materialization.get("document_id_field", "document_id"))
    seen_ids: set[str] = set()
    for line_number, line in enumerate(text_payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExternalSourceContractError(
                f"{parsed.source_id}:{line_number}: invalid JSONL"
            ) from exc
        if not isinstance(row, dict):
            raise ExternalSourceContractError(
                f"{parsed.source_id}:{line_number}: JSONL row must be an object"
            )
        document_id = _require_nonempty_string(
            row.get(document_id_field),
            label=f"{parsed.source_id}:{line_number}: {document_id_field}",
        )
        if document_id in seen_ids:
            raise ExternalSourceContractError(
                f"{parsed.source_id}: duplicate document_id: {document_id}"
            )
        seen_ids.add(document_id)
        text = row.get(text_field)
        if not isinstance(text, str):
            raise ExternalSourceContractError(
                f"{parsed.source_id}:{document_id}: {text_field} must be a string"
            )
        identity_payload = (
            f"{parsed.source_id}\0{parsed.source_version}\0{document_id}\0{text}".encode(
                "utf-8"
            )
        )
        digest = hashlib.sha256(identity_payload).hexdigest()[:24]
        yield {
            "record_id": f"external:{parsed.source_id}:{digest}",
            "source_id": parsed.source_id,
            "source_version": parsed.source_version,
            "stratum": parsed.stratum,
            "external": True,
            "project_authored": False,
            "raw_text": text,
        }
