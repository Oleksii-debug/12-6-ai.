"""Static Plan-1 environment/backend declarations, never Plan-8 acceptance authority.

A SUPPORTED declaration describes a compatible *contract surface*, not a physical
capability PASS. Every lookup remains UNQUALIFIED until Plan 8 independently
proves readiness. Unknown records fail closed.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twelve_six.integration.dependency_lock import (
    EXACT_PYTHON_VERSION,
    SUPPORTED_PROFILES,
    validate_lock_index,
)

SCHEMA_VERSION = "12-6.static-backend-compatibility.v1"
MAX_MATRIX_BYTES = 65536
_STATUSES = frozenset({"supported", "experimental", "unsupported"})
_TOKEN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = {"profile", "backend", "feature", "status", "evidence_ref"}


class CompatibilityError(ValueError):
    """Invalid, untrusted, or out-of-version static compatibility matrix."""


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CompatibilityError("duplicate matrix JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise CompatibilityError("nonfinite matrix JSON value")


def _trusted_file(root: Path, relative: str) -> Path:
    if type(relative) is not str or not relative or chr(92) in relative:
        raise CompatibilityError("unsafe matrix path")
    p = Path(relative)
    if p.is_absolute() or any(s in {".", ".."} for s in p.parts):
        raise CompatibilityError("matrix path traversal")
    root = root.resolve(strict=True)
    child = root
    for part in p.parts:
        child = child / part
        if child.is_symlink():
            raise CompatibilityError("symlink matrix path")
    if not child.is_file() or not child.resolve(strict=True).is_relative_to(root):
        raise CompatibilityError("matrix file missing or outside root")
    return child


@dataclass(frozen=True, slots=True)
class CompatibilityEntry:
    profile: str
    backend: str
    feature: str
    status: str
    evidence_ref: str | None

    def validate(self) -> None:
        if type(self.profile) is not str or self.profile not in SUPPORTED_PROFILES:
            raise CompatibilityError("unrecognized locked environment profile")
        for name in ("backend", "feature"):
            value = getattr(self, name)
            if type(value) is not str or _TOKEN.fullmatch(value) is None:
                raise CompatibilityError(f"noncanonical {name}")
        if type(self.status) is not str or self.status not in _STATUSES:
            raise CompatibilityError("unsupported static compatibility status")
        if self.evidence_ref is not None and (
            type(self.evidence_ref) is not str or _SHA.fullmatch(self.evidence_ref) is None
        ):
            raise CompatibilityError("invalid optional evidence identity")


@dataclass(frozen=True, slots=True)
class CompatibilityDecision:
    """A non-authoritative descriptor, not an acceptance/readiness verdict."""

    status: str
    qualification: str
    evidence_ref: str | None


@dataclass(frozen=True, slots=True)
class StaticBackendMatrix:
    schema_version: str
    python_version: str
    lock_index_sha256: str
    entries: tuple[CompatibilityEntry, ...]

    def validate(self, *, expected_lock_sha256: str) -> None:
        if self.schema_version != SCHEMA_VERSION or type(self.schema_version) is not str:
            raise CompatibilityError("unsupported static matrix schema")
        if self.python_version != EXACT_PYTHON_VERSION or type(self.python_version) is not str:
            raise CompatibilityError("static matrix Python version mismatch")
        if type(self.lock_index_sha256) is not str or _SHA.fullmatch(
            self.lock_index_sha256
        ) is None or self.lock_index_sha256 != expected_lock_sha256:
            raise CompatibilityError("dependency lock binding mismatch")
        if type(self.entries) is not tuple:
            raise CompatibilityError("entries must be immutable")
        for entry in self.entries:
            if type(entry) is not CompatibilityEntry:
                raise CompatibilityError("invalid entry implementation")
            entry.validate()
        keys = [(e.profile, e.backend, e.feature) for e in self.entries]
        if keys != sorted(set(keys)):
            raise CompatibilityError("duplicate or unordered compatibility entries")
        supported_lock_profiles = {
            e.profile for e in self.entries
            if e.backend == "python" and e.feature == "dependency_environment"
            and e.status == "supported"
        }
        if supported_lock_profiles != SUPPORTED_PROFILES:
            raise CompatibilityError("missing baseline dependency environment profile")

    def lookup(
        self, profile: object, backend: object, feature: object,
        *, expected_lock_sha256: str,
    ) -> CompatibilityDecision:
        self.validate(expected_lock_sha256=expected_lock_sha256)
        if any(type(v) is not str for v in (profile, backend, feature)):
            return CompatibilityDecision("unsupported", "UNQUALIFIED", None)
        for entry in self.entries:
            if (entry.profile, entry.backend, entry.feature) == (profile, backend, feature):
                return CompatibilityDecision(entry.status, "UNQUALIFIED", entry.evidence_ref)
        return CompatibilityDecision("unsupported", "UNQUALIFIED", None)


def parse_static_backend_matrix(
    raw: bytes, *, expected_lock_sha256: str,
) -> StaticBackendMatrix:
    if type(raw) is not bytes or len(raw) > MAX_MATRIX_BYTES:
        raise CompatibilityError("matrix must be bounded bytes")
    try:
        document = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise CompatibilityError("matrix is not strict UTF-8 JSON") from exc
    if type(document) is not dict or set(document) != {
        "schema_version", "python_version", "lock_index_sha256", "entries"
    }:
        raise CompatibilityError("unexpected static matrix fields")
    records = document["entries"]
    if type(records) is not list:
        raise CompatibilityError("matrix entries must be a list")
    entries = []
    for record in records:
        if type(record) is not dict or set(record) != _FIELDS:
            raise CompatibilityError("unexpected compatibility entry fields")
        entry = CompatibilityEntry(**record)
        entries.append(entry)
    matrix = StaticBackendMatrix(
        schema_version=document["schema_version"],
        python_version=document["python_version"],
        lock_index_sha256=document["lock_index_sha256"],
        entries=tuple(entries),
    )
    matrix.validate(expected_lock_sha256=expected_lock_sha256)
    return matrix


def load_static_backend_matrix(
    *, root: str | Path,
    matrix_path: str = "configs/compatibility/plan1_backend_matrix_v1.json",
) -> StaticBackendMatrix:
    """Bind static declarations to existing *fully verified* Plan-1 lock authority."""
    root_path = Path(root)
    index = validate_lock_index(
        root=root_path,
        index_path="requirements/locks/index.json",
    )
    raw = _trusted_file(root_path, matrix_path).read_bytes()
    return parse_static_backend_matrix(
        raw, expected_lock_sha256=index["index_sha256"],
    )


def matrix_blob_sha256(raw: bytes) -> str:
    if type(raw) is not bytes:
        raise CompatibilityError("matrix digest input must be bytes")
    return hashlib.sha256(raw).hexdigest()
