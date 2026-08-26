"""Fail-closed source acquisition receipts and bounded-memory resume mechanics."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import unquote, urlsplit

from .external_sources import validate_external_source_registry

RETRIEVAL_PLAN_SCHEMA = "12-6.source-retrieval-plan.v1"
RETRIEVAL_CHECKPOINT_SCHEMA = "12-6.source-retrieval-checkpoint.v1"
RETRIEVAL_RECEIPT_SCHEMA = "12-6.source-retrieval-receipt.v1"
RETRIEVAL_INVENTORY_SCHEMA = "12-6.source-retrieval-inventory.v1"
_ALLOWED_STORAGE_SCHEMES = frozenset({"file", "https", "s3", "gs", "hf", "az", "r2"})


class SourceAcquisitionError(ValueError):
    """Raised when source acquisition evidence is unsafe or inconsistent."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SourceAcquisitionError(f"{field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if text != text.lower() or len(text) != 64:
        raise SourceAcquisitionError(f"{field} must be lowercase SHA-256 hex")
    if any(char not in "0123456789abcdef" for char in text):
        raise SourceAcquisitionError(f"{field} must be lowercase SHA-256 hex")
    return text


def _require_non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceAcquisitionError(f"{field} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    result = _require_non_negative_int(value, field)
    if result == 0:
        raise SourceAcquisitionError(f"{field} must be positive")
    return result


def _validate_stable_uri(value: str, field: str) -> None:
    text = _require_text(value, field)
    parsed = urlsplit(text)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SourceAcquisitionError(f"{field} must not contain credentials/query/fragment")
    if parsed.scheme and parsed.scheme not in _ALLOWED_STORAGE_SCHEMES:
        raise SourceAcquisitionError(f"{field} uses unsupported URI scheme {parsed.scheme!r}")


def _path_from_file_uri(uri: str, field: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme != "file":
        raise SourceAcquisitionError(f"{field} must be a file:// URI for local execution")
    if parsed.netloc not in {"", "localhost"}:
        raise SourceAcquisitionError(f"{field} file:// URI must be local")
    return Path(unquote(parsed.path)).resolve()


@dataclass(frozen=True)
class SourceRetrievalPlan:
    """Exact source-version retrieval plan bound to immutable registry expectations."""

    source_registry_identity_sha256: str
    source_id: str
    source_version: str
    source_uri: str
    destination_uri: str
    expected_sha256: str
    expected_size_bytes: int
    upstream_version: str
    retrieval_method: str
    rights_status_observed: str
    chunk_size_bytes: int = 1024 * 1024
    max_inflight_chunks: int = 1

    def __post_init__(self) -> None:
        _require_sha256(
            self.source_registry_identity_sha256,
            "source_registry_identity_sha256",
        )
        for field in (
            "source_id",
            "source_version",
            "upstream_version",
            "retrieval_method",
            "rights_status_observed",
        ):
            _require_text(getattr(self, field), field)
        _validate_stable_uri(self.source_uri, "source_uri")
        _validate_stable_uri(self.destination_uri, "destination_uri")
        if self.source_uri == self.destination_uri:
            raise SourceAcquisitionError("source_uri and destination_uri must differ")
        _require_sha256(self.expected_sha256, "expected_sha256")
        _require_non_negative_int(self.expected_size_bytes, "expected_size_bytes")
        _require_positive_int(self.chunk_size_bytes, "chunk_size_bytes")
        if self.max_inflight_chunks != 1:
            raise SourceAcquisitionError("max_inflight_chunks must remain 1 for bounded-memory v1")

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": RETRIEVAL_PLAN_SCHEMA,
            **asdict(self),
            "rights_semantics": "OBSERVED_ONLY_NOT_APPROVAL",
            "training_eligibility_evaluated": False,
            "memory_contract": "one_chunk_plus_hash_state",
        }
        return {**core, "plan_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def plan_from_registered_source(
    source_registry: Mapping[str, Any],
    source_id: str,
    source_version: str,
    destination_uri: str,
    *,
    chunk_size_bytes: int = 1024 * 1024,
) -> SourceRetrievalPlan:
    """Bind acquisition mechanics to one exact registered source version without rights promotion."""

    requested_id = _require_text(source_id, "source_id")
    requested_version = _require_text(source_version, "source_version")
    sources = validate_external_source_registry(source_registry)
    matches = [
        source
        for source in sources
        if source.source_id == requested_id and source.source_version == requested_version
    ]
    if len(matches) != 1:
        raise SourceAcquisitionError(
            f"registered source version not found exactly once: {requested_id}@{requested_version}"
        )
    source = matches[0]
    return SourceRetrievalPlan(
        source_registry_identity_sha256=_require_sha256(
            source_registry.get("registry_identity_sha256"),
            "registry_identity_sha256",
        ),
        source_id=source.source_id,
        source_version=source.source_version,
        source_uri=source.snapshot.uri,
        destination_uri=destination_uri,
        expected_sha256=source.snapshot.sha256,
        expected_size_bytes=source.snapshot.size_bytes,
        upstream_version=source.snapshot.upstream_version,
        retrieval_method=source.snapshot.retrieval_method,
        rights_status_observed=source.rights.status,
        chunk_size_bytes=chunk_size_bytes,
    )


@dataclass(frozen=True)
class RetrievedChunk:
    index: int
    offset_bytes: int
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require_non_negative_int(self.index, "index")
        _require_non_negative_int(self.offset_bytes, "offset_bytes")
        _require_positive_int(self.size_bytes, "size_bytes")
        _require_sha256(self.sha256, "sha256")


@dataclass(frozen=True)
class RetrievalCheckpoint:
    """Contiguous completed-chunk evidence for a partial local destination."""

    plan_sha256: str
    expected_size_bytes: int
    chunk_size_bytes: int
    completed: tuple[RetrievedChunk, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.plan_sha256, "plan_sha256")
        _require_non_negative_int(self.expected_size_bytes, "expected_size_bytes")
        _require_positive_int(self.chunk_size_bytes, "chunk_size_bytes")
        expected_offset = 0
        for expected_index, chunk in enumerate(self.completed):
            if chunk.index != expected_index:
                raise SourceAcquisitionError("retrieval checkpoint chunk indexes must be contiguous")
            if chunk.offset_bytes != expected_offset:
                raise SourceAcquisitionError("retrieval checkpoint chunk offsets must be contiguous")
            expected_offset += chunk.size_bytes
            if expected_offset > self.expected_size_bytes:
                raise SourceAcquisitionError("retrieval checkpoint exceeds expected source size")
            is_complete = expected_offset == self.expected_size_bytes
            if not is_complete and chunk.size_bytes != self.chunk_size_bytes:
                raise SourceAcquisitionError(
                    "partial retrieval checkpoint must end on a full chunk boundary"
                )
        if self.completed and expected_offset < self.expected_size_bytes:
            if self.completed[-1].size_bytes != self.chunk_size_bytes:
                raise SourceAcquisitionError(
                    "partial retrieval checkpoint must end on a full chunk boundary"
                )

    @property
    def next_offset_bytes(self) -> int:
        return sum(chunk.size_bytes for chunk in self.completed)

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": RETRIEVAL_CHECKPOINT_SCHEMA,
            "plan_sha256": self.plan_sha256,
            "expected_size_bytes": self.expected_size_bytes,
            "chunk_size_bytes": self.chunk_size_bytes,
            "completed": [asdict(chunk) for chunk in self.completed],
            "next_offset_bytes": self.next_offset_bytes,
        }
        return {**core, "checkpoint_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def build_local_checkpoint(plan: SourceRetrievalPlan, partial_path: str | Path) -> RetrievalCheckpoint:
    """Hash an existing partial destination in bounded memory and return resume evidence."""

    path = Path(partial_path)
    completed: list[RetrievedChunk] = []
    offset = 0
    with path.open("rb") as handle:
        index = 0
        while True:
            chunk = handle.read(plan.chunk_size_bytes)
            if not chunk:
                break
            completed.append(
                RetrievedChunk(
                    index=index,
                    offset_bytes=offset,
                    size_bytes=len(chunk),
                    sha256=_sha256_bytes(chunk),
                )
            )
            offset += len(chunk)
            index += 1
    return RetrievalCheckpoint(
        plan_sha256=plan.manifest()["plan_sha256"],
        expected_size_bytes=plan.expected_size_bytes,
        chunk_size_bytes=plan.chunk_size_bytes,
        completed=tuple(completed),
    )


def assert_checkpoint_compatible(
    plan: SourceRetrievalPlan,
    checkpoint: RetrievalCheckpoint,
) -> None:
    if checkpoint.plan_sha256 != plan.manifest()["plan_sha256"]:
        raise SourceAcquisitionError("retrieval checkpoint is bound to a different plan")
    if checkpoint.expected_size_bytes != plan.expected_size_bytes:
        raise SourceAcquisitionError("retrieval checkpoint expected size differs from plan")
    if checkpoint.chunk_size_bytes != plan.chunk_size_bytes:
        raise SourceAcquisitionError("retrieval checkpoint chunk size differs from plan")


def _verify_checkpoint_against_source(
    source: BinaryIO,
    checkpoint: RetrievalCheckpoint,
) -> None:
    source.seek(0)
    for completed in checkpoint.completed:
        payload = source.read(completed.size_bytes)
        if len(payload) != completed.size_bytes or _sha256_bytes(payload) != completed.sha256:
            raise SourceAcquisitionError("partial destination does not match source prefix")


@dataclass(frozen=True)
class VerifiedRetrievalReceipt:
    """Deterministic byte-verification receipt. It explicitly does not approve training rights."""

    plan_sha256: str
    source_registry_identity_sha256: str
    source_id: str
    source_version: str
    source_uri: str
    destination_uri: str
    expected_sha256: str
    verified_sha256: str
    verified_size_bytes: int
    chunk_count: int
    chunk_manifest_sha256: str
    rights_status_observed: str

    def __post_init__(self) -> None:
        for field in (
            "plan_sha256",
            "source_registry_identity_sha256",
            "expected_sha256",
            "verified_sha256",
            "chunk_manifest_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        for field in ("source_id", "source_version", "rights_status_observed"):
            _require_text(getattr(self, field), field)
        _validate_stable_uri(self.source_uri, "source_uri")
        _validate_stable_uri(self.destination_uri, "destination_uri")
        _require_non_negative_int(self.verified_size_bytes, "verified_size_bytes")
        _require_non_negative_int(self.chunk_count, "chunk_count")
        if self.expected_sha256 != self.verified_sha256:
            raise SourceAcquisitionError("retrieval receipt expected/verified SHA-256 mismatch")

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": RETRIEVAL_RECEIPT_SCHEMA,
            **asdict(self),
            "verification": "PASS",
            "rights_semantics": "OBSERVED_ONLY_NOT_APPROVAL",
            "training_eligibility_evaluated": False,
        }
        return {**core, "receipt_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def _build_receipt(
    plan: SourceRetrievalPlan,
    chunks: Iterable[RetrievedChunk],
    verified_sha256: str,
    verified_size_bytes: int,
) -> VerifiedRetrievalReceipt:
    chunk_list = list(chunks)
    chunk_core = [asdict(chunk) for chunk in chunk_list]
    return VerifiedRetrievalReceipt(
        plan_sha256=plan.manifest()["plan_sha256"],
        source_registry_identity_sha256=plan.source_registry_identity_sha256,
        source_id=plan.source_id,
        source_version=plan.source_version,
        source_uri=plan.source_uri,
        destination_uri=plan.destination_uri,
        expected_sha256=plan.expected_sha256,
        verified_sha256=verified_sha256,
        verified_size_bytes=verified_size_bytes,
        chunk_count=len(chunk_list),
        chunk_manifest_sha256=_sha256_bytes(_canonical_json_bytes(chunk_core)),
        rights_status_observed=plan.rights_status_observed,
    )


def verify_and_stage_local_mirror(
    plan: SourceRetrievalPlan,
    source_path: str | Path,
    *,
    resume: bool = True,
) -> VerifiedRetrievalReceipt:
    """Copy a local mirror in bounded memory and atomically publish only exact registered bytes."""

    source = Path(source_path).resolve()
    destination = _path_from_file_uri(plan.destination_uri, "destination_uri")
    partial = destination.with_name(destination.name + ".partial")
    if source == destination or source == partial:
        raise SourceAcquisitionError("local source and destination/partial paths must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        raise SourceAcquisitionError("destination already exists; refuse silent overwrite")
    if partial.exists() and not resume:
        raise SourceAcquisitionError("partial destination exists but resume=false")

    checkpoint = (
        build_local_checkpoint(plan, partial)
        if partial.exists()
        else RetrievalCheckpoint(
            plan_sha256=plan.manifest()["plan_sha256"],
            expected_size_bytes=plan.expected_size_bytes,
            chunk_size_bytes=plan.chunk_size_bytes,
            completed=(),
        )
    )
    assert_checkpoint_compatible(plan, checkpoint)

    chunks = list(checkpoint.completed)
    total = checkpoint.next_offset_bytes
    digest = hashlib.sha256()
    if partial.exists():
        with partial.open("rb") as existing:
            while True:
                payload = existing.read(plan.chunk_size_bytes)
                if not payload:
                    break
                digest.update(payload)

    with source.open("rb") as source_handle:
        _verify_checkpoint_against_source(source_handle, checkpoint)
        source_handle.seek(total)
        mode = "ab" if partial.exists() else "wb"
        with partial.open(mode) as destination_handle:
            index = len(chunks)
            while True:
                payload = source_handle.read(plan.chunk_size_bytes)
                if not payload:
                    break
                if total + len(payload) > plan.expected_size_bytes:
                    raise SourceAcquisitionError("retrieved source exceeds expected registered size")
                destination_handle.write(payload)
                digest.update(payload)
                chunks.append(
                    RetrievedChunk(
                        index=index,
                        offset_bytes=total,
                        size_bytes=len(payload),
                        sha256=_sha256_bytes(payload),
                    )
                )
                total += len(payload)
                index += 1
            destination_handle.flush()
            os.fsync(destination_handle.fileno())

    verified_sha256 = digest.hexdigest()
    if total != plan.expected_size_bytes:
        raise SourceAcquisitionError(
            f"retrieved source size mismatch: expected {plan.expected_size_bytes}, got {total}"
        )
    if verified_sha256 != plan.expected_sha256:
        raise SourceAcquisitionError("retrieved source SHA-256 mismatch")
    os.replace(partial, destination)
    return _build_receipt(plan, chunks, verified_sha256, total)


def build_retrieval_inventory(
    source_registry: Mapping[str, Any],
    receipts: Iterable[VerifiedRetrievalReceipt],
) -> dict[str, Any]:
    """Bind verified byte receipts to the exact source registry without granting train eligibility."""

    sources = validate_external_source_registry(source_registry)
    source_map = {(item.source_id, item.source_version): item for item in sources}
    registry_identity = _require_sha256(
        source_registry.get("registry_identity_sha256"),
        "registry_identity_sha256",
    )
    receipt_list = sorted(receipts, key=lambda item: (item.source_id, item.source_version))
    seen: set[tuple[str, str]] = set()
    total_bytes = 0
    entries: list[dict[str, Any]] = []
    for receipt in receipt_list:
        key = (receipt.source_id, receipt.source_version)
        if key in seen:
            raise SourceAcquisitionError("retrieval inventory contains duplicate source version")
        seen.add(key)
        source = source_map.get(key)
        if source is None:
            raise SourceAcquisitionError("retrieval receipt references an unregistered source version")
        if receipt.source_registry_identity_sha256 != registry_identity:
            raise SourceAcquisitionError("retrieval receipt uses a different source registry identity")
        if receipt.expected_sha256 != source.snapshot.sha256:
            raise SourceAcquisitionError("retrieval receipt hash differs from registered snapshot")
        if receipt.verified_size_bytes != source.snapshot.size_bytes:
            raise SourceAcquisitionError("retrieval receipt size differs from registered snapshot")
        total_bytes += receipt.verified_size_bytes
        entries.append(receipt.manifest())
    if not entries:
        raise SourceAcquisitionError("retrieval inventory requires at least one verified receipt")
    core = {
        "schema_version": RETRIEVAL_INVENTORY_SCHEMA,
        "source_registry_identity_sha256": registry_identity,
        "verified_source_versions": len(entries),
        "verified_bytes": total_bytes,
        "receipts": entries,
        "rights_semantics": "INVENTORY_IS_NOT_TRAINING_APPROVAL",
    }
    return {**core, "inventory_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def validate_fsspec_runtime() -> str:
    """Return the installed fsspec version or fail without making it a base dependency."""

    try:
        installed = version("fsspec")
    except PackageNotFoundError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("remote source acquisition requires optional dependency fsspec") from exc
    return installed
