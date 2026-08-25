"""Production-oriented corpus provenance, decontamination, and streaming contracts."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

CORPUS_PLAN_SCHEMA = "12-6.corpus-streaming-plan.v1"
RESUME_SCHEMA = "12-6.corpus-resume-manifest.v1"
DEDUP_PLAN_SCHEMA = "12-6.dedup-plan.v1"
POLICY_SCHEMA = "12-6.record-policy-metadata.v1"
D06_REGISTRY_SCHEMA = "12-6.benchmark-registry.v1"
_ALLOWED_HOOK_STATUS = frozenset({"NOT_RUN", "PASS", "REJECT", "REVIEW_REQUIRED"})
_FORBIDDEN_TRAIN_USES = frozenset(
    {
        "train",
        "training",
        "pretrain",
        "pretraining",
        "finetune",
        "fine-tune",
        "sft",
        "dpo",
        "rl",
        "posttrain",
        "post-training",
    }
)


class CorpusFoundationError(ValueError):
    """Raised when corpus provenance or processing evidence is unsafe/inconsistent."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusFoundationError(f"{field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if text != text.lower() or len(text) != 64:
        raise CorpusFoundationError(f"{field} must be lowercase SHA-256 hex")
    if any(char not in "0123456789abcdef" for char in text):
        raise CorpusFoundationError(f"{field} must be lowercase SHA-256 hex")
    return text


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CorpusFoundationError(f"{field} must be a positive integer")
    return value


def _validate_stable_uri(value: str, field: str) -> None:
    text = _require_text(value, field)
    parsed = urlsplit(text)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise CorpusFoundationError(f"{field} must not contain credentials/query/fragment")


@dataclass(frozen=True)
class PolicyHookEvidence:
    """One non-authoritative data-policy hook result bound to exact tool/policy evidence."""

    hook_id: str
    status: str
    policy_version: str
    tool_ref: str
    executed_at: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        for field in ("hook_id", "policy_version", "tool_ref", "executed_at"):
            _require_text(getattr(self, field), field)
        if self.status not in _ALLOWED_HOOK_STATUS:
            raise CorpusFoundationError(f"unsupported hook status: {self.status}")
        _require_sha256(self.evidence_sha256, "evidence_sha256")


@dataclass(frozen=True)
class RecordPolicyMetadata:
    """Quality/LID/PII/copyright hook evidence; metadata does not grant source rights."""

    quality: PolicyHookEvidence
    language: PolicyHookEvidence
    pii: PolicyHookEvidence
    copyright: PolicyHookEvidence
    schema_version: str = POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_SCHEMA:
            raise CorpusFoundationError("unsupported policy metadata schema")

    def assert_passed(self) -> None:
        failures = [
            hook.hook_id
            for hook in (self.quality, self.language, self.pii, self.copyright)
            if hook.status != "PASS"
        ]
        if failures:
            raise CorpusFoundationError(
                "record policy metadata is not train-eligible; non-PASS hooks: "
                + ", ".join(sorted(failures))
            )

    def manifest(self) -> dict[str, Any]:
        core = asdict(self)
        return {**core, "metadata_sha256": _sha256_bytes(_canonical_json_bytes(core))}


@dataclass(frozen=True)
class DataTroveDedupPlan:
    """Deterministic scale seam for exact fingerprint + DataTrove MinHash near dedup."""

    source_registry_sha256: str
    reserved_registry_sha256: str
    input_uri: str
    output_uri: str
    logging_uri: str
    exact_key: str = "content_sha256"
    near_engine: str = "datatrove_minhash"
    minhash_signature_size: int = 128
    minhash_num_buckets: int = 14
    minhash_n_grams: int = 5
    tasks: int = 1
    workers: int = 1
    datatrove_version: str = "0.10.0"

    def __post_init__(self) -> None:
        _require_sha256(self.source_registry_sha256, "source_registry_sha256")
        _require_sha256(self.reserved_registry_sha256, "reserved_registry_sha256")
        for field in ("input_uri", "output_uri", "logging_uri"):
            _validate_stable_uri(getattr(self, field), field)
        if len({self.input_uri, self.output_uri, self.logging_uri}) != 3:
            raise CorpusFoundationError("input/output/logging URIs must be distinct")
        if self.exact_key != "content_sha256":
            raise CorpusFoundationError("exact dedup key must be content_sha256")
        if self.near_engine != "datatrove_minhash":
            raise CorpusFoundationError("near dedup engine must remain datatrove_minhash")
        for field in ("minhash_signature_size", "minhash_num_buckets", "minhash_n_grams"):
            _require_positive_int(getattr(self, field), field)
        _require_positive_int(self.tasks, "tasks")
        _require_positive_int(self.workers, "workers")
        if self.workers > self.tasks:
            raise CorpusFoundationError("workers cannot exceed tasks")
        if self.datatrove_version != "0.10.0":
            raise CorpusFoundationError("DataTrove version must be 0.10.0 until revalidated")

    def manifest(self) -> dict[str, Any]:
        core = {"schema_version": DEDUP_PLAN_SCHEMA, **asdict(self)}
        return {**core, "plan_sha256": _sha256_bytes(_canonical_json_bytes(core))}


class SQLiteExactDedupIndex:
    """Disk-backed exact dedup index for bounded-memory local/free mechanics tests."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS fingerprints (sha256 TEXT PRIMARY KEY) WITHOUT ROWID"
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self._connection.commit()
        else:
            self._connection.rollback()
        self._connection.close()

    def seen_or_add(self, fingerprint: str) -> bool:
        digest = _require_sha256(fingerprint, "fingerprint")
        cursor = self._connection.execute(
            "INSERT OR IGNORE INTO fingerprints(sha256) VALUES (?)", (digest,)
        )
        return cursor.rowcount == 0

    def commit(self) -> None:
        self._connection.commit()


@dataclass(frozen=True)
class StreamingShardPlan:
    """Deterministic hash sharding with bounded in-memory batching and Parquet output."""

    source_registry_sha256: str
    reserved_registry_sha256: str
    output_uri: str
    partition_salt: str
    shard_count: int
    max_records_in_memory: int
    row_group_target_bytes: int
    parquet_compression: str = "zstd"
    assignment: str = "sha256-record-id-mod-v1"

    def __post_init__(self) -> None:
        _require_sha256(self.source_registry_sha256, "source_registry_sha256")
        _require_sha256(self.reserved_registry_sha256, "reserved_registry_sha256")
        _validate_stable_uri(self.output_uri, "output_uri")
        _require_text(self.partition_salt, "partition_salt")
        _require_positive_int(self.shard_count, "shard_count")
        _require_positive_int(self.max_records_in_memory, "max_records_in_memory")
        _require_positive_int(self.row_group_target_bytes, "row_group_target_bytes")
        if self.parquet_compression not in {"zstd", "snappy", "none"}:
            raise CorpusFoundationError("unsupported Parquet compression")
        if self.assignment != "sha256-record-id-mod-v1":
            raise CorpusFoundationError("unsupported shard assignment")

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": CORPUS_PLAN_SCHEMA,
            **asdict(self),
            "output_format": "parquet",
            "filesystem": "fsspec",
        }
        return {**core, "plan_sha256": _sha256_bytes(_canonical_json_bytes(core))}

    def assign(self, record_id: str) -> int:
        identifier = _require_text(record_id, "record_id")
        digest = hashlib.sha256(f"{self.partition_salt}\0{identifier}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % self.shard_count


@dataclass(frozen=True)
class ShardArtifact:
    shard_index: int
    part_index: int
    uri: str
    sha256: str
    size_bytes: int
    records: int

    def __post_init__(self) -> None:
        if isinstance(self.shard_index, bool) or not isinstance(self.shard_index, int):
            raise CorpusFoundationError("shard_index must be an integer")
        if self.shard_index < 0:
            raise CorpusFoundationError("shard_index must be non-negative")
        if isinstance(self.part_index, bool) or not isinstance(self.part_index, int):
            raise CorpusFoundationError("part_index must be an integer")
        if self.part_index < 0:
            raise CorpusFoundationError("part_index must be non-negative")
        _validate_stable_uri(self.uri, "uri")
        _require_sha256(self.sha256, "sha256")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise CorpusFoundationError("size_bytes must be an integer")
        if self.size_bytes < 0:
            raise CorpusFoundationError("size_bytes must be non-negative")
        if isinstance(self.records, bool) or not isinstance(self.records, int):
            raise CorpusFoundationError("records must be an integer")
        if self.records < 0:
            raise CorpusFoundationError("records must be non-negative")


def build_resume_manifest(
    plan: StreamingShardPlan,
    artifacts: Iterable[ShardArtifact],
    *,
    dedup_plan_sha256: str,
) -> dict[str, Any]:
    dedup_hash = _require_sha256(dedup_plan_sha256, "dedup_plan_sha256")
    ordered = sorted(artifacts, key=lambda item: (item.shard_index, item.part_index))
    seen: set[tuple[int, int]] = set()
    next_part = {index: 0 for index in range(plan.shard_count)}
    for artifact in ordered:
        if artifact.shard_index >= plan.shard_count:
            raise CorpusFoundationError("artifact shard_index exceeds plan")
        key = (artifact.shard_index, artifact.part_index)
        if key in seen:
            raise CorpusFoundationError("duplicate shard/part artifact")
        seen.add(key)
        expected = next_part[artifact.shard_index]
        if artifact.part_index != expected:
            raise CorpusFoundationError(
                f"non-contiguous part indexes for shard {artifact.shard_index}: "
                f"expected {expected}, got {artifact.part_index}"
            )
        next_part[artifact.shard_index] = expected + 1
    core = {
        "schema_version": RESUME_SCHEMA,
        "plan_sha256": plan.manifest()["plan_sha256"],
        "source_registry_sha256": plan.source_registry_sha256,
        "reserved_registry_sha256": plan.reserved_registry_sha256,
        "dedup_plan_sha256": dedup_hash,
        "artifacts": [asdict(item) for item in ordered],
        "next_part_index_by_shard": {str(key): value for key, value in next_part.items()},
    }
    return {**core, "resume_manifest_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def validate_resume_manifest(
    manifest: Mapping[str, Any],
    plan: StreamingShardPlan,
    *,
    dedup_plan_sha256: str,
) -> None:
    if manifest.get("schema_version") != RESUME_SCHEMA:
        raise CorpusFoundationError("unsupported resume manifest schema")
    raw = manifest.get("artifacts")
    if not isinstance(raw, list):
        raise CorpusFoundationError("resume artifacts must be an array")
    artifacts: list[ShardArtifact] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise CorpusFoundationError("resume artifact must be an object")
        artifacts.append(ShardArtifact(**dict(item)))
    expected = build_resume_manifest(plan, artifacts, dedup_plan_sha256=dedup_plan_sha256)
    if dict(manifest) != expected:
        raise CorpusFoundationError("resume manifest identity/content mismatch")


def _validate_d06_manifest(manifest: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    if manifest.get("schema_version") != D06_REGISTRY_SCHEMA:
        raise CorpusFoundationError("unsupported D06 benchmark registry schema")
    raw = manifest.get("benchmarks")
    if not isinstance(raw, list):
        raise CorpusFoundationError("D06 benchmarks must be an array")
    payload = {"schema_version": D06_REGISTRY_SCHEMA, "benchmarks": raw}
    expected_hash = _sha256_bytes(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    if manifest.get("manifest_sha256") != expected_hash:
        raise CorpusFoundationError("D06 benchmark registry manifest hash mismatch")
    result: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise CorpusFoundationError("D06 benchmark entry must be an object")
        benchmark_id = _require_text(item.get("benchmark_id"), "benchmark_id")
        version = _require_text(item.get("version"), "version")
        source_id = _require_text(item.get("source_id"), "source_id")
        key = (benchmark_id, version)
        if key in seen:
            raise CorpusFoundationError("duplicate D06 benchmark identity")
        seen.add(key)
        if item.get("held_out") is not True:
            raise CorpusFoundationError(f"D06 benchmark {benchmark_id}@{version} must be held_out")
        uses = item.get("allowed_uses")
        if not isinstance(uses, list) or not uses:
            raise CorpusFoundationError("D06 allowed_uses must be a non-empty array")
        normalized = {_require_text(value, "allowed_use").lower() for value in uses}
        if normalized & _FORBIDDEN_TRAIN_USES:
            raise CorpusFoundationError(
                f"held-out D06 benchmark {benchmark_id}@{version} permits training use"
            )
        result.append({**dict(item), "source_id": source_id})
    return result


def reserved_registry_from_d06_manifest(
    manifest: Mapping[str, Any],
    fingerprints_by_source_id: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Translate exact D06 held-out identities into D03 reserved-fingerprint registry."""

    from .external_sources import ReservedSetSpec, build_reserved_fingerprint_registry

    benchmarks = _validate_d06_manifest(manifest)
    sets = []
    for item in benchmarks:
        source_id = item["source_id"]
        raw_fingerprints = fingerprints_by_source_id.get(source_id, ())
        fingerprints = tuple(
            sorted(_require_sha256(value, f"fingerprints[{source_id}]") for value in raw_fingerprints)
        )
        sets.append(
            ReservedSetSpec(
                set_id=f"d06:{item['benchmark_id']}",
                version=item["version"],
                source_id=source_id,
                purpose="evaluation",
                normalized_sha256=fingerprints,
            )
        )
    return build_reserved_fingerprint_registry(sets)


def assert_contamination_free(report: Mapping[str, Any]) -> None:
    required = ("source_id_overlap_count", "content_sha256_overlap_count")
    for field in required:
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CorpusFoundationError(f"invalid contamination report field: {field}")
    if report["source_id_overlap_count"] or report["content_sha256_overlap_count"]:
        raise CorpusFoundationError("contamination report contains reserved benchmark overlap")
