"""Production-oriented corpus provenance, decontamination, and streaming contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlsplit

from .external_sources import (
    ExternalDataContractError,
    ReservedSetSpec,
    build_reserved_fingerprint_registry,
    contamination_report,
    validate_external_source_registry,
)
from .scalable_ingestion import DATATROVE_VERSION

D06_BENCHMARK_SCHEMA = "12-6.benchmark-registry.v1"
POLICY_METADATA_SCHEMA = "12-6.record-policy-metadata.v1"
CORPUS_ELIGIBILITY_SCHEMA = "12-6.corpus-eligibility.v1"
EXACT_DEDUP_SCHEMA = "12-6.exact-dedup-plan.v1"
MINHASH_DEDUP_SCHEMA = "12-6.datatrove-minhash-plan.v1"
SHARD_PLAN_SCHEMA = "12-6.streaming-shard-plan.v1"
RESUME_SCHEMA = "12-6.streaming-resume-manifest.v1"

_ALLOWED_POLICY_CATEGORIES = frozenset({"quality", "language", "pii", "copyright"})
_ALLOWED_POLICY_STATUSES = frozenset({"PASS", "REJECT", "REVIEW_REQUIRED", "NOT_RUN"})
_REQUIRED_POLICY_CATEGORIES = frozenset(_ALLOWED_POLICY_CATEGORIES)
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
    """Raised when corpus eligibility, dedup, sharding, or resume evidence is unsafe."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusFoundationError(f"{field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if text != text.lower():
        raise CorpusFoundationError(f"{field} must be lowercase SHA-256 hex")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise CorpusFoundationError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _require_non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorpusFoundationError(f"{field} must be a non-negative integer")
    return value


def _validate_storage_uri(value: str, field: str) -> None:
    _require_text(value, field)
    parsed = urlsplit(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise CorpusFoundationError(f"{field} must not contain credentials/query/fragment")
    if parsed.scheme and parsed.scheme not in {"file", "s3", "gs", "hf", "az", "r2"}:
        raise CorpusFoundationError(f"{field} uses unsupported storage scheme {parsed.scheme!r}")


def _d06_manifest_sha256(payload: Mapping[str, Any]) -> str:
    core = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    canonical = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(canonical)


@dataclass(frozen=True)
class PolicyHookResult:
    """Per-record policy metadata. It never grants source-level training rights."""

    category: str
    status: str
    hook_id: str
    hook_version: str
    policy_ref: str
    detail_code: str

    def __post_init__(self) -> None:
        if self.category not in _ALLOWED_POLICY_CATEGORIES:
            raise CorpusFoundationError(f"unsupported policy category: {self.category}")
        if self.status not in _ALLOWED_POLICY_STATUSES:
            raise CorpusFoundationError(f"unsupported policy status: {self.status}")
        for field in ("hook_id", "hook_version", "policy_ref", "detail_code"):
            _require_text(getattr(self, field), field)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RecordPolicyMetadata:
    """Document-level provenance and policy results without storing document text."""

    record_id: str
    source_id: str
    source_version: str
    content_sha256: str
    normalized_size_bytes: int
    language: str
    hooks: tuple[PolicyHookResult, ...]

    def __post_init__(self) -> None:
        for field in ("record_id", "source_id", "source_version", "language"):
            _require_text(getattr(self, field), field)
        _require_sha256(self.content_sha256, "content_sha256")
        _require_non_negative_int(self.normalized_size_bytes, "normalized_size_bytes")
        if any(not isinstance(hook, PolicyHookResult) for hook in self.hooks):
            raise CorpusFoundationError("hooks must contain PolicyHookResult values")
        categories = [hook.category for hook in self.hooks]
        if len(set(categories)) != len(categories):
            raise CorpusFoundationError("record policy metadata contains duplicate categories")

    def assert_policy_clear(self) -> None:
        by_category = {hook.category: hook for hook in self.hooks}
        missing = sorted(_REQUIRED_POLICY_CATEGORIES - set(by_category))
        if missing:
            raise CorpusFoundationError(
                f"{self.record_id}: required policy hooks are missing: {missing}"
            )
        blocked = sorted(
            category
            for category, hook in by_category.items()
            if hook.status != "PASS"
        )
        if blocked:
            raise CorpusFoundationError(
                f"{self.record_id}: policy hooks are not clear for training: {blocked}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_METADATA_SCHEMA,
            "record_id": self.record_id,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "content_sha256": self.content_sha256,
            "normalized_size_bytes": self.normalized_size_bytes,
            "language": self.language,
            "hooks": [
                hook.to_dict()
                for hook in sorted(self.hooks, key=lambda item: item.category)
            ],
        }


def build_corpus_eligibility_manifest(
    source_registry: Mapping[str, Any],
    records: Iterable[RecordPolicyMetadata],
) -> dict[str, Any]:
    """Stream metadata and fail closed unless source rights and record hooks are clear."""

    sources = validate_external_source_registry(source_registry)
    source_map = {(source.source_id, source.source_version): source for source in sources}
    accepted_source_versions: set[tuple[str, str]] = set()
    record_count = 0
    normalized_bytes = 0
    stream_hash = hashlib.sha256()

    for record in records:
        if not isinstance(record, RecordPolicyMetadata):
            raise CorpusFoundationError("records must contain RecordPolicyMetadata values")
        source_key = (record.source_id, record.source_version)
        source = source_map.get(source_key)
        if source is None:
            raise CorpusFoundationError(
                f"{record.record_id}: source version is absent from the external registry"
            )
        try:
            source.assert_training_eligible()
        except ExternalDataContractError as exc:
            raise CorpusFoundationError(
                f"{record.record_id}: source version is not train-eligible: {exc}"
            ) from exc
        record.assert_policy_clear()
        record_count += 1
        normalized_bytes += record.normalized_size_bytes
        accepted_source_versions.add(source_key)
        stream_hash.update(_canonical_json_bytes(record.to_dict()))

    if record_count == 0:
        raise CorpusFoundationError("corpus eligibility requires at least one accepted record")

    core = {
        "schema_version": CORPUS_ELIGIBILITY_SCHEMA,
        "source_registry_identity_sha256": source_registry.get("registry_identity_sha256"),
        "record_count": record_count,
        "normalized_size_bytes": normalized_bytes,
        "source_versions": [
            {"source_id": source_id, "source_version": source_version}
            for source_id, source_version in sorted(accepted_source_versions)
        ],
        "required_policy_categories": sorted(_REQUIRED_POLICY_CATEGORIES),
        "metadata_stream_sha256": stream_hash.hexdigest(),
        "rights_and_policy_gate": "PASS",
    }
    return {**core, "manifest_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def reserved_registry_from_d06(
    benchmark_manifest: Mapping[str, Any],
    normalized_fingerprints: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Convert a canonical D06 benchmark registry manifest into D03 reserved identities."""

    if benchmark_manifest.get("schema_version") != D06_BENCHMARK_SCHEMA:
        raise CorpusFoundationError("unsupported D06 benchmark registry schema")
    raw_benchmarks = benchmark_manifest.get("benchmarks")
    if not isinstance(raw_benchmarks, list):
        raise CorpusFoundationError("D06 benchmark registry benchmarks must be an array")
    expected_hash = _d06_manifest_sha256(benchmark_manifest)
    if benchmark_manifest.get("manifest_sha256") != expected_hash:
        raise CorpusFoundationError("D06 benchmark registry manifest SHA-256 mismatch")

    fingerprint_map = normalized_fingerprints or {}
    seen_keys: set[str] = set()
    reserved: list[ReservedSetSpec] = []
    for item in raw_benchmarks:
        if not isinstance(item, Mapping):
            raise CorpusFoundationError("D06 benchmark entry must be an object")
        benchmark_id = _require_text(item.get("benchmark_id"), "benchmark_id")
        version_id = _require_text(item.get("version"), "version")
        source_id = _require_text(item.get("source_id"), "source_id")
        key = f"{benchmark_id}@{version_id}"
        if key in seen_keys:
            raise CorpusFoundationError(f"duplicate D06 benchmark key: {key}")
        seen_keys.add(key)
        if item.get("held_out") is not True:
            raise CorpusFoundationError(f"{key}: D06 benchmark must be held_out=true")
        allowed_uses = item.get("allowed_uses")
        if not isinstance(allowed_uses, list) or not allowed_uses:
            raise CorpusFoundationError(f"{key}: allowed_uses must be a non-empty array")
        normalized_uses = {
            _require_text(value, f"{key}.allowed_uses").lower() for value in allowed_uses
        }
        forbidden = sorted(normalized_uses & _FORBIDDEN_TRAIN_USES)
        if forbidden:
            raise CorpusFoundationError(
                f"{key}: held-out benchmark allows training uses: {forbidden}"
            )
        fingerprints = tuple(fingerprint_map.get(key, ()))
        reserved.append(
            ReservedSetSpec(
                set_id=benchmark_id,
                version=version_id,
                source_id=source_id,
                purpose="evaluation",
                normalized_sha256=fingerprints,
            )
        )

    unused = sorted(set(fingerprint_map) - seen_keys)
    if unused:
        raise CorpusFoundationError(f"fingerprints reference unknown D06 benchmark keys: {unused}")
    return build_reserved_fingerprint_registry(reserved)


def contamination_gate(
    records: Iterable[Mapping[str, Any]],
    reserved_registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Run D03 contamination accounting and reject any source/content overlap."""

    report = contamination_report(records, reserved_registry)
    if report["source_id_overlap_count"] or report["content_sha256_overlap_count"]:
        raise CorpusFoundationError(
            "training/evaluation contamination detected: "
            f"source={report['source_id_overlap_count']} "
            f"content={report['content_sha256_overlap_count']}"
        )
    return report


@dataclass(frozen=True)
class ExactDedupPlan:
    """Deterministic exact-content dedup partition plan for Parquet/fsspec execution."""

    corpus_identity_sha256: str
    input_uri: str
    survivor_uri: str
    duplicate_uri: str
    partitions: int = 256
    key_field: str = "content_sha256"

    def __post_init__(self) -> None:
        _require_sha256(self.corpus_identity_sha256, "corpus_identity_sha256")
        for field in ("input_uri", "survivor_uri", "duplicate_uri"):
            _validate_storage_uri(getattr(self, field), field)
        if len({self.input_uri, self.survivor_uri, self.duplicate_uri}) != 3:
            raise CorpusFoundationError(
                "exact dedup input/survivor/duplicate URIs must be distinct"
            )
        if isinstance(self.partitions, bool) or not isinstance(self.partitions, int):
            raise CorpusFoundationError("partitions must be an integer")
        if self.partitions <= 0:
            raise CorpusFoundationError("partitions must be positive")
        if self.key_field != "content_sha256":
            raise CorpusFoundationError("exact dedup key_field is fixed to content_sha256")

    def partition_for(self, content_sha256: str) -> int:
        digest = bytes.fromhex(_require_sha256(content_sha256, "content_sha256"))
        return int.from_bytes(digest[:8], "big") % self.partitions

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": EXACT_DEDUP_SCHEMA,
            **asdict(self),
            "algorithm": "sha256_exact_partitioned_sort_unique",
            "tie_break_fields": ["source_id", "source_version", "record_id"],
            "storage": "parquet_fsspec",
            "bounded_memory_contract": "one_hash_partition_at_a_time",
        }
        return {**core, "plan_sha256": _sha256_bytes(_canonical_json_bytes(core))}


@dataclass(frozen=True)
class DataTroveMinhashPlan:
    """Four-stage DataTrove MinHash seam over immutable Parquet shards."""

    corpus_identity_sha256: str
    input_parquet_uri: str
    signatures_uri: str
    buckets_uri: str
    remove_ids_uri: str
    output_parquet_uri: str
    logging_uri: str
    tasks: int
    workers: int
    num_buckets: int = 14
    hashes_per_bucket: int = 8
    n_grams: int = 5
    hash_precision: int = 64
    datatrove_version: str = DATATROVE_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.corpus_identity_sha256, "corpus_identity_sha256")
        uri_fields = (
            "input_parquet_uri",
            "signatures_uri",
            "buckets_uri",
            "remove_ids_uri",
            "output_parquet_uri",
            "logging_uri",
        )
        for field in uri_fields:
            _validate_storage_uri(getattr(self, field), field)
        if len({getattr(self, field) for field in uri_fields}) != len(uri_fields):
            raise CorpusFoundationError("DataTrove MinHash URIs must be distinct")
        for field in ("tasks", "workers", "num_buckets", "hashes_per_bucket", "n_grams"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CorpusFoundationError(f"{field} must be a positive integer")
        if self.workers > self.tasks:
            raise CorpusFoundationError("workers cannot exceed tasks")
        if self.hash_precision not in {32, 64}:
            raise CorpusFoundationError("hash_precision must be 32 or 64")
        if self.datatrove_version != DATATROVE_VERSION:
            raise CorpusFoundationError(
                f"DataTrove plan must pin {DATATROVE_VERSION} until compatibility is revalidated"
            )

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": MINHASH_DEDUP_SCHEMA,
            **asdict(self),
            "storage": "parquet_fsspec",
            "skip_completed": True,
            "pipeline": [
                "ParquetReader->MinhashDedupSignature",
                "MinhashDedupBuckets",
                "MinhashDedupCluster",
                "ParquetReader->MinhashDedupFilter->ParquetWriter",
            ],
        }
        return {**core, "plan_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def validate_datatrove_minhash_runtime(plan: DataTroveMinhashPlan) -> tuple[str, ...]:
    """Verify the pinned DataTrove runtime and exact MinHash/Parquet symbols exist."""

    try:
        installed_version = version("datatrove")
    except PackageNotFoundError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            f"DataTrove MinHash execution requires datatrove[io]=={plan.datatrove_version}"
        ) from exc
    if installed_version != plan.datatrove_version:
        raise RuntimeError(
            f"DataTrove runtime version mismatch: expected {plan.datatrove_version}, "
            f"got {installed_version}"
        )
    try:
        from datatrove.pipeline.dedup import MinhashDedupSignature
        from datatrove.pipeline.dedup.minhash import (
            MinhashConfig,
            MinhashDedupBuckets,
            MinhashDedupCluster,
            MinhashDedupFilter,
        )
        from datatrove.pipeline.readers import ParquetReader
        from datatrove.pipeline.writers import ParquetWriter
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("DataTrove MinHash/Parquet symbols are unavailable") from exc
    return tuple(
        item.__name__
        for item in (
            ParquetReader,
            MinhashConfig,
            MinhashDedupSignature,
            MinhashDedupBuckets,
            MinhashDedupCluster,
            MinhashDedupFilter,
            ParquetWriter,
        )
    )


@dataclass(frozen=True)
class StreamingShardPlan:
    """Worker-count-independent deterministic sharding with a per-record memory ceiling."""

    corpus_identity_sha256: str
    shard_count: int
    max_record_bytes: int = 4 * 1024 * 1024
    assignment_salt: str = "12-6-corpus-shard-v1"

    def __post_init__(self) -> None:
        _require_sha256(self.corpus_identity_sha256, "corpus_identity_sha256")
        if isinstance(self.shard_count, bool) or not isinstance(self.shard_count, int):
            raise CorpusFoundationError("shard_count must be an integer")
        if self.shard_count <= 0:
            raise CorpusFoundationError("shard_count must be positive")
        if isinstance(self.max_record_bytes, bool) or not isinstance(self.max_record_bytes, int):
            raise CorpusFoundationError("max_record_bytes must be an integer")
        if self.max_record_bytes <= 0:
            raise CorpusFoundationError("max_record_bytes must be positive")
        _require_text(self.assignment_salt, "assignment_salt")

    def shard_for_record_id(self, record_id: str) -> int:
        record = _require_text(record_id, "record_id")
        digest = hashlib.sha256(f"{self.assignment_salt}\0{record}".encode()).digest()
        return int.from_bytes(digest[:8], "big") % self.shard_count

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": SHARD_PLAN_SCHEMA,
            **asdict(self),
            "algorithm": "sha256(record_id)_mod_shard_count",
            "worker_count_independent": True,
            "streaming_memory_contract": "one_record_plus_writer_buffer",
        }
        return {**core, "plan_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def iter_shard(
    records: Iterable[RecordPolicyMetadata],
    plan: StreamingShardPlan,
    shard_id: int,
) -> Iterator[RecordPolicyMetadata]:
    """Yield one deterministic shard without materializing the input corpus."""

    if isinstance(shard_id, bool) or not isinstance(shard_id, int):
        raise CorpusFoundationError("shard_id must be an integer")
    if shard_id < 0 or shard_id >= plan.shard_count:
        raise CorpusFoundationError("shard_id is outside the shard plan")
    for record in records:
        if record.normalized_size_bytes > plan.max_record_bytes:
            raise CorpusFoundationError(
                f"{record.record_id}: normalized record exceeds max_record_bytes"
            )
        if plan.shard_for_record_id(record.record_id) == shard_id:
            yield record


@dataclass(frozen=True)
class ShardCompletion:
    shard_id: int
    record_count: int
    output_size_bytes: int
    output_sha256: str

    def __post_init__(self) -> None:
        _require_non_negative_int(self.shard_id, "shard_id")
        _require_non_negative_int(self.record_count, "record_count")
        _require_non_negative_int(self.output_size_bytes, "output_size_bytes")
        _require_sha256(self.output_sha256, "output_sha256")


@dataclass(frozen=True)
class StreamingResumeManifest:
    """Content-addressed completed-shard evidence used for deterministic resume."""

    plan_sha256: str
    shard_count: int
    completed: tuple[ShardCompletion, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.plan_sha256, "plan_sha256")
        if isinstance(self.shard_count, bool) or not isinstance(self.shard_count, int):
            raise CorpusFoundationError("shard_count must be an integer")
        if self.shard_count <= 0:
            raise CorpusFoundationError("shard_count must be positive")
        shard_ids = [item.shard_id for item in self.completed]
        if len(set(shard_ids)) != len(shard_ids):
            raise CorpusFoundationError("resume manifest contains duplicate shard completions")
        if any(shard_id >= self.shard_count for shard_id in shard_ids):
            raise CorpusFoundationError("resume manifest completion is outside shard_count")

    def pending_shards(self) -> tuple[int, ...]:
        completed = {item.shard_id for item in self.completed}
        return tuple(shard_id for shard_id in range(self.shard_count) if shard_id not in completed)

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": RESUME_SCHEMA,
            "plan_sha256": self.plan_sha256,
            "shard_count": self.shard_count,
            "completed": [
                asdict(item) for item in sorted(self.completed, key=lambda value: value.shard_id)
            ],
        }
        return {**core, "manifest_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def assert_resume_compatible(
    plan: StreamingShardPlan,
    resume: StreamingResumeManifest,
) -> None:
    manifest = plan.manifest()
    if resume.plan_sha256 != manifest["plan_sha256"]:
        raise CorpusFoundationError("resume manifest is bound to a different shard plan")
    if resume.shard_count != plan.shard_count:
        raise CorpusFoundationError("resume shard_count differs from the shard plan")
