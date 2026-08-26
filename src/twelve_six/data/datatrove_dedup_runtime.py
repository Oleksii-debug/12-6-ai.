"""Pinned DataTrove 0.10.0 MinHash execution for large-corpus near deduplication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from urllib.parse import urlsplit

DATATROVE_VERSION = "0.10.0"
RUNTIME_SCHEMA = "12-6.datatrove-minhash-runtime.v1"


class DataTroveMinhashError(ValueError):
    """Raised when a scalable near-dedup execution plan is unsafe or inconsistent."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _require_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataTroveMinhashError(f"{field} must be a non-empty string")
    return value.strip()


def _require_sha256(value: str, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DataTroveMinhashError(f"{field} must be lowercase SHA-256 hex")
    if value != value.lower() or any(char not in "0123456789abcdef" for char in value):
        raise DataTroveMinhashError(f"{field} must be lowercase SHA-256 hex")
    return value


def _require_positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DataTroveMinhashError(f"{field} must be a positive integer")
    return value


def _validate_uri(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataTroveMinhashError(f"{field} must be a non-empty URI/path")
    parsed = urlsplit(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise DataTroveMinhashError(f"{field} must not contain credentials/query/fragment")
    return value.rstrip("/")


def _join_uri(base: str, suffix: str) -> str:
    return f"{base.rstrip('/')}/{suffix.lstrip('/')}"


@dataclass(frozen=True)
class DataTroveMinhashSpec:
    """Executable DataTrove MinHash contract matching the public 0.10.0 API.

    ``language`` is deliberately mandatory. DataTrove's signature stage otherwise defaults
    to English, which is unsafe for this project's mixed Ukrainian/English/code corpus.
    Run language/modality partitions with an explicit tokenizer language and retain the
    partition identity in the manifest before any global successor corpus is promoted.
    """

    source_registry_sha256: str
    reserved_registry_sha256: str
    input_uri: str
    output_uri: str
    work_uri: str
    logging_uri: str
    language: str
    num_buckets: int = 14
    hashes_per_bucket: int = 8
    n_grams: int = 5
    hash_precision: int = 64
    seed: int = 1
    signature_tasks: int = 1
    workers: int = 1
    datatrove_version: str = DATATROVE_VERSION

    def __post_init__(self) -> None:
        _require_sha256(self.source_registry_sha256, "source_registry_sha256")
        _require_sha256(self.reserved_registry_sha256, "reserved_registry_sha256")
        _require_text(self.language, "language")
        uris = [
            _validate_uri(getattr(self, field), field)
            for field in ("input_uri", "output_uri", "work_uri", "logging_uri")
        ]
        if len(set(uris)) != len(uris):
            raise DataTroveMinhashError("input/output/work/logging URIs must be distinct")
        for field in (
            "num_buckets",
            "hashes_per_bucket",
            "n_grams",
            "seed",
            "signature_tasks",
            "workers",
        ):
            _require_positive_int(getattr(self, field), field)
        if self.workers > self.signature_tasks:
            raise DataTroveMinhashError("workers cannot exceed signature_tasks")
        if self.hash_precision != 64:
            raise DataTroveMinhashError("hash_precision must remain 64 until revalidated")
        if self.datatrove_version != DATATROVE_VERSION:
            raise DataTroveMinhashError(
                f"DataTrove version must remain {DATATROVE_VERSION} until revalidated"
            )

    @property
    def total_signature_hashes(self) -> int:
        return self.num_buckets * self.hashes_per_bucket

    def stage_topology(self) -> dict[str, dict[str, int]]:
        return {
            "signatures": {"tasks": self.signature_tasks, "workers": self.workers},
            "buckets": {
                "tasks": self.num_buckets,
                "workers": min(self.workers, self.num_buckets),
            },
            "cluster": {"tasks": 1, "workers": 1},
            "filter": {"tasks": self.signature_tasks, "workers": self.workers},
        }

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": RUNTIME_SCHEMA,
            **asdict(self),
            "total_signature_hashes": self.total_signature_hashes,
            "stage_topology": self.stage_topology(),
            "within_partition_near_dedup": True,
            "global_cross_partition_dedup_claimed": False,
            "output_semantics": (
                "datatrove_jsonl_preserving_id_text_and_remaining_fields_as_metadata"
            ),
        }
        return {
            **core,
            "plan_sha256": hashlib.sha256(_canonical_json_bytes(core)).hexdigest(),
        }


def _assert_runtime_version(expected: str) -> None:
    try:
        installed = version("datatrove")
    except PackageNotFoundError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "DataTrove MinHash execution requires the project data-scale optional dependencies"
        ) from exc
    if installed != expected:
        raise RuntimeError(
            f"DataTrove runtime version mismatch: expected {expected}, got {installed}"
        )


def build_datatrove_minhash_executors(spec: DataTroveMinhashSpec):
    """Build the four maintained DataTrove MinHash stages without executing them."""

    _assert_runtime_version(spec.datatrove_version)
    try:
        from datatrove.executor import LocalPipelineExecutor
        from datatrove.pipeline.dedup import MinhashDedupSignature
        from datatrove.pipeline.dedup.minhash import (
            MinhashConfig,
            MinhashDedupBuckets,
            MinhashDedupCluster,
            MinhashDedupFilter,
        )
        from datatrove.pipeline.readers import JsonlReader
        from datatrove.pipeline.writers.jsonl import JsonlWriter
        from datatrove.utils.hashing import HashConfig
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "DataTrove MinHash execution requires the project data-scale optional dependencies"
        ) from exc

    config = MinhashConfig(
        n_grams=spec.n_grams,
        num_buckets=spec.num_buckets,
        hashes_per_bucket=spec.hashes_per_bucket,
        seed=spec.seed,
        hash_config=HashConfig(precision=spec.hash_precision),
    )
    signatures = _join_uri(spec.work_uri, "signatures")
    buckets = _join_uri(spec.work_uri, "buckets")
    remove_ids = _join_uri(spec.work_uri, "remove_ids")
    removed = _join_uri(spec.work_uri, "removed")

    signature_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=spec.input_uri,
                add_file_path=False,
                shuffle_files=False,
            ),
            MinhashDedupSignature(
                output_folder=signatures,
                config=config,
                language=spec.language,
            ),
        ],
        tasks=spec.signature_tasks,
        workers=spec.workers,
        logging_dir=_join_uri(spec.logging_uri, "01_signatures"),
    )
    bucket_executor = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=signatures,
                output_folder=buckets,
                config=config,
            )
        ],
        tasks=spec.num_buckets,
        workers=min(spec.workers, spec.num_buckets),
        logging_dir=_join_uri(spec.logging_uri, "02_buckets"),
    )
    cluster_executor = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=buckets,
                output_folder=remove_ids,
                config=config,
            )
        ],
        tasks=1,
        workers=1,
        logging_dir=_join_uri(spec.logging_uri, "03_cluster"),
    )
    filter_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=spec.input_uri,
                add_file_path=False,
                shuffle_files=False,
            ),
            MinhashDedupFilter(
                input_folder=remove_ids,
                exclusion_writer=JsonlWriter(output_folder=removed),
            ),
            JsonlWriter(output_folder=spec.output_uri),
        ],
        tasks=spec.signature_tasks,
        workers=spec.workers,
        logging_dir=_join_uri(spec.logging_uri, "04_filter"),
    )
    return signature_executor, bucket_executor, cluster_executor, filter_executor


def run_datatrove_minhash(spec: DataTroveMinhashSpec) -> None:
    """Execute the four MinHash stages sequentially and retain their logs/artifacts."""

    for executor in build_datatrove_minhash_executors(spec):
        executor.run()
