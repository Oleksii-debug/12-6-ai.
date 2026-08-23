"""Backend-neutral scalable ingestion plans with an optional DataTrove executor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

DATATROVE_VERSION = "0.10.0"


class ScalableIngestionError(ValueError):
    """Raised when scalable ingestion configuration is unsafe or inconsistent."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()


def _validate_uri(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ScalableIngestionError(f"{field} must be a non-empty string")
    parsed = urlsplit(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ScalableIngestionError(f"{field} must not contain credentials/query/fragment")


@dataclass(frozen=True)
class DataTroveParquetPlan:
    """Deterministic plan for staging one immutable source snapshot to Parquet."""

    source_id: str
    source_version: str
    snapshot_sha256: str
    registry_identity_sha256: str
    input_uri: str
    input_format: str
    output_uri: str
    logging_uri: str
    tasks: int = 1
    workers: int = 1
    datatrove_version: str = DATATROVE_VERSION

    def __post_init__(self) -> None:
        for field in ("source_id", "source_version", "snapshot_sha256", "registry_identity_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ScalableIngestionError(f"{field} must be a non-empty string")
        for field in ("snapshot_sha256", "registry_identity_sha256"):
            value = getattr(self, field)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
                raise ScalableIngestionError(f"{field} must be a SHA-256 hex digest")
        if self.input_format not in {"jsonl", "parquet"}:
            raise ScalableIngestionError("input_format must be jsonl or parquet")
        for field in ("input_uri", "output_uri", "logging_uri"):
            _validate_uri(getattr(self, field), field)
        if self.output_uri == self.logging_uri:
            raise ScalableIngestionError("output_uri and logging_uri must differ")
        for field in ("tasks", "workers"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ScalableIngestionError(f"{field} must be a positive integer")
        if self.workers > self.tasks:
            raise ScalableIngestionError("workers cannot exceed tasks")
        if self.datatrove_version != DATATROVE_VERSION:
            raise ScalableIngestionError(
                f"plan must pin DataTrove {DATATROVE_VERSION} until compatibility is revalidated"
            )

    def manifest(self) -> dict[str, Any]:
        core = {
            "schema_version": "12-6.datatrove-parquet-plan.v1",
            **asdict(self),
            "output_format": "parquet",
            "skip_completed": True,
        }
        return {
            **core,
            "plan_sha256": hashlib.sha256(_canonical_json_bytes(core)).hexdigest(),
        }


def build_datatrove_executor(plan: DataTroveParquetPlan):
    """Build a DataTrove 0.10.0 local executor without making DataTrove a base dependency."""

    try:
        from datatrove.executor import LocalPipelineExecutor
        from datatrove.pipeline.readers import JsonlReader, ParquetReader
        from datatrove.pipeline.writers import ParquetWriter
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "DataTrove execution requires optional dependency datatrove[io]==0.10.0"
        ) from exc

    metadata = {
        "twelve_six_source_id": plan.source_id,
        "twelve_six_source_version": plan.source_version,
        "twelve_six_snapshot_sha256": plan.snapshot_sha256,
        "twelve_six_registry_identity_sha256": plan.registry_identity_sha256,
    }
    reader_class = JsonlReader if plan.input_format == "jsonl" else ParquetReader
    reader = reader_class(data_folder=plan.input_uri, default_metadata=metadata)
    writer = ParquetWriter(output_folder=plan.output_uri)
    return LocalPipelineExecutor(
        pipeline=[reader, writer],
        logging_dir=plan.logging_uri,
        tasks=plan.tasks,
        workers=plan.workers,
        skip_completed=True,
    )
