"""Data acquisition, provenance, filtering, deduplication, and corpus packaging."""

from .external_sources import (
    RIGHTS_APPROVED,
    RIGHTS_REJECTED,
    RIGHTS_REVIEW_REQUIRED,
    ExternalDataContractError,
    ExternalSourceSpec,
    ReservedSetSpec,
    RightsDecision,
    SnapshotSpec,
    build_external_source_registry,
    build_reserved_fingerprint_registry,
    contamination_report,
    validate_external_source_registry,
    validate_reserved_fingerprint_registry,
    verify_local_snapshot,
)
from .pipeline import DataContractError, PipelineConfig, build_dataset, language_id, normalize_text
from .scalable_ingestion import (
    DATATROVE_VERSION,
    DataTroveParquetPlan,
    ScalableIngestionError,
    build_datatrove_executor,
)

__all__ = [
    "DATATROVE_VERSION",
    "RIGHTS_APPROVED",
    "RIGHTS_REJECTED",
    "RIGHTS_REVIEW_REQUIRED",
    "DataContractError",
    "DataTroveParquetPlan",
    "ExternalDataContractError",
    "ExternalSourceSpec",
    "PipelineConfig",
    "ReservedSetSpec",
    "RightsDecision",
    "ScalableIngestionError",
    "SnapshotSpec",
    "build_dataset",
    "build_datatrove_executor",
    "build_external_source_registry",
    "build_reserved_fingerprint_registry",
    "contamination_report",
    "language_id",
    "normalize_text",
    "validate_external_source_registry",
    "validate_reserved_fingerprint_registry",
    "verify_local_snapshot",
]
