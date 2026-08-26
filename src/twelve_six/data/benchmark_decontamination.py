"""Auditable D06/D03 benchmark decontamination binding.

This module does not define benchmark semantics and does not introduce a second
near-deduplication engine. D06's ``12-6.benchmark-registry.v1`` is authoritative;
near matching is executed by the incumbent DataTrove 0.10.0 MinHash machinery
from :mod:`twelve_six.data.dedup_scale`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .corpus_foundation import CorpusFoundationError, reserved_registry_from_d06_manifest
from .dedup_scale import (
    DataTroveMinhashExecutionPlan,
    UnicodeRegexWordTokenizer,
    datatrove_minhash_config,
    validate_datatrove_runtime,
)

REFERENCE_BUNDLE_SCHEMA = "12-6.decontamination-reference-bundle.v1"
DECONTAMINATION_REPORT_SCHEMA = "12-6.benchmark-decontamination-report.v1"
PUBLICATION_MANIFEST_SCHEMA = "12-6.corpus-publication-manifest.v1"


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
    if len(text) != 64 or text != text.lower() or any(c not in "0123456789abcdef" for c in text):
        raise CorpusFoundationError(f"{field} must be lowercase SHA-256 hex")
    return text


def _record_value(record: Mapping[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata.get(key)
    return None


def _require_record_id(record: Mapping[str, Any], prefix: str) -> tuple[str, str]:
    source_id = _require_text(_record_value(record, "source_id"), f"{prefix}.source_id")
    document_id = _require_text(_record_value(record, "id"), f"{prefix}.id")
    return source_id, document_id


def benchmark_registry_identity(manifest: Mapping[str, Any]) -> str:
    """Validate through the incumbent D03 D06 bridge and return D06 identity."""

    reserved_registry_from_d06_manifest(manifest, {})
    return _require_sha256(manifest.get("manifest_sha256"), "benchmark_registry.manifest_sha256")


@dataclass(frozen=True)
class ReferenceRecord:
    source_id: str
    document_id: str
    content_sha256: str
    category: str
    evidence_ref: str

    def __post_init__(self) -> None:
        _require_text(self.source_id, "source_id")
        _require_text(self.document_id, "document_id")
        _require_sha256(self.content_sha256, "content_sha256")
        _require_text(self.category, "category")
        _require_text(self.evidence_ref, "evidence_ref")


def build_reference_bundle(
    *,
    benchmark_registry: Mapping[str, Any],
    references: Sequence[ReferenceRecord],
    rights_evidence_refs: Sequence[str],
    unavailable_probe_count: int = 0,
) -> dict[str, Any]:
    """Freeze all locally checkable held-out material without embedding its text."""

    registry_sha = benchmark_registry_identity(benchmark_registry)
    if isinstance(unavailable_probe_count, bool) or not isinstance(unavailable_probe_count, int):
        raise CorpusFoundationError("unavailable_probe_count must be an integer")
    if unavailable_probe_count < 0:
        raise CorpusFoundationError("unavailable_probe_count must be non-negative")
    rights = sorted({_require_text(value, "rights_evidence_ref") for value in rights_evidence_refs})
    if references and not rights:
        raise CorpusFoundationError("reference material requires local-check rights evidence")
    ordered = sorted(references, key=lambda item: (item.source_id, item.document_id, item.content_sha256))
    seen: set[tuple[str, str]] = set()
    for item in ordered:
        key = (item.source_id, item.document_id)
        if key in seen:
            raise CorpusFoundationError("duplicate decontamination reference document identity")
        seen.add(key)
    core = {
        "schema_version": REFERENCE_BUNDLE_SCHEMA,
        "benchmark_registry_sha256": registry_sha,
        "references": [asdict(item) for item in ordered],
        "rights_evidence_refs": rights,
        "unavailable_probe_count": unavailable_probe_count,
        "reference_count": len(ordered),
        "text_embedded": False,
    }
    return {**core, "reference_bundle_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def exact_matches(
    candidates: Iterable[Mapping[str, Any]],
    references: Sequence[ReferenceRecord],
) -> list[dict[str, Any]]:
    """Return deterministic exact benchmark/held-out matches with both identities."""

    by_hash: dict[str, list[ReferenceRecord]] = {}
    for reference in references:
        by_hash.setdefault(reference.content_sha256, []).append(reference)
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        source_id, document_id = _require_record_id(candidate, "candidate")
        digest = _require_sha256(candidate.get("content_sha256"), "candidate.content_sha256")
        for reference in by_hash.get(digest, ()):
            matches.append(
                {
                    "match_type": "exact_content_sha256",
                    "candidate_source_id": source_id,
                    "candidate_document_id": document_id,
                    "candidate_content_sha256": digest,
                    "reference_source_id": reference.source_id,
                    "reference_document_id": reference.document_id,
                    "reference_content_sha256": reference.content_sha256,
                    "reference_category": reference.category,
                    "decision": "REJECT_FROM_TRAINING",
                }
            )
    return sorted(
        matches,
        key=lambda item: (
            item["candidate_source_id"],
            item["candidate_document_id"],
            item["reference_source_id"],
            item["reference_document_id"],
        ),
    )


def near_match_records(
    removed_records: Iterable[Mapping[str, Any]],
    *,
    reference_bundle_sha256: str,
) -> list[dict[str, Any]]:
    """Describe DataTrove reference-index removals without inventing pair attribution.

    DataTrove 0.10.0's public filter output identifies the rejected candidate.
    The maintained index scope is cryptographically frozen, but the filter output
    does not expose the paired reference document identity. We record that limit
    explicitly instead of fabricating a pair.
    """

    scope_sha = _require_sha256(reference_bundle_sha256, "reference_bundle_sha256")
    rows: list[dict[str, Any]] = []
    for record in removed_records:
        source_id, document_id = _require_record_id(record, "near_removed")
        digest = _require_sha256(
            _record_value(record, "content_sha256"), "near_removed.content_sha256"
        )
        rows.append(
            {
                "match_type": "datatrove_minhash_reference_index",
                "candidate_source_id": source_id,
                "candidate_document_id": document_id,
                "candidate_content_sha256": digest,
                "reference_bundle_sha256": scope_sha,
                "reference_document_id": None,
                "decision": "REJECT_FROM_TRAINING",
                "attribution_limit": (
                    "DataTrove 0.10.0 filter output proves membership in the frozen reference "
                    "index match set but does not expose the paired reference document ID."
                ),
            }
        )
    return sorted(rows, key=lambda item: (item["candidate_source_id"], item["candidate_document_id"]))


def build_decontamination_report(
    *,
    benchmark_registry_sha256: str,
    reference_bundle_sha256: str,
    candidate_manifest_sha256: str,
    exact_match_rows: Sequence[Mapping[str, Any]],
    near_match_rows: Sequence[Mapping[str, Any]],
    known_semantic_match_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a publication gate report over one exact candidate/reference identity."""

    registry_sha = _require_sha256(benchmark_registry_sha256, "benchmark_registry_sha256")
    bundle_sha = _require_sha256(reference_bundle_sha256, "reference_bundle_sha256")
    candidate_sha = _require_sha256(candidate_manifest_sha256, "candidate_manifest_sha256")

    exact_rows = [dict(item) for item in exact_match_rows]
    near_rows = [dict(item) for item in near_match_rows]
    semantic_rows = [dict(item) for item in known_semantic_match_rows]
    unresolved_match_rows = [
        item
        for item in [*exact_rows, *near_rows, *semantic_rows]
        if item.get("decision") != "REJECT_FROM_TRAINING"
    ]
    publication_eligible = not unresolved_match_rows
    core = {
        "schema_version": DECONTAMINATION_REPORT_SCHEMA,
        "benchmark_registry_sha256": registry_sha,
        "reference_bundle_sha256": bundle_sha,
        "candidate_manifest_sha256": candidate_sha,
        "exact_matches": exact_rows,
        "near_matches": near_rows,
        "known_semantic_matches": semantic_rows,
        "counts": {
            "exact_matches": len(exact_rows),
            "near_matches": len(near_rows),
            "known_semantic_matches": len(semantic_rows),
            "unresolved_registered_matches": len(unresolved_match_rows),
        },
        "publication_eligible": publication_eligible,
        "semantic_universal_cleanliness_claimed": False,
        "residual_semantic_overlap_status": "UNKNOWN_BEYOND_REGISTERED_EXCLUSIONS",
        "limitations": [
            "Lexical MinHash cannot establish universal semantic cleanliness.",
            "Cross-language translations and paraphrases can survive lexical MinHash.",
            "Near-match paired reference document attribution is not exposed by the incumbent DataTrove filter output.",
            "Coverage is limited to the exact D06 registry identity and locally checkable reference bundle recorded here.",
        ],
    }
    return {**core, "decontamination_report_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def assert_fresh_decontamination(
    report: Mapping[str, Any], *, current_benchmark_registry_sha256: str
) -> None:
    """Fail closed when D06 registry identity differs from the completed pass."""

    if report.get("schema_version") != DECONTAMINATION_REPORT_SCHEMA:
        raise CorpusFoundationError("unsupported benchmark decontamination report schema")
    expected = _require_sha256(
        report.get("benchmark_registry_sha256"), "report.benchmark_registry_sha256"
    )
    current = _require_sha256(current_benchmark_registry_sha256, "current_benchmark_registry_sha256")
    if expected != current:
        raise CorpusFoundationError(
            "benchmark registry identity changed; a fresh decontamination pass is required"
        )


def build_corpus_publication_manifest(
    *,
    corpus_manifest_sha256: str,
    decontamination_report: Mapping[str, Any],
    current_benchmark_registry_sha256: str,
    output_files: Mapping[str, str],
) -> dict[str, Any]:
    """Bind publication to the exact D06 registry and completed decontamination pass."""

    assert_fresh_decontamination(
        decontamination_report,
        current_benchmark_registry_sha256=current_benchmark_registry_sha256,
    )
    corpus_sha = _require_sha256(corpus_manifest_sha256, "corpus_manifest_sha256")
    report_sha = _require_sha256(
        decontamination_report.get("decontamination_report_sha256"),
        "decontamination_report.decontamination_report_sha256",
    )
    outputs = {
        _require_text(name, "output_file_name"): _require_sha256(digest, f"output_files[{name}]")
        for name, digest in output_files.items()
    }
    if decontamination_report.get("publication_eligible") is not True:
        raise CorpusFoundationError("decontamination report does not permit corpus publication")
    core = {
        "schema_version": PUBLICATION_MANIFEST_SCHEMA,
        "corpus_manifest_sha256": corpus_sha,
        "decontamination_report_sha256": report_sha,
        "benchmark_registry_sha256": decontamination_report["benchmark_registry_sha256"],
        "reference_bundle_sha256": decontamination_report["reference_bundle_sha256"],
        "output_files": dict(sorted(outputs.items())),
        "publication_eligible": True,
        "semantic_universal_cleanliness_claimed": False,
    }
    return {**core, "publication_manifest_sha256": _sha256_bytes(_canonical_json_bytes(core))}


def run_datatrove_reference_filter(
    plan: DataTroveMinhashExecutionPlan,
    *,
    candidate_input: str | Path,
    workspace: str | Path,
    reference_index: str | Path,
) -> dict[str, Any]:
    """Run candidate-to-reference-only MinHash using the incumbent DataTrove engine."""

    validate_datatrove_runtime(plan)
    from datatrove.executor.local import LocalPipelineExecutor
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.dedup.minhash import (
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
    )
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers.jsonl import JsonlWriter

    workspace = Path(workspace)
    signatures = workspace / "candidate_reference_signatures"
    bucket_pairs = workspace / "candidate_reference_bucket_pairs"
    remove_ids = workspace / "candidate_reference_remove_ids"
    removed = workspace / "candidate_reference_removed"
    output = workspace / "candidate_reference_clean"
    logs = workspace / "logs" / "candidate_reference"
    config = datatrove_minhash_config(plan)

    stage1 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(str(candidate_input)),
            MinhashDedupSignature(
                output_folder=str(signatures),
                config=config,
                language=UnicodeRegexWordTokenizer(),
            ),
        ],
        tasks=plan.candidate_shards,
        workers=min(plan.workers, plan.candidate_shards),
        logging_dir=str(logs / "signatures"),
        skip_completed=True,
    )
    stage1.run()

    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=str(signatures),
                output_folder=str(bucket_pairs),
                index_folder=str(reference_index),
                config=config,
                only_dedup_in_index=True,
            )
        ],
        tasks=plan.num_buckets,
        workers=min(plan.workers, plan.num_buckets),
        logging_dir=str(logs / "buckets"),
        skip_completed=True,
    )
    stage2.run()

    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=str(bucket_pairs),
                output_folder=str(remove_ids),
                config=config,
            )
        ],
        tasks=1,
        workers=1,
        logging_dir=str(logs / "cluster"),
        skip_completed=True,
    )
    stage3.run()

    stage4 = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(str(candidate_input)),
            MinhashDedupFilter(
                input_folder=str(remove_ids),
                exclusion_writer=JsonlWriter(str(removed), compression=None),
            ),
            JsonlWriter(str(output), compression=None),
        ],
        tasks=plan.candidate_shards,
        workers=min(plan.workers, plan.candidate_shards),
        logging_dir=str(logs / "filter"),
        skip_completed=True,
    )
    stage4.run()
    return {
        "signatures": str(signatures),
        "bucket_pairs": str(bucket_pairs),
        "remove_ids": str(remove_ids),
        "removed": str(removed),
        "output": str(output),
        "reference_only": True,
        "skip_completed": True,
    }
