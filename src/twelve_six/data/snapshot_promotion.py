"""DATA-181 exact-object snapshot promotion through DATA-24/D03 gates.

This module does not decide rights. It consumes the already accepted DATA-21/22
object identities and the DATA-24 machine-readable rights registry. Promotion is
fail-closed on raw bytes, extraction output, evidence hashes, source manifests,
and repeated acquisition identity.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from twelve_six.data.external_sources import (
    EligibilityResolver,
    ExternalSourceSpec,
    verify_local_snapshot,
    validate_external_source_registry,
)
from twelve_six.data.multilingual_pretraining import (
    MultilingualDataError,
    PretrainingRecord,
    admit_for_pretraining,
    strict_normalize_utf8,
)
from twelve_six.data.pipeline import PipelineConfig, build_dataset
from twelve_six.data.source_intake import DownloadedBytes, Fetcher, bounded_http_fetch, extract_text

PROMOTION_SCHEMA = "12-6.data181-real-snapshot-promotion.v1"
REPORT_SCHEMA = "12-6.data181-real-snapshot-promotion-report.v1"
DATA24_REGISTRY_SCHEMA = "12-6.external-source-registry.v2"
DATA21_INTAKE_MANIFEST_SHA256 = "9d50c0baf98247c1babc5fca8dead5b1fa87264ad92ea62527c34e342a7dd735"
CANDIDATE_REGISTRY_IDENTITY_SHA256 = "678d250ac9910f58ab1b9113cf713a2fea52a6a21e7a8434e6434d95a8045214"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SnapshotPromotionError(ValueError):
    """Raised when DATA-181 cannot prove exact promotion identity."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise SnapshotPromotionError(f"{field} must be lowercase SHA-256")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SnapshotPromotionError(f"{path}: JSON root must be an object")
    return value


def load_promotion_plan(path: str | Path) -> dict[str, Any]:
    plan = _load_json(Path(path))
    if plan.get("schema_version") != PROMOTION_SCHEMA:
        raise SnapshotPromotionError("unsupported DATA-181 promotion plan schema")
    accepted = plan.get("accepted_intake")
    if not isinstance(accepted, Mapping):
        raise SnapshotPromotionError("accepted_intake must be an object")
    if accepted.get("intake_manifest_sha256") != DATA21_INTAKE_MANIFEST_SHA256:
        raise SnapshotPromotionError("DATA-21/22 intake manifest identity drift")
    if (
        accepted.get("candidate_registry_identity_sha256")
        != CANDIDATE_REGISTRY_IDENTITY_SHA256
    ):
        raise SnapshotPromotionError("DATA-21/22 candidate registry identity drift")
    if plan.get("local_free_only") is not True:
        raise SnapshotPromotionError("DATA-181 must remain LOCAL_FREE")
    objects = plan.get("objects")
    if not isinstance(objects, list) or len(objects) != 3:
        raise SnapshotPromotionError("DATA-181 requires exactly three accepted object promotions")
    keys: set[tuple[str, str]] = set()
    raw_hashes: set[str] = set()
    for item in objects:
        if not isinstance(item, dict):
            raise SnapshotPromotionError("promotion objects must be mappings")
        key = (str(item.get("promoted_source_id", "")), str(item.get("source_version", "")))
        if not all(key) or key in keys:
            raise SnapshotPromotionError("promoted source_id/source_version must be unique")
        keys.add(key)
        raw_sha = _require_sha(item.get("raw_sha256"), "raw_sha256")
        _require_sha(item.get("normalized_sha256"), "normalized_sha256")
        _require_sha(item.get("source_manifest_sha256"), "source_manifest_sha256")
        if raw_sha in raw_hashes:
            raise SnapshotPromotionError("accepted raw object identities must be unique")
        raw_hashes.add(raw_sha)
        for field in ("raw_bytes", "normalized_utf8_bytes"):
            value = item.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SnapshotPromotionError(f"{field} must be a positive integer")
    return plan


def _file_uri_path(repo_root: Path, uri: str) -> Path:
    parsed = urlsplit(uri)
    if parsed.scheme != "file":
        raise SnapshotPromotionError(f"DATA-181 expects file evidence/snapshot URI, got {uri!r}")
    raw = parsed.path
    if parsed.netloc:
        raw = f"{parsed.netloc}{raw}"
    raw = raw.lstrip("/")
    if not raw:
        raise SnapshotPromotionError(f"empty file URI path: {uri!r}")
    resolved = (repo_root / raw).resolve()
    root = repo_root.resolve()
    if root not in resolved.parents and resolved != root:
        raise SnapshotPromotionError(f"file URI escapes repository root: {uri!r}")
    return resolved


def load_and_verify_registry(
    repo_root: str | Path, plan: Mapping[str, Any], registry_path: str | Path
) -> tuple[dict[str, Any], tuple[ExternalSourceSpec, ...]]:
    root = Path(repo_root)
    registry = _load_json(Path(registry_path))
    if registry.get("schema_version") != DATA24_REGISTRY_SCHEMA:
        raise SnapshotPromotionError("DATA-181 requires DATA-24 registry v2")
    sources = validate_external_source_registry(registry)
    if registry.get("registry_identity_sha256") != plan.get(
        "canonical_registry_identity_sha256"
    ):
        raise SnapshotPromotionError("promotion plan/registry identity mismatch")

    planned = {
        (item["promoted_source_id"], item["source_version"]): item
        for item in plan["objects"]
    }
    actual = {(source.source_id, source.source_version): source for source in sources}
    if set(actual) != set(planned):
        raise SnapshotPromotionError("registry promoted source set differs from promotion plan")
    for key, item in planned.items():
        source = actual[key]
        if source.source_manifest_sha256 != item["source_manifest_sha256"]:
            raise SnapshotPromotionError(f"{key}: source manifest identity mismatch")
        if source.snapshot.sha256 != item["raw_sha256"]:
            raise SnapshotPromotionError(f"{key}: snapshot raw SHA mismatch")
        if source.snapshot.size_bytes != item["raw_bytes"]:
            raise SnapshotPromotionError(f"{key}: snapshot raw size mismatch")
        for evidence in source.rights.evidence_refs:
            evidence_path = _file_uri_path(root, evidence.uri)
            payload = evidence_path.read_bytes()
            if _sha256_bytes(payload) != evidence.sha256:
                raise SnapshotPromotionError(
                    f"{key}: rights evidence hash mismatch for {evidence.evidence_id}"
                )
    return registry, sources


def _verify_download(downloaded: DownloadedBytes, item: Mapping[str, Any], label: str) -> None:
    if len(downloaded.payload) != item["raw_bytes"]:
        raise SnapshotPromotionError(
            f"{item['promoted_source_id']}: {label} raw size drift "
            f"{len(downloaded.payload)} != {item['raw_bytes']}"
        )
    actual = _sha256_bytes(downloaded.payload)
    if actual != item["raw_sha256"]:
        raise SnapshotPromotionError(
            f"{item['promoted_source_id']}: {label} raw SHA drift {actual}"
        )


def _normalized_object(downloaded: DownloadedBytes, item: Mapping[str, Any]) -> tuple[str, str]:
    extracted, encoding = extract_text(downloaded, str(item["adapter"]))
    bounded = extracted[:50_000]
    normalized, _profile = strict_normalize_utf8(bounded)
    payload = normalized.encode("utf-8")
    if _sha256_bytes(payload) != item["normalized_sha256"]:
        raise SnapshotPromotionError(
            f"{item['promoted_source_id']}: extraction/normalization SHA drift"
        )
    if len(payload) != item["normalized_utf8_bytes"]:
        raise SnapshotPromotionError(
            f"{item['promoted_source_id']}: extraction/normalization size drift"
        )
    return normalized, encoding


def _chunk_text(text: str, *, max_chars: int = 1200, min_chars: int = 80) -> tuple[str, ...]:
    """Deterministic generic natural-text chunking before normal admission gates."""
    if max_chars < min_chars or min_chars < 20:
        raise SnapshotPromotionError("invalid chunk limits")
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            value = "\n".join(current).strip()
            if len(value) >= min_chars:
                chunks.append(value)
            current, current_len = [], 0

    for paragraph in paragraphs:
        pieces: list[str] = []
        if len(paragraph) <= max_chars:
            pieces = [paragraph]
        else:
            words = paragraph.split()
            part: list[str] = []
            part_len = 0
            for word in words:
                needed = len(word) if not part else len(word) + 1
                if part and part_len + needed > max_chars:
                    pieces.append(" ".join(part))
                    part = [word]
                    part_len = len(word)
                else:
                    part.append(word)
                    part_len += needed
            if part:
                pieces.append(" ".join(part))

        for piece in pieces:
            needed = len(piece) if not current else len(piece) + 1
            if current and current_len + needed > max_chars:
                flush()
            current.append(piece)
            current_len += needed
    flush()
    return tuple(chunks)


def _write_source_jsonl(
    path: Path, source: ExternalSourceSpec, admitted: list[dict[str, Any]]
) -> str:
    payload = b""
    for index, record in enumerate(admitted):
        row = {
            "document_id": f"{source.source_id}-{index:04d}",
            "language_hint": record["language"],
            "text": record["text"],
        }
        payload += _canonical_json_bytes(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def _build_d03_input(
    evidence_dir: Path,
    sources: Mapping[tuple[str, str], ExternalSourceSpec],
    admitted_by_source: Mapping[tuple[str, str], list[dict[str, Any]]],
    registry_identity: str,
) -> tuple[Path, Path]:
    input_dir = evidence_dir / "small-corpus-input"
    raw_dir = input_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for key in sorted(admitted_by_source):
        source = sources[key]
        admitted = admitted_by_source[key]
        if not admitted:
            raise SnapshotPromotionError(f"{source.source_id}: no chunks passed normal DATA-24 gate")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source.source_id)
        raw_rel = f"raw/{safe_name}.jsonl"
        raw_sha = _write_source_jsonl(input_dir / raw_rel, source, admitted)
        entries.append(
            {
                "source_id": source.source_id,
                "purpose": "pretraining",
                "raw_path": raw_rel,
                "content_sha256": raw_sha,
                "license": {
                    "status": source.rights.status,
                    "license_id": source.rights.license_id,
                    "policy_ref": source.rights.policy_ref,
                    "rights_evidence_ids": [
                        item.evidence_id for item in source.rights.evidence_refs
                    ],
                },
                "provenance": {
                    "synthetic": False,
                    "external_source": True,
                    "provider": source.provider,
                    "source_version": source.source_version,
                    "source_manifest_sha256": source.source_manifest_sha256,
                    "data24_registry_identity_sha256": registry_identity,
                },
            }
        )

    source_registry = {
        "schema_version": 1,
        "dataset_id": "data181-real-promoted-small-v1",
        "sources": entries,
    }
    contamination = {
        "schema_version": 1,
        "forbidden_normalized_sha256": [],
        "forbidden_source_purposes": [
            "benchmark",
            "evaluation_test",
            "heldout_test",
            "evaluation",
            "validation",
            "test",
        ],
    }
    source_registry_path = input_dir / "source_registry.json"
    contamination_path = input_dir / "contamination_registry.json"
    source_registry_path.write_bytes(_canonical_json_bytes(source_registry))
    contamination_path.write_bytes(_canonical_json_bytes(contamination))
    return source_registry_path, contamination_path


def _corpus_output_identity(path: Path) -> dict[str, str]:
    return {
        name: _sha256_bytes((path / name).read_bytes())
        for name in ("train.jsonl", "validation.jsonl", "manifest.json")
    }


def promote_snapshots(
    *,
    repo_root: str | Path,
    plan_path: str | Path,
    registry_path: str | Path,
    evidence_dir: str | Path,
    fetcher: Fetcher = bounded_http_fetch,
    max_download_bytes: int = 2_000_000,
) -> dict[str, Any]:
    """Acquire twice, materialize exact snapshots, admit normally, and build twice."""
    root = Path(repo_root)
    plan = load_promotion_plan(plan_path)
    registry, source_tuple = load_and_verify_registry(root, plan, registry_path)
    resolver = EligibilityResolver(registry)
    sources = {(item.source_id, item.source_version): item for item in source_tuple}
    output = Path(evidence_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalized_dir = output / "normalized"
    normalized_dir.mkdir(exist_ok=True)
    admitted_by_source: dict[tuple[str, str], list[dict[str, Any]]] = {}
    object_reports: list[dict[str, Any]] = []

    for item in plan["objects"]:
        key = (item["promoted_source_id"], item["source_version"])
        source = sources[key]
        decision = resolver.assert_model_training_eligible(
            source.source_id, source.source_version, source.source_manifest_sha256
        )
        first = fetcher(item["acquisition_url"], max_download_bytes)
        repeat = fetcher(item["acquisition_url"], max_download_bytes)
        _verify_download(first, item, "first acquisition")
        _verify_download(repeat, item, "repeat acquisition")
        if first.payload != repeat.payload:
            raise SnapshotPromotionError(
                f"{source.source_id}: repeat acquisition bytes differ despite checks"
            )

        snapshot_path = _file_uri_path(root, source.snapshot.uri)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(first.payload)
        verify_local_snapshot(source.snapshot, snapshot_path)

        normalized, encoding = _normalized_object(first, item)
        repeated_normalized, _ = _normalized_object(repeat, item)
        if repeated_normalized != normalized:
            raise SnapshotPromotionError(
                f"{source.source_id}: repeat extraction/normalization identity changed"
            )
        full_text_path = normalized_dir / f"{item['record_id']}.txt"
        full_text_path.write_text(normalized + "\n", encoding="utf-8", newline="\n")

        accepted_chunks: list[dict[str, Any]] = []
        rejected_chunks: list[dict[str, str]] = []
        for index, chunk in enumerate(_chunk_text(normalized)):
            record = PretrainingRecord(
                record_id=f"{item['record_id']}-chunk-{index:04d}",
                source_id=source.source_id,
                source_version=source.source_version,
                source_manifest_sha256=source.source_manifest_sha256,
                split="train",
                source_purpose="pretraining",
                modality="natural",
                text=chunk,
                language_hint=item["language"],
                external=True,
                rights_status=source.rights.status,
                allows_model_training=True,
            )
            try:
                admitted = admit_for_pretraining(record, eligibility_resolver=resolver)
            except MultilingualDataError as exc:
                rejected_chunks.append(
                    {"record_id": record.record_id, "reason": f"{type(exc).__name__}: {exc}"}
                )
                continue
            accepted_chunks.append(
                {
                    "record_id": admitted.record_id,
                    "language": admitted.language,
                    "normalized_sha256": admitted.normalized_sha256,
                    "text": admitted.normalized_text,
                }
            )
        if not accepted_chunks:
            raise SnapshotPromotionError(
                f"{source.source_id}: no normalized chunks passed normal DATA-24 admission"
            )
        admitted_by_source[key] = accepted_chunks
        admission_core = [
            {
                "record_id": row["record_id"],
                "language": row["language"],
                "normalized_sha256": row["normalized_sha256"],
            }
            for row in accepted_chunks
        ]
        object_reports.append(
            {
                "promoted_source_id": source.source_id,
                "source_version": source.source_version,
                "parent_source_id": item["parent_source_id"],
                "parent_source_identity_sha256": item["parent_source_identity_sha256"],
                "data21_record_id": item["record_id"],
                "raw_sha256": item["raw_sha256"],
                "raw_bytes": item["raw_bytes"],
                "normalized_sha256": item["normalized_sha256"],
                "normalized_utf8_bytes": item["normalized_utf8_bytes"],
                "decoded_encoding": encoding,
                "source_manifest_sha256": source.source_manifest_sha256,
                "registry_identity_sha256": decision.registry_identity_sha256,
                "rights_uses": {
                    "acquisition": decision.acquisition,
                    "storage": decision.storage,
                    "analysis": decision.analysis,
                    "model_training": decision.model_training,
                    "redistribution": decision.redistribution,
                },
                "rights_evidence_ids": list(decision.evidence_ids),
                "repeat_acquisition_same_raw_identity": True,
                "repeat_extraction_same_normalized_identity": True,
                "admitted_chunk_count": len(accepted_chunks),
                "rejected_chunk_count": len(rejected_chunks),
                "admitted_chunk_identity_sha256": _sha256_bytes(
                    _canonical_json_bytes(admission_core)
                ),
                "rejected_chunks": rejected_chunks,
                "snapshot_path": str(snapshot_path.relative_to(root)),
                "normalized_path": str(full_text_path.relative_to(output)),
            }
        )

    source_registry_path, contamination_path = _build_d03_input(
        output, sources, admitted_by_source, registry["registry_identity_sha256"]
    )
    config = PipelineConfig(
        split_seed="12-6-data181-real-small-v1",
        validation_fraction=0.20,
        min_chars=60,
        max_chars=1600,
        min_alpha_ratio=0.35,
        near_duplicate_threshold=0.92,
        near_duplicate_shingle_words=5,
        tiny_near_dedup_max_documents=5000,
    )
    first_build = output / "small-corpus-build-a"
    second_build = output / "small-corpus-build-b"
    first_manifest = build_dataset(
        source_registry_path, contamination_path, first_build, config=config
    )
    second_manifest = build_dataset(
        source_registry_path, contamination_path, second_build, config=config
    )
    first_outputs = _corpus_output_identity(first_build)
    second_outputs = _corpus_output_identity(second_build)
    if first_manifest["dataset_identity_sha256"] != second_manifest["dataset_identity_sha256"]:
        raise SnapshotPromotionError("repeat small-corpus build dataset identity drift")
    if first_outputs != second_outputs:
        raise SnapshotPromotionError("repeat small-corpus build output identity drift")

    core = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS",
        "authority_boundary": (
            "EXACT_DATA21_22_OBJECTS_PROMOTED_THROUGH_DATA24_RIGHTS_AND_NORMAL_D03_GATES"
        ),
        "local_free_only": True,
        "data21_22": plan["accepted_intake"],
        "data24": plan["rights_gate"],
        "canonical_registry_identity_sha256": registry["registry_identity_sha256"],
        "objects": object_reports,
        "small_corpus": {
            "dataset_identity_sha256": first_manifest["dataset_identity_sha256"],
            "stats": first_manifest["stats"],
            "output_sha256": first_outputs,
            "repeat_build_same_identity": True,
            "source_registry_sha256": _sha256_bytes(source_registry_path.read_bytes()),
            "contamination_registry_sha256": _sha256_bytes(contamination_path.read_bytes()),
            "gate_path": "DATA24 EligibilityResolver -> admit_for_pretraining -> incumbent D03 build_dataset",
        },
        "claims_explicitly_absent": [
            "representative_corpus",
            "production_corpus_freeze",
            "benchmark_clean_universal",
            "intelligence",
            "production_readiness",
            "alignment",
            "instruction_following",
        ],
    }
    report = {**core, "report_sha256": _sha256_bytes(_canonical_json_bytes(core))}
    (output / "promotion-report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
