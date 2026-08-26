"""NEXT100-065E record-granular successor for CPython accepted chunks.

The V6 lane correctly limits CPython capacity to DATA-228 accepted chunks but
concatenates those chunks into one comparison payload. That representation can
hide a near/fragment match affecting only one accepted record and creates
synthetic cross-record shingles at join boundaries.

V7 keeps all V6 authorities and capacity arithmetic unchanged while exposing
each accepted CPython chunk as its own dedup object. No corpus/training
promotion is performed here.
"""
from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping
from typing import Any

from twelve_six.data import cross_source_capacity_audit_v3 as v3
from twelve_six.data import cross_source_capacity_audit_v5 as v5
from twelve_six.data import cross_source_capacity_audit_v6 as v6

SCHEMA = "12-6.next100-065e-cross-source-dedup-report.v7"
WORKER_ID = "NEXT100-065E-CPYTHON-RECORD-GRANULARITY"
CPYTHON_ACCEPTED_RECORD_COUNT = 14
EXPECTED_SOURCE_OBJECT_COUNT = 44


class CrossSourceV7Error(RuntimeError):
    """Fail-closed V7 granularity error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossSourceV7Error(message)


def _cpython_record_rows(
    spec: Mapping[str, Any],
    accepted_payloads: list[bytes],
    accepted_chunk_indexes: list[int],
) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    """Turn accepted CPython chunks into independent dedup objects."""
    _require(
        len(accepted_payloads) == CPYTHON_ACCEPTED_RECORD_COUNT,
        "CPython accepted record-count drift",
    )
    _require(
        len(accepted_chunk_indexes) == len(accepted_payloads),
        "CPython accepted chunk-index cardinality drift",
    )
    _require(
        len(set(accepted_chunk_indexes)) == len(accepted_chunk_indexes),
        "CPython accepted chunk indexes are not unique",
    )

    expected_hashes = list(spec["accepted_normalized_sha256"])
    actual_hashes = [v5._sha256(payload) for payload in accepted_payloads]
    _require(
        actual_hashes == expected_hashes,
        "CPython accepted payload identity/order drift",
    )

    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    family = str(spec["source_family"])

    for accepted_index, (chunk_index, payload) in enumerate(
        zip(accepted_chunk_indexes, accepted_payloads, strict=True)
    ):
        digest = actual_hashes[accepted_index]
        source_id = f"{spec['source_id']}.accepted-chunk-{chunk_index:02d}"
        _require(source_id not in payloads, f"duplicate CPython source id: {source_id}")
        rows.append(
            {
                "source_id": source_id,
                "source_family": family,
                "stable_origin_id": spec["stable_origin_id"],
                "stable_object_id": f"sha256:{digest}",
                "modality": spec["modality"],
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": (
                    f"NEXT100-037@{spec['head_sha']} workflow "
                    f"{spec['dedicated_workflow_run']} accepted-only "
                    f"chunk {chunk_index}"
                ),
                "declared_capacity_bytes": len(payload),
                "expected_raw_bytes": len(payload),
                "expected_raw_sha256": digest,
                "acquisition_url": (
                    "materialized-v7://cpython-data228-accepted/"
                    f"{chunk_index:02d}"
                ),
                "origin_key": (
                    f"github:python/cpython:{spec['upstream_commit']}:"
                    f"{spec['upstream_path']}:accepted-chunk:{chunk_index}"
                ),
            }
        )
        payloads[source_id] = payload
        evidence.append(
            {
                "source_id": source_id,
                "source_family": family,
                "original_chunk_index": chunk_index,
                "accepted_index": accepted_index,
                "normalized_bytes": len(payload),
                "normalized_sha256": digest,
                "raw_text_emitted": False,
            }
        )

    return rows, payloads, evidence


def _materialize_cpython_records(
    spec: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, bytes],
    int,
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Reproduce DATA-228 filtering and preserve accepted record boundaries."""
    _require(
        spec.get("normalization_policy") == v5.CPYTHON_POLICY,
        "CPython policy drift",
    )
    _require(
        spec.get("dedicated_workflow_conclusion") == "success",
        "CPython authority not green",
    )
    _require(
        spec.get("training") == "ALLOWED_ACCEPTED_CHUNKS_ONLY",
        "CPython accepted-only boundary drift",
    )
    _require(
        spec.get("evaluation") == "NOT_SEPARATELY_ADMITTED",
        "CPython evaluation boundary drift",
    )

    raw = v5.v1.fetch_exact_source(str(spec["acquisition_url"]))
    _require(len(raw) == int(spec["raw_bytes"]), "CPython raw byte-count drift")
    _require(v5._sha256(raw) == spec["raw_sha256"], "CPython raw SHA-256 drift")
    _require(v5._git_blob_sha1(raw) == spec["git_blob_sha1"], "CPython Git blob drift")

    text = raw.decode("utf-8", errors="strict")
    normalized = v5.normalize_text(text[: int(spec["truncate_chars"])])
    normalized_bytes = normalized.encode("utf-8")
    _require(
        len(normalized_bytes) == int(spec["normalized_source_bytes"]),
        "CPython normalized source byte-count drift",
    )
    _require(
        v5._sha256(normalized_bytes) == spec["normalized_source_sha256"],
        "CPython normalized source SHA-256 drift",
    )

    chunking = spec["chunking"]
    chunks = v5._chunk_text(
        normalized,
        max_chars=int(chunking["max_chars"]),
        min_chars=int(chunking["min_chars"]),
    )
    _require(len(chunks) == int(spec["chunk_count"]), "CPython chunk-count drift")

    quality_config = v5._cpython_quality_config(spec)
    accepted_payloads: list[bytes] = []
    accepted_chunk_indexes: list[int] = []
    reasons: Counter[str] = Counter()

    for chunk_index, chunk in enumerate(chunks):
        reason = v5._quality_reason(chunk, quality_config)
        if reason is not None:
            reasons[reason] += 1
            continue
        accepted_payloads.append(v5.normalize_text(chunk).encode("utf-8"))
        accepted_chunk_indexes.append(chunk_index)

    _require(
        len(accepted_payloads) == int(spec["accepted_chunk_count"]),
        "CPython accepted count drift",
    )
    _require(
        len(chunks) - len(accepted_payloads) == int(spec["rejected_chunk_count"]),
        "CPython rejected count drift",
    )
    _require(
        dict(sorted(reasons.items())) == spec["rejection_reasons"],
        "CPython rejection-reason drift",
    )

    rows, payloads, record_evidence = _cpython_record_rows(
        spec,
        accepted_payloads,
        accepted_chunk_indexes,
    )
    capacity = sum(len(payload) for payload in accepted_payloads)
    _require(
        0 < capacity < int(spec["normalized_source_bytes"]),
        "CPython accepted-only capacity invariant failed",
    )
    _require(capacity == 15540, "CPython accepted capacity drift")

    source_evidence = {
        "source_id": spec["source_id"],
        "raw_bytes": len(raw),
        "raw_sha256": v5._sha256(raw),
        "git_blob_sha1": v5._git_blob_sha1(raw),
        "normalized_source_bytes": len(normalized_bytes),
        "normalized_source_sha256": v5._sha256(normalized_bytes),
        "chunk_count": len(chunks),
        "accepted_chunk_count": len(accepted_payloads),
        "accepted_chunk_indexes": accepted_chunk_indexes,
        "accepted_capacity_bytes": capacity,
        "rejected_chunk_count": len(chunks) - len(accepted_payloads),
        "rejection_reasons": dict(sorted(reasons.items())),
        "normalization_policy": v5.CPYTHON_POLICY,
        "comparison_representation": "RECORD_GRANULAR_NO_SYNTHETIC_JOIN",
        "raw_text_emitted": False,
    }
    return rows, payloads, capacity, source_evidence, record_evidence


def _family_counts(report: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(modality): int(summary["declared_source_family_count"])
        for modality, summary in report["terminal_candidates"]["by_modality"].items()
    }


def audit_live(
    base_inventory: Mapping[str, Any],
    v4_extension: Mapping[str, Any],
    v5_config: Mapping[str, Any],
    v6_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run V6 authorities with record-granular CPython dedup semantics."""
    v6._validate_config(v6_config, v5_config)

    merged, inherited_payloads, inherited_evidence = v5._materialize_v4(
        base_inventory,
        v4_extension,
    )
    mdn_payload, mdn_evidence = v5._materialize_mdn(v5_config["mdn"])
    (
        cpython_rows,
        cpython_payloads,
        cpython_capacity,
        cpython_source_evidence,
        cpython_record_evidence,
    ) = _materialize_cpython_records(v5_config["cpython"])
    numpy_rows, numpy_payloads, numpy_evidence = v6._materialize_numpy(
        v6_config["numpy"]
    )
    pg_rows, pg_payloads, pg_evidence = v6._materialize_gutenberg(
        v6_config["gutenberg"]
    )

    mdn = v5_config["mdn"]
    mdn_row = {
        "source_id": mdn["source_id"],
        "source_family": mdn["source_family"],
        "stable_origin_id": mdn["stable_origin_id"],
        "stable_object_id": mdn["stable_object_id"],
        "modality": mdn["modality"],
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": (
            f"NEXT100-038@{mdn['head_sha']} workflow "
            f"{mdn['dedicated_workflow_run']}"
        ),
        "declared_capacity_bytes": mdn["normalized_bytes"],
        "expected_raw_bytes": mdn["normalized_bytes"],
        "expected_raw_sha256": mdn["normalized_sha256"],
        "acquisition_url": "materialized-v7://mdn-prose-only",
        "origin_key": (
            f"github:mdn/content:{mdn['upstream_commit']}:{mdn['upstream_path']}"
        ),
    }

    final_inventory = copy.deepcopy(merged)
    final_inventory["sources"] = [
        *copy.deepcopy(merged["sources"]),
        mdn_row,
        *cpython_rows,
        *numpy_rows,
        *pg_rows,
    ]
    final_inventory["final_refresh_required"] = False
    final_inventory["terminal_refresh_cutoff_utc"] = "2026-08-26T20:10:00Z"
    final_inventory["terminal_refresh_rule"] = (
        "V7 preserves all V6 exact-green authorities and capacity while "
        "representing each accepted DATA-228 CPython chunk as an independent "
        "dedup object; no synthetic chunk concatenation is permitted."
    )

    payloads = dict(inherited_payloads)
    payloads[mdn["source_id"]] = mdn_payload
    payloads.update(cpython_payloads)
    payloads.update(numpy_payloads)
    payloads.update(pg_payloads)

    dedup = v3.audit_payloads(final_inventory, payloads)
    v3.verify_report(dedup)
    _require(
        dedup["source_count"] == EXPECTED_SOURCE_OBJECT_COUNT,
        "V7 source-count drift",
    )
    counts = _family_counts(dedup)
    _require(
        counts == {"uk": 4, "en": 5, "code": 5},
        f"V7 family-vector drift: {counts}",
    )

    fixed = v6_config["expected_vector"]["fixed_capacity_without_cpython_accepted_chunks"]
    expected_total = int(fixed["total"]) + cpython_capacity
    expected_en = int(fixed["en"]) + cpython_capacity
    scope = dedup["terminal_candidates"]
    _require(
        scope["declared_capacity_bytes_before"] == expected_total,
        "V7 total capacity arithmetic drift",
    )
    _require(
        scope["by_modality"]["en"]["declared_capacity_bytes_before"] == expected_en,
        "V7 EN capacity arithmetic drift",
    )

    core = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "parent_v6_worker_id": v6.WORKER_ID,
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "source_vector": {
            "source_object_count": dedup["source_count"],
            "source_family_counts": counts,
            "cpython_accepted_record_count": len(cpython_rows),
            "cpython_accepted_capacity_bytes": cpython_capacity,
            "source_capacity_bytes_before_global_dedup": expected_total,
            "source_capacity_by_modality_before_global_dedup": {
                "uk": int(fixed["uk"]),
                "en": expected_en,
                "code": int(fixed["code"]),
            },
            "conservative_unique_capacity_bytes_after_global_dedup": scope[
                "conservative_unique_capacity_bytes_after"
            ],
            "research_corpus_v1_acquisition_target_bytes": int(
                v6_config["expected_vector"]["research_corpus_v1_acquisition_target_bytes"]
            ),
            "pre_dedup_planning_gap_bytes": max(
                0,
                int(v6_config["expected_vector"]["research_corpus_v1_acquisition_target_bytes"])
                - expected_total,
            ),
        },
        "cpython_source_evidence": cpython_source_evidence,
        "cpython_record_evidence": cpython_record_evidence,
        "materialization_evidence": sorted(
            [*inherited_evidence, mdn_evidence, *numpy_evidence, *pg_evidence],
            key=lambda item: item["source_id"],
        ),
        "dedup_v3": dedup,
        "claim_boundary": copy.deepcopy(v6_config["claim_boundary"]),
        "remaining_blockers": [
            "SOURCE_CAPACITY_FAR_BELOW_20M_RESEARCH_TARGET",
            "FINAL_RECORD_GRANULARITY_QUALITY_PRIVACY_REVALIDATION",
            "EVALUATION_SELECTION_RESERVATIONS_AND_DECONTAMINATION",
            "IMMUTABLE_SPLIT_AND_PACKING",
            "POSTPACK_UNIQUE_LOSS_LEDGER",
            "TOKENIZER_FIT_AUTHORIZATION",
            "CHECKPOINT_D05_TERMINAL_INTEGRITY",
            "MATERIAL_COMPUTE_AUTHORIZATION",
        ],
        "raw_text_emitted": False,
    }
    return {**core, "report_sha256": v5._sha256(v5._canonical_bytes(core))}


def verify_report(report: Mapping[str, Any]) -> None:
    """Verify the V7 report and its record-granularity invariants."""
    _require(report.get("schema_version") == SCHEMA, "unsupported V7 report schema")
    expected_hash = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    _require(
        expected_hash == v5._sha256(v5._canonical_bytes(core)),
        "V7 report self-hash mismatch",
    )
    for key in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
        "raw_text_emitted",
    ):
        _require(report.get(key) is False, f"V7 execution boundary failed: {key}")

    vector = report["source_vector"]
    _require(
        vector.get("source_object_count") == EXPECTED_SOURCE_OBJECT_COUNT,
        "V7 report source count drift",
    )
    _require(
        vector.get("cpython_accepted_record_count") == CPYTHON_ACCEPTED_RECORD_COUNT,
        "V7 CPython record count drift",
    )
    _require(
        vector.get("cpython_accepted_capacity_bytes") == 15540,
        "V7 CPython capacity drift",
    )
    _require(
        vector.get("source_capacity_bytes_before_global_dedup") == 2045180,
        "V7 total capacity drift",
    )

    records = report["cpython_record_evidence"]
    _require(
        len(records) == CPYTHON_ACCEPTED_RECORD_COUNT,
        "V7 CPython evidence cardinality drift",
    )
    _require(
        len({row["source_id"] for row in records}) == len(records),
        "V7 CPython source ids are not unique",
    )
    _require(
        len({row["normalized_sha256"] for row in records}) == len(records),
        "V7 CPython accepted hashes are not unique",
    )
    _require(
        sum(int(row["normalized_bytes"]) for row in records) == 15540,
        "V7 CPython record byte accounting drift",
    )
    _require(
        report["cpython_source_evidence"].get("comparison_representation")
        == "RECORD_GRANULAR_NO_SYNTHETIC_JOIN",
        "V7 synthetic-join boundary weakened",
    )
    _require(
        all(row.get("raw_text_emitted") is False for row in records),
        "V7 CPython text evidence leaked",
    )

    for key in (
        "canonical_registry_replaced",
        "corpus_materialized",
        "decontamination_pass_claimed",
        "balance_release_claimed",
        "postpack_unique_loss_ledger_complete",
        "tokenizer_fit_authorized",
        "training_authorized",
        "paid_compute_authorized",
        "research_corpus_v1_terminal",
    ):
        _require(
            report["claim_boundary"].get(key) is False,
            f"V7 truth boundary weakened: {key}",
        )

    v3.verify_report(report["dedup_v3"])
