"""Deterministic candidate materialization for the qualified Lesia 1892 edition."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from twelve_six.data.wikisource_pd_api import (
    PageSnapshot,
    discover_index_titles,
    fetch_page_snapshot,
    request_json,
)
from twelve_six.data.wikisource_pd_contract import (
    INCUMBENT_AUTHORITY_SHA256,
    INDEX_REVISION_ID,
    SOURCE_FAMILY_ID,
    WikisourceIntakeError,
    normalize_rendered_text,
    validate_page_title,
    validate_ua_page_text,
)


@dataclass(frozen=True)
class Materialization:
    candidate_jsonl: bytes
    report: dict[str, Any]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    rendered = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"
    return rendered.encode("utf-8")


def _page_record(snapshot: PageSnapshot) -> dict[str, Any]:
    return {
        "source_id": f"ua.wikisource.lesia-1892.page{snapshot.page_number}",
        "source_family_id": SOURCE_FAMILY_ID,
        "language": "uk",
        "modality": "text",
        "page_number": snapshot.page_number,
        "page_title": snapshot.title,
        "page_revision_id": snapshot.revision_id,
        "normalized_sha256": snapshot.sha256,
        "normalized_utf8_bytes": snapshot.utf8_bytes,
        "training_eligible": False,
        "evaluation_eligible": False,
        "text": snapshot.normalized_text,
    }


def materialize_snapshots(snapshots: Iterable[PageSnapshot]) -> Materialization:
    ordered = sorted(snapshots, key=lambda row: row.page_number)
    if not ordered:
        raise WikisourceIntakeError("materialization requires at least one approved page")
    if len(ordered) > 112:
        raise WikisourceIntakeError("materialization exceeds edition page bound")
    page_numbers = [row.page_number for row in ordered]
    revisions = [row.revision_id for row in ordered]
    hashes = [row.sha256 for row in ordered]
    if len(set(page_numbers)) != len(page_numbers):
        raise WikisourceIntakeError("duplicate page number")
    if len(set(revisions)) != len(revisions):
        raise WikisourceIntakeError("duplicate revision id")
    if len(set(hashes)) != len(hashes):
        raise WikisourceIntakeError("exact duplicate page body requires explicit review")
    for row in ordered:
        if validate_page_title(row.title) != row.page_number:
            raise WikisourceIntakeError("page title/number mismatch")
        normalized = normalize_rendered_text(row.normalized_text)
        if normalized != row.normalized_text:
            raise WikisourceIntakeError("snapshot text is not canonical")
        validate_ua_page_text(normalized)
        payload = normalized.encode("utf-8")
        if _sha256(payload) != row.sha256 or len(payload) != row.utf8_bytes:
            raise WikisourceIntakeError("snapshot byte identity mismatch")
    records = [_page_record(row) for row in ordered]
    candidate_jsonl = b"".join(_canonical_json(record) for record in records)
    inventory = [
        {
            "page_number": row.page_number,
            "page_revision_id": row.revision_id,
            "normalized_sha256": row.sha256,
            "normalized_utf8_bytes": row.utf8_bytes,
        }
        for row in ordered
    ]
    report: dict[str, Any] = {
        "schema_version": "12-6.d03-wikisource-pd-edition-materialization.v1",
        "source_authority": {
            "incumbent_next100022_authority_sha256": INCUMBENT_AUTHORITY_SHA256,
            "index_revision_id": INDEX_REVISION_ID,
            "source_family_id": SOURCE_FAMILY_ID,
            "family_credit_added": False,
        },
        "candidate": {
            "page_count": len(ordered),
            "normalized_utf8_bytes": sum(row.utf8_bytes for row in ordered),
            "candidate_jsonl_sha256": _sha256(candidate_jsonl),
            "inventory": inventory,
        },
        "truth_boundary": {
            "canonical_capacity_credit_bytes": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "required_downstream_gates": [
                "GLOBAL_CROSS_SOURCE_DEDUP",
                "FRESH_RESERVED_EVALUATION_DECONTAMINATION",
                "POST_COMPOSITION_QUALITY_PRIVACY",
                "BALANCE_AND_FAMILY_CAPS",
                "CLUSTER_SAFE_SPLIT",
                "DETERMINISTIC_PACK_AND_TWO_CLEAN_BUILDS",
                "POSITIVE_EXACT_UNIQUE_LOSS_LEDGER",
            ],
        },
    }
    report["report_sha256"] = _sha256(_canonical_json(report))
    return Materialization(candidate_jsonl=candidate_jsonl, report=report)


def materialize_live(
    *,
    max_pages: int = 112,
    cadence_seconds: float = 0.55,
    get_json: Callable[[dict[str, str]], dict[str, Any]] = request_json,
) -> Materialization:
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or not 1 <= max_pages <= 112:
        raise WikisourceIntakeError("max_pages must be an integer in [1, 112]")
    if cadence_seconds < 0.5:
        raise WikisourceIntakeError("network request cadence must be at least 0.5 seconds")
    titles = discover_index_titles(get_json=get_json)[:max_pages]
    snapshots: list[PageSnapshot] = []
    for index, title in enumerate(titles):
        if index:
            time.sleep(cadence_seconds)
        snapshots.append(fetch_page_snapshot(title, get_json=get_json))
    return materialize_snapshots(snapshots)
