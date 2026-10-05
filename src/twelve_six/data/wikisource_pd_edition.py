"""Deterministic candidate materialization for the qualified Lesia 1892 edition."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from twelve_six.data.wikisource_pd_api import (
    SHORT_PAGE_REJECTION_REASON,
    PageSnapshot,
    RejectedPageCandidate,
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


@dataclass(frozen=True)
class PageRejection:
    page_number: int
    revision_id: int
    normalized_sha256: str
    normalized_utf8_bytes: int
    reason: str = SHORT_PAGE_REJECTION_REASON


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


def _rejection_record(rejection: PageRejection) -> dict[str, Any]:
    return {
        "page_number": rejection.page_number,
        "page_revision_id": rejection.revision_id,
        "normalized_sha256": rejection.normalized_sha256,
        "normalized_utf8_bytes": rejection.normalized_utf8_bytes,
        "reason": rejection.reason,
    }


def _validate_rejections(
    rejections: list[PageRejection],
    *,
    accepted_page_numbers: set[int],
    accepted_revision_ids: set[int],
) -> None:
    rejected_page_numbers = [row.page_number for row in rejections]
    rejected_revision_ids = [row.revision_id for row in rejections]
    if len(set(rejected_page_numbers)) != len(rejected_page_numbers):
        raise WikisourceIntakeError("duplicate rejected page number")
    if len(set(rejected_revision_ids)) != len(rejected_revision_ids):
        raise WikisourceIntakeError("duplicate rejected revision id")
    if accepted_page_numbers.intersection(rejected_page_numbers):
        raise WikisourceIntakeError("page cannot be both accepted and rejected")
    if accepted_revision_ids.intersection(rejected_revision_ids):
        raise WikisourceIntakeError("revision cannot be both accepted and rejected")
    for row in rejections:
        if not 1 <= row.page_number <= 112:
            raise WikisourceIntakeError("rejected page is outside the pinned edition bounds")
        if isinstance(row.revision_id, bool) or not isinstance(row.revision_id, int):
            raise WikisourceIntakeError("rejected page revision id is invalid")
        if row.revision_id <= 0:
            raise WikisourceIntakeError("rejected page revision id is invalid")
        if row.reason != SHORT_PAGE_REJECTION_REASON:
            raise WikisourceIntakeError("unsupported page rejection reason")
        if (
            isinstance(row.normalized_utf8_bytes, bool)
            or not isinstance(row.normalized_utf8_bytes, int)
            or not 0 < row.normalized_utf8_bytes < 64
        ):
            raise WikisourceIntakeError("short-page rejection byte count is outside gate")
        if re.fullmatch(r"[0-9a-f]{64}", row.normalized_sha256) is None:
            raise WikisourceIntakeError("rejected page hash is invalid")


def materialize_snapshots(
    snapshots: Iterable[PageSnapshot],
    *,
    rejected_pages: Iterable[PageRejection] = (),
) -> Materialization:
    ordered = sorted(snapshots, key=lambda row: row.page_number)
    rejected_ordered = sorted(rejected_pages, key=lambda row: row.page_number)
    if not ordered:
        raise WikisourceIntakeError("materialization requires at least one approved page")
    if len(ordered) + len(rejected_ordered) > 112:
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
    _validate_rejections(
        rejected_ordered,
        accepted_page_numbers=set(page_numbers),
        accepted_revision_ids=set(revisions),
    )
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
    rejected_inventory = [_rejection_record(row) for row in rejected_ordered]
    disposition_inventory = {
        "accepted": inventory,
        "rejected": rejected_inventory,
    }
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
        "disposition": {
            "accepted_page_count": len(ordered),
            "rejected_page_count": len(rejected_ordered),
            "observed_page_count": len(ordered) + len(rejected_ordered),
            "accepted_and_rejected_inventory_sha256": _sha256(
                _canonical_json(disposition_inventory)
            ),
            "rejected_inventory": rejected_inventory,
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


def _retry_after_seconds(error: urllib.error.HTTPError) -> float:
    value = error.headers.get("Retry-After") if error.headers is not None else None
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except ValueError:
        return 0.0
    if seconds < 0:
        return 0.0
    if seconds > 60:
        raise WikisourceIntakeError(
            "Wikisource Retry-After exceeds bounded LOCAL_FREE execution window"
        ) from error
    return seconds


def materialize_live(
    *,
    max_pages: int = 112,
    cadence_seconds: float = 1.1,
    max_429_attempts: int = 5,
    get_json: Callable[[dict[str, str]], dict[str, Any]] = request_json,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> Materialization:
    if isinstance(max_pages, bool) or not isinstance(max_pages, int) or not 1 <= max_pages <= 112:
        raise WikisourceIntakeError("max_pages must be an integer in [1, 112]")
    if (
        isinstance(cadence_seconds, bool)
        or not isinstance(cadence_seconds, (int, float))
        or cadence_seconds < 0.5
    ):
        raise WikisourceIntakeError("network request cadence must be at least 0.5 seconds")
    if (
        isinstance(max_429_attempts, bool)
        or not isinstance(max_429_attempts, int)
        or not 1 <= max_429_attempts <= 5
    ):
        raise WikisourceIntakeError("max_429_attempts must be an integer in [1, 5]")

    last_request_completed_at: float | None = None

    def paced_get_json(params: dict[str, str]) -> dict[str, Any]:
        nonlocal last_request_completed_at
        for attempt in range(max_429_attempts):
            if last_request_completed_at is not None:
                elapsed = monotonic_fn() - last_request_completed_at
                remaining = cadence_seconds - elapsed
                if remaining > 0:
                    sleep_fn(remaining)
            try:
                result = get_json(params)
            except urllib.error.HTTPError as exc:
                last_request_completed_at = monotonic_fn()
                if exc.code != 429 or attempt + 1 >= max_429_attempts:
                    raise
                retry_after = _retry_after_seconds(exc)
                exponential_backoff = min(30.0, float(2 ** (attempt + 1)))
                sleep_fn(max(cadence_seconds, retry_after, exponential_backoff))
                continue
            last_request_completed_at = monotonic_fn()
            return result
        raise WikisourceIntakeError("unreachable bounded 429 retry state")

    titles = discover_index_titles(get_json=paced_get_json)[:max_pages]
    snapshots: list[PageSnapshot] = []
    rejections: list[PageRejection] = []
    for title in titles:
        try:
            snapshots.append(fetch_page_snapshot(title, get_json=paced_get_json))
        except RejectedPageCandidate as exc:
            rejections.append(
                PageRejection(
                    page_number=exc.page_number,
                    revision_id=exc.revision_id,
                    normalized_sha256=exc.normalized_sha256,
                    normalized_utf8_bytes=exc.normalized_utf8_bytes,
                    reason=exc.reason,
                )
            )
    return materialize_snapshots(snapshots, rejected_pages=rejections)
