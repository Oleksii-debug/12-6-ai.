from __future__ import annotations

import hashlib
import json

import pytest

from twelve_six.data import wikisource_pd_edition as edition_module
from twelve_six.data.wikisource_pd_api import (
    SHORT_PAGE_REJECTION_REASON,
    PageSnapshot,
    RejectedPageCandidate,
    fetch_page_snapshot,
)
from twelve_six.data.wikisource_pd_contract import (
    APPROVED_CATEGORY,
    PAGE_PREFIX,
    WikisourceIntakeError,
    normalize_rendered_text,
    validate_ua_page_text,
)
from twelve_six.data.wikisource_pd_edition import (
    PageRejection,
    materialize_live,
    materialize_snapshots,
)


def _approved_metadata(title: str, revision_id: int) -> dict:
    return {
        "query": {
            "pages": [
                {
                    "title": title,
                    "revisions": [{"revid": revision_id}],
                    "proofread": {"quality": 4, "quality_text": APPROVED_CATEGORY},
                }
            ]
        }
    }


def _snapshot(page: int) -> PageSnapshot:
    normalized = normalize_rendered_text(
        "Український літературний текст достатньої довжини для accepted сторінки."
    )
    payload = normalized.encode("utf-8")
    return PageSnapshot(
        page_number=page,
        title=f"{PAGE_PREFIX}{page}",
        revision_id=600000 + page,
        normalized_text=normalized,
        sha256=hashlib.sha256(payload).hexdigest(),
        utf8_bytes=len(payload),
    )


def _rejection(page: int = 13, revision_id: int = 560107) -> PageRejection:
    normalized = normalize_rendered_text("Коротко")
    payload = normalized.encode("utf-8")
    return PageRejection(
        page_number=page,
        revision_id=revision_id,
        normalized_sha256=hashlib.sha256(payload).hexdigest(),
        normalized_utf8_bytes=len(payload),
    )


def test_short_page_gate_remains_exactly_64_bytes() -> None:
    with pytest.raises(WikisourceIntakeError, match="page body is too short"):
        validate_ua_page_text("а" * 31)
    validate_ua_page_text("а" * 32)


def test_exact_approved_short_page_becomes_text_free_rejection() -> None:
    title = f"{PAGE_PREFIX}13"
    secret_body = "Коротко"
    normalized = normalize_rendered_text(secret_body)
    payload = normalized.encode("utf-8")
    responses = iter(
        [
            _approved_metadata(title, 560107),
            {"parse": {"revid": 560107, "text": f"<div><p>{secret_body}</p></div>"}},
            _approved_metadata(title, 560107),
        ]
    )
    with pytest.raises(RejectedPageCandidate) as exc_info:
        fetch_page_snapshot(title, get_json=lambda _: next(responses))
    rejection = exc_info.value
    assert rejection.reason == SHORT_PAGE_REJECTION_REASON
    assert rejection.page_number == 13
    assert rejection.revision_id == 560107
    assert rejection.normalized_utf8_bytes == len(payload)
    assert rejection.normalized_sha256 == hashlib.sha256(payload).hexdigest()
    message = str(rejection)
    assert secret_body not in message
    assert title not in message


def test_short_non_ua_page_remains_fatal_not_rejected() -> None:
    title = f"{PAGE_PREFIX}13"
    body = "English words"
    responses = iter(
        [
            _approved_metadata(title, 560107),
            {"parse": {"revid": 560107, "text": f"<div><p>{body}</p></div>"}},
            _approved_metadata(title, 560107),
        ]
    )
    with pytest.raises(WikisourceIntakeError, match="not predominantly Ukrainian") as exc_info:
        fetch_page_snapshot(title, get_json=lambda _: next(responses))
    assert not isinstance(exc_info.value, RejectedPageCandidate)
    assert body not in str(exc_info.value)


def test_rejection_is_excluded_but_bound_into_deterministic_report() -> None:
    accepted = _snapshot(14)
    rejected = _rejection()
    one = materialize_snapshots([accepted], rejected_pages=[rejected])
    two = materialize_snapshots([accepted], rejected_pages=[rejected])
    assert one == two
    records = [json.loads(line) for line in one.candidate_jsonl.splitlines()]
    assert [row["page_number"] for row in records] == [14]
    disposition = one.report["disposition"]
    assert disposition["accepted_page_count"] == 1
    assert disposition["rejected_page_count"] == 1
    assert disposition["observed_page_count"] == 2
    rejected_inventory = disposition["rejected_inventory"]
    assert rejected_inventory == [
        {
            "page_number": 13,
            "page_revision_id": 560107,
            "normalized_sha256": rejected.normalized_sha256,
            "normalized_utf8_bytes": rejected.normalized_utf8_bytes,
            "reason": SHORT_PAGE_REJECTION_REASON,
        }
    ]
    combined = {
        "accepted": one.report["candidate"]["inventory"],
        "rejected": rejected_inventory,
    }
    canonical = (
        json.dumps(combined, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    assert disposition["accepted_and_rejected_inventory_sha256"] == hashlib.sha256(
        canonical
    ).hexdigest()
    assert "Коротко" not in json.dumps(one.report, ensure_ascii=False)
    assert one.report["truth_boundary"]["canonical_capacity_credit_bytes"] == 0


def test_live_materializer_skips_only_typed_short_rejection(monkeypatch) -> None:
    accepted = _snapshot(14)
    rejected = _rejection()
    titles = [f"{PAGE_PREFIX}13", f"{PAGE_PREFIX}14"]
    monkeypatch.setattr(edition_module, "discover_index_titles", lambda **_: titles)

    def fake_fetch(title: str, **_) -> PageSnapshot:
        if title.endswith("/13"):
            raise RejectedPageCandidate(
                page_number=rejected.page_number,
                revision_id=rejected.revision_id,
                normalized_sha256=rejected.normalized_sha256,
                normalized_utf8_bytes=rejected.normalized_utf8_bytes,
            )
        return accepted

    monkeypatch.setattr(edition_module, "fetch_page_snapshot", fake_fetch)
    result = materialize_live(
        max_pages=2,
        cadence_seconds=0.5,
        get_json=lambda _: {},
        sleep_fn=lambda _: None,
        monotonic_fn=lambda: 0.0,
    )
    assert result.report["candidate"]["page_count"] == 1
    assert result.report["disposition"]["rejected_page_count"] == 1


def test_live_materializer_keeps_other_failures_fatal(monkeypatch) -> None:
    titles = [f"{PAGE_PREFIX}13"]
    monkeypatch.setattr(edition_module, "discover_index_titles", lambda **_: titles)

    def fake_fetch(title: str, **_) -> PageSnapshot:
        raise WikisourceIntakeError("page body contains site chrome or raw HTML")

    monkeypatch.setattr(edition_module, "fetch_page_snapshot", fake_fetch)
    with pytest.raises(WikisourceIntakeError, match="site chrome"):
        materialize_live(
            max_pages=1,
            cadence_seconds=0.5,
            get_json=lambda _: {},
            sleep_fn=lambda _: None,
            monotonic_fn=lambda: 0.0,
        )


def test_invalid_rejection_cannot_bypass_minimum_gate() -> None:
    rejected = _rejection()
    invalid = PageRejection(
        page_number=rejected.page_number,
        revision_id=rejected.revision_id,
        normalized_sha256=rejected.normalized_sha256,
        normalized_utf8_bytes=64,
    )
    with pytest.raises(WikisourceIntakeError, match="byte count is outside gate"):
        materialize_snapshots([_snapshot(14)], rejected_pages=[invalid])
