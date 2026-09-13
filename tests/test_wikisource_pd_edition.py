from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from pathlib import Path

import pytest

from twelve_six.data.wikisource_pd_api import (
    PageSnapshot,
    discover_index_titles,
    fetch_page_snapshot,
)
from twelve_six.data.wikisource_pd_contract import (
    APPROVED_CATEGORY,
    INDEX_REVISION_ID,
    INDEX_TITLE,
    PAGE_PREFIX,
    SOURCE_FAMILY_ID,
    WikisourceIntakeError,
    normalize_rendered_text,
    validate_control_contract,
)
from twelve_six.data.wikisource_pd_edition import materialize_snapshots


def _snapshot(
    page: int,
    text: str = "Український літературний текст для перевіреної сторінки.\n",
) -> PageSnapshot:
    normalized = normalize_rendered_text(text)
    payload = normalized.encode("utf-8")
    return PageSnapshot(
        page_number=page,
        title=f"{PAGE_PREFIX}{page}",
        revision_id=500000 + page,
        normalized_text=normalized,
        sha256=hashlib.sha256(payload).hexdigest(),
        utf8_bytes=len(payload),
    )


def test_normalization_is_nfc_lf_and_stanza_stable() -> None:
    raw = "  Украі\u0308нський рядок\r\n\u00a0Другий український рядок\r\n\r\n\r\nТретій рядок  "
    normalized = normalize_rendered_text(raw)
    assert normalized == unicodedata.normalize("NFC", normalized)
    assert "\r" not in normalized
    assert "\u00a0" not in normalized
    assert normalized.endswith("\n")
    assert "\n\n\n" not in normalized


def test_index_discovery_is_numeric_sorted_and_pinned() -> None:
    calls = []

    def fake(params):
        calls.append(params)
        return {
            "parse": {
                "links": [
                    {"title": f"{PAGE_PREFIX}13"},
                    {"title": "Головна сторінка"},
                    {"title": f"{PAGE_PREFIX}3"},
                ]
            }
        }

    assert discover_index_titles(get_json=fake) == [f"{PAGE_PREFIX}3", f"{PAGE_PREFIX}13"]
    assert calls == [{"action": "parse", "oldid": str(INDEX_REVISION_ID), "prop": "links"}]


def _index_revision_metadata() -> dict:
    return {
        "query": {
            "pages": [
                {
                    "title": INDEX_TITLE,
                    "revisions": [{"revid": INDEX_REVISION_ID}],
                }
            ]
        }
    }


def test_index_fallback_selects_only_native_validated_pages() -> None:
    responses = iter(
        [
            {"parse": {"revid": INDEX_REVISION_ID, "links": []}},
            _index_revision_metadata(),
            {
                "query": {
                    "pages": [
                        {
                            "title": f"{PAGE_PREFIX}3",
                            "proofread": {"quality": 3, "quality_text": "Proofread"},
                        },
                        {
                            "title": f"{PAGE_PREFIX}13",
                            "proofread": {"quality": 4, "quality_text": APPROVED_CATEGORY},
                        },
                        {
                            "title": f"{PAGE_PREFIX}14",
                            "missing": True,
                        },
                    ]
                }
            },
            _index_revision_metadata(),
        ]
    )
    assert discover_index_titles(get_json=lambda _: next(responses)) == [f"{PAGE_PREFIX}13"]


def test_index_fallback_fails_if_no_validated_page_exists() -> None:
    responses = iter(
        [
            {"parse": {"revid": INDEX_REVISION_ID, "links": []}},
            _index_revision_metadata(),
            {
                "query": {
                    "pages": [
                        {
                            "title": f"{PAGE_PREFIX}3",
                            "proofread": {"quality": 3, "quality_text": "Proofread"},
                        }
                    ]
                }
            },
            _index_revision_metadata(),
        ]
    )
    with pytest.raises(WikisourceIntakeError, match="no validated numeric page"):
        discover_index_titles(get_json=lambda _: next(responses))


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


def test_fetch_page_requires_validated_quality_and_seals_revision() -> None:
    title = f"{PAGE_PREFIX}13"
    responses = iter(
        [
            _approved_metadata(title, 560107),
            {
                "parse": {
                    "revid": 560107,
                    "text": (
                        "<div><p>Український літературний текст достатньої "
                        "довжини для сторінки.</p></div>"
                    ),
                }
            },
            _approved_metadata(title, 560107),
        ]
    )
    snapshot = fetch_page_snapshot(title, get_json=lambda _: next(responses))
    assert snapshot.revision_id == 560107
    assert snapshot.page_number == 13
    assert snapshot.utf8_bytes > 64


def test_fetch_page_rejects_unapproved_page_before_render() -> None:
    title = f"{PAGE_PREFIX}13"
    calls = []

    def fake(params):
        calls.append(params)
        return {
            "query": {
                "pages": [
                    {
                        "title": title,
                        "revisions": [{"revid": 1}],
                        "proofread": {"quality": 3, "quality_text": "Proofread"},
                    }
                ]
            }
        }

    with pytest.raises(WikisourceIntakeError, match="approved"):
        fetch_page_snapshot(title, get_json=fake)
    assert len(calls) == 1


def test_fetch_page_rejects_revision_drift_after_exact_render() -> None:
    title = f"{PAGE_PREFIX}13"
    responses = iter(
        [
            _approved_metadata(title, 560107),
            {
                "parse": {
                    "revid": 560107,
                    "text": (
                        "<div><p>Український літературний текст достатньої "
                        "довжини для сторінки.</p></div>"
                    ),
                }
            },
            _approved_metadata(title, 560108),
        ]
    )
    with pytest.raises(WikisourceIntakeError, match="changed during exact render"):
        fetch_page_snapshot(title, get_json=lambda _: next(responses))


def test_fetch_page_rejects_approval_loss_after_exact_render() -> None:
    title = f"{PAGE_PREFIX}13"
    after = _approved_metadata(title, 560107)
    after["query"]["pages"][0]["proofread"] = {
        "quality": 3,
        "quality_text": "Proofread",
    }
    responses = iter(
        [
            _approved_metadata(title, 560107),
            {
                "parse": {
                    "revid": 560107,
                    "text": (
                        "<div><p>Український літературний текст достатньої "
                        "довжини для сторінки.</p></div>"
                    ),
                }
            },
            after,
        ]
    )
    with pytest.raises(WikisourceIntakeError, match="approval changed"):
        fetch_page_snapshot(title, get_json=lambda _: next(responses))


def test_materialization_is_deterministic_and_zero_credit() -> None:
    first = _snapshot(13)
    second = _snapshot(
        14,
        "Інший український літературний текст для наступної перевіреної сторінки.\n",
    )
    one = materialize_snapshots([second, first])
    two = materialize_snapshots([first, second])
    assert one == two
    report = one.report
    assert report["source_authority"]["source_family_id"] == SOURCE_FAMILY_ID
    assert report["source_authority"]["family_credit_added"] is False
    boundary = report["truth_boundary"]
    assert boundary["canonical_capacity_credit_bytes"] == 0
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["authorized_unique_loss_positions"] == 0
    assert boundary["model_training_executed"] is False
    assert boundary["paid_compute_used"] is False


def test_candidate_records_never_mark_training_or_evaluation_eligible() -> None:
    materialized = materialize_snapshots([_snapshot(13)])
    record = json.loads(materialized.candidate_jsonl)
    assert record["training_eligible"] is False
    assert record["evaluation_eligible"] is False


def test_report_is_text_free() -> None:
    text = "Унікальний український секретний маркер для тестової сторінки достатньої довжини.\n"
    materialized = materialize_snapshots([_snapshot(13, text)])
    serialized = json.dumps(materialized.report, ensure_ascii=False)
    assert "секретний маркер" not in serialized
    assert "text" not in materialized.report["candidate"]["inventory"][0]


def test_exact_duplicate_page_body_fails_closed() -> None:
    first = _snapshot(13)
    duplicate = PageSnapshot(
        page_number=14,
        title=f"{PAGE_PREFIX}14",
        revision_id=500014,
        normalized_text=first.normalized_text,
        sha256=first.sha256,
        utf8_bytes=first.utf8_bytes,
    )
    with pytest.raises(WikisourceIntakeError, match="duplicate page body"):
        materialize_snapshots([first, duplicate])


def test_snapshot_identity_mutation_fails_closed() -> None:
    source = _snapshot(13)
    bad = PageSnapshot(**{**source.__dict__, "sha256": "0" * 64})
    with pytest.raises(WikisourceIntakeError, match="byte identity"):
        materialize_snapshots([bad])


def test_title_outside_qualified_edition_fails_closed() -> None:
    source = _snapshot(13)
    bad = PageSnapshot(**{**source.__dict__, "title": "Сторінка:Інша книга/13"})
    with pytest.raises(WikisourceIntakeError, match="outside the qualified edition"):
        materialize_snapshots([bad])


def test_report_identity_detects_tamper() -> None:
    report = copy.deepcopy(materialize_snapshots([_snapshot(13)]).report)
    identity = report.pop("report_sha256")
    canonical = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == identity
    report["truth_boundary"]["training_authorized_bytes"] = 1
    changed = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    assert hashlib.sha256(changed).hexdigest() != identity


def test_non_ua_or_privacy_like_content_fails_closed() -> None:
    with pytest.raises(WikisourceIntakeError):
        materialize_snapshots([_snapshot(13, "English only content " * 20)])
    with pytest.raises(WikisourceIntakeError):
        materialize_snapshots([_snapshot(13, "Український текст user@example.com " * 8)])


def test_control_contract_is_fail_closed() -> None:
    contract = {
        "schema_version": "12-6.d03-wikisource-lesia1892-current-main.v1",
        "execution_class": "LOCAL_FREE",
        "incumbent_authority": {
            "head_sha": "84c51e42b6daa51796fd20d793b5ef1ff01cc9d2",
            "authority_identity_sha256": (
                "6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6"
            ),
        },
        "edition": {
            "index_revision_id": INDEX_REVISION_ID,
            "source_family_id": SOURCE_FAMILY_ID,
            "family_credit_added": False,
        },
        "acquisition": {"max_pages": 112, "minimum_request_cadence_seconds": 0.5},
        "truth_boundary": {
            "canonical_capacity_credit_bytes": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "optimizer_updates": 0,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
        },
    }
    validate_control_contract(contract)
    contract["truth_boundary"]["training_authorized_bytes"] = 1
    with pytest.raises(WikisourceIntakeError, match="zero-credit"):
        validate_control_contract(contract)


def test_repository_control_file_matches_runtime_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "configs/data/d03_wikisource_lesia1892_current_main_v1.json").read_text(
            encoding="utf-8"
        )
    )
    validate_control_contract(contract)
