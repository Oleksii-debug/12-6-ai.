from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data.source_intake import (
    DownloadedBytes,
    SourceIntakeError,
    extract_text,
    run_bounded_intake,
    validate_candidate_registry,
)

REGISTRY = Path("configs/data/external_source_candidates_ua_en_v1.json")


def _load() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _rehash(registry: dict) -> dict:
    core = dict(registry)
    core.pop("registry_identity_sha256", None)
    payload = (
        json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    registry["registry_identity_sha256"] = hashlib.sha256(payload).hexdigest()
    return registry


def test_candidate_registry_exact_counts_and_fail_closed_fetch_surface() -> None:
    registry = _load()
    sources = validate_candidate_registry(registry)
    assert len(sources) == 8
    eligible = [item for item in sources if item.eligibility_status == "ELIGIBLE"]
    blocked = [item for item in sources if item.eligibility_status == "BLOCKED_BY_RIGHTS"]
    assert {item.source_id for item in eligible} == {
        "ua.rada.open-data.laws-texts",
        "en.standardebooks.manual",
    }
    assert len(blocked) == 6
    assert sum(item.rights.status == "REJECTED" for item in sources) == 1
    assert sum(item.rights.status == "REVIEW_REQUIRED" for item in sources) == 5
    assert all(item.adapter is None and not item.acquisition_urls for item in blocked)


def test_candidate_registry_tamper_rejected() -> None:
    registry = _load()
    registry["sources"][0]["rights"]["allows_model_training"] = False
    with pytest.raises(SourceIntakeError, match="ELIGIBLE requires"):
        validate_candidate_registry(registry)


def test_cp1251_html_extraction() -> None:
    text = (
        "<html><head><meta charset='windows-1251'></head>"
        "<body><h1>Український закон</h1><p>Цей текст містить українські дані "
        "і мову для перевірки коректного декодування.</p></body></html>"
    )
    extracted, encoding = extract_text(
        DownloadedBytes(text.encode("cp1251"), "text/html"), "html_text"
    )
    assert encoding == "cp1251"
    assert "Український закон" in extracted
    assert "коректного декодування" in extracted


def test_bounded_intake_reuses_lid_and_exact_dedup_and_never_fetches_blocked(
    tmp_path: Path,
) -> None:
    registry = copy.deepcopy(_load())
    eligible = [
        source for source in registry["sources"] if source["eligibility_status"] == "ELIGIBLE"
    ]
    ua = next(source for source in eligible if source["language"] == "uk")
    en = next(source for source in eligible if source["language"] == "en")
    ua["acquisition_urls"] = ["https://example.invalid/ua"]
    en["acquisition_urls"] = [
        "https://example.invalid/en-a",
        "https://example.invalid/en-b",
    ]
    _rehash(registry)

    ua_html = (
        "<html><head><meta charset='utf-8'></head><body><article>"
        "Українська мова і дані для моделі. Цей закон описує права та обов'язки "
        "людей, а також те, що держава і суспільство мають робити для громадян. "
        "Ці дані потрібні для перевірки українського тексту."
        "</article></body></html>"
    ).encode("utf-8")
    en_text = (
        "This manual explains the language and typography for a book. "
        "The data and the model are not the topic; this text is a stable English "
        "fixture for testing extraction and exact duplicate staging."
    ).encode("utf-8")
    payloads = {
        "https://example.invalid/ua": DownloadedBytes(ua_html, "text/html; charset=utf-8"),
        "https://example.invalid/en-a": DownloadedBytes(en_text, "text/plain; charset=utf-8"),
        "https://example.invalid/en-b": DownloadedBytes(en_text, "text/plain; charset=utf-8"),
    }
    fetched: list[str] = []

    def fetcher(url: str, max_bytes: int) -> DownloadedBytes:
        fetched.append(url)
        assert max_bytes == 100_000
        return payloads[url]

    manifest = run_bounded_intake(
        registry,
        tmp_path,
        fetcher=fetcher,
        max_download_bytes=100_000,
        max_normalized_chars=20_000,
    )
    assert sorted(fetched) == sorted(payloads)
    assert manifest["source_counts"] == {
        "candidate": 8,
        "eligible": 2,
        "blocked_by_rights": 6,
        "rights_rejected": 1,
        "rights_review_required": 5,
    }
    assert manifest["record_counts"] == {
        "attempted": 3,
        "accepted": 2,
        "rejected": 1,
        "exact_duplicates": 1,
    }
    assert manifest["accepted_records_by_language"] == {"uk": 1, "en": 1}
    assert manifest["datatrove_handoff"]["runtime_version"] == "0.10.0"
    assert manifest["datatrove_handoff"]["near_dedup_status"] == "NOT_RUN_BOUNDED_SAMPLE"
    assert (tmp_path / "records.jsonl").is_file()
    assert len(list((tmp_path / "accepted_text").glob("*.txt"))) == 2


def test_fetcher_cannot_smuggle_over_bound_payload(tmp_path: Path) -> None:
    registry = copy.deepcopy(_load())
    for source in registry["sources"]:
        if source["eligibility_status"] == "ELIGIBLE":
            source["acquisition_urls"] = [f"https://example.invalid/{source['language']}"]
    _rehash(registry)

    def fetcher(url: str, max_bytes: int) -> DownloadedBytes:
        return DownloadedBytes(b"x" * (max_bytes + 1), "text/plain")

    manifest = run_bounded_intake(
        registry,
        tmp_path,
        fetcher=fetcher,
        max_download_bytes=64,
        max_normalized_chars=100,
    )
    assert manifest["record_counts"]["accepted"] == 0
    assert manifest["record_counts"]["rejected"] == 2
    assert all(
        "violated max_download_bytes" in item["failure_reason"]
        for item in manifest["records"]
    )
