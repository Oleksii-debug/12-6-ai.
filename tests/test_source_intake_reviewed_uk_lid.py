from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from twelve_six.data.source_intake import DownloadedBytes, run_bounded_intake

REGISTRY = Path("configs/data/external_source_candidates_ua_en_v1.json")


def _load() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _rehash(registry: dict) -> None:
    core = dict(registry)
    core.pop("registry_identity_sha256", None)
    payload = (
        json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    registry["registry_identity_sha256"] = hashlib.sha256(payload).hexdigest()


def _fixture_registry() -> dict:
    registry = copy.deepcopy(_load())
    for source in registry["sources"]:
        if source["eligibility_status"] != "ELIGIBLE":
            continue
        source["acquisition_urls"] = [
            f"https://example.invalid/{source['language']}"
        ]
    _rehash(registry)
    return registry


def _english_fixture() -> bytes:
    return (
        b"This manual contains stable English language data for the model and "
        b"provides enough words for deterministic source-intake validation."
    )


def test_reviewed_uk_source_accepts_dominant_cyrillic_with_latin_identifiers(
    tmp_path: Path,
) -> None:
    registry = _fixture_registry()
    ukrainian = (
        "Український закон визначає права та обов'язки громадян України. "
        "Цей закон і ці дані потрібні для української мови та перевірки моделі. "
        "Український текст залишається основним, а латинські позначення нижче "
        "є лише технічними ідентифікаторами змін до закону. "
    ) * 8 + " Amendment identifiers: VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX ABCDEF."
    payloads = {
        "https://example.invalid/uk": DownloadedBytes(
            ukrainian.encode(), "text/plain; charset=utf-8"
        ),
        "https://example.invalid/en": DownloadedBytes(
            _english_fixture(), "text/plain; charset=utf-8"
        ),
    }

    manifest = run_bounded_intake(
        registry,
        tmp_path,
        fetcher=lambda url, _max_bytes: payloads[url],
        max_download_bytes=100_000,
        max_normalized_chars=50_000,
    )

    assert manifest["accepted_records_by_language"] == {"uk": 1, "en": 1}
    uk_record = next(
        record
        for record in manifest["records"]
        if record.get("source_id") == "ua.rada.open-data.laws-texts"
    )
    assert uk_record["status"] == "ACCEPTED"
    assert uk_record["language"] == "uk"
    assert uk_record["language_reason"] == "uk-reviewed-source-dominant-cyrillic"
    assert uk_record["language_confidence"] > 0.8


def test_balanced_mixed_text_remains_rejected_for_reviewed_uk_source(
    tmp_path: Path,
) -> None:
    registry = _fixture_registry()
    ukrainian = (
        "Українська мова і дані для моделі, цей закон та права громадян України. "
    ) * 3
    english = (
        "This English section is intentionally substantial and balanced with the "
        "Ukrainian section so a reviewed source cannot override genuinely mixed text. "
    ) * 4
    payloads = {
        "https://example.invalid/uk": DownloadedBytes(
            (ukrainian + english).encode(), "text/plain; charset=utf-8"
        ),
        "https://example.invalid/en": DownloadedBytes(
            _english_fixture(), "text/plain; charset=utf-8"
        ),
    }

    manifest = run_bounded_intake(
        registry,
        tmp_path,
        fetcher=lambda url, _max_bytes: payloads[url],
        max_download_bytes=100_000,
        max_normalized_chars=50_000,
    )

    assert manifest["accepted_records_by_language"]["uk"] == 0
    uk_record = next(
        record
        for record in manifest["records"]
        if record.get("source_id") == "ua.rada.open-data.laws-texts"
    )
    assert uk_record["status"] == "REJECTED"
    assert "expected uk, detected mixed" in uk_record["failure_reason"]
