from __future__ import annotations

import json
from pathlib import Path

from twelve_six.data.pipeline import normalize_text
from twelve_six.data.ukrainian_normalization import (
    NORMALIZATION_SCHEMA,
    normalize_document,
    summarize_changes,
)

FIXTURE = Path("tests/fixtures/ukrainian_normalization_regression_v1.jsonl")
S0_RAW = Path("data/s0/raw/project_authored.jsonl")


def _fixture_cases() -> list[dict[str, object]]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def test_regression_corpus_exact_outputs_and_reason_counters() -> None:
    for case in _fixture_cases():
        result = normalize_document(
            case["input"],  # type: ignore[arg-type]
            modality=case["modality"],  # type: ignore[arg-type]
            source_id="regression",
            source_version="v1",
            raw_document_id=case["case_id"],  # type: ignore[arg-type]
        )
        assert result.text == case["expected"], case["case_id"]
        assert result.trace.reason_counts == case.get("reason_counts", {}), case["case_id"]
        assert result.trace.schema == NORMALIZATION_SCHEMA
        assert len(result.trace.raw_text_sha256) == 64
        assert len(result.trace.normalized_text_sha256) == 64


def test_meaningful_ukrainian_orthography_and_punctuation_are_not_folded() -> None:
    text = "Ґанок і ґрунт; гірка, їжак, йод. п'ять п’ять пʼять. «Так» — “так” – так. ①"
    normalized = normalize_document(text).text
    assert normalized == text
    assert "ґ" in normalized and "г" in normalized
    assert "і" in normalized and "ї" in normalized and "й" in normalized
    assert "'" in normalized and "’" in normalized and "ʼ" in normalized
    assert "—" in normalized and "–" in normalized and "①" in normalized


def test_no_global_lowercase_or_transliteration() -> None:
    text = "УкраЇНА, ҐРУНТ і Kyiv залишаються як у джерелі."
    normalized = normalize_document(text).text
    assert normalized == text
    assert "УкраЇНА" in normalized
    assert "ҐРУНТ" in normalized
    assert "Kyiv" in normalized


def test_code_path_only_canonicalizes_line_endings() -> None:
    raw = "def f():\r\n\tvalue  = ' ­①'\r\n\treturn value\r\n"
    result = normalize_document(raw, modality="code")
    assert result.text == "def f():\n\tvalue  = ' ­①'\n\treturn value\n"
    assert result.trace.reason_counts == {"crlf_to_lf": 3}
    assert " " in result.text
    assert "­" in result.text
    assert "①" in result.text
    assert "\t" in result.text
    assert "value  =" in result.text


def test_trace_binds_normalized_document_to_raw_identity() -> None:
    raw = "\ufeffУкраїнська мова\r\n"
    result = normalize_document(
        raw,
        source_id="ua-source",
        source_version="snapshot-7",
        raw_document_id="doc-42",
        raw_source_sha256="a" * 64,
    )
    trace = result.trace
    assert trace.source_id == "ua-source"
    assert trace.source_version == "snapshot-7"
    assert trace.raw_document_id == "doc-42"
    assert trace.raw_source_sha256 == "a" * 64
    assert trace.raw_text_sha256 != trace.normalized_text_sha256
    assert trace.normalized_text_sha256 == normalize_document(result.text).trace.normalized_text_sha256


def test_normalization_is_idempotent() -> None:
    first = normalize_document("\n<p>їжак&nbsp;і ґрунт</p>\r\n")
    second = normalize_document(first.text)
    assert second.text == first.text
    assert second.trace.reason_counts == {}


def test_d03_compatibility_wrapper_uses_new_nfc_policy() -> None:
    assert normalize_text("їжак ①") == "їжак ①"
    code = "x  = ' ①'\r\n"
    assert normalize_text(code, modality="code") == "x  = ' ①'\n"


def test_current_real_s0_ukrainian_sample_has_quantified_zero_drift() -> None:
    records = [json.loads(line) for line in S0_RAW.read_text(encoding="utf-8").splitlines()]
    uk_records = [record for record in records if record.get("language_hint") == "uk"]
    results = tuple(
        normalize_document(
            record["text"],
            source_id="project-authored-s0-fixture-v1",
            source_version="project-authored-s0-fixture-v1",
            raw_document_id=record["document_id"],
        )
        for record in uk_records
    )
    summary = summarize_changes(results)
    assert summary.documents == 6
    assert summary.changed_documents == 0
    assert summary.raw_codepoints == 811
    assert summary.normalized_codepoints == 811
    assert summary.codepoint_delta == 0
    assert summary.raw_byte_tokens == 1515
    assert summary.normalized_byte_tokens == 1515
    assert summary.byte_token_delta == 0
    assert summary.reason_counts == {}
