from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.measure_tokenizer_efficiency import CalibrationError, measure


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_measure_reports_real_utf8_fertility_by_stratum(tmp_path: Path) -> None:
    source = tmp_path / "slice.jsonl"
    write_jsonl(
        source,
        [
            {"language": "uk", "text": "Привіт"},
            {"language": "en", "text": "hello"},
            {"language": "code", "text": "x=1"},
        ],
    )

    report = measure(source, corpus_identity="sha256:test-corpus")

    assert report["current_tokenizer"]["loss_position_unit"] == "UTF8_BYTE_POSITION"
    assert report["by_language"]["uk"]["unicode_codepoints"] == 6
    assert report["by_language"]["uk"]["utf8_bytes"] == 12
    assert report["by_language"]["uk"]["utf8_bytes_per_codepoint"] == 2.0
    assert report["by_language"]["en"]["utf8_bytes_per_codepoint"] == 1.0
    assert report["by_language"]["code"]["utf8_bytes_per_codepoint"] == 1.0
    assert report["unit_policy"]["direct_byte_to_subword_scaling_law_conversion_allowed"] is False


def test_measure_requires_all_ua_en_code_strata(tmp_path: Path) -> None:
    source = tmp_path / "slice.jsonl"
    write_jsonl(
        source,
        [
            {"language": "uk", "text": "Текст"},
            {"language": "en", "text": "text"},
        ],
    )

    with pytest.raises(CalibrationError, match="missing: code"):
        measure(source, corpus_identity="sha256:test-corpus")


def test_reference_subword_counts_cannot_be_partial(tmp_path: Path) -> None:
    source = tmp_path / "slice.jsonl"
    write_jsonl(
        source,
        [
            {"language": "uk", "text": "Привіт", "reference_subword_tokens": 2},
            {"language": "en", "text": "hello"},
            {"language": "code", "text": "x=1", "reference_subword_tokens": 3},
        ],
    )

    with pytest.raises(CalibrationError, match="all or none"):
        measure(source, corpus_identity="sha256:test-corpus")


def test_reference_subword_counts_are_reported_without_unit_conversion(tmp_path: Path) -> None:
    source = tmp_path / "slice.jsonl"
    write_jsonl(
        source,
        [
            {"language": "uk", "text": "Привіт", "reference_subword_tokens": 2},
            {"language": "en", "text": "hello", "reference_subword_tokens": 1},
            {"language": "code", "text": "x=1", "reference_subword_tokens": 3},
        ],
    )

    report = measure(source, corpus_identity="sha256:test-corpus")

    assert report["overall"]["reference_subword_coverage"] == "COMPLETE"
    assert report["overall"]["utf8_bytes_per_reference_subword_token"] == round(20 / 6, 6)
    assert report["overall"]["codepoints_per_reference_subword_token"] == round(14 / 6, 6)
