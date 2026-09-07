from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools" / "retest_d03_languk_supreme_court.py"
spec = importlib.util.spec_from_file_location("languk_retest", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def cfg() -> dict:
    return mod.load_config()


def good_row(record_id: str = "2") -> dict:
    return {
        "id": record_id,
        "text": (
            "Верховний Суд України розглянув справу ОСОБА_1 за адресою АДРЕСА_1. "
            "Матеріали містять ІНФОРМАЦІЯ_1 та номер НОМЕР_1. "
            "Суд перевірив доводи сторін, застосував норми права та постановив рішення. "
            "Цей додатковий український текст забезпечує достатню довжину для тесту якості."
        ),
        "person_count": 1,
        "address_count": 1,
        "information_count": 1,
        "number_count": 1,
    }


def test_load_config_preserves_zero_credit_boundary() -> None:
    value = cfg()
    assert value["claim_boundary"]["training_authorized_bytes"] == 0
    assert value["claim_boundary"]["canonical_capacity_credited"] == 0
    assert value["claim_boundary"]["model_training_executed"] is False


def test_accepts_consistent_anonymized_ukrainian_row() -> None:
    ok, reason, text = mod.assess_row(good_row(), cfg())
    assert ok is True
    assert reason == "accepted"
    assert "ОСОБА_1" in text


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"person_count": 2}, "marker_mismatch_person_count"),
        ({"address_count": 0}, "marker_mismatch_address_count"),
        ({"information_count": 0}, "marker_mismatch_information_count"),
        ({"number_count": 0}, "marker_mismatch_number_count"),
    ],
)
def test_marker_count_mismatch_quarantines(mutation: dict, reason: str) -> None:
    row = good_row()
    row.update(mutation)
    ok, observed, _ = mod.assess_row(row, cfg())
    assert ok is False
    assert observed == reason


def test_missing_marker_count_schema_fails_closed() -> None:
    row = {"id": "1", "text": good_row()["text"]}
    ok, reason, _ = mod.assess_row(row, cfg())
    assert ok is False
    assert reason == "marker_count_schema_missing"


def test_email_and_phone_quarantine() -> None:
    email = good_row()
    email["text"] += " test@example.org"
    assert mod.assess_row(email, cfg())[1] == "email"
    phone = good_row()
    phone["text"] += " +380 67 123 45 67"
    assert mod.assess_row(phone, cfg())[1] == "phone"


def test_control_character_quarantine() -> None:
    row = good_row()
    row["text"] += "\x00"
    assert mod.assess_row(row, cfg())[1] == "control_character"


def test_low_ukrainian_quality_quarantines() -> None:
    row = good_row()
    row["text"] = "A" * 500
    row.update(person_count=0, address_count=0, information_count=0, number_count=0)
    assert mod.assess_row(row, cfg())[1] == "low_cyrillic_ratio"


def test_selection_is_stable_bounded_and_exact_deduped() -> None:
    value = cfg()
    value["selection"]["max_records"] = 2
    row_a = good_row("2")
    row_b = good_row("1")
    row_c = good_row("3")
    row_c["text"] = row_b["text"]
    accepted, reasons = mod.select_rows([row_a, row_c, row_b], value)
    assert [row["record_id"] for row in accepted] == ["1", "2"]
    assert len({row["normalized_sha256"] for row in accepted}) == 2
    assert reasons["accepted"] >= 2


def test_normalization_is_deterministic() -> None:
    text = "  Україна\r\n\r\n  Верховний   Суд  "
    assert mod.normalize(text) == "Україна\nВерховний Суд"
