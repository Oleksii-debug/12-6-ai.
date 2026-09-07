from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools" / "retest_d03_languk_supreme_court.py"
spec = importlib.util.spec_from_file_location("languk_retest", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

CONFIG = json.loads(
    Path("configs/data/d03_languk_supreme_court_retest_v1.json").read_text(
        encoding="utf-8"
    )
)


def _occurrences(text: str, token: str) -> list[dict[str, int | str]]:
    result: list[dict[str, int | str]] = []
    start = 0
    while True:
        idx = text.find(token, start)
        if idx < 0:
            break
        result.append({"start": idx, "end": idx + len(token), "text": token})
        start = idx + len(token)
    return result


def make_row(record_id: str = "116075957", repeats: int = 3) -> dict:
    sentence = (
        "Верховний Суд України розглянув матеріали ОСОБА_1 за адресою АДРЕСА_1, "
        "врахував ІНФОРМАЦІЯ_1 та реєстраційний НОМЕР_1 і застосував норми права. "
    )
    text = sentence * repeats + (
        "Українське судове рішення містить достатній правовий аналіз і мотивування."
    )
    specs = {
        "number": ("НОМЕР_1", "number_count", "number_occurrences"),
        "information": ("ІНФОРМАЦІЯ_1", "information_count", "information_occurrences"),
        "person": ("ОСОБА_1", "person_count", "person_occurrences"),
        "address": ("АДРЕСА_1", "address_count", "address_occurrences"),
    }
    row = {
        "id": record_id,
        "text": text,
        "number_count": 0,
        "information_count": 0,
        "person_count": 0,
        "address_count": 0,
        "sum_of_unique_entities": 4,
        "number_occurrences": [],
        "information_occurrences": [],
        "person_occurrences": [],
        "address_occurrences": [],
        "__index_level_0__": int(record_id) % 1000,
    }
    for token, count_field, occurrence_field in specs.values():
        occurrences = _occurrences(text, token)
        row[count_field] = len({str(item["text"]) for item in occurrences})
        row[occurrence_field] = occurrences
    return row


def test_load_config_preserves_exact_schema_and_zero_credit() -> None:
    value = mod.load_config()
    assert value["parquet_schema"]["required_fields"] == list(mod.EXPECTED_FIELDS)
    assert value["privacy"]["verify_occurrence_spans"] is True
    assert value["claim_boundary"]["training_authorized_bytes"] == 0
    assert value["claim_boundary"]["model_training_executed"] is False


def test_repeated_placeholder_occurrences_preserve_unique_entity_count() -> None:
    row = make_row(repeats=3)
    assert row["person_count"] == 1
    assert len(row["person_occurrences"]) == 3
    ok, reason, text = mod.assess_row(row, CONFIG)
    assert ok is True
    assert reason == "accepted"
    assert text.count("ОСОБА_1") == 3


def test_unique_entity_count_mismatch_fails_closed() -> None:
    row = make_row()
    row["person_count"] += 1
    with pytest.raises(mod.RetestError, match="unique_entity_count_mismatch_person_count"):
        mod.assess_row(row, CONFIG)


def test_two_distinct_person_markers_count_as_two_unique_entities() -> None:
    row = make_row(repeats=1)
    row["text"] += " Додатково у справі згадано ОСОБА_2 як учасника провадження."
    row["person_occurrences"] = [
        *_occurrences(row["text"], "ОСОБА_1"),
        *_occurrences(row["text"], "ОСОБА_2"),
    ]
    row["person_count"] = 2
    row["sum_of_unique_entities"] = 5
    ok, reason, text = mod.assess_row(row, CONFIG)
    assert ok is True
    assert reason == "accepted"
    assert "ОСОБА_2" in text


def test_stale_occurrence_span_fails_closed() -> None:
    row = make_row()
    row["person_occurrences"][0]["start"] += 1
    with pytest.raises(
        mod.RetestError, match="occurrence_span_text_mismatch_person_occurrences"
    ):
        mod.assess_row(row, CONFIG)


def test_untracked_placeholder_fails_closed() -> None:
    row = make_row()
    row["text"] += " ОСОБА_99"
    with pytest.raises(
        mod.RetestError, match="untracked_or_stale_occurrences_person_occurrences"
    ):
        mod.assess_row(row, CONFIG)


def test_occurrence_token_must_match_its_category() -> None:
    row = make_row()
    row["person_occurrences"][0]["text"] = "АДРЕСА_1"
    with pytest.raises(
        mod.RetestError, match="invalid_occurrence_token_person_occurrences"
    ):
        mod.assess_row(row, CONFIG)


def test_sum_of_unique_entities_is_bound() -> None:
    row = make_row()
    row["sum_of_unique_entities"] = 3
    with pytest.raises(mod.RetestError, match="sum_of_unique_entities_inconsistent"):
        mod.assess_row(row, CONFIG)


def test_sum_of_unique_entities_rejects_malformed_value() -> None:
    row = make_row()
    row["sum_of_unique_entities"] = True
    with pytest.raises(mod.RetestError, match="invalid_sum_of_unique_entities"):
        mod.assess_row(row, CONFIG)


def test_schema_missing_or_extra_field_fails_closed() -> None:
    missing = make_row()
    missing.pop("person_occurrences")
    with pytest.raises(mod.RetestError, match="parquet row field drift"):
        mod.assess_row(missing, CONFIG)
    extra = make_row()
    extra["unexpected"] = "x"
    with pytest.raises(mod.RetestError, match="parquet row field drift"):
        mod.assess_row(extra, CONFIG)


def test_email_phone_and_control_char_quarantine() -> None:
    email = make_row()
    email["text"] += " test@example.org"
    assert mod.assess_row(email, CONFIG)[1] == "email"
    phone = make_row()
    phone["text"] += " +380 67 123 45 67"
    assert mod.assess_row(phone, CONFIG)[1] == "phone"
    control = make_row()
    control["text"] += "\x00"
    assert mod.assess_row(control, CONFIG)[1] == "control_character"


def test_duplicate_source_id_fails_closed() -> None:
    row = make_row()
    with pytest.raises(mod.RetestError, match="duplicate source id"):
        mod.select_rows([row, copy.deepcopy(row)], CONFIG)


def test_selection_order_is_numeric_not_lexicographic() -> None:
    rows = [make_row("10"), make_row("2"), make_row("100")]
    for row in rows:
        row["text"] += f" Унікальний український додаток {row['id']}."
    accepted, _ = mod.select_rows(rows, CONFIG)
    assert [row["record_id"] for row in accepted] == ["2", "10", "100"]


def test_exact_normalized_duplicate_is_removed_with_one_disposition_per_row() -> None:
    first = make_row("2")
    second = make_row("3")
    second["text"] = first["text"]
    for field in mod.EXPECTED_FIELDS:
        if (
            field.endswith("_occurrences")
            or field.endswith("_count")
            or field == "sum_of_unique_entities"
        ):
            second[field] = copy.deepcopy(first[field])
    accepted, reasons = mod.select_rows([first, second], CONFIG)
    assert len(accepted) == 1
    assert reasons["accepted"] == 1
    assert reasons["exact_normalized_duplicate"] == 1
    assert sum(reasons.values()) == 2


def test_rejected_row_gets_one_terminal_disposition() -> None:
    accepted_row = make_row("2")
    rejected_row = make_row("3")
    rejected_row["text"] += " test@example.org"
    accepted, reasons = mod.select_rows([accepted_row, rejected_row], CONFIG)
    assert len(accepted) == 1
    assert reasons["accepted"] == 1
    assert reasons["email"] == 1
    assert sum(reasons.values()) == 2


def test_config_mutations_fail_closed() -> None:
    mutations = [
        ("parquet_schema", "reject_extra_fields", False),
        ("privacy", "verify_occurrence_spans", False),
        ("privacy", "require_complete_occurrence_annotation", False),
        ("runtime", "pyarrow", "latest"),
        ("selection", "order", "lexicographic"),
        ("claim_boundary", "training_authorized_bytes", 1),
    ]
    for section, key, value in mutations:
        cfg = copy.deepcopy(CONFIG)
        cfg[section][key] = value
        path = Path("/tmp/languk-mutated-config.json")
        path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(mod.RetestError):
            mod.load_config(path)


def test_normalization_is_deterministic() -> None:
    text = "  Україна\r\n\r\n  Верховний   Суд  "
    assert mod.normalize(text) == "Україна\nВерховний Суд"
