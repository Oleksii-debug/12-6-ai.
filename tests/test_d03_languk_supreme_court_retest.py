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


def _append_before_terminal_nul(row: dict, suffix: str) -> None:
    assert row["text"].endswith("\x00")
    row["text"] = row["text"][:-1] + suffix + "\x00"


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
        "information": (
            "ІНФОРМАЦІЯ_1",
            "information_count",
            "information_occurrences",
        ),
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
        row[count_field] = len(occurrences)
        row[occurrence_field] = occurrences
    row["text"] = text + "\x00"
    return row


def test_load_config_preserves_exact_schema_framing_and_zero_credit() -> None:
    value = mod.load_config()
    assert value["parquet_schema"]["required_fields"] == list(mod.EXPECTED_FIELDS)
    assert value["source_text_framing"]["terminal_codepoint"] == "U+0000"
    assert value["source_text_framing"]["required_count_per_row"] == 1
    assert value["source_text_framing"]["require_terminal_position"] is True
    assert value["source_text_framing"]["strip_after_annotation_validation"] is True
    assert value["source_text_framing"]["reject_internal_terminal_codepoint"] is True
    assert value["privacy"]["verify_occurrence_spans"] is True
    assert type(value["claim_boundary"]["training_authorized_bytes"]) is int
    assert value["claim_boundary"]["training_authorized_bytes"] == 0
    assert value["claim_boundary"]["model_training_executed"] is False


def test_repeated_placeholder_count_is_occurrence_count() -> None:
    row = make_row(repeats=3)
    assert row["person_count"] == 3
    assert len(row["person_occurrences"]) == 3
    ok, reason, text = mod.assess_row(row, CONFIG)
    assert ok is True
    assert reason == "accepted"
    assert text.count("ОСОБА_1") == 3
    assert "\x00" not in text


def test_terminal_nul_framing_is_required_and_stripped() -> None:
    row = make_row()
    assert row["text"].count("\x00") == 1
    assert row["text"].endswith("\x00")
    ok, reason, text = mod.assess_row(row, CONFIG)
    assert (ok, reason) == (True, "accepted")
    assert not text.endswith("\x00")
    assert "\x00" not in text


def test_missing_or_internal_nul_framing_is_quarantined() -> None:
    missing = make_row()
    missing["text"] = missing["text"][:-1]
    assert mod.assess_row(missing, CONFIG)[:2] == (
        False,
        "source_text_framing_inconsistent",
    )

    internal = make_row()
    internal["text"] = internal["text"][:-1] + "\x00X\x00"
    assert mod.assess_row(internal, CONFIG)[:2] == (
        False,
        "source_text_framing_inconsistent",
    )


def test_occurrence_count_mismatch_fails_closed_in_validator() -> None:
    row = make_row()
    row["person_count"] += 1
    with pytest.raises(mod.RetestError, match="occurrence_count_mismatch_person_count"):
        mod._validate_occurrences(row, row["text"])


def test_occurrence_count_mismatch_is_quarantined_by_selection() -> None:
    good = make_row("2")
    bad = make_row("3")
    bad["person_count"] += 1
    accepted, reasons = mod.select_rows([good, bad], CONFIG)
    assert [row["record_id"] for row in accepted] == ["2"]
    assert reasons["accepted"] == 1
    assert reasons["annotation_contract_inconsistent"] == 1
    assert sum(reasons.values()) == 2


def test_two_distinct_person_markers_keep_nonempty_category_sum() -> None:
    row = make_row(repeats=1)
    _append_before_terminal_nul(
        row, " Додатково у справі згадано ОСОБА_2 як учасника провадження."
    )
    row["person_occurrences"] = [
        *_occurrences(row["text"], "ОСОБА_1"),
        *_occurrences(row["text"], "ОСОБА_2"),
    ]
    row["person_count"] = 2
    row["sum_of_unique_entities"] = 4
    ok, reason, text = mod.assess_row(row, CONFIG)
    assert ok is True
    assert reason == "accepted"
    assert "ОСОБА_2" in text


def test_stale_occurrence_span_is_quarantined() -> None:
    row = make_row()
    row["person_occurrences"][0]["start"] += 1
    assert mod.assess_row(row, CONFIG)[:2] == (
        False,
        "annotation_contract_inconsistent",
    )


def test_untracked_placeholder_is_quarantined() -> None:
    row = make_row()
    _append_before_terminal_nul(row, " ОСОБА_99")
    assert mod.assess_row(row, CONFIG)[:2] == (
        False,
        "annotation_contract_inconsistent",
    )


@pytest.mark.parametrize(
    "token",
    [
        "ОСОБА_X",
        "ОСОБА_0",
        "ОСОБА_",
        "АДРЕСА_X",
        "ІНФОРМАЦІЯ_0",
        "НОМЕР_1X",
        "ОСОБА_1_2",
    ],
)
def test_malformed_anonymization_marker_prefix_is_quarantined(token: str) -> None:
    row = make_row()
    _append_before_terminal_nul(row, f" {token}")
    assert mod.assess_row(row, CONFIG)[:2] == (
        False,
        "annotation_contract_inconsistent",
    )


def test_new_canonical_marker_with_complete_annotation_is_accepted() -> None:
    row = make_row(repeats=1)
    _append_before_terminal_nul(
        row, " Додатково у справі згадано ОСОБА_99 як учасника провадження."
    )
    row["person_occurrences"] = [
        *_occurrences(row["text"], "ОСОБА_1"),
        *_occurrences(row["text"], "ОСОБА_99"),
    ]
    row["person_count"] = len(row["person_occurrences"])
    ok, reason, text = mod.assess_row(row, CONFIG)
    assert ok is True
    assert reason == "accepted"
    assert "ОСОБА_99" in text


def test_occurrence_token_category_mismatch_is_quarantined() -> None:
    row = make_row()
    row["person_occurrences"][0]["text"] = "АДРЕСА_1"
    assert mod.assess_row(row, CONFIG)[:2] == (
        False,
        "annotation_contract_inconsistent",
    )


def test_sum_of_unique_entities_is_bound_to_nonempty_categories() -> None:
    row = make_row()
    row["sum_of_unique_entities"] = 3
    assert mod.assess_row(row, CONFIG)[:2] == (
        False,
        "annotation_contract_inconsistent",
    )


def test_sum_of_unique_entities_rejects_malformed_value() -> None:
    row = make_row()
    row["sum_of_unique_entities"] = True
    assert mod.assess_row(row, CONFIG)[:2] == (
        False,
        "annotation_contract_inconsistent",
    )


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
    _append_before_terminal_nul(email, " test@example.org")
    assert mod.assess_row(email, CONFIG)[1] == "email"
    phone = make_row()
    _append_before_terminal_nul(phone, " +380 67 123 45 67")
    assert mod.assess_row(phone, CONFIG)[1] == "phone"
    control = make_row()
    _append_before_terminal_nul(control, "\x01")
    assert mod.assess_row(control, CONFIG)[1] == "control_character"


def test_duplicate_source_id_fails_closed() -> None:
    row = make_row()
    with pytest.raises(mod.RetestError, match="duplicate source id"):
        mod.select_rows([row, copy.deepcopy(row)], CONFIG)


def test_selection_order_is_numeric_not_lexicographic() -> None:
    rows = [make_row("10"), make_row("2"), make_row("100")]
    for row in rows:
        _append_before_terminal_nul(row, f" Унікальний український додаток {row['id']}.")
    accepted, _ = mod.select_rows(rows, CONFIG)
    assert [row["record_id"] for row in accepted] == ["2", "10", "100"]


def test_exact_normalized_duplicate_is_removed_with_one_disposition_per_row() -> None:
    first = make_row("2")
    second = make_row("3")
    second["text"] = first["text"]
    for field in mod.EXPECTED_FIELDS:
        if (
            field.endswith(("_occurrences", "_count"))
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
    _append_before_terminal_nul(rejected_row, " test@example.org")
    accepted, reasons = mod.select_rows([accepted_row, rejected_row], CONFIG)
    assert len(accepted) == 1
    assert reasons["accepted"] == 1
    assert reasons["email"] == 1
    assert sum(reasons.values()) == 2


def test_config_mutations_fail_closed() -> None:
    mutations = [
        ("parquet_schema", "reject_extra_fields", False),
        ("source_text_framing", "require_terminal_position", False),
        ("source_text_framing", "strip_after_annotation_validation", False),
        ("privacy", "verify_occurrence_spans", False),
        ("privacy", "require_complete_occurrence_annotation", False),
        ("runtime", "pyarrow", "latest"),
        ("selection", "order", "lexicographic"),
        ("claim_boundary", "training_authorized_bytes", 1),
        ("claim_boundary", "training_authorized_bytes", False),
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
