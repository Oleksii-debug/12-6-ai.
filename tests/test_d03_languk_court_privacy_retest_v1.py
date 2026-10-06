from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools" / "run_d03_languk_court_privacy_retest_v1.py"
SOURCE_CONTRACT_PATH = ROOT / "tools" / "retest_d03_languk_supreme_court.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load(RUNNER_PATH, "current_languk_retest")
source_contract = _load(SOURCE_CONTRACT_PATH, "historical_languk_contract")
source_contract_config = source_contract.load_config(
    ROOT / "configs" / "data" / "d03_languk_supreme_court_retest_v1.json"
)


def _occurrences(text: str, token: str) -> list[dict[str, int | str]]:
    result: list[dict[str, int | str]] = []
    start = 0
    while True:
        index = text.find(token, start)
        if index < 0:
            return result
        result.append({"start": index, "end": index + len(token), "text": token})
        start = index + len(token)


def _row(record_id: str = "116075957") -> dict:
    text = (
        "Верховний Суд України розглянув матеріали ОСОБА_1 за адресою АДРЕСА_1, "
        "врахував ІНФОРМАЦІЯ_1 та реєстраційний НОМЕР_1 і застосував норми права. "
        "Українське судове рішення містить достатній правовий аналіз."
    )
    values = {
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
        "__index_level_0__": 1,
    }
    bindings = (
        ("НОМЕР_1", "number_count", "number_occurrences"),
        ("ІНФОРМАЦІЯ_1", "information_count", "information_occurrences"),
        ("ОСОБА_1", "person_count", "person_occurrences"),
        ("АДРЕСА_1", "address_count", "address_occurrences"),
    )
    for token, count_key, occurrences_key in bindings:
        occurrences = _occurrences(text, token)
        values[count_key] = len(occurrences)
        values[occurrences_key] = occurrences
    values["text"] = text + "\x00"
    return values


def _prepare(row: dict):
    return runner._prepare_source_row(
        row,
        source_contract=source_contract,
        source_contract_config=source_contract_config,
        row_index=0,
    )


def test_valid_row_is_stripped_only_after_annotation_validation() -> None:
    row = _row()
    prepared, disposition = _prepare(row)
    assert disposition == "accepted"
    assert prepared is not None
    assert "\x00" not in prepared["normalized_text"]
    assert prepared["normalized_text"] == source_contract.normalize(row["text"][:-1])
    assert prepared["raw_text"].endswith("\x00")


@pytest.mark.parametrize("mutation", ["missing", "internal_and_terminal"])
def test_terminal_nul_framing_fails_closed(mutation: str) -> None:
    row = _row()
    if mutation == "missing":
        row["text"] = row["text"][:-1]
    else:
        row["text"] = row["text"][:-1] + "\x00X\x00"
    prepared, disposition = _prepare(row)
    assert prepared is None
    assert disposition == "source_text_framing_inconsistent"


def test_stale_occurrence_annotation_fails_before_sentinel_stripping() -> None:
    row = _row()
    row["person_occurrences"][0]["start"] += 1
    prepared, disposition = _prepare(row)
    assert prepared is None
    assert disposition == "annotation_contract_inconsistent"


def test_exact_annotation_schema_is_required() -> None:
    row = _row()
    row["unexpected"] = "x"
    with pytest.raises(runner.LangUkCourtRetestError, match="annotation schema drift"):
        _prepare(row)


def test_non_nul_control_remains_rejected() -> None:
    row = _row()
    row["text"] = row["text"][:-1] + "\x01\x00"
    prepared, disposition = _prepare(row)
    assert prepared is None
    assert disposition == "control_character"


def test_source_contract_does_not_replace_current_privacy_gate() -> None:
    row = _row()
    row["text"] = row["text"][:-1] + " test@example.org\x00"
    prepared, disposition = _prepare(row)
    assert disposition == "accepted"
    assert prepared is not None
    assert "test@example.org" in prepared["normalized_text"]


def test_source_contract_does_not_replace_current_quality_gate() -> None:
    row = _row()
    short = "Україна"
    row.update(
        {
            "text": short + "\x00",
            "number_count": 0,
            "information_count": 0,
            "person_count": 0,
            "address_count": 0,
            "sum_of_unique_entities": 0,
            "number_occurrences": [],
            "information_occurrences": [],
            "person_occurrences": [],
            "address_occurrences": [],
        }
    )
    prepared, disposition = _prepare(row)
    assert disposition == "accepted"
    assert prepared is not None
    assert prepared["normalized_text"] == short


def test_helper_does_not_mutate_source_annotation_object() -> None:
    row = _row()
    original = copy.deepcopy(row)
    prepared, disposition = _prepare(row)
    assert disposition == "accepted"
    assert prepared is not None
    assert row == original
