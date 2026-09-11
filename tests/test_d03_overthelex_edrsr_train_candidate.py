from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools" / "qualify_d03_overthelex_edrsr_train.py"
spec = importlib.util.spec_from_file_location("edrsr_qualify", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

CONFIG = json.loads(
    (Path(__file__).parents[1] / "configs/data/d03_overthelex_edrsr_train_candidate_v1.json").read_text(encoding="utf-8")
)


def make_row(doc_id: int = 108303379) -> dict:
    facts = (
        "Позивач [PERSON] звернувся до суду та зазначив, що за договором між сторонами "
        "виник спір. Суд дослідив письмові докази, пояснення учасників і норми українського "
        "законодавства. Відомості про адресу [ADDRESS], номер [NUMBER] та іншу інформацію "
        "[INFO] у відкритому тексті замінено анонімізаційними позначками."
    )
    return {
        "doc_id": doc_id,
        "justice_kind": 1,
        "justice_kind_name": "civil",
        "judgment_code": 3,
        "category_code": 1,
        "court_code": 123,
        "judge": "Public Official",
        "adjudication_date": "2022-01-01",
        "facts": facts,
        "dispositive": "MUST NEVER BE EMITTED",
        "outcome": "granted",
        "epoch": "full_scale",
        "year": 2022,
        "full_text_length": len(facts),
    }


def test_load_config_binds_train_only_and_zero_credit() -> None:
    cfg = mod.load_config(Path(__file__).parents[1] / "configs/data/d03_overthelex_edrsr_train_candidate_v1.json")
    assert cfg["source"]["sha256"] == mod.SOURCE_SHA256
    assert cfg["source"]["split"] == "train"
    assert cfg["evaluation_boundary"]["excluded_splits"] == ["validation", "test"]
    assert cfg["evaluation_boundary"]["reserved_evaluation_decontamination_required"] is True
    assert cfg["claim_boundary"]["training_authorized_bytes"] == 0


def test_assess_accepts_normalized_placeholders() -> None:
    ok, reason, text = mod.assess_row(make_row(), CONFIG)
    assert ok is True
    assert reason == "accepted"
    assert "[PERSON]" in text


def test_candidate_emits_facts_only_not_labels_or_metadata() -> None:
    rows, reasons = mod.select_rows([make_row()], CONFIG)
    assert reasons["accepted"] == 1
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {
        "record_id", "source_family", "language", "modality",
        "normalized_sha256", "normalized_bytes", "text",
    }
    assert "MUST NEVER BE EMITTED" not in row["text"]
    assert "Public Official" not in row["text"]
    assert "granted" not in row["text"]


@pytest.mark.parametrize(
    ("suffix", "reason"),
    [
        (" ОСОБА_1", "legacy_anonymization_marker"),
        (" test@example.org", "email"),
        (" +380 67 123 45 67", "phone"),
        ("\x00", "control_character"),
    ],
)
def test_privacy_failures_are_quarantined(suffix: str, reason: str) -> None:
    row = make_row()
    row["facts"] += suffix
    assert mod.assess_row(row, CONFIG)[1] == reason


def test_schema_drift_and_duplicate_ids_fail_closed() -> None:
    extra = make_row()
    extra["unexpected"] = "x"
    with pytest.raises(mod.QualificationError, match="parquet row field drift"):
        mod.assess_row(extra, CONFIG)
    row = make_row()
    with pytest.raises(mod.QualificationError, match="duplicate doc_id"):
        mod.select_rows([row, copy.deepcopy(row)], CONFIG)


def test_selection_is_numeric_and_exact_normalized_dedup_is_removed() -> None:
    first = make_row(20)
    second = make_row(3)
    second["facts"] += " Додатковий український абзац із поясненням права."
    duplicate = make_row(30)
    accepted, reasons = mod.select_rows([first, second, duplicate], CONFIG)
    assert [row["record_id"] for row in accepted] == ["3", "20"]
    assert reasons["exact_normalized_duplicate"] == 1


def test_candidate_byte_cap_is_fail_closed_and_deterministic() -> None:
    cfg = copy.deepcopy(CONFIG)
    one = make_row(1)
    two = make_row(2)
    two["facts"] += " Додатковий український текст."
    first_bytes = len(mod.normalize(one["facts"]).encode("utf-8"))
    cfg["selection"]["max_candidate_normalized_bytes"] = first_bytes
    accepted, reasons = mod.select_rows([two, one], cfg)
    assert [row["record_id"] for row in accepted] == ["1"]
    assert reasons["candidate_byte_cap_reached"] == 1


def test_config_mutations_fail_closed() -> None:
    base = Path(__file__).parents[1] / "configs/data/d03_overthelex_edrsr_train_candidate_v1.json"
    cfg = json.loads(base.read_text(encoding="utf-8"))
    mutations = [
        ("source", "file", "test.parquet"),
        ("selection", "text_column", "dispositive"),
        ("privacy", "reject_email", False),
        ("evaluation_boundary", "training_split_only", False),
        ("evaluation_boundary", "reserved_evaluation_decontamination_required", False),
        ("claim_boundary", "training_authorized_bytes", 1),
    ]
    for section, key, value in mutations:
        mutated = copy.deepcopy(cfg)
        mutated[section][key] = value
        path = Path("/tmp/d03-edrsr-mutated-config.json")
        path.write_text(json.dumps(mutated, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(mod.QualificationError):
            mod.load_config(path)


def test_normalization_is_deterministic() -> None:
    assert mod.normalize("  Україна\r\n\r\n  Суд   вирішив  ") == "Україна\nСуд вирішив"
