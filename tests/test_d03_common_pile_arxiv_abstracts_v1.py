from __future__ import annotations

import copy
import gzip
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1] / "tools" / "materialize_d03_common_pile_arxiv_abstracts_v1.py"
)
spec = importlib.util.spec_from_file_location("common_pile_arxiv_intake", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

CONFIG = mod.load_config()


def make_row(record_id: str = "2401.00001", suffix: str = "") -> dict:
    text = (
        "We present a reproducible scientific analysis of language models and "
        "optimization under constrained compute. The method compares deterministic "
        "training procedures, measures generalization on held-out observations, and "
        "reports statistically grounded results for future research. "
        f"{suffix}"
    )
    return {
        "id": record_id,
        "text": text,
        "source": "arxiv-abstracts",
        "created": "2024-01-01T00:00:00",
        "added": "2024-08-12T18:16:26.585370",
        "metadata": {
            "license": (
                "Creative Commons Zero - Public Domain - "
                "https://creativecommons.org/publicdomain/zero/1.0/"
            ),
            "full_text_license": "None",
            "authors": "Example Author",
        },
    }


def test_config_binds_merged_rights_and_exact_immutable_shard() -> None:
    cfg = mod.load_config()
    assert cfg["parent_rights"]["merged_pr"] == 769
    assert cfg["parent_rights"]["source_key"] == "arxiv_abstracts"
    assert cfg["source"]["revision"] == mod.SOURCE_REVISION
    assert cfg["source"]["sha256"] == mod.SOURCE_SHA256
    assert cfg["source"]["bytes"] == 232_616_926
    assert cfg["claim_boundary"]["training_authorized_bytes"] == 0
    assert cfg["claim_boundary"]["model_training_executed"] is False


def test_source_identity_mutation_fails_closed(tmp_path: Path) -> None:
    cfg = copy.deepcopy(CONFIG)
    cfg["source"]["sha256"] = "0" * 64
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(mod.ArxivAbstractsIntakeError, match="immutable source contract drift"):
        mod.load_config(path)


def test_rights_binding_mutation_fails_closed(tmp_path: Path) -> None:
    cfg = copy.deepcopy(CONFIG)
    cfg["parent_rights"]["required_rights_signal"] = "CC-BY"
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(mod.ArxivAbstractsIntakeError, match="parent rights binding drift"):
        mod.load_config(path)


def test_valid_row_is_normalized_and_retained() -> None:
    row = make_row()
    row["text"] = "  " + row["text"].replace(" ", "  ") + "\r\n"
    ok, reason, text = mod.assess_row(row, CONFIG)
    assert ok is True
    assert reason == "accepted"
    assert "\r" not in text
    assert "  " not in text


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (lambda row: row.update(source="other-source"), "source_label_mismatch"),
        (
            lambda row: row["metadata"].update(license="Creative Commons Attribution"),
            "metadata_license_mismatch",
        ),
        (lambda row: row.update(text=row["text"] + " research@example.org"), "email"),
        (lambda row: row.update(text=row["text"] + " +1 415 555 0123"), "phone"),
        (lambda row: row.update(text=row["text"] + "\x00"), "control_character"),
        (lambda row: row.update(text="Short abstract."), "too_short"),
        (lambda row: row.update(text="1 + 2 = 3; " * 80), "low_alpha_fraction"),
    ],
)
def test_row_level_filters_quarantine_without_credit(mutator, reason: str) -> None:
    row = make_row()
    mutator(row)
    ok, actual, text = mod.assess_row(row, CONFIG)
    assert ok is False
    assert actual == reason
    assert text == ""


def test_structural_schema_drift_is_source_fatal() -> None:
    row = make_row()
    row["unexpected"] = "x"
    with pytest.raises(mod.ArxivAbstractsIntakeError, match="source row field drift"):
        mod.assess_row(row, CONFIG)


def test_selection_is_deterministic_deduplicated_and_bounded() -> None:
    cfg = copy.deepcopy(CONFIG)
    cfg["selection"]["max_scanned_records"] = 5
    cfg["selection"]["max_retained_records"] = 2

    first = make_row("1", "first unique record")
    duplicate = copy.deepcopy(first)
    duplicate["id"] = "2"
    third = make_row("3", "third unique record")
    fourth = make_row("4", "fourth unique record")
    fifth = make_row("5", "fifth unique record")
    sixth_unscanned = make_row("6", "sixth unique record")

    accepted, reasons, scanned = mod.select_rows(
        [first, duplicate, third, fourth, fifth, sixth_unscanned],
        cfg,
    )
    assert scanned == 5
    assert [row["record_id"] for row in accepted] == ["1", "3"]
    assert reasons["accepted"] == 2
    assert reasons["exact_normalized_duplicate"] == 1
    assert reasons["retained_cap_reached"] == 2
    assert sum(reasons.values()) == 5


def test_gzip_jsonl_reader_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(b'{"id":"ok"}\n')
        handle.write(b'not-json\n')
    iterator = iter(mod._iter_gzip_jsonl(path))
    assert next(iterator) == {"id": "ok"}
    with pytest.raises(mod.ArxivAbstractsIntakeError, match="source JSONL is malformed"):
        next(iterator)


def test_report_is_text_free_zero_credit_evidence() -> None:
    accepted = [
        {
            "record_id": "1",
            "source_key": "arxiv_abstracts",
            "source_label": "arxiv-abstracts",
            "normalized_sha256": "a" * 64,
            "normalized_bytes": 321,
            "text": "not included in report",
        }
    ]
    payload = b'candidate bytes are not embedded in the report\n'
    report = mod.build_report(
        CONFIG,
        rights_registry_identity="b" * 64,
        accepted=accepted,
        reasons=Counter({"accepted": 1}),
        scanned=1,
        candidate_payload=payload,
    )
    encoded = json.dumps(report)
    assert "not included in report" not in encoded
    assert report["retained_records"] == 1
    assert report["retained_normalized_bytes"] == 321
    assert report["training_authorized_bytes"] == 0
    assert report["canonical_capacity_credited"] == 0
    assert report["family_credit_added"] == 0
    assert report["unique_causal_loss_positions_authorized"] == 0
    assert report["tokenizer_fit_authorized"] is False
    assert report["model_training_executed"] is False
    assert report["final_test_accessed"] is False
    assert report["paid_compute_used"] is False
    assert report["universal_pii_absence_claimed"] is False
