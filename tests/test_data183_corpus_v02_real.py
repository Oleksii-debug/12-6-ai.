import json
from pathlib import Path

import pytest

from twelve_six import data183_corpus_v02_real as d183
from twelve_six.checkpoint import sha256_file


def _write_fixture(root: Path, rows: list[dict]) -> dict:
    shard = root / "shards" / "part-00000.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "physical": {
            "shards": [
                {
                    "path": "shards/part-00000.jsonl",
                    "sha256": sha256_file(shard),
                }
            ]
        }
    }


def _row(record_id: str, text: str, split: str, stratum: str, origin: str, source: str) -> dict:
    return {
        "record_id": record_id,
        "text": text,
        "split": split,
        "stratum": stratum,
        "origin": origin,
        "source_id": source,
    }


def test_canonical_origin_contract_is_fail_closed():
    assert d183._canonical_origin("external_real") == "EXTERNAL_REAL"
    assert d183._canonical_origin("project_authored") == "PROJECT_AUTHORED"
    with pytest.raises(d183.Data183Error, match="unsupported origin"):
        d183._canonical_origin("synthetic_unknown")


def test_audit_reports_zero_normalized_cross_split_overlap_and_token_supply(tmp_path: Path):
    rows = [
        _row(
            "ua-train",
            "Український зовнішній документ для перевірки корпусу. " * 12,
            "train",
            "uk",
            "external_real",
            "external-ua-family",
        ),
        _row(
            "en-val",
            "Independent English external validation document with rights evidence. " * 12,
            "validation",
            "en",
            "external_real",
            "external-en-family",
        ),
        _row(
            "code-train",
            "def corpus_candidate(value):\n    return value + 1\n" * 20,
            "train",
            "code",
            "project_authored",
            "project-code-family",
        ),
    ]
    report = d183.audit_release(tmp_path, _write_fixture(tmp_path, rows))
    assert report["normalization_audit"]["train_validation_overlap"] == 0
    assert report["external_real_strata"] == ["en", "uk"]
    assert report["external_real_code_present"] is False
    assert report["project_authored_code_documents"] == 1
    supply = report["optimized_token_supply"]
    assert supply["total_target_tokens"] > 0
    assert supply["by_origin"]["EXTERNAL_REAL"] > 0
    assert supply["by_origin"]["PROJECT_AUTHORED"] > 0
    assert supply["by_source_family"]["external-ua-family"] > 0
    assert supply["by_source_family"]["project-code-family"] > 0


def test_audit_detects_overlap_after_independent_normalization(tmp_path: Path):
    train_text = "Café corpus overlap proof. " * 12
    validation_text = ("Cafe\u0301   corpus overlap proof.   " * 12).strip()
    rows = [
        _row("same-a", train_text, "train", "uk", "external_real", "ua"),
        _row("same-b", validation_text, "validation", "en", "external_real", "en"),
        _row(
            "code",
            "def unique_code(x):\n    return x * 2\n" * 20,
            "train",
            "code",
            "project_authored",
            "code",
        ),
    ]
    with pytest.raises(d183.Data183Error, match="normalized train-validation overlap"):
        d183.audit_release(tmp_path, _write_fixture(tmp_path, rows))


def test_duplicate_record_ids_fail_closed(tmp_path: Path):
    rows = [
        _row("dup", "Український текст. " * 30, "train", "uk", "external_real", "ua"),
        _row("dup", "English text. " * 30, "validation", "en", "external_real", "en"),
        _row(
            "code",
            "def f(x):\n    return x\n" * 30,
            "train",
            "code",
            "project_authored",
            "code",
        ),
    ]
    with pytest.raises(d183.Data183Error, match="duplicate record_id"):
        d183.audit_release(tmp_path, _write_fixture(tmp_path, rows))
