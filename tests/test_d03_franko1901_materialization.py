from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "materialize_d03_franko1901.py"
CONFIG_PATH = ROOT / "configs" / "data" / "d03_franko1901_exact_materialization_v1.json"

SPEC = importlib.util.spec_from_file_location("d03_franko1901", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def canonical_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_canonical_contract_validates() -> None:
    MODULE.validate_contract(canonical_config())


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        (("source", "source_git_blob_sha1", "0" * 40), "source authority drift"),
        (("rights_boundary", "modern_text_allowed", True), "LLM/enrichment fields"),
        (("truth_boundary", "training_authorized_bytes", 1), "truth boundary drift"),
        (("truth_boundary", "corpus_admitted", True), "truth boundary drift"),
        (("acquisition", "fetch_count_required", 1), "acquisition contract drift"),
        (("filter", "min_cyrillic_share_of_alpha", 0.0), "filter policy drift"),
    ],
)
def test_contract_mutations_fail_closed(
    mutation: tuple[str, str, object],
    expected_message: str,
) -> None:
    config = canonical_config()
    section, key, value = mutation
    config[section][key] = value
    with pytest.raises(RuntimeError, match=expected_message):
        MODULE.validate_contract(config)


def test_text_classifier_rejects_non_uk_and_payload_anomalies() -> None:
    policy = canonical_config()["filter"]
    assert MODULE.classify_text("Добра рада краща за золото.", policy) is None
    assert MODULE.classify_text("only English words here", policy) == "low_cyrillic_share"
    assert MODULE.classify_text(
        "Українська приповідка з довгим текстом і посиланням https://x.io",
        policy,
    ) == "url_like"
    assert MODULE.classify_text(
        "Українська приповідка має адресу пошти хтось@example.com у тексті",
        policy,
    ) == "email_like"
    assert MODULE.classify_text("Добрий\x01день український", policy) == "control_character"
    assert MODULE.classify_text("Український \ufffd текст", policy) == "replacement_character"


def test_parse_filter_is_deterministic_and_excludes_metadata() -> None:
    config = canonical_config()
    raw = (
        "prov_clean,term,letter,description\n"
        '"Добра рада краща за золото.","рада","Р","scholarly note"\n'
        '"Добра рада краща за золото.","дубль","Р","duplicate note"\n'
        '"Без праці нема добра.","праця","П","other note"\n'
        '"only English words here","","E","noise"\n'
    ).encode()

    first_jsonl, first_stats = MODULE.parse_and_filter(config, raw)
    second_jsonl, second_stats = MODULE.parse_and_filter(config, raw)

    assert first_jsonl == second_jsonl
    assert first_stats == second_stats
    assert first_stats["rows_seen"] == 4
    assert first_stats["accepted_rows"] == 2
    assert first_stats["rejected_by_reason"] == {
        "exact_duplicate": 1,
        "low_cyrillic_share": 1,
    }
    payload = first_jsonl.decode()
    assert "scholarly note" not in payload
    assert "duplicate note" not in payload
    assert '"text":"Добра рада краща за золото."' in payload
    assert '"text":"Без праці нема добра."' in payload


def test_normalization_preserves_historical_spelling() -> None:
    text = "  вбувсь, охвіра, робицця  "
    assert MODULE.normalize_source_text(text) == "вбувсь, охвіра, робицця"


def test_git_blob_identity_is_content_bound() -> None:
    assert MODULE.git_blob_sha1(b"test\n") == "9daeafb9864cf43055ae93beb0afd6c7d144bfa4"
    assert MODULE.git_blob_sha1(b"test\n") != MODULE.git_blob_sha1(b"test")


def test_report_keeps_zero_credit_boundary() -> None:
    config = canonical_config()
    raw = b"small synthetic bytes"
    stats = {
        "rows_seen": 1,
        "accepted_rows": 1,
        "rejected_rows": 0,
        "rejected_by_reason": {},
        "accepted_text_utf8_bytes": 10,
        "accepted_payload_jsonl_bytes": 20,
        "accepted_payload_jsonl_sha256": "a" * 64,
        "record_inventory_identity_sha256": "b" * 64,
        "aggregate_alphabetic_chars": 10,
        "aggregate_cyrillic_chars": 10,
        "aggregate_cyrillic_share_of_alpha": 1.0,
    }
    report = MODULE.build_report(config, raw, stats)
    truth = report["truth_boundary"]
    assert report["decision"] == "CANDIDATE_MATERIALIZED_ZERO_CREDIT"
    assert truth["canonical_capacity_credit_bytes"] == 0
    assert truth["training_authorized_bytes"] == 0
    assert truth["corpus_admitted"] is False
    assert truth["model_training_executed"] is False

    tampered = copy.deepcopy(report)
    tampered["truth_boundary"]["training_authorized_bytes"] = 1
    assert tampered != report
