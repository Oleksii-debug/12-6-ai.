from __future__ import annotations

import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/materialize_d03_common_pile_loc_books.py"
CONFIG_PATH = ROOT / "configs/data/d03_common_pile_loc_books_bounded_v1.json"
REGISTRY_PATH = ROOT / "configs/data/common_pile_source_rights_v1.json"

SPEC = importlib.util.spec_from_file_location("d03_common_pile_loc_books", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_canonical_config_and_merged_rights_row_validate() -> None:
    cfg = config()
    MODULE.validate_config(cfg)
    row = MODULE.load_rights_row(cfg, REGISTRY_PATH)
    assert row["key"] == "library_of_congress"
    assert row["rights_basis_class"] == "PUBLIC_DOMAIN_COLLECTION"
    assert row["canonical_training_authorized"] is False
    assert row["credited_bytes"] == 0
    assert row["final_test_excluded"] is True


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("rights_authority", "required_rights_basis_class", "SITEWIDE_OPEN_LICENSE", "rights"),
        ("source", "sha256", "0" * 64, "source authority"),
        ("source", "expected_metadata_license", "CC-BY-4.0", "source authority"),
        ("record_contract", "require_unique_source_ids", False, "uniqueness"),
        ("selection", "min_latin_share_of_alpha", 0.0, "selection policy"),
        ("privacy_quality", "reject_email_like", False, "privacy/quality"),
        ("claim_boundary", "training_authorized_bytes", 1, "claim boundary"),
        ("claim_boundary", "corpus_admitted", True, "claim boundary"),
    ],
)
def test_contract_mutations_fail_closed(
    section: str, key: str, value: object, message: str
) -> None:
    cfg = config()
    cfg[section][key] = value
    with pytest.raises(MODULE.MaterializationError, match=message):
        MODULE.validate_config(cfg)


def good_row(source_id: str, text: str) -> dict[str, object]:
    return {
        "id": source_id,
        "text": text,
        "source": "loc_books",
        "added": "2024-05-14T16:22:35.184395",
        "metadata": {
            "license": "Public Domain",
            "title": "Synthetic historical book",
            "author": "Example Author",
            "year": 1901,
            "language": "english",
            "item_url": "https://www.loc.gov/item/example",
            "text_file_url": "https://tiles.loc.gov/example.txt",
        },
    }


def test_row_contract_quarantines_rights_language_year_and_schema_drift() -> None:
    cfg = config()
    base = good_row("abc", "English prose " * 300)
    validated = MODULE.validate_row(base, cfg)
    assert isinstance(validated, tuple)

    bad = copy.deepcopy(base)
    bad["metadata"]["license"] = "Unknown"
    assert MODULE.validate_row(bad, cfg) == "rights_metadata_mismatch"

    bad = copy.deepcopy(base)
    bad["metadata"]["language"] = "french"
    assert MODULE.validate_row(bad, cfg) == "language_metadata_mismatch"

    bad = copy.deepcopy(base)
    bad["metadata"]["year"] = 1499
    assert MODULE.validate_row(bad, cfg) == "year_metadata_mismatch"

    bad = copy.deepcopy(base)
    bad["unexpected"] = True
    assert MODULE.validate_row(bad, cfg) == "row_schema_mismatch"


def test_classifier_rejects_privacy_and_non_english_payloads() -> None:
    cfg = config()
    cfg["selection"].update(
        {
            "min_text_utf8_bytes": 10,
            "max_text_utf8_bytes": 100000,
            "min_alphabetic_chars": 5,
            "min_alpha_fraction": 0.2,
            "min_latin_share_of_alpha": 0.8,
        }
    )
    assert MODULE.classify_text("A sound English historical paragraph." * 20, cfg) is None
    assert MODULE.classify_text("Contact archive@example.org for details." * 20, cfg) == "email_like"
    assert MODULE.classify_text("Call 202-555-0147 for details." * 30, cfg) == "phone_like"
    assert MODULE.classify_text("English text\x01 with control." * 30, cfg) == "control_character"
    assert MODULE.classify_text("English replacement \ufffd text." * 30, cfg) == "replacement_character"
    assert MODULE.classify_text("Український історичний текст. " * 30, cfg) == "low_latin_share"


def small_stream_config() -> dict[str, object]:
    cfg = config()
    cfg["selection"].update(
        {
            "max_scanned_decompressed_bytes": 200000,
            "min_candidate_normalized_utf8_bytes": 5000,
            "max_candidate_normalized_utf8_bytes": 9000,
            "min_text_utf8_bytes": 1000,
            "max_text_utf8_bytes": 5000,
            "min_alphabetic_chars": 500,
            "min_alpha_fraction": 0.4,
            "min_latin_share_of_alpha": 0.9,
        }
    )
    return cfg


def gzip_rows(rows: list[dict[str, object]]) -> io.BytesIO:
    raw = b"".join(MODULE.canonical_bytes(row) for row in rows)
    return io.BytesIO(gzip.compress(raw, mtime=0))


def test_stream_is_deterministic_zero_metadata_payload_and_exact_dedup() -> None:
    cfg = small_stream_config()
    text_a = ("This is clean public domain historical English prose for testing. " * 45).strip()
    text_b = ("Another distinct public domain English passage from an old volume. " * 45).strip()
    rows = [good_row("a", text_a), good_row("b", text_a), good_row("c", text_b)]

    outputs = []
    stats_runs = []
    for _ in range(2):
        output = io.BytesIO()
        stats = MODULE.materialize_stream(gzip_rows(rows), cfg, output)
        outputs.append(output.getvalue())
        stats_runs.append(stats)

    assert outputs[0] == outputs[1]
    assert stats_runs[0] == stats_runs[1]
    assert stats_runs[0]["accepted_records"] == 2
    assert stats_runs[0]["rejected_by_reason"]["exact_normalized_duplicate"] == 1
    payload = outputs[0].decode("utf-8")
    assert "Example Author" not in payload
    assert "loc.gov" not in payload
    assert '"source_family":"en.common-pile.library-of-congress"' in payload


def test_duplicate_source_id_fails_closed() -> None:
    cfg = small_stream_config()
    text_a = ("First clean historical English book passage. " * 70).strip()
    text_b = ("Second clean historical English book passage. " * 70).strip()
    rows = [good_row("same", text_a), good_row("same", text_b)]
    with pytest.raises(MODULE.MaterializationError, match="duplicate source id"):
        MODULE.materialize_stream(gzip_rows(rows), cfg, io.BytesIO())


def test_bounded_scan_fails_if_minimum_candidate_yield_is_not_met() -> None:
    cfg = small_stream_config()
    cfg["selection"]["min_candidate_normalized_utf8_bytes"] = 8000
    one = good_row("one", ("Only one acceptable passage is present. " * 50).strip())
    with pytest.raises(MODULE.MaterializationError, match="minimum not met"):
        MODULE.materialize_stream(gzip_rows([one]), cfg, io.BytesIO())


def test_exact_source_verification_is_byte_bound(tmp_path: Path) -> None:
    cfg = config()
    path = tmp_path / "source.gz"
    path.write_bytes(b"exact synthetic source")
    cfg["source"]["bytes"] = path.stat().st_size
    cfg["source"]["sha256"] = MODULE.sha256_bytes(path.read_bytes())
    digest, size = MODULE.verify_exact_source(cfg, path)
    assert digest == cfg["source"]["sha256"]
    assert size == cfg["source"]["bytes"]
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(MODULE.MaterializationError, match="byte-count mismatch"):
        MODULE.verify_exact_source(cfg, path)


def test_report_binds_zero_credit_and_contains_no_candidate_text() -> None:
    cfg = config()
    rights = MODULE.load_rights_row(cfg, REGISTRY_PATH)
    stats = {
        "rows_seen": 10,
        "scanned_decompressed_bytes": 6000000,
        "accepted_records": 3,
        "accepted_normalized_utf8_bytes": 5500000,
        "rejected_records": 7,
        "rejected_by_reason": {"email_like": 1},
        "candidate_jsonl_sha256": "a" * 64,
        "inventory_identity_sha256": "b" * 64,
    }
    report = MODULE.build_report(cfg, rights, cfg["source"]["sha256"], cfg["source"]["bytes"], stats)
    assert report["decision"] == "LOC_BOOKS_CANDIDATE_ONLY_ZERO_CREDIT"
    assert report["claim_boundary"]["training_authorized_bytes"] == 0
    assert report["claim_boundary"]["corpus_admitted"] is False
    assert report["claim_boundary"]["model_training_executed"] is False
    encoded = json.dumps(report, ensure_ascii=False)
    assert "Synthetic historical book" not in encoded
    assert "Example Author" not in encoded
