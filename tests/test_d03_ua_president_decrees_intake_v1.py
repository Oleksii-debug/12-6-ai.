from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_tool(filename: str, module_name: str):
    path = ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


producer = load_tool(
    "materialize_d03_ua_president_decrees_v1.py",
    "d03_president_decrees_producer",
)
validator = load_tool(
    "validate_d03_ua_president_decrees_v1.py",
    "d03_president_decrees_validator",
)


def decree_html(subject="Про Концепцію розвитку державної політики", body=None):
    body = body or (
        "Український нормативний текст про державну політику та розвиток. " * 60
    )
    return f"""<html><body>
<nav><a href="/documents/9992026-1">УКАЗ ПРЕЗИДЕНТА УКРАЇНИ №999/2026</a></nav>
<h1>УКАЗ ПРЕЗИДЕНТА УКРАЇНИ №873/2026</h1>
<div>{subject}</div>
<article>
<p>{body}</p>
<p>Президент України В.ЗЕЛЕНСЬКИЙ</p>
<p>4 вересня 2026 року</p>
</article>
<footer><h3>Новини</h3><p>неофіційний текст</p></footer>
</body></html>""".encode()


def valid_evidence():
    record = {
        "url": "https://www.president.gov.ua/documents/8732026-61461",
        "document_number": "873/2026",
        "raw_sha256_a": "a" * 64,
        "raw_sha256_b": "b" * 64,
        "raw_byte_identical": False,
        "normalized_sha256": "c" * 64,
        "normalized_bytes": 2500,
        "accepted": True,
    }
    evidence = {
        "schema": validator.SCHEMA,
        "status": "PASS_OBSERVED_YIELD",
        "generated_at_utc": "2026-09-07T20:00:00+00:00",
        "source": {
            "source_id": validator.SOURCE_ID,
            "family_id": validator.SOURCE_ID,
            "stratum": "uk",
            "catalog_url": "https://www.president.gov.ua/documents/decrees/",
            "allowed_origin": "https://www.president.gov.ua",
            "rights_scope": "OFFICIAL_DECREE_TEXT_ONLY",
            "website_blanket_license_not_used_as_training_authority": True,
            "ukraine_copyright_law": {
                "law": "2811-IX",
                "article": "8(1)(3)",
                "authority_url": "https://zakon.rada.gov.ua/laws/show/2811-20",
                "scope": "official acts only",
            },
        },
        "execution": {
            "class": "LOCAL_FREE",
            "max_catalog_pages": 2,
            "max_documents": 40,
            "request_delay_seconds": 1.05,
            "catalog_pages_observed": 1,
            "documents_probed": 1,
            "double_fetch_required": True,
            "final_test_accessed": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
        },
        "catalog_pages": [
            {
                "url": "https://www.president.gov.ua/documents/decrees/",
                "page": 1,
                "document_count": 1,
                "document_url_set_sha256": "d" * 64,
            }
        ],
        "observed_yield": {
            "accepted_documents": 1,
            "accepted_normalized_bytes": 2500,
            "one_conservative_family": True,
            "rejected_documents": 0,
            "rejection_counts": {},
        },
        "records": [record],
        "downstream_required": [
            "GLOBAL_EXACT_NEAR_FRAGMENT_LINEAGE_DEDUP",
            "RESERVED_EVALUATION_DECONTAMINATION",
            "POST_COMPOSITION_QUALITY_PRIVACY_BALANCE_FAMILY_CAPS",
            "CLUSTER_SAFE_SPLIT",
            "DETERMINISTIC_TOKENIZER_PACKING_DOUBLE_BUILD",
            "POSITIVE_EXACT_UNIQUE_CAUSAL_LOSS_LEDGER",
        ],
        "claims": {
            "canonical_capacity_credit_bytes": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "tokenizer_fit_authorized": False,
            "research_corpus_released": False,
            "learned_20m_claim": False,
        },
    }
    materialization = json.dumps(
        validator.canonical_materialization_view(evidence),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    evidence["materialization_identity_sha256"] = validator.sha256(materialization)
    identity_view = dict(evidence)
    identity_view.pop("generated_at_utc")
    identity_view.pop("materialization_identity_sha256")
    evidence["evidence_identity_sha256"] = validator.sha256(
        json.dumps(
            identity_view,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return evidence


def test_extract_uses_last_title_and_stops_before_footer():
    title, subject, lines = producer.extract_document(decree_html())
    assert title == "УКАЗ ПРЕЗИДЕНТА УКРАЇНИ №873/2026"
    assert subject.startswith("Про Концепцію")
    text = "\n".join(lines)
    assert "неофіційний текст" not in text
    assert "Президент України В.ЗЕЛЕНСЬКИЙ" not in text


def test_document_normalization_is_stable():
    _, _, lines_a = producer.extract_document(decree_html())
    _, _, lines_b = producer.extract_document(decree_html())
    assert producer.normalize_document(lines_a) == producer.normalize_document(lines_b)


def test_catalog_discovers_only_decree_title_links():
    title = "УКАЗ ПРЕЗИДЕНТА УКРАЇНИ №873/2026"
    html = (
        '<a href="/documents/8732026-61461">' + title + "</a>"
        '<a href="https://evil.example/documents/1">' + title + "</a>"
        '<a href="/news/1">news</a>'
    ).encode()
    assert producer.discover_document_urls(html, producer.CATALOG) == [
        "https://www.president.gov.ua/documents/8732026-61461"
    ]


def test_pagination_chooses_nearest_forward_page():
    html = (
        b'<a href="/documents/decrees?date-from=x&date-to=y&page=5">5</a>'
        b'<a href="/documents/decrees?date-from=x&date-to=y&page=3">3</a>'
    )
    assert producer.discover_next_catalog_url(html, producer.CATALOG, 2).endswith("page=3")


@pytest.mark.parametrize(
    "subject",
    [
        "Про призначення судді",
        "Про звільнення посадової особи",
        "Про відзначення державними нагородами України",
        "Про застосування спеціальних економічних та інших обмежувальних заходів",
        "Про питання громадянства України",
    ],
)
def test_personal_subjects_are_quarantined(subject):
    reasons = producer.privacy_reasons(subject, "звичайний текст")
    assert "subject_personnel_or_personal_measure" in reasons


def test_contact_data_is_quarantined():
    text = "email test@example.org тел. +380 67 123 45 67"
    reasons = producer.privacy_reasons("Про Концепцію", text)
    assert "email" in reasons
    assert "phone" in reasons


def test_many_initialized_names_are_quarantined():
    text = "Іваненко І. І. Петренко П. П. Сидоренко С. С. Коваленко К. К."
    assert "high_person_name_density" in producer.privacy_reasons("Про Концепцію", text)


def test_quality_rejects_tiny_text():
    assert "below_minimum_bytes" in producer.quality_reasons("коротко".encode())


def test_quality_rejects_non_ukrainian_text():
    payload = ("English only text " * 200).encode()
    assert "ua_letter_ratio" in producer.quality_reasons(payload)


def test_quality_accepts_long_ukrainian_text():
    payload = (
        "Це довгий український нормативний текст державної політики. " * 100
    ).encode()
    assert producer.quality_reasons(payload) == []


def test_canonical_url_refuses_foreign_origin():
    assert producer.canonical_url("https://example.com/documents/1", producer.CATALOG) is None


def test_normalize_document_nfc_and_whitespace():
    data = producer.normalize_document(["  один   два  ", "три\u00a0чотири"])
    assert data.decode() == "один два\nтри чотири\n"


def test_valid_evidence_passes():
    validator.validate(valid_evidence())


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda evidence: evidence["claims"].__setitem__("training_authorized_bytes", 1),
            "training credit",
        ),
        (
            lambda evidence: evidence["records"][0].__setitem__("text", "forbidden"),
            "raw text",
        ),
        (
            lambda evidence: evidence["records"][0].__setitem__(
                "url", "https://example.com/documents/8732026-61461"
            ),
            "foreign origin",
        ),
        (
            lambda evidence: evidence["observed_yield"].__setitem__(
                "accepted_normalized_bytes", 2501
            ),
            "identity drift",
        ),
    ],
)
def test_validator_fails_closed(mutator, message):
    evidence = valid_evidence()
    mutator(evidence)
    with pytest.raises(ValueError, match=".+"):
        validator.validate(evidence)
