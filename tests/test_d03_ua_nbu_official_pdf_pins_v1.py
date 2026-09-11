from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/materialize_d03_ua_nbu_official_pdf_pins_v1.py"
CONFIG = ROOT / "configs/data/d03_ua_nbu_official_pdf_pin_v1.json"

spec = importlib.util.spec_from_file_location("nbu_pdf_pins", TOOL)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def load_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def make_pdf(tag: bytes, size: int = 2400) -> bytes:
    assert size >= 64
    middle = (tag + b"\n") * max(1, (size - 32) // (len(tag) + 1))
    body = b"%PDF-1.7\n" + middle
    body = body[: max(16, size - 8)]
    return body + b"\n%%EOF\n"


def discovery_evidence(config, pdf_urls):
    parent = module.PARENT.load_config()
    documents = []
    for index, pdf_url in enumerate(pdf_urls, start=1):
        url = f"https://bank.gov.ua/ua/legislation/Resolution_0{index}012026_{index}"
        meta = {
            "url": url,
            "title": f"Постанова Правління Національного банку України від 0{index}.01.2026 № {index}",
            "official_pdf_urls": [pdf_url],
        }
        documents.append(
            {
                "document_url": url,
                "official_pdf_urls": [pdf_url],
                "page_metadata_sha256": module.PARENT.sha256(
                    (module.PARENT.canonical_json(meta) + "\n").encode("utf-8")
                ),
            }
        )
    evidence = {
        "schema": "12-6.d03-ua-nbu-official-resolutions-discovery-evidence.v1",
        "status": "DISCOVERY_ONLY_ZERO_CREDIT",
        "contract_identity_sha256": parent["contract_identity_sha256"],
        "source_id": parent["source"]["source_id"],
        "family_id": parent["source"]["family_id"],
        "discovered_documents": len(documents),
        "documents": documents,
        "raw_text_emitted": False,
        "durable_payload_bytes_emitted": 0,
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_used": False,
    }
    evidence["evidence_identity_sha256"] = module.PARENT.sha256(
        (module.PARENT.canonical_json(evidence) + "\n").encode("utf-8")
    )
    module.PARENT.validate_evidence(evidence, parent)
    assert evidence["contract_identity_sha256"] == config["base_authority"]["parent_contract_identity_sha256"]
    return evidence


def stable_fetcher(payloads):
    def fetch(url, config):
        body, content_type, final_url = payloads[url]
        return module.FetchedPdf(body=body, content_type=content_type, final_url=final_url)
    return fetch


def test_contract_validates_and_has_zero_credit():
    config = load_config()
    module.validate_config(config)
    assert config["claims"]["canonical_capacity_credit_bytes"] == 0
    assert config["claims"]["training_authorized_bytes"] == 0
    assert config["claims"]["text_materialized"] is False


def test_contract_promotion_fails_closed():
    config = load_config()
    promoted = deepcopy(config)
    promoted["claims"]["training_authorized_bytes"] = 1
    with pytest.raises(module.NbuPdfPinError):
        module.validate_config(promoted)


def test_stable_two_pdf_pins_are_body_free_and_deterministic():
    config = load_config()
    urls = [
        "https://bank.gov.ua/admin_uploads/law/01012026_1.pdf",
        "https://bank.gov.ua/admin_uploads/law/02012026_2.pdf",
    ]
    discovery = discovery_evidence(config, urls)
    payloads = {
        urls[0]: (make_pdf(b"alpha"), "application/pdf", urls[0]),
        urls[1]: (make_pdf(b"beta"), "application/octet-stream", urls[1]),
    }
    evidence = module.materialize_pdf_pins(discovery, config, stable_fetcher(payloads))
    module.validate_pin_evidence(evidence, config)
    assert evidence["pinned_pdfs"] == 2
    assert evidence["observed_pdf_bytes"] == sum(len(v[0]) for v in payloads.values())
    assert evidence["raw_pdf_bytes_emitted"] is False
    assert evidence["text_materialized"] is False
    assert evidence["training_authorized_bytes"] == 0
    assert all(set(pin) == {"pdf_url", "pdf_sha256", "pdf_bytes", "content_type"} for pin in evidence["pins"])


def test_second_fetch_byte_mutation_fails_closed():
    config = load_config()
    url = "https://bank.gov.ua/admin_uploads/law/01012026_1.pdf"
    discovery = discovery_evidence(config, [url])
    calls = 0

    def fetcher(requested, _config):
        nonlocal calls
        calls += 1
        body = make_pdf(b"first") if calls == 1 else make_pdf(b"second")
        return module.FetchedPdf(body=body, content_type="application/pdf", final_url=requested)

    with pytest.raises(module.NbuPdfPinError, match="independent PDF fetches differ"):
        module.materialize_pdf_pins(discovery, config, fetcher)


def test_malformed_pdf_markers_fail_closed():
    config = load_config()
    url = "https://bank.gov.ua/admin_uploads/law/01012026_1.pdf"
    discovery = discovery_evidence(config, [url])
    bad = b"NOT-A-PDF" + b"x" * 3000
    payloads = {url: (bad, "application/pdf", url)}
    with pytest.raises(module.NbuPdfPinError, match="PDF header marker missing"):
        module.materialize_pdf_pins(discovery, config, stable_fetcher(payloads))


def test_changed_final_url_fails_closed():
    config = load_config()
    url = "https://bank.gov.ua/admin_uploads/law/01012026_1.pdf"
    discovery = discovery_evidence(config, [url])
    payloads = {
        url: (
            make_pdf(b"same"),
            "application/pdf",
            "https://bank.gov.ua/admin_uploads/law/01012026_other.pdf",
        )
    }
    with pytest.raises(module.NbuPdfPinError, match="final PDF identity changed"):
        module.materialize_pdf_pins(discovery, config, stable_fetcher(payloads))


def test_wrong_content_type_fails_closed():
    config = load_config()
    url = "https://bank.gov.ua/admin_uploads/law/01012026_1.pdf"
    discovery = discovery_evidence(config, [url])
    payloads = {url: (make_pdf(b"same"), "text/html", url)}
    with pytest.raises(module.NbuPdfPinError, match="unexpected PDF content type"):
        module.materialize_pdf_pins(discovery, config, stable_fetcher(payloads))


def test_total_byte_bound_fails_closed():
    config = load_config()
    config = deepcopy(config)
    config["execution"]["hard_max_total_pdf_bytes"] = 4096
    config["contract_identity_sha256"] = module.self_identity(config, "contract_identity_sha256")
    module.validate_config(config)
    urls = [
        "https://bank.gov.ua/admin_uploads/law/01012026_1.pdf",
        "https://bank.gov.ua/admin_uploads/law/02012026_2.pdf",
    ]
    discovery = discovery_evidence(config, urls)
    payloads = {
        urls[0]: (make_pdf(b"alpha", 2400), "application/pdf", urls[0]),
        urls[1]: (make_pdf(b"beta", 2400), "application/pdf", urls[1]),
    }
    with pytest.raises(module.NbuPdfPinError, match="total PDF byte bound exceeded"):
        module.materialize_pdf_pins(discovery, config, stable_fetcher(payloads))


def test_parent_discovery_tamper_fails_closed():
    config = load_config()
    url = "https://bank.gov.ua/admin_uploads/law/01012026_1.pdf"
    discovery = discovery_evidence(config, [url])
    discovery = deepcopy(discovery)
    discovery["documents"][0]["official_pdf_urls"][0] = "https://evil.example/a.pdf"
    payloads = {url: (make_pdf(b"same"), "application/pdf", url)}
    with pytest.raises(module.NbuPdfPinError, match="parent discovery evidence invalid"):
        module.materialize_pdf_pins(discovery, config, stable_fetcher(payloads))


def test_pin_evidence_promotion_and_digest_tamper_fail_closed():
    config = load_config()
    url = "https://bank.gov.ua/admin_uploads/law/01012026_1.pdf"
    discovery = discovery_evidence(config, [url])
    payloads = {url: (make_pdf(b"same"), "application/pdf", url)}
    evidence = module.materialize_pdf_pins(discovery, config, stable_fetcher(payloads))

    promoted = deepcopy(evidence)
    promoted["training_authorized_bytes"] = 1
    with pytest.raises(module.NbuPdfPinError):
        module.validate_pin_evidence(promoted, config)

    tampered = deepcopy(evidence)
    tampered["pins"][0]["pdf_sha256"] = "0" * 64
    with pytest.raises(module.NbuPdfPinError):
        module.validate_pin_evidence(tampered, config)
