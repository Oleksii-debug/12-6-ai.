from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "materialize_d03_ua_nbu_pdftotext_v1.py"
CONFIG = ROOT / "configs" / "data" / "d03_ua_nbu_pdftotext_materialization_v1.json"

spec = importlib.util.spec_from_file_location("nbu_pdftotext", TOOL)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def load_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def with_identity(value: dict, field: str = "evidence_identity_sha256") -> dict:
    value.pop(field, None)
    value[field] = module.sha256((module.canonical_json(value) + "\n").encode("utf-8"))
    return value


def sample_inputs(config: dict):
    document_url = "https://bank.gov.ua/ua/legislation/Resolution_13012026_2"
    pdf_url = "https://bank.gov.ua/admin_uploads/law/13012026_2.pdf"
    pdf_bytes = b"%PDF-1.4\nsynthetic pinned bytes\n%%EOF\n"
    discovery = {
        "schema": "12-6.d03-ua-nbu-official-resolutions-discovery-evidence.v1",
        "status": "DISCOVERY_ONLY_ZERO_CREDIT",
        "contract_identity_sha256": config["base_authority"]["discovery_contract_identity_sha256"],
        "source_id": config["source"]["source_id"],
        "family_id": config["source"]["family_id"],
        "discovered_documents": 1,
        "documents": [{
            "document_url": document_url,
            "official_pdf_urls": [pdf_url],
            "page_metadata_sha256": "0" * 64,
        }],
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
    with_identity(discovery)
    pins = {
        "schema": "12-6.d03-ua-nbu-official-pdf-pins-evidence.v1",
        "status": "PDF_BYTES_PINNED_ZERO_CREDIT",
        "contract_identity_sha256": config["base_authority"]["parent_pdf_pin_contract_identity_sha256"],
        "parent_discovery_evidence_identity_sha256": discovery["evidence_identity_sha256"],
        "source_id": config["source"]["source_id"],
        "family_id": config["source"]["family_id"],
        "pinned_pdfs": 1,
        "observed_pdf_bytes": len(pdf_bytes),
        "pins": [{
            "pdf_url": pdf_url,
            "pdf_sha256": module.sha256(pdf_bytes),
            "pdf_bytes": len(pdf_bytes),
            "content_type": "application/pdf",
        }],
        "raw_pdf_bytes_emitted": False,
        "text_materialized": False,
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_used": False,
    }
    with_identity(pins)
    return discovery, pins, pdf_bytes


def test_current_parent_bindings_and_contract_identity_validate():
    config = load_config()
    module.validate_config(config)
    assert config["base_authority"]["parent_head_sha"] == "ea9436a84960e31282ccbddd7b3167be0e7c2362"
    assert config["base_authority"]["discovery_head_sha"] == "86a8a2a6f91eb4093bee414aaa8e8bbd3d895761"
    assert config["base_authority"]["parent_pdf_pin_contract_identity_sha256"] == "0532eea5d03014473a30cf78d19da5670530759d4cd4b15dda2c9fb7ac30912f"
    assert config["base_authority"]["discovery_contract_identity_sha256"] == "2e04335a2e6e665167b63f392d09756ae52e4f65f5253d9a037c48f5d191d9ec"
    assert config["claims"]["canonical_capacity_credit_bytes"] == 0
    assert config["claims"]["training_authorized_bytes"] == 0


def test_injected_materialization_is_deterministic_and_evidence_body_free():
    config = load_config()
    discovery, pins, pdf_bytes = sample_inputs(config)
    text = ("Постанова Національного банку України. Deterministic text.\n" * 2).encode("utf-8")

    def fetcher(url, expected_bytes, cfg):
        assert expected_bytes == len(pdf_bytes)
        return pdf_bytes

    def extractor(payload, cfg):
        assert payload == pdf_bytes
        return text

    first_artifact, first_evidence = module.materialize_text(
        discovery, pins, config,
        fetcher=fetcher,
        extractor=extractor,
        extractor_version="25.06.0",
    )
    second_artifact, second_evidence = module.materialize_text(
        discovery, pins, config,
        fetcher=fetcher,
        extractor=extractor,
        extractor_version="25.06.0",
    )
    assert first_artifact == second_artifact
    assert first_evidence == second_evidence
    assert first_evidence["raw_text_emitted_in_evidence"] is False
    assert first_evidence["text_artifact_is_training_authority"] is False
    assert first_evidence["canonical_capacity_credit_bytes"] == 0
    assert all("text" not in record for record in first_evidence["records"])


def test_wrong_extractor_version_fails_closed():
    config = load_config()
    discovery, pins, pdf_bytes = sample_inputs(config)
    with pytest.raises(module.NbuTextMaterializationError, match="extractor version mismatch"):
        module.materialize_text(
            discovery,
            pins,
            config,
            fetcher=lambda *_: pdf_bytes,
            extractor=lambda *_: b"valid extracted text over thirty-two bytes\n",
            extractor_version="25.03.0",
        )


def test_refetched_pdf_hash_mismatch_fails_closed():
    config = load_config()
    discovery, pins, pdf_bytes = sample_inputs(config)
    wrong = bytearray(pdf_bytes)
    wrong[8] ^= 1
    with pytest.raises(module.NbuTextMaterializationError, match="refetched PDF bytes do not match exact pin"):
        module.materialize_text(
            discovery,
            pins,
            config,
            fetcher=lambda *_: bytes(wrong),
            extractor=lambda *_: b"valid extracted text over thirty-two bytes\n",
            extractor_version="25.06.0",
        )


def test_cross_origin_source_and_evidence_promotion_fail_closed():
    config = load_config()
    discovery, pins, _ = sample_inputs(config)
    bad_source = deepcopy(discovery)
    bad_source["documents"][0]["document_url"] = "https://evil.example/ua/legislation/Resolution_13012026_2"
    with_identity(bad_source)
    with pytest.raises(module.NbuTextMaterializationError, match="cross-origin/non-https"):
        module.validate_discovery_evidence(bad_source, config)

    promoted = deepcopy(pins)
    promoted["training_authorized_bytes"] = 1
    with_identity(promoted)
    with pytest.raises(module.NbuTextMaterializationError, match="pin truth boundary drift"):
        module.validate_pin_evidence(promoted, config)



def test_duplicate_document_entry_fails_closed():
    config = load_config()
    discovery, _, _ = sample_inputs(config)
    discovery["documents"].append(deepcopy(discovery["documents"][0]))
    discovery["discovered_documents"] = 2
    with_identity(discovery)
    with pytest.raises(module.NbuTextMaterializationError, match="duplicate discovery document"):
        module.validate_discovery_evidence(discovery, config)


def test_one_pdf_cannot_be_owned_by_multiple_documents():
    config = load_config()
    discovery, _, _ = sample_inputs(config)
    shared_pdf = discovery["documents"][0]["official_pdf_urls"][0]
    discovery["documents"].append({
        "document_url": "https://bank.gov.ua/ua/legislation/Resolution_14012026_3",
        "official_pdf_urls": [shared_pdf],
        "page_metadata_sha256": "1" * 64,
    })
    discovery["discovered_documents"] = 2
    with_identity(discovery)
    with pytest.raises(module.NbuTextMaterializationError, match="one PDF is mapped to multiple documents"):
        module.validate_discovery_evidence(discovery, config)


def test_resealed_manifest_cannot_substitute_document_provenance():
    config = load_config()
    discovery, pins, pdf_bytes = sample_inputs(config)
    text = b"valid extracted text over thirty-two bytes for provenance identity\n"
    _, evidence = module.materialize_text(
        discovery,
        pins,
        config,
        fetcher=lambda *_: pdf_bytes,
        extractor=lambda *_: text,
        extractor_version="25.06.0",
    )
    tampered = deepcopy(evidence)
    tampered["records"][0]["document_url"] = (
        "https://bank.gov.ua/ua/legislation/Resolution_12012026_1"
    )
    with_identity(tampered)
    with pytest.raises(module.NbuTextMaterializationError, match="record provenance identity mismatch"):
        module.validate_materialization_evidence(tampered, config)


def test_resealed_discovery_related_pdf_fails_before_fetch():
    config = load_config()
    discovery, pins, pdf_bytes = sample_inputs(config)
    discovery["documents"][0]["official_pdf_urls"] = [
        "https://bank.gov.ua/admin_uploads/law/Resolution_24022022_18_kp.pdf"
    ]
    with_identity(discovery)
    pins["parent_discovery_evidence_identity_sha256"] = discovery["evidence_identity_sha256"]
    with_identity(pins)
    calls = 0

    def fetcher(*_args):
        nonlocal calls
        calls += 1
        return pdf_bytes

    with pytest.raises(module.NbuTextMaterializationError, match="primary PDF identity mismatch"):
        module.materialize_text(
            discovery,
            pins,
            config,
            fetcher=fetcher,
            extractor=lambda *_: b"valid extracted text over thirty-two bytes\n",
            extractor_version="25.06.0",
        )
    assert calls == 0


def test_duplicate_json_keys_fail_before_pdftotext_authority(tmp_path: Path):
    discovery_path = tmp_path / "duplicate-discovery.json"
    discovery_path.write_text(
        '{"documents":[],"documents":[{}]}',
        encoding="utf-8",
    )
    with pytest.raises(module.NbuTextMaterializationError, match="duplicate JSON key"):
        module.load_json_object(discovery_path, context="discovery evidence")

    pins_path = tmp_path / "duplicate-pins.json"
    pins_path.write_text(
        '{"pins":[],"pins":[{}]}',
        encoding="utf-8",
    )
    with pytest.raises(module.NbuTextMaterializationError, match="duplicate JSON key"):
        module.load_json_object(pins_path, context="PDF pin evidence")
