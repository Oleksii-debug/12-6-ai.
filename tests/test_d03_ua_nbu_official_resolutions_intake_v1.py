from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "validate_d03_ua_nbu_official_resolutions_intake_v1.py"
CONFIG = ROOT / "configs" / "data" / "d03_ua_nbu_official_resolutions_intake_v1.json"

spec = importlib.util.spec_from_file_location("nbu_intake", TOOL)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def load_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


CATALOG_A = """
<html><head><title>Нормативна база</title></head><body>
<a href="/ua/legislation/Resolution_13012026_2">r2</a>
<a href="https://bank.gov.ua/ua/legislation/Resolution_22012026_10">r10</a>
<a href="/ua/news/all/not-legislation">news</a>
<a href="https://evil.example/ua/legislation/Resolution_13012026_3">evil</a>
</body></html>
"""
CATALOG_B = CATALOG_A.replace("<body>", "<body><div>dynamic chrome</div>")

DOC_2_A = """
<html><head><title>Постанова Правління Національного банку України від 13.01.2026 № 2</title></head>
<body><a href="/admin_uploads/law/13012026_2.pdf?v=16">PDF</a></body></html>
"""
DOC_2_B = DOC_2_A.replace("<body>", "<body><div>transport chrome</div>")
DOC_10 = """
<html><head><title>Постанова Правління Національного банку України від 22.01.2026 № 10</title></head>
<body><a href="https://bank.gov.ua/admin_uploads/law/22012026_10.pdf">PDF</a></body></html>
"""


def test_contract_validates_and_remains_zero_credit():
    config = load_config()
    module.validate_config(config)
    assert config["claims"]["training_authorized_bytes"] == 0
    assert config["claims"]["authorized_unique_loss_positions"] == 0
    assert config["claims"]["paid_compute_authorized"] is False


def test_contract_identity_and_truth_boundary_fail_closed():
    config = load_config()
    mutated = deepcopy(config)
    mutated["claims"]["training_authorized_bytes"] = 1
    with pytest.raises(module.NbuIntakeError):
        module.validate_config(mutated)
    mutated = deepcopy(config)
    mutated["rights"]["blanket_site_rights_claimed"] = True
    with pytest.raises(module.NbuIntakeError):
        module.validate_config(mutated)


def test_catalog_discovers_only_same_origin_resolution_urls():
    config = load_config()
    assert module.discover_resolution_urls(CATALOG_A, config) == (
        "https://bank.gov.ua/ua/legislation/Resolution_13012026_2",
        "https://bank.gov.ua/ua/legislation/Resolution_22012026_10",
    )


def test_query_fragment_are_not_part_of_pdf_identity():
    config = load_config()
    assert module.canonical_pdf_url(
        "https://bank.gov.ua/admin_uploads/law/13012026_2.pdf?v=16#viewer",
        "https://bank.gov.ua/ua/legislation/Resolution_13012026_2",
        config,
    ) == "https://bank.gov.ua/admin_uploads/law/13012026_2.pdf"


def test_external_and_non_resolution_paths_fail_closed():
    config = load_config()
    for url in (
        "http://bank.gov.ua/ua/legislation/Resolution_13012026_2",
        "https://evil.example/ua/legislation/Resolution_13012026_2",
        "https://bank.gov.ua/ua/news/all/Resolution_13012026_2",
    ):
        with pytest.raises(module.NbuIntakeError):
            module.canonical_document_url(url, config["source"]["catalog_url"], config)


def test_resolution_page_requires_exact_scope_and_official_pdf():
    config = load_config()
    page = module.inspect_resolution_page(DOC_2_A, "https://bank.gov.ua/ua/legislation/Resolution_13012026_2", config)
    assert page.official_pdf_urls == ("https://bank.gov.ua/admin_uploads/law/13012026_2.pdf",)
    bad_title = DOC_2_A.replace("Постанова Правління Національного банку України", "Новина Національного банку України")
    with pytest.raises(module.NbuIntakeError):
        module.inspect_resolution_page(bad_title, "https://bank.gov.ua/ua/legislation/Resolution_13012026_2", config)
    external_pdf = DOC_2_A.replace("/admin_uploads/law/13012026_2.pdf?v=16", "https://cdn.example/13012026_2.pdf")
    with pytest.raises(module.NbuIntakeError):
        module.inspect_resolution_page(external_pdf, "https://bank.gov.ua/ua/legislation/Resolution_13012026_2", config)


def test_two_fetch_body_free_discovery_is_deterministic():
    config = load_config()
    fetches = {
        "https://bank.gov.ua/ua/legislation/Resolution_13012026_2": [DOC_2_A, DOC_2_B],
        "https://bank.gov.ua/ua/legislation/Resolution_22012026_10": [DOC_10, DOC_10],
    }
    evidence = module.build_body_free_discovery_evidence(config, [CATALOG_A, CATALOG_B], fetches)
    module.validate_evidence(evidence, config)
    assert evidence["discovered_documents"] == 2
    assert evidence["raw_text_emitted"] is False
    assert evidence["durable_payload_bytes_emitted"] == 0
    assert evidence["canonical_capacity_credit_bytes"] == 0


def test_catalog_instability_fails_closed():
    config = load_config()
    changed = CATALOG_A.replace('<a href="/ua/legislation/Resolution_13012026_2">r2</a>', "")
    with pytest.raises(module.NbuIntakeError):
        module.build_body_free_discovery_evidence(config, [CATALOG_A, changed], {})


def test_document_pdf_instability_fails_closed():
    config = load_config()
    catalog = '<html><body><a href="/ua/legislation/Resolution_13012026_2">r2</a></body></html>'
    changed_pdf = DOC_2_A.replace("13012026_2.pdf", "13012026_2_v2.pdf")
    with pytest.raises(module.NbuIntakeError):
        module.build_body_free_discovery_evidence(config, [catalog, catalog], {"https://bank.gov.ua/ua/legislation/Resolution_13012026_2": [DOC_2_A, changed_pdf]})


def test_evidence_promotion_and_tamper_fail_closed():
    config = load_config()
    catalog = '<html><body><a href="/ua/legislation/Resolution_13012026_2">r2</a></body></html>'
    evidence = module.build_body_free_discovery_evidence(config, [catalog, catalog], {"https://bank.gov.ua/ua/legislation/Resolution_13012026_2": [DOC_2_A, DOC_2_A]})
    promoted = deepcopy(evidence)
    promoted["training_authorized_bytes"] = 10
    with pytest.raises(module.NbuIntakeError):
        module.validate_evidence(promoted, config)
    tampered = deepcopy(evidence)
    tampered["documents"][0]["official_pdf_urls"][0] = "https://evil.example/a.pdf"
    with pytest.raises(module.NbuIntakeError):
        module.validate_evidence(tampered, config)
