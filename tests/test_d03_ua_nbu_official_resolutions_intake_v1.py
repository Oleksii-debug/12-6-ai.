from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

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


def _catalog_page(*document_ids: str) -> str:
    links = "".join(
        f'<a href="/ua/legislation/Resolution_01012026_{value}">{value}</a>'
        for value in document_ids
    )
    return f"<html><body>{links}</body></html>"


def _requested_page(url: str) -> int:
    query = parse_qs(urlsplit(url).query, keep_blank_values=True)
    return int(query["page"][0])


def test_catalog_search_url_is_exact_and_rejects_bool_integer_aliases():
    config = load_config()
    url = module.catalog_search_url(config, page=2, per_page=40)
    parts = urlsplit(url)
    query = parse_qs(parts.query, keep_blank_values=True)
    assert parts.scheme == "https"
    assert parts.netloc == "bank.gov.ua"
    assert parts.path == "/ua/legislation/search"
    assert query["page"] == ["2"]
    assert query["perPage"] == ["40"]
    assert set(query) == {
        "from",
        "metaKeywords",
        "number",
        "page",
        "perPage",
        "publicationDate",
        "title",
        "to",
        "type",
    }
    with pytest.raises(module.NbuIntakeError, match="page must be an exact integer"):
        module.catalog_search_url(config, page=True, per_page=40)
    with pytest.raises(module.NbuIntakeError, match="per_page must be an exact integer"):
        module.catalog_search_url(config, page=1, per_page=False)


def test_bounded_catalog_crawl_progresses_without_pagination_anchors():
    config = load_config()
    pages = {
        1: _catalog_page("1", "2", "3"),
        2: _catalog_page("4", "5", "6"),
    }
    fetched: list[str] = []

    def fetch(url: str) -> tuple[str, str]:
        fetched.append(url)
        return pages[_requested_page(url)], url

    requested, documents = module.crawl_resolution_catalog(
        config,
        fetch,
        target_documents=5,
        per_page=40,
    )
    assert requested == tuple(fetched)
    assert [_requested_page(url) for url in requested] == [1, 2]
    assert len(documents) == 6
    assert documents == tuple(sorted(documents))


def test_catalog_crawl_rejects_response_query_substitution():
    config = load_config()

    def fetch(url: str) -> tuple[str, str]:
        final = url.replace("page=1", "page=2")
        return _catalog_page("1"), final

    with pytest.raises(module.NbuIntakeError, match="catalog response query value drift"):
        module.crawl_resolution_catalog(
            config,
            fetch,
            target_documents=1,
            per_page=40,
        )


def test_catalog_crawl_rejects_repeated_or_no_progress_page():
    config = load_config()

    def fetch(url: str) -> tuple[str, str]:
        page = _requested_page(url)
        body = _catalog_page("1") if page in {1, 2} else _catalog_page("2")
        return body, url

    with pytest.raises(module.NbuIntakeError, match="catalog page document tuple repeated"):
        module.crawl_resolution_catalog(
            config,
            fetch,
            target_documents=2,
            per_page=40,
        )


def test_catalog_response_rejects_cross_origin_and_extra_query_fields():
    config = load_config()
    expected = module.catalog_search_url(config, page=1, per_page=40)
    with pytest.raises(module.NbuIntakeError, match="escaped canonical origin/path"):
        module.validate_catalog_response_url(
            "https://evil.example/ua/legislation/search?page=1&perPage=40",
            expected,
            config,
        )
    with pytest.raises(module.NbuIntakeError, match="query field count drift"):
        module.validate_catalog_response_url(
            expected + "&unexpected=1",
            expected,
            config,
        )


OUT_OF_SCOPE = """
<html><head><title>Рішення Національного банку України</title></head>
<body><a href="/admin_uploads/law/13012026_2.pdf">PDF</a></body></html>
"""
NO_PDF = """
<html><head><title>Постанова Правління Національного банку України № 2</title></head>
<body><p>no attachment</p></body></html>
"""


def test_resolution_pair_deterministically_rejects_known_scope_misses():
    config = load_config()
    url = "https://bank.gov.ua/ua/legislation/Resolution_13012026_2"
    page, reason = module.inspect_resolution_page_pair(
        [OUT_OF_SCOPE, OUT_OF_SCOPE], url, config
    )
    assert page is None
    assert reason == "not a proven NBU Board resolution"

    page, reason = module.inspect_resolution_page_pair([NO_PDF, NO_PDF], url, config)
    assert page is None
    assert reason == "no official resolution PDF"


def test_resolution_pair_rejects_eligibility_drift_between_fetches():
    config = load_config()
    url = "https://bank.gov.ua/ua/legislation/Resolution_13012026_2"
    with pytest.raises(
        module.NbuIntakeError,
        match="unstable document eligibility/metadata",
    ):
        module.inspect_resolution_page_pair([DOC_2_A, OUT_OF_SCOPE], url, config)


def test_selected_document_builder_never_silently_drops_scope_miss():
    config = load_config()
    url = "https://bank.gov.ua/ua/legislation/Resolution_13012026_2"
    with pytest.raises(module.NbuIntakeError, match="selected document is outside scope"):
        module.build_body_free_discovery_evidence_from_documents(
            config,
            [url],
            {url: [OUT_OF_SCOPE, OUT_OF_SCOPE]},
        )


RELATED_ATTACHMENTS = """
<html><head><title>Постанова Правління Національного банку України від 13.01.2026 № 2</title></head>
<body>
<a href="/admin_uploads/law/13012026_2.pdf">primary</a>
<a href="/admin_uploads/law/Allres_13012026_2.pdf">consolidated</a>
<a href="/admin_uploads/law/Tabl_zmyny_13012026_2.pdf">table</a>
<a href="/admin_uploads/law/Resolution_24022022_18_kp.pdf">control copy</a>
<a href="/admin_uploads/law/Resolution_24022022_18_kp_eng.pdf">english control copy</a>
</body></html>
"""

ONLY_RELATED_ATTACHMENT = """
<html><head><title>Постанова Правління Національного банку України від 13.01.2026 № 2</title></head>
<body><a href="/admin_uploads/law/Resolution_24022022_18_kp.pdf">related control copy</a></body></html>
"""


def test_resolution_page_keeps_only_exact_primary_pdf():
    config = load_config()
    url = "https://bank.gov.ua/ua/legislation/Resolution_13012026_2"
    page = module.inspect_resolution_page(RELATED_ATTACHMENTS, url, config)
    assert page.official_pdf_urls == (
        "https://bank.gov.ua/admin_uploads/law/13012026_2.pdf",
    )


def test_related_attachment_without_exact_primary_is_deterministic_scope_reject():
    config = load_config()
    url = "https://bank.gov.ua/ua/legislation/Resolution_13012026_2"
    with pytest.raises(module.NbuIntakeError, match="no exact primary resolution PDF"):
        module.inspect_resolution_page(ONLY_RELATED_ATTACHMENT, url, config)
    page, reason = module.inspect_resolution_page_pair(
        [ONLY_RELATED_ATTACHMENT, ONLY_RELATED_ATTACHMENT],
        url,
        config,
    )
    assert page is None
    assert reason == "no exact primary resolution PDF"


def test_resealed_discovery_evidence_cannot_add_related_pdf():
    config = load_config()
    fetches = {
        "https://bank.gov.ua/ua/legislation/Resolution_13012026_2": [DOC_2_A, DOC_2_A],
    }
    catalog = '<html><body><a href="/ua/legislation/Resolution_13012026_2">r2</a></body></html>'
    evidence = module.build_body_free_discovery_evidence(
        config,
        [catalog, catalog],
        fetches,
    )
    tampered = deepcopy(evidence)
    tampered["documents"][0]["official_pdf_urls"].append(
        "https://bank.gov.ua/admin_uploads/law/Resolution_24022022_18_kp.pdf"
    )
    tampered["documents"][0]["official_pdf_urls"].sort()
    tampered["evidence_identity_sha256"] = module.self_identity(
        tampered,
        "evidence_identity_sha256",
    )
    with pytest.raises(module.NbuIntakeError, match="bad primary PDF set"):
        module.validate_evidence(tampered, config)


def test_resealed_discovery_evidence_cannot_substitute_related_pdf():
    config = load_config()
    fetches = {
        "https://bank.gov.ua/ua/legislation/Resolution_13012026_2": [DOC_2_A, DOC_2_A],
    }
    catalog = '<html><body><a href="/ua/legislation/Resolution_13012026_2">r2</a></body></html>'
    evidence = module.build_body_free_discovery_evidence(
        config,
        [catalog, catalog],
        fetches,
    )
    tampered = deepcopy(evidence)
    tampered["documents"][0]["official_pdf_urls"] = [
        "https://bank.gov.ua/admin_uploads/law/Resolution_24022022_18_kp.pdf"
    ]
    tampered["evidence_identity_sha256"] = module.self_identity(
        tampered,
        "evidence_identity_sha256",
    )
    with pytest.raises(module.NbuIntakeError, match="non-primary PDF"):
        module.validate_evidence(tampered, config)
