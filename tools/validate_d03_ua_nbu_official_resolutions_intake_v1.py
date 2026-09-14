from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/data/d03_ua_nbu_official_resolutions_intake_v1.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class NbuIntakeError(ValueError):
    pass


@dataclass(frozen=True)
class ResolutionPage:
    url: str
    title: str
    official_pdf_urls: tuple[str, ...]


class Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []
        self.title: list[str] = []
        self.in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() == "title":
            self.in_title = True
        if tag.casefold() == "a":
            self.hrefs.extend(v for k, v in attrs if k.casefold() == "href" and v)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title and (value := " ".join(data.split())):
            self.title.append(value)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def self_identity(value: Mapping[str, Any], field: str) -> str:
    clone = deepcopy(dict(value))
    clone.pop(field, None)
    return sha256((canonical_json(clone) + "\n").encode())


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "12-6.d03-ua-nbu-official-resolutions-intake.v1":
        raise NbuIntakeError("schema drift")
    if config.get("status") != "PREPARED_DISCOVERY_ZERO_CREDIT":
        raise NbuIntakeError("status drift")
    base = config.get("base_authority")
    if not isinstance(base, Mapping) or base.get("repository") != "Oleksii-debug/12-6-ai.":
        raise NbuIntakeError("base authority drift")
    if not isinstance(base.get("git_sha"), str) or not HEX40.fullmatch(base["git_sha"]):
        raise NbuIntakeError("bad base sha")
    if base.get("lane") != "D03" or base.get("control_issue") != 548:
        raise NbuIntakeError("control binding drift")

    source = config.get("source")
    required_source = {
        "source_id": "ua.nbu.official-resolutions",
        "family_id": "ua.nbu.official-resolutions",
        "stratum": "uk",
        "catalog_url": "https://bank.gov.ua/ua/legislation",
        "allowed_origin": "https://bank.gov.ua",
        "required_title_prefix": "Постанова Правління Національного банку України",
        "one_conservative_family": True,
    }
    if not isinstance(source, Mapping) or any(source.get(k) != v for k, v in required_source.items()):
        raise NbuIntakeError("source authority drift")
    for key in ("document_path_regex", "official_pdf_path_regex"):
        try:
            re.compile(str(source[key]))
        except (KeyError, re.error) as exc:
            raise NbuIntakeError(f"bad {key}") from exc

    rights = config.get("rights")
    if not isinstance(rights, Mapping):
        raise NbuIntakeError("rights missing")
    if rights.get("decision") != "OFFICIAL_NBU_RESOLUTION_TEXT_ONLY_CANDIDATE":
        raise NbuIntakeError("rights decision drift")
    if rights.get("article") != "8(1)(3)" or rights.get("blanket_site_rights_claimed") is not False:
        raise NbuIntakeError("rights boundary weakened")
    if rights.get("legal_conclusion_claimed") is not False or len(rights.get("exclude", [])) < 5:
        raise NbuIntakeError("rights boundary weakened")

    discovery = config.get("discovery_contract")
    required_discovery = {
        "class": "LOCAL_FREE",
        "catalog_fetches_required": 2,
        "document_page_fetches_required": 2,
        "same_origin_https_only": True,
        "query_fragment_forbidden_for_canonical_identity": True,
        "stable_document_set_required": True,
        "stable_official_pdf_set_required": True,
        "durable_evidence_body_free": True,
    }
    if not isinstance(discovery, Mapping) or any(discovery.get(k) != v for k, v in required_discovery.items()):
        raise NbuIntakeError("discovery contract drift")
    if not isinstance(discovery.get("hard_max_documents"), int) or not 1 <= discovery["hard_max_documents"] <= 250:
        raise NbuIntakeError("unsafe hard bound")

    claims = config.get("claims")
    required_claims = {
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_authorized": False,
        "learned_20m_claim": False,
    }
    if not isinstance(claims, Mapping) or any(claims.get(k) != v for k, v in required_claims.items()):
        raise NbuIntakeError("truth boundary drift")
    ident = config.get("contract_identity_sha256")
    if not isinstance(ident, str) or not HEX64.fullmatch(ident) or self_identity(config, "contract_identity_sha256") != ident:
        raise NbuIntakeError("contract identity mismatch")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise NbuIntakeError("config root")
    validate_config(config)
    return config


def _url(raw: str, base: str, config: Mapping[str, Any], kind: str) -> str:
    parts = urlsplit(urljoin(base, html.unescape(raw.strip())))
    allowed = urlsplit(config["source"]["allowed_origin"])
    if parts.scheme != "https" or parts.netloc.casefold() != allowed.netloc.casefold():
        raise NbuIntakeError("cross-origin/non-https URL")
    if parts.username or parts.password or parts.port not in (None, 443):
        raise NbuIntakeError("non-canonical authority")
    parts = parts._replace(query="", fragment="")
    key = "document_path_regex" if kind == "document" else "official_pdf_path_regex"
    if not re.fullmatch(config["source"][key], parts.path):
        raise NbuIntakeError("path outside allowlist")
    return urlunsplit(("https", parts.netloc.casefold(), parts.path, "", ""))


def canonical_document_url(raw: str, base: str, config: Mapping[str, Any]) -> str:
    return _url(raw, base, config, "document")


def canonical_pdf_url(raw: str, base: str, config: Mapping[str, Any]) -> str:
    return _url(raw, base, config, "pdf")


def parse(raw_html: str) -> Parser:
    if not raw_html:
        raise NbuIntakeError("empty HTML")
    parser = Parser()
    parser.feed(raw_html)
    parser.close()
    return parser


def discover_resolution_urls(catalog_html: str, config: Mapping[str, Any]) -> tuple[str, ...]:
    validate_config(config)
    found = set()
    for href in parse(catalog_html).hrefs:
        try:
            found.add(canonical_document_url(href, config["source"]["catalog_url"], config))
        except NbuIntakeError:
            pass
    return tuple(sorted(found))


def inspect_resolution_page(document_html: str, document_url: str, config: Mapping[str, Any]) -> ResolutionPage:
    validate_config(config)
    url = canonical_document_url(document_url, config["source"]["catalog_url"], config)
    parser = parse(document_html)
    title = " ".join(parser.title)
    if not title.startswith(config["source"]["required_title_prefix"]):
        raise NbuIntakeError("not a proven NBU Board resolution")
    pdfs = set()
    for href in parser.hrefs:
        try:
            pdfs.add(canonical_pdf_url(href, url, config))
        except NbuIntakeError:
            pass
    if not pdfs:
        raise NbuIntakeError("no official resolution PDF")
    return ResolutionPage(url, title, tuple(sorted(pdfs)))


def build_body_free_discovery_evidence(config: Mapping[str, Any], catalog_fetches: Sequence[str], document_fetches: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    validate_config(config)
    if len(catalog_fetches) != 2:
        raise NbuIntakeError("catalog fetch count")
    discoveries = [discover_resolution_urls(item, config) for item in catalog_fetches]
    if discoveries[0] != discoveries[1] or not discoveries[0]:
        raise NbuIntakeError("unstable/empty discovery")
    urls = discoveries[0]
    if len(urls) > config["discovery_contract"]["hard_max_documents"]:
        raise NbuIntakeError("hard bound exceeded")
    documents = []
    for url in urls:
        payloads = document_fetches.get(url)
        if payloads is None or len(payloads) != 2:
            raise NbuIntakeError("document fetch count")
        pages = [inspect_resolution_page(item, url, config) for item in payloads]
        if pages[0] != pages[1]:
            raise NbuIntakeError("unstable document metadata")
        page = pages[0]
        meta = {"url": page.url, "title": page.title, "official_pdf_urls": list(page.official_pdf_urls)}
        documents.append({"document_url": page.url, "official_pdf_urls": list(page.official_pdf_urls), "page_metadata_sha256": sha256((canonical_json(meta) + "\n").encode())})
    evidence = {
        "schema": "12-6.d03-ua-nbu-official-resolutions-discovery-evidence.v1",
        "status": "DISCOVERY_ONLY_ZERO_CREDIT",
        "contract_identity_sha256": config["contract_identity_sha256"],
        "source_id": config["source"]["source_id"],
        "family_id": config["source"]["family_id"],
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
    evidence["evidence_identity_sha256"] = sha256((canonical_json(evidence) + "\n").encode())
    return evidence


def validate_evidence(evidence: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    validate_config(config)
    if evidence.get("schema") != "12-6.d03-ua-nbu-official-resolutions-discovery-evidence.v1" or evidence.get("status") != "DISCOVERY_ONLY_ZERO_CREDIT":
        raise NbuIntakeError("evidence authority drift")
    if evidence.get("contract_identity_sha256") != config["contract_identity_sha256"]:
        raise NbuIntakeError("contract binding")
    documents = evidence.get("documents")
    if not isinstance(documents, list) or not documents or evidence.get("discovered_documents") != len(documents):
        raise NbuIntakeError("document evidence")
    seen = set()
    for item in documents:
        url = canonical_document_url(str(item.get("document_url", "")), config["source"]["catalog_url"], config)
        if url in seen:
            raise NbuIntakeError("duplicate document")
        seen.add(url)
        pdfs = item.get("official_pdf_urls")
        if not isinstance(pdfs, list) or not pdfs or sorted(set(pdfs)) != pdfs:
            raise NbuIntakeError("bad PDF set")
        for pdf in pdfs:
            canonical_pdf_url(str(pdf), url, config)
        if not HEX64.fullmatch(str(item.get("page_metadata_sha256", ""))):
            raise NbuIntakeError("bad metadata hash")
    required = {
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
    if any(evidence.get(k) != v for k, v in required.items()):
        raise NbuIntakeError("evidence promotion")
    ident = evidence.get("evidence_identity_sha256")
    if not isinstance(ident, str) or self_identity(evidence, "evidence_identity_sha256") != ident:
        raise NbuIntakeError("evidence identity")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--evidence", type=Path)
    args = ap.parse_args()
    config = load_config(args.config)
    if args.evidence:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        validate_evidence(evidence, config)
        print(f"PASS evidence_identity_sha256={evidence['evidence_identity_sha256']}")
    else:
        print(f"PASS contract_identity_sha256={config['contract_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
