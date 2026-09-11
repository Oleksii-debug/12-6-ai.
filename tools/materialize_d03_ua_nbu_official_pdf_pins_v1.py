from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import urllib.request
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/d03_ua_nbu_official_pdf_pin_v1.json"
PARENT_TOOL = ROOT / "tools/validate_d03_ua_nbu_official_resolutions_intake_v1.py"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class NbuPdfPinError(ValueError):
    pass


@dataclass(frozen=True)
class FetchedPdf:
    body: bytes
    final_url: str
    content_type: str


def _load_parent():
    spec = importlib.util.spec_from_file_location("nbu_parent_intake", PARENT_TOOL)
    if spec is None or spec.loader is None:
        raise NbuPdfPinError("parent intake module unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PARENT = _load_parent()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def self_identity(value: Mapping[str, Any], field: str) -> str:
    clone = deepcopy(dict(value))
    clone.pop(field, None)
    return sha256((canonical_json(clone) + "\n").encode("utf-8"))


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "12-6.d03-ua-nbu-official-pdf-pin.v1":
        raise NbuPdfPinError("schema drift")
    if config.get("status") != "PREPARED_PDF_BYTE_PIN_ZERO_CREDIT":
        raise NbuPdfPinError("status drift")
    base = config.get("base_authority")
    if not isinstance(base, Mapping):
        raise NbuPdfPinError("base authority missing")
    if base.get("repository") != "Oleksii-debug/12-6-ai." or base.get("parent_pr") != 898:
        raise NbuPdfPinError("base authority drift")
    if not isinstance(base.get("parent_head_sha"), str) or not HEX40.fullmatch(base["parent_head_sha"]):
        raise NbuPdfPinError("parent head malformed")
    if not isinstance(base.get("parent_contract_identity_sha256"), str) or not HEX64.fullmatch(base["parent_contract_identity_sha256"]):
        raise NbuPdfPinError("parent contract identity malformed")
    if base.get("lane") != "D03" or base.get("control_issue") != 548:
        raise NbuPdfPinError("control binding drift")

    source = config.get("source")
    if not isinstance(source, Mapping):
        raise NbuPdfPinError("source missing")
    if source.get("source_id") != "ua.nbu.official-resolutions" or source.get("family_id") != "ua.nbu.official-resolutions":
        raise NbuPdfPinError("source drift")
    if source.get("allowed_origin") != "https://bank.gov.ua":
        raise NbuPdfPinError("origin drift")
    try:
        re.compile(str(source["official_pdf_path_regex"]))
    except (KeyError, re.error) as exc:
        raise NbuPdfPinError("PDF allowlist invalid") from exc

    execution = config.get("execution")
    required_execution = {
        "class": "LOCAL_FREE",
        "fetches_per_pdf": 2,
        "same_origin_redirects_only": True,
        "byte_identical_replay_required": True,
        "durable_evidence_body_free": True,
    }
    if not isinstance(execution, Mapping) or any(execution.get(k) != v for k, v in required_execution.items()):
        raise NbuPdfPinError("execution boundary drift")
    numeric_bounds = {
        "hard_max_documents": (1, 120),
        "hard_max_pdfs": (1, 240),
        "min_pdf_bytes": (1, 4096),
        "hard_max_pdf_bytes": (4096, 100_000_000),
        "hard_max_total_pdf_bytes": (4096, 1_000_000_000),
        "request_timeout_seconds": (1, 120),
    }
    for key, (lo, hi) in numeric_bounds.items():
        value = execution.get(key)
        if not isinstance(value, int) or not lo <= value <= hi:
            raise NbuPdfPinError(f"unsafe {key}")

    pdf_structure = config.get("pdf_structure")
    if not isinstance(pdf_structure, Mapping):
        raise NbuPdfPinError("pdf_structure missing")
    if pdf_structure.get("pdf_header_within_first_bytes") != 1024:
        raise NbuPdfPinError("PDF header boundary drift")
    if pdf_structure.get("eof_marker_within_last_bytes") != 2048:
        raise NbuPdfPinError("PDF EOF boundary drift")
    allowed = pdf_structure.get("allowed_content_types")
    if allowed != ["application/pdf", "application/octet-stream"]:
        raise NbuPdfPinError("content-type allowlist drift")

    claims = config.get("claims")
    expected_claims = {
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "text_materialized": False,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_authorized": False,
        "learned_20m_claim": False,
    }
    if not isinstance(claims, Mapping) or any(claims.get(k) != v for k, v in expected_claims.items()):
        raise NbuPdfPinError("claims promotion detected")

    identity = config.get("contract_identity_sha256")
    if not isinstance(identity, str) or not HEX64.fullmatch(identity):
        raise NbuPdfPinError("contract identity malformed")
    if self_identity(config, "contract_identity_sha256") != identity:
        raise NbuPdfPinError("contract identity mismatch")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NbuPdfPinError("config root must be an object")
    validate_config(value)
    return value


def _canonical_pdf_url(url: str, config: Mapping[str, Any]) -> str:
    parts = urlsplit(url)
    origin = urlsplit(config["source"]["allowed_origin"])
    if parts.scheme != "https" or parts.netloc.casefold() != origin.netloc.casefold():
        raise NbuPdfPinError("cross-origin/non-https PDF URL")
    if parts.username or parts.password or parts.port not in (None, 443):
        raise NbuPdfPinError("non-canonical PDF authority")
    if parts.query or parts.fragment:
        raise NbuPdfPinError("pinned PDF URL must already be canonical")
    if not re.fullmatch(config["source"]["official_pdf_path_regex"], parts.path):
        raise NbuPdfPinError("PDF path outside allowlist")
    return f"https://{origin.netloc}{parts.path}"


class _SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = config

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        try:
            canonical = _canonical_pdf_url(newurl, self.config)
        except NbuPdfPinError as exc:
            raise urllib.error.HTTPError(newurl, code, str(exc), headers, fp) from exc
        return super().redirect_request(req, fp, code, msg, headers, canonical)


def network_fetch(url: str, config: Mapping[str, Any]) -> FetchedPdf:
    canonical = _canonical_pdf_url(url, config)
    opener = urllib.request.build_opener(_SameOriginRedirect(config))
    request = urllib.request.Request(canonical, headers={"User-Agent": "12-6-ai-d03-nbu-pdf-pin/1.0 (+source-audit)"}, method="GET")
    timeout = config["execution"]["request_timeout_seconds"]
    with opener.open(request, timeout=timeout) as response:
        final_url = _canonical_pdf_url(response.geturl(), config)
        content_type = response.headers.get_content_type().casefold()
        hard = config["execution"]["hard_max_pdf_bytes"]
        body = response.read(hard + 1)
    if len(body) > hard:
        raise NbuPdfPinError("PDF exceeds per-file hard byte bound")
    return FetchedPdf(body=body, final_url=final_url, content_type=content_type)


def _verify_pdf(payload: FetchedPdf, requested_url: str, config: Mapping[str, Any]) -> None:
    requested = _canonical_pdf_url(requested_url, config)
    if payload.final_url != requested:
        raise NbuPdfPinError("final PDF identity changed")
    if payload.content_type.casefold() not in config["pdf_structure"]["allowed_content_types"]:
        raise NbuPdfPinError("unexpected PDF content type")
    body = payload.body
    if len(body) < config["execution"]["min_pdf_bytes"]:
        raise NbuPdfPinError("PDF below minimum byte bound")
    if len(body) > config["execution"]["hard_max_pdf_bytes"]:
        raise NbuPdfPinError("PDF exceeds hard byte bound")
    head_window = config["pdf_structure"]["pdf_header_within_first_bytes"]
    if b"%PDF-" not in body[:head_window]:
        raise NbuPdfPinError("PDF header marker missing")
    tail_window = config["pdf_structure"]["eof_marker_within_last_bytes"]
    if b"%%EOF" not in body[-tail_window:]:
        raise NbuPdfPinError("PDF EOF marker missing")


def materialize_pdf_pins(discovery_evidence: Mapping[str, Any], config: Mapping[str, Any], fetcher: Callable[[str, Mapping[str, Any]], FetchedPdf] = network_fetch) -> dict[str, Any]:
    validate_config(config)
    if discovery_evidence.get("contract_identity_sha256") != config["base_authority"]["parent_contract_identity_sha256"]:
        raise NbuPdfPinError("discovery evidence uses unexpected parent contract")
    try:
        PARENT.validate_evidence(discovery_evidence, PARENT.load_config())
    except Exception as exc:
        raise NbuPdfPinError("parent discovery evidence invalid") from exc

    documents = discovery_evidence["documents"]
    if len(documents) > config["execution"]["hard_max_documents"]:
        raise NbuPdfPinError("discovery document bound exceeded")
    urls: list[str] = []
    for document in documents:
        for raw in document["official_pdf_urls"]:
            canonical = _canonical_pdf_url(raw, config)
            if canonical not in urls:
                urls.append(canonical)
    urls.sort()
    if not urls or len(urls) > config["execution"]["hard_max_pdfs"]:
        raise NbuPdfPinError("PDF count outside hard bound")

    total = 0
    pins: list[dict[str, Any]] = []
    for url in urls:
        first = fetcher(url, config)
        second = fetcher(url, config)
        _verify_pdf(first, url, config)
        _verify_pdf(second, url, config)
        if first.body != second.body:
            raise NbuPdfPinError("independent PDF fetches differ")
        digest = sha256(first.body)
        byte_count = len(first.body)
        total += byte_count
        if total > config["execution"]["hard_max_total_pdf_bytes"]:
            raise NbuPdfPinError("total PDF byte bound exceeded")
        pins.append({"pdf_url": url, "pdf_sha256": digest, "pdf_bytes": byte_count, "content_type": first.content_type.casefold()})

    evidence: dict[str, Any] = {
        "schema": "12-6.d03-ua-nbu-official-pdf-pins-evidence.v1",
        "status": "PDF_BYTES_PINNED_ZERO_CREDIT",
        "contract_identity_sha256": config["contract_identity_sha256"],
        "parent_discovery_evidence_identity_sha256": discovery_evidence["evidence_identity_sha256"],
        "source_id": config["source"]["source_id"],
        "family_id": config["source"]["family_id"],
        "pinned_pdfs": len(pins),
        "observed_pdf_bytes": total,
        "pins": pins,
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
    evidence["evidence_identity_sha256"] = sha256((canonical_json(evidence) + "\n").encode("utf-8"))
    return evidence


def validate_pin_evidence(evidence: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    validate_config(config)
    if evidence.get("schema") != "12-6.d03-ua-nbu-official-pdf-pins-evidence.v1":
        raise NbuPdfPinError("evidence schema drift")
    if evidence.get("status") != "PDF_BYTES_PINNED_ZERO_CREDIT":
        raise NbuPdfPinError("evidence status drift")
    if evidence.get("contract_identity_sha256") != config["contract_identity_sha256"]:
        raise NbuPdfPinError("evidence contract mismatch")
    pins = evidence.get("pins")
    if not isinstance(pins, list) or not pins or evidence.get("pinned_pdfs") != len(pins):
        raise NbuPdfPinError("pin list invalid")
    if len(pins) > config["execution"]["hard_max_pdfs"]:
        raise NbuPdfPinError("pin count exceeds hard bound")
    total = 0
    seen = set()
    for item in pins:
        if not isinstance(item, Mapping):
            raise NbuPdfPinError("malformed pin")
        url = _canonical_pdf_url(str(item.get("pdf_url", "")), config)
        if url in seen:
            raise NbuPdfPinError("duplicate PDF pin")
        seen.add(url)
        digest = item.get("pdf_sha256")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise NbuPdfPinError("bad PDF digest")
        size = item.get("pdf_bytes")
        if not isinstance(size, int) or not config["execution"]["min_pdf_bytes"] <= size <= config["execution"]["hard_max_pdf_bytes"]:
            raise NbuPdfPinError("bad PDF size")
        content_type = item.get("content_type")
        if content_type not in config["pdf_structure"]["allowed_content_types"]:
            raise NbuPdfPinError("bad PDF content type")
        total += size
    if evidence.get("observed_pdf_bytes") != total or total > config["execution"]["hard_max_total_pdf_bytes"]:
        raise NbuPdfPinError("PDF total mismatch")

    required = {
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
    if any(evidence.get(k) != v for k, v in required.items()):
        raise NbuPdfPinError("evidence promotion detected")
    identity = evidence.get("evidence_identity_sha256")
    if not isinstance(identity, str) or self_identity(evidence, "evidence_identity_sha256") != identity:
        raise NbuPdfPinError("evidence identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pin exact NBU official PDF bytes from validated discovery evidence.")
    parser.add_argument("discovery_evidence", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    discovery = json.loads(args.discovery_evidence.read_text(encoding="utf-8"))
    evidence = materialize_pdf_pins(discovery, config)
    validate_pin_evidence(evidence, config)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS pinned_pdfs={evidence['pinned_pdfs']} observed_pdf_bytes={evidence['observed_pdf_bytes']} evidence_identity_sha256={evidence['evidence_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
