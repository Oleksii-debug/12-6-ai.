from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/data/d03_ua_nbu_pdftotext_materialization_v1.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
VERSION_RE = re.compile(r"pdftotext version ([0-9]+(?:\.[0-9]+)+)", re.IGNORECASE)


class NbuTextMaterializationError(ValueError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def self_identity(value: Mapping[str, Any], field: str) -> str:
    clone = deepcopy(dict(value))
    clone.pop(field, None)
    return sha256((canonical_json(clone) + "\n").encode("utf-8"))


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "12-6.d03-ua-nbu-pdftotext-materialization.v1":
        raise NbuTextMaterializationError("schema drift")
    if config.get("status") != "PREPARED_TEXT_MATERIALIZATION_ZERO_CREDIT":
        raise NbuTextMaterializationError("status drift")

    base = config.get("base_authority")
    if not isinstance(base, Mapping):
        raise NbuTextMaterializationError("base authority missing")
    expected_base = {
        "repository": "Oleksii-debug/12-6-ai.",
        "parent_pr": 912,
        "discovery_pr": 898,
        "lane": "D03",
        "control_issue": 548,
    }
    if any(base.get(k) != v for k, v in expected_base.items()):
        raise NbuTextMaterializationError("base authority drift")
    for key in ("parent_head_sha", "discovery_head_sha"):
        if not isinstance(base.get(key), str) or not HEX40.fullmatch(base[key]):
            raise NbuTextMaterializationError(f"malformed {key}")
    for key in (
        "parent_pdf_pin_contract_identity_sha256",
        "discovery_contract_identity_sha256",
    ):
        if not isinstance(base.get(key), str) or not HEX64.fullmatch(base[key]):
            raise NbuTextMaterializationError(f"malformed {key}")

    source = config.get("source")
    if not isinstance(source, Mapping):
        raise NbuTextMaterializationError("source missing")
    required_source = {
        "source_id": "ua.nbu.official-resolutions",
        "family_id": "ua.nbu.official-resolutions",
        "allowed_origin": "https://bank.gov.ua",
    }
    if any(source.get(k) != v for k, v in required_source.items()):
        raise NbuTextMaterializationError("source drift")
    for key in ("document_path_regex", "official_pdf_path_regex"):
        value = source.get(key)
        if not isinstance(value, str):
            raise NbuTextMaterializationError(f"missing {key}")
        try:
            re.compile(value)
        except re.error as exc:
            raise NbuTextMaterializationError(f"invalid {key}") from exc

    extractor = config.get("extractor")
    required_extractor = {
        "backend": "poppler-pdftotext",
        "executable": "pdftotext",
        "required_version": "25.06.0",
        "arguments": ["-enc", "UTF-8", "-nopgbrk"],
        "ocr_allowed": False,
        "shell_allowed": False,
        "network_allowed_by_extractor": False,
        "double_extract_required": True,
        "exact_output_bytes_required": True,
    }
    if not isinstance(extractor, Mapping) or any(
        extractor.get(k) != v for k, v in required_extractor.items()
    ):
        raise NbuTextMaterializationError("extractor contract drift")

    materialization = config.get("materialization")
    required_materialization = {
        "class": "LOCAL_FREE",
        "one_record_per_pdf": True,
        "utf8_strict": True,
        "nul_forbidden": True,
        "durable_evidence_body_free": True,
        "text_artifact_is_training_authority": False,
    }
    if not isinstance(materialization, Mapping) or any(
        materialization.get(k) != v for k, v in required_materialization.items()
    ):
        raise NbuTextMaterializationError("materialization contract drift")
    numeric_bounds = {
        "hard_max_records": (1, 240),
        "min_text_bytes": (1, 4096),
        "hard_max_text_bytes_per_record": (4096, 50_000_000),
        "hard_max_total_text_bytes": (4096, 500_000_000),
        "source_fetch_timeout_seconds": (1, 120),
    }
    for key, (lo, hi) in numeric_bounds.items():
        value = materialization.get(key)
        if not isinstance(value, int) or not lo <= value <= hi:
            raise NbuTextMaterializationError(f"unsafe {key}")

    claims = config.get("claims")
    required_claims = {
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "text_materialization_execution_authorized": True,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_authorized": False,
        "learned_20m_claim": False,
    }
    if not isinstance(claims, Mapping) or any(claims.get(k) != v for k, v in required_claims.items()):
        raise NbuTextMaterializationError("claims truth boundary drift")

    identity = config.get("contract_identity_sha256")
    if not isinstance(identity, str) or not HEX64.fullmatch(identity):
        raise NbuTextMaterializationError("contract identity malformed")
    if self_identity(config, "contract_identity_sha256") != identity:
        raise NbuTextMaterializationError("contract identity mismatch")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise NbuTextMaterializationError("config root must be object")
    validate_config(value)
    return value


def _canonical_source_url(url: str, config: Mapping[str, Any], kind: str) -> str:
    parts = urlsplit(url)
    origin = urlsplit(config["source"]["allowed_origin"])
    if parts.scheme != "https" or parts.netloc.casefold() != origin.netloc.casefold():
        raise NbuTextMaterializationError(f"{kind} URL is cross-origin/non-https")
    if parts.username or parts.password or parts.port not in (None, 443):
        raise NbuTextMaterializationError(f"{kind} authority is non-canonical")
    if parts.query or parts.fragment:
        raise NbuTextMaterializationError(f"{kind} URL must already be canonical")
    key = "document_path_regex" if kind == "document" else "official_pdf_path_regex"
    if not re.fullmatch(config["source"][key], parts.path):
        raise NbuTextMaterializationError(f"{kind} path outside allowlist")
    return f"https://{origin.netloc}{parts.path}"


def validate_discovery_evidence(evidence: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, str]:
    if evidence.get("schema") != "12-6.d03-ua-nbu-official-resolutions-discovery-evidence.v1":
        raise NbuTextMaterializationError("discovery schema drift")
    if evidence.get("status") != "DISCOVERY_ONLY_ZERO_CREDIT":
        raise NbuTextMaterializationError("discovery status drift")
    if evidence.get("contract_identity_sha256") != config["base_authority"]["discovery_contract_identity_sha256"]:
        raise NbuTextMaterializationError("discovery contract mismatch")
    if evidence.get("source_id") != config["source"]["source_id"] or evidence.get("family_id") != config["source"]["family_id"]:
        raise NbuTextMaterializationError("discovery source mismatch")
    documents = evidence.get("documents")
    if not isinstance(documents, list) or not documents:
        raise NbuTextMaterializationError("discovery documents missing")
    if evidence.get("discovered_documents") != len(documents) or len(documents) > config["materialization"]["hard_max_records"]:
        raise NbuTextMaterializationError("discovery document count invalid")

    pdf_to_document: dict[str, str] = {}
    for item in documents:
        if not isinstance(item, Mapping):
            raise NbuTextMaterializationError("malformed discovery document")
        document_url = _canonical_source_url(str(item.get("document_url", "")), config, "document")
        pdfs = item.get("official_pdf_urls")
        if not isinstance(pdfs, list) or not pdfs or sorted(set(pdfs)) != pdfs:
            raise NbuTextMaterializationError("discovery PDF set invalid")
        for raw_pdf in pdfs:
            pdf_url = _canonical_source_url(str(raw_pdf), config, "pdf")
            if pdf_url in pdf_to_document:
                raise NbuTextMaterializationError("one PDF is mapped to multiple documents")
            pdf_to_document[pdf_url] = document_url
        if not HEX64.fullmatch(str(item.get("page_metadata_sha256", ""))):
            raise NbuTextMaterializationError("discovery page metadata hash malformed")

    truth = {
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
    if any(evidence.get(k) != v for k, v in truth.items()):
        raise NbuTextMaterializationError("discovery truth boundary drift")
    identity = evidence.get("evidence_identity_sha256")
    if not isinstance(identity, str) or not HEX64.fullmatch(identity):
        raise NbuTextMaterializationError("discovery evidence identity malformed")
    if self_identity(evidence, "evidence_identity_sha256") != identity:
        raise NbuTextMaterializationError("discovery evidence identity mismatch")
    return pdf_to_document


def validate_pin_evidence(evidence: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if evidence.get("schema") != "12-6.d03-ua-nbu-official-pdf-pins-evidence.v1":
        raise NbuTextMaterializationError("pin schema drift")
    if evidence.get("status") != "PDF_BYTES_PINNED_ZERO_CREDIT":
        raise NbuTextMaterializationError("pin status drift")
    if evidence.get("contract_identity_sha256") != config["base_authority"]["parent_pdf_pin_contract_identity_sha256"]:
        raise NbuTextMaterializationError("pin contract mismatch")
    if evidence.get("source_id") != config["source"]["source_id"] or evidence.get("family_id") != config["source"]["family_id"]:
        raise NbuTextMaterializationError("pin source mismatch")
    pins = evidence.get("pins")
    if not isinstance(pins, list) or not pins or evidence.get("pinned_pdfs") != len(pins):
        raise NbuTextMaterializationError("pin list invalid")
    if len(pins) > config["materialization"]["hard_max_records"]:
        raise NbuTextMaterializationError("pin count exceeds hard bound")

    by_url: dict[str, Mapping[str, Any]] = {}
    observed = 0
    for item in pins:
        if not isinstance(item, Mapping):
            raise NbuTextMaterializationError("malformed pin")
        pdf_url = _canonical_source_url(str(item.get("pdf_url", "")), config, "pdf")
        if pdf_url in by_url:
            raise NbuTextMaterializationError("duplicate PDF pin")
        digest = item.get("pdf_sha256")
        size = item.get("pdf_bytes")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise NbuTextMaterializationError("PDF pin hash malformed")
        if not isinstance(size, int) or size <= 0:
            raise NbuTextMaterializationError("PDF pin size malformed")
        if item.get("content_type") not in ("application/pdf", "application/octet-stream"):
            raise NbuTextMaterializationError("PDF pin content type invalid")
        observed += size
        by_url[pdf_url] = item
    if evidence.get("observed_pdf_bytes") != observed:
        raise NbuTextMaterializationError("PDF pin byte total mismatch")

    truth = {
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
    if any(evidence.get(k) != v for k, v in truth.items()):
        raise NbuTextMaterializationError("pin truth boundary drift")
    identity = evidence.get("evidence_identity_sha256")
    if not isinstance(identity, str) or not HEX64.fullmatch(identity):
        raise NbuTextMaterializationError("pin evidence identity malformed")
    if self_identity(evidence, "evidence_identity_sha256") != identity:
        raise NbuTextMaterializationError("pin evidence identity mismatch")
    return by_url


class _SameIdentityRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, config: Mapping[str, Any], requested_url: str) -> None:
        super().__init__()
        self.config = config
        self.requested_url = requested_url

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        try:
            canonical = _canonical_source_url(newurl, self.config, "pdf")
            if canonical != self.requested_url:
                raise NbuTextMaterializationError("PDF redirect changed identity")
        except NbuTextMaterializationError as exc:
            raise urllib.error.HTTPError(newurl, code, str(exc), headers, fp) from exc
        return super().redirect_request(req, fp, code, msg, headers, canonical)


def network_fetch_pdf(url: str, expected_bytes: int, config: Mapping[str, Any]) -> bytes:
    canonical = _canonical_source_url(url, config, "pdf")
    opener = urllib.request.build_opener(_SameIdentityRedirect(config, canonical))
    request = urllib.request.Request(
        canonical,
        headers={"User-Agent": "12-6-ai-d03-nbu-pdftotext/1.0 (+source-audit)"},
        method="GET",
    )
    timeout = config["materialization"]["source_fetch_timeout_seconds"]
    with opener.open(request, timeout=timeout) as response:
        final = _canonical_source_url(response.geturl(), config, "pdf")
        if final != canonical:
            raise NbuTextMaterializationError("PDF final identity changed")
        body = response.read(expected_bytes + 1)
    if len(body) != expected_bytes:
        raise NbuTextMaterializationError("refetched PDF size differs from pin")
    return body


def probe_extractor_version(config: Mapping[str, Any]) -> str:
    command = [config["extractor"]["executable"], "-v"]
    try:
        completed = subprocess.run(command, capture_output=True, check=False, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise NbuTextMaterializationError("pdftotext version probe failed") from exc
    output = (completed.stdout + completed.stderr).decode("utf-8", errors="replace")
    match = VERSION_RE.search(output)
    if completed.returncode != 0 or match is None:
        raise NbuTextMaterializationError("pdftotext version unavailable")
    version = match.group(1)
    if version != config["extractor"]["required_version"]:
        raise NbuTextMaterializationError(
            f"pdftotext version mismatch: required {config['extractor']['required_version']}, got {version}"
        )
    return version


def extract_once(pdf_bytes: bytes, config: Mapping[str, Any]) -> bytes:
    with tempfile.TemporaryDirectory(prefix="12-6-nbu-pdftotext-") as temp_dir:
        pdf_path = Path(temp_dir) / "input.pdf"
        text_path = Path(temp_dir) / "output.txt"
        pdf_path.write_bytes(pdf_bytes)
        command = [
            config["extractor"]["executable"],
            *config["extractor"]["arguments"],
            str(pdf_path),
            str(text_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, check=False, timeout=60)
        except (OSError, subprocess.SubprocessError) as exc:
            raise NbuTextMaterializationError("pdftotext execution failed") from exc
        if completed.returncode != 0 or not text_path.is_file():
            detail = completed.stderr.decode("utf-8", errors="replace")[:500]
            raise NbuTextMaterializationError(f"pdftotext failed: {detail}")
        return text_path.read_bytes()


def _validate_text_bytes(text: bytes, config: Mapping[str, Any]) -> str:
    minimum = config["materialization"]["min_text_bytes"]
    maximum = config["materialization"]["hard_max_text_bytes_per_record"]
    if not minimum <= len(text) <= maximum:
        raise NbuTextMaterializationError("extracted text size outside bounds")
    if b"\x00" in text:
        raise NbuTextMaterializationError("NUL byte in extracted text")
    try:
        decoded = text.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise NbuTextMaterializationError("extracted text is not strict UTF-8") from exc
    if not decoded.strip():
        raise NbuTextMaterializationError("extracted text is blank")
    return decoded


def materialize_text(
    discovery_evidence: Mapping[str, Any],
    pin_evidence: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    fetcher: Callable[[str, int, Mapping[str, Any]], bytes] = network_fetch_pdf,
    extractor: Callable[[bytes, Mapping[str, Any]], bytes] = extract_once,
    extractor_version: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    validate_config(config)
    pdf_to_document = validate_discovery_evidence(discovery_evidence, config)
    pins = validate_pin_evidence(pin_evidence, config)
    if pin_evidence.get("parent_discovery_evidence_identity_sha256") != discovery_evidence.get("evidence_identity_sha256"):
        raise NbuTextMaterializationError("pin/discovery evidence lineage mismatch")
    if set(pins) != set(pdf_to_document):
        raise NbuTextMaterializationError("pin set does not equal discovery PDF set")

    version = extractor_version if extractor_version is not None else probe_extractor_version(config)
    if version != config["extractor"]["required_version"]:
        raise NbuTextMaterializationError("extractor version mismatch")

    total_text_bytes = 0
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for pdf_url in sorted(pins):
        pin = pins[pdf_url]
        pdf_bytes = fetcher(pdf_url, int(pin["pdf_bytes"]), config)
        if len(pdf_bytes) != pin["pdf_bytes"] or sha256(pdf_bytes) != pin["pdf_sha256"]:
            raise NbuTextMaterializationError("refetched PDF bytes do not match exact pin")

        first = extractor(pdf_bytes, config)
        second = extractor(pdf_bytes, config)
        if first != second:
            raise NbuTextMaterializationError("independent text extractions differ")
        text = _validate_text_bytes(first, config)
        text_digest = sha256(first)
        total_text_bytes += len(first)
        if total_text_bytes > config["materialization"]["hard_max_total_text_bytes"]:
            raise NbuTextMaterializationError("total extracted text byte bound exceeded")
        document_url = pdf_to_document[pdf_url]
        record_identity_input = {
            "document_url": document_url,
            "pdf_url": pdf_url,
            "pdf_sha256": pin["pdf_sha256"],
            "text_sha256": text_digest,
        }
        record_id = sha256((canonical_json(record_identity_input) + "\n").encode("utf-8"))
        record = {
            "schema": "12-6.d03-ua-nbu-pdftotext-record.v1",
            "record_id": record_id,
            "source_id": config["source"]["source_id"],
            "family_id": config["source"]["family_id"],
            "document_url": document_url,
            "pdf_url": pdf_url,
            "pdf_sha256": pin["pdf_sha256"],
            "text_sha256": text_digest,
            "text_bytes": len(first),
            "extractor_backend": config["extractor"]["backend"],
            "extractor_version": version,
            "extractor_arguments": list(config["extractor"]["arguments"]),
            "text": text,
        }
        records.append(record)
        manifest.append({k: v for k, v in record.items() if k != "text"})

    if not records or len(records) > config["materialization"]["hard_max_records"]:
        raise NbuTextMaterializationError("materialized record count outside hard bound")
    artifact = b"".join(
        (canonical_json(record) + "\n").encode("utf-8") for record in records
    )
    evidence: dict[str, Any] = {
        "schema": "12-6.d03-ua-nbu-pdftotext-materialization-evidence.v1",
        "status": "TEXT_MATERIALIZED_ZERO_CREDIT",
        "contract_identity_sha256": config["contract_identity_sha256"],
        "discovery_evidence_identity_sha256": discovery_evidence["evidence_identity_sha256"],
        "pdf_pin_evidence_identity_sha256": pin_evidence["evidence_identity_sha256"],
        "source_id": config["source"]["source_id"],
        "family_id": config["source"]["family_id"],
        "extractor_backend": config["extractor"]["backend"],
        "extractor_version": version,
        "extractor_arguments": list(config["extractor"]["arguments"]),
        "materialized_records": len(records),
        "observed_text_bytes": total_text_bytes,
        "text_artifact_bytes": len(artifact),
        "text_artifact_sha256": sha256(artifact),
        "records": manifest,
        "raw_text_emitted_in_evidence": False,
        "text_artifact_is_training_authority": False,
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_used": False,
    }
    evidence["evidence_identity_sha256"] = sha256(
        (canonical_json(evidence) + "\n").encode("utf-8")
    )
    validate_materialization_evidence(evidence, config)
    return artifact, evidence


def validate_materialization_evidence(evidence: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    validate_config(config)
    if evidence.get("schema") != "12-6.d03-ua-nbu-pdftotext-materialization-evidence.v1":
        raise NbuTextMaterializationError("materialization evidence schema drift")
    if evidence.get("status") != "TEXT_MATERIALIZED_ZERO_CREDIT":
        raise NbuTextMaterializationError("materialization evidence status drift")
    if evidence.get("contract_identity_sha256") != config["contract_identity_sha256"]:
        raise NbuTextMaterializationError("materialization contract mismatch")
    if evidence.get("extractor_backend") != config["extractor"]["backend"] or evidence.get("extractor_version") != config["extractor"]["required_version"]:
        raise NbuTextMaterializationError("extractor evidence mismatch")
    if evidence.get("extractor_arguments") != config["extractor"]["arguments"]:
        raise NbuTextMaterializationError("extractor arguments mismatch")
    records = evidence.get("records")
    if not isinstance(records, list) or not records or evidence.get("materialized_records") != len(records):
        raise NbuTextMaterializationError("record manifest invalid")
    if len(records) > config["materialization"]["hard_max_records"]:
        raise NbuTextMaterializationError("record manifest exceeds hard bound")
    total = 0
    ids = set()
    for item in records:
        if not isinstance(item, Mapping) or "text" in item:
            raise NbuTextMaterializationError("evidence must remain body-free")
        record_id = item.get("record_id")
        if not isinstance(record_id, str) or not HEX64.fullmatch(record_id) or record_id in ids:
            raise NbuTextMaterializationError("record identity invalid")
        ids.add(record_id)
        _canonical_source_url(str(item.get("document_url", "")), config, "document")
        _canonical_source_url(str(item.get("pdf_url", "")), config, "pdf")
        if not HEX64.fullmatch(str(item.get("pdf_sha256", ""))) or not HEX64.fullmatch(str(item.get("text_sha256", ""))):
            raise NbuTextMaterializationError("record hash invalid")
        text_bytes = item.get("text_bytes")
        if not isinstance(text_bytes, int) or not config["materialization"]["min_text_bytes"] <= text_bytes <= config["materialization"]["hard_max_text_bytes_per_record"]:
            raise NbuTextMaterializationError("record text size invalid")
        if item.get("extractor_backend") != config["extractor"]["backend"] or item.get("extractor_version") != config["extractor"]["required_version"]:
            raise NbuTextMaterializationError("record extractor identity mismatch")
        if item.get("extractor_arguments") != config["extractor"]["arguments"]:
            raise NbuTextMaterializationError("record extractor arguments mismatch")
        total += text_bytes
    if evidence.get("observed_text_bytes") != total or total > config["materialization"]["hard_max_total_text_bytes"]:
        raise NbuTextMaterializationError("materialized text total mismatch")
    if not HEX64.fullmatch(str(evidence.get("text_artifact_sha256", ""))) or not isinstance(evidence.get("text_artifact_bytes"), int) or evidence["text_artifact_bytes"] <= 0:
        raise NbuTextMaterializationError("text artifact identity malformed")

    truth = {
        "raw_text_emitted_in_evidence": False,
        "text_artifact_is_training_authority": False,
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_used": False,
    }
    if any(evidence.get(k) != v for k, v in truth.items()):
        raise NbuTextMaterializationError("materialization evidence promotion detected")
    identity = evidence.get("evidence_identity_sha256")
    if not isinstance(identity, str) or not HEX64.fullmatch(identity) or self_identity(evidence, "evidence_identity_sha256") != identity:
        raise NbuTextMaterializationError("materialization evidence identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize deterministic NBU official-resolution text with qualified pdftotext.")
    parser.add_argument("discovery_evidence", type=Path)
    parser.add_argument("pdf_pin_evidence", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--text-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    discovery = json.loads(args.discovery_evidence.read_text(encoding="utf-8"))
    pins = json.loads(args.pdf_pin_evidence.read_text(encoding="utf-8"))
    if not isinstance(discovery, dict) or not isinstance(pins, dict):
        raise NbuTextMaterializationError("input evidence root must be object")
    artifact, evidence = materialize_text(discovery, pins, config)
    args.text_output.write_bytes(artifact)
    args.evidence_output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "PASS "
        f"records={evidence['materialized_records']} "
        f"observed_text_bytes={evidence['observed_text_bytes']} "
        f"text_artifact_sha256={evidence['text_artifact_sha256']} "
        f"evidence_identity_sha256={evidence['evidence_identity_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
