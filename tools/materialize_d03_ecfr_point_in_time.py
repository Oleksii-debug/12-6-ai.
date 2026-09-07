#!/usr/bin/env python3
"""Bounded, fail-closed point-in-time eCFR materialization for D03.

This tool proves transport and deterministic parsing only. It never grants corpus,
family, tokenizer, training, loss-position, or paid-compute credit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from io import BufferedIOBase
from pathlib import Path
from typing import Any, NamedTuple

REQUEST_SCHEMA = "12-6.d03-ecfr-point-in-time-materialization-request.v1"
REQUEST_STATUS = "READY_FOR_NETWORK_EXECUTION_ZERO_CREDIT"
EVIDENCE_SCHEMA = "12-6.d03-ecfr-point-in-time-materialization-evidence.v1"
SOURCE_ID = "en.us.ecfr.regulations"
FAMILY_ID = "us.federal-regulations.ecfr"
TITLES_ENDPOINT = "https://www.ecfr.gov/api/versioner/v1/titles.json"
FULL_TITLE_RE = re.compile(
    r"https://www\.ecfr\.gov/api/versioner/v1/full/(\d{4}-\d{2}-\d{2})/title-(\d+)\.xml"
)
ALLOWED_CONTENT_TYPES = {
    "application/xml",
    "text/xml",
    "application/octet-stream",
}
ZERO_CLAIMS = {
    "canonical_capacity_credit_bytes": 0,
    "family_credit": 0,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "tokenizer_fit_authorized": False,
    "model_training_executed": False,
    "optimizer_updates": 0,
    "final_test_accessed": False,
    "paid_compute_authorized": False,
    "research_corpus_v1_released": False,
    "learned_20m_claim": False,
}


class MaterializationError(ValueError):
    """Raised when a materialization request or acquisition fails closed."""


class FetchResult(NamedTuple):
    status: int
    final_url: str
    content_type: str
    body: bytes


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


def _canonical_sha256(payload: dict[str, Any], identity_field: str) -> str:
    clone = copy.deepcopy(payload)
    clone.pop(identity_field, None)
    encoded = json.dumps(
        clone,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(request, dict), "request must be a JSON object")
    _require(request.get("schema") == REQUEST_SCHEMA, "unexpected request schema")
    _require(request.get("status") == REQUEST_STATUS, "request status drift")

    identity = request.get("request_identity_sha256")
    _require(_is_lower_sha256(identity), "request identity must be lowercase SHA-256")
    _require(
        identity == _canonical_sha256(request, "request_identity_sha256"),
        "request identity mismatch",
    )

    authority = request.get("authority")
    _require(isinstance(authority, dict), "authority block missing")
    _require(authority.get("source_id") == SOURCE_ID, "source identity drift")
    _require(authority.get("family_id") == FAMILY_ID, "family identity drift")
    _require(authority.get("stratum") == "en", "eCFR request must remain EN")
    _require(authority.get("titles_endpoint") == TITLES_ENDPOINT, "titles endpoint drift")
    metadata_date = authority.get("titles_metadata_date")
    _require(isinstance(metadata_date, str), "titles metadata date missing")
    _require(authority.get("import_in_progress") is False, "metadata import was in progress")
    reserved = authority.get("reserved_titles")
    _require(isinstance(reserved, list) and all(isinstance(x, int) for x in reserved), "reserved titles invalid")
    _require(35 in reserved, "known reserved title 35 must remain excluded")
    _require(
        authority.get("metadata_observation_authority") == "DISCOVERY_ONLY",
        "metadata observation cannot become training authority",
    )

    selection = request.get("selection")
    _require(isinstance(selection, dict), "selection block missing")
    date = selection.get("date")
    title = selection.get("title")
    url = selection.get("url")
    _require(isinstance(date, str), "selection date missing")
    _require(isinstance(title, int) and 1 <= title <= 50, "title must be in observed range 1..50")
    _require(title not in reserved, "reserved title cannot be materialized")
    _require(isinstance(url, str), "selection URL missing")
    match = FULL_TITLE_RE.fullmatch(url)
    _require(match is not None, "selection must use exact historical full-title URL without query/fragment")
    assert match is not None
    _require(match.group(1) == date, "URL date does not match selection date")
    _require(int(match.group(2)) == title, "URL title does not match selection title")
    _require(date <= metadata_date, "selection date exceeds frozen metadata availability")

    bounds = request.get("bounds")
    _require(isinstance(bounds, dict), "bounds block missing")
    max_bytes = bounds.get("max_response_bytes")
    _require(isinstance(max_bytes, int) and 1 <= max_bytes <= 32 * 1024 * 1024, "max_response_bytes out of bounded range")
    _require(bounds.get("acquisitions_required") == 2, "exactly two acquisitions are required")
    _require(bounds.get("timeout_seconds") in range(1, 121), "timeout_seconds out of range")
    _require(bounds.get("allow_redirects") is False, "redirects must fail closed")
    _require(bounds.get("require_identical_bytes") is True, "byte equality must be required")
    _require(bounds.get("require_xml_parse") is True, "XML parse must be required")

    claims = request.get("claims")
    _require(claims == ZERO_CLAIMS, "materialization request cannot grant scientific/training credit")
    return request


def _read_bounded(stream: BufferedIOBase | Any, max_bytes: int) -> bytes:
    data = stream.read(max_bytes + 1)
    _require(isinstance(data, bytes), "HTTP response body must be bytes")
    _require(len(data) <= max_bytes, "HTTP response exceeds max_response_bytes")
    return data


def _fetch_once(url: str, max_bytes: int, timeout_seconds: int) -> FetchResult:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/xml,text/xml;q=0.9,application/octet-stream;q=0.5",
            "User-Agent": "12-6-ai-d03-ecfr-materializer/1.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.getcode())
            final_url = response.geturl()
            content_type = response.headers.get_content_type().lower()
            body = _read_bounded(response, max_bytes)
    except urllib.error.HTTPError as exc:
        raise MaterializationError(f"HTTP error {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise MaterializationError(f"network error for {url}: {exc.reason}") from exc
    return FetchResult(status=status, final_url=final_url, content_type=content_type, body=body)


def _xml_stats(raw: bytes) -> dict[str, Any]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise MaterializationError(f"malformed XML: {exc}") from exc

    element_count = 0
    text_fragments: list[str] = []
    for element in root.iter():
        element_count += 1
        for fragment in (element.text, element.tail):
            if fragment:
                normalized = " ".join(fragment.split())
                if normalized:
                    text_fragments.append(normalized)

    normalized_text = "\n".join(text_fragments).encode("utf-8")
    return {
        "root_tag": root.tag,
        "element_count": element_count,
        "normalized_text_fragment_count": len(text_fragments),
        "normalized_text_bytes": len(normalized_text),
        "normalized_text_sha256": hashlib.sha256(normalized_text).hexdigest(),
    }


def materialize(
    request: dict[str, Any],
    *,
    fetcher: Callable[[str, int, int], FetchResult] = _fetch_once,
) -> tuple[bytes, dict[str, Any]]:
    validate_request(request)
    selection = request["selection"]
    bounds = request["bounds"]
    url = selection["url"]
    max_bytes = bounds["max_response_bytes"]
    timeout_seconds = bounds["timeout_seconds"]

    first = fetcher(url, max_bytes, timeout_seconds)
    second = fetcher(url, max_bytes, timeout_seconds)
    acquisitions = [first, second]
    for index, result in enumerate(acquisitions, start=1):
        _require(result.status == 200, f"acquisition {index} returned HTTP {result.status}")
        _require(result.final_url == url, f"acquisition {index} redirected or changed final URL")
        _require(
            result.content_type.lower() in ALLOWED_CONTENT_TYPES,
            f"acquisition {index} unexpected content type {result.content_type!r}",
        )
        _require(len(result.body) <= max_bytes, f"acquisition {index} exceeded byte bound")

    _require(first.body == second.body, "repeat acquisitions are not byte-identical")
    raw_sha256 = hashlib.sha256(first.body).hexdigest()
    stats = _xml_stats(first.body)
    acquisition_records = [
        {
            "ordinal": index,
            "http_status": result.status,
            "final_url": result.final_url,
            "content_type": result.content_type.lower(),
            "raw_bytes": len(result.body),
            "raw_sha256": hashlib.sha256(result.body).hexdigest(),
        }
        for index, result in enumerate(acquisitions, start=1)
    ]

    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "status": "TRANSPORT_AND_XML_PARSE_PASS_ZERO_CREDIT",
        "request_identity_sha256": request["request_identity_sha256"],
        "source_id": SOURCE_ID,
        "family_id": FAMILY_ID,
        "selection": copy.deepcopy(selection),
        "acquisitions": acquisition_records,
        "byte_identical": True,
        "raw_bytes": len(first.body),
        "raw_sha256": raw_sha256,
        "xml": stats,
        "record_extraction_status": "NOT_RUN",
        "rights_and_provenance_status": "NOT_RUN",
        "quality_language_privacy_status": "NOT_RUN",
        "global_dedup_status": "NOT_RUN",
        "evaluation_decontamination_status": "NOT_RUN",
        "claims": copy.deepcopy(ZERO_CLAIMS),
    }
    evidence["evidence_identity_sha256"] = _canonical_sha256(
        evidence, "evidence_identity_sha256"
    )
    return first.body, evidence


def load_request(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read request: {exc}") from exc
    validate_request(data)
    return data


def _safe_output_stem(request: dict[str, Any]) -> str:
    selection = request["selection"]
    return f"ecfr-{selection['date']}-title-{selection['title']}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "request",
        nargs="?",
        type=Path,
        default=Path("configs/data/d03_ecfr_point_in_time_request_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--retain-raw",
        action="store_true",
        help="Retain the verified XML locally; evidence JSON is always written.",
    )
    args = parser.parse_args()

    try:
        request = load_request(args.request)
        raw, evidence = materialize(request)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = _safe_output_stem(request)
        evidence_path = args.output_dir / f"{stem}.evidence.json"
        evidence_path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raw_path: Path | None = None
        if args.retain_raw:
            raw_path = args.output_dir / f"{stem}.xml"
            raw_path.write_bytes(raw)
    except (MaterializationError, OSError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 1

    print(
        json.dumps(
            {
                "valid": True,
                "status": evidence["status"],
                "evidence_path": str(evidence_path),
                "raw_path": str(raw_path) if raw_path is not None else None,
                "raw_bytes": evidence["raw_bytes"],
                "raw_sha256": evidence["raw_sha256"],
                "evidence_identity_sha256": evidence["evidence_identity_sha256"],
                "canonical_capacity_credit_bytes": 0,
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
