#!/usr/bin/env python3
"""Deterministic local-only eCFR section extraction for D03."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/data/d03_ecfr_section_extraction_v1.json")
EXPECTED_REQUEST_PATH = Path("configs/data/d03_ecfr_point_in_time_request_v1.json")
REQUEST_PATH = EXPECTED_REQUEST_PATH
EXPECTED_REQUEST_BLOB_SHA1 = "92fb471260020dffc3b9d53ca98581cf700b0916"
REQUEST_ID = "d9955ff11713a358164513d912d39ba107f0073c75bed15be64afe2cc8c3a2e4"
SOURCE_ID = "en.us.ecfr.regulations"
FAMILY_ID = "us.federal-regulations.ecfr"
MAX_INPUT_BYTES = 8 * 1024 * 1024
FORBIDDEN_DECL_TEXT_RE = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
XML_DECL_ENCODING_RE = re.compile(
    r"^\ufeff?\s*<\?xml\b[^>]*\bencoding\s*=\s*['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)

ROOT_KEYS = {
    "schema_version", "execution_profile", "project_authority",
    "source_binding", "xml_contract", "output_contract", "truth_boundary",
}
AUTH_KEYS = {
    "swarm_control_issue", "ownership_issue", "parent_issue", "source_request_pr",
    "request_path", "request_git_blob_sha1",
}
SOURCE_KEYS = {"request_identity_sha256", "date", "title", "url", "source_id", "family_id"}
XML_KEYS = {
    "official_guide_url", "root_tag", "section_tag", "section_type",
    "required_section_attributes", "ancestor_tags", "max_input_bytes",
    "forbid_dtd_or_entity_declarations",
}
OUTPUT_KEYS = {"record_schema", "summary_schema", "order", "json"}
TRUTH_KEYS = {
    "real_data_executed", "payload_extracted", "source_rights_decision_made",
    "canonical_capacity_credit_bytes", "family_credit", "training_authorized_bytes",
    "authorized_unique_loss_positions", "tokenizer_fit_authorized",
    "model_training_executed", "optimizer_updates", "final_test_accessed",
    "paid_compute_used", "foreign_pretrained_weights",
}
ZERO_INT_KEYS = {
    "canonical_capacity_credit_bytes", "family_credit", "training_authorized_bytes",
    "authorized_unique_loss_positions", "optimizer_updates",
}
EXPECTED_SOURCE = {
    "request_identity_sha256": REQUEST_ID,
    "date": "2026-09-03",
    "title": 1,
    "url": "https://www.ecfr.gov/api/versioner/v1/full/2026-09-03/title-1.xml",
    "source_id": SOURCE_ID,
    "family_id": FAMILY_ID,
}
EXPECTED_XML = {
    "official_guide_url": "https://www.govinfo.gov/bulkdata/ECFR/resources/ECFR-XML-User-Guide.pdf",
    "root_tag": "DLPSTEXTCLASS",
    "section_tag": "DIV8",
    "section_type": "SECTION",
    "required_section_attributes": ["N", "NODE"],
    "ancestor_tags": ["DIV1", "DIV2", "DIV3", "DIV4", "DIV5", "DIV6", "DIV7"],
    "max_input_bytes": MAX_INPUT_BYTES,
    "forbid_dtd_or_entity_declarations": True,
}
EXPECTED_OUTPUT = {
    "record_schema": "12-6.d03-ecfr-section-record.v1",
    "summary_schema": "12-6.d03-ecfr-section-extraction-summary.v1",
    "order": "DOCUMENT_ORDER",
    "json": "UTF8_CANONICAL_JSONL",
}


class ExtractionError(ValueError):
    """Raised when extraction authority or payload fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExtractionError(message)


def _keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    _require(type(value) is dict, f"{label} must be an object")
    assert isinstance(value, dict)
    _require(set(value) == expected, f"{label} keys must be exact")
    return value


def _strict_int(value: object, label: str, expected: int | None = None) -> int:
    _require(type(value) is int, f"{label} must be a strict integer")
    assert type(value) is int
    _require(value >= 0, f"{label} must be non-negative")
    if expected is not None:
        _require(value == expected, f"{label} mismatch")
    return value


def _strict_bool(value: object, label: str, expected: bool) -> bool:
    _require(type(value) is bool and value is expected, f"{label} mismatch")
    assert type(value) is bool
    return value


def _git_blob_sha1(raw: bytes) -> str:
    prefix = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(prefix + raw).hexdigest()  # noqa: S324


def _validate_request() -> dict[str, Any]:
    _require(REQUEST_PATH == EXPECTED_REQUEST_PATH, "request path substitution")
    raw = REQUEST_PATH.read_bytes()
    _require(_git_blob_sha1(raw) == EXPECTED_REQUEST_BLOB_SHA1, "request Git blob mismatch")
    request = json.loads(raw)
    _require(type(request) is dict, "request must be an object")
    assert isinstance(request, dict)
    _require(request.get("request_identity_sha256") == REQUEST_ID, "request identity drift")
    selection = request.get("selection")
    _require(type(selection) is dict, "request selection missing")
    assert isinstance(selection, dict)
    _require(selection.get("date") == "2026-09-03", "request date drift")
    _strict_int(selection.get("title"), "request title", 1)
    _require(selection.get("url") == EXPECTED_SOURCE["url"], "request URL drift")
    authority = request.get("authority")
    _require(type(authority) is dict, "request authority missing")
    assert isinstance(authority, dict)
    _require(authority.get("source_id") == SOURCE_ID, "request source drift")
    _require(authority.get("family_id") == FAMILY_ID, "request family drift")
    return request


def validate_config(config: dict[str, Any], *, verify_request: bool = True) -> dict[str, Any]:
    root = _keys(config, ROOT_KEYS, "config root")
    _require(root["schema_version"] == "12-6.d03-ecfr-section-extraction.v1", "schema drift")
    _require(root["execution_profile"] == "LOCAL_FREE", "execution profile drift")

    auth = _keys(root["project_authority"], AUTH_KEYS, "project_authority")
    expected_auth = {
        "swarm_control_issue": 723, "ownership_issue": 1121, "parent_issue": 672,
        "source_request_pr": 707, "request_path": str(EXPECTED_REQUEST_PATH),
        "request_git_blob_sha1": EXPECTED_REQUEST_BLOB_SHA1,
    }
    for key, expected in expected_auth.items():
        if type(expected) is int:
            _strict_int(auth[key], f"project_authority.{key}", expected)
        else:
            _require(auth[key] == expected, f"project_authority.{key} drift")

    source = _keys(root["source_binding"], SOURCE_KEYS, "source_binding")
    for key, expected in EXPECTED_SOURCE.items():
        if key == "title":
            _strict_int(source[key], "source_binding.title", expected)
        else:
            _require(source[key] == expected, f"source_binding.{key} drift")

    xml = _keys(root["xml_contract"], XML_KEYS, "xml_contract")
    for key, expected in EXPECTED_XML.items():
        if key == "max_input_bytes":
            _strict_int(xml[key], f"xml_contract.{key}", expected)
        elif type(expected) is bool:
            _strict_bool(xml[key], f"xml_contract.{key}", expected)
        else:
            _require(xml[key] == expected, f"xml_contract.{key} drift")

    output = _keys(root["output_contract"], OUTPUT_KEYS, "output_contract")
    _require(output == EXPECTED_OUTPUT, "output contract drift")

    truth = _keys(root["truth_boundary"], TRUTH_KEYS, "truth_boundary")
    for key in ZERO_INT_KEYS:
        _strict_int(truth[key], f"truth_boundary.{key}", 0)
    for key in TRUTH_KEYS - ZERO_INT_KEYS:
        _strict_bool(truth[key], f"truth_boundary.{key}", False)

    if verify_request:
        _validate_request()
    return config


def load_config(path: Path = CONFIG_PATH, *, verify_request: bool = True) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExtractionError("config is invalid JSON") from exc
    _require(type(config) is dict, "config must be an object")
    assert isinstance(config, dict)
    return validate_config(config, verify_request=verify_request)


def _norm(text: str | None) -> str:
    return " ".join((text or "").split())


def _decode_xml_text(raw: bytes) -> str:
    _require(b"\x00" not in raw, "XML input must be canonical UTF-8 without NUL bytes")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExtractionError("XML input must be canonical UTF-8") from exc
    declaration = XML_DECL_ENCODING_RE.search(text)
    if declaration is not None:
        encoding = declaration.group(1).lower().replace("_", "-")
        _require(encoding in {"utf-8", "utf8"}, "XML declaration must specify UTF-8")
    _require(
        FORBIDDEN_DECL_TEXT_RE.search(text) is None,
        "DTD/entity declarations are prohibited",
    )
    return text


def _head(element: ET.Element) -> str:
    direct = [child for child in element if child.tag == "HEAD"]
    _require(len(direct) <= 1, "multiple direct HEAD elements")
    return _norm(" ".join(direct[0].itertext())) if direct else ""


def _structural_ancestors(
    element: ET.Element,
    parent: dict[ET.Element, ET.Element],
    root: ET.Element,
) -> list[ET.Element]:
    allowed = set(EXPECTED_XML["ancestor_tags"])
    direct_parent = parent.get(element)
    _require(
        direct_parent is not None and direct_parent.tag in allowed,
        "DIV8 direct parent must be DIV1..DIV7",
    )

    chain: list[ET.Element] = []
    current = direct_parent
    while current is not root:
        _require(current.tag in allowed, "section ancestor chain must contain only DIV1..DIV7")
        chain.append(current)
        current = parent.get(current)
        _require(current is not None, "section ancestor chain does not terminate at title root")
    chain.reverse()
    _require(bool(chain), "section structural ancestry is empty")
    levels = [int(node.tag[3:]) for node in chain]
    _require(
        all(left < right for left, right in zip(levels, levels[1:])),
        "section DIV ancestor levels must be strictly increasing",
    )
    return chain


def _ancestor_provenance(chain: list[ET.Element]) -> list[dict[str, Any]]:
    return [
        {
            "tag": element.tag,
            "type": element.attrib.get("TYPE"),
            "n": element.attrib.get("N"),
            "node": element.attrib.get("NODE"),
            "head": _head(element),
        }
        for element in chain
    ]


def extract_sections(raw: bytes, config: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    validate_config(config)
    _require(type(raw) is bytes, "raw XML must be bytes")
    _require(0 < len(raw) <= MAX_INPUT_BYTES, "raw XML byte bound violated")
    xml_text = _decode_xml_text(raw)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ExtractionError(f"malformed XML: {exc}") from exc
    _require(root.tag == "DLPSTEXTCLASS", "unexpected eCFR root tag")

    parent = {child: node for node in root.iter() for child in node}
    records: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    seen_ids: set[str] = set()

    for element in root.iter():
        if element.tag != "DIV8":
            continue
        _require(element.attrib.get("TYPE") == "SECTION", "DIV8 must have TYPE=SECTION")
        ancestors = _structural_ancestors(element, parent, root)
        number = _norm(element.attrib.get("N"))
        node = _norm(element.attrib.get("NODE"))
        _require(bool(number), "section N missing")
        _require(bool(node), "section NODE missing")
        _require(node not in seen_nodes, "duplicate section NODE")
        seen_nodes.add(node)

        fragments = [_norm(fragment) for fragment in element.itertext()]
        text = "\n".join(fragment for fragment in fragments if fragment)
        _require(bool(text), "section normalized text is empty")
        text_raw = text.encode("utf-8")
        record_id = hashlib.sha256(
            f"{REQUEST_ID}\0{node}\0{number}".encode("utf-8")
        ).hexdigest()
        _require(record_id not in seen_ids, "record id collision")
        seen_ids.add(record_id)
        records.append(
            {
                "schema": EXPECTED_OUTPUT["record_schema"],
                "record_id": record_id,
                "request_identity_sha256": REQUEST_ID,
                "source_id": SOURCE_ID,
                "family_id": FAMILY_ID,
                "date": EXPECTED_SOURCE["date"],
                "title": EXPECTED_SOURCE["title"],
                "section_n": number,
                "node": node,
                "head": _head(element),
                "ancestors": _ancestor_provenance(ancestors),
                "embedded_in_ecfr_xml": True,
                "normalized_text": text,
                "normalized_text_bytes": len(text_raw),
                "normalized_text_sha256": hashlib.sha256(text_raw).hexdigest(),
                "source_rights_decision": "NOT_MADE",
                "training_eligible": False,
            }
        )

    _require(bool(records), "no DIV8 TYPE=SECTION records found")
    lines = [
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for record in records
    ]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    summary = {
        "schema": EXPECTED_OUTPUT["summary_schema"],
        "request_identity_sha256": REQUEST_ID,
        "input_raw_bytes": len(raw),
        "input_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "record_count": len(records),
        "normalized_text_bytes": sum(r["normalized_text_bytes"] for r in records),
        "jsonl_bytes": len(payload),
        "jsonl_sha256": hashlib.sha256(payload).hexdigest(),
        "source_rights_decision_made": False,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
    }
    return payload, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    config = load_config(args.config)
    payload, summary = extract_sections(args.xml.read_bytes(), config)
    args.out_jsonl.write_bytes(payload)
    args.out_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
