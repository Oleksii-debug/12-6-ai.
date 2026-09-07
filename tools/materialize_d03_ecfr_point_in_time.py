#!/usr/bin/env python3
"""Materialize exact point-in-time eCFR title XML with zero training credit."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import validate_d03_ecfr_point_in_time_contract as contract

DEFAULT_CONFIG = contract.DEFAULT_CONFIG
REPORT_SCHEMA = "12-6.d03-ecfr-point-in-time-materialization-report.v1"
CHUNK_BYTES = 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")
XML_RUN = re.compile(r"\s+|[^\s]+", re.UNICODE)
FORBIDDEN_XML = (b"<!doctype", b"<!entity")
MaterializationError = contract.ContractError
EXPECTED_CONTRACT_IDENTITY = contract.EXPECTED_CONTRACT_IDENTITY


def _require(condition: bool, message: str) -> None:
    contract.require(condition, message)


def _report_identity(report: Mapping[str, Any]) -> str:
    core = copy.deepcopy(dict(report))
    core.pop("report_identity_sha256", None)
    return contract.sha256(contract.canonical_bytes(core))


def load_contract(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return contract.load_contract(path)


def _validate_final_url(expected_url: str, final_url: str, allowed_hosts: list[str]) -> None:
    expected = urllib.parse.urlsplit(expected_url)
    observed = urllib.parse.urlsplit(final_url)
    _require(observed.scheme == "https", "final URL must remain HTTPS")
    _require(observed.port in (None, 443), "final URL used a non-default HTTPS port")
    _require(observed.username is None and observed.password is None, "credentials in final URL")
    _require(observed.fragment == "", "fragment in final URL")
    _require((observed.hostname or "").lower() in allowed_hosts, "redirect left eCFR hosts")
    _require(observed.path == expected.path, "redirect changed exact historical path")
    _require(observed.query == expected.query, "redirect changed exact historical query")


def _stream_response(response: BinaryIO, destination: Path, max_bytes: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with destination.open("xb") as output:
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            _require(total <= max_bytes, "eCFR object exceeded byte cap")
            digest.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    _require(total > 0, "eCFR object was empty")
    return total, digest.hexdigest()


def _download_once(
    url: str,
    destination: Path,
    network: Mapping[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    _require(not destination.exists() and not destination.is_symlink(), "destination exists")
    request = urllib.request.Request(
        url,
        method="GET",
        headers={
            "User-Agent": network["user_agent"],
            "Accept": "application/xml,text/xml;q=0.9",
            "Accept-Encoding": network["accept_encoding"],
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            _require(status == 200, f"unexpected HTTP status: {status}")
            final_url = response.geturl()
            _validate_final_url(url, final_url, list(network["allowed_hosts"]))
            encoding = response.headers.get("Content-Encoding")
            _require(encoding in (None, "", "identity"), "content encoding must be identity")
            kind = str(response.headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
            _require(kind in network["allowed_content_types"], f"unexpected content type: {kind}")
            length = response.headers.get("Content-Length")
            declared = None
            if length is not None:
                try:
                    declared = int(length)
                except ValueError as exc:
                    raise MaterializationError("malformed Content-Length") from exc
                _require(0 < declared <= int(network["max_object_bytes"]), "Content-Length cap")
            size_bytes, sha256 = _stream_response(
                response,
                destination,
                int(network["max_object_bytes"]),
            )
            if declared is not None:
                _require(size_bytes == declared, "download size differs from Content-Length")
    except (OSError, urllib.error.URLError) as exc:
        destination.unlink(missing_ok=True)
        raise MaterializationError(f"eCFR acquisition failed: {exc}") from exc
    return {
        "url": url,
        "final_url": final_url,
        "content_type": kind,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _require_identical(first: Mapping[str, Any], second: Mapping[str, Any]) -> None:
    for field, label in (
        ("size_bytes", "size"),
        ("sha256", "SHA-256"),
        ("final_url", "final URL"),
        ("content_type", "content type"),
    ):
        _require(first.get(field) == second.get(field), f"double-fetch {label} mismatch")


def _scan_forbidden_xml(path: Path) -> None:
    tail = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            probe = (tail + chunk).lower()
            for marker in FORBIDDEN_XML:
                _require(marker not in probe, f"forbidden XML declaration: {marker.decode()}")
            tail = probe[-32:]


@dataclass
class _TextIdentityTarget:
    digest: Any
    byte_count: int = 0
    character_count: int = 0
    started: bool = False
    pending_space: bool = False

    def start(self, tag: str, attrs: Mapping[str, str]) -> None:
        del tag, attrs

    def end(self, tag: str) -> None:
        del tag

    def data(self, data: str) -> None:
        for match in XML_RUN.finditer(data):
            piece = match.group(0)
            if piece.isspace():
                self.pending_space = self.started
                continue
            if self.started and self.pending_space:
                self.digest.update(b" ")
                self.byte_count += 1
                self.character_count += 1
            raw = piece.encode("utf-8")
            self.digest.update(raw)
            self.byte_count += len(raw)
            self.character_count += len(piece)
            self.started = True
            self.pending_space = False

    def close(self) -> dict[str, Any]:
        return {
            "normalized_text_sha256": self.digest.hexdigest(),
            "normalized_text_utf8_bytes": self.byte_count,
            "normalized_text_characters": self.character_count,
        }


def extract_text_identity(path: Path) -> dict[str, Any]:
    _scan_forbidden_xml(path)
    parser = ET.XMLParser(target=_TextIdentityTarget(hashlib.sha256()))
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(CHUNK_BYTES):
                parser.feed(chunk)
        result = parser.close()
    except ET.ParseError as exc:
        raise MaterializationError(f"invalid XML payload: {exc}") from exc
    _require(result["normalized_text_utf8_bytes"] > 0, "XML yielded no character data")
    return result


def materialize(
    config: Mapping[str, Any],
    workspace: Path,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    network = config["network_policy"]
    effective_timeout = float(timeout if timeout is not None else network["timeout_seconds"])
    _require(effective_timeout > 0, "timeout must be positive")
    workspace.mkdir(parents=True, exist_ok=True)
    objects: list[dict[str, Any]] = []
    total_raw = 0
    total_text = 0
    for item in config["objects"]:
        stem = f"ecfr-{item['date']}-title-{item['title']}"
        first_path = workspace / f"{stem}-a.xml"
        second_path = workspace / f"{stem}-b.xml"
        first = _download_once(item["url"], first_path, network, timeout=effective_timeout)
        second = _download_once(item["url"], second_path, network, timeout=effective_timeout)
        _require_identical(first, second)
        text = extract_text_identity(first_path)
        total_raw += int(first["size_bytes"])
        total_text += int(text["normalized_text_utf8_bytes"])
        _require(total_raw <= int(network["max_total_bytes"]), "aggregate byte cap exceeded")
        second_path.unlink(missing_ok=True)
        objects.append({
            "date": item["date"],
            "title": item["title"],
            "url": item["url"],
            "final_url": first["final_url"],
            "content_type": first["content_type"],
            "raw_bytes": first["size_bytes"],
            "raw_sha256": first["sha256"],
            "two_byte_identical_acquisitions": True,
            **text,
        })
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "worker_id": contract.WORKER_ID,
        "execution_profile": "LOCAL_FREE",
        "contract_identity_sha256": config["contract_identity_sha256"],
        "parent": {
            "pr": contract.PARENT_PR,
            "head_sha": contract.PARENT_HEAD,
            "main_merge_sha": contract.PARENT_MAIN,
            "contract_identity_sha256": contract.PARENT_IDENTITY,
        },
        "source_id": contract.SOURCE_ID,
        "family_id": contract.FAMILY_ID,
        "objects": objects,
        "aggregate": {
            "object_count": len(objects),
            "raw_bytes": total_raw,
            "normalized_text_utf8_bytes": total_text,
        },
        "rights_boundary": {
            "decision": "REVIEW_REQUIRED_ZERO_CREDIT",
            "rights_and_provenance_complete": False,
            "automatic_training_eligibility": False,
        },
        "claim_boundary": copy.deepcopy(config["claim_boundary"]),
        "raw_text_emitted_in_report": False,
        "next_gate": "RIGHTS_AND_PROVENANCE_CLASSIFICATION",
    }
    report["report_identity_sha256"] = _report_identity(report)
    return report


def verify_report(config: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == REPORT_SCHEMA, "report schema drift")
    _require(report.get("worker_id") == contract.WORKER_ID, "report worker drift")
    _require(report.get("execution_profile") == "LOCAL_FREE", "report profile drift")
    _require(
        report.get("contract_identity_sha256") == config["contract_identity_sha256"],
        "report contract drift",
    )
    identity = report.get("report_identity_sha256")
    _require(isinstance(identity, str) and HEX64.fullmatch(identity) is not None, "report hash")
    _require(identity == _report_identity(report), "report self-hash mismatch")
    _require(report.get("source_id") == contract.SOURCE_ID, "report source drift")
    _require(report.get("family_id") == contract.FAMILY_ID, "report family drift")
    _require(report.get("raw_text_emitted_in_report") is False, "raw text report forbidden")
    objects = report.get("objects")
    _require(isinstance(objects, list) and len(objects) == len(config["objects"]), "object count")
    for expected, observed in zip(config["objects"], objects, strict=True):
        for field in ("date", "title", "url"):
            _require(observed.get(field) == expected[field], f"report {field} drift")
        _require(observed.get("two_byte_identical_acquisitions") is True, "double-fetch proof")
        _require(int(observed.get("raw_bytes", 0)) > 0, "raw bytes missing")
        _require(HEX64.fullmatch(str(observed.get("raw_sha256", ""))) is not None, "raw hash")
        _require(int(observed.get("normalized_text_utf8_bytes", 0)) > 0, "text bytes missing")
        _require(
            HEX64.fullmatch(str(observed.get("normalized_text_sha256", ""))) is not None,
            "text hash",
        )
    _require(report.get("claim_boundary") == config["claim_boundary"], "claim boundary drift")
    _require(report.get("next_gate") == "RIGHTS_AND_PROVENANCE_CLASSIFICATION", "next gate")


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temp_name = handle.name
    os.replace(temp_name, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--verify-report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_contract(args.config)
        if args.verify_report is not None:
            verify_report(config, json.loads(args.verify_report.read_text(encoding="utf-8")))
            print("PASS_ECFR_POINT_IN_TIME_REPORT")
            return 0
        _require(args.workspace is not None, "--workspace is required")
        _require(args.report_output is not None, "--report-output is required")
        report = materialize(config, args.workspace, timeout=args.timeout)
        verify_report(config, report)
        _write_report(args.report_output, report)
    except (MaterializationError, OSError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("D03_ECFR_POINT_IN_TIME=MATERIALIZED_ZERO_CREDIT")
    print("REPORT_SHA256=" + report["report_identity_sha256"])
    print("RAW_BYTES=" + str(report["aggregate"]["raw_bytes"]))
    print("NORMALIZED_TEXT_BYTES=" + str(report["aggregate"]["normalized_text_utf8_bytes"]))
    print("TRAINING_AUTHORIZED_BYTES=0")
    print("NEXT=RIGHTS_AND_PROVENANCE_CLASSIFICATION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
