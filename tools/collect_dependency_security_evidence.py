#!/usr/bin/env python3
"""Collect current OSV/PyPI evidence for the exact committed dependency locks."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from twelve_six.integration.dependency_lock import canonical_json_bytes
from twelve_six.integration.dependency_security import (
    build_lock_sbom,
    build_security_evidence,
    unique_components,
    write_json,
)

OSV_BATCH_URL = "https://api.osv.dev/v1/querybatch"
PYPI_BASE_URL = "https://pypi.org/pypi"
USER_AGENT = "12-6-ai-dependency-evidence/1"


class EvidenceCollectionError(RuntimeError):
    """Raised when an external evidence source cannot be queried completely."""


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _request_json(url: str, *, payload: dict[str, Any] | None = None) -> tuple[Any, str]:
    data = None
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read()
            return json.loads(raw), hashlib.sha256(raw).hexdigest()
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(attempt)
    raise EvidenceCollectionError(f"failed to query dependency evidence source: {url}") from last_error


def _collect_osv(components: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], str]:
    queries = [
        {
            "package": {"ecosystem": "PyPI", "name": component["name"]},
            "version": component["version"],
        }
        for component in components
    ]
    response, response_sha256 = _request_json(OSV_BATCH_URL, payload={"queries": queries})
    if not isinstance(response, dict) or not isinstance(response.get("results"), list):
        raise EvidenceCollectionError("OSV querybatch returned an invalid response")
    results = response["results"]
    if len(results) != len(components):
        raise EvidenceCollectionError("OSV querybatch result count mismatch")
    records: dict[str, dict[str, Any]] = {}
    for component, result in zip(components, results, strict=True):
        if not isinstance(result, dict):
            raise EvidenceCollectionError("OSV result must be an object")
        vulnerabilities = result.get("vulns", [])
        if not isinstance(vulnerabilities, list):
            raise EvidenceCollectionError("OSV vulnerabilities must be a list")
        compact: list[dict[str, Any]] = []
        for vulnerability in vulnerabilities:
            if not isinstance(vulnerability, dict) or not isinstance(vulnerability.get("id"), str):
                raise EvidenceCollectionError("OSV vulnerability entry is malformed")
            compact.append(
                {
                    "id": vulnerability["id"],
                    "aliases": sorted(
                        alias for alias in vulnerability.get("aliases", []) if isinstance(alias, str)
                    ),
                    "modified": vulnerability.get("modified"),
                }
            )
        compact.sort(key=lambda item: item["id"])
        records[component["key"]] = {
            "status": "QUERIED",
            "response_sha256": _json_sha256(result),
            "vulnerabilities": compact,
        }
    return records, response_sha256


def _collect_pypi(components: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], str]:
    records: dict[str, dict[str, Any]] = {}
    response_hashes: list[str] = []
    for component in components:
        name = urllib.parse.quote(component["name"], safe="")
        version = urllib.parse.quote(component["version"], safe="")
        response, response_sha256 = _request_json(f"{PYPI_BASE_URL}/{name}/{version}/json")
        response_hashes.append(response_sha256)
        if not isinstance(response, dict) or not isinstance(response.get("info"), dict):
            raise EvidenceCollectionError("PyPI metadata response is malformed")
        info = response["info"]
        expression = info.get("license_expression")
        if not isinstance(expression, str) or not expression.strip():
            expression = None
        license_field = info.get("license")
        if not isinstance(license_field, str) or not license_field.strip():
            license_field = None
        classifiers = sorted(
            classifier
            for classifier in info.get("classifiers", [])
            if isinstance(classifier, str) and classifier.startswith("License ::")
        )
        declared = expression is not None or license_field is not None or bool(classifiers)
        records[component["key"]] = {
            "status": "DECLARED" if declared else "UNRESOLVED",
            "license_expression": expression,
            "license_field_present": license_field is not None,
            "license_field_sha256": (
                hashlib.sha256(license_field.encode()).hexdigest() if license_field is not None else None
            ),
            "license_classifiers": classifiers,
            "metadata_sha256": response_sha256,
        }
    return records, hashlib.sha256("".join(sorted(response_hashes)).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--sbom-out", type=Path, required=True)
    parser.add_argument("--evidence-out", type=Path, required=True)
    parser.add_argument("--fail-on-review-required", action="store_true")
    args = parser.parse_args()

    sbom = build_lock_sbom(root=args.root, source_sha=args.source_sha)
    components = unique_components(sbom)
    osv_records, osv_batch_sha256 = _collect_osv(components)
    pypi_records, pypi_set_sha256 = _collect_pypi(components)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    evidence = build_security_evidence(
        sbom=sbom,
        generated_at=generated_at,
        license_records=pypi_records,
        advisory_records=osv_records,
        scan_sources={
            "osv": {
                "status": "SUCCESS",
                "endpoint": OSV_BATCH_URL,
                "retrieved_at": generated_at,
                "batch_response_sha256": osv_batch_sha256,
            },
            "pypi": {
                "status": "SUCCESS",
                "endpoint": PYPI_BASE_URL,
                "retrieved_at": generated_at,
                "response_set_sha256": pypi_set_sha256,
            },
        },
    )
    write_json(args.sbom_out, sbom)
    write_json(args.evidence_out, evidence)
    print(f"sbom_sha256={sbom['sbom_sha256']}")
    print(f"evidence_sha256={evidence['evidence_sha256']}")
    print(f"status={evidence['status']}")
    if args.fail_on_review_required and evidence["status"] != "EVIDENCE_COMPLETE_NO_REVIEW_FINDINGS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
