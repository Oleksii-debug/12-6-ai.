"""Successor cross-source dedup with sealed NIST PDF materialization.

NEXT100-065 deliberately excluded sources whose dedicated terminal workflows had
not completed at its refresh cutoff.  NEXT100-034/NIST became terminal later,
but its authority seals PDF byte identities plus deterministic normalized-text
hashes rather than committing the normalized payload itself.

This module composes the immutable NEXT100-065 inventory with late terminal
sources and materializes the exact NIST comparison payload before delegating to
the already-audited V3 lineage/dedup engine.  No model training is performed.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.data import cross_source_capacity_audit as v1
from twelve_six.data import cross_source_capacity_audit_v3 as v3

SCHEMA = "12-6.next100-076-cross-source-dedup.v4"
REPORT_SCHEMA = "12-6.next100-076-cross-source-dedup-report.v4"
ALGORITHM = "next100-076-v3-lineage-dedup-plus-sealed-materialization-v1"
NIST_POLICY = "NEXT100_034_NIST_PDF_BODY_NFKC_BOUNDED_V1"
MAX_NIST_NORMALIZED_BYTES = 20_000
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
BLANK_RE = re.compile(r"\n{3,}")


class CrossSourceV4Error(RuntimeError):
    """Fail-closed successor inventory/materialization error."""


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CrossSourceV4Error(f"{field} must be lowercase SHA-256")
    return value


def _validate_inventory_header(inventory: Mapping[str, Any]) -> None:
    if inventory.get("schema_version") != SCHEMA:
        raise CrossSourceV4Error("unsupported successor inventory schema")
    if inventory.get("local_free_only") is not True:
        raise CrossSourceV4Error("inventory must bind LOCAL_FREE only")
    if inventory.get("model_training_executed") is not False:
        raise CrossSourceV4Error("inventory must bind model_training_executed=false")
    base = inventory.get("base_inventory")
    if not isinstance(base, Mapping):
        raise CrossSourceV4Error("base_inventory must be a mapping")
    for key in ("acquisition_url", "expected_git_blob_sha1", "authority_head_sha"):
        value = base.get(key)
        if not isinstance(value, str) or not value:
            raise CrossSourceV4Error(f"base_inventory requires {key}")
    if re.fullmatch(r"[0-9a-f]{40}", str(base["expected_git_blob_sha1"])) is None:
        raise CrossSourceV4Error("base inventory Git blob identity is invalid")
    rows = inventory.get("additional_sources")
    if not isinstance(rows, list) or not rows:
        raise CrossSourceV4Error("successor inventory requires additional_sources")
    if not isinstance(inventory.get("additional_lineage_edges", []), list):
        raise CrossSourceV4Error("additional_lineage_edges must be a list")


def _load_base_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    _validate_inventory_header(inventory)
    base = inventory["base_inventory"]
    payload = v1.fetch_exact_source(str(base["acquisition_url"]))
    if v1._git_blob_sha1(payload) != base["expected_git_blob_sha1"]:
        raise CrossSourceV4Error("base inventory Git blob identity drift")
    try:
        decoded = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CrossSourceV4Error("base inventory is not canonical UTF-8 JSON") from exc
    if not isinstance(decoded, dict) or decoded.get("schema_version") != v3.INVENTORY_SCHEMA:
        raise CrossSourceV4Error("base inventory is not NEXT100-065 V3 authority")
    return decoded


def _validate_nist_row(row: Mapping[str, Any]) -> None:
    source_id = row.get("source_id")
    if row.get("comparison_normalization") != NIST_POLICY:
        raise CrossSourceV4Error(f"{source_id}: unsupported late-source materialization policy")
    if row.get("source_family") != "en.usgov.nist.technical-series":
        raise CrossSourceV4Error(f"{source_id}: unexpected NIST family identity")
    if row.get("modality") != "en" or row.get("evidence_status") != "DEDICATED_TERMINAL":
        raise CrossSourceV4Error(f"{source_id}: NIST source must be terminal English evidence")
    start_page = row.get("pdf_start_page")
    if not isinstance(start_page, int) or isinstance(start_page, bool) or start_page <= 0:
        raise CrossSourceV4Error(f"{source_id}: invalid pdf_start_page")
    expected_bytes = row.get("expected_comparison_bytes")
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or not (1 <= expected_bytes <= MAX_NIST_NORMALIZED_BYTES):
        raise CrossSourceV4Error(f"{source_id}: invalid expected comparison byte count")
    if row.get("declared_capacity_bytes") != expected_bytes:
        raise CrossSourceV4Error(f"{source_id}: declared capacity must equal sealed normalized bytes")
    _validate_sha256(row.get("expected_comparison_sha256"), field=f"{source_id}.expected_comparison_sha256")
    prefix = row.get("expected_extractor_version_prefix")
    if not isinstance(prefix, str) or not prefix.startswith("pdftotext version "):
        raise CrossSourceV4Error(f"{source_id}: extractor version must be pinned")


def _expand_inventory(inventory: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    _validate_inventory_header(inventory)
    base_sources = base.get("sources")
    base_edges = base.get("lineage_edges", [])
    if not isinstance(base_sources, list) or not isinstance(base_edges, list):
        raise CrossSourceV4Error("base inventory structure changed")

    additional = inventory["additional_sources"]
    for raw in additional:
        if not isinstance(raw, Mapping):
            raise CrossSourceV4Error("additional source must be a mapping")
        _validate_nist_row(raw)

    source_ids = [row.get("source_id") for row in base_sources + additional if isinstance(row, Mapping)]
    if len(source_ids) != len(set(source_ids)):
        raise CrossSourceV4Error("successor source IDs overlap the immutable base inventory")

    transformed_sources: list[dict[str, Any]] = [dict(row) for row in base_sources]
    for raw in additional:
        row = dict(raw)
        # V3 consumes the effective text payload.  Preserve upstream PDF identity
        # in the successor inventory/evidence, but bind V3's raw fields to the
        # sealed normalized bytes after materialization.
        row["expected_raw_bytes"] = row["expected_comparison_bytes"]
        row["expected_raw_sha256"] = row["expected_comparison_sha256"]
        row.pop("expected_git_blob_sha1", None)
        row.pop("comparison_normalization", None)
        row.pop("expected_comparison_bytes", None)
        row.pop("expected_comparison_sha256", None)
        row.pop("expected_extractor_version_prefix", None)
        row.pop("pdf_start_page", None)
        transformed_sources.append(row)

    return {
        "schema_version": v3.INVENTORY_SCHEMA,
        "worker_id": str(inventory.get("worker_id", "NEXT100-076-CORPUS-CONVERGENCE")),
        "local_free_only": True,
        "model_training_executed": False,
        "sources": transformed_sources,
        "lineage_edges": [dict(edge) for edge in base_edges]
        + [dict(edge) for edge in inventory.get("additional_lineage_edges", [])],
    }


def _normalize_nist_extracted(text: str) -> bytes:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = BLANK_RE.sub("\n\n", text).strip() + "\n"
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_NIST_NORMALIZED_BYTES:
        return encoded

    prefix = encoded[:MAX_NIST_NORMALIZED_BYTES]
    while True:
        try:
            candidate = prefix.decode("utf-8")
            break
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    cut = candidate.rfind("\n\n")
    if cut < 12_000:
        cut = candidate.rfind("\n")
    if cut < 12_000:
        raise CrossSourceV4Error("cannot find safe deterministic NIST truncation boundary")
    return (candidate[:cut].rstrip() + "\n").encode("utf-8")


def _pdftotext_version() -> str:
    result = subprocess.run(
        ["pdftotext", "-v"], text=True, capture_output=True, check=True
    )
    lines = (result.stderr or result.stdout).splitlines()
    if not lines:
        raise CrossSourceV4Error("pdftotext did not report a version")
    return lines[0].strip()


def _materialize_nist(row: Mapping[str, Any], pdf: bytes) -> tuple[bytes, dict[str, Any]]:
    source_id = str(row["source_id"])
    # Upstream PDF identity is the terminal authority boundary.
    original = dict(row)
    original.pop("expected_comparison_bytes", None)
    original.pop("expected_comparison_sha256", None)
    original.pop("expected_extractor_version_prefix", None)
    original.pop("comparison_normalization", None)
    original.pop("pdf_start_page", None)
    v1._verify_payload(original, pdf)
    if not pdf.startswith(b"%PDF-"):
        raise CrossSourceV4Error(f"{source_id}: upstream payload is not PDF")

    version = _pdftotext_version()
    expected_prefix = str(row["expected_extractor_version_prefix"])
    if not version.startswith(expected_prefix):
        raise CrossSourceV4Error(
            f"{source_id}: extractor drift: {version!r} does not match {expected_prefix!r}"
        )

    with tempfile.TemporaryDirectory(prefix="next100-076-nist-") as tmp:
        root = Path(tmp)
        pdf_path = root / "source.pdf"
        text_path = root / "source.txt"
        pdf_path.write_bytes(pdf)
        subprocess.run(
            [
                "pdftotext",
                "-f",
                str(row["pdf_start_page"]),
                "-nopgbrk",
                "-enc",
                "UTF-8",
                str(pdf_path),
                str(text_path),
            ],
            check=True,
        )
        normalized = _normalize_nist_extracted(text_path.read_text(encoding="utf-8"))

    expected_bytes = int(row["expected_comparison_bytes"])
    expected_sha = str(row["expected_comparison_sha256"])
    if len(normalized) != expected_bytes:
        raise CrossSourceV4Error(f"{source_id}: sealed normalized byte count drift")
    if v1._sha256(normalized) != expected_sha:
        raise CrossSourceV4Error(f"{source_id}: sealed normalized SHA-256 drift")

    return normalized, {
        "source_id": source_id,
        "policy": NIST_POLICY,
        "extractor_version": version,
        "upstream_raw_bytes": len(pdf),
        "upstream_raw_sha256": v1._sha256(pdf),
        "materialized_bytes": len(normalized),
        "materialized_sha256": v1._sha256(normalized),
    }


def audit_payloads(
    inventory: Mapping[str, Any],
    *,
    base_inventory: Mapping[str, Any],
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    expanded = _expand_inventory(inventory, base_inventory)
    late_rows = {str(row["source_id"]): row for row in inventory["additional_sources"]}
    expected_ids = {str(row["source_id"]) for row in expanded["sources"]}
    if set(payloads) != expected_ids:
        raise CrossSourceV4Error("payload coverage must equal the exact converged inventory")

    materialized = dict(payloads)
    evidence: list[dict[str, Any]] = []
    for source_id, row in late_rows.items():
        materialized[source_id], item = _materialize_nist(row, payloads[source_id])
        evidence.append(item)

    delegated = v3.audit_payloads(expanded, materialized)
    core = dict(delegated)
    core.pop("report_sha256", None)
    core["schema_version"] = REPORT_SCHEMA
    core["algorithm"] = ALGORITHM
    core["worker_id"] = str(inventory.get("worker_id", "NEXT100-076-CORPUS-CONVERGENCE"))
    core["base_inventory"] = dict(inventory["base_inventory"])
    core["late_terminal_materialization"] = sorted(evidence, key=lambda item: item["source_id"])
    core["matching_authority"] = (
        "NEXT100-065 V3 exact/near/fragment/code/lineage semantics after exact sealed "
        "NEXT100-034 NIST PDF-to-training-text materialization"
    )
    return {**core, "report_sha256": v1._sha256(v1._canonical_bytes(core))}


def audit_live(inventory: Mapping[str, Any]) -> dict[str, Any]:
    base = _load_base_inventory(inventory)
    expanded = _expand_inventory(inventory, base)
    late = {str(row["source_id"]): row for row in inventory["additional_sources"]}
    payloads: dict[str, bytes] = {}
    for row in expanded["sources"]:
        source_id = str(row["source_id"])
        if source_id in late:
            url = str(late[source_id]["acquisition_url"])
        else:
            url = str(row["acquisition_url"])
        payloads[source_id] = v1.fetch_exact_source(url)
    return audit_payloads(inventory, base_inventory=base, payloads=payloads)


def verify_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise CrossSourceV4Error("unsupported successor report schema")
    expected = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    if expected != v1._sha256(v1._canonical_bytes(core)):
        raise CrossSourceV4Error("successor report self-hash mismatch")
    if report.get("local_free_only") is not True:
        raise CrossSourceV4Error("LOCAL_FREE invariant failed")
    if report.get("model_training_executed") is not False:
        raise CrossSourceV4Error("model-training invariant failed")
    if report.get("raw_text_emitted") is not False:
        raise CrossSourceV4Error("raw text emission is forbidden")
    scope = report.get("terminal_candidates")
    if not isinstance(scope, Mapping):
        raise CrossSourceV4Error("terminal candidate summary missing")
    if scope.get("conservative_unique_capacity_bytes_after", 0) > scope.get("declared_capacity_bytes_before", -1):
        raise CrossSourceV4Error("deduplication inflated capacity")


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    verify_report(report)
    v1.write_report(report, path)
