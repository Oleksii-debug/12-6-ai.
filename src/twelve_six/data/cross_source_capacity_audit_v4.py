"""NEXT100-065B converged cross-source deduplication V4.

V4 deliberately reuses the terminal V3 duplicate/lineage engine.  Its new job is
source convergence: bind the exact late terminal source objects and materialize
source-specific comparison payloads (notably the NIST PDF-body policy) before
V3 is allowed to evaluate the full 21-object vector.
"""
from __future__ import annotations

import copy
import hashlib
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

SCHEMA = "12-6.next100-065b-cross-source-dedup-report.v4"
INVENTORY_SCHEMA = "12-6.next100-065b-cross-source-dedup.v4"
NIST_POLICY = "NEXT100_034_NIST_PDF_BODY_NFKC_BOUNDED_V1"
MAX_NIST_BYTES = 20_000
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
BLANK_RE = re.compile(r"\n{3,}")


class CrossSourceV4Error(RuntimeError):
    """Fail-closed V4 convergence or source-materialization error."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _normalize_nist_extracted(text: str) -> bytes:
    """Exact NEXT100-034 deterministic bounded PDF-body normalization."""

    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = BLANK_RE.sub("\n\n", text).strip() + "\n"
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_NIST_BYTES:
        return encoded

    prefix = encoded[:MAX_NIST_BYTES]
    while True:
        try:
            candidate = prefix.decode("utf-8")
            break
        except UnicodeDecodeError:
            prefix = prefix[:-1]
            if not prefix:
                raise CrossSourceV4Error("NIST UTF-8 truncation exhausted payload")
    cut = candidate.rfind("\n\n")
    if cut < 12_000:
        cut = candidate.rfind("\n")
    if cut < 12_000:
        raise CrossSourceV4Error("NIST deterministic truncation boundary missing")
    return (candidate[:cut].rstrip() + "\n").encode("utf-8")


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise CrossSourceV4Error(f"{field} must be lowercase SHA-256")
    return value


def _validate_extension(
    base_inventory: Mapping[str, Any], extension: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if extension.get("schema_version") != INVENTORY_SCHEMA:
        raise CrossSourceV4Error("unsupported V4 inventory schema")
    for flag in ("local_free_only",):
        if extension.get(flag) is not True:
            raise CrossSourceV4Error(f"{flag} must be true")
    for flag in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
    ):
        if extension.get(flag) is not False:
            raise CrossSourceV4Error(f"{flag} must be false")
    if base_inventory.get("schema_version") != v3.INVENTORY_SCHEMA:
        raise CrossSourceV4Error("base inventory is not NEXT100-065 V3")

    base_meta = extension.get("base_v3")
    convergence = extension.get("registry_convergence")
    if not isinstance(base_meta, Mapping) or not isinstance(convergence, Mapping):
        raise CrossSourceV4Error("base/convergence authority bindings are required")
    if base_meta.get("head_sha") != "efc278cec0e4773eb4ff405bf4b4d24ee63b5d13":
        raise CrossSourceV4Error("unexpected NEXT100-065 V3 authority head")
    if convergence.get("head_sha") != "9ad8f74b12a2e991b7934356a88dd9a1f6ff3f41":
        raise CrossSourceV4Error("unexpected NEXT100-063 convergence authority head")

    additions_raw = extension.get("additions")
    if not isinstance(additions_raw, list) or not additions_raw:
        raise CrossSourceV4Error("V4 additions must be a non-empty list")
    additions: list[dict[str, Any]] = []
    seen = {row.get("source_id") for row in base_inventory.get("sources", [])}
    for item in additions_raw:
        if not isinstance(item, Mapping):
            raise CrossSourceV4Error("addition row must be an object")
        row = dict(item)
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            raise CrossSourceV4Error("source IDs must be non-empty and globally unique")
        seen.add(source_id)
        if row.get("evidence_status") != "DEDICATED_TERMINAL":
            raise CrossSourceV4Error(f"{source_id}: only terminal dedicated evidence is creditable")
        capacity = row.get("declared_capacity_bytes")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise CrossSourceV4Error(f"{source_id}: invalid declared capacity")
        for field in ("source_family", "stable_origin_id", "stable_object_id", "modality", "authority_ref", "acquisition_url"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise CrossSourceV4Error(f"{source_id}: missing {field}")
        if row.get("comparison_normalization") == NIST_POLICY:
            if row.get("modality") != "en":
                raise CrossSourceV4Error(f"{source_id}: NIST policy requires English modality")
            start_page = row.get("pdf_start_page")
            if not isinstance(start_page, int) or isinstance(start_page, bool) or start_page <= 0:
                raise CrossSourceV4Error(f"{source_id}: invalid pdf_start_page")
            for field in ("expected_pdf_sha256", "expected_comparison_sha256"):
                _validate_sha256(row.get(field), field=f"{source_id}.{field}")
            for field in ("expected_pdf_bytes", "expected_comparison_bytes"):
                value = row.get(field)
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise CrossSourceV4Error(f"{source_id}: invalid {field}")
            if row["expected_comparison_bytes"] != capacity:
                raise CrossSourceV4Error(f"{source_id}: NIST capacity must equal bounded normalized bytes")
        else:
            if "comparison_normalization" in row:
                raise CrossSourceV4Error(f"{source_id}: unsupported comparison normalization")
            _validate_sha256(row.get("expected_raw_sha256"), field=f"{source_id}.expected_raw_sha256")
            raw_bytes = row.get("expected_raw_bytes")
            if not isinstance(raw_bytes, int) or isinstance(raw_bytes, bool) or raw_bytes <= 0:
                raise CrossSourceV4Error(f"{source_id}: invalid expected_raw_bytes")
        additions.append(row)

    expected = extension.get("expected_pre_dedup")
    if not isinstance(expected, Mapping):
        raise CrossSourceV4Error("expected_pre_dedup binding is required")
    base_rows = base_inventory.get("sources")
    if not isinstance(base_rows, list):
        raise CrossSourceV4Error("base source list missing")
    if len(base_rows) + len(additions) != expected.get("source_object_count"):
        raise CrossSourceV4Error("pre-dedup source object count drift")

    by_modality = {"uk": 0, "en": 0, "code": 0}
    for row in [*base_rows, *additions]:
        modality = row.get("modality")
        if modality not in by_modality:
            raise CrossSourceV4Error(f"unsupported modality: {modality!r}")
        by_modality[modality] += int(row["declared_capacity_bytes"])
    bound_capacity = expected.get("capacity_bytes")
    if not isinstance(bound_capacity, Mapping):
        raise CrossSourceV4Error("expected capacity binding missing")
    if any(by_modality[key] != bound_capacity.get(key) for key in by_modality):
        raise CrossSourceV4Error(
            f"pre-dedup capacity drift: computed={by_modality}, bound={dict(bound_capacity)}"
        )
    if sum(by_modality.values()) != bound_capacity.get("total"):
        raise CrossSourceV4Error("pre-dedup total capacity drift")
    return additions


def _nist_payload(row: Mapping[str, Any], raw_pdf: bytes) -> tuple[bytes, dict[str, Any]]:
    source_id = str(row["source_id"])
    if len(raw_pdf) != row["expected_pdf_bytes"] or _sha256(raw_pdf) != row["expected_pdf_sha256"]:
        raise CrossSourceV4Error(f"{source_id}: upstream NIST PDF byte identity drift")
    if not raw_pdf.startswith(b"%PDF-"):
        raise CrossSourceV4Error(f"{source_id}: acquired payload is not PDF")

    with tempfile.TemporaryDirectory(prefix="next100-065b-nist-") as tmp:
        root = Path(tmp)
        pdf_path = root / "source.pdf"
        txt_path = root / "source.txt"
        pdf_path.write_bytes(raw_pdf)
        try:
            completed = subprocess.run(
                [
                    "pdftotext",
                    "-f",
                    str(row["pdf_start_page"]),
                    "-nopgbrk",
                    "-enc",
                    "UTF-8",
                    str(pdf_path),
                    str(txt_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise CrossSourceV4Error(f"{source_id}: pdftotext extraction failed") from exc
        del completed
        try:
            extracted = txt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise CrossSourceV4Error(f"{source_id}: pdftotext output is not UTF-8") from exc

    normalized = _normalize_nist_extracted(extracted)
    if len(normalized) != row["expected_comparison_bytes"]:
        raise CrossSourceV4Error(f"{source_id}: NIST normalized byte-count drift")
    if _sha256(normalized) != row["expected_comparison_sha256"]:
        raise CrossSourceV4Error(f"{source_id}: NIST normalization identity drift")
    return normalized, {
        "source_id": source_id,
        "pdf_bytes": len(raw_pdf),
        "pdf_sha256": _sha256(raw_pdf),
        "comparison_bytes": len(normalized),
        "comparison_sha256": _sha256(normalized),
        "normalization_policy": NIST_POLICY,
        "pdf_start_page": row["pdf_start_page"],
    }


def _v3_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    if row.get("comparison_normalization") != NIST_POLICY:
        return result
    result["expected_raw_bytes"] = row["expected_comparison_bytes"]
    result["expected_raw_sha256"] = row["expected_comparison_sha256"]
    for key in (
        "expected_pdf_bytes",
        "expected_pdf_sha256",
        "comparison_normalization",
        "pdf_start_page",
        "expected_comparison_bytes",
        "expected_comparison_sha256",
    ):
        result.pop(key, None)
    # V3 audit_payloads consumes supplied bytes and never re-opens this URL.  Keep
    # a descriptive immutable pseudo-origin rather than pretending normalized
    # bytes are the upstream PDF acquisition URL.
    result["acquisition_url"] = f"materialized-v4://{result['source_id']}"
    return result


def compose_v3_inventory(
    base_inventory: Mapping[str, Any], extension: Mapping[str, Any]
) -> dict[str, Any]:
    additions = _validate_extension(base_inventory, extension)
    merged = copy.deepcopy(dict(base_inventory))
    merged["sources"] = [
        *copy.deepcopy(list(base_inventory["sources"])),
        *[_v3_row(row) for row in additions],
    ]
    merged["final_refresh_required"] = False
    merged["terminal_refresh_cutoff_utc"] = "2026-08-26T19:28:45Z"
    merged["terminal_refresh_rule"] = (
        "NEXT100-065B consumes exact terminal authorities frozen by NEXT100-063; "
        "uncredited queued/RETEST/PROBE sources remain excluded."
    )
    return merged


def audit_live(
    base_inventory: Mapping[str, Any], extension: Mapping[str, Any]
) -> dict[str, Any]:
    additions = _validate_extension(base_inventory, extension)
    merged = compose_v3_inventory(base_inventory, extension)
    payloads: dict[str, bytes] = {}
    acquisition_evidence: list[dict[str, Any]] = []

    for row in base_inventory["sources"]:
        payloads[row["source_id"]] = v1.fetch_exact_source(row["acquisition_url"])
    for row in additions:
        raw = v1.fetch_exact_source(row["acquisition_url"])
        if row.get("comparison_normalization") == NIST_POLICY:
            materialized, evidence = _nist_payload(row, raw)
            payloads[row["source_id"]] = materialized
            acquisition_evidence.append(evidence)
        else:
            payloads[row["source_id"]] = raw
            acquisition_evidence.append(
                {
                    "source_id": row["source_id"],
                    "raw_bytes": len(raw),
                    "raw_sha256": _sha256(raw),
                    "normalization_policy": "V3_GENERIC_FROM_VERIFIED_RAW",
                }
            )

    dedup = v3.audit_payloads(merged, payloads)
    v3.verify_report(dedup)
    core = {
        "schema_version": SCHEMA,
        "worker_id": "NEXT100-065B-CROSSSOURCE-DEDUP-V4",
        "issue": extension.get("issue"),
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "base_v3_head_sha": extension["base_v3"]["head_sha"],
        "registry_convergence_head_sha": extension["registry_convergence"]["head_sha"],
        "expected_pre_dedup": copy.deepcopy(extension["expected_pre_dedup"]),
        "acquisition_evidence": sorted(acquisition_evidence, key=lambda item: item["source_id"]),
        "dedup_v3": dedup,
        "claim_boundary": copy.deepcopy(extension["claim_boundary"]),
    }
    return {**core, "report_sha256": _sha256(_canonical_bytes(core))}


def verify_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA:
        raise CrossSourceV4Error("unsupported V4 report schema")
    if report.get("local_free_only") is not True:
        raise CrossSourceV4Error("report must remain LOCAL_FREE")
    for flag in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
    ):
        if report.get(flag) is not False:
            raise CrossSourceV4Error(f"report {flag} must remain false")
    dedup = report.get("dedup_v3")
    if not isinstance(dedup, Mapping):
        raise CrossSourceV4Error("embedded V3 dedup report missing")
    v3.verify_report(dedup)
    expected = report.get("expected_pre_dedup")
    if not isinstance(expected, Mapping):
        raise CrossSourceV4Error("pre-dedup authority binding missing")
    evidence = report.get("acquisition_evidence")
    if not isinstance(evidence, list) or len(evidence) != 10:
        raise CrossSourceV4Error("all ten late terminal objects require acquisition evidence")
    claim = report.get("claim_boundary")
    if not isinstance(claim, Mapping) or claim.get("training_authorized") is not False:
        raise CrossSourceV4Error("V4 must not authorize training")
    supplied_sha = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    if supplied_sha != _sha256(_canonical_bytes(core)):
        raise CrossSourceV4Error("V4 report SHA-256 mismatch")


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    verify_report(report)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(dict(report)))
