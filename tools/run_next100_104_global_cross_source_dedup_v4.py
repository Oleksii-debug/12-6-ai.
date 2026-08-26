#!/usr/bin/env python3
"""Run the NEXT100-104 successor global cross-source deduplication gate.

This composes the terminal NEXT100-065 inventory with the positive-capacity
late authorities converged by NEXT100-063. NIST PDFs are deterministically
converted to the exact admitted bounded training payload before the common
DATA-232/NEXT100-065 matcher sees them.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from twelve_six.data import cross_source_capacity_audit as v1
from twelve_six.data.cross_source_capacity_audit_v3 import (
    audit_payloads,
    verify_report as verify_v3_report,
)

EXTENSION_SCHEMA = "12-6.next100-104-global-cross-source-dedup-extension.v4"
REPORT_SCHEMA = "12-6.next100-104-global-cross-source-dedup-report.v4"
NIST_POLICY = "NEXT100_034_NIST_PDF_BODY_NFKC_BOUNDED_V1"
DIRECT_POLICIES = {"DIRECT_PINNED_UTF8_SNAPSHOT", "DIRECT_PINNED_NORMALIZED_UTF8_SNAPSHOT"}
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
BLANK_RE = re.compile(r"\n{3,}")
MAX_NIST_NORMALIZED_BYTES = 20_000


class Next100104Error(RuntimeError):
    """Fail-closed NEXT100-104 authority or execution error."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()  # noqa: S324 - Git object identity, not security


def _read_json(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise Next100104Error(f"JSON root must be an object: {path}")
    return value


def _read_json_exact_blob(path: str | Path, expected_blob_sha1: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    actual = _git_blob_sha1(raw)
    if actual != expected_blob_sha1:
        raise Next100104Error(f"Git blob identity drift for {path}: {actual} != {expected_blob_sha1}")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise Next100104Error(f"JSON root must be an object: {path}")
    return value


def normalize_nist_extracted(text: str) -> bytes:
    """Reproduce the terminal NEXT100-034 bounded NIST training-text policy."""
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
        raise Next100104Error("NIST normalization cannot find safe deterministic truncation boundary")
    return (candidate[:cut].rstrip() + "\n").encode("utf-8")


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "12-6-ai-NEXT100-104/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def _fetch_nist_training_payload(row: Mapping[str, Any]) -> bytes:
    pdf = _download(str(row["acquisition_url"]))
    if not pdf.startswith(b"%PDF-"):
        raise Next100104Error(f"{row['source_id']}: upstream payload is not PDF")
    if len(pdf) != row.get("upstream_pdf_bytes") or _sha256(pdf) != row.get("upstream_pdf_sha256"):
        raise Next100104Error(f"{row['source_id']}: upstream PDF identity drift")

    with tempfile.TemporaryDirectory(prefix="next100-104-nist-") as tmp:
        root = Path(tmp)
        pdf_path = root / "source.pdf"
        txt_path = root / "source.txt"
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
                str(txt_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = normalize_nist_extracted(txt_path.read_text(encoding="utf-8"))

    if len(payload) != row["expected_raw_bytes"] or _sha256(payload) != row["expected_raw_sha256"]:
        raise Next100104Error(f"{row['source_id']}: admitted NIST training-payload identity drift")
    return payload


def _capacity_by_modality(rows: list[dict[str, Any]]) -> dict[str, int]:
    result = {"uk": 0, "en": 0, "code": 0}
    for row in rows:
        modality = row.get("modality")
        if modality not in result:
            raise Next100104Error(f"unsupported modality: {modality}")
        capacity = row.get("declared_capacity_bytes")
        if not isinstance(capacity, int) or capacity <= 0:
            raise Next100104Error(f"{row.get('source_id')}: capacity must be positive integer")
        result[modality] += capacity
    result["total"] = sum(result.values())
    return result


def _families_by_modality(rows: list[dict[str, Any]]) -> dict[str, int]:
    families: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        families[str(row["modality"])].add(str(row["source_family"]))
    result = {modality: len(families.get(modality, set())) for modality in ("uk", "en", "code")}
    result["total"] = sum(result.values())
    return result


def build_inventory(
    base: Mapping[str, Any],
    convergence: Mapping[str, Any],
    extension: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    if extension.get("schema_version") != EXTENSION_SCHEMA:
        raise Next100104Error("unsupported extension schema")
    for flag in ("local_free_only",):
        if extension.get(flag) is not True:
            raise Next100104Error(f"{flag} invariant failed")
    for flag in ("model_training_executed", "tokenizer_fit_executed", "paid_compute_used", "final_test_payload_read"):
        if extension.get(flag) is not False:
            raise Next100104Error(f"{flag} invariant failed")

    late = extension.get("sources")
    if not isinstance(late, list) or not late:
        raise Next100104Error("extension sources must be nonempty")
    late_rows = [dict(row) for row in late if isinstance(row, Mapping)]
    if len(late_rows) != len(late):
        raise Next100104Error("every extension source must be an object")
    if len({row.get("source_id") for row in late_rows}) != len(late_rows):
        raise Next100104Error("extension source_id values must be unique")

    ext_expected = extension["extension_expected"]
    if _capacity_by_modality(late_rows) != ext_expected["capacity_bytes"]:
        raise Next100104Error("extension capacity vector drift")
    if len(late_rows) != ext_expected["source_object_count"]:
        raise Next100104Error("extension source-object count drift")

    base_rows_raw = base.get("sources")
    if not isinstance(base_rows_raw, list) or not base_rows_raw:
        raise Next100104Error("base inventory sources missing")
    base_rows = [dict(row) for row in base_rows_raw]
    if _capacity_by_modality(base_rows) != extension["base_inventory"]["expected_capacity_bytes"]:
        raise Next100104Error("base capacity vector drift")
    if len(base_rows) != extension["base_inventory"]["expected_source_object_count"]:
        raise Next100104Error("base source-object count drift")

    combined_rows = base_rows + late_rows
    if len({row.get("source_id") for row in combined_rows}) != len(combined_rows):
        raise Next100104Error("combined source inventory contains duplicate source_id")
    if len(combined_rows) != ext_expected["combined_source_object_count"]:
        raise Next100104Error("combined source-object count drift")

    capacity = _capacity_by_modality(combined_rows)
    families = _families_by_modality(combined_rows)
    parent = extension["parent_convergence"]
    if capacity != parent["expected_pre_dedup_capacity_bytes"]:
        raise Next100104Error("combined capacity does not reproduce parent convergence vector")
    if families != parent["expected_pre_dedup_family_counts"]:
        raise Next100104Error("combined family counts do not reproduce parent convergence vector")

    conv_vector = convergence.get("converged_pre_successor_dedup_vector", {})
    if conv_vector.get("numeric_capacity_bytes") != capacity:
        raise Next100104Error("parent convergence capacity authority drift")
    if conv_vector.get("independent_family_counts") != families:
        raise Next100104Error("parent convergence family authority drift")
    if convergence.get("claim_boundary", {}).get("safe_result") != "SOURCE_AUTHORITY_VECTOR_CONVERGED_FOR_NEXT_DEDUP_ITERATION":
        raise Next100104Error("parent convergence is not authorized for the next dedup iteration")

    inventory = copy.deepcopy(dict(base))
    inventory["sources"] = combined_rows
    inventory["lineage_edges"] = list(base.get("lineage_edges", [])) + list(extension.get("lineage_edges", []))
    inventory["terminal_refresh_cutoff_utc"] = "2026-08-26T19:26:40Z"
    inventory["final_refresh_required"] = False
    return inventory, capacity, families


def acquire_payloads(inventory: Mapping[str, Any]) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for raw in inventory["sources"]:
        row = dict(raw)
        policy = row.get("payload_policy")
        if policy == NIST_POLICY:
            payload = _fetch_nist_training_payload(row)
        elif policy in DIRECT_POLICIES or policy is None:
            payload = v1.fetch_exact_source(str(row["acquisition_url"]))
        else:
            raise Next100104Error(f"{row['source_id']}: unsupported payload policy {policy}")
        payloads[str(row["source_id"])] = payload
    return payloads


def _post_capacity(audit: Mapping[str, Any]) -> dict[str, int]:
    by_modality = audit["terminal_candidates"]["by_modality"]
    result = {
        modality: int(by_modality[modality]["conservative_unique_capacity_bytes_after"])
        for modality in ("uk", "en", "code")
    }
    result["total"] = int(audit["terminal_candidates"]["conservative_unique_capacity_bytes_after"])
    return result


def _effective_origins(audit: Mapping[str, Any]) -> dict[str, int]:
    by_modality = audit["terminal_candidates"]["by_modality"]
    result = {
        modality: int(by_modality[modality]["effective_independent_origin_count"])
        for modality in ("uk", "en", "code")
    }
    result["total"] = int(audit["terminal_candidates"]["effective_independent_origin_count"])
    return result


def make_report(
    audit: Mapping[str, Any],
    pre_capacity: Mapping[str, int],
    family_counts: Mapping[str, int],
    extension: Mapping[str, Any],
) -> dict[str, Any]:
    post = _post_capacity(audit)
    origins = _effective_origins(audit)
    family_minimum_pass = all(family_counts[m] >= 2 and origins[m] >= 2 for m in ("uk", "en", "code"))
    core: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "worker_id": "NEXT100-104-GLOBAL-CROSS-SOURCE-DEDUP-V4",
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "parent_convergence_head_sha": extension["parent_convergence"]["head_sha"],
        "source_object_count": audit["source_count"],
        "pre_dedup_capacity_bytes": dict(pre_capacity),
        "post_dedup_capacity_bytes": post,
        "duplicate_discount_bytes": {key: int(pre_capacity[key]) - int(post[key]) for key in ("uk", "en", "code", "total")},
        "pre_dedup_family_counts": dict(family_counts),
        "post_dedup_effective_origin_counts": origins,
        "family_minimum_required_per_stratum": 2,
        "family_minimum_status": "PASS" if family_minimum_pass else "BLOCKED",
        "next_gate": "BALANCE_DIVERSITY_RETEST" if family_minimum_pass else "SOURCE_ACQUISITION_REQUIRED",
        "dedup_audit": dict(audit),
        "claim_boundary": {
            "source_authority_only": True,
            "canonical_registry_rewritten": False,
            "corpus_identity_claimed": False,
            "decontamination_claimed": False,
            "unique_loss_positions_claimed": False,
            "tokenizer_fit_authorized": False,
            "learned_20m_campaign_authorized": False,
            "learned_20m_checkpoint_claimed": False,
            "safe_result": "GLOBAL_CROSS_SOURCE_DEDUP_COMPLETE_FOR_BALANCE_RETEST" if family_minimum_pass else "GLOBAL_CROSS_SOURCE_DEDUP_COMPLETE_SOURCE_GAP_REMAINS",
        },
    }
    return {**core, "report_sha256": _sha256(_canonical_bytes(core))}


def verify_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise Next100104Error("unsupported report schema")
    expected = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    if expected != _sha256(_canonical_bytes(core)):
        raise Next100104Error("report self-hash mismatch")
    if report.get("local_free_only") is not True:
        raise Next100104Error("LOCAL_FREE invariant failed")
    for flag in ("model_training_executed", "tokenizer_fit_executed", "paid_compute_used", "final_test_payload_read"):
        if report.get(flag) is not False:
            raise Next100104Error(f"{flag} invariant failed")
    if report.get("source_object_count") != 21:
        raise Next100104Error("successor dedup must cover exactly 21 positive-capacity source objects")
    verify_v3_report(report["dedup_audit"])
    pre = report["pre_dedup_capacity_bytes"]
    post = report["post_dedup_capacity_bytes"]
    if any(int(post[key]) > int(pre[key]) for key in ("uk", "en", "code", "total")):
        raise Next100104Error("deduplication inflated capacity")
    family_counts = report["pre_dedup_family_counts"]
    origins = report["post_dedup_effective_origin_counts"]
    family_minimum_pass = all(int(family_counts[m]) >= 2 and int(origins[m]) >= 2 for m in ("uk", "en", "code"))
    expected_next = "BALANCE_DIVERSITY_RETEST" if family_minimum_pass else "SOURCE_ACQUISITION_REQUIRED"
    if report.get("next_gate") != expected_next:
        raise Next100104Error("next-gate decision is inconsistent with family/origin evidence")


def run(base_path: str, convergence_path: str, extension_path: str) -> dict[str, Any]:
    extension = _read_json(extension_path)
    base = _read_json_exact_blob(base_path, extension["base_inventory"]["config_blob_sha1"])
    convergence = _read_json_exact_blob(
        convergence_path,
        extension["parent_convergence"]["config_blob_sha1"],
    )
    inventory, capacity, families = build_inventory(base, convergence, extension)
    payloads = acquire_payloads(inventory)
    audit = audit_payloads(inventory, payloads)
    verify_v3_report(audit)
    report = make_report(audit, capacity, families, extension)
    verify_report(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    execute = sub.add_parser("run")
    execute.add_argument("--base-inventory", required=True)
    execute.add_argument("--convergence", required=True)
    execute.add_argument("--extension", required=True)
    execute.add_argument("--report", required=True)

    verify = sub.add_parser("verify")
    verify.add_argument("--report", required=True)

    args = parser.parse_args()
    if args.command == "verify":
        verify_report(_read_json(args.report))
        return 0

    report = run(args.base_inventory, args.convergence, args.extension)
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "source_object_count": report["source_object_count"],
        "pre_dedup_capacity_bytes": report["pre_dedup_capacity_bytes"],
        "post_dedup_capacity_bytes": report["post_dedup_capacity_bytes"],
        "family_minimum_status": report["family_minimum_status"],
        "next_gate": report["next_gate"],
        "report_sha256": report["report_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
