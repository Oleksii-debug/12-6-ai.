#!/usr/bin/env python3
"""Run NEXT100-105 successor global cross-source dedup V4.

The dedup semantics are intentionally delegated to the already-audited NEXT100-065
V3 engine.  This runner only expands the exact source inventory and materializes
late terminal authorities byte-identically before handing all payloads to V3.
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
from pathlib import Path
from typing import Any

from twelve_six.data import cross_source_capacity_audit as v1
from twelve_six.data import cross_source_capacity_audit_v3 as v3

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/next100_105_cross_source_dedup_v4.json"
BASE_PATH = ROOT / "configs/data/next100_065_cross_source_dedup_v3.json"
CONVERGENCE_PATH = ROOT / "configs/data/next100_063_source_registry_convergence_v1.json"
EXPECTED_CONVERGENCE_HEAD = "9ad8f74b12a2e991b7934356a88dd9a1f6ff3f41"
EXPECTED_BASE_BLOB = "c1e05f09490e25f6fed765dfb70d900717528f4d"
NIST_PDFTOTEXT_VERSION = "24.02.0"
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
BLANK_RE = re.compile(r"\n{3,}")
MAX_NIST_NORMALIZED_BYTES = 20_000


class DedupV4Error(RuntimeError):
    """Fail-closed V4 materialization or authority error."""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "12-6-ai-NEXT100-105/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _verify_payload(row: dict[str, Any], payload: bytes) -> bytes:
    if len(payload) != int(row["expected_raw_bytes"]):
        raise DedupV4Error(f"{row['source_id']}: normalized payload byte drift")
    if _sha256(payload) != row["expected_raw_sha256"]:
        raise DedupV4Error(f"{row['source_id']}: normalized payload SHA-256 drift")
    return payload


def _canonical_pinned_text(row: dict[str, Any]) -> bytes:
    raw = _download(str(row["acquisition_url"]))
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DedupV4Error(f"{row['source_id']}: pinned text is not strict UTF-8") from exc
    # KMu authority normalizes the committed body snapshot to one terminal LF.
    # The already-normalized Verba snapshot is unchanged by the same operation.
    payload = (text.strip() + "\n").encode("utf-8")
    return _verify_payload(row, payload)


def _normalize_nist_extracted(text: str) -> bytes:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = BLANK_RE.sub("\n\n", text).strip() + "\n"
    encoded = text.encode("utf-8")
    if len(encoded) > MAX_NIST_NORMALIZED_BYTES:
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
            raise DedupV4Error("NIST deterministic truncation boundary unavailable")
        text = candidate[:cut].rstrip() + "\n"
    return text.encode("utf-8")


def _pdftotext_version() -> str:
    proc = subprocess.run(["pdftotext", "-v"], text=True, capture_output=True, check=True)
    line = (proc.stderr or proc.stdout).splitlines()[0]
    if NIST_PDFTOTEXT_VERSION not in line:
        raise DedupV4Error(
            f"pdftotext drift: require {NIST_PDFTOTEXT_VERSION}, observed {line!r}"
        )
    return line


def _materialize_nist(row: dict[str, Any]) -> bytes:
    _pdftotext_version()
    pdf = _download(str(row["acquisition_url"]))
    if len(pdf) != int(row["upstream_raw_bytes"]):
        raise DedupV4Error(f"{row['source_id']}: upstream PDF byte drift")
    if _sha256(pdf) != row["upstream_raw_sha256"]:
        raise DedupV4Error(f"{row['source_id']}: upstream PDF SHA-256 drift")
    if not pdf.startswith(b"%PDF-"):
        raise DedupV4Error(f"{row['source_id']}: upstream object is not a PDF")

    with tempfile.TemporaryDirectory(prefix="next100-105-nist-") as tmp:
        pdf_path = Path(tmp) / "source.pdf"
        txt_path = Path(tmp) / "source.txt"
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
        )
        payload = _normalize_nist_extracted(txt_path.read_text(encoding="utf-8"))
    return _verify_payload(row, payload)


def materialize_late_source(row: dict[str, Any]) -> bytes:
    kind = row.get("materializer")
    if kind == "PINNED_GITHUB_TEXT":
        return _canonical_pinned_text(row)
    if kind == "NIST_PDF_NORMALIZED_V2":
        return _materialize_nist(row)
    raise DedupV4Error(f"{row.get('source_id')}: unsupported materializer {kind!r}")


def _validate_contract(config: dict[str, Any], base: dict[str, Any], convergence: dict[str, Any]) -> None:
    if config.get("schema_version") != "12-6.next100-105-cross-source-dedup.v4":
        raise DedupV4Error("wrong V4 config schema")
    if config.get("execution_profile") != "LOCAL_FREE":
        raise DedupV4Error("V4 must remain LOCAL_FREE")
    for key in ("model_training_executed", "tokenizer_fit_executed", "paid_compute_used", "final_test_payload_read"):
        if config.get(key) is not False:
            raise DedupV4Error(f"{key} must be false")

    conv_binding = config["convergence_authority"]
    if conv_binding.get("head_sha") != EXPECTED_CONVERGENCE_HEAD:
        raise DedupV4Error("convergence head binding drift")
    if convergence.get("worker_id") != "NEXT100-063-SOURCE-REGISTRY-CONVERGENCE":
        raise DedupV4Error("wrong source-convergence authority")
    vector = convergence.get("converged_pre_successor_dedup_vector", {})
    if vector.get("numeric_capacity_bytes") != conv_binding.get("expected_pre_dedup_bytes"):
        raise DedupV4Error("converged pre-dedup byte vector drift")
    if vector.get("independent_family_counts") != conv_binding.get("expected_family_counts"):
        raise DedupV4Error("converged family vector drift")
    if vector.get("numeric_source_object_count") != conv_binding.get("expected_numeric_source_objects"):
        raise DedupV4Error("converged numeric object count drift")

    base_binding = config["incumbent_dedup_authority"]
    if base_binding.get("config_blob_sha1") != EXPECTED_BASE_BLOB:
        raise DedupV4Error("incumbent V3 blob binding drift")
    if base.get("worker_id") != "NEXT100-065-CROSSSOURCE-DEDUP-V3":
        raise DedupV4Error("wrong incumbent dedup inventory")
    if len(base.get("sources", [])) != int(base_binding["source_objects"]):
        raise DedupV4Error("incumbent V3 source count drift")

    late = config.get("late_sources")
    if not isinstance(late, list) or len(late) != 10:
        raise DedupV4Error("V4 requires exactly ten newly credited late objects")
    ids = [row.get("source_id") for row in late]
    if any(not isinstance(source_id, str) or not source_id for source_id in ids):
        raise DedupV4Error("late source_id must be nonempty")
    if len(set(ids)) != len(ids):
        raise DedupV4Error("late source_id collision")
    if set(ids) & {row["source_id"] for row in base["sources"]}:
        raise DedupV4Error("late source collides with incumbent source_id")
    if sum(int(row["declared_capacity_bytes"]) for row in late) != 70_170:
        raise DedupV4Error("late numeric capacity drift")
    if sum(int(row["declared_capacity_bytes"]) for row in base["sources"]) != 243_970:
        raise DedupV4Error("incumbent numeric capacity drift")
    if 243_970 + 70_170 != conv_binding["expected_pre_dedup_bytes"]["total"]:
        raise DedupV4Error("expanded input capacity does not bind convergence authority")


def build_expanded_inventory(config: dict[str, Any], base: dict[str, Any], convergence: dict[str, Any]) -> dict[str, Any]:
    _validate_contract(config, base, convergence)
    inventory = copy.deepcopy(base)
    # Keep the V3 inventory schema so the exact audited V3 engine validates it.
    inventory["worker_id"] = "NEXT100-105-CROSSSOURCE-DEDUP-V4"
    inventory["terminal_refresh_cutoff_utc"] = "NEXT100-063@9ad8f74b12a2e991b7934356a88dd9a1f6ff3f41"
    inventory["sources"].extend(copy.deepcopy(config["late_sources"]))
    if len(inventory["sources"]) != 21:
        raise DedupV4Error("expanded inventory must contain 21 numeric source objects")
    return inventory


def run(report_path: Path, inventory_path: Path | None = None) -> dict[str, Any]:
    config = _load(CONFIG_PATH)
    base = _load(BASE_PATH)
    convergence = _load(CONVERGENCE_PATH)
    inventory = build_expanded_inventory(config, base, convergence)

    payloads: dict[str, bytes] = {}
    base_ids = {row["source_id"] for row in base["sources"]}
    for row in inventory["sources"]:
        if row["source_id"] in base_ids:
            payloads[row["source_id"]] = v1.fetch_exact_source(row["acquisition_url"])
        else:
            payloads[row["source_id"]] = materialize_late_source(row)

    if set(payloads) != {row["source_id"] for row in inventory["sources"]}:
        raise DedupV4Error("materialized payload coverage drift")

    report = v3.audit_payloads(inventory, payloads)
    v3.verify_report(report)
    v3.write_report(report, report_path)
    if inventory_path is not None:
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(
            json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--expanded-inventory")
    args = parser.parse_args()
    run(
        Path(args.report),
        Path(args.expanded_inventory) if args.expanded_inventory else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
