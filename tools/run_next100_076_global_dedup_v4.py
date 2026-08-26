#!/usr/bin/env python3
"""Execute NEXT100-076 successor global cross-source deduplication.

Late authorities enter the global graph as the exact training-authorized payload
that their terminal source authority sealed. Transport wrappers are verified
first and are never substituted for normalized training capacity.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from twelve_six.data import cross_source_capacity_audit as v1
from twelve_six.data import cross_source_capacity_audit_v3 as v3

SCHEMA = "12-6.next100-076-global-dedup-report.v4"
CONFIG_SCHEMA = "12-6.next100-076-global-dedup.v4"
WORKER = "NEXT100-076-GLOBAL-DEDUP-V4-SOL"
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
BLANK_RE = re.compile(r"\n{3,}")
MAX_NIST_NORMALIZED_BYTES = 20_000


class Next100076Error(RuntimeError):
    """Fail-closed successor-dedup error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Next100076Error(message)


def _sha(payload: bytes) -> str:
    return v1._sha256(payload)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def _verify_blob(path: Path, expected: str, label: str) -> bytes:
    payload = path.read_bytes()
    actual = v1._git_blob_sha1(payload)
    _require(actual == expected, f"{label} Git blob drift: {actual} != {expected}")
    return payload


def _load_bound(root: Path, spec: dict[str, Any], label: str) -> dict[str, Any]:
    path = root / str(spec["path"])
    payload = _verify_blob(path, str(spec["blob_sha1"]), label)
    value = json.loads(payload.decode("utf-8"))
    _require(isinstance(value, dict), f"{label} root must be object")
    return value


def normalize_kmu(raw: bytes) -> bytes:
    """Reproduce NEXT100-026 normalization exactly."""
    text = unicodedata.normalize("NFKC", raw.decode("utf-8", errors="strict"))
    lines = (" ".join(line.split()) for line in text.splitlines())
    return ("\n".join(line for line in lines if line).strip() + "\n").encode("utf-8")


def normalize_nist_pdf(raw_pdf: bytes, *, start_page: int) -> bytes:
    """Reproduce NEXT100-034 bounded pdftotext normalization exactly."""
    _require(raw_pdf.startswith(b"%PDF-"), "NIST source wrapper is not a PDF")
    with tempfile.TemporaryDirectory(prefix="next100-076-nist-") as tmp:
        root = Path(tmp)
        pdf = root / "source.pdf"
        text_path = root / "source.txt"
        pdf.write_bytes(raw_pdf)
        subprocess.run(
            [
                "pdftotext",
                "-f",
                str(start_page),
                "-nopgbrk",
                "-enc",
                "UTF-8",
                str(pdf),
                str(text_path),
            ],
            check=True,
        )
        text = text_path.read_text(encoding="utf-8")
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
        _require(cut >= 12_000, "cannot find deterministic NIST truncation boundary")
        encoded = (candidate[:cut].rstrip() + "\n").encode("utf-8")
    return encoded


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    _require(end >= 0, "unterminated MDN frontmatter")
    frontmatter: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip().strip('"')
    return frontmatter, text[end + 5 :]


def _strip_mdn_fenced_code(text: str) -> str:
    out: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.lstrip()
        if fence is None and (stripped.startswith("```") or stripped.startswith("~~~")):
            fence = stripped[:3]
            continue
        if fence is not None:
            if stripped.startswith(fence):
                fence = None
            continue
        out.append(line)
    _require(fence is None, "unterminated MDN fenced code block")
    return "\n".join(out)


def normalize_mdn_prose(raw: bytes) -> bytes:
    """Reproduce NEXT100-038 MDN_PROSE_ONLY_MARKDOWN_V1 exactly."""
    text = unicodedata.normalize(
        "NFKC",
        raw.decode("utf-8", errors="strict").replace("\r\n", "\n").replace("\r", "\n"),
    )
    _, text = _parse_frontmatter(text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = _strip_mdn_fenced_code(text)
    prose_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            prose_lines.append("")
            continue
        if stripped.startswith("![") or "<img" in stripped.lower() or "<picture" in stripped.lower():
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        if re.search(r"\{\{\s*(Embed|InteractiveExample|LiveSample|EmbedGHLiveSample)", line, flags=re.I):
            continue
        line = re.sub(r"`+[^`\n]*`+", " ", line)
        line = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", line)
        line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
        line = re.sub(r"\{\{[^{}]*\}\}", " ", line)
        line = re.sub(r"<[^>]+>", " ", line)
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line)
        line = re.sub(r"^\s*>\s?", "", line)
        line = line.replace("**", "").replace("__", "").replace("~~", "")
        prose_lines.append(re.sub(r"\s+", " ", line).strip())
    normalized_lines: list[str] = []
    previous_blank = True
    for line in prose_lines:
        blank = not line
        if blank and previous_blank:
            continue
        normalized_lines.append(line)
        previous_blank = blank
    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()
    return ("\n".join(normalized_lines) + "\n").encode("utf-8")


def _payload_row(
    *,
    source_id: str,
    family: str,
    origin: str,
    object_id: str,
    modality: str,
    authority_ref: str,
    payload: bytes,
    capacity: int,
    provenance: str,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "source_family": family,
        "stable_origin_id": origin,
        "stable_object_id": object_id,
        "modality": modality,
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": authority_ref,
        "declared_capacity_bytes": capacity,
        "expected_raw_bytes": len(payload),
        "expected_raw_sha256": _sha(payload),
        "acquisition_url": provenance,
        "origin_key": object_id,
    }


def _load_convergence(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    spec = config["convergence_authority"]
    path = repo_root / str(spec["path"])
    payload = _verify_blob(path, str(spec["blob_sha1"]), "NEXT100-063 convergence")
    convergence = json.loads(payload.decode("utf-8"))
    _require(
        convergence.get("schema_version") == "12-6.next100-063-source-registry-convergence.v1",
        "NEXT100-063 schema mismatch",
    )
    _require(convergence.get("local_free_only") is True, "NEXT100-063 is not LOCAL_FREE")
    _require(convergence.get("model_training_executed") is False, "NEXT100-063 claims training")
    vector = convergence.get("converged_pre_successor_dedup_vector", {})
    _require(
        vector.get("numeric_capacity_bytes") == spec["expected_pre_dedup_capacity_bytes"],
        "NEXT100-063 capacity vector drift",
    )
    _require(
        vector.get("independent_family_counts") == spec["expected_pre_dedup_family_counts"],
        "NEXT100-063 family vector drift",
    )
    _require(
        vector.get("numeric_source_object_count") == spec["expected_numeric_source_objects"],
        "NEXT100-063 source-object count drift",
    )
    return convergence


def _late_row(convergence: dict[str, Any], worker: str) -> dict[str, Any]:
    rows = [row for row in convergence.get("late_authorities", []) if row.get("worker_id") == worker]
    _require(len(rows) == 1, f"missing/duplicate convergence row: {worker}")
    row = rows[0]
    _require(row.get("workflow_conclusion") == "success", f"{worker}: source workflow not success")
    _require(row.get("training_authorized") is True, f"{worker}: training not authorized")
    _require(row.get("evaluation_authorized") is False, f"{worker}: evaluation permission drift")
    return row


def _verify_late_binding(spec: dict[str, Any], row: dict[str, Any], label: str) -> None:
    pairs = {
        "head_sha": spec["head_sha"],
        "authority_blob_sha1": spec["blob_sha1"],
        "family_id": spec["family_id"],
        "terminal_status": spec["terminal_status"],
    }
    for key, expected in pairs.items():
        _require(row.get(key) == expected, f"{label}: convergence binding drift: {key}")


def _base_inventory(repo_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    spec = config["base_dedup_authority"]
    path = repo_root / str(spec["path"])
    payload = _verify_blob(path, str(spec["blob_sha1"]), "NEXT100-065 base inventory")
    inventory = json.loads(payload.decode("utf-8"))
    rows = inventory.get("sources", [])
    _require(len(rows) == spec["expected_source_objects"], "base source-object count drift")
    _require(
        sum(int(row["declared_capacity_bytes"]) for row in rows) == spec["expected_capacity_bytes"],
        "base declared capacity drift",
    )
    return inventory


def _kmu_payloads(root: Path, spec: dict[str, Any], conv: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    authority = _load_bound(root, spec, "KMu authority")
    _require(authority.get("verdict") == "ADMIT", "KMu authority is not ADMIT")
    _require(authority.get("rights", {}).get("training") == "ALLOWED_PRETRAINING", "KMu training right drift")
    records = authority.get("records", [])
    _require(len(records) == spec["expected_objects"], "KMu record count drift")
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for record in records:
        raw = (root / str(record["raw_path"])).read_bytes()
        _require(len(raw) == record["raw_bytes"], f"KMu raw byte drift: {record['id']}")
        _require(_sha(raw) == record["raw_sha256"], f"KMu raw hash drift: {record['id']}")
        normalized = normalize_kmu(raw)
        _require(len(normalized) == record["normalized_bytes"], f"KMu normalized byte drift: {record['id']}")
        _require(_sha(normalized) == record["normalized_sha256"], f"KMu normalized hash drift: {record['id']}")
        source_id = "ua.kmu.secretariat." + str(record["id"])
        rows.append(
            _payload_row(
                source_id=source_id,
                family=str(spec["family_id"]),
                origin="publisher:kmu-secretariat-news",
                object_id="sha256:" + str(record["normalized_sha256"]),
                modality="uk",
                authority_ref=f"NEXT100-026 exact head {spec['head_sha']} workflow {conv['workflow_run']}",
                payload=normalized,
                capacity=int(record["normalized_bytes"]),
                provenance="authority-materialized://next100-026/" + str(record["raw_path"]),
            )
        )
        payloads[source_id] = normalized
    _require(sum(len(value) for value in payloads.values()) == spec["expected_capacity_bytes"], "KMu capacity drift")
    edges = [
        {
            "left_source_id": rows[index - 1]["source_id"],
            "right_source_id": rows[index]["source_id"],
            "relation": "sibling_same_origin",
            "capacity_collapsing": False,
            "independence_collapsing": True,
            "evidence": "six Secretariat-authored KMu news snapshots share one admitted source family",
        }
        for index in range(1, len(rows))
    ]
    return rows, payloads, edges


def _nist_payloads(root: Path, spec: dict[str, Any], conv: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bytes], list[dict[str, Any]]]:
    terminal = _load_bound(root, spec, "NIST terminal authority")
    supporting_bytes = _verify_blob(root / str(spec["supporting_path"]), str(spec["supporting_blob_sha1"]), "NIST supporting authority")
    supporting = json.loads(supporting_bytes.decode("utf-8"))
    _require(terminal.get("terminal_status") == "ADMIT", "NIST terminal authority is not ADMIT")
    _require(terminal.get("rights", {}).get("model_training") == "ALLOWED_WITH_NIST_SOURCE_PROVENANCE", "NIST training right drift")
    detailed = {row["publication_id"]: row for row in supporting.get("admit", [])}
    admitted = terminal.get("admit", [])
    _require(len(admitted) == spec["expected_objects"], "NIST admitted object count drift")
    version = subprocess.run(["pdftotext", "-v"], text=True, capture_output=True, check=True)
    version_text = (version.stderr or version.stdout).strip()
    _require(str(spec["required_pdftotext_version"]) in version_text, f"pdftotext version drift: {version_text}")
    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for sealed in admitted:
        publication_id = str(sealed["publication_id"])
        detail = detailed.get(publication_id)
        _require(isinstance(detail, dict), f"NIST supporting row missing: {publication_id}")
        raw = v1.fetch_exact_source(str(detail["pdf_url"]))
        _require(len(raw) == sealed["raw_bytes"], f"NIST raw byte drift: {publication_id}")
        _require(_sha(raw) == sealed["raw_sha256"], f"NIST raw hash drift: {publication_id}")
        normalized = normalize_nist_pdf(raw, start_page=int(detail["pdf_start_page"]))
        _require(len(normalized) == sealed["normalized_utf8_bytes"], f"NIST normalized byte drift: {publication_id}")
        _require(_sha(normalized) == sealed["normalized_sha256"], f"NIST normalized hash drift: {publication_id}")
        source_id = "en.nist." + publication_id.lower().replace(".", "-")
        rows.append(
            _payload_row(
                source_id=source_id,
                family=str(spec["family_id"]),
                origin="publisher:nist-technical-series",
                object_id="sha256:" + str(sealed["normalized_sha256"]),
                modality="en",
                authority_ref=f"NEXT100-034 exact head {spec['head_sha']} workflow {conv['workflow_run']}",
                payload=normalized,
                capacity=int(sealed["normalized_utf8_bytes"]),
                provenance="authority-normalized://nist/" + publication_id,
            )
        )
        payloads[source_id] = normalized
    _require(sum(len(value) for value in payloads.values()) == spec["expected_capacity_bytes"], "NIST capacity drift")
    edges = [
        {
            "left_source_id": rows[index - 1]["source_id"],
            "right_source_id": rows[index]["source_id"],
            "relation": "sibling_same_origin",
            "capacity_collapsing": False,
            "independence_collapsing": True,
            "evidence": "three bounded NIST publications share one technical-series source family",
        }
        for index in range(1, len(rows))
    ]
    return rows, payloads, edges


def _mdn_payloads(root: Path, spec: dict[str, Any], conv: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    authority = _load_bound(root, spec, "MDN authority")
    _require(authority.get("verdict") == "ADMIT_PROSE_ONLY", "MDN authority is not ADMIT_PROSE_ONLY")
    boundary = authority.get("claim_boundary", {})
    _require(boundary.get("prose_only") is True and boundary.get("code_admitted") is False, "MDN prose/code boundary drift")
    pages = authority.get("pages", [])
    _require(len(pages) == spec["expected_objects"] == 1, "MDN page count drift")
    page = pages[0]
    upstream = authority.get("upstream", {})
    upstream_commit = str(upstream["commit"])
    path = str(page["path"])
    raw_url = f"https://raw.githubusercontent.com/mdn/content/{upstream_commit}/{path}"
    raw = v1.fetch_exact_source(raw_url)
    _require(len(raw) == page["raw_bytes"], "MDN raw byte drift")
    _require(_sha(raw) == page["raw_sha256"], "MDN raw SHA-256 drift")
    _require(v1._git_blob_sha1(raw) == page["git_blob_sha1"], "MDN Git blob drift")
    normalized = normalize_mdn_prose(raw)
    _require(len(normalized) == page["normalized_bytes"], "MDN normalized byte drift")
    _require(_sha(normalized) == page["normalized_sha256"], "MDN normalized hash drift")
    _require(len(normalized) == spec["expected_capacity_bytes"], "MDN capacity drift")
    source_id = "en.mdn.webdocs.prose.compression"
    row = _payload_row(
        source_id=source_id,
        family=str(spec["family_id"]),
        origin="github:mdn/content",
        object_id="sha256:" + str(page["normalized_sha256"]),
        modality="en",
        authority_ref=f"NEXT100-038 exact head {spec['head_sha']} workflow {conv['workflow_run']}",
        payload=normalized,
        capacity=len(normalized),
        provenance="authority-normalized://mdn/" + path,
    )
    return [row], {source_id: normalized}


def _nomis_payloads(root: Path, spec: dict[str, Any], conv: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    authority = _load_bound(root, spec, "Nomis authority")
    _require(authority.get("verdict") == "ADMIT", "Nomis authority is not ADMIT")
    _require(authority.get("scope", {}).get("training_admitted") is True, "Nomis training right drift")
    snapshot = authority.get("snapshot", {})
    normalized = (root / str(snapshot["normalized_path"])).read_bytes()
    _require(len(normalized) == snapshot["normalized_bytes"], "Nomis normalized byte drift")
    _require(_sha(normalized) == snapshot["normalized_sha256"], "Nomis normalized hash drift")
    _require(len(normalized) == spec["expected_capacity_bytes"], "Nomis capacity drift")
    source_id = "ua.verba.nomis1864.bounded24"
    row = _payload_row(
        source_id=source_id,
        family=str(spec["family_id"]),
        origin="edition:nomis:ukrainian-proverbs:1864",
        object_id="sha256:" + str(snapshot["normalized_sha256"]),
        modality="uk",
        authority_ref=f"NEXT100-027 exact head {spec['head_sha']} workflow {conv['workflow_run']}",
        payload=normalized,
        capacity=len(normalized),
        provenance="authority-materialized://next100-027/" + str(snapshot["normalized_path"]),
    )
    return [row], {source_id: normalized}


def _build_and_audit(
    *,
    repo_root: Path,
    config: dict[str, Any],
    kmu_root: Path,
    nist_root: Path,
    mdn_root: Path,
    nomis_root: Path,
) -> dict[str, Any]:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "V4 config schema mismatch")
    _require(config.get("worker_id") == WORKER, "V4 worker mismatch")
    for key, expected in (
        ("local_free_only", True),
        ("model_training_executed", False),
        ("tokenizer_fit_executed", False),
        ("paid_compute_used", False),
        ("final_test_payload_read", False),
    ):
        _require(config.get(key) is expected, f"unsafe V4 boundary: {key}")

    convergence = _load_convergence(repo_root, config)
    base = _base_inventory(repo_root, config)
    expanded = json.loads(json.dumps(base))
    payloads = {row["source_id"]: v1.fetch_exact_source(row["acquisition_url"]) for row in expanded["sources"]}
    specs = config["late_authorities"]
    bindings: dict[str, dict[str, Any]] = {}
    for key in ("kmu", "nist", "mdn", "nomis", "cpython_docs"):
        spec = specs[key]
        conv = _late_row(convergence, str(spec["worker_id"]))
        _verify_late_binding(spec, conv, key.upper())
        bindings[key] = conv
    _require(
        bindings["cpython_docs"].get("numeric_capacity_bytes") == specs["cpython_docs"]["expected_capacity_credit_bytes"]
        and bindings["cpython_docs"].get("independent_family_credit") == specs["cpython_docs"]["expected_family_credit"],
        "CPython docs received premature capacity/family credit",
    )

    kmu_rows, kmu_payloads, kmu_edges = _kmu_payloads(kmu_root, specs["kmu"], bindings["kmu"])
    nist_rows, nist_payloads, nist_edges = _nist_payloads(nist_root, specs["nist"], bindings["nist"])
    mdn_rows, mdn_payloads = _mdn_payloads(mdn_root, specs["mdn"], bindings["mdn"])
    nomis_rows, nomis_payloads = _nomis_payloads(nomis_root, specs["nomis"], bindings["nomis"])
    expanded["sources"].extend(kmu_rows + nist_rows + mdn_rows + nomis_rows)
    expanded["lineage_edges"].extend(kmu_edges + nist_edges)
    payloads.update(kmu_payloads)
    payloads.update(nist_payloads)
    payloads.update(mdn_payloads)
    payloads.update(nomis_payloads)

    acceptance = config["acceptance"]
    _require(len(expanded["sources"]) == acceptance["expected_source_objects"], "expanded source count drift")
    declared = sum(int(row["declared_capacity_bytes"]) for row in expanded["sources"])
    _require(declared == acceptance["expected_declared_capacity_bytes_before"], "expanded capacity drift")
    _require(set(payloads) == {row["source_id"] for row in expanded["sources"]}, "payload coverage drift")

    report = v3.audit_payloads(expanded, payloads)
    v3.verify_report(report)
    scope = report["terminal_candidates"]
    _require(scope["source_count"] == acceptance["expected_source_objects"], "report source count drift")
    _require(scope["declared_capacity_bytes_before"] == acceptance["expected_declared_capacity_bytes_before"], "report pre-dedup capacity drift")
    _require(scope["stable_origin_count"] == acceptance["expected_pre_dedup_stable_origins"], "stable-origin count drift")
    for modality, expected in acceptance["expected_modality_capacity_before"].items():
        _require(scope["by_modality"][modality]["declared_capacity_bytes_before"] == expected, f"{modality} pre-dedup capacity drift")
    _require(scope["conservative_unique_capacity_bytes_after"] <= scope["declared_capacity_bytes_before"], "dedup inflated capacity")

    core = {
        "schema_version": SCHEMA,
        "worker_id": WORKER,
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "convergence_head": config["convergence_authority"]["head_sha"],
        "convergence_revalidated_by_successor": True,
        "authority_payload_policy": "VERIFY_WRAPPER_IDENTITY_THEN_COMPARE_AUTHORITY_APPROVED_TRAINING_PAYLOAD",
        "cpython_docs_capacity_credit": 0,
        "dedup_report": report,
        "post_dedup_effective_origins": {
            modality: scope["by_modality"][modality]["effective_independent_origin_count"]
            for modality in ("uk", "en", "code")
        },
        "next_gate": "BALANCE_DIVERSITY_RETEST",
        "corpus_materialization_claimed": False,
        "research_corpus_v1_released": False,
        "learned_20m_checkpoint_claimed": False,
    }
    return {**core, "report_sha256": _sha(v1._canonical_bytes(core))}


def verify_envelope(report: dict[str, Any], config: dict[str, Any]) -> None:
    _require(report.get("schema_version") == SCHEMA, "report schema mismatch")
    expected = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    _require(expected == _sha(v1._canonical_bytes(core)), "V4 report self-hash mismatch")
    for key, value in (
        ("local_free_only", True),
        ("model_training_executed", False),
        ("tokenizer_fit_executed", False),
        ("paid_compute_used", False),
        ("final_test_payload_read", False),
        ("convergence_revalidated_by_successor", True),
        ("corpus_materialization_claimed", False),
        ("research_corpus_v1_released", False),
        ("learned_20m_checkpoint_claimed", False),
    ):
        _require(report.get(key) is value, f"report truth boundary failed: {key}")
    _require(report.get("next_gate") == "BALANCE_DIVERSITY_RETEST", "next gate drift")
    nested = report.get("dedup_report")
    _require(isinstance(nested, dict), "nested V3 report missing")
    v3.verify_report(nested)
    acceptance = config["acceptance"]
    scope = nested["terminal_candidates"]
    _require(scope["source_count"] == acceptance["expected_source_objects"], "verified source count drift")
    _require(scope["declared_capacity_bytes_before"] == acceptance["expected_declared_capacity_bytes_before"], "verified capacity drift")
    _require(scope["conservative_unique_capacity_bytes_after"] <= acceptance["expected_declared_capacity_bytes_before"], "verified post-dedup capacity inflation")


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(v1._canonical_bytes(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--repo-root", default=".")
    run.add_argument("--config", default="configs/data/next100_076_global_dedup_v4.json")
    run.add_argument("--kmu-root", required=True)
    run.add_argument("--nist-root", required=True)
    run.add_argument("--mdn-root", required=True)
    run.add_argument("--nomis-root", required=True)
    run.add_argument("--report", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--config", default="configs/data/next100_076_global_dedup_v4.json")
    verify.add_argument("--report", required=True)
    args = parser.parse_args()
    try:
        config = _load_json(Path(args.config))
        if args.command == "run":
            report = _build_and_audit(
                repo_root=Path(args.repo_root),
                config=config,
                kmu_root=Path(args.kmu_root),
                nist_root=Path(args.nist_root),
                mdn_root=Path(args.mdn_root),
                nomis_root=Path(args.nomis_root),
            )
            verify_envelope(report, config)
            _write(Path(args.report), report)
        else:
            verify_envelope(_load_json(Path(args.report)), config)
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
        Next100076Error,
        v1.CapacityAuditError,
        v3.CrossSourceV3Error,
    ) as exc:
        print(f"NEXT100-076 FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
