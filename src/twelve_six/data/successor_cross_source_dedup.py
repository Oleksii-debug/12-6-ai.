"""NEXT100-071 successor global cross-source deduplication intake.

This layer composes terminal late authorities with the NEXT100-065 V3
inventory without weakening the underlying fail-closed dedup semantics.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from twelve_six.data import cross_source_capacity_audit as v1
from twelve_six.data import cross_source_capacity_audit_v3 as v3

SCHEMA = "12-6.next100-071-successor-cross-source-dedup-report.v1"
CONFIG_SCHEMA = "12-6.next100-071-successor-cross-source-dedup.v1"
WORKER = "NEXT100-071-GLOBAL-CROSS-SOURCE-DEDUP"
NIST_POLICY = "NEXT100_034_NIST_PDF_BODY_NFKC_BOUNDED_V1"
MDN_POLICY = "MDN_PROSE_ONLY_MARKDOWN_V1"
MAX_NIST_NORMALIZED_BYTES = 20_000
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
BLANK_RE = re.compile(r"\n{3,}")


class SuccessorDedupError(RuntimeError):
    """Fail-closed successor intake or report error."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"{rendered}\n".encode()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SuccessorDedupError(message)


def _normalize_nist_extracted(text: str, *, max_bytes: int = MAX_NIST_NORMALIZED_BYTES) -> bytes:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    text = unicodedata.normalize("NFKC", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = BLANK_RE.sub("\n\n", text).strip() + "\n"
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return encoded

    prefix = encoded[:max_bytes]
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
        raise SuccessorDedupError("cannot find safe deterministic NIST truncation boundary")
    return (candidate[:cut].rstrip() + "\n").encode("utf-8")


def _parse_mdn_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        raise SuccessorDedupError("unterminated MDN frontmatter")
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
    if fence is not None:
        raise SuccessorDedupError("unterminated MDN fenced code block")
    return "\n".join(out)


def _normalize_mdn_prose(raw: bytes) -> bytes:
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise SuccessorDedupError("MDN source is not strict UTF-8") from exc
    text = unicodedata.normalize("NFKC", decoded.replace("\r\n", "\n").replace("\r", "\n"))
    _, text = _parse_mdn_frontmatter(text)
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
        line = re.sub(r"\s+", " ", line).strip()
        prose_lines.append(line)

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


def _verify_transformed_payload(row: Mapping[str, Any], raw: bytes, normalized: bytes) -> None:
    _require(len(raw) == row.get("upstream_raw_bytes"), f"{row.get('source_id')}: upstream byte count drift")
    _require(_sha256(raw) == row.get("upstream_raw_sha256"), f"{row.get('source_id')}: upstream SHA-256 drift")
    expected_blob = row.get("upstream_git_blob_sha1")
    if expected_blob is not None:
        _require(_git_blob_sha1(raw) == expected_blob, f"{row.get('source_id')}: upstream git blob drift")
    _require(len(normalized) == row.get("expected_raw_bytes"), f"{row.get('source_id')}: normalized byte count drift")
    _require(
        _sha256(normalized) == row.get("expected_raw_sha256"),
        f"{row.get('source_id')}: normalized SHA-256 drift",
    )


def _materialize_nist(
    row: Mapping[str, Any],
    *,
    fetcher: Callable[[str], bytes],
    pdftotext: str = "pdftotext",
) -> bytes:
    raw_pdf = fetcher(str(row["acquisition_url"]))
    start_page = row.get("pdf_start_page")
    _require(isinstance(start_page, int) and start_page >= 1, f"{row.get('source_id')}: invalid pdf_start_page")
    _require(raw_pdf.startswith(b"%PDF-"), f"{row.get('source_id')}: upstream source is not a PDF")
    _require(len(raw_pdf) == row.get("upstream_raw_bytes"), f"{row.get('source_id')}: upstream PDF size drift")
    _require(_sha256(raw_pdf) == row.get("upstream_raw_sha256"), f"{row.get('source_id')}: upstream PDF SHA-256 drift")

    with tempfile.TemporaryDirectory(prefix="next100-071-nist-") as tmp:
        root = Path(tmp)
        pdf_path = root / "source.pdf"
        txt_path = root / "source.txt"
        pdf_path.write_bytes(raw_pdf)
        try:
            subprocess.run(
                [
                    pdftotext,
                    "-f",
                    str(start_page),
                    "-nopgbrk",
                    "-enc",
                    "UTF-8",
                    str(pdf_path),
                    str(txt_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SuccessorDedupError(f"{row.get('source_id')}: pdftotext materialization failed") from exc
        try:
            extracted = txt_path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeError) as exc:
            raise SuccessorDedupError(f"{row.get('source_id')}: invalid pdftotext UTF-8 output") from exc

    normalized = _normalize_nist_extracted(extracted)
    _verify_transformed_payload(row, raw_pdf, normalized)
    return normalized


def _materialize_mdn(row: Mapping[str, Any], *, fetcher: Callable[[str], bytes]) -> bytes:
    raw = fetcher(str(row["acquisition_url"]))
    normalized = _normalize_mdn_prose(raw)
    _verify_transformed_payload(row, raw, normalized)
    return normalized


def _validate_config(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "successor config schema mismatch")
    _require(config.get("worker_id") == WORKER, "successor worker mismatch")
    for key, expected in (
        ("local_free_only", True),
        ("model_training_executed", False),
        ("tokenizer_fit_executed", False),
        ("paid_compute_used", False),
        ("final_test_payload_read", False),
    ):
        _require(config.get(key) is expected, f"unsafe boundary: {key}")

    late = config.get("late_numeric_sources")
    _require(isinstance(late, list) and late, "late_numeric_sources must be nonempty")
    late_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in late:
        _require(isinstance(raw, Mapping), "late source row must be an object")
        row = dict(raw)
        source_id = row.get("source_id")
        _require(isinstance(source_id, str) and source_id and source_id not in seen, "late source ids must be unique")
        seen.add(source_id)
        _require(row.get("evidence_status") == "DEDICATED_TERMINAL", f"{source_id}: source is not terminal")
        _require(isinstance(row.get("authority_ref"), str) and row["authority_ref"], f"{source_id}: authority_ref missing")
        policy = row.get("materialization_policy")
        if policy is not None:
            _require(policy in {NIST_POLICY, MDN_POLICY}, f"{source_id}: unknown materialization policy")
            for key in ("upstream_raw_bytes", "upstream_raw_sha256"):
                _require(row.get(key) is not None, f"{source_id}: missing {key}")
            if policy == NIST_POLICY:
                _require(row.get("pdf_start_page") is not None, f"{source_id}: missing pdf_start_page")
            if policy == MDN_POLICY:
                _require(row.get("upstream_git_blob_sha1") is not None, f"{source_id}: missing upstream_git_blob_sha1")
        late_rows.append(row)

    exclusions = config.get("zero_credit_exclusions")
    _require(isinstance(exclusions, list) and exclusions, "zero_credit_exclusions must be explicit")
    cpython = [
        row
        for row in exclusions
        if isinstance(row, Mapping) and row.get("worker_id") == "NEXT100-037-DATA-EN-PYTHON-DOCS"
    ]
    _require(len(cpython) == 1, "CPython zero-credit authority exclusion must be explicit")
    _require(cpython[0].get("numeric_capacity_bytes") == 0, "CPython received premature numeric capacity")
    _require(cpython[0].get("independent_family_credit") == 0, "CPython received premature family credit")
    _require(
        cpython[0].get("reason") == "BLOCKED_ACCEPTED_CHUNK_BYTE_LEDGER_NOT_MATERIALIZED",
        "CPython exclusion reason changed",
    )

    expected = config.get("expected_pre_dedup")
    _require(isinstance(expected, Mapping), "expected_pre_dedup missing")
    return late_rows, dict(expected)


def _count_families(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    modalities = sorted({str(row["modality"]) for row in rows})
    for modality in modalities:
        result[modality] = len({str(row["source_family"]) for row in rows if row["modality"] == modality})
    result["total"] = sum(result.values())
    return result


def _capacity_vector(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    modalities = sorted({str(row["modality"]) for row in rows})
    for modality in modalities:
        result[modality] = sum(int(row["declared_capacity_bytes"]) for row in rows if row["modality"] == modality)
    result["total"] = sum(result.values())
    return result


def build_successor_inventory(base_inventory: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    late_rows, expected = _validate_config(config)
    _require(base_inventory.get("schema_version") == v3.INVENTORY_SCHEMA, "base inventory schema mismatch")
    _require(base_inventory.get("local_free_only") is True, "base inventory must be LOCAL_FREE")
    _require(base_inventory.get("model_training_executed") is False, "base inventory training boundary changed")
    base_sources = base_inventory.get("sources")
    base_edges = base_inventory.get("lineage_edges")
    _require(isinstance(base_sources, list) and base_sources, "base inventory sources missing")
    _require(isinstance(base_edges, list), "base inventory lineage edges missing")

    rows = [dict(row) for row in base_sources] + late_rows
    ids = [row.get("source_id") for row in rows]
    _require(len(ids) == len(set(ids)), "successor source_id collision")

    expected_capacity = expected.get("numeric_capacity_bytes")
    expected_families = expected.get("independent_family_counts")
    _require(isinstance(expected_capacity, Mapping), "expected capacity vector missing")
    _require(isinstance(expected_families, Mapping), "expected family vector missing")
    _require(_capacity_vector(rows) == dict(expected_capacity), "pre-dedup capacity vector mismatch")
    _require(_count_families(rows) == dict(expected_families), "pre-dedup family vector mismatch")
    _require(len(rows) == expected.get("source_object_count"), "pre-dedup source object count mismatch")

    return {
        "schema_version": v3.INVENTORY_SCHEMA,
        "worker_id": WORKER,
        "local_free_only": True,
        "model_training_executed": False,
        "sources": rows,
        "lineage_edges": [dict(edge) for edge in base_edges],
    }


def materialize_successor_payloads(
    inventory: Mapping[str, Any],
    *,
    fetcher: Callable[[str], bytes] = v1.fetch_exact_source,
    pdftotext: str = "pdftotext",
) -> dict[str, bytes]:
    rows = inventory.get("sources")
    _require(isinstance(rows, list) and rows, "successor inventory sources missing")
    payloads: dict[str, bytes] = {}
    for row in rows:
        source_id = str(row["source_id"])
        policy = row.get("materialization_policy")
        if policy == NIST_POLICY:
            payload = _materialize_nist(row, fetcher=fetcher, pdftotext=pdftotext)
        elif policy == MDN_POLICY:
            payload = _materialize_mdn(row, fetcher=fetcher)
        else:
            payload = fetcher(str(row["acquisition_url"]))
        payloads[source_id] = payload
    return payloads


def _post_dedup_handoff(v3_report: Mapping[str, Any], minimum: int) -> dict[str, Any]:
    scope = v3_report.get("terminal_candidates")
    _require(isinstance(scope, Mapping), "V3 terminal candidate scope missing")
    by_modality = scope.get("by_modality")
    _require(isinstance(by_modality, Mapping), "V3 modality scope missing")
    required = ("uk", "en", "code")
    independent = {
        modality: int(by_modality.get(modality, {}).get("effective_independent_origin_count", 0))
        for modality in required
    }
    passes = all(independent[modality] >= minimum for modality in required)
    capacities = {
        modality: int(by_modality.get(modality, {}).get("conservative_unique_capacity_bytes_after", 0))
        for modality in required
    }
    capacities["total"] = int(scope.get("conservative_unique_capacity_bytes_after", 0))
    independent["total"] = sum(independent.values())
    return {
        "post_dedup_capacity_bytes": capacities,
        "post_dedup_effective_independent_origin_counts": independent,
        "family_minimum_required_per_stratum": minimum,
        "family_minimum_status": "PASS_POST_DEDUP_MINIMUM" if passes else "FAIL_POST_DEDUP_MINIMUM",
        "next_gate": "BALANCE_DIVERSITY_RETEST" if passes else "SOURCE_FAMILY_REMEDIATION",
        "corpus_materialization_authorized": False,
        "tokenizer_fit_authorized": False,
        "learned_20m_campaign_authorized": False,
    }


def audit_successor_live(
    base_inventory: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    fetcher: Callable[[str], bytes] = v1.fetch_exact_source,
    pdftotext: str = "pdftotext",
) -> dict[str, Any]:
    inventory = build_successor_inventory(base_inventory, config)
    payloads = materialize_successor_payloads(inventory, fetcher=fetcher, pdftotext=pdftotext)
    v3_report = v3.audit_payloads(inventory, payloads)
    v3.verify_report(v3_report)
    minimum = int(config["expected_pre_dedup"]["family_minimum_required_per_stratum"])
    handoff = _post_dedup_handoff(v3_report, minimum)
    core = {
        "schema_version": SCHEMA,
        "worker_id": WORKER,
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "source_count": int(v3_report["source_count"]),
        "input_pre_dedup": dict(config["expected_pre_dedup"]),
        "zero_credit_exclusions": [dict(row) for row in config["zero_credit_exclusions"]],
        "v3_report": v3_report,
        "handoff": handoff,
        "claim_boundary": {
            "canonical_registry_rewritten": False,
            "research_corpus_released": False,
            "evaluation_decontamination_complete": False,
            "unique_loss_ledger_complete": False,
            "learned_20m_checkpoint_claimed": False,
            "learned_100m_checkpoint_claimed": False,
        },
        "raw_text_emitted": False,
    }
    return {**core, "report_sha256": _sha256(_canonical_bytes(core))}


def verify_successor_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == SCHEMA, "successor report schema mismatch")
    expected_hash = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    _require(expected_hash == _sha256(_canonical_bytes(core)), "successor report self-hash mismatch")
    for key, expected in (
        ("local_free_only", True),
        ("model_training_executed", False),
        ("tokenizer_fit_executed", False),
        ("paid_compute_used", False),
        ("final_test_payload_read", False),
        ("raw_text_emitted", False),
    ):
        _require(report.get(key) is expected, f"unsafe report boundary: {key}")
    v3_report = report.get("v3_report")
    _require(isinstance(v3_report, Mapping), "embedded V3 report missing")
    v3.verify_report(v3_report)
    input_scope = report.get("input_pre_dedup")
    _require(isinstance(input_scope, Mapping), "input pre-dedup scope missing")
    _require(report.get("source_count") == input_scope.get("source_object_count"), "source count drift")
    handoff = report.get("handoff")
    _require(isinstance(handoff, Mapping), "handoff missing")
    _require(handoff.get("corpus_materialization_authorized") is False, "corpus was authorized prematurely")
    _require(handoff.get("tokenizer_fit_authorized") is False, "tokenizer fit was authorized prematurely")
    _require(handoff.get("learned_20m_campaign_authorized") is False, "20M campaign was authorized prematurely")


def write_successor_report(report: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_canonical_bytes(report))
