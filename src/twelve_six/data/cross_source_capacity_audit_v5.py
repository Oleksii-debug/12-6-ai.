"""NEXT100-065C global cross-source dedup V5.

V5 re-executes the full NEXT100-065B/V4 source materialization and then adds
exact terminal MDN prose plus only the 14 DATA-228-accepted CPython docs chunks.
It deliberately refuses to credit the two CPython privacy-rejected chunks.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.data import cross_source_capacity_audit as v1
from twelve_six.data import cross_source_capacity_audit_v3 as v3
from twelve_six.data import cross_source_capacity_audit_v4 as v4
from twelve_six.data.pipeline import PipelineConfig, _quality_reason, normalize_text

SCHEMA = "12-6.next100-065c-cross-source-dedup-report.v5"
INVENTORY_SCHEMA = "12-6.next100-065c-cross-source-dedup.v5"
WORKER_ID = "NEXT100-065C-CROSSSOURCE-DEDUP-V5"
MDN_POLICY = "MDN_PROSE_ONLY_MARKDOWN_V1"
CPYTHON_POLICY = "DATA228_PLAIN_TEXT_NFKC_LF_PER_LINE_WHITESPACE_DROP_EMPTY_STRIP_V1"


class CrossSourceV5Error(RuntimeError):
    """Fail-closed V5 source-materialization or authority error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossSourceV5Error(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CrossSourceV5Error(f"{path}: JSON root must be an object")
    return value


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        raise CrossSourceV5Error("unterminated MDN frontmatter")
    frontmatter: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip().strip('"')
    return frontmatter, text[end + 5 :]


def _strip_fenced_code(text: str) -> tuple[str, int]:
    out: list[str] = []
    fence: str | None = None
    removed = 0
    for line in text.splitlines():
        stripped = line.lstrip()
        if fence is None and (
            stripped.startswith("```") or stripped.startswith("~~~")
        ):
            fence = stripped[:3]
            removed += 1
            continue
        if fence is not None:
            removed += 1
            if stripped.startswith(fence):
                fence = None
            continue
        out.append(line)
    if fence is not None:
        raise CrossSourceV5Error("unterminated MDN fenced code block")
    return "\n".join(out), removed


def _normalize_mdn_prose(raw: bytes) -> tuple[bytes, dict[str, int]]:
    """Reproduce NEXT100-038 MDN_PROSE_ONLY_MARKDOWN_V1 exactly."""
    text = unicodedata.normalize(
        "NFKC",
        raw.decode("utf-8", errors="strict")
        .replace("\r\n", "\n")
        .replace("\r", "\n"),
    )
    _, text = _parse_frontmatter(text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text, fenced_lines = _strip_fenced_code(text)
    stats = {
        "fenced_code_lines_removed": fenced_lines,
        "image_lines_removed": 0,
        "table_lines_removed": 0,
        "embed_macro_lines_removed": 0,
        "inline_code_spans_removed": 0,
    }
    prose_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            prose_lines.append("")
            continue
        if (
            stripped.startswith("![")
            or "<img" in stripped.lower()
            or "<picture" in stripped.lower()
        ):
            stats["image_lines_removed"] += 1
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            stats["table_lines_removed"] += 1
            continue
        if re.search(
            r"\{\{\s*(Embed|InteractiveExample|LiveSample|EmbedGHLiveSample)",
            line,
            flags=re.I,
        ):
            stats["embed_macro_lines_removed"] += 1
            continue
        spans = re.findall(r"`+[^`\n]*`+", line)
        stats["inline_code_spans_removed"] += len(spans)
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
    normalized = ("\n".join(normalized_lines) + "\n").encode("utf-8")
    return normalized, stats


def _materialize_mdn(spec: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    _require(spec.get("normalization_policy") == MDN_POLICY, "MDN policy drift")
    _require(spec.get("dedicated_workflow_conclusion") == "success", "MDN authority not green")
    _require(spec.get("training") == "ALLOWED_UNDER_LICENSE_TERMS", "MDN training rights drift")
    _require(spec.get("evaluation") == "NOT_SEPARATELY_ADMITTED", "MDN evaluation boundary drift")
    raw = v1.fetch_exact_source(str(spec["acquisition_url"]))
    _require(len(raw) == spec["raw_bytes"], "MDN raw byte-count drift")
    _require(_sha256(raw) == spec["raw_sha256"], "MDN raw SHA-256 drift")
    _require(_git_blob_sha1(raw) == spec["git_blob_sha1"], "MDN Git blob drift")
    normalized, stats = _normalize_mdn_prose(raw)
    _require(len(normalized) == spec["normalized_bytes"], "MDN normalized byte-count drift")
    _require(_sha256(normalized) == spec["normalized_sha256"], "MDN normalized SHA-256 drift")
    return normalized, {
        "source_id": spec["source_id"],
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "git_blob_sha1": _git_blob_sha1(raw),
        "comparison_bytes": len(normalized),
        "comparison_sha256": _sha256(normalized),
        "normalization_policy": MDN_POLICY,
        "removal_stats": stats,
    }


def _chunk_text(text: str, *, max_chars: int, min_chars: int) -> tuple[str, ...]:
    """Reproduce the DATA-228 generic natural-text chunker exactly."""
    _require(max_chars >= min_chars >= 20, "invalid CPython chunk limits")
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            value = "\n".join(current).strip()
            if len(value) >= min_chars:
                chunks.append(value)
            current, current_len = [], 0

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            pieces = [paragraph]
        else:
            pieces: list[str] = []
            words = paragraph.split()
            part: list[str] = []
            part_len = 0
            for word in words:
                needed = len(word) if not part else len(word) + 1
                if part and part_len + needed > max_chars:
                    pieces.append(" ".join(part))
                    part = [word]
                    part_len = len(word)
                else:
                    part.append(word)
                    part_len += needed
            if part:
                pieces.append(" ".join(part))
        for piece in pieces:
            needed = len(piece) if not current else len(piece) + 1
            if current and current_len + needed > max_chars:
                flush()
            current.append(piece)
            current_len += needed
    flush()
    return tuple(chunks)


def _cpython_quality_config(spec: Mapping[str, Any]) -> PipelineConfig:
    policy = spec["quality_policy"]
    _require(policy.get("reject_control_characters") is True, "CPython control gate weakened")
    _require(policy.get("reject_email") is True, "CPython email gate weakened")
    _require(policy.get("reject_phone") is True, "CPython phone gate weakened")
    return PipelineConfig(
        split_seed="12-6-data228-probe-v1",
        validation_fraction=0.20,
        min_chars=int(policy["min_chars"]),
        max_chars=int(policy["max_chars"]),
        min_alpha_ratio=float(policy["min_alpha_ratio"]),
        near_duplicate_threshold=0.92,
        near_duplicate_shingle_words=5,
        tiny_near_dedup_max_documents=5000,
    )


def _materialize_cpython(spec: Mapping[str, Any]) -> tuple[bytes, int, dict[str, Any]]:
    _require(spec.get("normalization_policy") == CPYTHON_POLICY, "CPython policy drift")
    _require(spec.get("dedicated_workflow_conclusion") == "success", "CPython authority not green")
    _require(spec.get("training") == "ALLOWED_ACCEPTED_CHUNKS_ONLY", "CPython accepted-only boundary drift")
    _require(spec.get("evaluation") == "NOT_SEPARATELY_ADMITTED", "CPython evaluation boundary drift")
    raw = v1.fetch_exact_source(str(spec["acquisition_url"]))
    _require(len(raw) == spec["raw_bytes"], "CPython raw byte-count drift")
    _require(_sha256(raw) == spec["raw_sha256"], "CPython raw SHA-256 drift")
    _require(_git_blob_sha1(raw) == spec["git_blob_sha1"], "CPython Git blob drift")
    text = raw.decode("utf-8", errors="strict")
    normalized = normalize_text(text[: int(spec["truncate_chars"])])
    normalized_bytes = normalized.encode("utf-8")
    _require(
        len(normalized_bytes) == spec["normalized_source_bytes"],
        "CPython normalized source byte-count drift",
    )
    _require(
        _sha256(normalized_bytes) == spec["normalized_source_sha256"],
        "CPython normalized source SHA-256 drift",
    )
    chunking = spec["chunking"]
    chunks = _chunk_text(
        normalized,
        max_chars=int(chunking["max_chars"]),
        min_chars=int(chunking["min_chars"]),
    )
    _require(len(chunks) == spec["chunk_count"], "CPython chunk-count drift")
    quality_config = _cpython_quality_config(spec)
    accepted: list[bytes] = []
    accepted_hashes: list[str] = []
    reasons: Counter[str] = Counter()
    for chunk in chunks:
        reason = _quality_reason(chunk, quality_config)
        if reason is not None:
            reasons[reason] += 1
            continue
        payload = normalize_text(chunk).encode("utf-8")
        accepted.append(payload)
        accepted_hashes.append(_sha256(payload))
    _require(len(accepted) == spec["accepted_chunk_count"], "CPython accepted count drift")
    _require(accepted_hashes == spec["accepted_normalized_sha256"], "CPython accepted identity/order drift")
    _require(len(chunks) - len(accepted) == spec["rejected_chunk_count"], "CPython rejected count drift")
    _require(dict(sorted(reasons.items())) == spec["rejection_reasons"], "CPython rejection-reason drift")
    capacity = sum(len(item) for item in accepted)
    payload = b"\n\n".join(accepted)
    _require(0 < capacity < int(spec["normalized_source_bytes"]), "CPython accepted-only capacity invariant failed")
    return payload, capacity, {
        "source_id": spec["source_id"],
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "git_blob_sha1": _git_blob_sha1(raw),
        "normalized_source_bytes": len(normalized_bytes),
        "normalized_source_sha256": _sha256(normalized_bytes),
        "chunk_count": len(chunks),
        "accepted_chunk_count": len(accepted),
        "accepted_chunk_sha256": accepted_hashes,
        "accepted_capacity_bytes": capacity,
        "comparison_payload_bytes": len(payload),
        "comparison_payload_sha256": _sha256(payload),
        "rejected_chunk_count": len(chunks) - len(accepted),
        "rejection_reasons": dict(sorted(reasons.items())),
        "normalization_policy": CPYTHON_POLICY,
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == INVENTORY_SCHEMA, "unsupported V5 config schema")
    _require(config.get("worker_id") == WORKER_ID, "V5 worker id drift")
    _require(config.get("local_free_only") is True, "LOCAL_FREE boundary weakened")
    for key in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
    ):
        _require(config.get(key) is False, f"execution boundary weakened: {key}")
    base = config["base_v4"]
    _require(base.get("head_sha") == "5738bb8bac8fda058d5ae9c1361c4a0c3756f360", "V4 head drift")
    _require(base.get("source_object_count") == 21, "V4 source count drift")
    _require(base.get("source_family_counts") == {"uk": 4, "en": 2, "code": 4}, "V4 family vector drift")
    _require(base.get("source_capacity_bytes") == {"uk": 100856, "en": 144151, "code": 69133, "total": 314140}, "V4 capacity vector drift")
    expected = config["expected_vector"]
    _require(expected.get("source_object_count") == 23, "V5 expected source count drift")
    _require(expected.get("source_family_counts") == {"uk": 4, "en": 4, "code": 4}, "V5 expected family vector drift")
    _require(expected.get("fixed_capacity_without_cpython_accepted_chunks") == {"uk": 100856, "en": 150643, "code": 69133, "total": 320632}, "V5 fixed-capacity vector drift")
    boundary = config["claim_boundary"]
    for key in (
        "canonical_registry_replaced",
        "corpus_materialized",
        "decontamination_pass_claimed",
        "balance_release_claimed",
        "postpack_unique_loss_ledger_complete",
        "tokenizer_fit_authorized",
        "training_authorized",
        "paid_compute_authorized",
        "research_corpus_v1_terminal",
    ):
        _require(boundary.get(key) is False, f"claim boundary weakened: {key}")


def _materialize_v4(
    base_inventory: Mapping[str, Any], v4_extension: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, bytes], list[dict[str, Any]]]:
    additions = v4._validate_extension(base_inventory, v4_extension)
    merged = v4.compose_v3_inventory(base_inventory, v4_extension)
    payloads: dict[str, bytes] = {}
    evidence: list[dict[str, Any]] = []
    for row in base_inventory["sources"]:
        payloads[row["source_id"]] = v1.fetch_exact_source(row["acquisition_url"])
    for row in additions:
        raw = v1.fetch_exact_source(row["acquisition_url"])
        if row.get("comparison_normalization") == v4.NIST_POLICY:
            materialized, item = v4._nist_payload(row, raw)
            payloads[row["source_id"]] = materialized
            evidence.append(item)
        else:
            payloads[row["source_id"]] = raw
            evidence.append(
                {
                    "source_id": row["source_id"],
                    "raw_bytes": len(raw),
                    "raw_sha256": _sha256(raw),
                    "normalization_policy": "V4_GENERIC_FROM_VERIFIED_RAW",
                }
            )
    v4_dedup = v3.audit_payloads(merged, payloads)
    v3.verify_report(v4_dedup)
    return merged, payloads, evidence


def _family_counts(report: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(modality): int(summary["declared_source_family_count"])
        for modality, summary in report["terminal_candidates"]["by_modality"].items()
    }


def audit_live(
    base_inventory: Mapping[str, Any],
    v4_extension: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_config(config)
    merged, payloads, inherited_evidence = _materialize_v4(base_inventory, v4_extension)
    _require(len(merged["sources"]) == 21, "inherited V4 source count drift")

    mdn_payload, mdn_evidence = _materialize_mdn(config["mdn"])
    cpython_payload, cpython_capacity, cpython_evidence = _materialize_cpython(config["cpython"])

    mdn = config["mdn"]
    cpython = config["cpython"]
    mdn_row = {
        "source_id": mdn["source_id"],
        "source_family": mdn["source_family"],
        "stable_origin_id": mdn["stable_origin_id"],
        "stable_object_id": mdn["stable_object_id"],
        "modality": mdn["modality"],
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": f"NEXT100-038@{mdn['head_sha']} workflow {mdn['dedicated_workflow_run']}",
        "declared_capacity_bytes": mdn["normalized_bytes"],
        "expected_raw_bytes": mdn["normalized_bytes"],
        "expected_raw_sha256": mdn["normalized_sha256"],
        "acquisition_url": "materialized-v5://mdn-prose-only",
        "origin_key": f"github:mdn/content:{mdn['upstream_commit']}:{mdn['upstream_path']}",
    }
    cpython_row = {
        "source_id": cpython["source_id"],
        "source_family": cpython["source_family"],
        "stable_origin_id": cpython["stable_origin_id"],
        "stable_object_id": cpython["stable_object_id"],
        "modality": cpython["modality"],
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": f"NEXT100-037@{cpython['head_sha']} workflow {cpython['dedicated_workflow_run']} accepted-only",
        "declared_capacity_bytes": cpython_capacity,
        "expected_raw_bytes": len(cpython_payload),
        "expected_raw_sha256": _sha256(cpython_payload),
        "acquisition_url": "materialized-v5://cpython-data228-accepted-chunks",
        "origin_key": f"github:python/cpython:{cpython['upstream_commit']}:{cpython['upstream_path']}:accepted-only",
    }

    final_inventory = copy.deepcopy(merged)
    final_inventory["sources"] = [*copy.deepcopy(merged["sources"]), mdn_row, cpython_row]
    final_inventory["final_refresh_required"] = False
    final_inventory["terminal_refresh_cutoff_utc"] = "2026-08-26T19:45:00Z"
    final_inventory["terminal_refresh_rule"] = (
        "V5 re-executes V4 and adds exact-green NEXT100-038 plus accepted-only NEXT100-037; "
        "failed/nonterminal sibling authorities receive zero credit."
    )
    payloads = dict(payloads)
    payloads[mdn["source_id"]] = mdn_payload
    payloads[cpython["source_id"]] = cpython_payload

    dedup = v3.audit_payloads(final_inventory, payloads)
    v3.verify_report(dedup)
    _require(dedup["source_count"] == 23, "V5 source-count drift")
    counts = _family_counts(dedup)
    _require(counts == config["expected_vector"]["source_family_counts"], f"V5 family-vector drift: {counts}")
    scope = dedup["terminal_candidates"]
    fixed = config["expected_vector"]["fixed_capacity_without_cpython_accepted_chunks"]
    expected_total = int(fixed["total"]) + cpython_capacity
    expected_en = int(fixed["en"]) + cpython_capacity
    _require(scope["declared_capacity_bytes_before"] == expected_total, "V5 total capacity arithmetic drift")
    _require(scope["by_modality"]["en"]["declared_capacity_bytes_before"] == expected_en, "V5 EN capacity arithmetic drift")
    _require(cpython_capacity != config["expected_vector"]["full_cpython_normalized_bytes_must_not_be_credited"], "full CPython normalized bytes were incorrectly credited")

    core = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "base_v4_head_sha": config["base_v4"]["head_sha"],
        "source_vector": {
            "source_object_count": dedup["source_count"],
            "source_family_counts": counts,
            "fixed_capacity_without_cpython_accepted_chunks": copy.deepcopy(fixed),
            "cpython_accepted_capacity_bytes": cpython_capacity,
            "source_capacity_bytes_before_global_dedup": expected_total,
            "source_capacity_by_modality_before_global_dedup": {
                "uk": fixed["uk"],
                "en": expected_en,
                "code": fixed["code"],
            },
            "conservative_unique_capacity_bytes_after_global_dedup": scope[
                "conservative_unique_capacity_bytes_after"
            ],
        },
        "materialization_evidence": sorted(
            [*inherited_evidence, mdn_evidence, cpython_evidence],
            key=lambda item: item["source_id"],
        ),
        "dedup_v3": dedup,
        "claim_boundary": copy.deepcopy(config["claim_boundary"]),
        "remaining_blockers": [
            "SOURCE_CAPACITY_FAR_BELOW_20M_RESEARCH_TARGET",
            "FINAL_RECORD_GRANULARITY_QUALITY_PRIVACY_REVALIDATION",
            "EVALUATION_SELECTION_RESERVATIONS_AND_DECONTAMINATION",
            "IMMUTABLE_SPLIT_AND_PACKING",
            "POSTPACK_UNIQUE_LOSS_LEDGER",
            "TOKENIZER_FIT_AUTHORIZATION",
            "CHECKPOINT_D05_TERMINAL_INTEGRITY",
            "MATERIAL_COMPUTE_AUTHORIZATION"
        ],
        "raw_text_emitted": False,
    }
    return {**core, "report_sha256": _sha256(_canonical_bytes(core))}


def verify_report(report: Mapping[str, Any]) -> None:
    _require(report.get("schema_version") == SCHEMA, "unsupported V5 report schema")
    expected = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    _require(expected == _sha256(_canonical_bytes(core)), "V5 report self-hash mismatch")
    _require(report.get("local_free_only") is True, "V5 LOCAL_FREE invariant failed")
    for key in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
        "raw_text_emitted",
    ):
        _require(report.get(key) is False, f"V5 execution/text boundary failed: {key}")
    vector = report["source_vector"]
    _require(vector.get("source_object_count") == 23, "V5 report source count drift")
    _require(vector.get("source_family_counts") == {"code": 4, "en": 4, "uk": 4}, "V5 report family vector drift")
    cp_capacity = vector.get("cpython_accepted_capacity_bytes")
    _require(isinstance(cp_capacity, int) and 0 < cp_capacity < 17901, "V5 CPython accepted capacity invalid")
    _require(vector.get("source_capacity_bytes_before_global_dedup") == 320632 + cp_capacity, "V5 report total arithmetic drift")
    cp_evidence = next(
        item
        for item in report["materialization_evidence"]
        if item["source_id"] == "en.python.docs.tutorial-introduction"
    )
    _require(cp_evidence.get("accepted_chunk_count") == 14, "V5 CPython accepted count drift")
    _require(cp_evidence.get("rejected_chunk_count") == 2, "V5 CPython rejected count drift")
    _require(cp_evidence.get("rejection_reasons") == {"pii_phone": 2}, "V5 CPython privacy rejection drift")
    boundary = report["claim_boundary"]
    for key in (
        "canonical_registry_replaced",
        "corpus_materialized",
        "decontamination_pass_claimed",
        "balance_release_claimed",
        "postpack_unique_loss_ledger_complete",
        "tokenizer_fit_authorized",
        "training_authorized",
        "paid_compute_authorized",
        "research_corpus_v1_terminal",
    ):
        _require(boundary.get(key) is False, f"V5 truth boundary weakened: {key}")
    v3.verify_report(report["dedup_v3"])


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    v1.write_report(report, path)


def load_inputs(
    base_inventory_path: str | Path,
    v4_extension_path: str | Path,
    v5_config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        _load_json(base_inventory_path),
        _load_json(v4_extension_path),
        _load_json(v5_config_path),
    )
