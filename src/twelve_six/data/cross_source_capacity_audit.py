"""DATA-298 cross-source deduplication and conservative capacity accounting.

This module audits already-declared source objects.  It does not grant rights,
admit sources, or replace the DATA-230 corpus authority.  Matching thresholds and
normalization semantics are inherited from DATA-232 so capacity accounting cannot
silently use a weaker duplicate definition.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

from twelve_six.data._data232_decontamination_matching import (
    DEFAULT_THRESHOLDS,
    TOKEN_RE,
    code_skeleton_tokens,
    normalize_for_contamination,
)

SCHEMA = "12-6.data298-cross-source-capacity-audit.v1"
ALGORITHM = "data298-data232-cluster-capacity-v1"
MAX_SOURCE_BYTES = 2_000_000
COLLAPSE_MATCH_TYPES = {
    "origin_alias",
    "raw_exact",
    "normalized_exact",
    "near_match",
    "document_fragment",
    "code_fork_copy",
}
STATUS_SCOPES = {
    "canonical_registry": {"REGISTRY_TERMINAL"},
    "terminal_evidence": {"REGISTRY_TERMINAL", "DEDICATED_TERMINAL"},
    "all_observed": {"REGISTRY_TERMINAL", "DEDICATED_TERMINAL", "PROBE_NONTERMINAL"},
}


class CapacityAuditError(RuntimeError):
    """Fail-closed DATA-298 input or verification error."""


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    prefix = b"blob " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + payload).hexdigest()


def _shingles(tokens: Sequence[str], width: int) -> frozenset[str]:
    if not tokens:
        return frozenset()
    if len(tokens) < width:
        return frozenset({"\x1f".join(tokens)})
    return frozenset("\x1f".join(tokens[i : i + width]) for i in range(len(tokens) - width + 1))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def _containment(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / min(len(left), len(right)) if left and right else 0.0


def _validate_inventory(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    if inventory.get("schema_version") != "12-6.data298-cross-source-inventory.v1":
        raise CapacityAuditError("unsupported inventory schema")
    if inventory.get("local_free_only") is not True:
        raise CapacityAuditError("inventory must be LOCAL_FREE only")
    rows = inventory.get("sources")
    if not isinstance(rows, list) or not rows:
        raise CapacityAuditError("inventory requires sources")
    seen_ids: set[str] = set()
    result: list[dict[str, Any]] = []
    allowed_status = set().union(*STATUS_SCOPES.values())
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise CapacityAuditError("source entry must be an object")
        row = dict(raw)
        required_strings = (
            "source_id",
            "source_family",
            "modality",
            "evidence_status",
            "acquisition_url",
            "origin_key",
        )
        for key in required_strings:
            if not isinstance(row.get(key), str) or not row[key]:
                raise CapacityAuditError(f"source missing {key}")
        if row["source_id"] in seen_ids:
            raise CapacityAuditError(f"duplicate source_id: {row['source_id']}")
        seen_ids.add(row["source_id"])
        if row["evidence_status"] not in allowed_status:
            raise CapacityAuditError(f"unsupported evidence_status: {row['evidence_status']}")
        if row["modality"] not in {"en", "uk", "ua", "text", "code"}:
            raise CapacityAuditError(f"unsupported modality: {row['modality']}")
        for key in ("declared_capacity_bytes", "expected_raw_bytes"):
            if not isinstance(row.get(key), int) or row[key] <= 0:
                raise CapacityAuditError(f"source requires positive {key}")
        expected_sha = row.get("expected_raw_sha256")
        expected_blob = row.get("expected_git_blob_sha1")
        if expected_sha is None and expected_blob is None:
            raise CapacityAuditError("source requires immutable SHA-256 or Git blob identity")
        if expected_sha is not None and not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha)):
            raise CapacityAuditError("expected_raw_sha256 must be lowercase SHA-256")
        if expected_blob is not None and not re.fullmatch(r"[0-9a-f]{40}", str(expected_blob)):
            raise CapacityAuditError("expected_git_blob_sha1 must be lowercase Git SHA-1")
        result.append(row)
    return result


def fetch_exact_source(url: str) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "12-6-ai-DATA-298/1.0", "Accept-Encoding": "identity"},
    )
    with urlopen(request, timeout=30) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_SOURCE_BYTES:
            raise CapacityAuditError(f"oversized source: {url}")
        payload = response.read(MAX_SOURCE_BYTES + 1)
    if len(payload) > MAX_SOURCE_BYTES:
        raise CapacityAuditError(f"oversized source: {url}")
    return payload


def _verify_payload(row: Mapping[str, Any], payload: bytes) -> None:
    if len(payload) != row["expected_raw_bytes"]:
        raise CapacityAuditError(
            f"{row['source_id']}: raw size changed: {len(payload)} != {row['expected_raw_bytes']}"
        )
    expected_sha = row.get("expected_raw_sha256")
    if expected_sha is not None and _sha256(payload) != expected_sha:
        raise CapacityAuditError(f"{row['source_id']}: raw SHA-256 mismatch")
    expected_blob = row.get("expected_git_blob_sha1")
    if expected_blob is not None and _git_blob_sha1(payload) != expected_blob:
        raise CapacityAuditError(f"{row['source_id']}: Git blob SHA-1 mismatch")


def _fingerprint(row: Mapping[str, Any], payload: bytes) -> dict[str, Any]:
    _verify_payload(row, payload)
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CapacityAuditError(f"{row['source_id']}: source is not strict UTF-8") from exc
    modality = str(row["modality"])
    normalized = normalize_for_contamination(text, modality)
    tokens = tuple(TOKEN_RE.findall(normalized))
    is_code = modality == "code"
    width = int(DEFAULT_THRESHOLDS["code_shingle_tokens"] if is_code else DEFAULT_THRESHOLDS["natural_shingle_tokens"])
    skeleton = code_skeleton_tokens(text) if is_code else ()
    return {
        "row": dict(row),
        "text": text,
        "raw_sha256": _sha256(payload),
        "raw_bytes": len(payload),
        "normalized_sha256": _sha256(normalized.encode("utf-8")),
        "normalized_utf8_bytes": len(normalized.encode("utf-8")),
        "tokens": tokens,
        "shingles": _shingles(tokens, width),
        "skeleton": skeleton,
        "skeleton_shingles": _shingles(skeleton, int(DEFAULT_THRESHOLDS["code_shingle_tokens"])),
    }


def _edge_lines(text: str) -> frozenset[str]:
    raw_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    candidates = raw_lines[:10] + raw_lines[-10:]
    lines: set[str] = set()
    for line in candidates:
        value = normalize_for_contamination(line, "text")
        if 32 <= len(value) <= 320:
            lines.add(value)
    return frozenset(lines)


def _publisher_boilerplate(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    shared = _edge_lines(left["text"]) & _edge_lines(right["text"])
    shared_chars = sum(len(line) for line in shared)
    if shared_chars < 80:
        return None
    return {
        "match_type": "publisher_boilerplate",
        "score": shared_chars,
        "shared_edge_line_count": len(shared),
        "shared_edge_characters": shared_chars,
        "capacity_collapsing": False,
    }


def _pair_matches(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    a = left["row"]
    b = right["row"]
    matches: list[dict[str, Any]] = []

    def add(kind: str, score: float, *, collapsing: bool = True) -> None:
        matches.append(
            {
                "left_source_id": a["source_id"],
                "right_source_id": b["source_id"],
                "left_source_family": a["source_family"],
                "right_source_family": b["source_family"],
                "match_type": kind,
                "score": round(float(score), 12),
                "cross_source_family": a["source_family"] != b["source_family"],
                "capacity_collapsing": collapsing,
            }
        )

    if a["origin_key"] == b["origin_key"]:
        add("origin_alias", 1.0)
    if left["raw_sha256"] == right["raw_sha256"]:
        add("raw_exact", 1.0)
    if left["normalized_sha256"] == right["normalized_sha256"] and left["raw_sha256"] != right["raw_sha256"]:
        add("normalized_exact", 1.0)

    same_code = a["modality"] == b["modality"] == "code"
    both_natural = a["modality"] != "code" and b["modality"] != "code"
    if same_code or both_natural:
        near = _jaccard(left["shingles"], right["shingles"])
        near_limit = float(
            DEFAULT_THRESHOLDS["code_near_jaccard"] if same_code else DEFAULT_THRESHOLDS["natural_near_jaccard"]
        )
        if near >= near_limit and left["normalized_sha256"] != right["normalized_sha256"]:
            add("near_match", near)

        min_tokens = int(
            DEFAULT_THRESHOLDS["code_fragment_min_tokens"] if same_code else DEFAULT_THRESHOLDS["natural_fragment_min_tokens"]
        )
        frag_limit = float(
            DEFAULT_THRESHOLDS["code_fragment_containment"] if same_code else DEFAULT_THRESHOLDS["natural_fragment_containment"]
        )
        frag = _containment(left["shingles"], right["shingles"])
        if min(len(left["tokens"]), len(right["tokens"])) >= min_tokens and frag >= frag_limit:
            if not any(x["match_type"] in {"raw_exact", "normalized_exact"} for x in matches):
                add("document_fragment", frag)

    if same_code:
        copy = _jaccard(left["skeleton_shingles"], right["skeleton_shingles"])
        if min(len(left["skeleton"]), len(right["skeleton"])) >= int(DEFAULT_THRESHOLDS["code_copy_min_tokens"]):
            if copy >= float(DEFAULT_THRESHOLDS["code_copy_jaccard"]):
                add("code_fork_copy", copy)

    boilerplate = _publisher_boilerplate(left, right)
    if boilerplate is not None:
        boilerplate.update(
            {
                "left_source_id": a["source_id"],
                "right_source_id": b["source_id"],
                "left_source_family": a["source_family"],
                "right_source_family": b["source_family"],
                "cross_source_family": a["source_family"] != b["source_family"],
            }
        )
        matches.append(boilerplate)
    return matches


def _components(source_ids: Sequence[str], collapse_pairs: Sequence[tuple[str, str]]) -> list[list[str]]:
    parent = {source_id: source_id for source_id in source_ids}

    def find(value: str) -> str:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            nxt = parent[value]
            parent[value] = root
            value = nxt
        return root

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            if a > b:
                a, b = b, a
            parent[b] = a

    for left, right in collapse_pairs:
        union(left, right)
    groups: dict[str, list[str]] = defaultdict(list)
    for source_id in source_ids:
        groups[find(source_id)].append(source_id)
    return [sorted(group) for _, group in sorted(groups.items())]


def _scope_summary(
    fingerprints: Sequence[dict[str, Any]],
    matches: Sequence[dict[str, Any]],
    statuses: set[str],
) -> dict[str, Any]:
    selected = [fp for fp in fingerprints if fp["row"]["evidence_status"] in statuses]
    ids = {fp["row"]["source_id"] for fp in selected}
    by_id = {fp["row"]["source_id"]: fp for fp in selected}
    collapse_pairs = [
        (m["left_source_id"], m["right_source_id"])
        for m in matches
        if m["capacity_collapsing"]
        and m["match_type"] in COLLAPSE_MATCH_TYPES
        and m["left_source_id"] in ids
        and m["right_source_id"] in ids
    ]
    components = _components(sorted(ids), collapse_pairs)
    duplicate_components = [component for component in components if len(component) > 1]

    declared_before = sum(fp["row"]["declared_capacity_bytes"] for fp in selected)
    conservative_after = sum(
        max(by_id[source_id]["row"]["declared_capacity_bytes"] for source_id in component)
        for component in components
    )
    raw_before = sum(fp["raw_bytes"] for fp in selected)
    raw_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fp in selected:
        raw_groups[fp["raw_sha256"]].append(fp)
    raw_exact_after = sum(max(fp["raw_bytes"] for fp in group) for group in raw_groups.values())

    families = sorted({fp["row"]["source_family"] for fp in selected})
    family_pairs = sorted(
        {
            tuple(sorted((m["left_source_family"], m["right_source_family"])))
            for m in matches
            if m["capacity_collapsing"]
            and m["match_type"] in COLLAPSE_MATCH_TYPES
            and m["cross_source_family"]
            and m["left_source_id"] in ids
            and m["right_source_id"] in ids
        }
    )
    effective_family_components = _components(families, family_pairs)

    return {
        "source_count": len(selected),
        "declared_family_count": len(families),
        "effective_independent_family_count": len(effective_family_components),
        "raw_bytes_before": raw_before,
        "raw_exact_unique_bytes_after": raw_exact_after,
        "declared_capacity_bytes_before": declared_before,
        "conservative_unique_capacity_bytes_after": conservative_after,
        "duplicate_discount_bytes": declared_before - conservative_after,
        "duplicate_discount_fraction": round((declared_before - conservative_after) / declared_before, 12)
        if declared_before
        else 0.0,
        "duplicate_cluster_count": len(duplicate_components),
        "duplicate_clusters": duplicate_components,
        "family_clusters": effective_family_components,
    }


def audit_payloads(inventory: Mapping[str, Any], payloads: Mapping[str, bytes]) -> dict[str, Any]:
    rows = _validate_inventory(inventory)
    expected_ids = {row["source_id"] for row in rows}
    if set(payloads) != expected_ids:
        missing = sorted(expected_ids - set(payloads))
        extra = sorted(set(payloads) - expected_ids)
        raise CapacityAuditError(f"payload coverage mismatch missing={missing} extra={extra}")
    fingerprints = [_fingerprint(row, payloads[row["source_id"]]) for row in rows]
    pair_matches: list[dict[str, Any]] = []
    for i, left in enumerate(fingerprints):
        for right in fingerprints[i + 1 :]:
            pair_matches.extend(_pair_matches(left, right))
    pair_matches.sort(key=lambda x: (x["left_source_id"], x["right_source_id"], x["match_type"]))

    public_sources = [
        {
            "source_id": fp["row"]["source_id"],
            "source_family": fp["row"]["source_family"],
            "modality": fp["row"]["modality"],
            "evidence_status": fp["row"]["evidence_status"],
            "declared_capacity_bytes": fp["row"]["declared_capacity_bytes"],
            "verified_raw_bytes": fp["raw_bytes"],
            "verified_raw_sha256": fp["raw_sha256"],
            "normalized_utf8_bytes": fp["normalized_utf8_bytes"],
            "normalized_sha256": fp["normalized_sha256"],
            "origin_key_sha256": _sha256(fp["row"]["origin_key"].encode("utf-8")),
        }
        for fp in fingerprints
    ]
    public_sources.sort(key=lambda x: x["source_id"])

    core = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "local_free_only": True,
        "source_admission_authority": False,
        "model_training_executed": False,
        "matching_authority": "DATA-232 deterministic overlap and code-copy semantics",
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "source_count": len(fingerprints),
        "sources": public_sources,
        "matches": pair_matches,
        "scopes": {
            name: _scope_summary(fingerprints, pair_matches, statuses)
            for name, statuses in STATUS_SCOPES.items()
        },
        "capacity_policy": {
            "claim_metric": "conservative_unique_capacity_bytes_after",
            "duplicate_cluster_rule": "count at most the largest declared capacity member in each connected duplicate cluster",
            "publisher_boilerplate_rule": "report edge boilerplate but do not collapse capacity on boilerplate alone",
            "nonterminal_rule": "PROBE_NONTERMINAL sources appear only in all_observed scope",
        },
        "raw_text_emitted": False,
    }
    return {**core, "report_sha256": _sha256(_canonical_bytes(core))}


def audit_live(inventory: Mapping[str, Any]) -> dict[str, Any]:
    rows = _validate_inventory(inventory)
    payloads = {row["source_id"]: fetch_exact_source(row["acquisition_url"]) for row in rows}
    return audit_payloads(inventory, payloads)


def verify_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA:
        raise CapacityAuditError("unsupported report schema")
    expected = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    if expected != _sha256(_canonical_bytes(core)):
        raise CapacityAuditError("report self-hash mismatch")
    if report.get("raw_text_emitted") is not False:
        raise CapacityAuditError("raw text emission is forbidden")
    if report.get("local_free_only") is not True or report.get("model_training_executed") is not False:
        raise CapacityAuditError("LOCAL_FREE/no-training invariant failed")
    for scope in report.get("scopes", {}).values():
        if scope["conservative_unique_capacity_bytes_after"] > scope["declared_capacity_bytes_before"]:
            raise CapacityAuditError("deduplication inflated capacity")


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
