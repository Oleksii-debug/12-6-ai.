"""DATA-298 cross-source deduplication and conservative capacity accounting."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
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
    rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"{rendered}\n".encode()


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
    return frozenset(
        "\x1f".join(tokens[index : index + width])
        for index in range(len(tokens) - width + 1)
    )


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
    allowed_status = set().union(*STATUS_SCOPES.values())
    seen_ids: set[str] = set()
    result: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise CapacityAuditError("source entry must be an object")
        row = dict(raw)
        for key in (
            "source_id",
            "source_family",
            "modality",
            "evidence_status",
            "acquisition_url",
            "origin_key",
        ):
            if not isinstance(row.get(key), str) or not row[key]:
                raise CapacityAuditError(f"source missing {key}")
        source_id = row["source_id"]
        if source_id in seen_ids:
            raise CapacityAuditError(f"duplicate source_id: {source_id}")
        seen_ids.add(source_id)
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
        raise CapacityAuditError(f"{row['source_id']}: raw size changed")
    expected_sha = row.get("expected_raw_sha256")
    if expected_sha is not None and _sha256(payload) != expected_sha:
        raise CapacityAuditError(f"{row['source_id']}: raw SHA-256 mismatch")
    expected_blob = row.get("expected_git_blob_sha1")
    if expected_blob is not None and _git_blob_sha1(payload) != expected_blob:
        raise CapacityAuditError(f"{row['source_id']}: Git blob SHA-1 mismatch")


def _fingerprint(row: Mapping[str, Any], payload: bytes) -> dict[str, Any]:
    _verify_payload(row, payload)
    try:
        text = payload.decode(errors="strict")
    except UnicodeDecodeError as exc:
        raise CapacityAuditError(f"{row['source_id']}: source is not strict UTF-8") from exc
    modality = str(row["modality"])
    normalized = normalize_for_contamination(text, modality)
    tokens = tuple(TOKEN_RE.findall(normalized))
    is_code = modality == "code"
    width_key = "code_shingle_tokens" if is_code else "natural_shingle_tokens"
    skeleton = code_skeleton_tokens(text) if is_code else ()
    normalized_bytes = normalized.encode()
    return {
        "row": dict(row),
        "text": text,
        "raw_sha256": _sha256(payload),
        "raw_bytes": len(payload),
        "normalized_sha256": _sha256(normalized_bytes),
        "normalized_utf8_bytes": len(normalized_bytes),
        "tokens": tokens,
        "shingles": _shingles(tokens, int(DEFAULT_THRESHOLDS[width_key])),
        "skeleton": skeleton,
        "skeleton_shingles": _shingles(
            skeleton,
            int(DEFAULT_THRESHOLDS["code_shingle_tokens"]),
        ),
    }


def _publisher_boilerplate(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    def edge_lines(text: str) -> frozenset[str]:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        values = {
            normalize_for_contamination(line, "text")
            for line in lines[:10] + lines[-10:]
        }
        return frozenset(value for value in values if 32 <= len(value) <= 320)

    shared = edge_lines(left["text"]) & edge_lines(right["text"])
    shared_chars = sum(map(len, shared))
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
    a, b = left["row"], right["row"]
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
    normalized_exact = left["normalized_sha256"] == right["normalized_sha256"]
    if normalized_exact and left["raw_sha256"] != right["raw_sha256"]:
        add("normalized_exact", 1.0)

    same_code = a["modality"] == b["modality"] == "code"
    both_natural = a["modality"] != "code" and b["modality"] != "code"
    if same_code or both_natural:
        prefix = "code" if same_code else "natural"
        near = _jaccard(left["shingles"], right["shingles"])
        if near >= float(DEFAULT_THRESHOLDS[f"{prefix}_near_jaccard"]) and not normalized_exact:
            add("near_match", near)
        containment = _containment(left["shingles"], right["shingles"])
        enough_tokens = min(len(left["tokens"]), len(right["tokens"])) >= int(
            DEFAULT_THRESHOLDS[f"{prefix}_fragment_min_tokens"]
        )
        exact_seen = any(
            match["match_type"] in {"raw_exact", "normalized_exact"}
            for match in matches
        )
        if (
            enough_tokens
            and containment >= float(DEFAULT_THRESHOLDS[f"{prefix}_fragment_containment"])
            and not exact_seen
        ):
            add("document_fragment", containment)

    if same_code:
        copy_score = _jaccard(left["skeleton_shingles"], right["skeleton_shingles"])
        enough_skeleton = min(len(left["skeleton"]), len(right["skeleton"])) >= int(
            DEFAULT_THRESHOLDS["code_copy_min_tokens"]
        )
        if enough_skeleton and copy_score >= float(DEFAULT_THRESHOLDS["code_copy_jaccard"]):
            add("code_fork_copy", copy_score)

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


def _components(ids: Sequence[str], pairs: Sequence[tuple[str, str]]) -> list[list[str]]:
    parent = {value: value for value in ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left, right in pairs:
        a, b = find(left), find(right)
        if a != b:
            if a > b:
                a, b = b, a
            parent[b] = a
    groups: dict[str, list[str]] = defaultdict(list)
    for value in ids:
        groups[find(value)].append(value)
    return [sorted(group) for _, group in sorted(groups.items())]


def _scope_summary(
    fingerprints: Sequence[dict[str, Any]],
    matches: Sequence[dict[str, Any]],
    statuses: set[str],
) -> dict[str, Any]:
    selected = [item for item in fingerprints if item["row"]["evidence_status"] in statuses]
    ids = {item["row"]["source_id"] for item in selected}
    by_id = {item["row"]["source_id"]: item for item in selected}
    collapse = [
        (match["left_source_id"], match["right_source_id"])
        for match in matches
        if match["capacity_collapsing"]
        and match["match_type"] in COLLAPSE_MATCH_TYPES
        and match["left_source_id"] in ids
        and match["right_source_id"] in ids
    ]
    components = _components(sorted(ids), collapse)
    duplicates = [component for component in components if len(component) > 1]
    before = sum(item["row"]["declared_capacity_bytes"] for item in selected)
    after = sum(
        max(by_id[source_id]["row"]["declared_capacity_bytes"] for source_id in component)
        for component in components
    )
    raw_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        raw_groups[item["raw_sha256"]].append(item)
    families = sorted({item["row"]["source_family"] for item in selected})
    family_pairs = sorted(
        {
            tuple(sorted((match["left_source_family"], match["right_source_family"])))
            for match in matches
            if match["capacity_collapsing"]
            and match["match_type"] in COLLAPSE_MATCH_TYPES
            and match["cross_source_family"]
            and match["left_source_id"] in ids
            and match["right_source_id"] in ids
        }
    )
    family_components = _components(families, family_pairs)
    discount = before - after
    return {
        "source_count": len(selected),
        "declared_family_count": len(families),
        "effective_independent_family_count": len(family_components),
        "raw_bytes_before": sum(item["raw_bytes"] for item in selected),
        "raw_exact_unique_bytes_after": sum(
            max(item["raw_bytes"] for item in group) for group in raw_groups.values()
        ),
        "declared_capacity_bytes_before": before,
        "conservative_unique_capacity_bytes_after": after,
        "duplicate_discount_bytes": discount,
        "duplicate_discount_fraction": round(discount / before, 12) if before else 0.0,
        "duplicate_cluster_count": len(duplicates),
        "duplicate_clusters": duplicates,
        "family_clusters": family_components,
    }


def audit_payloads(inventory: Mapping[str, Any], payloads: Mapping[str, bytes]) -> dict[str, Any]:
    rows = _validate_inventory(inventory)
    expected_ids = {row["source_id"] for row in rows}
    if set(payloads) != expected_ids:
        missing = sorted(expected_ids - set(payloads))
        extra = sorted(set(payloads) - expected_ids)
        raise CapacityAuditError(f"payload coverage mismatch missing={missing} extra={extra}")
    fingerprints = [_fingerprint(row, payloads[row["source_id"]]) for row in rows]
    matches = [
        match
        for index, left in enumerate(fingerprints)
        for right in fingerprints[index + 1 :]
        for match in _pair_matches(left, right)
    ]
    matches.sort(key=lambda item: (item["left_source_id"], item["right_source_id"], item["match_type"]))
    sources = [
        {
            "source_id": item["row"]["source_id"],
            "source_family": item["row"]["source_family"],
            "modality": item["row"]["modality"],
            "evidence_status": item["row"]["evidence_status"],
            "declared_capacity_bytes": item["row"]["declared_capacity_bytes"],
            "verified_raw_bytes": item["raw_bytes"],
            "verified_raw_sha256": item["raw_sha256"],
            "normalized_utf8_bytes": item["normalized_utf8_bytes"],
            "normalized_sha256": item["normalized_sha256"],
            "origin_key_sha256": _sha256(item["row"]["origin_key"].encode()),
        }
        for item in fingerprints
    ]
    sources.sort(key=lambda item: item["source_id"])
    core = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "local_free_only": True,
        "source_admission_authority": False,
        "model_training_executed": False,
        "matching_authority": "DATA-232 deterministic overlap and code-copy semantics",
        "thresholds": dict(DEFAULT_THRESHOLDS),
        "source_count": len(fingerprints),
        "sources": sources,
        "matches": matches,
        "scopes": {
            name: _scope_summary(fingerprints, matches, statuses)
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
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    target.write_text(f"{rendered}\n", encoding="utf-8", newline="\n")
