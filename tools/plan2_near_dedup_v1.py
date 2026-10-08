"""Plan 2 S7: deterministic near/mirror/template families over S6 exact survivors.

This adapter consumes verified S6/S5/S4 evidence, never replaces the S6 exact
authority, and never admits data to training, evaluation, or tokenizer fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools import plan2_exact_dedup_v1 as exact
from tools.plan2_physical_materialization_v1 import _atomic_write, _json, _read_destination

SCHEMA = "12-6.plan2-near-dedup-candidate.v1"
POLICY = "NFKC_CASEFOLD_TOKEN_TEMPLATE_CHAR5_COMPLETE_LINK_V1"
AUDIT = "configs/data/plan2_near_dedup_audit_v1.json"


class NearDedupError(ValueError):
    """A source, matcher, family, audit or publication is unverified."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise NearDedupError(message)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fold(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+", _fold(value), flags=re.UNICODE)


def _chargrams(value: str) -> set[str]:
    text = _fold(value)
    return {text[i:i + 5] for i in range(max(0, len(text) - 4))}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def match_kind(left: str, right: str) -> str | None:
    """Conservative deterministic similarity; no embedding/external model."""
    if not isinstance(left, str) or not isinstance(right, str):
        raise NearDedupError("matcher requires decoded text")
    a, b = _fold(left), _fold(right)
    ta, tb = _tokens(a), _tokens(b)
    # Short generic phrases must not form near families.
    if min(len(ta), len(tb)) < 7 or min(len(a), len(b)) < 45:
        return None
    if a == b:
        return "MIRROR_CASEFOLD"
    if (len(ta) == len(tb) and ta != tb and
            any(re.search(r"\d", t) for t in ta) and
            len(set(ta) & set(tb)) >= 5 and
            [re.sub(r"\d+", "<NUM>", t) for t in ta] ==
            [re.sub(r"\d+", "<NUM>", t) for t in tb]):
        return "TEMPLATE_NUMERIC_VARIANT"
    if (min(len(a), len(b)) / max(len(a), len(b)) >= 0.75 and
            len(set(ta) & set(tb)) >= 6 and
            _jaccard(set(ta), set(tb)) >= 0.75 and
            _jaccard(_chargrams(a), _chargrams(b)) >= 0.84):
        return "NEAR_CHAR5_AND_TOKEN"
    return None


def inspect_near(
    cohorts: Sequence[tuple[Mapping[str, Any], bytes, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Reverify S6 and retain at most one survivor per complete-link family."""
    try:
        prior = exact.inspect_exact(cohorts)
    except (ValueError, KeyError, TypeError) as exc:
        raise NearDedupError("S6 exact/privacy authority invalid") from exc
    chosen = set(prior["retained_record_ids"])
    inventory = {r["record_id"]: r for r in prior["record_inventory"]}
    _need(len(inventory) == prior["input_candidate_record_count"],
          "S6 record inventory incomplete")
    text_by_id: dict[str, str] = {}
    for normalized, payload, _receipt in cohorts:
        source_id = normalized["source_id"]
        for item in normalized["records"]:
            rid = f"{source_id}:r{item['index']:08d}"
            if rid not in chosen:
                continue
            raw = payload[item["start_byte"]:item["end_byte"]]
            _need(rid in inventory and _sha(raw) ==
                  inventory[rid]["normalized_record_sha256"],
                  "S6 survivor bytes drift")
            try:
                text_by_id[rid] = raw.decode("utf-8", "strict")
            except UnicodeDecodeError as exc:
                raise NearDedupError("S6 survivor encoding drift") from exc
    _need(set(text_by_id) == chosen, "S6 survivors missing or duplicated")
    # Stable record IDs, not caller iteration order, determine every decision.
    groups: list[list[str]] = []
    for rid in sorted(chosen):
        for group in groups:
            if all(match_kind(text_by_id[rid], text_by_id[peer]) for peer in group):
                group.append(rid)
                break
        else:
            groups.append([rid])
    families: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for group in groups:
        if len(group) < 2:
            continue
        representative = group[0]
        kinds = sorted({match_kind(text_by_id[a], text_by_id[b])
                        for i, a in enumerate(group) for b in group[i + 1:]})
        families.append({
            "family_id_sha256": _sha(_canonical(group)),
            "representative_record_id": representative,
            "members": group,
            "pair_match_kinds": kinds,
            "retained_family_cap": 1,
        })
        excluded.extend({
            "record_id": rid, "representative_record_id": representative,
            "reason": "NEAR_FAMILY_CAP_V1",
        } for rid in group[1:])
    removed_ids = {row["record_id"] for row in excluded}
    kept = sorted(chosen - removed_ids)
    core = {
        "schema_version": SCHEMA,
        "similarity_policy": POLICY,
        "upstream_exact_manifest_sha256": prior["manifest_sha256"],
        "input_exact_survivor_count": len(chosen),
        "retained_record_count": len(kept),
        "suppressed_near_record_count": len(excluded),
        "retained_record_ids": kept,
        "excluded_near_records": sorted(excluded, key=lambda x: x["record_id"]),
        "near_families": families,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
        "raw_text_emitted": False,
    }
    return {**core, "manifest_sha256": _sha(_canonical(core))}


def audit_samples(root: Path) -> dict[str, Any]:
    """Versioned FP/FN audit; report bounded recall rather than zero FN."""
    path = root / AUDIT
    _need(path.is_file() and not path.is_symlink(), "S7 audit fixture missing")
    raw = path.read_bytes()
    try:
        data = _json(raw)
    except (ValueError, UnicodeError) as exc:
        raise NearDedupError("S7 audit fixture invalid") from exc
    _need(isinstance(data, dict) and set(data) == {"schema", "samples"} and
          data["schema"] == "12-6.plan2-near-audit.v1" and
          type(data["samples"]) is list and len(data["samples"]) >= 6,
          "S7 audit schema drift")
    fp = fn = tp = tn = 0
    ids: set[str] = set()
    for row in data["samples"]:
        _need(isinstance(row, dict) and set(row) == {"id", "a", "b", "is_related"}
              and type(row["id"]) is str and row["id"] not in ids and
              type(row["a"]) is str and type(row["b"]) is str and
              type(row["is_related"]) is bool, "S7 audit row invalid")
        ids.add(row["id"])
        expected, actual = row["is_related"], match_kind(row["a"], row["b"]) is not None
        if expected and actual:
            tp += 1
        elif expected:
            fn += 1
        elif actual:
            fp += 1
        else:
            tn += 1
    _need(fp == 0 and tp >= 3 and tp / (tp + fn) >= 0.75 and tn >= 2,
          "S7 false-positive or false-negative audit gate failed")
    return {"schema_version": data["schema"], "fixture_sha256": _sha(raw),
            "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn}


def stage_near(root: Path, destination: Path) -> dict[str, Any]:
    _need(not any(p.is_symlink() for p in (destination, *destination.parents)),
          "symlink destination")
    audit = audit_samples(root)
    upstream = exact.stage_exact(root, destination / "exact")
    prefix = destination / "exact" / "privacy"
    try:
        normalized = json.loads(_read_destination(
            prefix / "normalization" / "normalization-manifest.json"))
        payload = _read_destination(prefix / "normalization" / "cohort" /
                                    "normalized.utf8")
        receipt = json.loads(_read_destination(prefix / "privacy-manifest.json"))
    except (OSError, ValueError) as exc:
        raise NearDedupError("S5/S6 staged authority missing") from exc
    candidate = inspect_near([(normalized, payload, receipt)])
    _need(candidate["upstream_exact_manifest_sha256"] == upstream["manifest_sha256"],
          "S6 upstream exact identity changed")
    core = {k: v for k, v in candidate.items() if k != "manifest_sha256"}
    core["versioned_audit"] = audit
    result = {**core, "manifest_sha256": _sha(_canonical(core))}
    target = destination / "near-dedup-manifest.json"
    expected = _canonical(result)
    if target.exists() or target.is_symlink():
        _need(_read_destination(target) == expected, "immutable near manifest drift")
    else:
        _atomic_write(destination, target, expected)
    _need(_read_destination(target) == expected, "near manifest readback drift")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan 2 Section 7 near-dedup")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    receipt = stage_near(args.root, args.out_dir)
    print(json.dumps({"status": "PASS_SOURCE_CANDIDATE_ONLY",
                      "manifest_sha256": receipt["manifest_sha256"],
                      "retained_record_count": receipt["retained_record_count"],
                      "training_corpus_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
