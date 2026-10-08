"""Plan 2 / Section 7: versioned near/global dedup over verified S6 candidates.

This is an evidence adapter: S6 remains the sole exact dedup and S5 remains the
sole privacy authority. No corpus, tokenizer or evaluation admission is granted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools import plan2_exact_dedup_v1 as exact
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-near-family-candidate.v1"
POLICY_SCHEMA = "12-6.plan2-near-dedup-policy.v1"
POLICY_FILE = "configs/data/plan2_near_policy_v1.json"
AUDIT_FILE = "configs/data/plan2_near_audit_samples_v1.json"
POLICY = {
    "schema_version": POLICY_SCHEMA,
    "algorithm": "NFKC_CASEFOLD_WORD_TRIGRAM_SET_JACCARD_CONTAINMENT_V1",
    "jaccard_min": 0.82,
    "containment_min": 0.94,
    "containment_length_ratio_min": 0.50,
    "min_near_shingles": 3,
    "family_cap": 1,
    "max_candidate_pairs": 200000,
    "output": "TEXT_FREE_CLUSTER_PROVENANCE_ONLY",
    "admission": "SOURCE_CANDIDATE_ONLY",
}


class NearDedupError(ValueError):
    """Untrusted upstream cohort, policy or immutable near-family publication."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise NearDedupError(message)


def _canonical(value: Any) -> bytes:
    return exact._canonical(value)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_json(root: Path, relative: str) -> Any:
    path = root / relative
    _require(path.is_file() and not path.is_symlink(),
             "missing or symlinked near dedup authority")
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, val in values:
            _require(key not in result, "duplicate authority JSON key")
            result[key] = val
        return result
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=pairs)
    except (UnicodeError, ValueError) as exc:
        raise NearDedupError("invalid near dedup authority JSON") from exc


def _policy(value: Mapping[str, Any]) -> None:
    _require(isinstance(value, Mapping) and dict(value) == POLICY,
             "unsupported near dedup policy")


def _shingles(raw: bytes) -> frozenset[str]:
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeError as exc:
        raise NearDedupError("non-UTF8 candidate record") from exc
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text.casefold()))
    if len(words) < 3:
        return frozenset()
    return frozenset(" ".join(words[i:i + 3]) for i in range(len(words) - 2))


def _near(left: frozenset[str], right: frozenset[str]) -> bool:
    if min(len(left), len(right)) < POLICY["min_near_shingles"]:
        return False
    common = len(left & right)
    union = len(left) + len(right) - common
    ratio = min(len(left), len(right)) / max(len(left), len(right))
    return (
        common / union >= POLICY["jaccard_min"]
        or (common / min(len(left), len(right)) >= POLICY["containment_min"]
            and ratio >= POLICY["containment_length_ratio_min"])
    )


def audit_samples(samples: Mapping[str, Any]) -> dict[str, Any]:
    """Versioned synthetic ground-truth audit; reports both FP and FN."""
    _require(isinstance(samples, Mapping)
             and samples.get("schema_version") == "12-6.plan2-near-audit.v1"
             and isinstance(samples.get("pairs"), list)
             and len(samples["pairs"]) >= 4,
             "missing versioned near-dedup audit pairs")
    identities = set()
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for row in samples["pairs"]:
        _require(type(row) is dict
                 and set(row) == {"id", "left", "right", "related"}
                 and type(row["id"]) is str and row["id"] not in identities
                 and type(row["left"]) is str and type(row["right"]) is str
                 and type(row["related"]) is bool,
                 "malformed or duplicate audit pair")
        identities.add(row["id"])
        outcome = _near(_shingles(row["left"].encode()),
                        _shingles(row["right"].encode()))
        if outcome:
            confusion["tp" if row["related"] else "fp"] += 1
        else:
            confusion["fn" if row["related"] else "tn"] += 1
    _require(confusion["tp"] and confusion["tn"],
             "audit lacks observable positive or negative controls")
    _require(confusion["fp"] == 0 and confusion["fn"] == 0,
             "near threshold fails versioned audit samples")
    return {
        "audit_schema": samples["schema_version"],
        "audit_sha256": _sha(_canonical(samples)),
        "confusion": confusion,
        "sample_count": len(samples["pairs"]),
    }


def inspect_near(
    cohorts: Sequence[tuple[Mapping[str, Any], bytes, Mapping[str, Any]]],
    *, audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify full S6 lineage, then cluster only previously retained records."""
    prior = exact.inspect_exact(cohorts)
    _require(prior["training_corpus_authorized"] is False
             and prior["evaluation_authorized"] is False,
             "S6 authority escalation")
    required = set(prior["retained_record_ids"])
    payloads: dict[str, bytes] = {}
    for normalized, payload, receipt in cohorts:
        clean = {r["record_id"] for r in receipt["clean_record_hashes"]}
        for part in normalized["records"]:
            rid = f"{normalized['source_id']}:r{part['index']:08d}"
            if rid in clean and rid in required:
                _require(rid not in payloads, "duplicate retained record")
                payloads[rid] = payload[part["start_byte"]:part["end_byte"]]
    _require(set(payloads) == required, "incomplete verified S6 retained record data")
    fingerprints = {key: _shingles(raw) for key, raw in sorted(payloads.items())}
    parent = {key: key for key in fingerprints}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        roots = sorted((find(a), find(b)))
        parent[roots[1]] = roots[0]

    # Inverted shingle index: disjoint records never enter the pair comparator.
    posting: dict[str, list[str]] = defaultdict(list)
    comparisons: set[tuple[str, str]] = set()
    evidence: list[dict[str, Any]] = []
    for rid in sorted(fingerprints):
        sig = fingerprints[rid]
        candidates = set()
        for key in sig:
            candidates.update(posting[key])
        for other in sorted(candidates):
            _require(len(comparisons) < POLICY["max_candidate_pairs"],
                     "candidate pair resource bound exceeded; not silent pass")
            pair = (other, rid)
            comparisons.add(pair)
            if _near(fingerprints[other], sig):
                union(other, rid)
                evidence.append({
                    "left_record_id": other,
                    "right_record_id": rid,
                    "reason": POLICY["algorithm"],
                })
        for key in sig:
            posting[key].append(rid)
    groups: dict[str, list[str]] = defaultdict(list)
    for rid in sorted(parent):
        groups[find(rid)].append(rid)
    families = []
    excluded = []
    for members in sorted(groups.values()):
        if len(members) <= POLICY["family_cap"]:
            continue
        representative = members[0]
        identity = _sha(_canonical({"members": members, "policy": POLICY_SCHEMA}))
        families.append({
            "family_sha256": identity,
            "representative_record_id": representative,
            "members": members,
            "family_cap": POLICY["family_cap"],
        })
        for rid in members[1:]:
            excluded.append({
                "record_id": rid,
                "representative_record_id": representative,
                "family_sha256": identity,
                "reason": "NEAR_OR_MIRROR_FAMILY_CAP",
            })
    retained = sorted(set(required) - {r["record_id"] for r in excluded})
    _require(len(retained) + len(excluded) == len(required),
             "near family accounting mismatch")
    core = {
        "schema_version": SCHEMA,
        "policy": POLICY,
        "policy_sha256": _sha(_canonical(POLICY)),
        "parent_exact_manifest_sha256": prior["manifest_sha256"],
        "audit": audit_samples(audit),
        "input_exact_retained_count": len(required),
        "near_retained_count": len(retained),
        "near_excluded_count": len(excluded),
        "candidate_pairs_compared": len(comparisons),
        "retained_record_ids": retained,
        "excluded_near_records": excluded,
        "near_families": families,
        "match_evidence": sorted(evidence, key=lambda x: (
            x["left_record_id"], x["right_record_id"])),
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
        "raw_text_emitted": False,
    }
    return {**core, "manifest_sha256": _sha(_canonical(core))}


def stage_near(root: Path, destination: Path) -> dict[str, Any]:
    _require(not any(p.is_symlink() for p in (destination, *destination.parents)),
             "symlink destination")
    root = root.resolve()
    _policy(_load_json(root, POLICY_FILE))
    audit = _load_json(root, AUDIT_FILE)
    upstream = exact.stage_exact(root, destination / "exact")
    location = destination / "exact" / "privacy"
    normalized = json.loads(_read_destination(
        location / "normalization" / "normalization-manifest.json"))
    payload = _read_destination(location / "normalization" / "cohort" / "normalized.utf8")
    privacy_receipt = json.loads(_read_destination(
        location / "privacy-manifest.json"))
    result = inspect_near([(normalized, payload, privacy_receipt)], audit=audit)
    _require(result["parent_exact_manifest_sha256"] == upstream["manifest_sha256"],
             "upstream exact dedup identity mismatch")
    target = destination / "near-dedup-manifest.json"
    expected = _canonical(result)
    if target.exists() or target.is_symlink():
        _require(_read_destination(target) == expected,
                 "immutable near dedup manifest drift")
    else:
        _atomic_write(destination, target, expected)
    _require(_read_destination(target) == expected, "near dedup readback drift")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan-2 S7 near family candidate")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = stage_near(args.root, args.out_dir)
    print(json.dumps({
        "status": "PASS_SOURCE_CANDIDATE_ONLY",
        "manifest_sha256": receipt["manifest_sha256"],
        "near_retained_count": receipt["near_retained_count"],
        "near_excluded_count": receipt["near_excluded_count"],
        "training_corpus_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
