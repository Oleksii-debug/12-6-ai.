"""Plan 2 Section 6: global exact-record dedup of verified S5 candidates.

Reuses S3/S4 identity and the incumbent G06 S5 privacy authority. It does not
admit sources, change privacy decisions, or grant training/tokenizer/eval rights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools import plan2_privacy_gate_v1 as privacy
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-exact-dedup-candidate.v1"
EQUIVALENCE = "S4_VERIFIED_NORMALIZED_UTF8_RECORD_BYTES_SHA256_V1"


class ExactDedupError(ValueError):
    """An unverified cohort, equivalence collision or publication drift."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ExactDedupError(message)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def inspect_exact(
    cohorts: Sequence[tuple[Mapping[str, Any], bytes, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Re-execute S5 privacy and deduplicate globally, irrespective of input order.

    Each tuple is (S4 normalization receipt, S4 UTF-8 bytes, S5 privacy receipt).
    The S5 receipt is never trusted merely because its own SHA is well formed.
    """
    _require(isinstance(cohorts, (tuple, list)) and bool(cohorts),
             "empty or invalid accepted candidate cohort")
    sources: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    bytes_by_digest: dict[str, bytes] = {}
    known_sources: set[str] = set()
    for normalized, payload, receipt in cohorts:
        _require(isinstance(normalized, Mapping) and isinstance(receipt, Mapping)
                 and type(payload) is bytes, "invalid cohort members")
        _require(receipt.get("training_corpus_authorized") is False
                 and receipt.get("tokenizer_fit_authorized") is False
                 and receipt.get("evaluation_authorized") is False,
                 "upstream candidate authority escalation")
        try:
            actual = privacy.inspect_normalized(
                normalized, payload,
                policy_sha256=receipt["policy_sha256"],
                tombstones=receipt["tombstones"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ExactDedupError("upstream normalization/privacy proof invalid") from exc
        _require(dict(receipt) == actual, "privacy receipt changed or was forged")
        source_id = actual["source_id"]
        _require(source_id not in known_sources, "duplicate source identity")
        known_sources.add(source_id)
        sources.append({
            "source_id": source_id,
            "normalization_manifest_sha256": normalized["manifest_sha256"],
            "privacy_manifest_sha256": actual["manifest_sha256"],
            "normalized_member_sha256": normalized["normalized_sha256"],
            "normalized_member_bytes": normalized["normalized_bytes"],
            "candidate_record_count": actual["candidate_record_count"],
            "excluded_record_count": len(actual["excluded_records"]),
        })
        clean = actual["clean_record_hashes"]
        index = {row["record_id"]: row["sha256"] for row in clean}
        _require(len(index) == len(clean) == actual["candidate_record_count"],
                 "ambiguous or duplicated privacy record identity")
        excluded = {row["record_id"] for row in actual["excluded_records"]}
        _require(not excluded.intersection(index), "excluded records reintroduced")
        offsets = normalized["records"]
        for part in offsets:
            record_id = f"{source_id}:r{part['index']:08d}"
            if record_id not in index:
                continue
            raw = payload[part["start_byte"]:part["end_byte"]]
            digest = _sha(raw)
            _require(digest == part["sha256"] == index[record_id],
                     "survivor content disagrees with verified S4/S5")
            prior = bytes_by_digest.setdefault(digest, raw)
            _require(prior == raw, "exact digest collision: fail closed")
            records.append({
                "record_id": record_id, "source_id": source_id,
                "normalized_record_sha256": digest,
                "normalized_record_bytes": len(raw),
                "equivalence_sha256": digest,
            })
    records.sort(key=lambda r: r["record_id"])
    _require(len({r["record_id"] for r in records}) == len(records),
             "global record identity collision")
    buckets: dict[str, list[str]] = defaultdict(list)
    for record in records:
        buckets[record["equivalence_sha256"]].append(record["record_id"])
    retained: list[str] = []
    removed: list[dict[str, str]] = []
    families: list[dict[str, Any]] = []
    for identity, ids in sorted(buckets.items()):
        representative = ids[0]  # ids are globally sorted
        retained.append(representative)
        for duplicate in ids[1:]:
            removed.append({
                "record_id": duplicate, "representative_record_id": representative,
                "equivalence_sha256": identity,
                "reason": "EXACT_NORMALIZED_RECORD",
            })
        if len(ids) > 1:
            families.append({
                "equivalence_sha256": identity, "representative_record_id": representative,
                "members": ids,
            })
    member_buckets: dict[str, list[str]] = defaultdict(list)
    for source in sources:
        member_buckets[source["normalized_member_sha256"]].append(source["source_id"])
    member_families = [
        {"normalized_member_sha256": key, "source_ids": sorted(ids)}
        for key, ids in sorted(member_buckets.items()) if len(ids) > 1
    ]
    core = {
        "schema_version": SCHEMA,
        "equivalence_policy": EQUIVALENCE,
        "sources": sorted(sources, key=lambda s: s["source_id"]),
        "input_candidate_record_count": len(records),
        "retained_record_count": len(retained),
        "exact_duplicate_record_count": len(removed),
        "retained_record_ids": sorted(retained),
        "excluded_exact_records": sorted(removed, key=lambda x: x["record_id"]),
        "duplicate_families": families,
        "duplicate_member_families": member_families,
        "record_inventory": records,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "evaluation_authorized": False,
        "raw_text_emitted": False,
    }
    return {**core, "manifest_sha256": _sha(_canonical(core))}


def stage_exact(root: Path, destination: Path) -> dict[str, Any]:
    """Restart-safe, immutable publication chained to physical S5 authority."""
    _require(not any(p.is_symlink() for p in (destination, *destination.parents)),
             "symlink destination")
    receipt = privacy.stage_privacy(root, destination / "privacy")
    staged = destination / "privacy" / "normalization"
    try:
        normalized = json.loads(
            _read_destination(staged / "normalization-manifest.json"))
        payload = _read_destination(staged / "cohort" / "normalized.utf8")
    except (OSError, ValueError) as exc:
        raise ExactDedupError("upstream physical normalization missing") from exc
    result = inspect_exact([(normalized, payload, receipt)])
    target = destination / "exact-dedup-manifest.json"
    expected = _canonical(result)
    if target.exists() or target.is_symlink():
        _require(_read_destination(target) == expected,
                 "immutable exact dedup manifest drift")
    else:
        _atomic_write(destination, target, expected)
    _require(_read_destination(target) == expected, "exact dedup readback drift")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan-2 S6 exact dedup candidate")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    options = parser.parse_args()
    result = stage_exact(options.root, options.out_dir)
    print(json.dumps({
        "status": "PASS_SOURCE_CANDIDATE_ONLY",
        "manifest_sha256": result["manifest_sha256"],
        "retained_record_count": result["retained_record_count"],
        "exact_duplicate_record_count": result["exact_duplicate_record_count"],
        "training_corpus_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
