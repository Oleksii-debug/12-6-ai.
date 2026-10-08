"""Plan 2 Section 6 exact, text-free dedup over admitted S3/S4/S5 candidates.

Never competes with incumbent D03 near-match/cross-origin matcher authority:
this Plan-2 adapter collapses *only byte-identical SHA256+size* records,
normalized source content and physical cohort members. It never grants corpus,
tokenizer, evaluation or optimization exposure authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twelve_six.data.privacy_filter_v3 import assert_hash_safe_evidence

from tools import plan2_normalization_evidence_v1 as norm
from tools import plan2_privacy_gate_v1 as privacy
from tools.plan2_physical_materialization_v1 import (
    SCHEMA as PHYSICAL_SCHEMA,
    _atomic_write,
    _read_destination,
    stage_candidate_cohort,
)

SCHEMA = "12-6.plan2-exact-dedup.v1"
EQUIVALENCE = "exact-sha256+exact-byte-length+kind+normalization-policy.v1"
SURVIVOR_RULE = "lexicographically-smallest-item-id.v1"


class ExactDedupError(ValueError):
    """Upstream identity, cohort equivalence or immutable output is invalid."""


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise ExactDedupError(reason)


def _sha(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _hash(value: Any) -> str:
    return _sha(_canonical(value))


def _is_sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


def _count(value: Any, label: str, *, positive: bool = False) -> int:
    _need(type(value) is int and value >= (1 if positive else 0),
          f"invalid {label} count")
    return value


def _manifest(value: Any, schema: str, label: str) -> Mapping[str, Any]:
    _need(isinstance(value, Mapping) and
          value.get("schema_version") == schema, f"{label} schema drift")
    core = dict(value)
    digest = core.pop("manifest_sha256", None)
    _need(_is_sha(digest) and digest == _hash(core),
          f"{label} manifest hash drift")
    _need(value.get("training_corpus_authorized") is False and
          value.get("tokenizer_fit_authorized") is False and
          value.get("evaluation_authorized") is False,
          f"{label} improperly promoted")
    return value


def _verified_cohort(value: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _need(isinstance(value, Mapping) and set(value) ==
          {"physical", "normalization", "privacy"}, "cohort schema drift")
    ph = _manifest(value["physical"], PHYSICAL_SCHEMA, "physical")
    no = _manifest(value["normalization"], norm.MANIFEST_SCHEMA, "normalization")
    pr = _manifest(value["privacy"], privacy.SCHEMA, "privacy")
    source = ph.get("source_id")
    _need(isinstance(source, str) and source and
          source == no.get("source_id") == pr.get("source_id"),
          "inconsistent source identity")
    _need(ph.get("source_level_candidate_only") is True and
          ph.get("manifest_sha256") == no.get("upstream_materialization_sha256") and
          no.get("manifest_sha256") == pr.get("normalization_manifest_sha256"),
          "broken physical-normalization-privacy lineage")
    _need(_is_sha(no.get("policy_sha256")) and
          _is_sha(pr.get("policy_sha256")) and
          _is_sha(pr.get("g06_execution_identity_sha256")),
          "policy or G06 execution identity malformed")
    _need(_is_sha(no.get("raw_sha256")) and
          _is_sha(no.get("normalized_sha256")), "normalized identity missing")
    _need(pr.get("input_record_count") == no.get("record_count"),
          "privacy/normalization record count drift")
    rows = no.get("records")
    clean = pr.get("clean_record_hashes")
    rejected = pr.get("excluded_records")
    _need(isinstance(rows, list) and isinstance(clean, list) and
          isinstance(rejected, list) and
          len(rows) == _count(no.get("record_count"), "record"),
          "invalid record or privacy inventory")
    _need(len(clean) == pr.get("candidate_record_count") and
          len(clean) + len(rejected) == len(rows),
          "privacy cohort partition drift")
    indexed: dict[str, tuple[str, int]] = {}
    start = 0
    for index, row in enumerate(rows):
        _need(isinstance(row, Mapping) and set(row) ==
              {"index", "start_byte", "end_byte", "sha256"},
              "record schema drift")
        end = row.get("end_byte")
        _need(type(row["index"]) is int and row["index"] == index and
              type(row["start_byte"]) is int and row["start_byte"] == start and
              type(end) is int and end > start and
              _is_sha(row.get("sha256")), "record ordering or hash drift")
        indexed[f"{source}:r{index:08d}"] = (row["sha256"], end - start)
        start = end
    _need(start == no.get("normalized_bytes") and
          _count(start, "normalized bytes", positive=True) > 0,
          "normalized record extent drift")
    observed: set[str] = set()
    record_entries = []
    for row in clean:
        _need(isinstance(row, Mapping) and
              set(row) == {"record_id", "sha256"}, "clean record schema drift")
        item = row["record_id"]
        _need(item in indexed and item not in observed and
              row["sha256"] == indexed[item][0],
              "clean record not bound to normalization")
        observed.add(item)
        record_entries.append({"kind": "record", "id": item,
                               "source_id": source, "sha256": row["sha256"],
                               "bytes": indexed[item][1],
                               "policy_sha256": no["policy_sha256"]})
    for row in rejected:
        _need(isinstance(row, Mapping) and
              set(row) == {"record_id", "reason"},
              "excluded record schema drift")
        item = row["record_id"]
        _need(item in indexed and item not in observed and
              row["reason"] in {"REDACT", "EXCLUDE", "QUARANTINE", "REMOVAL"},
              "excluded record partition drift")
        observed.add(item)
    _need(set(indexed) == observed, "privacy record missing or duplicated")
    member_entries = []
    for name, expected_hash in (("raw", no["raw_sha256"]),
                                ("normalized", no["normalized_sha256"])):
        member = ph.get(name)
        _need(isinstance(member, Mapping) and
              _is_sha(member.get("sha256")) and
              member["sha256"] == expected_hash and
              _count(member.get("bytes"), "member", positive=True) > 0,
              "physical member identity drift")
        if name == "normalized":
            _need(member["bytes"] == no["normalized_bytes"],
                  "normalized member extent drift")
        member_entries.append({"kind": "member", "id": f"{source}:{name}",
                               "source_id": source, "sha256": member["sha256"],
                               "bytes": member["bytes"],
                               "policy_sha256": "raw-physical-bytes.v1"})
    content_entry = {"kind": "content", "id": f"{source}:normalized-content",
                     "source_id": source, "sha256": no["normalized_sha256"],
                     "bytes": no["normalized_bytes"],
                     "policy_sha256": no["policy_sha256"]}
    roots = {"source_id": source,
             "physical_manifest_sha256": ph["manifest_sha256"],
             "normalization_manifest_sha256": no["manifest_sha256"],
             "privacy_manifest_sha256": pr["manifest_sha256"],
             "privacy_execution_sha256": pr["g06_execution_identity_sha256"]}
    return {"roots": roots, "policy_sha256": no["policy_sha256"],
            "privacy_policy_sha256": pr["policy_sha256"]}, [
                content_entry, *member_entries, *record_entries]


def build_exact_dedup(cohorts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Deterministically collapse only exact byte-level equivalence classes."""
    _need(isinstance(cohorts, Sequence) and not isinstance(cohorts, (str, bytes))
          and bool(cohorts), "at least one authenticated cohort required")
    metadata = []
    all_entries: list[dict[str, Any]] = []
    for cohort in cohorts:
        info, entries = _verified_cohort(cohort)
        metadata.append(info)
        all_entries.extend(entries)
    sources = [info["roots"]["source_id"] for info in metadata]
    _need(len(set(sources)) == len(sources), "duplicate source lineage")
    _need(len({x["policy_sha256"] for x in metadata}) == 1 and
          len({x["privacy_policy_sha256"] for x in metadata}) == 1,
          "incompatible normalization or privacy policy")
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in all_entries:
        key = (entry["kind"], entry["policy_sha256"],
               entry["sha256"], entry["bytes"])
        grouped[key].append(entry)
    survivors: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    discarded = {"content": 0, "member": 0, "record": 0}
    for key, members in sorted(grouped.items()):
        members.sort(key=lambda r: r["id"])
        canonical_id = members[0]["id"]
        equivalence = _hash({"kind": key[0], "policy": key[1],
                             "sha256": key[2], "bytes": key[3]})
        survivors.append({"kind": key[0], "id": canonical_id,
                          "source_id": members[0]["source_id"],
                          "sha256": key[2], "bytes": key[3],
                          "equivalence_sha256": equivalence})
        discarded[key[0]] += len(members) - 1
        if len(members) > 1:
            families.append({"kind": key[0], "equivalence_sha256": equivalence,
                             "survivor_id": canonical_id,
                             "member_ids": [m["id"] for m in members],
                             "member_source_ids": [m["source_id"] for m in members]})
    survivors.sort(key=lambda row: (row["kind"], row["id"]))
    families.sort(key=lambda row: (row["kind"], row["survivor_id"]))
    counts = {kind: {"input": sum(x["kind"] == kind for x in all_entries),
                     "surviving": sum(x["kind"] == kind for x in survivors),
                     "duplicate_removed": discarded[kind]}
              for kind in ("content", "member", "record")}
    core = {"schema_version": SCHEMA, "equivalence_rule": EQUIVALENCE,
            "survivor_rule": SURVIVOR_RULE,
            "normalization_policy_sha256": metadata[0]["policy_sha256"],
            "privacy_policy_sha256": metadata[0]["privacy_policy_sha256"],
            "input_cohorts": sorted((m["roots"] for m in metadata),
                                    key=lambda row: row["source_id"]),
            "counts": counts, "survivors": survivors,
            "duplicate_families": families,
            "training_corpus_authorized": False,
            "tokenizer_fit_authorized": False,
            "evaluation_authorized": False,
            "exact_dedup_only": True,
            "near_match_authority": "incumbent D03 — not replicated by Plan 2"}
    result = {**core, "manifest_sha256": _hash(core)}
    assert_hash_safe_evidence(result)
    return result


def stage_exact_dedup(root: Path, destination: Path) -> dict[str, Any]:
    """Rebuild trusted S3/S4/S5 and immutable S6; no external network/compute."""
    _need(not any(p.is_symlink() for p in (destination, *destination.parents)),
          "symlink output destination")
    root = root.resolve()
    privacy_receipt = privacy.stage_privacy(root, destination / "privacy")
    normalization = norm.stage_normalization(
        root, destination / "privacy" / "normalization")
    physical = stage_candidate_cohort(
        root, destination / "privacy" / "normalization" / "cohort")
    result = build_exact_dedup([{"physical": physical,
                                 "normalization": normalization,
                                 "privacy": privacy_receipt}])
    output = destination / "exact-dedup-manifest.json"
    expected = _canonical(result)
    if output.exists() or output.is_symlink():
        _need(_read_destination(output) == expected,
              "immutable exact dedup candidate changed; use new destination")
    else:
        _atomic_write(destination, output, expected)
    _need(_read_destination(output) == expected, "dedup readback mismatch")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan-2 exact hashed candidate dedup")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = stage_exact_dedup(args.root, args.out_dir)
    print(json.dumps({"status": "PASS_CANDIDATE_ONLY",
                      "manifest_sha256": result["manifest_sha256"],
                      "exact_duplicate_record_count": (
                          result["counts"]["record"]["duplicate_removed"]),
                      "training_corpus_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
