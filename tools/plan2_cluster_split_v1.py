"""Plan 2 Section 10: candidate-only cluster-safe three-way split over S9.

This adapter consumes the accepted S9 selection and S7 near-family authority.
It does not redefine dedup, obtain raw reserved evaluation data, or authorize training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "12-6.plan2-cluster-safe-split.v1"
POLICY_SCHEMA = "12-6.plan2-cluster-split-policy.v1"
POLICY_PATH = "configs/data/plan2_cluster_split_policy_v1.json"
POLICY_GIT_BLOB = "6abca62603be0abf540e2f7d3e1490be5fc135cc"
PARTITIONS = ("train", "validation", "test")


class Plan2ClusterSplitError(ValueError):
    """Upstream selection, split policy, or immutable output is invalid."""


def _need(condition: bool, why: str) -> None:
    if not condition:
        raise Plan2ClusterSplitError(why)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for key, value in pairs:
        _need(key not in data, "duplicate split policy JSON key")
        data[key] = value
    return data


def _not_constant(_: str) -> None:
    raise Plan2ClusterSplitError("nonfinite split policy")


def parse_policy(raw: bytes) -> dict[str, Any]:
    _need(type(raw) is bytes and _git_blob(raw) == POLICY_GIT_BLOB,
          "unapproved split policy Git blob")
    try:
        obj = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_pairs,
                         parse_constant=_not_constant)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise Plan2ClusterSplitError("split policy malformed") from exc
    _need(type(obj) is dict and set(obj) == {
        "schema_version", "revision", "purpose", "seed", "validation_percent",
        "test_percent",
    }, "split policy schema keys")
    _need(obj["schema_version"] == POLICY_SCHEMA
          and obj["purpose"] == "LOCAL_FREE_CANDIDATE_SPLIT"
          and type(obj["revision"]) is str and bool(obj["revision"].strip())
          and type(obj["seed"]) is str
          and re.fullmatch(r"[0-9a-f]{64}", obj["seed"]) is not None,
          "split policy authority invalid")
    valid = (type(obj["validation_percent"]) is int
             and type(obj["test_percent"]) is int
             and 1 <= obj["validation_percent"] <= 30
             and 1 <= obj["test_percent"] <= 30
             and obj["validation_percent"] + obj["test_percent"] < 50)
    _need(valid, "split policy fractions invalid")
    return obj


def _check_receipt(receipt: Mapping[str, Any], key: str,
                   schema: str) -> None:
    _need(type(receipt) is dict and receipt.get("schema_version") == schema,
          f"{key} schema invalid")
    claimed = receipt.get(key)
    _need(type(claimed) is str and re.fullmatch(r"[0-9a-f]{64}", claimed)
          is not None, f"{key} missing")
    core = {k: v for k, v in receipt.items() if k != key}
    _need(_sha(_canonical(core)) == claimed, f"{key} self-hash mismatch")


def _part_sizes(n: int, policy: Mapping[str, Any]) -> tuple[int, int]:
    _need(n >= 3, "at least three independent S7 clusters required")
    val = max(1, (n * policy["validation_percent"] + 50) // 100)
    test = max(1, (n * policy["test_percent"] + 50) // 100)
    _need(val + test < n, "split would empty train partition")
    return val, test


def compose_split(
    s9: Mapping[str, Any], s7: Mapping[str, Any], policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Assign whole verified S7 near families, never individual members."""
    _check_receipt(s9, "dataset_candidate_sha256",
                   "12-6.plan2-corpus-mixture-candidate.v1")
    _check_receipt(s7, "manifest_sha256", "12-6.plan2-near-dedup-candidate.v1")
    _need(s9.get("training_corpus_authorized") is False
          and s9.get("tokenizer_fit_authorized") is False
          and s9.get("real_final_test_material_accessed") is False,
          "S9 candidate-only claim boundary widened")
    ids = s9.get("selected_record_ids")
    retained = s7.get("retained_record_ids")
    excluded = s9.get("excluded")
    _need(type(ids) is list and bool(ids)
          and all(type(x) is str and bool(x) for x in ids)
          and len(set(ids)) == len(ids)
          and ids == sorted(ids)
          and type(s9.get("selected_record_count")) is int
          and len(ids) == s9["selected_record_count"],
          "S9 selected record identities invalid")
    _need(type(retained) is list and len(set(retained)) == len(retained)
          and set(ids).issubset(set(retained)),
          "S9 selection does not belong to S7 survivor boundary")
    _need(type(excluded) is list
          and all(type(row) is dict and type(row.get("record_id")) is str
                  for row in excluded)
          and set(ids).isdisjoint({row["record_id"] for row in excluded}),
          "S9 excluded record entered split")
    _need(s9.get("physical_s8_manifest_sha256") is not None
          and s9.get("upstream_s8_manifest_sha256") is not None,
          "S9 physical S8 binding missing")
    _need(s7.get("retained_family_cap") is None,
          "unexpected alternate S7 family authority")

    family_by_member: dict[str, str] = {}
    groups = s7.get("near_families")
    _need(type(groups) is list, "S7 family evidence missing")
    for group in groups:
        _need(type(group) is dict
              and type(group.get("members")) is list
              and len(group["members"]) >= 2
              and len(set(group["members"])) == len(group["members"])
              and group.get("members") == sorted(group["members"])
              and group.get("representative_record_id") == group["members"][0]
              and group.get("retained_family_cap") == 1
              and group.get("family_id_sha256") == _sha(_canonical(group["members"])),
              "S7 near family proof invalid")
        for member in group["members"]:
            _need(member not in family_by_member, "overlapping S7 family membership")
            family_by_member[member] = group["family_id_sha256"]
        _need(set(group["members"]) & set(retained)
              == {group["representative_record_id"]},
              "S7 near duplicate family cap broken")

    clusters: dict[str, list[str]] = {}
    for rid in ids:
        cluster = family_by_member.get(rid, _sha(_canonical([rid])))
        clusters.setdefault(cluster, []).append(rid)
    _need(sum(map(len, clusters.values())) == len(ids),
          "split cluster accounting mismatch")
    n_val, n_test = _part_sizes(len(clusters), policy)
    ranked = sorted(clusters, key=lambda key: (
        _sha((policy["seed"] + ":" + key).encode()), key))
    assigned: dict[str, list[str]] = {key: [] for key in PARTITIONS}
    cluster_splits: dict[str, str] = {}
    for i, cluster in enumerate(ranked):
        partition = ("test" if i < n_test else
                     "validation" if i < n_test + n_val else "train")
        cluster_splits[cluster] = partition
        assigned[partition].extend(clusters[cluster])
    assigned = {part: sorted(records) for part, records in assigned.items()}
    _need(all(assigned.values())
          and set().union(*(set(p) for p in assigned.values())) == set(ids)
          and sum(map(len, assigned.values())) == len(ids),
          "train/validation/test coverage or isolation failed")
    core = {
        "schema_version": SCHEMA,
        "policy_schema_version": POLICY_SCHEMA,
        "policy_revision": policy["revision"],
        "policy_sha256": _sha(_canonical(policy)),
        "seed": policy["seed"],
        "upstream_s9_dataset_candidate_sha256": s9["dataset_candidate_sha256"],
        "upstream_s7_near_manifest_sha256": s7["manifest_sha256"],
        "upstream_s8_manifest_sha256": s9["upstream_s8_manifest_sha256"],
        "cluster_authority": "S7_COMPLETE_LINK_NEAR_FAMILY_WITH_SINGLETONS",
        "selected_record_count": len(ids),
        "cluster_count": len(clusters),
        "splits": assigned,
        "split_cluster_counts": {part: sum(x == part for x in cluster_splits.values())
                                 for part in PARTITIONS},
        "record_cluster_sha256": dict(sorted((rid, cluster)
                                              for cluster, members in clusters.items()
                                              for rid in members)),
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "real_final_test_material_accessed": False,
        "production_test_release_authorized": False,
        "paid_compute_used": False,
    }
    return {**core, "split_manifest_sha256": _sha(_canonical(core))}


def verify_split(manifest: Mapping[str, Any], s9: Mapping[str, Any],
                 s7: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    _need(dict(manifest) == compose_split(s9, s7, policy),
          "cluster split semantic identity/content mismatch")


def stage_split(root: Path, destination: Path) -> dict[str, Any]:
    """Replay S3→S9, publish immutable S10 split and verify exact bytes."""
    from tools import plan2_corpus_mixture_v1 as mixture
    from tools.plan2_physical_materialization_v1 import (
        _atomic_write, _json, _read_destination, _read_source,
    )

    _need(not any(x.is_symlink() for x in (destination, *destination.parents)),
          "symlink split destination")
    try:
        raw = _read_source(root, POLICY_PATH)
        policy = parse_policy(raw)
        s9 = mixture.stage_mixture(root, destination / "mixture")
        s7 = _json(_read_destination(destination / "mixture" / "firewall" /
                                     "near" / "near-dedup-manifest.json"))
        manifest = compose_split(s9, s7, policy)
        verify_split(manifest, s9, s7, policy)
        target = destination / "cluster-split-manifest.json"
        payload = _canonical(manifest)
        if target.exists() or target.is_symlink():
            _need(_read_destination(target) == payload, "immutable split manifest drift")
        else:
            _atomic_write(destination, target, payload)
        _need(_read_destination(target) == payload, "split readback drift")
        return manifest
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise Plan2ClusterSplitError("physical S10 split staging denied") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan2 S10 LOCAL_FREE split")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = stage_split(args.root, args.out_dir)
    print(json.dumps({"status": "PASS_LOCAL_FREE_CANDIDATE_ONLY",
                      "split_manifest_sha256": result["split_manifest_sha256"],
                      "split_record_counts": {k: len(v) for k, v in result["splits"].items()},
                      "training_corpus_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
