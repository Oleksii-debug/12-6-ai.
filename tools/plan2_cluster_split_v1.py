"""Plan 2 S10: strict document-family train/validation/test split candidate.

Reuses the pinned incumbent cluster split mechanics. The transient
training_eligible=True SplitRecord is a mechanics adapter ONLY; release remains
fail-closed. One-document candidate cannot be split without leakage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools import plan2_corpus_mixture_v1 as mixture
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination
from twelve_six import split_robustness as canonical

SCHEMA = "12-6.plan2-cluster-safe-split-candidate.v1"
POLICY_SCHEMA = "12-6.plan2-cluster-split-policy.v1"
POLICY_PATH = "configs/data/plan2_cluster_split_policy_v1.json"
SPLIT_BLOB = "e49518f2e431dc8576005be4938c1f15881897c4"
POLICY_GIT_BLOB = "ee5adc07d137cf846048000b986b3254fdbe30dc"


class Plan2SplitError(ValueError):
    """Invalid upstream, cluster mapping, policy, or output identity."""


def _need(ok: bool, message: str) -> None:
    if not ok:
        raise Plan2SplitError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return mixture._canonical(value)


def _git_blob(raw: bytes) -> str:
    return mixture._git_blob(raw)


def _policy(raw: bytes) -> dict[str, Any]:
    _need(_git_blob(raw) == POLICY_GIT_BLOB, "unapproved split policy blob")
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise Plan2SplitError("malformed split policy") from exc
    _need(type(value) is dict and set(value) == {
        "schema_version", "revision", "seed", "test_fraction",
        "validation_fraction_remaining", "cluster_boundary", "purpose",
    }, "split policy fields drift")
    _need(value["schema_version"] == POLICY_SCHEMA
          and value["purpose"] == "LOCAL_FREE_CANDIDATE_ONLY"
          and value["cluster_boundary"] == "WHOLE_DOCUMENT_SOURCE_ID"
          and type(value["revision"]) is str and bool(value["revision"])
          and type(value["seed"]) is str
          and re.fullmatch("[0-9a-f]{64}", value["seed"]) is not None,
          "split policy authority invalid")
    for key in ("test_fraction", "validation_fraction_remaining"):
        _need(type(value[key]) is float and 0 < value[key] < 0.5,
              "invalid split fraction")
    return value


def _verify_incumbent() -> None:
    raw = Path(canonical.__file__).read_bytes()
    _need(_git_blob(raw) == SPLIT_BLOB,
          "canonical split mechanics changed without requalification")


def _verify_s9(s9: Mapping[str, Any]) -> list[str]:
    _need(type(s9) is dict and s9.get("schema_version") == mixture.SCHEMA,
          "unverified S9 split boundary")
    expected = s9.get("dataset_candidate_sha256")
    core = {k: v for k, v in s9.items() if k != "dataset_candidate_sha256"}
    _need(type(expected) is str and expected == _sha(_canonical(core)),
          "S9 candidate identity mismatch")
    _need(s9.get("training_corpus_authorized") is False
          and s9.get("tokenizer_fit_authorized") is False
          and s9.get("real_final_test_material_accessed") is False,
          "S9 permissions widened")
    ids = s9.get("selected_record_ids")
    _need(type(ids) is list and len(ids) >= 4
          and all(type(rid) is str and rid for rid in ids)
          and ids == sorted(set(ids))
          and s9.get("selected_record_count") == len(ids),
          "S9 selected identity set invalid")
    return ids


def build_cluster_split(
    s9: Mapping[str, Any], rows: Sequence[Mapping[str, str]],
    policy: Mapping[str, Any], *, fixture: bool = False,
) -> dict[str, Any]:
    """Assign whole document clusters, never individual normalized line records."""
    selected = set(_verify_s9(s9))
    trusted = _policy((Path(__file__).resolve().parents[1] / POLICY_PATH).read_bytes())
    _need(_canonical(policy) == _canonical(trusted), "unapproved split policy mutation")
    _verify_incumbent()
    _need(type(policy) is dict and policy.get("cluster_boundary") ==
          "WHOLE_DOCUMENT_SOURCE_ID", "untrusted cluster boundary")
    seen: set[str] = set()
    projections = []
    for row in rows:
        _need(type(row) is dict and set(row) == {
            "record_id", "source_id", "text"
        }, "untrusted split row shape")
        rid, source, value = row["record_id"], row["source_id"], row["text"]
        _need(type(rid) is str and rid in selected and rid not in seen,
              "extra/duplicate split record")
        _need(type(source) is str and rid.startswith(source + ":r")
              and type(value) is str and bool(value.strip()),
              "split record source/text invalid")
        seen.add(rid)
        # This is a transient adapter to existing mechanics, NOT training admission.
        projections.append(canonical.SplitRecord(
            id=rid, text=value, source_id=source,
            modality="text", content_sha256=_sha(value.encode("utf-8")),
            near_duplicate_cluster_id="document:" + source,
            training_eligible=True, purpose="pretraining_eligible",
        ))
    _need(seen == selected, "missing S9 selected records")
    try:
        test_clusters = set(canonical._choose_validation_clusters(
            projections, seed=policy["seed"] + ":test",
            validation_fraction=policy["test_fraction"]))
        test = {r.id for r in projections if r.near_duplicate_cluster_id in test_clusters}
        other = [r for r in projections if r.id not in test]
        val_clusters = set(canonical._choose_validation_clusters(
            other, seed=policy["seed"] + ":validation",
            validation_fraction=policy["validation_fraction_remaining"]))
        validation = {r.id for r in other if r.near_duplicate_cluster_id in val_clusters}
        train = selected - validation - test
        _need(bool(train) and bool(validation) and bool(test),
              "three-way cluster-safe split has empty partition")
    except canonical.SplitRobustnessError as exc:
        raise Plan2SplitError("insufficient independent document families") from exc
    assignments = {rid: ("test" if rid in test else
                         "validation" if rid in validation else "train")
                   for rid in sorted(selected)}
    families: dict[str, set[str]] = {}
    for row in rows:
        families.setdefault(row["source_id"], set()).add(assignments[row["record_id"]])
    _need(all(len(v) == 1 for v in families.values()),
          "document-family leakage between partitions")
    core = {
        "schema_version": SCHEMA,
        "purpose": "LOCAL_FREE_SYNTHETIC_COMPONENT" if fixture else "S9_CANDIDATE_ONLY",
        "policy_revision": policy["revision"],
        "policy_sha256": _sha(_canonical(policy)),
        "upstream_s9_dataset_candidate_sha256": s9["dataset_candidate_sha256"],
        "canonical_split_git_blob": SPLIT_BLOB,
        "canonical_corpus_identity": canonical.eligible_corpus_identity(projections),
        "canonical_dedup_relations_identity": canonical.dedup_relations_identity(projections),
        "cluster_boundary": "WHOLE_DOCUMENT_SOURCE_ID",
        "seed": policy["seed"],
        "train_record_ids": sorted(train),
        "validation_record_ids": sorted(validation),
        "test_record_ids": sorted(test),
        "record_assignments": assignments,
        "document_cluster_count": len(families),
        "cluster_leakage_count": 0,
        "training_corpus_authorized": False,
        "evaluation_release_authorized": False,
        "tokenizer_fit_authorized": False,
        "paid_compute_used": False,
        "actual_model_tokens": None,
    }
    return {**core, "split_manifest_sha256": _sha(_canonical(core))}


def _synthetic_rows() -> tuple[dict[str, Any], list[dict[str, str]]]:
    rows = [
        {"record_id": f"fixture.document.{i:03d}:r{j:08d}",
         "source_id": f"fixture.document.{i:03d}",
         "text": f"Independent public synthetic document {i}, line {j}, safe fixture."}
        for i in range(16) for j in range(2)
    ]
    core = {
        "schema_version": mixture.SCHEMA,
        "selected_record_ids": sorted(r["record_id"] for r in rows),
        "selected_record_count": len(rows),
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "real_final_test_material_accessed": False,
        "fixture_only": True,
    }
    return {**core, "dataset_candidate_sha256": _sha(_canonical(core))}, rows


def stage_fixture(root: Path, destination: Path) -> dict[str, Any]:
    _need(not any(p.is_symlink() for p in (destination, *destination.parents)),
          "symlink split destination")
    raw = (root / POLICY_PATH).read_bytes()
    policy = _policy(raw)
    s9, rows = _synthetic_rows()
    result = build_cluster_split(s9, rows, policy, fixture=True)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "cluster-split-manifest.json"
    blob = _canonical(result)
    try:
        if target.exists() or target.is_symlink():
            _need(_read_destination(target) == blob, "immutable split manifest drift")
        else:
            _atomic_write(destination, target, blob)
        _need(_read_destination(target) == blob, "split readback drift")
    except (OSError, ValueError) as exc:
        raise Plan2SplitError("split manifest publication failed") from exc
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan2 S10 LOCAL_FREE fixture")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    receipt = stage_fixture(args.root, args.out_dir)
    print(json.dumps({"status": "PASS_FIXTURE_ONLY",
                      "split_manifest_sha256": receipt["split_manifest_sha256"],
                      "cluster_leakage_count": receipt["cluster_leakage_count"],
                      "training_corpus_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
