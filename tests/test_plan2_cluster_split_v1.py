"""Plan 2 S10: group-safe 3-way split and versioned fail-closed contracts."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import plan2_cluster_split_v1 as split

ROOT = Path(__file__).resolve().parents[1]
POLICY = (ROOT / split.POLICY_PATH).read_bytes()


def _receipt(rows: dict, key: str) -> dict:
    return {**rows, key: split._sha(split._canonical(rows))}


def _inputs(n: int = 30):
    ids = [f"source:r{i:08d}" for i in range(n)]
    group = ids[:2]
    near = {
        "schema_version": "12-6.plan2-near-dedup-candidate.v1",
        "retained_record_ids": ids[0:1] + ids[2:],
        "near_families": [{
            "family_id_sha256": split._sha(split._canonical(group)),
            "representative_record_id": group[0],
            "members": group,
            "retained_family_cap": 1,
        }],
    }
    near = _receipt(near, "manifest_sha256")
    selected = ids[0:1] + ids[2:]
    mix = _receipt({
        "schema_version": "12-6.plan2-corpus-mixture-candidate.v1",
        "selected_record_ids": selected,
        "selected_record_count": len(selected),
        "excluded": [{"record_id": group[1], "reason": "NEAR_FAMILY_CAP_V1"}],
        "physical_s8_manifest_sha256": "b" * 64,
        "upstream_s8_manifest_sha256": "a" * 64,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "real_final_test_material_accessed": False,
    }, "dataset_candidate_sha256")
    return mix, near


def test_policy_is_pinned_and_fraction_exact():
    policy = split.parse_policy(POLICY)
    assert policy["test_percent"] == policy["validation_percent"] == 10
    assert split._git_blob(POLICY) == split.POLICY_GIT_BLOB


def test_three_way_disjoint_deterministic_and_no_claim_widening():
    mix, near = _inputs()
    policy = split.parse_policy(POLICY)
    first = split.compose_split(mix, near, policy)
    second = split.compose_split(copy.deepcopy(mix), copy.deepcopy(near), policy)
    assert first == second
    split.verify_split(first, mix, near, policy)
    assert set(first["splits"]) == set(split.PARTITIONS)
    assert all(first["splits"].values())
    assert set(first["splits"]["train"]).isdisjoint(first["splits"]["validation"])
    assert set(first["splits"]["train"]).isdisjoint(first["splits"]["test"])
    assert set(first["splits"]["validation"]).isdisjoint(first["splits"]["test"])
    assert first["selected_record_count"] == sum(map(len, first["splits"].values()))
    assert first["record_cluster_sha256"]["source:r00000000"] == (
        near["near_families"][0]["family_id_sha256"])
    assert "source:r00000001" not in json.dumps(first)
    assert first["training_corpus_authorized"] is False
    assert first["production_test_release_authorized"] is False


def test_different_seed_changes_split_identity_even_when_counts_same():
    mix, near = _inputs()
    policy = split.parse_policy(POLICY)
    a = split.compose_split(mix, near, policy)
    changed = dict(policy)
    changed["seed"] = "c" * 64
    b = split.compose_split(mix, near, changed)
    assert a["split_manifest_sha256"] != b["split_manifest_sha256"]


@pytest.mark.parametrize("mutation", [
    "missing", "overlap", "different-source", "duplicate", "count", "hash",
    "training", "s7-tamper", "fake-family", "multi-member", "invalid-s7",
])
def test_rejects_forgery_or_upstream_leakage(mutation):
    mix, near = _inputs()
    if mutation == "missing":
        mix["selected_record_ids"] = mix["selected_record_ids"][:-1]
    if mutation == "overlap":
        mix["excluded"].append({"record_id": mix["selected_record_ids"][0]})
    if mutation == "different-source":
        mix["selected_record_ids"][0] = "foreign:r00000000"
    if mutation == "duplicate":
        mix["selected_record_ids"].append(mix["selected_record_ids"][-1])
    if mutation == "count":
        mix["selected_record_count"] += 1
    if mutation == "hash":
        mix["dataset_candidate_sha256"] = "c" * 64
    if mutation == "training":
        mix["training_corpus_authorized"] = True
    if mutation == "s7-tamper":
        near["manifest_sha256"] = "f" * 64
    if mutation == "fake-family":
        near["near_families"][0]["family_id_sha256"] = "f" * 64
    if mutation == "multi-member":
        near["retained_record_ids"].append("source:r00000001")
    if mutation == "invalid-s7":
        near["schema_version"] = "attacker.v1"
    if mutation not in {"hash", "s7-tamper"}:
        if mutation != "invalid-s7":
            if mutation in {"fake-family", "multi-member"}:
                near = _receipt({k: v for k, v in near.items() if k != "manifest_sha256"},
                                "manifest_sha256")
            else:
                mix = _receipt({k: v for k, v in mix.items()
                                if k != "dataset_candidate_sha256"},
                               "dataset_candidate_sha256")
    with pytest.raises(split.Plan2ClusterSplitError):
        split.compose_split(mix, near, split.parse_policy(POLICY))


def test_too_few_independent_clusters_fail_closed():
    mix, near = _inputs(3)
    with pytest.raises(split.Plan2ClusterSplitError, match="three independent"):
        split.compose_split(mix, near, split.parse_policy(POLICY))


def test_verifier_rejects_self_resealed_assignment_movement():
    mix, near = _inputs()
    policy = split.parse_policy(POLICY)
    real = split.compose_split(mix, near, policy)
    tampered = copy.deepcopy(real)
    rid = tampered["splits"]["test"].pop(0)
    tampered["splits"]["train"].append(rid)
    tampered = _receipt({k: v for k, v in tampered.items()
                         if k != "split_manifest_sha256"}, "split_manifest_sha256")
    with pytest.raises(split.Plan2ClusterSplitError, match="semantic"):
        split.verify_split(tampered, mix, near, policy)


def test_policy_rejects_malformed_and_duplicates():
    assert split.parse_policy(POLICY)
    cases = [POLICY.replace(b'"validation_percent": 10',
                            b'"validation_percent": true'),
             POLICY.replace(b'"validation_percent": 10',
                            b'"validation_percent": 35'),
             POLICY.replace(b'"validation_percent": 10',
                            b'"validation_percent": 10, "validation_percent": 10'),
             POLICY.replace(b'"revision": ', b'"unknown": 42, "revision": ')]
    for case in cases:
        with pytest.raises(split.Plan2ClusterSplitError):
            split.parse_policy(case)
