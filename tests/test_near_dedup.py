from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from twelve_six.data.near_dedup import (
    NearDedupPolicy,
    calibration_records,
    load_calibration,
    lsh_detection_probability,
    policy_candidates,
    run_datatrove_policy,
    score_calibration,
    select_policy,
    surviving_corpus_identity,
)


def test_policy_grid_is_small_and_contains_data12_incumbent() -> None:
    candidates = policy_candidates()
    assert set(candidates) == {"natural", "code"}
    assert len(candidates["natural"]) == 3
    assert len(candidates["code"]) == 3
    incumbent = next(p for p in candidates["natural"] if p.name == "natural_incumbent_9g_14x8")
    assert incumbent.n_grams == 9
    assert incumbent.num_buckets == 14
    assert incumbent.hashes_per_bucket == 8
    assert incumbent.signature_size == 112
    assert 0.68 < incumbent.lsh_similarity_at_50pct_detection < 0.69
    assert 0.92 < lsh_detection_probability(0.8, buckets=14, hashes_per_bucket=8) < 0.93


def test_code_policy_is_more_conservative_in_lsh_banding() -> None:
    candidates = policy_candidates()
    natural = next(p for p in candidates["natural"] if p.name == "natural_incumbent_9g_14x8")
    code = next(p for p in candidates["code"] if p.name == "code_strict_5g_10x10")
    assert code.lsh_similarity_at_50pct_detection > natural.lsh_similarity_at_50pct_detection
    assert code.manifest()["semantic_deduplication_claimed"] is False


def test_select_policy_keeps_acceptable_preferred_incumbent() -> None:
    policies = policy_candidates()["natural"]
    scored = [(policy, {"recall": 1.0, "false_removal_risk": 0.0}) for policy in policies]
    selected = select_policy(scored, preferred_policy_name="natural_incumbent_9g_14x8")
    assert selected.name == "natural_incumbent_9g_14x8"


def test_select_policy_rejects_unsafe_preferred_policy() -> None:
    policies = policy_candidates()["code"]
    scored = []
    for policy in policies:
        risk = 0.5 if policy.name == "code_strict_5g_10x10" else 0.0
        scored.append((policy, {"recall": 1.0, "false_removal_risk": risk}))
    selected = select_policy(scored, preferred_policy_name="code_strict_5g_10x10")
    assert selected.name != "code_strict_5g_10x10"


def test_surviving_corpus_identity_binds_policy_and_survivors() -> None:
    records = [
        {"id": "a", "text": "alpha beta gamma", "metadata": {"source_id": "s", "raw_identity": "raw-a"}},
        {"id": "b", "text": "delta epsilon zeta", "metadata": {"source_id": "s", "raw_identity": "raw-b"}},
    ]
    natural = NearDedupPolicy("n", "natural", 9, 14, 8)
    code = NearDedupPolicy("c", "code", 5, 10, 10)
    one = surviving_corpus_identity(records, selected_policies={"natural": natural, "code": code}, input_corpus_identity="f" * 64)
    changed_policy = surviving_corpus_identity(
        records,
        selected_policies={"natural": NearDedupPolicy("n2", "natural", 9, 12, 9), "code": code},
        input_corpus_identity="f" * 64,
    )
    fewer = surviving_corpus_identity(records[:1], selected_policies={"natural": natural, "code": code}, input_corpus_identity="f" * 64)
    assert one["surviving_corpus_identity"] != changed_policy["surviving_corpus_identity"]
    assert one["surviving_corpus_identity"] != fewer["surviving_corpus_identity"]


def test_calibration_schema_has_required_categories() -> None:
    calibration = load_calibration(Path("data/calibration/near_dedup_v1.json"))
    categories = {pair["category"] for pair in calibration["pairs"]}
    assert {
        "true_near_copy",
        "boilerplate",
        "translation",
        "code_fork",
        "legitimate_similar_document",
    } <= categories
    assert calibration_records(calibration, "natural")
    assert calibration_records(calibration, "code")


@pytest.mark.skipif(importlib.util.find_spec("datatrove") is None, reason="DataTrove optional outside DATA-30 lane")
def test_datatrove_cluster_provenance_restart_and_scoring(tmp_path: Path) -> None:
    calibration = load_calibration(Path("data/calibration/near_dedup_v1.json"))
    records = calibration_records(calibration, "natural")
    policy = next(p for p in policy_candidates()["natural"] if p.name == "natural_incumbent_9g_14x8")
    first = run_datatrove_policy(records, policy=policy, workspace=tmp_path / "work")
    rerun = run_datatrove_policy(records, policy=policy, workspace=tmp_path / "work")
    metrics = score_calibration(calibration, modality="natural", execution=first)

    assert first["engine"]["name"] == "DataTrove MinHash"
    assert first["engine"]["second_dedup_engine_created"] is False
    assert first["restart"]["signature_rerun_byte_identical"] is True
    assert first["survivor_ids"] == rerun["survivor_ids"]
    assert first["clusters"] == rerun["clusters"]
    assert metrics["positive_pairs"] >= 3
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["false_removal_risk"] <= 1.0
    for cluster in first["clusters"]:
        assert cluster["representative_record_id"] in {
            member["record_id"] for member in cluster["members"]
        }
