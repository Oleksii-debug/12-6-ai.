from __future__ import annotations

import hashlib

import pytest

from twelve_six.split_robustness import (
    SplitFamilySpec,
    SplitRecord,
    SplitRobustnessError,
    assert_checkpoint_split_binding,
    assert_run_split_binding,
    audit_cluster_leakage,
    build_split_family,
    dedup_relations_identity,
    eligible_corpus_identity,
    legacy_record_hash_assignments,
    pairwise_ranking_stability,
    verify_split_family_manifest,
)


def _records() -> list[SplitRecord]:
    records: list[SplitRecord] = []
    for cluster_index in range(60):
        for member in range(2):
            text = (
                f"Project-authored cluster {cluster_index:03d}, member {member}. "
                "Deterministic validation data must remain outside optimization while "
                f"near-duplicate relatives stay together. Marker {cluster_index * 11 + member}."
            )
            records.append(
                SplitRecord(
                    id=f"doc-{cluster_index:03d}-{member}",
                    text=text,
                    source_id=f"source-{cluster_index % 3}",
                    modality=("uk", "en", "code")[cluster_index % 3],
                    content_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    near_duplicate_cluster_id=f"cluster-{cluster_index:03d}",
                )
            )
    return records


def _family() -> tuple[list[SplitRecord], dict[str, object]]:
    records = _records()
    spec = SplitFamilySpec(
        eligible_corpus_sha256=eligible_corpus_identity(records),
        dedup_relations_sha256=dedup_relations_identity(records),
        variant_seeds=("split-a", "split-b", "split-c", "split-d"),
        validation_fraction=0.10,
    )
    return records, build_split_family(records, spec)


def test_split_family_is_deterministic_distinct_and_cluster_safe() -> None:
    records, family = _family()
    verify_split_family_manifest(records, family)

    repeated = build_split_family(
        records,
        SplitFamilySpec(
            eligible_corpus_sha256=eligible_corpus_identity(records),
            dedup_relations_sha256=dedup_relations_identity(records),
            variant_seeds=("split-a", "split-b", "split-c", "split-d"),
            validation_fraction=0.10,
        ),
    )
    assert repeated == family
    assert len(set(family["variant_split_identities"])) == 4
    assert family["cluster_straddles_across_variants"] == 0
    assert set(family["shared_train_record_ids"]).isdisjoint(
        family["validation_union_record_ids"]
    )

    for variant in family["variants"]:
        assignments = {record_id: "train" for record_id in variant["train_record_ids"]}
        assignments.update(
            {record_id: "validation" for record_id in variant["validation_record_ids"]}
        )
        assert audit_cluster_leakage(records, assignments) == []


def test_record_level_hash_semantics_can_straddle_known_near_duplicate_clusters() -> None:
    records = _records()
    leakage_counts = []
    for seed in ("split-a", "split-b", "split-c", "split-d"):
        assignments = legacy_record_hash_assignments(
            records, seed=seed, validation_fraction=0.10
        )
        leakage_counts.append(len(audit_cluster_leakage(records, assignments)))
    assert any(count > 0 for count in leakage_counts)


def test_split_family_rejects_tampering_and_dedup_relation_drift() -> None:
    records, family = _family()
    family["shared_train_record_ids"] = list(family["shared_train_record_ids"])[1:]
    with pytest.raises(SplitRobustnessError, match="identity/content mismatch"):
        verify_split_family_manifest(records, family)

    records = _records()
    altered = list(records)
    victim = altered[0]
    altered[0] = SplitRecord(
        id=victim.id,
        text=victim.text,
        source_id=victim.source_id,
        modality=victim.modality,
        content_sha256=victim.content_sha256,
        near_duplicate_cluster_id="different-cluster",
    )
    _, clean_family = _family()
    with pytest.raises(SplitRobustnessError, match="dedup relation"):
        verify_split_family_manifest(altered, clean_family)


def test_run_and_checkpoint_binding_require_exact_split_family_sha256() -> None:
    _, family = _family()
    identity = family["split_family_identity_sha256"]
    corpus = family["eligible_corpus_sha256"]
    run_manifest = {
        "data": {
            "split_identity": identity,
            "eligible_corpus_sha256": corpus,
        }
    }
    assert assert_run_split_binding(run_manifest, family) == identity

    checkpoint = {
        "training_config": {
            "data": {
                "split_identity": identity,
            }
        }
    }
    assert assert_checkpoint_split_binding(checkpoint, family) == identity

    bad_run = {"data": {"split_identity": "human-readable-v1", "eligible_corpus_sha256": corpus}}
    with pytest.raises(SplitRobustnessError, match="SHA-256"):
        assert_run_split_binding(bad_run, family)

    bad_checkpoint = {
        "training_config": {"data": {"split_identity": "0" * 64}}
    }
    with pytest.raises(SplitRobustnessError, match="does not match"):
        assert_checkpoint_split_binding(bad_checkpoint, family)


def test_benchmark_or_test_records_are_rejected_before_split_construction() -> None:
    text = "This project fixture is marked as test material and must be rejected."
    with pytest.raises(SplitRobustnessError, match="forbidden/non-training purpose"):
        SplitRecord(
            id="heldout-test",
            text=text,
            source_id="reserved",
            modality="en",
            content_sha256=hashlib.sha256(text.encode()).hexdigest(),
            near_duplicate_cluster_id="reserved-cluster",
            purpose="heldout_test",
        )


def test_pairwise_ranking_stability_detects_rank_reversal() -> None:
    stable = pairwise_ranking_stability(
        {"small": [5.0, 5.1, 5.2], "large": [4.5, 4.7, 4.8]}
    )
    assert stable["all_pairs_stable"] is True

    unstable = pairwise_ranking_stability(
        {"small": [5.0, 4.0, 5.2], "large": [4.5, 4.7, 4.8]}
    )
    assert unstable["all_pairs_stable"] is False
    assert unstable["pairs"][0]["rank_reversal_count"] == 1
