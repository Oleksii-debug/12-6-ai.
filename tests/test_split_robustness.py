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
    bind_split_evidence,
    build_split_family,
    dedup_relations_identity,
    eligible_corpus_identity,
    legacy_record_hash_assignments,
    pairwise_ranking_stability,
    split_sensitivity,
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
    # The corpus identity transitively binds the cluster relation, so this drift
    # fails even before the separately retained dedup-relations identity check.
    with pytest.raises(SplitRobustnessError, match="eligible corpus"):
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

    bad_run = {
        "data": {
            "split_identity": "human-readable-v1",
            "eligible_corpus_sha256": corpus,
        }
    }
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


@pytest.mark.parametrize(
    "invalid_metric",
    [True, float("nan"), float("inf"), float("-inf")],
)
def test_split_metric_evidence_rejects_nonfinite_and_bool_values(
    invalid_metric: float,
) -> None:
    with pytest.raises(SplitRobustnessError, match="finite real number"):
        split_sensitivity([1.0, invalid_metric])

    with pytest.raises(SplitRobustnessError, match="finite real number"):
        pairwise_ranking_stability(
            {
                "small": [5.0, invalid_metric],
                "large": [4.5, 4.7],
            }
        )


@pytest.mark.parametrize(
    "values",
    [
        [1e308, -1e308],
        [-1e308, 1e308],
    ],
)
def test_split_sensitivity_rejects_nonfinite_derived_outputs(
    values: list[float],
) -> None:
    with pytest.raises(
        SplitRobustnessError,
        match="split sensitivity derived metric must be a finite real number",
    ):
        split_sensitivity(values)


def test_split_metric_evidence_rejects_huge_integer_without_overflow_leak() -> None:
    huge = 10**400
    with pytest.raises(SplitRobustnessError, match="finite real number"):
        split_sensitivity([1, huge])
    with pytest.raises(SplitRobustnessError, match="finite real number"):
        pairwise_ranking_stability(
            {
                "small": [5, huge],
                "large": [4, 6],
            }
        )


def test_split_metric_evidence_keeps_finite_integer_and_float_semantics() -> None:
    sensitivity = split_sensitivity([1, 2.5, 4])
    assert sensitivity["mean"] == 2.5

    stable = pairwise_ranking_stability(
        {"small": [5, 5.1], "large": [4.5, 4]}
    )
    assert stable["all_pairs_stable"] is True


@pytest.mark.parametrize("values", [[1e308, 1e308], [1e308, 1e308, 1e308]])
def test_split_sensitivity_normalizes_statistics_overflow(
    values: list[float],
) -> None:
    # Every input is finite, but statistics.fmean overflows internally.
    with pytest.raises(SplitRobustnessError, match="statistics overflowed"):
        split_sensitivity(values)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_split_evidence_rejects_nested_nonfinite_numbers(invalid: float) -> None:
    family = {
        "split_family_identity_sha256": "1" * 64,
        "eligible_corpus_sha256": "2" * 64,
    }
    with pytest.raises(SplitRobustnessError, match="finite and serializable"):
        bind_split_evidence({"nested": {"score": invalid}}, family)


def test_split_evidence_rejects_unserializable_and_overdeep_values() -> None:
    family = {
        "split_family_identity_sha256": "1" * 64,
        "eligible_corpus_sha256": "2" * 64,
    }
    with pytest.raises(SplitRobustnessError, match="finite and serializable"):
        bind_split_evidence({"nested": object()}, family)
    nested: list[object] = []
    for _ in range(1500):
        nested = [nested]
    with pytest.raises(SplitRobustnessError, match="finite and serializable"):
        bind_split_evidence({"nested": nested}, family)


def test_split_evidence_retains_deterministic_finite_identity() -> None:
    family = {
        "split_family_identity_sha256": "1" * 64,
        "eligible_corpus_sha256": "2" * 64,
    }
    payload = {"nested": {"score": 1.25, "count": 2}}
    first = bind_split_evidence(payload, family)
    second = bind_split_evidence(payload, family)
    assert first == second
    assert len(first["evidence_sha256"]) == 64
    assert first["nested"] == payload["nested"]
    assert payload == {"nested": {"score": 1.25, "count": 2}}
