from __future__ import annotations

import hashlib

import pytest

from twelve_six.data.corpus_foundation import CorpusFoundationError
from twelve_six.data.dedup_scale import (
    DataTroveMinhashExecutionPlan,
    build_dedup_output_manifest,
    build_training_eligibility_envelope,
    minhash_lsh_detection_probability,
)

H = "a" * 64
R = "b" * 64
I = "c" * 64
M = "d" * 64


def _plan() -> DataTroveMinhashExecutionPlan:
    return DataTroveMinhashExecutionPlan(
        source_registry_sha256=H,
        reserved_registry_sha256=R,
        input_manifest_sha256=I,
        workspace_uri="file:///tmp/data12",
        candidate_shards=8,
        workers=2,
    )


def test_scale_plan_matches_datatrove_010_signature_topology_and_is_deterministic() -> None:
    plan = _plan()
    first = plan.manifest()
    second = plan.manifest()
    assert first == second
    assert plan.signature_size == 14 * 8 == 112
    assert first["topology"]["candidate_signature_tasks"] == 8
    assert first["topology"]["candidate_bucket_tasks"] == 14
    assert first["semantic_cleanliness_claimed"] is False
    assert len(first["plan_sha256"]) == 64
    probability = minhash_lsh_detection_probability(
        0.8, num_buckets=14, hashes_per_bucket=8
    )
    assert 0.90 < probability < 0.95


def test_output_manifest_and_training_eligibility_fail_closed_on_known_semantic_overlap() -> None:
    plan = _plan()
    manifest = build_dedup_output_manifest(
        plan=plan,
        input_records=1000,
        exact_survivors=900,
        final_survivors=800,
        output_files={"00000.jsonl": hashlib.sha256(b"output").hexdigest()},
        metrics_sha256=M,
    )
    envelope = build_training_eligibility_envelope(
        output_manifest=manifest,
        source_rights_eligible=True,
        record_policy_eligible=True,
        exact_reserved_overlap_count=0,
        lexical_reserved_overlap_count=0,
        known_semantic_overlap_count=1,
        experiment_acceptance_pass=True,
    )
    assert envelope["decontamination_eligible"] is False
    assert envelope["training_eligible"] is False
    assert envelope["semantic_universal_cleanliness_claimed"] is False

    clean = build_training_eligibility_envelope(
        output_manifest=manifest,
        source_rights_eligible=True,
        record_policy_eligible=True,
        exact_reserved_overlap_count=0,
        lexical_reserved_overlap_count=0,
        known_semantic_overlap_count=0,
        experiment_acceptance_pass=True,
    )
    assert clean["training_eligible"] is True


def test_scale_plan_rejects_unvalidated_runtime_identity() -> None:
    with pytest.raises(CorpusFoundationError, match="must remain pinned"):
        DataTroveMinhashExecutionPlan(
            source_registry_sha256=H,
            reserved_registry_sha256=R,
            input_manifest_sha256=I,
            workspace_uri="file:///tmp/data12",
            datatrove_version="0.11.0",
        )
