from __future__ import annotations

import hashlib
import json

import pytest

from twelve_six.data import (
    RIGHTS_APPROVED,
    RIGHTS_REVIEW_REQUIRED,
    CorpusFoundationError,
    DataTroveMinhashPlan,
    ExactDedupPlan,
    ExternalSourceSpec,
    PolicyHookResult,
    RecordPolicyMetadata,
    RightsDecision,
    ShardCompletion,
    SnapshotSpec,
    StreamingResumeManifest,
    StreamingShardPlan,
    assert_resume_compatible,
    build_corpus_eligibility_manifest,
    build_external_source_registry,
    contamination_gate,
    iter_shard,
    reserved_registry_from_d06,
)


def _source(status: str = RIGHTS_APPROVED) -> ExternalSourceSpec:
    payload = b"immutable source"
    rights = RightsDecision(
        status=status,
        license_id="CC-BY-4.0",
        terms_url="https://example.invalid/terms",
        allows_model_training=status == RIGHTS_APPROVED,
        allows_derivatives=True,
        allows_redistribution=True,
        policy_ref="policy://rights/42",
        reviewed_at="2026-08-24T00:00:00Z",
        reviewer_ref="role://data-rights-owner",
    )
    snapshot = SnapshotSpec(
        uri="hf://datasets/example/source/snapshots/commit-abc/source.jsonl",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        retrieved_at="2026-08-24T00:00:00Z",
        upstream_version="commit-abc",
        retrieval_method="immutable_snapshot_copy",
    )
    return ExternalSourceSpec(
        source_id="external-source-a",
        source_version="commit-abc",
        provider="example-provider",
        source_url="https://example.invalid/source",
        source_kind="jsonl",
        purpose="pretraining",
        synthetic=False,
        benchmark_material=False,
        held_out=False,
        snapshot=snapshot,
        rights=rights,
    )


def _record(record_id: str = "r1", *, pii: str = "PASS", size: int = 8) -> RecordPolicyMetadata:
    hooks = tuple(
        PolicyHookResult(
            category=category,
            status=pii if category == "pii" else "PASS",
            hook_id=f"hook-{category}",
            hook_version="v1",
            policy_ref=f"policy://records/{category}/v1",
            detail_code="fixture",
        )
        for category in ("quality", "language", "pii", "copyright")
    )
    content = (record_id.encode() * (size // max(len(record_id), 1) + 1))[:size]
    return RecordPolicyMetadata(
        record_id=record_id,
        source_id="external-source-a",
        source_version="commit-abc",
        content_sha256=hashlib.sha256(content).hexdigest(),
        normalized_size_bytes=len(content),
        language="en",
        hooks=hooks,
    )


def _d06_manifest(*, allowed_uses: list[str] | None = None) -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": "12-6.benchmark-registry.v1",
        "benchmarks": [
            {
                "benchmark_id": "heldout-a",
                "version": "v1",
                "source_id": "heldout-source",
                "held_out": True,
                "allowed_uses": allowed_uses or ["evaluation"],
                "license_id": None,
                "source_url": None,
                "notes": "fixture",
            }
        ],
    }
    canonical = json.dumps(core, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**core, "manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def test_corpus_eligibility_binds_rights_policy_and_registry_identity() -> None:
    registry = build_external_source_registry([_source()])
    records = [_record("one"), _record("two")]
    first = build_corpus_eligibility_manifest(registry, records)
    assert first == build_corpus_eligibility_manifest(registry, records)
    assert first["record_count"] == 2
    assert first["rights_and_policy_gate"] == "PASS"
    assert first["source_registry_identity_sha256"] == registry["registry_identity_sha256"]


@pytest.mark.parametrize(
    ("registry", "records", "match"),
    [
        (build_external_source_registry([_source()]), [], "at least one accepted record"),
        (
            build_external_source_registry([_source(RIGHTS_REVIEW_REQUIRED)]),
            [_record()],
            "not train-eligible",
        ),
        (
            build_external_source_registry([_source()]),
            [_record(pii="REVIEW_REQUIRED")],
            "not clear",
        ),
    ],
)
def test_corpus_eligibility_fails_closed(registry, records, match: str) -> None:
    with pytest.raises(CorpusFoundationError, match=match):
        build_corpus_eligibility_manifest(registry, records)


def test_policy_metadata_requires_all_four_categories() -> None:
    record = _record()
    incomplete = RecordPolicyMetadata(
        record_id=record.record_id,
        source_id=record.source_id,
        source_version=record.source_version,
        content_sha256=record.content_sha256,
        normalized_size_bytes=record.normalized_size_bytes,
        language=record.language,
        hooks=tuple(hook for hook in record.hooks if hook.category != "copyright"),
    )
    with pytest.raises(CorpusFoundationError, match="required policy hooks are missing"):
        incomplete.assert_policy_clear()


def test_d06_bridge_verifies_manifest_and_reserved_fingerprint() -> None:
    fingerprint = hashlib.sha256(b"benchmark").hexdigest()
    reserved = reserved_registry_from_d06(
        _d06_manifest(), {"heldout-a@v1": [fingerprint]}
    )
    assert reserved["sets"][0]["source_id"] == "heldout-source"
    assert reserved["sets"][0]["normalized_sha256"] == [fingerprint]

    tampered = _d06_manifest()
    tampered["manifest_sha256"] = "0" * 64
    with pytest.raises(CorpusFoundationError, match="manifest SHA-256 mismatch"):
        reserved_registry_from_d06(tampered)

    with pytest.raises(CorpusFoundationError, match="allows training uses"):
        reserved_registry_from_d06(_d06_manifest(allowed_uses=["evaluation", "training"]))


def test_contamination_injections_reject_source_and_exact_content() -> None:
    fingerprint = hashlib.sha256(b"benchmark").hexdigest()
    reserved = reserved_registry_from_d06(
        _d06_manifest(), {"heldout-a@v1": [fingerprint]}
    )
    with pytest.raises(CorpusFoundationError, match="contamination detected"):
        contamination_gate(
            [{"id": "a", "source_id": "heldout-source", "content_sha256": "1" * 64}],
            reserved,
        )
    with pytest.raises(CorpusFoundationError, match="contamination detected"):
        contamination_gate(
            [{"id": "b", "source_id": "clean-source", "content_sha256": fingerprint}],
            reserved,
        )
    clean = contamination_gate(
        [{"id": "c", "source_id": "clean-source", "content_sha256": "2" * 64}],
        reserved,
    )
    assert clean["source_id_overlap_count"] == clean["content_sha256_overlap_count"] == 0


def test_exact_dedup_plan_is_deterministic_and_partition_bounded() -> None:
    plan = ExactDedupPlan(
        corpus_identity_sha256="1" * 64,
        input_uri="file:///corpus/input",
        survivor_uri="file:///corpus/survivors",
        duplicate_uri="file:///corpus/duplicates",
        partitions=64,
    )
    digest = hashlib.sha256(b"same").hexdigest()
    assert 0 <= plan.partition_for(digest) < 64
    assert plan.partition_for(digest) == plan.partition_for(digest)
    assert plan.manifest()["bounded_memory_contract"] == "one_hash_partition_at_a_time"


def test_exact_dedup_plan_rejects_uri_aliasing() -> None:
    with pytest.raises(CorpusFoundationError, match="must be distinct"):
        ExactDedupPlan("1" * 64, "file:///same", "file:///same", "file:///duplicates")


def test_datatrove_minhash_plan_is_pinned_restartable_and_topology_safe() -> None:
    kwargs = {
        "corpus_identity_sha256": "1" * 64,
        "input_parquet_uri": "file:///m/input",
        "signatures_uri": "file:///m/signatures",
        "buckets_uri": "file:///m/buckets",
        "remove_ids_uri": "file:///m/remove",
        "output_parquet_uri": "file:///m/output",
        "logging_uri": "file:///m/logs",
    }
    plan = DataTroveMinhashPlan(**kwargs, tasks=8, workers=4)
    manifest = plan.manifest()
    assert manifest["datatrove_version"] == "0.10.0"
    assert manifest["skip_completed"] is True
    assert "MinhashDedupSignature" in manifest["pipeline"][0]
    with pytest.raises(CorpusFoundationError, match="workers cannot exceed tasks"):
        DataTroveMinhashPlan(**kwargs, tasks=1, workers=2)


def test_sharding_is_order_and_worker_independent_and_streaming() -> None:
    plan = StreamingShardPlan("3" * 64, shard_count=4, max_record_bytes=16)
    ids = ["a", "b", "c", "d"]
    assert {item: plan.shard_for_record_id(item) for item in ids} == {
        item: plan.shard_for_record_id(item) for item in reversed(ids)
    }
    assert plan.manifest()["worker_count_independent"] is True
    records = [_record(f"r{index}") for index in range(20)]
    shard = list(iter_shard(iter(records), plan, 2))
    assert all(plan.shard_for_record_id(record.record_id) == 2 for record in shard)

    too_large = _record("large", size=17)
    target = plan.shard_for_record_id(too_large.record_id)
    with pytest.raises(CorpusFoundationError, match="exceeds max_record_bytes"):
        list(iter_shard([too_large], plan, target))


def test_resume_manifest_binds_exact_shard_plan() -> None:
    plan = StreamingShardPlan("4" * 64, shard_count=4)
    resume = StreamingResumeManifest(
        plan_sha256=plan.manifest()["plan_sha256"],
        shard_count=4,
        completed=(ShardCompletion(0, 10, 100, "5" * 64), ShardCompletion(2, 12, 120, "6" * 64)),
    )
    assert resume.pending_shards() == (1, 3)
    assert_resume_compatible(plan, resume)
    with pytest.raises(CorpusFoundationError, match="different shard plan"):
        assert_resume_compatible(plan, StreamingResumeManifest("7" * 64, 4, ()))
