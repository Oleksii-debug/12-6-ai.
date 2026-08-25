from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data.corpus_foundation import (
    CorpusFoundationError,
    DataTroveDedupPlan,
    PolicyHookEvidence,
    RecordPolicyMetadata,
    ShardArtifact,
    SQLiteExactDedupIndex,
    StreamingShardPlan,
    assert_contamination_free,
    build_resume_manifest,
    validate_resume_manifest,
)

H = "a" * 64
R = "b" * 64


def _hook(hook_id: str, status: str = "PASS") -> PolicyHookEvidence:
    return PolicyHookEvidence(hook_id, status, "v1", "tool@1", "2026-08-24T00:00:00Z", H)


def test_policy_hooks_fail_closed_until_every_hook_passes() -> None:
    metadata = RecordPolicyMetadata(
        quality=_hook("quality"),
        language=_hook("language"),
        pii=_hook("pii", "REVIEW_REQUIRED"),
        copyright=_hook("copyright"),
    )
    with pytest.raises(CorpusFoundationError, match="pii"):
        metadata.assert_passed()


def test_sqlite_exact_dedup_is_disk_backed_and_deterministic(tmp_path: Path) -> None:
    first = hashlib.sha256(b"alpha").hexdigest()
    second = hashlib.sha256(b"beta").hexdigest()
    with SQLiteExactDedupIndex(tmp_path / "dedup.sqlite3") as index:
        assert index.seen_or_add(first) is False
        assert index.seen_or_add(first) is True
        assert index.seen_or_add(second) is False
        index.commit()


def test_streaming_shard_assignment_and_resume_are_deterministic() -> None:
    plan = StreamingShardPlan(H, R, "file:///tmp/shards", "s1", 4, 128, 1_000_000)
    assert plan.assign("record-42") == plan.assign("record-42")
    artifacts = [
        ShardArtifact(0, 0, "file:///tmp/shards/0-0.parquet", H, 123, 4),
        ShardArtifact(0, 1, "file:///tmp/shards/0-1.parquet", R, 124, 5),
    ]
    resume = build_resume_manifest(plan, artifacts, dedup_plan_sha256=H)
    validate_resume_manifest(resume, plan, dedup_plan_sha256=H)
    tampered = json.loads(json.dumps(resume))
    tampered["artifacts"][0]["records"] += 1
    with pytest.raises(CorpusFoundationError, match="mismatch"):
        validate_resume_manifest(tampered, plan, dedup_plan_sha256=H)


def test_resume_rejects_non_contiguous_part_indexes() -> None:
    plan = StreamingShardPlan(H, R, "file:///tmp/shards", "s1", 2, 64, 1000)
    with pytest.raises(CorpusFoundationError, match="non-contiguous"):
        build_resume_manifest(
            plan,
            [ShardArtifact(0, 1, "file:///tmp/shards/0-1.parquet", H, 1, 1)],
            dedup_plan_sha256=H,
        )


def test_datatrove_plan_binds_registry_and_scale_parameters() -> None:
    plan = DataTroveDedupPlan(
        H,
        R,
        "file:///tmp/in",
        "file:///tmp/out",
        "file:///tmp/logs",
        tasks=8,
        workers=4,
    )
    manifest = plan.manifest()
    assert manifest["near_engine"] == "datatrove_minhash"
    assert len(manifest["plan_sha256"]) == 64


def test_contamination_gate_rejects_injected_overlap() -> None:
    with pytest.raises(CorpusFoundationError, match="overlap"):
        assert_contamination_free(
            {"source_id_overlap_count": 1, "content_sha256_overlap_count": 1}
        )
    assert_contamination_free({"source_id_overlap_count": 0, "content_sha256_overlap_count": 0})


def test_d06_bridge_rejects_tamper_and_detects_injected_content_overlap() -> None:
    from twelve_six.data.corpus_foundation import reserved_registry_from_d06_manifest
    from twelve_six.data.external_sources import contamination_report

    fingerprint = hashlib.sha256(b"reserved benchmark sample").hexdigest()
    payload = {
        "schema_version": "12-6.benchmark-registry.v1",
        "benchmarks": [
            {
                "benchmark_id": "synthetic-eval",
                "version": "v1",
                "source_id": "eval/source/v1",
                "held_out": True,
                "allowed_uses": ["evaluation"],
                "license_id": None,
                "source_url": None,
                "notes": None,
            }
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest = {**payload, "manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest()}
    reserved = reserved_registry_from_d06_manifest(
        manifest, {"eval/source/v1": [fingerprint]}
    )
    report = contamination_report(
        [{"id": "train-injection", "source_id": "other", "content_sha256": fingerprint}],
        reserved,
    )
    assert report["content_sha256_overlap_count"] == 1
    with pytest.raises(CorpusFoundationError, match="overlap"):
        assert_contamination_free(report)

    tampered = json.loads(json.dumps(manifest))
    tampered["benchmarks"][0]["version"] = "v2"
    with pytest.raises(CorpusFoundationError, match="hash mismatch"):
        reserved_registry_from_d06_manifest(tampered, {})
