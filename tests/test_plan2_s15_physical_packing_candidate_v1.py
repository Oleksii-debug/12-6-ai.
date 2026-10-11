"""S15 physical packing must preserve every real token and fail closed."""
from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from tools import plan2_deterministic_packing_v1 as packing
from tools import plan2_s15_physical_packing_candidate_v1 as physical

ROOT = Path(__file__).resolve().parents[1]


def test_real_shards_integrity_and_same_path_restart(tmp_path: Path) -> None:
    out = tmp_path / "shards"
    receipt = physical.stage(ROOT, out)
    assert receipt["decision"] == "REAL_PHYSICAL_PACKING_CANDIDATE_NOT_RELEASE"
    assert receipt["target_count"] > 0
    assert len(receipt["cluster_split_sha256"]) == 64
    assert receipt["cluster_split_sha256"] != receipt["physical_split_probe_sha256"]
    assert receipt["block_count"] > 0
    assert receipt["shards"]
    segment_map = receipt["source_segment_byte_map"]
    assert [r["record_id"] for r in segment_map] == receipt["train_record_ids"]
    assert segment_map[0]["source_byte_start"] == 0
    assert segment_map[-1]["source_byte_end"] == receipt["train_document_bytes"]
    assert all(a["source_byte_end"] == b["source_byte_start"]
               for a, b in pairwise(segment_map))
    assert all(len(x["source_sha256"]) == 64 for x in segment_map)

    assert receipt["source_token_shard_mapping"]
    assert sum(x["target_count"] for x in receipt["shards"]) == (
        receipt["target_count"]
    )
    assert all(len(x["sha256"]) == 64 for x in receipt["shards"])
    assert receipt["physical_s9_admitted"] is False
    assert receipt["training_corpus_authorized"] is False
    assert receipt["production_release_authorized"] is False
    assert receipt["terminal_done"] is False
    for entry in receipt["shards"]:
        raw = (out / entry["path"]).read_bytes()
        assert packing.digest(raw) == entry["sha256"]
        assert len(raw) == entry["byte_count"]
        assert len(json.loads(raw)["blocks"]) == entry["block_count"]
    assert json.loads((out / physical.OUTPUT).read_bytes()) == receipt
    assert physical.stage(ROOT, out) == receipt


def test_changed_shard_fails_before_successful_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    member = "shards/shard-000000.json"
    proof = {"schema_version": physical.SCHEMA,
             "decision": "REAL_PHYSICAL_PACKING_CANDIDATE_NOT_RELEASE",
             "production_release_authorized": False}
    monkeypatch.setattr(physical, "build",
                        lambda _root: (proof, {member: b'{"blocks":[]}\n'}))
    out = tmp_path / "out"
    physical.stage(tmp_path, out)
    (out / member).write_bytes(b'{"blocks":[{"target":9}]}\n')
    with pytest.raises(physical.PhysicalPackingDenied, match="immutable"):
        physical.stage(tmp_path, out)


def test_immutable_manifest_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    member = "shards/shard-000000.json"
    proof = {"schema_version": physical.SCHEMA,
             "production_release_authorized": False}
    monkeypatch.setattr(physical, "build",
                        lambda _root: (proof, {member: b"clean\n"}))
    out = tmp_path / "out"
    physical.stage(tmp_path, out)
    (out / physical.OUTPUT).write_bytes(b'{"production_release_authorized":true}\n')
    with pytest.raises(physical.PhysicalPackingDenied, match="immutable"):
        physical.stage(tmp_path, out)


def test_symlink_publication_denied_without_reading_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(physical, "build", lambda _: pytest.fail("source read"))
    with pytest.raises(physical.PhysicalPackingDenied, match="symlink"):
        physical.stage(tmp_path, link)


def test_stale_extra_shard_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    member = "shards/shard-000000.json"
    proof = {"schema_version": physical.SCHEMA,
             "production_release_authorized": False}
    monkeypatch.setattr(physical, "build",
                        lambda _root: (proof, {member: b"clean\n"}))
    out = tmp_path / "out"
    physical.stage(tmp_path, out)
    (out / "shards" / "shard-stale.json").write_bytes(b"stale\n")
    with pytest.raises(physical.PhysicalPackingDenied, match="unexpected"):
        physical.stage(tmp_path, out)
