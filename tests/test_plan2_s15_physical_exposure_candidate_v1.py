"""S15 actual physical S14 exposure replay is unique, durable and non-release."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_deterministic_packing_v1 as packing
from tools import plan2_exposure_ledger_v1 as exposure
from tools import plan2_s15_physical_exposure_candidate_v1 as real

ROOT = Path(__file__).resolve().parents[1]


def test_real_physical_exposure_chain_and_restart(tmp_path: Path) -> None:
    receipt = real.stage(ROOT, tmp_path / "ledger")
    assert receipt["decision"] == "PHYSICAL_ORDERED_TARGETS_CANDIDATE_NOT_RELEASE"
    assert receipt["target_count"] > 0
    assert receipt["source_shard_count"] == len(receipt["exposure_shards"])
    assert sum(x["exposure_count"] for x in receipt["exposure_shards"]) == (
        receipt["target_count"]
    )
    assert len(receipt["chain_head_sha256"]) == 64
    assert receipt["training_corpus_authorized"] is False
    assert receipt["production_release_authorized"] is False
    assert receipt["optimizer_effect_authorized"] is False
    assert receipt["terminal_done"] is False
    for shard in receipt["exposure_shards"]:
        raw = (tmp_path / "ledger" / shard["path"]).read_bytes()
        assert real.books.sha(raw) == shard["sha256"]
        assert len(json.loads(raw)["ordered_exposures"]) == shard["exposure_count"]
    assert real.stage(ROOT, tmp_path / "ledger") == receipt


def _one_block() -> tuple[dict, dict[str, bytes]]:
    sid = "en.physical.book"
    record_id = sid + ":r50000000"
    source_shard = "shards/shard-000000.json"
    tokens = [258, 100] + [256] * (packing.BLOCK - 2)
    mask = [1, 1] + [0] * (packing.BLOCK - 2)
    block = {
        "record_id": record_id, "source_id": sid,
        "source_sha256": "a" * 64,
        "token_offset": 0, "target_count": 2,
        "input_ids": [257, 42] + [256] * (packing.BLOCK - 2),
        "target_ids": tokens, "attention_mask": mask, "loss_mask": mask,
    }
    data = packing.canonical({
        "schema_version": "12-6.plan2-token-shard.v1",
        "shard_index": 0, "block_size": packing.BLOCK, "blocks": [block],
    })
    meta = {
        "path": source_shard, "sha256": packing.digest(data),
        "block_count": 1, "target_count": 2, "byte_count": len(data),
    }
    manifest = {
        "target_count": 2, "block_count": 1,
        "block_size": packing.BLOCK,
        "production_release_authorized": False,
        "physical_s9_admitted": False,
        "source_token_shard_mapping": [{
            "record_id": record_id, "source_id": sid,
            "source_sha256": block["source_sha256"],
            "token_offset": 0, "target_count": 2,
            "shard_path": source_shard, "block_ordinal": 0,
        }],
        "shards": [meta],
        "cluster_split_sha256": "b" * 64,
        "tokenizer_manifest_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
    }
    return manifest, {source_shard: data}


def test_s14_exact_target_identity_on_physical_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = _one_block()
    monkeypatch.setattr(real.physical, "build", lambda _root: fixed)
    receipt, files = real.build(tmp_path)
    assert receipt["target_count"] == 2
    data = json.loads(files["exposures/shard-000000.json"])
    entries = data["ordered_exposures"]
    assert [x["block_index"] for x in entries] == [0, 0]
    assert [x["token_offset"] for x in entries] == [0, 1]
    for entry in entries:
        basis = {
            "contract": exposure.CONTRACT,
            "split_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "record_id": entry["record_id"],
            "source_sha256": "a" * 64,
            "token_offset": entry["token_offset"],
            "target_token_id": entry["target_token_id"],
        }
        assert entry["target_id"] == exposure.hashed(basis)
        assert entry["exposure_id"] == exposure.hashed({
            "packing_sha256": "d" * 64, "target_id": entry["target_id"],
            "global_index": entry["index"],
        })


def test_changed_physical_shard_hash_blocks_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, blocks = _one_block()
    manifest["shards"][0]["sha256"] = "e" * 64
    monkeypatch.setattr(real.physical, "build", lambda _root: (manifest, blocks))
    with pytest.raises(real.PhysicalExposureDenied, match="SHA"):
        real.build(tmp_path)


def test_changed_target_mask_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, blocks = _one_block()
    name = "shards/shard-000000.json"
    parsed = json.loads(blocks[name])
    parsed["blocks"][0]["loss_mask"][1] = 0
    blocks[name] = packing.canonical(parsed)
    manifest["shards"][0]["sha256"] = packing.digest(blocks[name])
    monkeypatch.setattr(real.physical, "build", lambda _root: (manifest, blocks))
    with pytest.raises(real.PhysicalExposureDenied, match="mask"):
        real.build(tmp_path)


def test_symlink_publication_is_denied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(real, "build", lambda _: pytest.fail("unsafe source read"))
    with pytest.raises(real.PhysicalExposureDenied, match="symlink"):
        real.stage(tmp_path, link)


def test_reused_s14_builder_exactly_matches_incumbent_and_publisher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = _one_block()
    monkeypatch.setattr(real.physical, "build", lambda _: fixed)
    existing, existing_bytes = real.build(tmp_path)
    reused, reused_bytes = real.build_from_packed(*fixed)
    assert reused == existing
    assert reused_bytes == existing_bytes
    first = real.stage_manifest(reused, reused_bytes, tmp_path / "reused")
    second = real.stage(tmp_path, tmp_path / "existing")
    assert first == second == existing
    assert (tmp_path / "reused" / real.OUTPUT).read_bytes() == (
        tmp_path / "existing" / real.OUTPUT
    ).read_bytes()
    assert real.stage_manifest(reused, reused_bytes, tmp_path / "reused") == existing
