"""Plan2 S13: fixture-only deterministic shards, mapping and recovery."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import plan2_deterministic_packing_v1 as p

ROOT = Path(__file__).resolve().parents[1]


def test_replay_and_explicit_parent_no_authority():
    manifest, shards = p.build(ROOT)
    assert (manifest, shards) == p.build(ROOT)
    p.verify(ROOT, manifest, shards)
    assert manifest["target_count"] > 0
    assert manifest["block_count"] == len(manifest["source_token_shard_mapping"])
    assert not manifest["training_corpus_authorized"]
    assert not manifest["production_release_authorized"]
    assert not manifest["paid_compute_used"]
    assert all(ref["sha256"] == p.digest(shards[ref["path"]])
               for ref in manifest["shards"])


def test_masks_offsets_record_boundaries_and_shard_rollover():
    manifest, shards = p.build(ROOT)
    seen = {}
    for ref in manifest["shards"]:
        obj = json.loads(shards[ref["path"]])
        assert obj["block_size"] == p.BLOCK
        assert len(obj["blocks"]) <= p.PER_SHARD
        for block in obj["blocks"]:
            n = block["target_count"]
            assert 0 < n <= p.BLOCK
            assert len(block["input_ids"]) == len(block["target_ids"]) == p.BLOCK
            assert block["attention_mask"] == block["loss_mask"]
            assert block["loss_mask"] == [1] * n + [0] * (p.BLOCK-n)
            assert block["input_ids"][n:] == [256] * (p.BLOCK-n)
            assert block["target_ids"][n:] == [256] * (p.BLOCK-n)
            seen.setdefault(block["record_id"], []).append(block)
    assert set(seen) == set(manifest["train_record_ids"])
    for blocks in seen.values():
        offset = 0
        for block in blocks:
            assert block["token_offset"] == offset
            offset += block["target_count"]
        assert offset


def test_restart_clean_rebuild_interruption_and_corruption(tmp_path):
    manifest, shards = p.build(ROOT)
    partial = tmp_path / "partial"
    (partial / "shards").mkdir(parents=True)
    name, raw = next(iter(shards.items()))
    (partial / name).write_bytes(raw)
    assert p.stage(ROOT, partial) == manifest
    assert p.stage(ROOT, partial) == manifest
    assert p.stage(ROOT, tmp_path / "clean") == manifest
    for name in ("packing-manifest.json", *shards):
        assert (partial / name).read_bytes() == (tmp_path / "clean" / name).read_bytes()
    victim = partial / next(iter(shards))
    victim.write_bytes(b"{}")
    with pytest.raises(p.PackingDenied):
        p.stage(ROOT, partial)


def test_missing_extra_and_symlink_shard_denied(tmp_path):
    out = tmp_path / "data"
    manifest = p.stage(ROOT, out)
    victim = out / manifest["shards"][0]["path"]
    victim.unlink()
    with pytest.raises(p.PackingDenied):
        p.stage(ROOT, out)
    victim.symlink_to(out / "packing-manifest.json")
    with pytest.raises(p.PackingDenied):
        p.stage(ROOT, out)
    alias = tmp_path / "alias"
    alias.symlink_to(out, target_is_directory=True)
    with pytest.raises(p.PackingDenied):
        p.stage(ROOT, alias / "child")


def test_resealed_manifest_and_fake_shards_rejected():
    original, shards = p.build(ROOT)
    for key, value in [
        ("train_record_ids", []), ("tokenizer_identity", {}),
        ("training_corpus_authorized", True), ("target_count", 0),
    ]:
        attack = copy.deepcopy(original)
        attack[key] = value
        core = {k: v for k, v in attack.items() if k != "manifest_sha256"}
        attack["manifest_sha256"] = p.digest(p.canonical(core))
        with pytest.raises(p.PackingDenied):
            p.verify(ROOT, attack, shards)
    attack_shards = dict(shards)
    attack_shards[next(iter(shards))] = b"{}\n"
    with pytest.raises(p.PackingDenied):
        p.verify(ROOT, original, attack_shards)


def test_duplicate_record_rejected_and_no_eval_test_rows():
    frozen, selected, tokenizer, rows = p.inputs(ROOT)
    assert len(rows) == frozen["train_record_count"]
    assert {r["record_id"] for r in rows}.isdisjoint(selected["test_record_ids"])
    assert {r["record_id"] for r in rows}.isdisjoint(selected["validation_record_ids"])
    with pytest.raises(p.PackingDenied):
        p.pack(tokenizer, rows + [rows[0]])


def test_resumable_reader_restart_boundary_and_empty_end(tmp_path):
    out = tmp_path / "candidate"
    manifest = p.stage(ROOT, out)
    all_blocks = p.read_blocks(ROOT, out)
    assert len(all_blocks) == manifest["block_count"]
    assert [item["block_index"] for item in all_blocks] == list(range(len(all_blocks)))
    assert all(item["next_block"] == item["block_index"] + 1 for item in all_blocks)
    for cursor in (0, 1, p.PER_SHARD, len(all_blocks) // 2, len(all_blocks)):
        assert all_blocks[:cursor] + p.read_blocks(ROOT, out, start_block=cursor) == all_blocks
    assert p.read_blocks(ROOT, out, start_block=len(all_blocks)) == []


@pytest.mark.parametrize("cursor", [-1, True, None, 1.0, "1", 100000])
def test_reader_rejects_invalid_cursor(tmp_path, cursor):
    out = tmp_path / "candidate"
    p.stage(ROOT, out)
    with pytest.raises(p.PackingDenied):
        p.read_blocks(ROOT, out, start_block=cursor)


def test_reader_denies_corruption_missing_extra_and_symlink(tmp_path):
    out = tmp_path / "candidate"
    m = p.stage(ROOT, out)
    assert p.read_blocks(ROOT, out)
    shard = out / m["shards"][0]["path"]
    original = shard.read_bytes()
    shard.write_bytes(b"{}\\n")
    with pytest.raises(p.PackingDenied):
        p.read_blocks(ROOT, out)
    shard.write_bytes(original)
    shard.unlink()
    with pytest.raises(p.PackingDenied):
        p.read_blocks(ROOT, out)
    shard.write_bytes(original)
    extra = out / "shards" / "undeclared.json"
    extra.write_bytes(b"{}")
    with pytest.raises(p.PackingDenied):
        p.read_blocks(ROOT, out)
    extra.unlink()
    alias = tmp_path / "alias"
    alias.symlink_to(out, target_is_directory=True)
    with pytest.raises(p.PackingDenied):
        p.read_blocks(ROOT, alias)
    manifest = out / "packing-manifest.json"
    manifest.write_bytes(b"{}")
    with pytest.raises(p.PackingDenied):
        p.read_blocks(ROOT, out)


def test_interrupted_temporary_files_are_cleaned_but_unsafe_links_denied(tmp_path):
    out = tmp_path / "candidate"
    (out / "shards").mkdir(parents=True)
    partial = out / "shards" / ".plan2-partial-abandoned"
    partial.write_bytes(b"uncommitted")
    (out / ".plan2-partial-abandoned").write_bytes(b"uncommitted")
    manifest = p.stage(ROOT, out)
    assert manifest["shards"]
    assert not partial.exists()
    assert not (out / ".plan2-partial-abandoned").exists()
    (out / "shards" / ".plan2-partial-hostile").symlink_to(out / "packing-manifest.json")
    with pytest.raises(p.PackingDenied):
        p.stage(ROOT, out)


def test_undeclared_destination_member_denied(tmp_path):
    out = tmp_path / "candidate"
    p.stage(ROOT, out)
    (out / "unrecognized-record.json").write_text("{}", encoding="utf-8")
    with pytest.raises(p.PackingDenied):
        p.stage(ROOT, out)
