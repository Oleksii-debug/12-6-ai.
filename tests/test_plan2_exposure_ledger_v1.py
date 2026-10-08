"""Plan2 S14 LOCAL_FREE exact ordered exposure/restart/negative fixture tests."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools import plan2_deterministic_packing_v1 as packing
from tools import plan2_exposure_ledger_v1 as p

ROOT = Path(__file__).resolve().parents[1]


def publish(tmp_path):
    shards = tmp_path / "shards-published"
    packing.stage(ROOT, shards)
    out = tmp_path / "ledger"
    manifest = p.stage(ROOT, shards, out)
    return shards, out, manifest


def test_replay_partition_and_exact_token_mapping(tmp_path):
    shards, out, manifest = publish(tmp_path)
    assert manifest == p.build(ROOT, shards) == p.stage(ROOT, shards, out)
    m, source_shards = packing.build(ROOT)
    assert manifest["block_count"] == m["block_count"]
    assert manifest["target_count"] == m["target_count"]
    assert not manifest["training_corpus_authorized"]
    assert not manifest["optimizer_effect_authorized"]
    assert not manifest["paid_compute_used"]
    entries = p.read_exposures(ROOT, shards, out)
    assert len(entries) == manifest["target_count"]
    assert [row["index"] for row in entries] == list(range(len(entries)))
    assert len({row["target_id"] for row in entries}) == len(entries)
    assert len({row["exposure_id"] for row in entries}) == len(entries)
    assert entries[-1]["chain_sha256"] == manifest["chain_head_sha256"]
    assert {row["record_id"] for row in entries}.issubset(set(m["train_record_ids"]))
    assert source_shards


def test_every_shard_rollover_and_last_block_cursor(tmp_path):
    shards, out, manifest = publish(tmp_path)
    all_rows = p.read_exposures(ROOT, shards, out)
    for i in (0, 1, 31, 32, 127, 128, len(all_rows)//2, len(all_rows)):
        if i > len(all_rows):
            continue
        receipt = p.resume_receipt(manifest, i)
        suffix = p.read_exposures(ROOT, shards, out, start_target=i,
                                  expected_resume=receipt)
        assert all_rows[:i] + suffix == all_rows
        assert suffix[0]["index"] == i if suffix else i == len(all_rows)
    assert p.read_exposures(ROOT, shards, out,
                            start_target=len(all_rows)) == []


@pytest.mark.parametrize("bad", [-1, True, None, 1.5, "0", 999999])
def test_invalid_cursor_denied(tmp_path, bad):
    shards, out, manifest = publish(tmp_path)
    with pytest.raises(p.ExposureDenied):
        p.read_exposures(ROOT, shards, out, start_target=bad)
    with pytest.raises(p.ExposureDenied):
        p.resume_receipt(manifest, bad)


def test_wrong_or_forged_resume_denied(tmp_path):
    shards, out, manifest = publish(tmp_path)
    receipt = p.resume_receipt(manifest, 17)
    for candidate, cursor in ((receipt, 18), ({**receipt, "next_target": 18}, 17),
                              ({**receipt, "manifest_sha256": "0"*64}, 17),
                              ("untrusted", 17)):
        with pytest.raises(p.ExposureDenied):
            p.read_exposures(ROOT, shards, out, start_target=cursor,
                             expected_resume=candidate)


def test_corrupted_and_self_resealed_publication_denied(tmp_path):
    shards, out, manifest = publish(tmp_path)
    target = out / p.FILE
    before = target.read_bytes()
    target.write_bytes(b"{}")
    with pytest.raises(p.ExposureDenied):
        p.read_exposures(ROOT, shards, out)
    with pytest.raises(p.ExposureDenied):
        p.stage(ROOT, shards, out)
    target.write_bytes(before)
    altered = copy.deepcopy(manifest)
    altered["ordered_exposures"][2]["token_offset"] += 1
    altered["manifest_sha256"] = p.hashed({k: v for k, v in altered.items()
                                             if k != "manifest_sha256"})
    target.write_bytes(p.canonical(altered))
    with pytest.raises(p.ExposureDenied):
        p.read_exposures(ROOT, shards, out)


def test_missing_extra_or_symlink_denied(tmp_path):
    shards, out, manifest = publish(tmp_path)
    (out / "unexpected").write_bytes(b"extra")
    with pytest.raises(p.ExposureDenied):
        p.read_exposures(ROOT, shards, out)
    (out / "unexpected").unlink()
    alias = tmp_path / "link"
    alias.symlink_to(out, target_is_directory=True)
    with pytest.raises(p.ExposureDenied):
        p.read_exposures(ROOT, shards, alias)
    (out / p.FILE).unlink()
    with pytest.raises(p.ExposureDenied):
        p.read_exposures(ROOT, shards, out)


def test_shard_corruption_or_external_replacement_denied(tmp_path):
    shards, out, manifest = publish(tmp_path)
    m, _ = packing.build(ROOT)
    victim = shards / m["shards"][0]["path"]
    victim.write_bytes(b"{}")
    with pytest.raises(packing.PackingDenied):
        p.read_exposures(ROOT, shards, out)


def test_interrupted_stage_recovery(tmp_path):
    shards = tmp_path / "s13"
    packing.stage(ROOT, shards)
    out = tmp_path / "s14"
    out.mkdir()
    (out / ".plan2-partial-interrupted").write_bytes(b"partial")
    published = p.stage(ROOT, shards, out)
    assert not (out / ".plan2-partial-interrupted").exists()
    clean = p.stage(ROOT, shards, tmp_path / "clean")
    assert published == clean
    assert (out / p.FILE).read_bytes() == (tmp_path / "clean" / p.FILE).read_bytes()
