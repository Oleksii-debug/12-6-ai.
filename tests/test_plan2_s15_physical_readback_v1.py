"""S15 independent physical S13->S14 published replay and tamper negatives."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import plan2_deterministic_packing_v1 as packing
from tools import plan2_exposure_ledger_v1 as exposure
from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_physical_readback_v1 as readback


def _receipt(core: dict) -> dict:
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def _fixture(root: Path) -> tuple[Path, Path, dict, dict]:
    pd, ed = root / "packing", root / "exposure"
    (pd / "shards").mkdir(parents=True)
    (ed / "exposures").mkdir(parents=True)
    name = "shards/shard-000000.json"
    rid = "en.book:r50000000"
    source = "en.book"
    n = 2
    pad = packing.BLOCK - n
    block = {
        "record_id": rid,
        "source_id": source,
        "source_sha256": "a" * 64,
        "token_offset": 0,
        "target_count": n,
        "input_ids": [257, 97] + [256] * pad,
        "target_ids": [97, 258] + [256] * pad,
        "attention_mask": [1, 1] + [0] * pad,
        "loss_mask": [1, 1] + [0] * pad,
    }
    raw = packing.canonical({
        "schema_version": "12-6.plan2-token-shard.v1",
        "shard_index": 0,
        "block_size": packing.BLOCK,
        "blocks": [block],
    })
    (pd / name).write_bytes(raw)
    original = {
        "schema_version": "12-6.plan2-s15-physical-packed-shards-candidate.v1",
        "decision": "REAL_PHYSICAL_PACKING_CANDIDATE_NOT_RELEASE",
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
        "cluster_split_sha256": "b" * 64,
        "tokenizer_manifest_sha256": "c" * 64,
        "target_count": n,
        "block_count": 1,
        "block_size": packing.BLOCK,
        "train_document_bytes": 2,
        "train_record_ids": [rid],
        "source_segment_byte_map": [{
            "record_id": rid, "source_id": source,
            "source_byte_start": 0, "source_byte_end": 2,
            "source_sha256": "a" * 64,
        }],
        "shards": [{
            "path": name,
            "sha256": books.sha(raw),
            "block_count": 1,
            "target_count": n,
            "byte_count": len(raw),
        }],
        "source_token_shard_mapping": [{
            "shard_path": name,
            "block_ordinal": 0,
            "record_id": rid,
            "source_id": source,
            "source_sha256": "a" * 64,
            "token_offset": 0,
            "target_count": n,
        }],
    }
    packed = _receipt(original)
    events = []
    chain = exposure.ZERO
    for i, token in enumerate([97, 258]):
        basis = {
            "contract": exposure.CONTRACT,
            "split_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "record_id": rid,
            "source_sha256": "a" * 64,
            "token_offset": i,
            "target_token_id": token,
        }
        target_id = exposure.hashed(basis)
        exposure_id = exposure.hashed({
            "packing_sha256": packed["manifest_sha256"],
            "target_id": target_id,
            "global_index": i,
        })
        row = {
            "index": i, "record_id": rid,
            "token_offset": i, "block_index": 0,
            "shard_path": name, "position_in_block": i,
            "target_token_id": token, "target_id": target_id,
            "exposure_id": exposure_id,
        }
        chain = exposure.hashed({"previous": chain, "exposure": row})
        events.append({**row, "chain_sha256": chain})
    out = "exposures/shard-000000.json"
    raw_e = books.canonical({
        "schema_version": "12-6.plan2-s15-physical-exposure-shard.v1",
        "source_shard_sha256": books.sha(raw),
        "source_shard_path": name,
        "ordered_exposures": events,
    })
    (ed / out).write_bytes(raw_e)
    ledger = _receipt({
        "schema_version": "12-6.plan2-s15-physical-ordered-exposure-candidate.v1",
        "decision": "PHYSICAL_ORDERED_TARGETS_CANDIDATE_NOT_RELEASE",
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
        "packing_manifest_sha256": packed["manifest_sha256"],
        "cluster_split_sha256": "b" * 64,
        "tokenizer_manifest_sha256": "c" * 64,
        "target_count": n,
        "block_count": 1,
        "chain_head_sha256": chain,
        "exposure_shards": [{
            "source_shard_path": name,
            "path": out,
            "sha256": books.sha(raw_e),
            "byte_count": len(raw_e),
            "exposure_count": n,
            "chain_tail_sha256": chain,
        }],
    })
    (pd / "physical-shard-candidate.json").write_bytes(books.canonical(packed))
    (ed / "physical-exposure-candidate.json").write_bytes(books.canonical(ledger))
    return pd, ed, packed, ledger


def _verify(pd: Path, ed: Path, packed: dict, ledger: dict) -> dict:
    return readback.verify(
        pd, ed,
        expected_packed_sha256=packed["manifest_sha256"],
        expected_exposure_sha256=ledger["manifest_sha256"],
    )


def test_physical_replay_hashes_and_boundary(tmp_path: Path) -> None:
    pd, ed, packed, ledger = _fixture(tmp_path)
    proof = _verify(pd, ed, packed, ledger)
    assert proof["readback_target_count"] == 2
    assert proof["readback_block_count"] == 1
    assert proof["last_chain_sha256"] == ledger["chain_head_sha256"]
    assert proof["production_release_authorized"] is False
    assert proof["terminal_done"] is False
    assert len(proof["readback_sha256"]) == 64
    assert _verify(pd, ed, packed, ledger) == proof


@pytest.mark.parametrize("target", [
    "packing-bytes", "exposure-bytes", "orphan", "manifest-forgery",
])
def test_physical_replay_tamper_fails(
    tmp_path: Path, target: str,
) -> None:
    pd, ed, packed, ledger = _fixture(tmp_path)
    if target == "packing-bytes":
        (pd / "shards" / "shard-000000.json").write_bytes(b"{}\n")
    elif target == "exposure-bytes":
        member = ed / "exposures" / "shard-000000.json"
        doc = json.loads(member.read_bytes())
        doc["ordered_exposures"][1]["target_token_id"] = 99
        member.write_bytes(books.canonical(doc))
    elif target == "orphan":
        (pd / "shards" / "orphan.json").write_bytes(b"{}\n")
    else:
        m = pd / "physical-shard-candidate.json"
        doc = json.loads(m.read_bytes())
        doc["production_release_authorized"] = True
        m.write_bytes(books.canonical(doc))
    with pytest.raises(readback.PhysicalReadbackDenied):
        _verify(pd, ed, packed, ledger)


def test_symlinked_physical_exposure_denied(tmp_path: Path) -> None:
    pd, ed, packed, ledger = _fixture(tmp_path)
    member = ed / "exposures" / "shard-000000.json"
    member.unlink()
    member.symlink_to(pd / "shards" / "shard-000000.json")
    with pytest.raises(readback.PhysicalReadbackDenied, match="linked"):
        _verify(pd, ed, packed, ledger)


def test_resigned_exposure_token_still_fails_source_replay(
    tmp_path: Path,
) -> None:
    pd, ed, packed, ledger = _fixture(tmp_path)
    name = "exposures/shard-000000.json"
    member = ed / name
    event_doc = json.loads(member.read_bytes())
    event_doc["ordered_exposures"][1]["target_token_id"] = 110
    new_bytes = books.canonical(event_doc)
    member.write_bytes(new_bytes)
    # Even when attacker recomputes both JSON hashes, source-token equality,
    # target/exposure IDs and causal chain must remain independently checked.
    core = {k: v for k, v in ledger.items() if k != "manifest_sha256"}
    core["exposure_shards"][0]["sha256"] = books.sha(new_bytes)
    core["exposure_shards"][0]["byte_count"] = len(new_bytes)
    resign = _receipt(core)
    (ed / "physical-exposure-candidate.json").write_bytes(
        books.canonical(resign)
    )
    with pytest.raises(readback.PhysicalReadbackDenied, match="replay"):
        _verify(pd, ed, packed, resign)


def test_resigned_packing_mask_still_fails_target_replay(
    tmp_path: Path,
) -> None:
    pd, ed, packed, ledger = _fixture(tmp_path)
    member = pd / "shards/shard-000000.json"
    doc = json.loads(member.read_bytes())
    doc["blocks"][0]["loss_mask"][1] = 0
    raw = books.canonical(doc)
    member.write_bytes(raw)
    core = {k: v for k, v in packed.items() if k != "manifest_sha256"}
    core["shards"][0]["sha256"] = books.sha(raw)
    core["shards"][0]["byte_count"] = len(raw)
    resign = _receipt(core)
    (pd / "physical-shard-candidate.json").write_bytes(
        books.canonical(resign)
    )
    with pytest.raises(readback.PhysicalReadbackDenied):
        _verify(pd, ed, resign, ledger)
