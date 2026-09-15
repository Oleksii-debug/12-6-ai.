from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_d03_clean_g05_g06_replay_v1.py"
SPEC = importlib.util.spec_from_file_location("d03_clean_g05_g06_replay_v1", TOOL)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def _clean_evidence() -> dict:
    core = {
        "schema_version": "12-6.d03-nomis-free-data526-successor-evidence.v1",
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": "f" * 40,
        "source_report_sha256": replay.PHYSICAL_SOURCE_REPORT_SHA256,
        "survivor_authority_sha256": replay.PHYSICAL_SURVIVOR_AUTHORITY_SHA256,
        "raw_text_emitted_to_durable_evidence": False,
        "clean_data526": {
            "record_count": replay.CLEAN_RECORD_COUNT,
            "source_object_count": replay.CLEAN_SOURCE_OBJECT_COUNT,
            "total_payload_bytes": replay.CLEAN_TOTAL_PAYLOAD_BYTES,
            "record_payload_jsonl_sha256": replay.CLEAN_RECORD_PAYLOAD_JSONL_SHA256,
            "record_inventory_digest_sha256": replay.CLEAN_RECORD_INVENTORY_SHA256,
            "payload_inventory_digest_sha256": replay.CLEAN_PAYLOAD_INVENTORY_SHA256,
        },
        "truth_boundary": {
            "tokenizer_fit_authorized": False,
            "authorized_training_exposure": 0,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "learned_weights_created": False,
            "final_test_payload_read": False,
            "paid_compute_used": False,
        },
    }
    return {
        **core,
        "evidence_identity_sha256": replay.sha256(replay.canonical(core)),
    }


def _receipt(run_id: str, g05_raw: bytes, g06_raw: bytes) -> dict:
    core = {
        "schema_version": replay.SCHEMA,
        "execution_profile": replay.EXECUTION_PROFILE,
        "run_id": run_id,
        "execution_head_sha": "a" * 40,
        "runner_git_blob_sha1": "b" * 40,
        "dependencies": {},
        "fresh_reconstruction_evidence_identity_sha256": "c" * 64,
        "clean_record_payload_jsonl_sha256": replay.CLEAN_RECORD_PAYLOAD_JSONL_SHA256,
        "clean_record_inventory_digest_sha256": replay.CLEAN_RECORD_INVENTORY_SHA256,
        "clean_payload_inventory_digest_sha256": replay.CLEAN_PAYLOAD_INVENTORY_SHA256,
        "covered_record_count": replay.CLEAN_RECORD_COUNT,
        "covered_payload_bytes": replay.CLEAN_TOTAL_PAYLOAD_BYTES,
        "covered_source_object_count": replay.CLEAN_SOURCE_OBJECT_COUNT,
        "g05_g06_input_rows_sha256": "d" * 64,
        "g05_execution_identity_sha256": "e" * 64,
        "g05_authority_sha256": replay.sha256(g05_raw),
        "g06_execution_identity_sha256": "f" * 64,
        "g06_authority_sha256": replay.sha256(g06_raw),
        "truth_boundary": dict(replay.TRUTH_BOUNDARY),
    }
    return {
        **core,
        "replay_execution_authority_sha256": replay.sha256(replay.cjson(core)),
    }


def test_runtime_bindings_are_exactly_pinned() -> None:
    observed = replay.validate_runtime_bindings(ROOT)
    assert observed[str(replay.QUALITY_MODULE_PATH)] == replay.QUALITY_MODULE_BLOB_SHA1
    assert observed[str(replay.PRIVACY_MODULE_PATH)] == replay.PRIVACY_MODULE_BLOB_SHA1
    assert observed[str(replay.CLEAN_MATERIALIZER_PATH)] == replay.CLEAN_MATERIALIZER_BLOB_SHA1


def test_truth_boundary_stays_zero_credit() -> None:
    boundary = replay.TRUTH_BOUNDARY
    assert boundary["clean_g05_g06_replay_executed"] is True
    assert boundary["reserved_evaluation_decontamination_executed"] is False
    assert boundary["current_retained_corpus_launch_authoritative"] is False
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["authorized_optimized_target_exposure"] == 0
    assert boundary["optimizer_updates_executed_on_real_targets"] == 0
    assert boundary["training_executed"] is False
    assert boundary["learned_weights_created"] is False
    assert boundary["final_test_outcomes_read"] is False
    assert boundary["paid_compute_used"] is False


def test_clean_evidence_is_exact_root_bound_and_self_hashed() -> None:
    evidence = _clean_evidence()
    assert replay.validate_clean_evidence(evidence) == evidence["evidence_identity_sha256"]
    evidence["clean_data526"]["record_count"] = 273
    body = dict(evidence)
    body.pop("evidence_identity_sha256")
    evidence["evidence_identity_sha256"] = replay.sha256(replay.canonical(body))
    with pytest.raises(replay.CleanG05G06ReplayError, match="record_count drift"):
        replay.validate_clean_evidence(evidence)


def test_inventory_recomputes_roots_and_rejects_nomis(monkeypatch) -> None:
    rows = [
        {
            "record_id": "a", "source_id": "source-a", "family": "family-a",
            "modality": "en", "payload_sha256": replay.sha256(b"abc"),
            "payload_bytes": 3,
        },
        {
            "record_id": "b", "source_id": "source-b", "family": "family-b",
            "modality": "uk", "payload_sha256": replay.sha256("ї".encode()),
            "payload_bytes": len("ї".encode()),
        },
    ]
    record_root = replay.sha256(replay.canonical(rows))
    payload_root = replay.sha256(replay.canonical([
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]))
    total = sum(row["payload_bytes"] for row in rows)
    monkeypatch.setattr(replay, "CLEAN_RECORD_COUNT", 2)
    monkeypatch.setattr(replay, "CLEAN_SOURCE_OBJECT_COUNT", 2)
    monkeypatch.setattr(replay, "CLEAN_TOTAL_PAYLOAD_BYTES", total)
    monkeypatch.setattr(replay, "CLEAN_RECORD_INVENTORY_SHA256", record_root)
    monkeypatch.setattr(replay, "CLEAN_PAYLOAD_INVENTORY_SHA256", payload_root)
    inventory = {
        "schema_version": "12-6.data526-record-inventory.v1",
        "record_count": 2,
        "total_payload_bytes": total,
        "record_inventory_digest_sha256": record_root,
        "payload_inventory_digest_sha256": payload_root,
        "records": rows,
    }
    assert replay.normalize_inventory(inventory) == rows
    inventory["records"][0]["family"] = replay.NOMIS_FAMILY
    with pytest.raises(replay.CleanG05G06ReplayError, match="Nomis family"):
        replay.normalize_inventory(inventory)


def test_pair_requires_distinct_replays_and_identical_g05_g06(tmp_path: Path) -> None:
    g05_raw = b'{"authority":"g05"}\n'
    g06_raw = b'{"authority":"g06"}\n'
    receipt_a = _receipt("run-a", g05_raw, g06_raw)
    receipt_b = _receipt("run-b", g05_raw, g06_raw)
    paths = {}
    for name, raw in {
        "replay-a.json": replay.cjson(receipt_a),
        "replay-b.json": replay.cjson(receipt_b),
        "g05-a.json": g05_raw,
        "g05-b.json": g05_raw,
        "g06-a.json": g06_raw,
        "g06-b.json": g06_raw,
    }.items():
        path = tmp_path / name
        path.write_bytes(raw)
        paths[name] = path
    out = tmp_path / "pair.json"
    pair = replay.verify_pair(
        replay_a=paths["replay-a.json"], replay_b=paths["replay-b.json"],
        g05_a=paths["g05-a.json"], g05_b=paths["g05-b.json"],
        g06_a=paths["g06-a.json"], g06_b=paths["g06-b.json"], output=out,
    )
    assert pair["two_independent_replays_agree"] is True
    assert json.loads(out.read_text())["pair_execution_authority_sha256"] == pair[
        "pair_execution_authority_sha256"
    ]

    paths["replay-b.json"].write_bytes(replay.cjson(_receipt("run-a", g05_raw, g06_raw)))
    with pytest.raises(replay.CleanG05G06ReplayError, match="run ids must be distinct"):
        replay.verify_pair(
            replay_a=paths["replay-a.json"], replay_b=paths["replay-b.json"],
            g05_a=paths["g05-a.json"], g05_b=paths["g05-b.json"],
            g06_a=paths["g06-a.json"], g06_b=paths["g06-b.json"], output=out,
        )
