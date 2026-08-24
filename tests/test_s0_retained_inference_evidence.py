from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.checkpoint import CheckpointIntegrityError, hash_json, verify_checkpoint
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.inference.s0_artifact_evidence import (
    SCHEMA,
    collect_retained_s0_inference_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def test_retained_real_s0_checkpoint_proves_first_party_inference(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    report = collect_retained_s0_inference_evidence(
        ROOT,
        "a" * 40,
        output,
        train_steps=4,
        seed=20260825,
        verify_checkout=False,
    )

    assert report["schema"] == SCHEMA
    assert report["status"] == "PASS"
    assert report["candidate"]["canonical_base"] == "random_init"
    assert report["candidate"]["pretraining_only"] is True
    assert report["candidate"]["foreign_pretrained_weights"] is False
    assert report["candidate"]["behavioral_alignment_weights"] is False
    assert report["training"]["parameter_count"] == 10_140
    assert report["training"]["steps"] == 4
    assert report["checkpoint"]["serialization_pickle"] is False
    assert report["checkpoint"]["corrupt_checkpoint_rejected"] is True
    assert report["inference"]["parity"]["passed"] is True
    assert report["inference"]["parity"]["max_abs_error"] == 0.0
    assert report["inference"]["parity"]["max_rel_error"] == 0.0
    assert report["inference"]["seeded_sampling"]["repeatable"] is True
    assert report["inference"]["stop_semantics"] == {
        "token_stop": "stop_token",
        "text_stop": "stop_string",
    }
    assert report["inference"]["context"]["exact_limit_stop"] == "context_limit"
    assert report["inference"]["context"]["over_limit_rejected"] is True
    assert report["inference"]["openai_compatible_raw_completion_equal"] is True
    assert report["inference"]["chat_semantics"] is False
    assert report["truth_boundary"]["windows_nvda_live_execution"] == "NOT_TESTED"

    checkpoint = output / "checkpoint"
    manifest = verify_checkpoint(checkpoint)
    assert manifest["checkpoint_id"] == report["checkpoint"]["checkpoint_id"]
    backend = load_first_party_backend(checkpoint)
    assert backend.diagnostics()["git_sha"] == "a" * 40

    on_disk = json.loads((output / "inference_evidence.json").read_text(encoding="utf-8"))
    report_hash = on_disk.pop("report_sha256")
    assert hash_json(on_disk) == report_hash
    assert report_hash == report["report_sha256"]


def test_retained_evidence_rejects_stale_identity_and_dirty_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="candidate_sha"):
        collect_retained_s0_inference_evidence(
            ROOT,
            "short",
            tmp_path / "bad-sha",
            train_steps=1,
            verify_checkout=False,
        )

    dirty = tmp_path / "dirty"
    dirty.mkdir()
    (dirty / "unexpected.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="pre-existing"):
        collect_retained_s0_inference_evidence(
            ROOT,
            "b" * 40,
            dirty,
            train_steps=1,
            verify_checkout=False,
        )


def test_retained_checkpoint_tamper_still_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    collect_retained_s0_inference_evidence(
        ROOT,
        "c" * 40,
        output,
        train_steps=1,
        verify_checkout=False,
    )
    weights = output / "checkpoint" / "weights.safetensors"
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(payload)

    with pytest.raises(CheckpointIntegrityError, match="checksum mismatch"):
        load_first_party_backend(output / "checkpoint")
