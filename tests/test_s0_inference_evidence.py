from __future__ import annotations

import copy
from pathlib import Path

import pytest

from twelve_six.inference.s0_evidence import (
    SCHEMA,
    collect_s0_inference_evidence,
    validate_s0_inference_evidence,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_SHA = "a" * 40


def test_real_trained_checkpoint_reload_inference_evidence(tmp_path: Path) -> None:
    output_dir = tmp_path / "inference-evidence"
    report = collect_s0_inference_evidence(
        REPO_ROOT,
        TEST_SHA,
        output_dir,
        train_steps=4,
        seed=20260825,
        verify_checkout=False,
    )

    assert report["schema"] == SCHEMA
    assert report["candidate"]["sha"] == TEST_SHA
    assert report["candidate"]["canonical_base"] == "random_init"
    assert report["candidate"]["parameter_count"] == 10_140
    assert report["training_fixture"]["steps"] == 4
    assert report["checkpoint"]["serialization_pickle"] is False
    assert (output_dir / "checkpoint-v1" / "model.safetensors").is_file()
    assert (output_dir / "inference_evidence.json").is_file()

    assert report["greedy"]["direct_reload_equal"] is True
    assert report["seeded_sampling"]["repeatable"] is True
    assert report["seeded_sampling"]["direct_reload_equal"] is True
    assert report["parity"]["passed"] is True
    assert report["parity"]["atol"] == 0.0
    assert report["parity"]["rtol"] == 0.0
    assert report["parity"]["failures"] == []

    stops = report["stop_semantics"]
    assert stops["token_stop_reason"] == "stop_token"
    assert stops["string_stop_reason"] == "stop_string"
    assert stops["string_stop_stripped_to_empty"] is True
    assert stops["context_stop_reason"] == "context_limit"
    assert stops["over_context_prompt_rejected"] is True

    assert report["openai_completion_handoff"]["raw_completion_matches_greedy"] is True
    assert report["openai_completion_handoff"]["chat_semantics_supported"] is False
    assert all(report["fail_closed"].values())
    validate_s0_inference_evidence(report, expected_candidate_sha=TEST_SHA)


def test_inference_evidence_validator_rejects_tamper(tmp_path: Path) -> None:
    report = collect_s0_inference_evidence(
        REPO_ROOT,
        TEST_SHA,
        tmp_path / "evidence",
        train_steps=4,
        seed=20260826,
        verify_checkout=False,
    )
    tampered = copy.deepcopy(report)
    tampered["seeded_sampling"]["repeatable"] = False

    with pytest.raises(ValueError, match="sampling"):
        validate_s0_inference_evidence(tampered, expected_candidate_sha=TEST_SHA)


def test_inference_evidence_rejects_stale_or_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="40-hex"):
        collect_s0_inference_evidence(
            REPO_ROOT,
            "abc",
            tmp_path / "invalid-sha",
            train_steps=4,
            verify_checkout=False,
        )

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(FileExistsError, match="absent or empty"):
        collect_s0_inference_evidence(
            REPO_ROOT,
            TEST_SHA,
            occupied,
            train_steps=4,
            verify_checkout=False,
        )
