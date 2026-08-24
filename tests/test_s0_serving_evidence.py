from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from twelve_six.inference.s0_serving_evidence import (
    S0ServingEvidenceError,
    collect_serving_evidence,
    validate_serving_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


@pytest.fixture(scope="module")
def serving_bundle(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("s0-serving-evidence")
    evidence = collect_serving_evidence(
        ROOT,
        candidate_sha=_git_head(),
        output_dir=output,
        train_steps=4,
        seed=20260825,
    )
    return evidence, output


def test_real_trained_checkpoint_reaches_loopback_http(serving_bundle) -> None:
    evidence, output = serving_bundle

    assert evidence["training"]["optimizer_steps"] == 4
    assert evidence["training"]["tokens_seen"] > 0
    assert (
        evidence["training"]["initial_model_state_sha256"]
        != evidence["training"]["trained_model_state_sha256"]
    )
    assert evidence["checkpoint"]["reload_verified"] is True
    assert evidence["checkpoint"]["direct_reloaded_logits_exact"] is True
    assert evidence["http"]["health_status"] == 200
    assert evidence["http"]["models_status"] == 200
    assert evidence["http"]["greedy_status"] == 200
    assert evidence["http"]["sample_status_a"] == 200
    assert evidence["http"]["sample_status_b"] == 200
    assert evidence["http"]["context_overflow_status"] == 400
    assert evidence["http"]["chat_status"] == 404
    assert evidence["parity"] == {
        "greedy_direct_vs_http": True,
        "sample_direct_vs_http": True,
        "sample_http_repeatable": True,
    }

    checkpoint = output / "checkpoint"
    assert (checkpoint / "manifest.json").is_file()
    assert (checkpoint / "MANIFEST.sha256").is_file()
    assert (checkpoint / "weights.safetensors").is_file()
    assert (output / "serving_evidence.json").is_file()


def test_evidence_is_privacy_safe_and_truth_bounded(serving_bundle) -> None:
    evidence, _ = serving_bundle
    encoded = json.dumps(evidence, sort_keys=True)

    assert "12-6 Base serving probe" not in encoded
    assert evidence["http"]["host"] == "127.0.0.1"
    assert evidence["http"]["externally_exposed"] is False
    assert all(value is False for value in evidence["claims"].values())
    validate_serving_evidence(evidence, expected_candidate_sha=_git_head())


def test_validator_rejects_stale_candidate(serving_bundle) -> None:
    evidence, _ = serving_bundle
    with pytest.raises(S0ServingEvidenceError, match="stale"):
        validate_serving_evidence(evidence, expected_candidate_sha="0" * 40)


def test_validator_rejects_transport_parity_drift(serving_bundle) -> None:
    evidence, _ = serving_bundle
    tampered = copy.deepcopy(evidence)
    tampered["parity"]["greedy_direct_vs_http"] = False

    with pytest.raises(S0ServingEvidenceError, match="greedy HTTP parity failed"):
        validate_serving_evidence(tampered, expected_candidate_sha=_git_head())


def test_validator_rejects_evidence_hash_tamper(serving_bundle) -> None:
    evidence, _ = serving_bundle
    tampered = copy.deepcopy(evidence)
    tampered["evidence_sha256"] = "0" * 64

    with pytest.raises(S0ServingEvidenceError, match="evidence hash mismatch"):
        validate_serving_evidence(tampered, expected_candidate_sha=_git_head())
