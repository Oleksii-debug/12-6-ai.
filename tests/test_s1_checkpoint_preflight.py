from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from typing import Any

import pytest

from twelve_six.checkpoint import hash_json
from twelve_six.checkpoint.s1_preflight import (
    AUTHORITY,
    FIXTURE_SCOPE,
    REPOSITORY,
    SCHEMA,
    collect_s1_checkpoint_preflight,
    validate_s1_checkpoint_preflight,
)

ROOT = Path(__file__).resolve().parents[1]


def _head() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert len(value) in {40, 64}
    return value


def _rehash(payload: dict[str, Any]) -> None:
    material = dict(payload)
    material.pop("evidence_sha256", None)
    payload["evidence_sha256"] = hash_json(material)


def _valid_minimal() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "repository": REPOSITORY,
        "candidate_sha": "a" * 40,
        "s1_architecture_status": "ENGINEERING_CANDIDATE_NOT_FROZEN",
        "s1_tokenizer_selected": False,
        "s1_data_selected": False,
        "fixture_scope": FIXTURE_SCOPE,
        "canonical_binding": {
            "accepted": False,
            "rejected_as_expected": True,
            "reason": "ModelSpec/tokenizer vocab mismatch: model=512, tokenizer=256",
        },
        "checkpoint": {"save_verified": True, "pickle": False},
        "resume": {"model_state_exact": True, "trainer_state_exact": True},
        "constraints": {
            "paid_compute": False,
            "promotion_claimed": False,
            "s1_quality_claimed": False,
        },
    }
    _rehash(payload)
    return payload


def test_real_s1_checkpoint_preflight_is_exact_but_noncanonical(tmp_path: Path) -> None:
    head = _head()
    evidence = collect_s1_checkpoint_preflight(
        ROOT,
        head,
        tmp_path / "s1-checkpoint-preflight",
        total_steps=2,
        split_step=1,
        seed=20260825,
    )

    assert evidence["candidate_sha"] == head
    assert evidence["authority"] == AUTHORITY
    assert evidence["model"]["parameter_count"] == 107_856
    assert evidence["model"]["model_vocab_size"] == 512
    assert evidence["fixture"]["tokenizer_vocab_size"] == 256
    assert evidence["s1_tokenizer_selected"] is False
    assert evidence["s1_data_selected"] is False
    assert evidence["canonical_binding"]["accepted"] is False
    assert evidence["canonical_binding"]["rejected_as_expected"] is True
    assert "ModelSpec/tokenizer vocab mismatch" in evidence["canonical_binding"]["reason"]
    assert evidence["checkpoint"]["save_verified"] is True
    assert evidence["checkpoint"]["pickle"] is False
    assert evidence["checkpoint"]["format"] == "12-6-checkpoint"
    assert evidence["checkpoint"]["format_version"] == 1
    assert evidence["resume"]["model_state_exact"] is True
    assert evidence["resume"]["trainer_state_exact"] is True
    assert evidence["resume"]["baseline_tokens_seen"] == evidence["resume"]["resumed_tokens_seen"]
    assert (tmp_path / "s1-checkpoint-preflight/checkpoint/manifest.json").is_file()
    assert (tmp_path / "s1-checkpoint-preflight/s1-checkpoint-preflight.json").is_file()
    validate_s1_checkpoint_preflight(evidence, expected_candidate_sha=head)


def test_validator_rejects_premature_s1_tokenizer_claim() -> None:
    payload = copy.deepcopy(_valid_minimal())
    payload["s1_tokenizer_selected"] = True
    _rehash(payload)

    with pytest.raises(ValueError, match="must not select an S1 tokenizer"):
        validate_s1_checkpoint_preflight(payload, expected_candidate_sha="a" * 40)


def test_validator_rejects_quality_or_promotion_overclaim() -> None:
    payload = copy.deepcopy(_valid_minimal())
    payload["constraints"]["s1_quality_claimed"] = True
    _rehash(payload)
    with pytest.raises(ValueError, match="cannot claim S1 quality"):
        validate_s1_checkpoint_preflight(payload, expected_candidate_sha="a" * 40)

    payload = copy.deepcopy(_valid_minimal())
    payload["constraints"]["promotion_claimed"] = True
    _rehash(payload)
    with pytest.raises(ValueError, match="cannot grant promotion"):
        validate_s1_checkpoint_preflight(payload, expected_candidate_sha="a" * 40)


def test_validator_rejects_stale_candidate_and_tamper() -> None:
    payload = _valid_minimal()
    with pytest.raises(ValueError, match="candidate SHA is stale"):
        validate_s1_checkpoint_preflight(payload, expected_candidate_sha="0" * 40)

    payload = copy.deepcopy(payload)
    payload["resume"]["model_state_exact"] = False
    with pytest.raises(ValueError, match="interrupted/resumed preflight is not exact"):
        validate_s1_checkpoint_preflight(payload, expected_candidate_sha="a" * 40)
