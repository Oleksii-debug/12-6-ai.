from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from typing import Any

import pytest

from twelve_six.checkpoint import hash_json
from twelve_six.inference.s0_http_evidence import (
    REPOSITORY,
    SCHEMA,
    collect_s0_http_parity_evidence,
    validate_http_parity_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def _head() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    assert len(value) in {40, 64}
    return value


def _rehash(payload: dict[str, Any]) -> None:
    material = dict(payload)
    material.pop("evidence_sha256", None)
    payload["evidence_sha256"] = hash_json(material)


def _valid_evidence() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "candidate": {
            "repository": REPOSITORY,
            "sha": "a" * 40,
            "random_init_pretraining_only": True,
        },
        "training": {
            "optimizer_steps": 1,
            "tokens_seen": 10,
            "paid_compute": False,
        },
        "checkpoint": {"checkpoint_id": "b" * 64},
        "parity": {
            "passed": True,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
        },
        "http": {
            "health_ok": True,
            "model_list_ok": True,
            "greedy_matches_direct": True,
            "sampled_matches_direct": True,
            "seeded_sampling_repeatable": True,
            "stop_matches_direct": True,
            "context_limit_matches_direct": True,
            "over_context_rejected": True,
            "chat_rejected": True,
        },
        "raw_base_semantics": {
            "hidden_prompt": False,
            "chat_roles": False,
            "instruction_template": False,
            "alignment_behavior": False,
        },
    }
    _rehash(payload)
    return payload


def test_real_trained_checkpoint_reloads_and_matches_loopback_http(tmp_path: Path) -> None:
    head = _head()
    evidence = collect_s0_http_parity_evidence(
        ROOT,
        head,
        tmp_path / "http-parity",
        train_steps=2,
        seed=20260825,
    )

    assert evidence["schema"] == SCHEMA
    assert evidence["candidate"]["sha"] == head
    assert evidence["candidate"]["random_init_pretraining_only"] is True
    assert evidence["candidate"]["parameter_count"] == 10_140
    assert evidence["training"]["optimizer_steps"] == 2
    assert evidence["training"]["tokens_seen"] > 0
    assert evidence["training"]["paid_compute"] is False
    assert evidence["checkpoint"]["serialization_pickle"] is False
    assert evidence["checkpoint"]["git_sha"] == head
    assert evidence["parity"]["passed"] is True
    assert evidence["parity"]["max_abs_error"] == 0.0
    assert evidence["parity"]["max_rel_error"] == 0.0
    assert all(
        evidence["http"][key] is True
        for key in (
            "health_ok",
            "model_list_ok",
            "greedy_matches_direct",
            "sampled_matches_direct",
            "seeded_sampling_repeatable",
            "stop_matches_direct",
            "context_limit_matches_direct",
            "over_context_rejected",
            "chat_rejected",
            "loopback_only",
        )
    )
    assert (tmp_path / "http-parity/checkpoint/manifest.json").is_file()
    assert (tmp_path / "http-parity/s0-http-parity-evidence.json").is_file()
    validate_http_parity_evidence(evidence, expected_candidate_sha=head)


def test_http_parity_validator_rejects_semantic_tamper() -> None:
    tampered = copy.deepcopy(_valid_evidence())
    tampered["http"]["sampled_matches_direct"] = False
    _rehash(tampered)

    with pytest.raises(ValueError, match="required HTTP parity"):
        validate_http_parity_evidence(tampered, expected_candidate_sha="a" * 40)


def test_http_parity_validator_rejects_stale_candidate() -> None:
    evidence = _valid_evidence()

    with pytest.raises(ValueError, match="candidate SHA is stale"):
        validate_http_parity_evidence(
            evidence,
            expected_candidate_sha="0" * 40,
        )
