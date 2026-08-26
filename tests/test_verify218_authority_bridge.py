from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from twelve_six.checkpoint import hash_json
from twelve_six.verify218_authority_bridge import (
    PRODUCER_ARTIFACT_NAME,
    PRODUCER_WORKFLOW_RUN_ID,
    Verify218BridgeError,
    _resolve_roles,
    _verify_detailed_authority,
)
from twelve_six.verify218_learned_10m import STATE, WORKER
from twelve_six.verify218_resume_probe import _trainer_config


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _detailed() -> dict:
    value = {
        "schema": "12-6.verify218-learned-10m-independent.v1",
        "worker": WORKER,
        "state": STATE,
        "checkpoints": {
            "best": {"checkpoint_id": "a" * 64},
            "final": {"checkpoint_id": "b" * 64},
        },
        "boundaries": {
            "training_executed": False,
            "optimizer_updates": 0,
            "foreign_pretrained_weights": False,
            "evaluation_mutated_model": False,
        },
    }
    value["identity_sha256"] = hash_json(value)
    return value


def _artifact_fixture(root: Path) -> None:
    _write_json(
        root / "scale141-evidence" / "fresh-verification.json",
        {
            "ladder_common_evaluation": {
                "all_scheduled": {
                    "0": {"evaluation": {"bits_per_byte": 8.0}},
                    "500000": {"evaluation": {"bits_per_byte": 2.0}},
                    "1000000": {"evaluation": {"bits_per_byte": 1.0}},
                    "1500000": {"evaluation": {"bits_per_byte": 1.2}},
                    "2000000": {"evaluation": {"bits_per_byte": 1.4}},
                }
            }
        },
    )
    _write_json(
        root / "scale141-evidence" / "retained" / "index.json",
        {
            "roles": {
                "best": {
                    "target_optimized_tokens": 1_000_000,
                    "checkpoint_id": "a" * 64,
                },
                "final": {
                    "target_optimized_tokens": 2_000_000,
                    "checkpoint_id": "b" * 64,
                },
            }
        },
    )


def test_bridge_binds_terminal_learn217_source() -> None:
    assert PRODUCER_ARTIFACT_NAME == "learn217-terminal-10m-learned-base"
    assert PRODUCER_WORKFLOW_RUN_ID == 32_952_787_070


def test_detailed_authority_self_hash_and_truth_boundary_fail_closed() -> None:
    value = _detailed()
    _verify_detailed_authority(value)

    corrupt = deepcopy(value)
    corrupt["boundaries"]["optimizer_updates"] = 1
    corrupt["identity_sha256"] = hash_json(
        {key: item for key, item in corrupt.items() if key != "identity_sha256"}
    )
    with pytest.raises(Verify218BridgeError, match="optimizer updates"):
        _verify_detailed_authority(corrupt)


def test_role_resolution_uses_all_scheduled_evidence(tmp_path: Path) -> None:
    _artifact_fixture(tmp_path)
    resolved = _resolve_roles(tmp_path, _detailed())
    assert resolved["best_target_optimized_tokens"] == 1_000_000
    assert resolved["final_target_optimized_tokens"] == 2_000_000
    assert resolved["best_checkpoint_id"] == "a" * 64
    assert resolved["final_checkpoint_id"] == "b" * 64


def test_role_resolution_rejects_retained_best_substitution(tmp_path: Path) -> None:
    _artifact_fixture(tmp_path)
    index_path = tmp_path / "scale141-evidence" / "retained" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["roles"]["best"]["target_optimized_tokens"] = 500_000
    _write_json(index_path, index)
    with pytest.raises(Verify218BridgeError, match="best role"):
        _resolve_roles(tmp_path, _detailed())


def test_resume_probe_reconstructs_tuple_betas() -> None:
    identity = {
        "training_config": {
            "trainer": {
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "max_steps": 10,
                "warmup_steps": 0,
                "scheduler": "constant",
                "gradient_accumulation_steps": 1,
                "gradient_clip_norm": 1.0,
                "precision": "fp32",
                "seed": 7,
                "deterministic_algorithms": True,
                "deterministic_warn_only": False,
            }
        }
    }
    config = _trainer_config(identity)
    assert config.betas == (0.9, 0.95)
    assert config.seed == 7
