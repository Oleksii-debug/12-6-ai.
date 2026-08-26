from __future__ import annotations

from copy import deepcopy

import pytest

from twelve_six.model341_launch_readiness import (
    INIT_IDENTITY_SHA256,
    MODEL_IDENTITY_SHA256,
    MODEL_SOURCE_SHA,
    MODEL_STAGE,
    PARAMETER_COUNT,
    SCHEMA,
    LaunchReadinessError,
    assess_model341_launch,
)

SHA = "a" * 64
GIT = "b" * 40


def complete_packet() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "binding": {
            "stage": MODEL_STAGE,
            "source_sha": MODEL_SOURCE_SHA,
            "model_identity_sha256": MODEL_IDENTITY_SHA256,
            "init_identity_sha256": INIT_IDENTITY_SHA256,
            "parameter_count": PARAMETER_COUNT,
        },
        "tokenizer": {
            "status": "TERMINAL_SELECTED",
            "kind": "utf8-byte",
            "version": "s0-byte-v1",
            "vocab_size": 256,
            "identity_sha256": SHA,
            "authority_ref": "TOKENIZER-AUTHORITY",
        },
        "corpus": {
            "status": "TERMINAL_PASS",
            "corpus_manifest_sha256": SHA,
            "split_manifest_sha256": SHA,
            "packing_manifest_sha256": SHA,
            "unique_loss_ledger_sha256": SHA,
            "decontamination_manifest_sha256": SHA,
            "budget_unit": "post_pack_unique_causal_loss_positions",
            "measurement_method": "post_pack_loss_mask_ledger",
            "source_bytes": 500_000_000,
            "unique_causal_loss_positions": 120_000_000,
        },
        "checkpoint_recovery": {
            "status": "TERMINAL_PASS",
            "authority_ref": "D05-TERMINAL",
            "authority_sha": GIT,
        },
        "learned_ladder": {
            "3m": {
                "status": "INDEPENDENT_PASS",
                "authority_ref": "VERIFY-219",
                "authority_sha": GIT,
            },
            "10m": {
                "status": "INDEPENDENT_PASS",
                "authority_ref": "VERIFY-218",
                "authority_sha": GIT,
            },
        },
        "evaluation_firewall": {
            "status": "TERMINAL_PASS",
            "selection_validation_identity_sha256": SHA,
            "final_test_registry_sha256": SHA,
            "exclusion_manifest_sha256": SHA,
            "final_test_access_before_terminal": False,
        },
        "training_recipe": {
            "status": "TERMINAL_PASS",
            "optimizer": "AdamW",
            "scheduler": "cosine",
            "precision": "bf16",
            "learning_rate": 0.0003,
            "warmup_steps": 100,
            "seeds": [341],
            "target_unique_loss_positions": 103_067_200,
            "checkpoint_interval_steps": 100,
            "stop_rules": ["nonfinite_loss", "checkpoint_integrity_failure"],
        },
        "compute_plan": {
            "status": "TERMINAL_PASS",
            "estimated_flops": 12_000_000_000_000_000,
            "max_wall_minutes": 180,
            "hardware": "qualified accelerator target",
            "max_cost_usd": 50.0,
        },
        "authorization": {
            "compute_status": "NOT_AUTHORIZED",
            "training_status": "NOT_AUTHORIZED",
            "scope": "NONE",
            "authority_ref": None,
        },
        "smoke_result": {
            "status": "NOT_RUN",
            "authority_ref": None,
            "authority_sha": None,
        },
    }


def test_complete_science_never_synthesizes_financial_authorization() -> None:
    result = assess_model341_launch(complete_packet())
    assert result.scientific_packet_complete is True
    assert result.ready_for_authorization_request is True
    assert result.bounded_smoke_authorized is False
    assert result.long_training_authorized is False
    assert "explicit_compute_and_training_authorization_missing" in result.blockers


def test_source_bytes_cannot_substitute_for_unique_loss_ledger() -> None:
    packet = complete_packet()
    corpus = packet["corpus"]
    assert isinstance(corpus, dict)
    corpus["unique_causal_loss_positions"] = 0
    corpus["unique_loss_ledger_sha256"] = None
    result = assess_model341_launch(packet)
    assert "unique_causal_loss_positions_missing" in result.blockers
    assert "corpus_unique_loss_ledger_sha256_missing" in result.blockers
    assert result.ready_for_authorization_request is False


def test_wrong_measurement_unit_fails_closed() -> None:
    packet = complete_packet()
    corpus = packet["corpus"]
    assert isinstance(corpus, dict)
    corpus["budget_unit"] = "source_bytes"
    result = assess_model341_launch(packet)
    assert "corpus_budget_unit_invalid" in result.blockers


def test_training_budget_cannot_exceed_unique_authorized_positions() -> None:
    packet = complete_packet()
    recipe = packet["training_recipe"]
    assert isinstance(recipe, dict)
    recipe["target_unique_loss_positions"] = 120_000_001
    result = assess_model341_launch(packet)
    assert "training_budget_exceeds_unique_corpus_authority" in result.blockers


def test_bounded_smoke_requires_explicit_scope_and_authority() -> None:
    packet = complete_packet()
    authorization = packet["authorization"]
    assert isinstance(authorization, dict)
    authorization.update(
        compute_status="COMPUTE_AUTHORIZED",
        training_status="TRAINING_AUTHORIZED",
        scope="BOUNDED_SMOKE",
        authority_ref="OWNER-AUTH-001",
    )
    result = assess_model341_launch(packet)
    assert result.bounded_smoke_authorized is True
    assert result.long_training_authorized is False


def test_long_training_requires_terminal_smoke_evidence() -> None:
    packet = complete_packet()
    authorization = packet["authorization"]
    assert isinstance(authorization, dict)
    authorization.update(
        compute_status="COMPUTE_AUTHORIZED",
        training_status="TRAINING_AUTHORIZED",
        scope="LONG_TRAINING",
        authority_ref="OWNER-AUTH-002",
    )
    blocked = assess_model341_launch(packet)
    assert blocked.long_training_authorized is False
    assert "bounded_smoke_not_terminal_pass" in blocked.blockers

    smoke = packet["smoke_result"]
    assert isinstance(smoke, dict)
    smoke.update(status="TERMINAL_PASS", authority_ref="SMOKE-RESULT", authority_sha=GIT)
    passed = assess_model341_launch(packet)
    assert passed.long_training_authorized is True


def test_exact_model_identity_is_mandatory() -> None:
    packet = complete_packet()
    binding = packet["binding"]
    assert isinstance(binding, dict)
    binding["parameter_count"] = 20_000_000
    binding["model_identity_sha256"] = "c" * 64
    result = assess_model341_launch(packet)
    assert "parameter_count_not_bound" in result.blockers
    assert "model_identity_not_bound" in result.blockers


def test_bad_schema_is_malformed_not_incomplete() -> None:
    packet = deepcopy(complete_packet())
    packet["schema"] = "other"
    with pytest.raises(LaunchReadinessError, match="unsupported launch packet schema"):
        assess_model341_launch(packet)
