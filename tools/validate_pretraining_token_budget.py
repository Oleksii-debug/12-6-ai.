#!/usr/bin/env python3
"""Validate the scientific pretraining token-budget control."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs/control/pretraining_token_budget_v1.json"

EXPECTED_20M_PARAMS = 20_613_440
REFERENCE_TOKENS_PER_PARAMETER = 20


def canonical_identity(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("evidence_identity_sha256", None)
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stage(value: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [row for row in value["stages"] if row["name"] == name]
    assert len(matches) == 1
    return matches[0]


def validate(value: dict[str, Any]) -> None:
    assert value["schema"] == "12-6.pretraining-token-budget.v1"
    assert value["repository"] == "Oleksii-debug/12-6-ai."
    assert value["execution_profile"] == "LOCAL_FREE"
    assert value["training_executed"] is False
    assert value["paid_compute_used"] is False
    assert value["compute_authorized"] is False

    policy = value["policy"]
    assert policy["compute_optimal_reference_tokens_per_parameter"] == REFERENCE_TOKENS_PER_PARAMETER
    assert policy["no_replay_for_budget_credit"] is True
    assert policy["count_only_training_authorized_loss_positions"] is True
    assert policy["tokenizer_identity_must_be_bound"] is True
    assert policy["corpus_identity_must_be_bound"] is True
    assert policy["checkpoint_integrity_must_be_terminal"] is True

    model20 = stage(value, "MODEL-341-20M")
    assert model20["parameter_count"] == EXPECTED_20M_PARAMS
    assert model20["compute_optimal_reference_tokens"] == EXPECTED_20M_PARAMS * REFERENCE_TOKENS_PER_PARAMETER
    assert model20["current_preregistered_unique_target_request"] == 20_000_000
    assert model20["current_preregistered_unique_target_request"] < model20["compute_optimal_reference_tokens"]
    assert model20["current_request_classification"] == (
        "PIPELINE_PILOT_NOT_SCIENCE_COMPLETE_20M_BASELINE"
    )

    model100 = stage(value, "100M")
    assert model100["parameter_count"] == 100_000_000
    assert model100["compute_optimal_reference_tokens"] == 2_000_000_000

    model1b = stage(value, "1B")
    assert model1b["parameter_count"] == 1_000_000_000
    assert model1b["compute_optimal_reference_tokens"] == 20_000_000_000

    decision = value["current_decision"]
    assert decision["20m_architecture_ready_for_pilot_after_blockers"] is True
    assert decision["20m_quality_baseline_data_budget_ready"] is False
    assert decision["100m_training_ready"] is False
    assert decision["1b_training_ready"] is False

    rules = value["promotion_rules"]
    assert rules["first_learned_checkpoint"]["quality_claim_allowed"] is False
    assert rules["science_complete_size_baseline"]["quality_claim_allowed"] is True
    assert rules["larger_model_scale_up"]["requires_previous_stage_learned"] is True
    assert rules["larger_model_scale_up"]["requires_terminal_checkpoint_integrity"] is True

    assert canonical_identity(value) == value["evidence_identity_sha256"]


def main() -> int:
    value = json.loads(CONTROL.read_text(encoding="utf-8"))
    validate(value)
    model20 = stage(value, "MODEL-341-20M")
    print("PRETRAINING_TOKEN_BUDGET=PASS")
    print("20M_PARAMS=" + str(model20["parameter_count"]))
    print("20M_PILOT_TARGETS=" + str(model20["current_preregistered_unique_target_request"]))
    print("20M_REFERENCE_TARGETS=" + str(model20["compute_optimal_reference_tokens"]))
    print("20M_CLASS=" + model20["current_request_classification"])
    print("TOKEN_BUDGET_EVIDENCE_SHA256=" + value["evidence_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
