#!/usr/bin/env python3
"""Validate the scientific pretraining token-budget control without assert semantics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs/control/pretraining_token_budget_v1.json"

EXPECTED_20M_PARAMS = 20_613_440
REFERENCE_TOKENS_PER_PARAMETER = 20


class TokenBudgetValidationError(ValueError):
    """Raised when the pretraining token budget violates a control invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TokenBudgetValidationError(message)


def canonical_identity(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("evidence_identity_sha256", None)
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def stage(value: dict[str, Any], name: str) -> dict[str, Any]:
    stages = value.get("stages")
    require(isinstance(stages, list), "stages must be a list")
    matches = [row for row in stages if isinstance(row, dict) and row.get("name") == name]
    require(len(matches) == 1, f"expected exactly one stage named {name}")
    return matches[0]


def validate(value: dict[str, Any]) -> None:
    require(value.get("schema") == "12-6.pretraining-token-budget.v1", "unexpected schema")
    require(value.get("repository") == "Oleksii-debug/12-6-ai.", "unexpected repository")
    require(value.get("execution_profile") == "LOCAL_FREE", "execution profile must remain LOCAL_FREE")
    require(value.get("training_executed") is False, "control file cannot claim training execution")
    require(value.get("paid_compute_used") is False, "control file cannot claim paid compute")
    require(value.get("compute_authorized") is False, "control file cannot authorize compute")

    policy = value.get("policy", {})
    require(
        policy.get("compute_optimal_reference_tokens_per_parameter") == REFERENCE_TOKENS_PER_PARAMETER,
        "unexpected compute-optimal planning ratio",
    )
    require(policy.get("no_replay_for_budget_credit") is True, "replay cannot receive budget credit")
    require(
        policy.get("count_only_training_authorized_loss_positions") is True,
        "only training-authorized loss positions may count",
    )
    require(policy.get("tokenizer_identity_must_be_bound") is True, "tokenizer identity binding is mandatory")
    require(policy.get("corpus_identity_must_be_bound") is True, "corpus identity binding is mandatory")
    require(
        policy.get("checkpoint_integrity_must_be_terminal") is True,
        "terminal checkpoint integrity is mandatory",
    )

    model20 = stage(value, "MODEL-341-20M")
    require(model20.get("parameter_count") == EXPECTED_20M_PARAMS, "wrong MODEL-341 parameter count")
    require(
        model20.get("compute_optimal_reference_tokens")
        == EXPECTED_20M_PARAMS * REFERENCE_TOKENS_PER_PARAMETER,
        "wrong 20M planning reference token count",
    )
    require(
        model20.get("current_preregistered_unique_target_request") == 20_000_000,
        "wrong current 20M pilot target request",
    )
    require(
        model20["current_preregistered_unique_target_request"] < model20["compute_optimal_reference_tokens"],
        "pilot must remain below the science-reference target in this authority",
    )
    require(
        model20.get("current_request_classification")
        == "PIPELINE_PILOT_NOT_SCIENCE_COMPLETE_20M_BASELINE",
        "20M pilot cannot be promoted to a science-complete baseline",
    )

    model100 = stage(value, "100M")
    require(model100.get("parameter_count") == 100_000_000, "wrong 100M stage parameter count")
    require(model100.get("compute_optimal_reference_tokens") == 2_000_000_000, "wrong 100M reference tokens")

    model1b = stage(value, "1B")
    require(model1b.get("parameter_count") == 1_000_000_000, "wrong 1B stage parameter count")
    require(model1b.get("compute_optimal_reference_tokens") == 20_000_000_000, "wrong 1B reference tokens")

    decision = value.get("current_decision", {})
    require(
        decision.get("20m_architecture_ready_for_pilot_after_blockers") is True,
        "20M mechanics decision drifted unexpectedly",
    )
    require(
        decision.get("20m_quality_baseline_data_budget_ready") is False,
        "20M quality data budget must remain not ready",
    )
    require(decision.get("100m_training_ready") is False, "100M training must remain blocked")
    require(decision.get("1b_training_ready") is False, "1B training must remain blocked")

    rules = value.get("promotion_rules", {})
    first = rules.get("first_learned_checkpoint", {})
    science = rules.get("science_complete_size_baseline", {})
    larger = rules.get("larger_model_scale_up", {})
    require(first.get("quality_claim_allowed") is False, "first learned checkpoint cannot imply quality baseline")
    require(science.get("quality_claim_allowed") is True, "science-complete baseline rule drifted")
    require(larger.get("requires_previous_stage_learned") is True, "larger scale must require learned prior stage")
    require(
        larger.get("requires_terminal_checkpoint_integrity") is True,
        "larger scale must require terminal checkpoint integrity",
    )

    require(
        canonical_identity(value) == value.get("evidence_identity_sha256"),
        "pretraining token-budget evidence identity mismatch",
    )


def main() -> int:
    value = json.loads(CONTROL.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "control root must be a JSON object")
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
