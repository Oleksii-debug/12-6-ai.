#!/usr/bin/env python3
"""Validate tokenizer-aware scientific pretraining budget controls."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "configs" / "control" / "pretraining_token_budget_v1.json"

EXPECTED_PARAMS = 20_613_440
EXPECTED_PILOT_BYTE_POSITIONS = 20_000_000
EXPECTED_TOKENIZER = "s0-byte-v1"
EXPECTED_TOKENIZER_TYPE = "utf8-byte"


class TokenBudgetValidationError(ValueError):
    """Raised when the pretraining budget violates a fail-closed invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TokenBudgetValidationError(message)


def stage(value: dict[str, Any], name: str) -> dict[str, Any]:
    stages = value.get("stages")
    require(isinstance(stages, list), "stages must be a list")
    matches = [row for row in stages if isinstance(row, dict) and row.get("name") == name]
    require(len(matches) == 1, f"expected exactly one stage named {name}")
    return matches[0]


def validate(value: dict[str, Any]) -> None:
    require(value.get("schema") == "12-6.pretraining-token-budget.v1", "unexpected schema")
    require(value.get("semantics_revision") == 2, "unit-corrected semantics revision is required")
    require(value.get("status") == "ACTIVE_FAIL_CLOSED_UNIT_CORRECTED", "budget must be unit-corrected and fail closed")
    require(value.get("repository") == "Oleksii-debug/12-6-ai.", "unexpected repository")
    require(value.get("execution_profile") == "LOCAL_FREE", "execution profile must remain LOCAL_FREE")
    require(value.get("training_executed") is False, "control cannot claim training execution")
    require(value.get("paid_compute_used") is False, "control cannot claim paid compute")
    require(value.get("compute_authorized") is False, "control cannot authorize compute")

    tok = value.get("tokenizer_authority", {})
    require(tok.get("tokenizer_version") == EXPECTED_TOKENIZER, "wrong tokenizer authority")
    require(tok.get("type") == EXPECTED_TOKENIZER_TYPE, "current tokenizer must be utf8-byte")
    require(tok.get("vocab_size") == 256 and tok.get("byte_values") == 256, "byte tokenizer vocabulary drift")
    require(tok.get("model_loss_position_unit") == "UTF8_BYTE_POSITION", "wrong current loss-position unit")

    units = value.get("unit_policy", {})
    require(units.get("external_scaling_reference_unit") == "SOURCE_REPORTED_TOKEN", "wrong external reference unit")
    require(units.get("current_model_training_unit") == "UTF8_BYTE_POSITION", "wrong current training unit")
    require(units.get("cross_tokenizer_numeric_conversion_allowed") is False, "cross-tokenizer numeric conversion must stay forbidden")
    require(units.get("direct_chinchilla_reference_as_byte_budget_allowed") is False, "Chinchilla tokens cannot become a direct byte budget")

    policy = value.get("policy", {})
    for key in (
        "no_replay_for_budget_credit",
        "count_only_training_authorized_loss_positions",
        "tokenizer_identity_must_be_bound",
        "corpus_identity_must_be_bound",
        "checkpoint_integrity_must_be_terminal",
        "quality_claim_requires_held_out_evaluation",
        "larger_scale_requires_measured_scaling_curve",
        "paid_compute_requires_explicit_authorization",
    ):
        require(policy.get(key) is True, f"mandatory policy gate missing: {key}")

    model20 = stage(value, "MODEL-341-20M")
    require(model20.get("parameter_count") == EXPECTED_PARAMS, "wrong MODEL-341 parameter count")
    require(model20.get("tokenizer_type") == EXPECTED_TOKENIZER_TYPE, "20M stage tokenizer drift")
    require(model20.get("current_preregistered_unique_byte_positions") == EXPECTED_PILOT_BYTE_POSITIONS, "wrong 20M engineering pilot byte budget")
    require(model20.get("direct_reference_byte_positions") is None, "must not fabricate a Chinchilla-derived byte target")
    require(model20.get("current_request_fraction_of_external_reference") is None, "must not compare byte positions numerically to external token count")
    require(model20.get("external_reference_unit") == "SOURCE_REPORTED_TOKEN", "external reference unit must remain explicit")
    require(model20.get("external_hypothetical_reference_tokens_at_20_per_parameter") == EXPECTED_PARAMS * 20, "external planning anchor arithmetic drift")
    require(model20.get("current_request_classification") == "ENGINEERING_PILOT_NOT_SCIENCE_COMPLETE_20M_BASELINE", "pilot cannot be promoted to a science-complete baseline")

    for name, params in (("100M", 100_000_000), ("1B", 1_000_000_000)):
        item = stage(value, name)
        require(item.get("parameter_count") == params, f"wrong {name} parameter count")
        require(item.get("external_hypothetical_reference_tokens_at_20_per_parameter") == params * 20, f"wrong {name} external planning anchor")
        require(item.get("external_reference_unit") == "SOURCE_REPORTED_TOKEN", f"wrong {name} external reference unit")
        require(item.get("direct_reference_byte_positions") is None, f"{name} must not have a fabricated byte budget")
        require(item.get("status") == "FUTURE_SCALE_STAGE_BLOCKED", f"{name} must remain blocked")

    decision = value.get("current_decision", {})
    require(decision.get("20m_architecture_ready_for_pilot_after_blockers") is True, "20M mechanics decision drift")
    require(decision.get("20m_engineering_pilot_requested_positions") == EXPECTED_PILOT_BYTE_POSITIONS, "20M decision byte budget drift")
    require(decision.get("20m_quality_baseline_data_budget_ready") is False, "20M quality budget must remain not ready")
    require(decision.get("20m_science_complete_byte_budget") is None, "science-complete byte budget must remain undefined")
    require(decision.get("100m_training_ready") is False, "100M training must remain blocked")
    require(decision.get("1b_training_ready") is False, "1B training must remain blocked")

    calibration = set(value.get("mandatory_calibration_before_science_budget", []))
    mandatory = {
        "MEASURE_BYTES_CHARACTERS_AND_SUBWORD_TOKENS_PER_DOCUMENT_BY_UA_EN_CODE",
        "COMPARE_BYTE_VS_LEARNED_SUBWORD_ON_IDENTICAL_CLEAN_CORPUS_SLICE",
        "REPORT_HELD_OUT_NLL_NORMALIZED_BY_BYTE_AND_BY_CHARACTER",
        "MEASURE_TRAINING_FLOPS_OR_WALLCLOCK_PER_EFFECTIVE_TEXT_UNIT",
        "MEASURE_CONTEXT_SEMANTIC_SPAN_BY_LANGUAGE_AND_DOMAIN",
        "FIT_FLOP_NORMALIZED_LEARNING_CURVE_BEFORE_100M_PARAMETER_PROMOTION",
    }
    require(mandatory.issubset(calibration), "mandatory tokenizer/FLOP calibration gates are missing")

    rules = value.get("promotion_rules", {})
    first = rules.get("first_learned_checkpoint", {})
    science = rules.get("science_complete_size_baseline", {})
    larger = rules.get("larger_model_scale_up", {})
    require(first.get("quality_claim_allowed") is False, "first learned checkpoint cannot imply quality baseline")
    require(science.get("numeric_byte_budget") is None, "science baseline cannot have a byte budget before calibration")
    require(science.get("status") == "UNDEFINED_PENDING_CALIBRATION", "science budget status must remain calibration-blocked")
    require(science.get("quality_claim_allowed") is False, "science-complete quality claim is premature")
    require(larger.get("requires_previous_stage_learned") is True, "larger scale must require a learned previous stage")
    require(larger.get("requires_terminal_checkpoint_integrity") is True, "larger scale must require checkpoint integrity")
    require(larger.get("requires_tokenizer_efficiency_evidence") is True, "larger scale must require tokenizer evidence")
    require(larger.get("requires_flop_normalized_scaling_curve") is True, "larger scale must require FLOP-normalized scaling")

    require(value.get("evidence_identity") == "GIT_COMMIT_AND_BLOB_BOUND", "evidence must be Git-bound")


def main() -> int:
    value = json.loads(CONTROL.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "control root must be a JSON object")
    validate(value)
    model20 = stage(value, "MODEL-341-20M")
    print("PRETRAINING_TOKEN_BUDGET=PASS_UNIT_CORRECTED")
    print("LOSS_POSITION_UNIT=UTF8_BYTE_POSITION")
    print("20M_PARAMS=" + str(model20["parameter_count"]))
    print("20M_PILOT_BYTE_POSITIONS=" + str(model20["current_preregistered_unique_byte_positions"]))
    print("SCIENCE_COMPLETE_BYTE_BUDGET=UNDEFINED_PENDING_CALIBRATION")
    print("100M_READY=false")
    print("1B_READY=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
