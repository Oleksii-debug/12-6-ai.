#!/usr/bin/env python3
"""Fail-closed, tokenizer-aware validator for the 20M training ladder."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "control" / "20m_training_ladder_v1.json"
EXPECTED_SCHEMA = "12-6.20m-training-ladder.v1"
EXPECTED_MODEL_PARAMETERS = 20_613_440
EXPECTED_MODEL_HEAD = "e4ff486fd90802fc123bebf60eed4e59196a98df"
EXPECTED_MODEL_SPEC = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
EXPECTED_PILOT_BYTE_POSITIONS = 20_000_000


class LadderValidationError(ValueError):
    """Raised when the training ladder violates a fail-closed invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LadderValidationError(message)


def validate(data: dict[str, Any]) -> None:
    require(data.get("schema") == EXPECTED_SCHEMA, "unexpected schema")
    require(data.get("semantics_revision") == 2, "unit-corrected semantics revision is required")
    require(data.get("status") == "ACTIVE_FAIL_CLOSED_UNIT_CORRECTED", "ladder must be unit-corrected and fail closed")

    authority = data.get("authority", {})
    require(authority.get("parameter_count") == EXPECTED_MODEL_PARAMETERS, "wrong 20M parameter authority")
    require(authority.get("model_head_sha") == EXPECTED_MODEL_HEAD, "wrong MODEL-341 head")
    require(authority.get("model_spec_sha256") == EXPECTED_MODEL_SPEC, "wrong MODEL-341 ModelSpec")
    require(authority.get("random_init_only") is True, "20M authority must remain random-init only")
    require(authority.get("tokenizer_version") == "s0-byte-v1", "wrong tokenizer version")
    require(authority.get("tokenizer_type") == "utf8-byte", "wrong tokenizer type")
    require(authority.get("vocab_size") == 256, "wrong byte vocabulary size")
    require(authority.get("loss_position_unit") == "UTF8_BYTE_POSITION", "wrong loss-position unit")

    ladder = data.get("training_ladder", {})
    pilot = ladder.get("engineering_pilot", {})
    require(pilot.get("requested_unique_optimized_byte_positions") == EXPECTED_PILOT_BYTE_POSITIONS, "wrong 20M engineering pilot byte budget")
    require(pilot.get("runnable_now") is False, "pilot must remain blocked while exact data authority is zero")
    require(pilot.get("claim_ceiling") == "PIPELINE_AND_LEARNING_SIGNAL_ONLY_NO_GENERAL_BASE_QUALITY_CLAIM", "pilot cannot imply a general Base quality claim")

    science = ladder.get("science_complete_20m_budget", {})
    require(science.get("unit") == "UTF8_BYTE_POSITION", "science budget unit must be explicit")
    require(science.get("value") is None, "science-complete byte budget must remain undefined before calibration")
    require(science.get("status") == "UNDEFINED_PENDING_TOKENIZER_AND_FLOP_CALIBRATION", "science budget must remain calibration-blocked")

    anchor = ladder.get("external_chinchilla_style_anchor", {})
    require(anchor.get("rounded_source_reported_tokens_per_parameter") == 20, "unexpected external planning ratio")
    require(anchor.get("hypothetical_source_reported_tokens_for_same_parameter_count") == EXPECTED_MODEL_PARAMETERS * 20, "external anchor arithmetic drift")
    require(anchor.get("direct_conversion_to_byte_positions") is False, "external tokens cannot be directly converted to byte positions")
    require(anchor.get("status") == "REFERENCE_ONLY_NOT_TRAINING_TARGET_NOT_COMPUTE_AUTHORIZATION", "external anchor cannot authorize a training target")

    sequence = ladder.get("decision_sequence")
    require(isinstance(sequence, list) and len(sequence) >= 6, "measured decision sequence is incomplete")
    require(sequence[0] == "L0_20M_BYTE_POSITION_ENGINEERING_PILOT", "first stage must remain the byte-position engineering pilot")
    require("L2_FLOP_NORMALIZED_BYTE_VS_SUBWORD_ABLATION" in sequence, "byte-vs-subword FLOP ablation is mandatory")
    require(sequence[-1] == "L5_CONSIDER_100M_PARAMETER_PROMOTION_ONLY_FROM_MEASURED_EVIDENCE", "100M promotion must remain evidence-gated")

    future = data.get("future_scale_reference")
    require(isinstance(future, list) and len(future) == 2, "100M and 1B future anchors are required")
    expected = {100_000_000: 2_000_000_000, 1_000_000_000: 20_000_000_000}
    observed: dict[int, int] = {}
    for item in future:
        params = item.get("nominal_parameters")
        tokens = item.get("external_hypothetical_source_reported_tokens_at_20_per_parameter")
        require(type(params) is int and type(tokens) is int, "future anchors must use integer parameter/token counts")
        require(tokens == params * 20, "future external anchor arithmetic drift")
        require(item.get("direct_reference_byte_positions") is None, "future scale must not fabricate byte budgets")
        require(item.get("status") == "EXTERNAL_ANCHOR_ONLY_FUTURE_STAGE_BLOCKED", "future scale must remain blocked")
        observed[params] = tokens
    require(observed == expected, "future parameter anchors must bind 100M and 1B")

    gate = data.get("current_data_gate", {})
    require(gate.get("corpus_identity") is None, "must not fabricate a corpus identity")
    require(gate.get("shard_identity") is None, "must not fabricate a shard identity")
    require(gate.get("authorized_unique_optimized_byte_positions") == 0, "authorized unique byte positions must stay zero until exact data authority exists")
    require(gate.get("long_training_runnable") is False, "long training must remain blocked")

    requirements = set(data.get("promotion_requirements", []))
    mandatory = {
        "TERMINAL_EXACT_CORPUS_AND_SHARD_IDENTITIES",
        "TOKENIZER_IDENTITY_LOCKED_TO_CHECKPOINT_LINEAGE",
        "D05_CORRUPTION_MATRIX_PASS_BEFORE_TARGET_MUTATION",
        "CHECKPOINT_SAVE_LOAD_RESUME_AND_RNG_CONTINUATION_REQUALIFIED_ON_MODEL341",
        "TOKENIZER_EFFICIENCY_CALIBRATION_BY_UA_EN_CODE",
        "FLOP_NORMALIZED_BYTE_VS_SUBWORD_SCALING_CURVE",
        "CONTEXT_SEMANTIC_SPAN_CALIBRATION_BY_LANGUAGE",
        "EXPLICIT_COMPUTE_AUTHORIZATION_FOR_ANY_MATERIALLY_PAID_LONG_RUN",
    }
    require(mandatory.issubset(requirements), "mandatory promotion gates are missing")

    compute = data.get("compute_boundary", {})
    require(compute.get("execution_profile_now") == "LOCAL_FREE", "current execution profile must be LOCAL_FREE")
    require(compute.get("material_paid_compute_authorized") is False, "paid compute must remain unauthorized")
    require(compute.get("long_training_authorized") is False, "long training must remain unauthorized")
    require(compute.get("training_executed_by_this_change") is False, "control change cannot claim training execution")


def load_and_validate(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "config root must be an object")
    validate(data)
    return data


def main() -> int:
    data = load_and_validate()
    pilot = data["training_ladder"]["engineering_pilot"]["requested_unique_optimized_byte_positions"]
    print(f"PASS 20M ladder: loss_unit=UTF8_BYTE_POSITION pilot={pilot} science_byte_budget=undefined 100m_ready=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
