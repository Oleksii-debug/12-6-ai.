#!/usr/bin/env python3
"""Fail-closed validator for the S4 ~100M admission contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path("configs/control/s4_100m_admission_v1.json")
TARGET_PARAMETERS = 100_000_000
EXPECTED_SCHEMA = "12-6.s4-100m-admission.v1"
EXPECTED_PREDECESSOR_SHA = "e4ff486fd90802fc123bebf60eed4e59196a98df"
EXPECTED_TOKEN_BUDGET_SCHEMA = "12-6.pretraining-token-budget.v1"
EXPECTED_TOKEN_BUDGET_BLOB = "4b2371fd20c9a0b96bf26902dc2c29d195ea96f4"
BLOCKING_GATE_VALUES = {"BLOCKED", "NO"}


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _require_mapping(value: object, label: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{label} must be a mapping")
        return {}
    return value


def parameter_count(model: dict[str, Any]) -> int:
    """Count parameters for the current bias-free tied decoder contract."""
    fields = (
        "vocab_size",
        "d_model",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "d_ff",
    )
    values: dict[str, int] = {}
    for field in fields:
        value = model.get(field)
        if not _is_positive_int(value):
            raise ValueError(f"model.{field} must be a positive integer")
        values[field] = value

    d_model = values["d_model"]
    n_heads = values["n_heads"]
    n_kv_heads = values["n_kv_heads"]
    head_dim = values["head_dim"]
    if d_model != n_heads * head_dim:
        raise ValueError("d_model must equal n_heads * head_dim")
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    if model.get("tie_word_embeddings") is not True:
        raise ValueError("S4 admission candidates must use tied word embeddings")
    if model.get("activation") != "swiglu":
        raise ValueError("S4 admission candidates must use SwiGLU")
    if model.get("norm_kind") != "rmsnorm":
        raise ValueError("S4 admission candidates must use RMSNorm")
    if model.get("position_embedding") != "rope":
        raise ValueError("S4 admission candidates must use RoPE")

    vocab_size = values["vocab_size"]
    n_layers = values["n_layers"]
    d_ff = values["d_ff"]
    kv_width = n_kv_heads * head_dim

    token_embedding = vocab_size * d_model
    attention_per_layer = 2 * d_model * d_model + 2 * d_model * kv_width
    mlp_per_layer = 3 * d_model * d_ff
    norms_per_layer = 2 * d_model
    final_norm = d_model
    return token_embedding + n_layers * (
        attention_per_layer + mlp_per_layer + norms_per_layer
    ) + final_norm


def _validate_token_budget_authority(
    contract: dict[str, Any], errors: list[str]
) -> dict[str, Any]:
    authority = _require_mapping(
        contract.get("token_budget_authority"), "token_budget_authority", errors
    )
    if authority.get("path") != "configs/control/pretraining_token_budget_v1.json":
        errors.append("token budget authority must point to pretraining_token_budget_v1.json")
    if authority.get("schema") != EXPECTED_TOKEN_BUDGET_SCHEMA:
        errors.append("token budget authority schema mismatch")
    if authority.get("blob_sha") != EXPECTED_TOKEN_BUDGET_BLOB:
        errors.append("token budget authority blob drifted from the reviewed upstream snapshot")
    if authority.get("policy_owner") != "UPSTREAM_TOKEN_BUDGET_CONTROLLER":
        errors.append("S4 admission must not claim ownership of token-budget policy")
    if authority.get("compute_optimal_reference_tokens_per_parameter") != 20:
        errors.append("token budget authority must preserve the reviewed 20 tpp planning reference")
    if authority.get("predecessor_20m_reference_tokens") != 412_268_800:
        errors.append("token budget authority 20M reference mismatch")
    if authority.get("canonical_100m_reference_tokens") != 2_000_000_000:
        errors.append("token budget authority 100M reference mismatch")
    if authority.get("canonical_1b_reference_tokens") != 20_000_000_000:
        errors.append("token budget authority 1B reference mismatch")
    if authority.get("current_20m_request_classification") != (
        "PIPELINE_PILOT_NOT_SCIENCE_COMPLETE_20M_BASELINE"
    ):
        errors.append("20M request must remain classified as a pipeline pilot")
    return authority


def validate_contract(contract: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    computed: dict[str, int] = {}

    if contract.get("schema") != EXPECTED_SCHEMA:
        errors.append(f"schema must be {EXPECTED_SCHEMA}")
    if contract.get("execution_profile") != "LOCAL_FREE":
        errors.append("execution_profile must remain LOCAL_FREE")

    predecessor = _require_mapping(contract.get("predecessor_20m"), "predecessor_20m", errors)
    if predecessor.get("head_sha") != EXPECTED_PREDECESSOR_SHA:
        errors.append("predecessor_20m.head_sha drifted from MODEL-341 authority")
    if predecessor.get("mechanical_qualification") != "PASS":
        errors.append("predecessor mechanical qualification must be PASS")
    if predecessor.get("long_training_performed") is not False:
        errors.append("contract must not claim predecessor long training")

    token_budget_authority = _validate_token_budget_authority(contract, errors)

    selection = _require_mapping(contract.get("selection_rule"), "selection_rule", errors)
    max_deviation = selection.get("maximum_target_deviation_fraction")
    if not isinstance(max_deviation, (int, float)) or isinstance(max_deviation, bool):
        errors.append("selection_rule.maximum_target_deviation_fraction must be numeric")
        max_deviation = 0.0
    if selection.get("selected_candidate") is not None:
        errors.append("v1 research contract must not preselect an S4 candidate")

    candidates = contract.get("research_candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        errors.append("research_candidates must contain at least two controls")
        candidates = []

    seen_ids: set[str] = set()
    for index, raw_candidate in enumerate(candidates):
        candidate = _require_mapping(raw_candidate, f"research_candidates[{index}]", errors)
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            errors.append(f"research_candidates[{index}].candidate_id must be non-empty")
            continue
        if candidate_id in seen_ids:
            errors.append(f"duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)

        model = _require_mapping(candidate.get("model"), f"{candidate_id}.model", errors)
        try:
            actual = parameter_count(model)
        except ValueError as exc:
            errors.append(f"{candidate_id}: {exc}")
            continue
        computed[candidate_id] = actual

        expected = candidate.get("expected_parameters")
        if expected != actual:
            errors.append(
                f"{candidate_id}: expected_parameters={expected!r} but formula gives {actual}"
            )
        deviation = abs(actual - TARGET_PARAMETERS) / TARGET_PARAMETERS
        declared_deviation = candidate.get("target_deviation_fraction")
        if not isinstance(declared_deviation, (int, float)) or isinstance(
            declared_deviation, bool
        ):
            errors.append(f"{candidate_id}: target_deviation_fraction must be numeric")
        elif abs(float(declared_deviation) - deviation) > 1e-12:
            errors.append(f"{candidate_id}: target_deviation_fraction is inconsistent")
        if deviation > float(max_deviation):
            errors.append(
                f"{candidate_id}: target deviation {deviation:.6f} exceeds {max_deviation:.6f}"
            )
        n_heads = model.get("n_heads")
        n_kv_heads = model.get("n_kv_heads")
        if isinstance(n_heads, int) and isinstance(n_kv_heads, int) and n_kv_heads >= n_heads:
            errors.append(f"{candidate_id}: research controls must exercise GQA, not MHA")

    tokenizer_gate = _require_mapping(contract.get("tokenizer_gate"), "tokenizer_gate", errors)
    if tokenizer_gate.get("production_decision") != "BLOCKED_REQUIRES_EVIDENCE":
        errors.append("tokenizer production decision must remain evidence-gated")

    data_gate = _require_mapping(contract.get("data_gate"), "data_gate", errors)
    if data_gate.get("exact_final_corpus_identity") is not None:
        errors.append("v1 snapshot must not fabricate a final corpus identity")
    if data_gate.get("exact_shard_identity") is not None:
        errors.append("v1 snapshot must not fabricate a shard identity")
    if data_gate.get("authorized_unique_no_replay_loss_positions") != 0:
        errors.append("v1 snapshot must preserve zero authorized real loss positions")

    gates = _require_mapping(contract.get("gates"), "gates", errors)
    has_blocking_gate = any(value in BLOCKING_GATE_VALUES for value in gates.values())
    if not has_blocking_gate:
        errors.append("at least one fail-closed gate must remain unresolved in this snapshot")
    if gates.get("material_compute_authorized") != "NO":
        errors.append("material compute must remain unauthorized")

    if contract.get("promotion_allowed") is not False:
        errors.append("promotion_allowed must be false")
    if contract.get("long_training_authorized") is not False:
        errors.append("long_training_authorized must be false")
    if contract.get("paid_compute_used") is not False:
        errors.append("paid_compute_used must be false")

    references = _require_mapping(
        contract.get("training_budget_research_reference"),
        "training_budget_research_reference",
        errors,
    )
    tpp = references.get("reference_only_tokens_per_parameter")
    authority_tpp = token_budget_authority.get("compute_optimal_reference_tokens_per_parameter")
    if tpp != authority_tpp:
        errors.append("local scaling reference must exactly inherit upstream token-budget policy")
    if not isinstance(tpp, (int, float)) or isinstance(tpp, bool) or tpp <= 0:
        errors.append("reference_only_tokens_per_parameter must be positive")
    else:
        expected_reference = int(round(float(tpp) * int(predecessor.get("parameter_count", 0))))
        if references.get("predecessor_20m_reference_tokens") != expected_reference:
            errors.append("predecessor scaling reference token count is inconsistent")
        for candidate_id, actual in computed.items():
            key = (
                "s4_gqa_exactish_reference_tokens"
                if candidate_id == "S4-GQA-EXACTISH-v1"
                else "s4_gqa_aligned_reference_tokens"
            )
            if references.get(key) != int(round(float(tpp) * actual)):
                errors.append(f"{candidate_id}: scaling reference token count is inconsistent")

    terminal = _require_mapping(contract.get("terminal_decision"), "terminal_decision", errors)
    if terminal.get("status") != (
        "BLOCK_S4_LONG_TRAINING_CONTINUE_LOCAL_FREE_RESEARCH_AND_ENGINEERING"
    ):
        errors.append("terminal_decision.status must remain fail-closed")

    return errors, computed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json", action="store_true", help="emit machine-readable result")
    args = parser.parse_args()

    try:
        contract = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result = {"status": "FAIL", "errors": [str(exc)], "computed_parameters": {}}
        print(json.dumps(result, sort_keys=True) if args.json else f"FAIL: {exc}")
        return 1

    if not isinstance(contract, dict):
        result = {
            "status": "FAIL",
            "errors": ["top-level config must be a JSON object"],
            "computed_parameters": {},
        }
        print(json.dumps(result, sort_keys=True) if args.json else "FAIL: invalid top-level object")
        return 1

    errors, computed = validate_contract(contract)
    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "computed_parameters": computed,
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
    elif errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
    else:
        print("PASS")
        for candidate_id, count in sorted(computed.items()):
            print(f"- {candidate_id}: {count:,} parameters")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
