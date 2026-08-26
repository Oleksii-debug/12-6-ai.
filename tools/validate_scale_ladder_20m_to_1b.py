#!/usr/bin/env python3
"""Validate the additive 20M -> 1B engineering scale ladder without allocating models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/control/scale_ladder_20m_to_1b_v1.json"


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _count_parameters(model: dict[str, Any]) -> int:
    """Exact algebra for the current tied, bias-free decoder ModelSpec family."""

    if model.get("attention_bias") is not False:
        raise ValueError("scale-ladder algebra requires attention_bias=false")
    if model.get("mlp_bias") is not False:
        raise ValueError("scale-ladder algebra requires mlp_bias=false")
    if model.get("lm_head_bias") is not False:
        raise ValueError("scale-ladder algebra requires lm_head_bias=false")
    if model.get("tie_word_embeddings") is not True:
        raise ValueError("scale-ladder algebra requires tied word embeddings")
    if model.get("final_norm") is not True:
        raise ValueError("scale-ladder algebra requires final_norm=true")
    if model.get("activation") != "swiglu":
        raise ValueError("scale-ladder algebra requires SwiGLU")
    if model.get("norm_kind") != "rmsnorm" or model.get("norm_placement") != "pre":
        raise ValueError("scale-ladder algebra requires pre-RMSNorm")

    vocab = int(model["vocab_size"])
    d_model = int(model["d_model"])
    layers = int(model["n_layers"])
    n_heads = int(model["n_heads"])
    n_kv_heads = int(model["n_kv_heads"])
    head_dim = int(model["head_dim"])
    d_ff = int(model["d_ff"])

    q_width = n_heads * head_dim
    kv_width = n_kv_heads * head_dim
    if q_width != d_model:
        raise ValueError("current scale ladder requires n_heads * head_dim == d_model")
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_kv_heads must divide n_heads")

    embedding = vocab * d_model
    attention = d_model * q_width + 2 * d_model * kv_width + q_width * d_model
    swiglu = 3 * d_model * d_ff
    two_layer_norms = 2 * d_model
    final_norm = d_model
    return embedding + layers * (attention + swiglu + two_layer_norms) + final_norm


def validate(manifest_path: Path = MANIFEST) -> dict[str, Any]:
    manifest = _load(manifest_path)
    if manifest.get("schema_version") != "12-6.scale-ladder-20m-to-1b.v1":
        raise ValueError("unexpected scale ladder schema")
    if manifest.get("local_free_only") is not True:
        raise ValueError("scale ladder must remain LOCAL_FREE only")
    if manifest.get("paid_compute_used") is not False:
        raise ValueError("paid compute must remain false in this engineering contract")
    if manifest.get("training_executed") is not False:
        raise ValueError("validator contract cannot claim model training")

    global_policy = manifest.get("global_policy")
    if not isinstance(global_policy, dict):
        raise TypeError("global_policy must be an object")
    required_true = (
        "candidate_geometry_is_not_learned_evidence",
        "candidate_geometry_is_not_compute_authorization",
        "candidate_geometry_is_not_stage_promotion",
        "material_training_requires_explicit_compute_authorization",
        "production_tokenizer_decision_required_before_scale_promotion",
        "held_out_evaluation_required_before_scale_promotion",
        "checkpoint_resume_evidence_required_before_scale_promotion",
        "preceding_learned_stage_pass_required",
    )
    for key in required_true:
        if global_policy.get(key) is not True:
            raise ValueError(f"global fail-closed policy missing: {key}")

    stages = manifest.get("stages")
    if not isinstance(stages, list) or len(stages) != 4:
        raise ValueError("scale ladder must bind exactly 20M, 100M, 400M, and 1B")

    reports: list[dict[str, Any]] = []
    previous_target = 0
    for stage in stages:
        if not isinstance(stage, dict):
            raise TypeError("stage entry must be an object")
        target = int(stage["target_parameters"])
        expected = int(stage["expected_parameters"])
        tolerance = float(stage["max_relative_target_error"])
        if target <= previous_target:
            raise ValueError("scale targets must increase strictly")
        previous_target = target

        if stage.get("training_authorized") is not False:
            raise ValueError(f"{stage['stage_id']} must not authorize training")
        if stage.get("promotion_allowed") is not False:
            raise ValueError(f"{stage['stage_id']} must not allow promotion")
        if stage.get("requires_preceding_stage_pass") is not True:
            raise ValueError(f"{stage['stage_id']} must require preceding-stage evidence")
        gates = stage.get("required_training_gates")
        if not isinstance(gates, list) or "MATERIAL_COMPUTE_AUTHORIZATION" not in gates:
            raise ValueError(f"{stage['stage_id']} is missing the material compute gate")

        config_path = ROOT / str(stage["config_path"])
        config = _load(config_path)
        if int(config["target_parameters"]) != target:
            raise ValueError(f"{stage['stage_id']} target does not match its config")
        if int(config["expected_parameters"]) != expected:
            raise ValueError(f"{stage['stage_id']} expected count does not match its config")
        if config.get("canonical_base") != "random_init":
            raise ValueError(f"{stage['stage_id']} must remain scratch random-init")
        if config.get("expected_model_identity_sha256") != stage.get("expected_model_identity_sha256"):
            raise ValueError(f"{stage['stage_id']} ModelSpec identity binding mismatch")

        model = config.get("model")
        if not isinstance(model, dict):
            raise TypeError(f"{stage['stage_id']} config is missing model")
        attention = stage.get("attention")
        if not isinstance(attention, dict):
            raise TypeError(f"{stage['stage_id']} attention contract must be an object")
        for key in ("n_heads", "n_kv_heads", "head_dim"):
            if int(model[key]) != int(attention[key]):
                raise ValueError(f"{stage['stage_id']} attention mismatch for {key}")
        expected_kind = "MHA" if int(model["n_heads"]) == int(model["n_kv_heads"]) else "GQA"
        if attention.get("kind") != expected_kind:
            raise ValueError(f"{stage['stage_id']} attention kind mismatch")
        if int(model["vocab_size"]) != int(stage["tokenizer_vocab_size"]):
            raise ValueError(f"{stage['stage_id']} tokenizer vocabulary mismatch")
        if int(model["max_seq_len"]) != int(stage["context_tokens"]):
            raise ValueError(f"{stage['stage_id']} context mismatch")

        algebra_count = _count_parameters(model)
        if algebra_count != expected:
            raise ValueError(
                f"{stage['stage_id']} parameter algebra mismatch: expected {expected}, got {algebra_count}"
            )
        relative_error = abs(expected - target) / target
        if relative_error > tolerance:
            raise ValueError(
                f"{stage['stage_id']} target error {relative_error:.6f} exceeds {tolerance:.6f}"
            )

        top_level_promotion = config.get("promotion_allowed")
        if top_level_promotion is True:
            raise ValueError(f"{stage['stage_id']} source config unexpectedly permits promotion")
        nested_scale05 = config.get("scale05_execution")
        if isinstance(nested_scale05, dict):
            if nested_scale05.get("promotion_allowed") is True:
                raise ValueError("S5 source config unexpectedly permits promotion")
            if nested_scale05.get("compute_authorized") is True:
                raise ValueError("S5 source config unexpectedly authorizes compute")

        if target >= 400_000_000 and expected_kind == "GQA":
            if not any(str(gate).startswith("NATIVE_GQA") for gate in gates):
                raise ValueError(f"{stage['stage_id']} requires an explicit native-GQA runtime gate")

        reports.append(
            {
                "stage_id": stage["stage_id"],
                "target_parameters": target,
                "exact_parameters": algebra_count,
                "relative_target_error": (expected - target) / target,
                "attention_kind": expected_kind,
                "context_tokens": int(model["max_seq_len"]),
                "training_authorized": False,
                "promotion_allowed": False,
            }
        )

    transitions = manifest.get("cross_stage_gates")
    if not isinstance(transitions, list) or len(transitions) != len(stages) - 1:
        raise ValueError("cross_stage_gates must bind every adjacent stage")
    for index, transition in enumerate(transitions):
        if transition.get("from") != stages[index]["stage_id"]:
            raise ValueError("cross-stage gate source ordering mismatch")
        if transition.get("to") != stages[index + 1]["stage_id"]:
            raise ValueError("cross-stage gate target ordering mismatch")
        if not str(transition.get("decision", "")).startswith("BLOCK_UNTIL_"):
            raise ValueError("every cross-stage transition must fail closed")

    decision = manifest.get("current_decision")
    if decision != "BLOCK_MATERIAL_SCALE_TRAINING_CONTINUE_LOCAL_FREE_ENGINEERING_AND_DATA_READINESS":
        raise ValueError("unexpected current scale-ladder decision")

    return {
        "schema_version": manifest["schema_version"],
        "verdict": "PASS_CONTRACT_VALID",
        "current_decision": decision,
        "stages": reports,
    }


def main() -> int:
    print(json.dumps(validate(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
