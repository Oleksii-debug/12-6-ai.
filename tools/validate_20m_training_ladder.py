#!/usr/bin/env python3
"""Fail-closed validator for the 20M training/scaling ladder."""

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


class LadderValidationError(ValueError):
    """Raised when the training ladder violates a fail-closed invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LadderValidationError(message)


def validate(data: dict[str, Any]) -> None:
    _require(data.get("schema") == EXPECTED_SCHEMA, "unexpected schema")
    _require(data.get("status") == "ACTIVE_FAIL_CLOSED", "ladder must be fail closed")

    authority = data.get("authority", {})
    params = authority.get("parameter_count")
    _require(type(params) is int and params == EXPECTED_MODEL_PARAMETERS, "wrong 20M parameter authority")
    _require(authority.get("model_head_sha") == EXPECTED_MODEL_HEAD, "wrong MODEL-341 head")
    _require(authority.get("model_spec_sha256") == EXPECTED_MODEL_SPEC, "wrong MODEL-341 ModelSpec")
    _require(authority.get("random_init_only") is True, "20M authority must remain random-init only")

    ladder = data.get("training_ladder", {})
    pilot = ladder.get("engineering_pilot", {})
    _require(
        pilot.get("requested_unique_optimized_targets") == 20_000_000,
        "engineering pilot must bind the preregistered 20M optimized-target scale",
    )
    _require(pilot.get("runnable_now") is False, "pilot must remain blocked while corpus authority is zero")
    _require(
        pilot.get("claim_ceiling") == "PIPELINE_AND_LEARNING_SIGNAL_ONLY_NO_GENERAL_BASE_QUALITY_CLAIM",
        "pilot must not be relabeled as a general Base quality claim",
    )

    scientific = ladder.get("scientific_reference", {})
    tpp = scientific.get("tokens_per_parameter")
    reference_targets = scientific.get("reference_unique_optimized_targets")
    _require(type(tpp) is int and tpp == 20, "planning reference must remain 20 tokens per parameter")
    _require(
        reference_targets == params * tpp,
        "scientific reference target must equal exact parameter_count * tokens_per_parameter",
    )
    _require(
        scientific.get("status") == "REFERENCE_ONLY_NOT_COMPUTE_AUTHORIZATION",
        "scientific reference must never imply compute authorization",
    )
    _require(bool(scientific.get("extrapolation_warning")), "small-model extrapolation warning is required")

    milestones = ladder.get("milestones")
    _require(isinstance(milestones, list) and len(milestones) >= 4, "at least four cumulative milestones are required")
    targets = [m.get("cumulative_unique_optimized_targets") for m in milestones]
    _require(all(type(v) is int and v > 0 for v in targets), "milestone targets must be positive integers")
    _require(targets == sorted(targets) and len(targets) == len(set(targets)), "milestones must be strictly increasing")
    _require(targets[0] == pilot["requested_unique_optimized_targets"], "first milestone must be the engineering pilot")
    _require(targets[-1] == reference_targets, "last milestone must reach the scientific reference")

    future = data.get("future_scale_reference")
    _require(isinstance(future, list) and len(future) == 2, "100M and 1B planning references are required")
    expected_future = {100_000_000: 2_000_000_000, 1_000_000_000: 20_000_000_000}
    observed_future: dict[int, int] = {}
    for item in future:
        p = item.get("nominal_parameters")
        tokens = item.get("reference_tokens_at_20_per_parameter")
        _require(type(p) is int and type(tokens) is int, "future scale references must be integers")
        _require(tokens == p * tpp, "future scale reference must preserve the planning ratio")
        _require(item.get("status") == "PLANNING_REFERENCE_ONLY", "future scale entries are not run authorization")
        observed_future[p] = tokens
    _require(observed_future == expected_future, "future scale references must bind nominal 100M and 1B stages")

    gate = data.get("current_data_gate", {})
    _require(gate.get("corpus_identity") is None, "v1 must not fabricate a corpus identity")
    _require(gate.get("shard_identity") is None, "v1 must not fabricate a shard identity")
    _require(gate.get("authorized_unique_optimized_targets") == 0, "current authorized unique target count must stay zero")
    _require(gate.get("long_training_runnable") is False, "long training must remain blocked")

    requirements = set(data.get("promotion_requirements", []))
    mandatory = {
        "TERMINAL_EXACT_CORPUS_AND_SHARD_IDENTITIES",
        "TOKENIZER_IDENTITY_LOCKED_TO_CHECKPOINT_LINEAGE",
        "D05_CORRUPTION_MATRIX_PASS_BEFORE_TARGET_MUTATION",
        "CHECKPOINT_SAVE_LOAD_RESUME_AND_RNG_CONTINUATION_REQUALIFIED_ON_MODEL341",
        "EXPLICIT_COMPUTE_AUTHORIZATION_FOR_ANY_MATERIALLY_PAID_LONG_RUN",
    }
    _require(mandatory.issubset(requirements), "mandatory promotion gates are missing")

    compute = data.get("compute_boundary", {})
    _require(compute.get("execution_profile_now") == "LOCAL_FREE", "current execution profile must be LOCAL_FREE")
    _require(compute.get("material_paid_compute_authorized") is False, "paid compute must not be authorized here")
    _require(compute.get("long_training_authorized") is False, "long training must not be authorized here")
    _require(compute.get("training_executed_by_this_change") is False, "this control-plane change must not claim training")


def load_and_validate(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "config root must be an object")
    validate(data)
    return data


def main() -> int:
    data = load_and_validate()
    params = data["authority"]["parameter_count"]
    reference = data["training_ladder"]["scientific_reference"]["reference_unique_optimized_targets"]
    print(f"PASS 20M training ladder: parameters={params} reference_targets={reference} long_training_authorized=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
