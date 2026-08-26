from __future__ import annotations

import copy

from twelve_six.environment_parity import compare_traces, decision_policy
from twelve_six.milestone150_env160_entry import json_normalize


def _trace(*, locked: bool = True) -> dict:
    fp = {
        "exact_locked_runtime": locked,
        "python": {"version": "3.11.16" if locked else "3.13.5"},
        "torch": {"version": "2.13.0" if locked else "2.10.0+cpu"},
        "platform": {"release": "x"},
    }
    tensor = {"sha256": "a" * 64, "values": [1.0, 2.0]}
    state = {"state_sha256": "b" * 64, "tensors": {"w": tensor}}
    gradients = {"gradient_sha256": "c" * 64, "tensors": {"w": tensor}}
    checkpoint = {
        "checkpoint_id": "d" * 64,
        "identity": {"step": 1, "tokens_seen": 13},
    }
    return {
        "source_sha": "e" * 40,
        "environment_fingerprint": fp,
        "model": {
            "model_spec_sha256": "f" * 64,
            "parameter_count": 10,
            "init_spec_sha256": "1" * 64,
        },
        "optimizer": {"name": "AdamW", "config": {"betas": [0.9, 0.95]}},
        "inputs": {"train": [1]},
        "token_counters": {"optimizer_step": 3, "tokens_seen": 39},
        "checkpoints": {
            "step_1": checkpoint,
            "step_3": {
                "checkpoint_id": "2" * 64,
                "identity": {"step": 3, "tokens_seen": 39},
            },
        },
        "heldout_evaluation": {"loss": 2.0, "non_mutation_passed": True},
        "initial": {
            "weights": state,
            "logits": tensor,
            "loss": 3.0,
            "gradients": gradients,
        },
        "updates": {
            "state_after_step_1": state,
            "state_after_step_3": state,
            "step_metrics": [{"loss": 3.0, "tokens": 13}],
        },
    }


def test_json_normalize_persisted_tuple_semantics() -> None:
    assert json_normalize({"betas": (0.9, 0.95)}) == {"betas": [0.9, 0.95]}


def test_identical_trace_is_bitwise_pass() -> None:
    report = compare_traces(_trace(), _trace())
    assert report["classification"] == "PASS_BITWISE"
    assert report["scientific_authority"] is True


def test_cross_version_small_float_difference_is_debug_only_tolerance_pass() -> None:
    canonical = _trace(locked=True)
    candidate = _trace(locked=False)
    candidate["initial"]["logits"]["values"][0] += 5e-7
    report = compare_traces(canonical, candidate)
    assert report["classification"] == "PASS_NUMERIC_TOLERANCE"
    assert report["scientific_authority"] is False


def test_semantic_drift_fails_closed() -> None:
    canonical = _trace()
    candidate = copy.deepcopy(canonical)
    candidate["token_counters"]["tokens_seen"] += 1
    report = compare_traces(canonical, candidate)
    assert report["classification"] == "SEMANTIC_DRIFT"


def test_numeric_drift_requires_exact_head() -> None:
    canonical = _trace()
    candidate = copy.deepcopy(canonical)
    candidate["initial"]["logits"]["values"][0] += 1e-2
    report = compare_traces(canonical, candidate)
    assert report["classification"] == "NUMERIC_DRIFT_REQUIRES_EXACT_HEAD"


def test_policy_requires_exact_head_for_scientific_decisions() -> None:
    policy = decision_policy()
    joined = " ".join(policy["exact_head_locked_rerun_mandatory_for"])
    assert "cross-scale" in joined
    assert policy["source_equivalent_never_upgrades_itself_to_authority"] is True
