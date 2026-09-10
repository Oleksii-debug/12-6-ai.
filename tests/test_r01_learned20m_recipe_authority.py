from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.learned20m_recipe import (
    MODEL341,
    RecipeValidationError,
    bind_terminal_authorities,
    blocked_template,
    identity_sha256,
    readiness_fragment,
    validate_policy,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs/research/r01_learned20m_recipe_authority_v1.json"


def load_policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def authority(seed: str = "a"):
    return {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": seed * 40,
        "evidence_sha256": seed * 64,
        "terminal": True,
        "workflow_run_id": 123456,
        "workflow_conclusion": "success",
    }


def bindings(capacity: int = 15_000_000):
    return {
        "code": {"git_sha": "c" * 40},
        "model": copy.deepcopy(MODEL341),
        "tokenizer": {
            "authority": authority("1"),
            "identity_sha256": "2" * 64,
            "decision": "BYTE_BASELINE_RETAINED",
        },
        "d04": {
            "authority": authority("3"),
            "unique_nonignored_causal_loss_positions": capacity,
            "next_exposure_identity_sha256": "4" * 64,
            "train_trace_identity_sha256": "5" * 64,
            "corpus_manifest_sha256": "6" * 64,
            "split_sha256": "7" * 64,
            "packing_sha256": "8" * 64,
        },
        "d05": {
            "authority": authority("9"),
            "status": "PASS",
            "checkpoint_contract_identity_sha256": "a" * 64,
            "fresh_process_resume_equivalence": True,
        },
        "d06": {
            "authority": authority("b"),
            "status": "PASS",
            "evaluation_firewall_identity_sha256": "c" * 64,
            "selection_validation_identity_sha256": "d" * 64,
            "final_test_sealed": True,
        },
    }


def test_checked_in_policy_is_self_hashed_and_valid():
    policy = validate_policy(load_policy())
    body = {k: v for k, v in policy.items() if k != "policy_identity_sha256"}
    assert policy["policy_identity_sha256"] == identity_sha256(body)
    assert policy["recipe"]["learning_rate"] == 0.00022
    assert policy["truth_boundary"]["authorized_optimized_targets"] == 0


def test_template_is_blocked_and_cannot_self_authorize():
    template = blocked_template(load_policy())
    assert template["status"] == "BLOCKED_TEMPLATE"
    assert template["training_authorized"] is False
    assert template["compute_authorized"] is False
    assert template["authorized_optimized_targets"] == 0
    assert template["optimizer_updates_executed"] == 0


def test_terminal_binding_qualifies_recipe_but_not_training():
    session = bind_terminal_authorities(load_policy(), bindings())
    assert session["status"] == "QUALIFIED_RECIPE_ONLY"
    assert session["qualified_runtime_unique_loss_positions"] == 15_000_000
    assert session["training_recipe_status"] == "QUALIFIED"
    assert session["training_authorized"] is False
    assert session["compute_authorized"] is False
    assert session["authorized_optimized_targets"] == 0


def test_runtime_budget_caps_at_twenty_million():
    session = bind_terminal_authorities(load_policy(), bindings(25_000_000))
    assert session["qualified_runtime_unique_loss_positions"] == 20_000_000


def test_capacity_below_meaningful_floor_fails_closed():
    with pytest.raises(RecipeValidationError, match="below LEARN-345 meaningful floor"):
        bind_terminal_authorities(load_policy(), bindings(9_999_999))


def test_bool_capacity_is_not_an_integer_alias():
    data = bindings()
    data["d04"]["unique_nonignored_causal_loss_positions"] = True
    with pytest.raises(RecipeValidationError, match="positive integer"):
        bind_terminal_authorities(load_policy(), data)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("recipe", "learning_rate"), 0.00016),
        (("recipe", "scheduler"), "cosine"),
        (("recipe", "warmup_steps"), 1),
        (("recipe", "precision"), "bf16"),
        (("recipe", "sequence_length"), 1024),
        (("recipe", "gradient_accumulation_steps"), 2),
        (("truth_boundary", "training_authorized"), True),
        (("truth_boundary", "compute_authorized"), True),
        (("truth_boundary", "authorized_optimized_targets"), 1),
    ],
)
def test_policy_drift_fails_closed(path, value):
    policy = load_policy()
    policy[path[0]][path[1]] = value
    body = {k: v for k, v in policy.items() if k != "policy_identity_sha256"}
    policy["policy_identity_sha256"] = identity_sha256(body)
    with pytest.raises(RecipeValidationError):
        validate_policy(policy)


def test_unknown_policy_field_fails_closed():
    policy = load_policy()
    policy["surprise"] = "unsafe"
    with pytest.raises(RecipeValidationError, match="keys mismatch"):
        validate_policy(policy)


def test_nonfinite_numeric_rejected_by_canonical_identity():
    with pytest.raises(RecipeValidationError, match="non-finite"):
        identity_sha256({"x": float("nan")})


def test_substituted_model_fails_closed():
    data = bindings()
    data["model"]["parameter_count"] += 1
    with pytest.raises(RecipeValidationError, match="MODEL-341"):
        bind_terminal_authorities(load_policy(), data)


def test_nonterminal_or_failed_authority_fails_closed():
    data = bindings()
    data["d04"]["authority"]["terminal"] = False
    with pytest.raises(RecipeValidationError, match="terminal"):
        bind_terminal_authorities(load_policy(), data)

    data = bindings()
    data["d04"]["authority"]["workflow_conclusion"] = "failure"
    with pytest.raises(RecipeValidationError, match="success"):
        bind_terminal_authorities(load_policy(), data)


def test_tokenizer_substitution_or_unresolved_decision_fails_closed():
    data = bindings()
    data["tokenizer"]["decision"] = "UNRESOLVED"
    with pytest.raises(RecipeValidationError, match="tokenizer decision"):
        bind_terminal_authorities(load_policy(), data)


def test_d05_resume_equivalence_is_mandatory():
    data = bindings()
    data["d05"]["fresh_process_resume_equivalence"] = False
    with pytest.raises(RecipeValidationError, match="resume equivalence"):
        bind_terminal_authorities(load_policy(), data)


def test_d06_final_test_must_remain_sealed():
    data = bindings()
    data["d06"]["final_test_sealed"] = False
    with pytest.raises(RecipeValidationError, match="final test"):
        bind_terminal_authorities(load_policy(), data)


def test_exact_next_exposure_and_trace_are_identity_bound():
    first = bind_terminal_authorities(load_policy(), bindings())
    changed = bindings()
    changed["d04"]["next_exposure_identity_sha256"] = "e" * 64
    second = bind_terminal_authorities(load_policy(), changed)
    assert first["bindings_identity_sha256"] != second["bindings_identity_sha256"]
    assert first["session_identity_sha256"] != second["session_identity_sha256"]


def test_unknown_binding_field_fails_closed():
    data = bindings()
    data["d04"]["extra"] = "not allowed"
    with pytest.raises(RecipeValidationError, match="keys mismatch"):
        bind_terminal_authorities(load_policy(), data)


def test_readiness_fragment_matches_existing_training_recipe_shape():
    session = bind_terminal_authorities(load_policy(), bindings(12_345_678))
    fragment = readiness_fragment(session, authority("e"))
    assert fragment["status"] == "QUALIFIED"
    assert fragment["seed_count"] == 1
    assert fragment["requested_unique_loss_positions"] == 12_345_678
    assert fragment["requested_total_training_exposures"] == 12_345_678
    assert fragment["max_exposures_per_unique_position"] == 1
    assert len(fragment["config_sha256"]) == 64
    assert len(fragment["stopping_policy_sha256"]) == 64


def test_readiness_fragment_rejects_session_self_authorization():
    session = bind_terminal_authorities(load_policy(), bindings())
    session["training_authorized"] = True
    with pytest.raises(RecipeValidationError, match="training_authorized"):
        readiness_fragment(session, authority("e"))
