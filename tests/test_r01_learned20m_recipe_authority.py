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
from twelve_six.packing.core import DEFAULT_SEQUENCE_LENGTH, PACKING_CONFIG_HASH

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
            "packing_sha256": PACKING_CONFIG_HASH,
            "sequence_length": DEFAULT_SEQUENCE_LENGTH,
            "tokenizer_identity_sha256": "2" * 64,
        },
        "d05": {
            "authority": authority("9"),
            "status": "PASS",
            "checkpoint_contract_identity_sha256": "a" * 64,
            "fresh_process_resume_equivalence": True,
            "next_exposure_identity_sha256": "4" * 64,
            "train_trace_identity_sha256": "5" * 64,
        },
        "d06": {
            "authority": authority("b"),
            "status": "PASS",
            "evaluation_firewall_identity_sha256": "c" * 64,
            "selection_validation_identity_sha256": "d" * 64,
            "primary_selection_metric": "BITS_PER_BYTE",
            "required_held_out_strata": ["ua", "en", "code"],
            "final_test_sealed": True,
            "final_test_may_influence_selection": False,
        },
    }


def trusted_authorities(data=None):
    data = data or bindings()
    return {
        role: copy.deepcopy(data[role]["authority"])
        for role in ("tokenizer", "d04", "d05", "d06")
    }


def bind(data=None, capacity: int = 15_000_000):
    data = data or bindings(capacity)
    return bind_terminal_authorities(
        load_policy(), data, trusted_authorities=trusted_authorities(data)
    )


def test_checked_in_policy_is_self_hashed_and_valid():
    policy = validate_policy(load_policy())
    body = {k: v for k, v in policy.items() if k != "policy_identity_sha256"}
    assert policy["policy_identity_sha256"] == identity_sha256(body)
    assert policy["recipe"]["learning_rate"] == 0.00022
    assert policy["recipe"]["sequence_length"] == DEFAULT_SEQUENCE_LENGTH
    assert policy["truth_boundary"]["authorized_optimized_targets"] == 0


def test_template_is_blocked_and_cannot_self_authorize():
    template = blocked_template(load_policy())
    assert template["status"] == "BLOCKED_TEMPLATE"
    assert template["trusted_authorities_identity_sha256"] is None
    assert template["training_authorized"] is False
    assert template["compute_authorized"] is False
    assert template["authorized_optimized_targets"] == 0
    assert template["optimizer_updates_executed"] == 0


def test_terminal_binding_qualifies_recipe_but_not_training():
    session = bind()
    assert session["status"] == "QUALIFIED_RECIPE_ONLY"
    assert session["qualified_runtime_unique_loss_positions"] == 15_000_000
    assert len(session["trusted_authorities_identity_sha256"]) == 64
    assert session["training_recipe_status"] == "QUALIFIED"
    assert session["training_authorized"] is False
    assert session["compute_authorized"] is False
    assert session["authorized_optimized_targets"] == 0


def test_runtime_budget_caps_at_twenty_million():
    session = bind(capacity=25_000_000)
    assert session["qualified_runtime_unique_loss_positions"] == 20_000_000


def test_capacity_below_meaningful_floor_fails_closed():
    data = bindings(9_999_999)
    with pytest.raises(RecipeValidationError, match="below LEARN-345 meaningful floor"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )


def test_bool_capacity_is_not_an_integer_alias():
    data = bindings()
    data["d04"]["unique_nonignored_causal_loss_positions"] = True
    with pytest.raises(RecipeValidationError, match="positive integer"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("recipe", "learning_rate"), 0.00016),
        (("recipe", "scheduler"), "cosine"),
        (("recipe", "warmup_steps"), 1),
        (("recipe", "precision"), "bf16"),
        (("recipe", "sequence_length"), 256),
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
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )


def test_nonterminal_or_failed_authority_fails_closed():
    data = bindings()
    expected = trusted_authorities(data)
    data["d04"]["authority"]["terminal"] = False
    with pytest.raises(RecipeValidationError, match="terminal"):
        bind_terminal_authorities(load_policy(), data, trusted_authorities=expected)

    data = bindings()
    expected = trusted_authorities(data)
    data["d04"]["authority"]["workflow_conclusion"] = "failure"
    with pytest.raises(RecipeValidationError, match="success"):
        bind_terminal_authorities(load_policy(), data, trusted_authorities=expected)


def test_self_consistent_packet_authority_substitution_fails_against_trusted_set():
    data = bindings()
    expected = trusted_authorities(data)
    data["d04"]["authority"] = authority("e")
    with pytest.raises(RecipeValidationError, match="trusted role-bound"):
        bind_terminal_authorities(load_policy(), data, trusted_authorities=expected)


def test_trusted_role_set_is_closed_world_and_mandatory():
    data = bindings()
    expected = trusted_authorities(data)
    expected["extra"] = authority("e")
    with pytest.raises(RecipeValidationError, match="trusted_authorities keys mismatch"):
        bind_terminal_authorities(load_policy(), data, trusted_authorities=expected)
    with pytest.raises(RecipeValidationError, match="out-of-packet role map"):
        bind_terminal_authorities(load_policy(), data, trusted_authorities=None)


def test_tokenizer_substitution_or_unresolved_decision_fails_closed():
    data = bindings()
    data["tokenizer"]["decision"] = "UNRESOLVED"
    with pytest.raises(RecipeValidationError, match="tokenizer decision"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )


def test_d04_sequence_and_packing_must_match_canonical_contract():
    data = bindings()
    data["d04"]["sequence_length"] = 256
    with pytest.raises(RecipeValidationError, match="sequence length"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )

    data = bindings()
    data["d04"]["packing_sha256"] = "8" * 64
    with pytest.raises(RecipeValidationError, match="packing identity"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )


def test_d04_tokenizer_identity_must_match_terminal_tokenizer():
    data = bindings()
    data["d04"]["tokenizer_identity_sha256"] = "e" * 64
    with pytest.raises(RecipeValidationError, match="tokenizer identity"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )


def test_d05_resume_equivalence_and_d04_identity_coherence_are_mandatory():
    data = bindings()
    data["d05"]["fresh_process_resume_equivalence"] = False
    with pytest.raises(RecipeValidationError, match="resume equivalence"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )

    data = bindings()
    data["d05"]["next_exposure_identity_sha256"] = "e" * 64
    with pytest.raises(RecipeValidationError, match="next-exposure"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )

    data = bindings()
    data["d05"]["train_trace_identity_sha256"] = "e" * 64
    with pytest.raises(RecipeValidationError, match="train-trace"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )


def test_d06_binds_bpb_ua_en_code_and_final_test_firewall():
    data = bindings()
    data["d06"]["primary_selection_metric"] = "LOSS"
    with pytest.raises(RecipeValidationError, match="primary selection metric"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )

    data = bindings()
    data["d06"]["required_held_out_strata"] = ["ua", "en"]
    with pytest.raises(RecipeValidationError, match="ua/en/code"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )

    data = bindings()
    data["d06"]["final_test_sealed"] = False
    with pytest.raises(RecipeValidationError, match="final test"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )

    data = bindings()
    data["d06"]["final_test_may_influence_selection"] = True
    with pytest.raises(RecipeValidationError, match="may not influence"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )


def test_exact_next_exposure_and_trace_are_identity_bound():
    first_data = bindings()
    first = bind(first_data)
    changed = bindings()
    changed["d04"]["next_exposure_identity_sha256"] = "e" * 64
    changed["d05"]["next_exposure_identity_sha256"] = "e" * 64
    second = bind(changed)
    assert first["bindings_identity_sha256"] != second["bindings_identity_sha256"]
    assert first["session_identity_sha256"] != second["session_identity_sha256"]


def test_unknown_binding_field_fails_closed():
    data = bindings()
    data["d04"]["extra"] = "not allowed"
    with pytest.raises(RecipeValidationError, match="keys mismatch"):
        bind_terminal_authorities(
            load_policy(), data, trusted_authorities=trusted_authorities(data)
        )


def test_readiness_fragment_matches_existing_training_recipe_shape():
    data = bindings(12_345_678)
    session = bind(data)
    fragment = readiness_fragment(session, authority("e"))
    assert fragment["status"] == "QUALIFIED"
    assert fragment["seed_count"] == 1
    assert fragment["requested_unique_loss_positions"] == 12_345_678
    assert fragment["requested_total_training_exposures"] == 12_345_678
    assert fragment["max_exposures_per_unique_position"] == 1
    assert len(fragment["config_sha256"]) == 64
    assert len(fragment["stopping_policy_sha256"]) == 64


def test_readiness_fragment_rejects_session_self_authorization():
    session = bind()
    session["training_authorized"] = True
    with pytest.raises(RecipeValidationError, match="training_authorized"):
        readiness_fragment(session, authority("e"))
