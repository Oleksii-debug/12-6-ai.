from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.learned20m_recipe import (
    MODEL341,
    RecipeValidationError,
    bind_terminal_authorities,
    identity_sha256,
    validate_policy,
)
from twelve_six.packing.core import DEFAULT_SEQUENCE_LENGTH, PACKING_CONFIG_HASH

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs/research/r01_learned20m_recipe_authority_v1.json"


def load_policy():
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def authority(seed: str):
    return {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": seed * 40,
        "evidence_sha256": seed * 64,
        "terminal": True,
        "workflow_run_id": 123456,
        "workflow_conclusion": "success",
    }


def bindings():
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
            "unique_nonignored_causal_loss_positions": 15_000_000,
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


def trusted_authorities(data):
    return {
        role: copy.deepcopy(data[role]["authority"])
        for role in ("tokenizer", "d04", "d05", "d06")
    }


def bind(data):
    trusted = trusted_authorities(data)
    return bind_terminal_authorities(
        load_policy(),
        data,
        trusted_authorities=trusted,
        expected_trusted_authorities_identity_sha256=identity_sha256(trusted),
    )


def set_path(value, path, replacement):
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("truth_boundary", "training_authorized"), 0),
        (("truth_boundary", "authorized_optimized_targets"), False),
        (("recipe", "micro_batch_size"), True),
        (("recipe", "warmup_steps"), False),
        (("recipe", "sequence_length"), float(DEFAULT_SEQUENCE_LENGTH)),
        (("recipe", "gradient_clip_norm"), 1),
        (("runtime_budget", "max_exposures_per_unique_position"), True),
        (("checkpoint_and_evaluation", "final_test_may_influence_selection"), 0),
        (
            ("source_authorities", "model341", "parameter_count"),
            float(MODEL341["parameter_count"]),
        ),
    ],
)
def test_resealed_policy_equal_valued_wrong_types_fail_closed(path, replacement):
    policy = load_policy()
    set_path(policy, path, replacement)
    body = {
        key: value
        for key, value in policy.items()
        if key != "policy_identity_sha256"
    }
    policy["policy_identity_sha256"] = identity_sha256(body)

    with pytest.raises(RecipeValidationError):
        validate_policy(policy)


@pytest.mark.parametrize(
    ("path", "replacement", "error"),
    [
        (
            ("model", "parameter_count"),
            float(MODEL341["parameter_count"]),
            "MODEL-341",
        ),
        (
            ("d04", "sequence_length"),
            float(DEFAULT_SEQUENCE_LENGTH),
            "sequence length",
        ),
    ],
)
def test_binding_equal_valued_wrong_numeric_types_fail_closed(
    path,
    replacement,
    error,
):
    data = bindings()
    set_path(data, path, replacement)

    with pytest.raises(RecipeValidationError, match=error):
        bind(data)


def test_valid_typed_contract_still_qualifies_recipe_only():
    session = bind(bindings())
    assert session["status"] == "QUALIFIED_RECIPE_ONLY"
    assert session["training_authorized"] is False
    assert session["compute_authorized"] is False
    assert session["authorized_optimized_targets"] == 0
    assert session["optimizer_updates_executed"] == 0
