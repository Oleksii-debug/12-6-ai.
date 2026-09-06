from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools.validate_train344b_model341_optimizer_mechanics import (
    ContractError,
    canonical_identity,
    load_contract,
    validate_dependency_firewall,
    validate_exact_model,
    validate_frozen_optimizer,
)

ROOT = Path(__file__).resolve().parents[1]


def test_contract_self_hash_and_frozen_optimizer() -> None:
    contract = load_contract(ROOT)
    assert contract["identity_sha256"] == canonical_identity(contract)
    validate_frozen_optimizer(contract)
    validate_dependency_firewall(contract)


def test_exact_model341_identity_gate() -> None:
    contract = load_contract(ROOT)
    gate = validate_exact_model(ROOT, contract)
    assert gate["parameter_count"] == 20_613_440
    assert gate["model_spec_sha256"] == "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
    assert gate["optimizer_updates_authorized_for_synthetic_probe"] == 96
    assert gate["learned_corpus_optimizer_updates_authorized"] == 0


def test_broader_18m_22m_window_cannot_substitute_for_exact_model() -> None:
    contract = load_contract(ROOT)
    mutated = copy.deepcopy(contract)
    mutated["target_model"]["parameter_count"] = 20_000_000
    with pytest.raises(ContractError, match="config expected parameter count drift|runtime parameter count drift"):
        validate_exact_model(ROOT, mutated)


def test_optimizer_grid_drift_fails_closed() -> None:
    contract = load_contract(ROOT)
    mutated = copy.deepcopy(contract)
    mutated["optimizer"]["learning_rate_candidates"] = [0.0001, 0.0002, 0.0003]
    with pytest.raises(ContractError, match="frozen TRAIN-344 optimizer semantics drift"):
        validate_frozen_optimizer(mutated)


def test_dependency_firewall_cannot_authorize_learned_updates() -> None:
    contract = load_contract(ROOT)
    mutated = copy.deepcopy(contract)
    mutated["dependency_firewall"]["learned_corpus_optimizer_updates_authorized"] = 1
    with pytest.raises(ContractError, match="learned-corpus optimizer updates must remain zero"):
        validate_dependency_firewall(mutated)
