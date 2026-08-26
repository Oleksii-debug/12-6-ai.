from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools" / "validate_s4_100m_admission.py"
CONFIG_PATH = ROOT / "configs" / "control" / "s4_100m_admission_v1.json"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_s4_100m_admission", VALIDATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_contract() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_s4_admission_contract_passes() -> None:
    validator = _load_validator()
    errors, computed = validator.validate_contract(_load_contract())

    assert errors == []
    assert computed == {
        "S4-GQA-EXACTISH-v1": 100_000_512,
        "S4-GQA-ALIGNED-v1": 100_885_248,
    }


def test_parameter_count_matches_incumbent_mha_reference() -> None:
    validator = _load_validator()
    incumbent_model = {
        "vocab_size": 256,
        "d_model": 768,
        "n_layers": 13,
        "n_heads": 12,
        "n_kv_heads": 12,
        "head_dim": 64,
        "d_ff": 2304,
        "activation": "swiglu",
        "norm_kind": "rmsnorm",
        "position_embedding": "rope",
        "tie_word_embeddings": True,
    }

    assert validator.parameter_count(incumbent_model) == 99_897_600


def test_expected_parameter_drift_fails_closed() -> None:
    validator = _load_validator()
    contract = deepcopy(_load_contract())
    contract["research_candidates"][0]["expected_parameters"] += 1

    errors, _ = validator.validate_contract(contract)

    assert any("expected_parameters" in error for error in errors)


def test_invalid_gqa_geometry_fails_closed() -> None:
    validator = _load_validator()
    contract = deepcopy(_load_contract())
    contract["research_candidates"][0]["model"]["n_kv_heads"] = 5

    errors, _ = validator.validate_contract(contract)

    assert any("n_heads must be divisible by n_kv_heads" in error for error in errors)


def test_compute_or_training_authorization_cannot_be_silently_enabled() -> None:
    validator = _load_validator()
    contract = deepcopy(_load_contract())
    contract["long_training_authorized"] = True
    contract["gates"]["material_compute_authorized"] = "YES"

    errors, _ = validator.validate_contract(contract)

    assert "long_training_authorized must be false" in errors
    assert "material compute must remain unauthorized" in errors


def test_v1_cannot_fabricate_corpus_identity_or_capacity() -> None:
    validator = _load_validator()
    contract = deepcopy(_load_contract())
    contract["data_gate"]["exact_final_corpus_identity"] = "invented"
    contract["data_gate"]["authorized_unique_no_replay_loss_positions"] = 1

    errors, _ = validator.validate_contract(contract)

    assert "v1 snapshot must not fabricate a final corpus identity" in errors
    assert "v1 snapshot must preserve zero authorized real loss positions" in errors
