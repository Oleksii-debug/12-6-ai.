from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/r01_byte_token_unit_firewall_v1.json"
VALIDATOR = ROOT / "tools/validate_r01_byte_token_unit_firewall.py"

spec = importlib.util.spec_from_file_location("r01_unit_firewall", VALIDATOR)
assert spec is not None and spec.loader is not None
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_frozen_unit_firewall_is_valid() -> None:
    assert validator.validate_firewall(_load()) == []


def test_token_ratio_cannot_become_execution_budget() -> None:
    data = copy.deepcopy(_load())
    data["source_reported_token_ratios_are_execution_budgets"] = True
    errors = validator.validate_firewall(data)
    assert any("execution budgets" in error for error in errors)


def test_token_ratio_cannot_be_directly_converted_to_byte_positions() -> None:
    data = copy.deepcopy(_load())
    data["source_reported_token_ratios_may_be_converted_to_byte_positions"] = True
    errors = validator.validate_firewall(data)
    assert any("converted directly" in error for error in errors)


def test_science_complete_budget_cannot_be_invented_before_calibration() -> None:
    data = copy.deepcopy(_load())
    data["science_complete_20m_byte_position_budget"] = 412_268_800
    errors = validator.validate_firewall(data)
    assert any("must remain undefined" in error for error in errors)


def test_engineering_20m_pilot_cannot_masquerade_as_complete_budget() -> None:
    data = copy.deepcopy(_load())
    data["engineering_pilot_is_science_complete_budget"] = True
    errors = validator.validate_firewall(data)
    assert any("must not masquerade" in error for error in errors)


def test_required_flop_calibration_cannot_be_removed() -> None:
    data = copy.deepcopy(_load())
    data["required_calibrations"].remove("flop_normalized_byte_vs_subword_ablation")
    errors = validator.validate_firewall(data)
    assert any("calibration set" in error for error in errors)


def test_tokenizer_agnostic_bpb_calibration_cannot_be_removed() -> None:
    data = copy.deepcopy(_load())
    data["required_calibrations"].remove("tokenizer_agnostic_bits_per_byte")
    errors = validator.validate_firewall(data)
    assert any("calibration set" in error for error in errors)


def test_cross_tokenizer_primary_metric_must_remain_bpb() -> None:
    data = copy.deepcopy(_load())
    data["cross_tokenizer_primary_metric"] = "TOKEN_PERPLEXITY"
    errors = validator.validate_firewall(data)
    assert any("BITS_PER_BYTE" in error for error in errors)


def test_token_nll_cannot_rank_different_tokenizers() -> None:
    data = copy.deepcopy(_load())
    data["token_nll_or_perplexity_may_rank_tokenizers"] = True
    errors = validator.validate_firewall(data)
    assert any("must not rank" in error for error in errors)


def test_bpb_authority_cannot_be_dropped() -> None:
    data = copy.deepcopy(_load())
    data["bits_per_byte_authority_required"] = False
    errors = validator.validate_firewall(data)
    assert any("bits-per-byte authority" in error for error in errors)


def test_reference_ratio_alone_cannot_promote_100m() -> None:
    data = copy.deepcopy(_load())
    data["promotion_to_100m_allowed_from_reference_ratio_alone"] = True
    errors = validator.validate_firewall(data)
    assert any("100M promotion" in error for error in errors)


def test_firewall_cannot_authorize_long_or_paid_training() -> None:
    for field in ("long_training_authorized", "paid_compute_authorized"):
        data = copy.deepcopy(_load())
        data[field] = True
        errors = validator.validate_firewall(data)
        assert any(field.split("_")[0] in error.lower() for error in errors)
