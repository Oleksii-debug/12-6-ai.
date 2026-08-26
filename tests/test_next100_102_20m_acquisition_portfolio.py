from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/data/next100_102_20m_acquisition_portfolio_v1.json"
VALIDATOR_PATH = ROOT / "tools/validate_next100_102_20m_acquisition_portfolio.py"

_spec = importlib.util.spec_from_file_location("next100_102_validator", VALIDATOR_PATH)
assert _spec is not None and _spec.loader is not None
_validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_validator)


def _load() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_canonical_portfolio_passes() -> None:
    report = _validator.validate(_load())
    assert report["status"] == "PASS"
    assert report["remaining_gap_bytes"] == {
        "uk": 8_899_144,
        "en": 6_855_849,
        "code": 3_930_867,
        "total": 19_685_860,
    }
    assert report["planning_budget_bytes"] == {
        "uk": 10_000_000,
        "en": 7_200_000,
        "code": 4_200_000,
        "total": 21_400_000,
    }
    assert report["credited_capacity_from_portfolio_now_bytes"] == 0


def test_research_candidate_cannot_be_promoted_to_capacity() -> None:
    data = _load()
    data["portfolio_lanes"][0]["candidate_authority_status"] = "ADMIT"
    data["portfolio_lanes"][0]["capacity_credit_now_bytes"] = 1
    with pytest.raises(_validator.ValidationError):
        _validator.validate(data)


def test_nonterminal_source_convergence_cannot_be_relabelled_terminal() -> None:
    data = _load()
    data["planning_input"]["source_convergence_status_at_claim"] = "TERMINAL"
    with pytest.raises(_validator.ValidationError):
        _validator.validate(data)


def test_single_family_lane_cannot_exceed_target_family_cap() -> None:
    data = _load()
    lane = next(row for row in data["portfolio_lanes"] if row["lane_id"] == "CODE-SQLALCHEMY-MIT")
    lane["planning_budget_bytes"] = 2_400_001
    data["portfolio_budget_summary"]["planning_budget_bytes"]["code"] += 1_000_001
    data["portfolio_budget_summary"]["planning_budget_bytes"]["total"] += 1_000_001
    with pytest.raises(_validator.ValidationError):
        _validator.validate(data)


def test_portfolio_budget_must_cover_each_current_gap() -> None:
    data = _load()
    lane = next(row for row in data["portfolio_lanes"] if row["lane_id"] == "EN-OPEN-PROSE-INDEPENDENT-FAMILIES")
    lane["planning_budget_bytes"] = 1
    data["portfolio_budget_summary"]["planning_budget_bytes"]["en"] = 5_600_001
    data["portfolio_budget_summary"]["planning_budget_bytes"]["total"] = 19_800_001
    with pytest.raises(_validator.ValidationError):
        _validator.validate(data)


def test_research_evidence_cannot_become_source_authority() -> None:
    data = _load()
    data["research_evidence"][0]["authority_effect"] = "TRAINING_AUTHORIZED"
    with pytest.raises(_validator.ValidationError):
        _validator.validate(data)


def test_unsafe_training_or_paid_compute_claim_is_rejected() -> None:
    for field in ("model_training_executed", "tokenizer_fit_executed", "paid_compute_used"):
        data = copy.deepcopy(_load())
        data[field] = True
        with pytest.raises(_validator.ValidationError):
            _validator.validate(data)


def test_required_dedup_gate_cannot_be_removed() -> None:
    data = _load()
    data["portfolio_lanes"][0]["hard_filters"].remove("GLOBAL_DEDUP")
    with pytest.raises(_validator.ValidationError):
        _validator.validate(data)


def test_downstream_order_cannot_skip_corpus_decontamination() -> None:
    data = _load()
    data["downstream_order"].remove("EVALUATION_DECONTAMINATION")
    with pytest.raises(_validator.ValidationError):
        _validator.validate(data)
