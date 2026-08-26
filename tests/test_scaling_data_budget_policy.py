from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/scaling/data_budget_policy_v1.json"
VALIDATOR = ROOT / "tools/validate_scaling_data_budget_policy.py"

spec = importlib.util.spec_from_file_location("scaling_data_budget_validator", VALIDATOR)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def load_policy() -> dict:
    return json.loads(POLICY.read_text(encoding="utf-8"))


def _recompute_stage_budget(stage: dict) -> None:
    parameter_count = stage["parameter_count"]
    stage["reference_training_token_exposures"] = {
        "20x": parameter_count * 20,
        "50x": parameter_count * 50,
        "100x": parameter_count * 100,
    }
    stage["approx_dense_training_flops_at_20x"] = 6 * parameter_count * (
        parameter_count * 20
    )


def test_current_policy_passes() -> None:
    report = module.validate(load_policy())
    assert report == {
        "status": "PASS",
        "stage_count": 3,
        "primary_parameter_count": 20_613_440,
        "primary_20x_training_token_exposures": 412_268_800,
        "volatile_source_snapshot_embedded": False,
        "source_bytes_are_token_authority": False,
        "token_exposures_are_unique_positions": False,
        "policy_can_authorize_training": False,
    }


def test_bytes_cannot_be_promoted_to_training_tokens() -> None:
    broken = copy.deepcopy(load_policy())
    broken["truth_boundary"]["source_capacity_bytes_are_training_tokens"] = True
    with pytest.raises(ValueError, match="bytes must never be token authority"):
        module.validate(broken)


def test_source_capacity_target_cannot_be_promoted_to_training_budget() -> None:
    broken = copy.deepcopy(load_policy())
    broken["truth_boundary"]["source_capacity_target_is_training_budget"] = True
    with pytest.raises(ValueError, match="must never be training budgets"):
        module.validate(broken)


def test_training_exposures_cannot_be_relabelled_unique() -> None:
    broken = copy.deepcopy(load_policy())
    broken["truth_boundary"]["training_token_exposures_are_unique_loss_positions"] = True
    with pytest.raises(ValueError, match="must not be treated as unique loss positions"):
        module.validate(broken)


def test_20x_reference_is_derived_from_parameter_count() -> None:
    broken = copy.deepcopy(load_policy())
    broken["stages"][0]["reference_training_token_exposures"]["20x"] += 1
    with pytest.raises(ValueError, match="20M_PRIMARY 20x token-exposure reference drift"):
        module.validate(broken)


def test_exact_20m_primary_parameter_identity_cannot_drift() -> None:
    broken = copy.deepcopy(load_policy())
    broken["stages"][0]["parameter_count"] = 20_000_000
    _recompute_stage_budget(broken["stages"][0])
    with pytest.raises(ValueError, match="20M_PRIMARY parameter_count drift"):
        module.validate(broken)


def test_100m_stage_identity_cannot_be_relabelled_with_consistent_math() -> None:
    broken = copy.deepcopy(load_policy())
    broken["stages"][1]["parameter_count"] = 10_000_000
    _recompute_stage_budget(broken["stages"][1])
    with pytest.raises(ValueError, match="100M_TARGET parameter_count drift"):
        module.validate(broken)


def test_1b_stage_identity_cannot_be_relabelled_with_consistent_math() -> None:
    broken = copy.deepcopy(load_policy())
    broken["stages"][2]["parameter_count"] = 100_000_000
    _recompute_stage_budget(broken["stages"][2])
    with pytest.raises(ValueError, match="1B_TARGET parameter_count drift"):
        module.validate(broken)


def test_20x_reference_cannot_become_hard_minimum() -> None:
    broken = copy.deepcopy(load_policy())
    broken["truth_boundary"]["planning_reference_is_minimum_training_requirement"] = True
    with pytest.raises(ValueError, match="cannot become a hard minimum"):
        module.validate(broken)


def test_paid_compute_cannot_be_authorized_by_research_policy() -> None:
    broken = copy.deepcopy(load_policy())
    broken["truth_boundary"]["paid_compute_authorized"] = True
    with pytest.raises(ValueError, match="must not authorize paid compute"):
        module.validate(broken)


def test_scaling_policy_cannot_self_authorize_training() -> None:
    broken = copy.deepcopy(load_policy())
    broken["truth_boundary"]["this_policy_can_authorize_training"] = True
    with pytest.raises(ValueError, match="must not self-authorize training"):
        module.validate(broken)


def test_unique_token_evidence_cannot_be_removed() -> None:
    broken = copy.deepcopy(load_policy())
    broken["long_training_gate"]["required_evidence"].remove(
        "exact_post_tokenization_unique_train_token_count"
    )
    with pytest.raises(ValueError, match="required evidence set drift"):
        module.validate(broken)


def test_replay_policy_cannot_be_removed() -> None:
    broken = copy.deepcopy(load_policy())
    broken["long_training_gate"]["required_evidence"].remove(
        "preregistered_replay_policy_and_epoch_cap"
    )
    with pytest.raises(ValueError, match="required evidence set drift"):
        module.validate(broken)


def test_explicit_compute_authorization_cannot_be_removed() -> None:
    broken = copy.deepcopy(load_policy())
    broken["long_training_gate"]["required_evidence"].remove(
        "explicit_compute_authorization"
    )
    with pytest.raises(ValueError, match="required evidence set drift"):
        module.validate(broken)


def test_volatile_live_source_snapshot_is_rejected() -> None:
    broken = copy.deepcopy(load_policy())
    broken["current_20m_observation"] = {
        "source_registry_pr": 538,
        "observed_source_capacity_bytes": 303_374,
    }
    with pytest.raises(ValueError, match="volatile live source snapshots"):
        module.validate(broken)


def test_source_registry_cannot_be_relabelled_as_corpus_identity() -> None:
    broken = copy.deepcopy(load_policy())
    broken["data_authority_contract"]["source_registry_is_corpus_identity"] = True
    with pytest.raises(ValueError, match="data authority firewall missing"):
        module.validate(broken)


def test_source_registry_cannot_be_token_authority() -> None:
    broken = copy.deepcopy(load_policy())
    broken["data_authority_contract"]["source_registry_is_token_count_authority"] = True
    with pytest.raises(ValueError, match="data authority firewall missing"):
        module.validate(broken)


def test_unique_vs_replayed_accounting_cannot_be_disabled() -> None:
    broken = copy.deepcopy(load_policy())
    broken["data_authority_contract"][
        "promotion_requires_unique_vs_replayed_exposure_accounting"
    ] = False
    with pytest.raises(ValueError, match="data authority invariant missing"):
        module.validate(broken)


def test_stage_cannot_embed_volatile_readiness_status() -> None:
    broken = copy.deepcopy(load_policy())
    broken["stages"][0]["status_source"] = "READY"
    with pytest.raises(ValueError, match="must not embed volatile readiness status"):
        module.validate(broken)
