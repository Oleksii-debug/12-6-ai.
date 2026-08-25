from __future__ import annotations

import json
import math
from pathlib import Path

from twelve_six.model import InitSpec, ModelSpec
from twelve_six.training.init_seed_experiment import (
    CANDIDATES,
    _candidate_init,
)


CONFIG = Path("configs/experiments/model19_init_seeds_100k_500k.json")


def test_model19_exact_small_candidate_set_and_geometry() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert tuple(payload["candidates"]) == CANDIDATES
    scales = {item["label"]: item for item in payload["scales"]}
    spec100 = ModelSpec.from_dict(scales["research_100k"]["model"])
    spec500 = ModelSpec.from_dict(scales["research_500k"]["model"])
    assert spec100.parameter_count() == scales["research_100k"]["expected_parameters"] == 95_568
    assert spec500.parameter_count() == scales["research_500k"]["expected_parameters"] == 467_808
    assert spec100.identity_sha256() == scales["research_100k"]["expected_model_identity_sha256"]
    assert spec500.identity_sha256() == scales["research_500k"]["expected_model_identity_sha256"]


def test_width_reference_candidate_matches_incumbent_formula() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    base = InitSpec.from_dict(payload["base_init"])
    assert base.identity_sha256() == payload["expected_base_init_identity_sha256"]
    width100 = _candidate_init(
        base,
        candidate="s1_width_reference_control",
        d_model=48,
        width_reference=48,
    )
    width500 = _candidate_init(
        base,
        candidate="s1_width_reference_control",
        d_model=96,
        width_reference=48,
    )
    bad500 = _candidate_init(
        base,
        candidate="unscaled_residual_control",
        d_model=96,
        width_reference=48,
    )
    assert width100.identity_sha256() == base.identity_sha256()
    assert math.isclose(width500.std, 0.02 * math.sqrt(0.5))
    assert width500.residual_branch_scale == "sqrt_2_layers"
    assert bad500.std == 0.02
    assert bad500.residual_branch_scale == "none"


def test_model19_decision_and_rejection_gates_are_predeclared() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert payload["decision"]["min_relative_heldout_improvement_for_v2"] == 0.01
    assert payload["rejection_gates"] == {
        "clip_fraction_threshold": 0.5,
        "max_final_residual_rms_ratio_to_default": 1.5,
        "max_initial_grad_ratio_to_default": 3.0,
    }
    assert payload["truth_boundary"]["canonical_initspec_change_authorized"] is False
