from __future__ import annotations

import pytest

from twelve_six.training.clip_10m_transfer import (
    derive_clip_thresholds,
    select_weakest_intervention,
)


def test_thresholds_are_observation_derived_and_do_not_inject_clip_one() -> None:
    thresholds = derive_clip_thresholds([2.1, 2.2, 2.4, 2.5, 2.7, 2.9, 3.2, 3.6])
    assert thresholds[0] is None
    assert 1.0 not in thresholds
    assert thresholds == [None, 3.6]


def _candidate(label: str, threshold, bpb: float, spikes: int, warnings: int, freq: float):
    return {
        "label": label,
        "gradient_clip_norm": threshold,
        "summary": {
            "status": "PASS",
            "final_bpb": bpb,
            "loss_spikes": {"count": spikes},
            "depth_warning_steps": list(range(warnings)),
            "clip_frequency": freq,
        },
    }


def test_selection_prefers_no_clip_when_quality_and_stability_are_preserved() -> None:
    result = select_weakest_intervention([
        _candidate("clip_none", None, 3.00, 0, 0, 0.0),
        _candidate("clip_4", 4.0, 2.99, 0, 0, 0.08),
    ])
    assert result["selected_label"] == "clip_none"
    assert result["selected_gradient_clip_norm"] is None


def test_selection_uses_highest_threshold_when_no_clip_is_not_quality_equivalent() -> None:
    result = select_weakest_intervention([
        _candidate("clip_none", None, 3.20, 2, 3, 0.0),
        _candidate("clip_4", 4.0, 3.00, 0, 0, 0.08),
        _candidate("clip_3", 3.0, 2.99, 0, 0, 0.15),
    ])
    assert result["selected_label"] == "clip_4"


def test_threshold_derivation_requires_real_window() -> None:
    with pytest.raises(Exception):
        derive_clip_thresholds([2.0, 2.1, 2.2])
