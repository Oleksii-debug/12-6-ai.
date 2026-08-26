from __future__ import annotations

from twelve_six.training.clip_10m_final import _decision, _derive_threshold_plan


def _summary(*, bpb: float, clip_frequency: float, post_p95: float, post_max: float):
    return {
        "mean_final_bpb": bpb,
        "final_bpb_by_seed": [bpb, bpb, bpb],
        "global_preclip_norm": {"profile": {"p95": 10.0, "max": 20.0}},
        "global_postclip_norm": {"profile": {"p95": post_p95, "max": post_max}},
        "update_weight_ratio": {"profile": {"p95": 0.01}},
        "clip_frequency_by_seed": [clip_frequency, clip_frequency, clip_frequency],
        "mean_clip_frequency": clip_frequency,
        "loss_spike_count_by_seed": [0, 0, 0],
        "depth_warning_count_by_seed": [0, 0, 0],
        "numerical_failures": [],
    }


def test_threshold_plan_is_narrow_and_weakest_first() -> None:
    plan = _derive_threshold_plan([1.0 + index * 0.1 for index in range(32)])
    assert plan[0]["gradient_clip_norm"] is None
    clipped = [float(item["gradient_clip_norm"]) for item in plan[1:]]
    assert 1 <= len(clipped) <= 2
    assert clipped == sorted(clipped, reverse=True)


def test_decision_prefers_highest_eligible_threshold() -> None:
    specs = [
        {"label": "unclipped", "gradient_clip_norm": None},
        {"label": "clip_p95", "gradient_clip_norm": 12.0},
        {"label": "clip_p90", "gradient_clip_norm": 10.0},
    ]
    summaries = {
        "unclipped": _summary(bpb=4.50, clip_frequency=0.0, post_p95=10.0, post_max=20.0),
        "clip_p95": _summary(bpb=4.505, clip_frequency=0.08, post_p95=9.9, post_max=17.0),
        "clip_p90": _summary(bpb=4.504, clip_frequency=0.15, post_p95=9.0, post_max=15.0),
    }
    result = _decision(specs, summaries)
    assert result["verdict"] == "SELECT_CLIPPING_POLICY"
    assert result["selected_label"] == "clip_p95"
    assert result["selected_gradient_clip_norm"] == 12.0


def test_decision_rejects_quality_regression() -> None:
    specs = [
        {"label": "unclipped", "gradient_clip_norm": None},
        {"label": "clip_p95", "gradient_clip_norm": 12.0},
    ]
    summaries = {
        "unclipped": _summary(bpb=4.50, clip_frequency=0.0, post_p95=10.0, post_max=20.0),
        "clip_p95": _summary(bpb=4.53, clip_frequency=0.08, post_p95=9.0, post_max=15.0),
    }
    result = _decision(specs, summaries)
    assert result["verdict"] == "NO_USABLE_CLIPPING_THRESHOLD"
    assert result["selected_gradient_clip_norm"] is None
