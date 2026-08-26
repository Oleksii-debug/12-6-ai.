from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from twelve_six import milestone150_learned_base_ladder as m150
from twelve_six.memorization import build_canary_suite


def _runner():
    path = Path(__file__).parents[1] / "tools/run_recover178_memorization.py"
    spec = importlib.util.spec_from_file_location("recover178_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_m150_geometry_is_reused() -> None:
    assert m150.model_spec("100k").parameter_count() == 95_568
    assert m150.model_spec("500k").parameter_count() == 467_808
    assert m150.model_spec("1m").parameter_count() == 1_037_696


def test_exposure_cycle_has_exact_declared_repetitions() -> None:
    r = _runner()
    cfg = r._config(Path(__file__).parents[1])
    suite = r._suite(cfg)
    events = r._schedule(cfg, suite, 0)
    assert len(events) == 100
    counts = {item.canary_id: 0 for item in suite.canaries}
    for event in events:
        if event is not None:
            counts[str(event["canary_id"])] += 1
    for item in suite.canaries:
        assert counts[item.canary_id] == item.exposure_per_cycle
    assert all(counts[item.canary_id] == 0 for item in suite.canaries if item.control)


def test_public_canary_manifest_never_contains_canary_strings() -> None:
    r = _runner()
    suite = build_canary_suite()
    public = suite.public()
    r._assert_public_safe(public)
    assert public["text_emitted"] is False
    for item in public["canaries"]:
        assert "prefix" not in item
        assert "continuation" not in item
        assert "prefix_sha256" in item
        assert "continuation_sha256" in item


def test_stop_thresholds_bind_at_random_init_and_are_self_hashed() -> None:
    r = _runner()
    cfg = r._config(Path(__file__).parents[1])
    curve = [
        {
            "exposure_per_cycle": 0,
            "nll_per_token_median": 5.0,
            "nll_per_token_mad": 0.1,
            "rank_median": 8.0,
            "rank_percentile_median": 0.5,
            "exact_recovery_rate": 0.0,
            "candidate_count": 16,
            "canary_count": 3,
        },
        {
            "exposure_per_cycle": 16,
            "nll_per_token_median": 5.0,
            "nll_per_token_mad": 0.1,
            "rank_median": 8.0,
            "rank_percentile_median": 0.5,
            "exact_recovery_rate": 0.0,
            "candidate_count": 16,
            "canary_count": 3,
        },
    ]
    binding = r._bind_stop_policy(curve, cfg, model_state_sha256="0" * 64)
    assert binding["bound_before_optimizer_update"] is True
    assert binding["bound_at"] == "random_init_before_any_optimizer_update"
    r._check_self_hash(binding)


def test_public_safety_guard_rejects_raw_text_fields() -> None:
    r = _runner()
    with pytest.raises(r.RecoverError):
        r._assert_public_safe({"text": "must never enter report"})
