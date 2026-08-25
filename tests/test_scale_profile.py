from __future__ import annotations

import copy

import pytest
import torch

from twelve_six.scale_profile import (
    ANALYTICAL_STAGE,
    _analytical_meta_spec,
    _build_report,
    _power_law_extrapolate,
    _tensor_bytes,
    validate_report,
)

SOURCE_SHA = "a" * 40


def _row(stage: str, parameters: int, seconds: float) -> dict:
    parameter_bytes = parameters * 4
    optimized_tokens = 127
    return {
        "schema": "12-6.scale-profile-stage.v1",
        "origin": "OBSERVED",
        "stage": stage,
        "source_sha": SOURCE_SHA,
        "geometry": {"parameter_count": parameters},
        "workload": {
            "sequence_length": 128,
            "optimized_tokens_per_step": optimized_tokens,
            "generation_new_tokens": 4,
        },
        "measurements": {
            "construction": {"seconds": {"median": seconds / 8}},
            "forward": {"seconds": {"median": seconds / 4}},
            "canonical_train_microbatch_forward_backward_update": {
                "seconds": {"median": seconds},
                "optimized_tokens_per_second": optimized_tokens / seconds,
                "phase_decomposition": "NOT_EXPOSED_BY_PUBLIC_TRAINER_SEAM",
            },
            "checkpoint_save": {
                "seconds": {"median": seconds / 3},
                "bytes": {"median": parameter_bytes * 3},
            },
            "greedy_generation_stateless": {"seconds": {"median": seconds / 2}},
            "parameter_bytes": parameter_bytes,
            "optimizer_tensor_bytes": parameter_bytes * 2,
        },
        "timing_metrics_in_deterministic_state_fingerprint": False,
    }


def _rows() -> list[dict]:
    return [
        _row("S1", 107_856, 0.01),
        _row("S2", 1_066_112, 0.08),
        _row("S3", 10_059_840, 0.70),
    ]


def test_analytical_geometry_is_approximately_100m() -> None:
    spec = _analytical_meta_spec()
    assert spec.parameter_count() == 100_017_216
    assert 95_000_000 <= spec.parameter_count() <= 105_000_000


def test_power_law_fit_preserves_positive_scale() -> None:
    fit = _power_law_extrapolate(
        _rows(),
        lambda row: row["measurements"][
            "canonical_train_microbatch_forward_backward_update"
        ]["seconds"]["median"],
        100_017_216,
    )
    assert fit["estimate"] > 0.70
    assert fit["exponent"] > 0
    assert fit["max_observed_fit_relative_error"] >= 0


def test_report_separates_observed_and_analytical() -> None:
    report = _build_report(SOURCE_SHA, _rows())
    validate_report(report, source_sha=SOURCE_SHA)
    assert [row["stage"] for row in report["observed"]] == ["S1", "S2", "S3"]
    assert report["extrapolated"][0]["stage"] == ANALYTICAL_STAGE
    assert report["extrapolated"][0]["origin"] == "EXTRAPOLATED_ANALYTICAL"
    assert report["scope"]["w5_s0_profiler"] == "NOT_MODIFIED_NOT_REEXECUTED"
    assert report["decision_support"]["compute_plan_handoff"]["c01_files_modified"] is False


def test_validator_rejects_timing_fingerprint_overclaim() -> None:
    report = _build_report(SOURCE_SHA, _rows())
    tampered = copy.deepcopy(report)
    tampered["observed"][1]["timing_metrics_in_deterministic_state_fingerprint"] = True
    tampered.pop("report_sha256")
    with pytest.raises(ValueError, match="timing metrics"):
        validate_report(tampered, source_sha=SOURCE_SHA)


def test_validator_rejects_canonical_100m_overclaim() -> None:
    report = _build_report(SOURCE_SHA, _rows())
    tampered = copy.deepcopy(report)
    tampered["extrapolated"][0]["geometry"]["status"] = "CANONICAL_STAGE"
    tampered.pop("report_sha256")
    with pytest.raises(ValueError, match="must not claim canonical"):
        validate_report(tampered, source_sha=SOURCE_SHA)


def test_tensor_byte_accounting_walks_nested_state() -> None:
    state = {
        "a": torch.zeros(8, dtype=torch.float32),
        "nested": [torch.zeros(4, dtype=torch.int64), {"x": 7}],
    }
    assert _tensor_bytes(state) == 8 * 4 + 4 * 8
