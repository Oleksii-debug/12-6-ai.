from __future__ import annotations

import pytest

from twelve_six.training.scale_launch_gate import (
    ScaleLaunchGateError,
    evaluate_launch_gate,
    project_cost,
)


def _plan() -> dict[str, object]:
    return {
        "source_sha": "a" * 40,
        "tokenizer_identity": "tokenizer-sha",
        "corpus_manifest_sha256": "corpus-sha",
        "architecture_identity": "architecture-sha",
        "precision": "bf16",
        "gpu_class": "H100_SXM_80GB",
        "gpu_count": 1,
        "target_tokens": 20_971_520_000,
        "eur_per_gpu_hour": 2.83,
        "budget_eur": 10_000.0,
        "reserve_fraction": 0.20,
        "minimum_global_tokens_per_second": 8_000.0,
        "maximum_peak_hbm_fraction": 0.90,
        "minimum_qualification_tokens": 134_217_728,
    }


def _evidence() -> dict[str, object]:
    return {
        "measurement_kind": "GPU_MEASURED",
        "source_sha": "a" * 40,
        "tokenizer_identity": "tokenizer-sha",
        "corpus_manifest_sha256": "corpus-sha",
        "architecture_identity": "architecture-sha",
        "precision": "bf16",
        "gpu_class": "H100_SXM_80GB",
        "global_tokens_per_second": 12_000.0,
        "peak_hbm_fraction": 0.82,
        "qualification_tokens": 134_217_728,
        "loss_decreased": True,
        "non_finite_steps": 0,
        "checkpoint_roundtrip_passed": True,
        "resume_continuity_passed": True,
        "data_cursor_resume_passed": True,
        "distributed_same_topology_passed": False,
    }


def test_technical_pass_still_requires_external_owner_authorization() -> None:
    report = evaluate_launch_gate(_plan(), _evidence())

    assert report.technical_qualified is True
    assert report.owner_authorized is False
    assert report.launch_allowed is False
    assert report.projection is not None
    assert "COMPUTE_AUTHORIZED" in report.reasons[-1]


def test_explicit_authorization_allows_only_a_technically_qualified_plan() -> None:
    report = evaluate_launch_gate(
        _plan(),
        _evidence(),
        authorization="COMPUTE_AUTHORIZED",
    )

    assert report.technical_qualified is True
    assert report.owner_authorized is True
    assert report.launch_allowed is True
    assert report.reasons == ()


def test_cpu_throughput_is_never_accepted_for_paid_cost_projection() -> None:
    evidence = _evidence()
    evidence["measurement_kind"] = "CPU_MEASURED"

    with pytest.raises(ScaleLaunchGateError, match="GPU_MEASURED"):
        project_cost(_plan(), evidence)

    report = evaluate_launch_gate(_plan(), evidence, authorization="COMPUTE_AUTHORIZED")
    assert report.technical_qualified is False
    assert report.launch_allowed is False
    assert report.projection is None
    assert any("GPU" in reason for reason in report.reasons)


def test_unfrozen_tokenizer_or_corpus_fails_closed() -> None:
    plan = _plan()
    plan["tokenizer_identity"] = "NOT_FROZEN"

    with pytest.raises(ScaleLaunchGateError, match="tokenizer_identity"):
        evaluate_launch_gate(plan, _evidence())


def test_multi_gpu_plan_requires_real_distributed_training_evidence() -> None:
    plan = _plan()
    plan["gpu_count"] = 4

    report = evaluate_launch_gate(plan, _evidence(), authorization="COMPUTE_AUTHORIZED")
    assert report.technical_qualified is False
    assert report.launch_allowed is False
    assert any("multi-GPU" in reason for reason in report.reasons)


def test_cost_projection_respects_budget_reserve() -> None:
    evidence = _evidence()
    evidence["global_tokens_per_second"] = 500.0

    report = evaluate_launch_gate(_plan(), evidence, authorization="COMPUTE_AUTHORIZED")

    assert report.projection is not None
    assert report.projection.projected_compute_eur > report.projection.spend_ceiling_eur
    assert report.launch_allowed is False
    assert any("budget" in reason for reason in report.reasons)
