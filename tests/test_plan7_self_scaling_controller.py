"""Plan 7 S3 bounded value/cost, stop, negative and worker recovery tests."""
from dataclasses import replace

import pytest

from twelve_six.self_scaling_controller import (
    ACTIONS,
    ScalingDecisionRejected,
    ScalingEnvelope,
    ScalingProposal,
    choose_scaling_action,
    verify_scaling_decision,
)


def envelope(**changes):
    e = ScalingEnvelope(
        version=1, run_id="synthetic-run", epoch=4,
        modelspec_sha256="a" * 64, protocol_sha256="b" * 64,
        remaining_cost_units=20.0, min_expected_gain=0.1,
        min_value_per_cost=0.05, min_observations=3,
        max_peak_memory_bytes=50_000, max_flops=100_000,
        max_workers=2,
    )
    return replace(e, **changes)


def proposal(action="MEMORY_RAG", **changes):
    p = ScalingProposal(
        action=action, evidence_sha256="c" * 64,
        observations=5, measured_gain=4.0, expected_gain=3.0,
        measured_cost_units=2.0, projected_cost_units=3.0,
        projected_peak_memory_bytes=1000, projected_flops=1500,
        workers_required=1, healthy_workers=2,
    )
    return replace(p, **changes)


def test_five_modes_measured_value_per_cost_selection():
    opts = tuple(proposal(a, projected_cost_units=i+1.0) for i, a in enumerate(ACTIONS))
    decision = choose_scaling_action(envelope(), opts)
    assert decision.action == "SAME_SIZE_LEARNING"
    assert decision.reason == "evidence_driven_local_free_advisory_only"
    assert decision.expected_gain == 3.0
    assert decision.compute_authorized is False
    assert decision.training_authorized is False
    assert decision.model_promotion_authorized is False
    assert verify_scaling_decision(envelope(), opts, decision)
    assert choose_scaling_action(envelope(), tuple(reversed(opts))) == decision


def test_prefers_value_over_absolute_gain():
    opts = (proposal("LARGER_MODEL", measured_gain=10.0, expected_gain=8.0,
                     projected_cost_units=19.0), proposal("TOOL_USE",
                     measured_gain=2.5, expected_gain=2.0, projected_cost_units=1.0))
    assert choose_scaling_action(envelope(), opts).action == "TOOL_USE"


@pytest.mark.parametrize("overrides,reason", [
    ({"resource_class": "PAID_GPU"}, "paid_or_nonlocal_compute_denied"),
    ({"paid_compute_requested": True}, "paid_or_nonlocal_compute_denied"),
    ({"observations": 1}, "insufficient_measurements"),
    ({"healthy_workers": 0}, "worker_capacity_or_failure"),
    ({"workers_required": 3}, "worker_capacity_or_failure"),
    ({"projected_peak_memory_bytes": 50001}, "resource_or_cost_budget_exceeded"),
    ({"projected_flops": 100001}, "resource_or_cost_budget_exceeded"),
    ({"projected_cost_units": 21}, "resource_or_cost_budget_exceeded"),
    ({"measured_gain": 0}, "insufficient_measured_or_expected_gain"),
    ({"expected_gain": 0}, "insufficient_measured_or_expected_gain"),
])
def test_fail_closed_without_launch_grant(overrides, reason):
    p = proposal(**overrides)
    result = choose_scaling_action(envelope(), (p,))
    assert result.action == "STOP"
    assert result.rejected == (f"MEMORY_RAG:{reason}",)
    assert result.compute_authorized is False


def test_worker_loss_falls_back_to_healthy_path_and_restart_is_exact():
    failed = proposal("LARGER_MODEL", healthy_workers=0)
    fallback = proposal("MEMORY_RAG", measured_gain=1.0, expected_gain=1.0)
    opts = (failed, fallback)
    decision = choose_scaling_action(envelope(), opts)
    assert decision.action == "MEMORY_RAG"
    assert "LARGER_MODEL:worker_capacity_or_failure" in decision.rejected
    assert choose_scaling_action(envelope(), opts) == decision
    assert verify_scaling_decision(envelope(), opts, decision)
    assert not verify_scaling_decision(envelope(epoch=3), opts, decision)
    assert not verify_scaling_decision(envelope(protocol_sha256="d"*64), opts, decision)
    assert not verify_scaling_decision(envelope(), (replace(fallback, evidence_sha256="e"*64), failed), decision)
    assert not verify_scaling_decision(envelope(), opts, replace(decision, action="LARGER_MODEL"))
    assert not verify_scaling_decision(envelope(), opts, replace(decision, compute_authorized=True))
    assert not verify_scaling_decision(envelope(), opts, replace(decision, receipt_sha256="f"*64))


def test_explicit_halt_and_exhausted_budget():
    assert choose_scaling_action(envelope(halted=True), (proposal(),)).reason == "halted"
    assert choose_scaling_action(envelope(remaining_cost_units=0.0), (proposal(),)).reason == "budget_exhausted"
    assert choose_scaling_action(envelope(min_value_per_cost=10.0), (proposal(),)).action == "STOP"
    assert choose_scaling_action(envelope(), ()).action == "STOP"


@pytest.mark.parametrize("changes", [
    {"observations": True}, {"expected_gain": float("nan")},
    {"measured_gain": float("inf")}, {"projected_cost_units": -1},
    {"evidence_sha256": "BAD"}, {"action": "RUN_PAID_TRAINING"},
    {"workers_required": 0}, {"paid_compute_requested": 1},
])
def test_malformed_or_adversarial_proposals_refused(changes):
    with pytest.raises(ScalingDecisionRejected):
        proposal(**changes)


@pytest.mark.parametrize("changes", [
    {"version": 2}, {"run_id": ""}, {"epoch": -1},
    {"remaining_cost_units": float("inf")},
    {"remaining_cost_units": -0.01},
    {"min_value_per_cost": float("nan")},
    {"max_workers": True}, {"modelspec_sha256": "bad"},
])
def test_bad_budget_or_identity_packets_refused(changes):
    with pytest.raises(ScalingDecisionRejected):
        envelope(**changes)


def test_duplicate_actions_or_unbounded_packet_refused():
    with pytest.raises(ScalingDecisionRejected, match="duplicate"):
        choose_scaling_action(envelope(), (proposal(), proposal()))
    with pytest.raises(ScalingDecisionRejected, match="bounded"):
        choose_scaling_action(envelope(), (proposal(),)*6)
    with pytest.raises(ScalingDecisionRejected, match="tuple"):
        choose_scaling_action(envelope(), [proposal()])


def test_local_fallback_after_paid_candidate_is_denied():
    opts = (proposal("ARCHITECTURE_CHANGE", resource_class="PAID_GPU"),
            proposal("SAME_SIZE_LEARNING"))
    result = choose_scaling_action(envelope(), opts)
    assert result.action == "SAME_SIZE_LEARNING"
    assert result.training_authorized is False
    assert result.rejected == ("ARCHITECTURE_CHANGE:paid_or_nonlocal_compute_denied",)


def test_measured_cost_cannot_be_hidden_by_low_projection():
    costly = proposal("LARGER_MODEL", measured_gain=6.0, expected_gain=5.0,
                      measured_cost_units=19.0, projected_cost_units=1.0)
    cheap = proposal("TOOL_USE", measured_gain=2.0, expected_gain=2.0,
                     measured_cost_units=1.0, projected_cost_units=1.0)
    assert choose_scaling_action(envelope(), (costly, cheap)).action == "TOOL_USE"
