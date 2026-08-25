from __future__ import annotations

from twelve_six.cpu_training_perf import (
    PROFILE_SCHEMA,
    REFERENCE_TOKENS_PER_STEP,
    CPUTrainingProfile,
    _bounded_interop_candidates,
    _bounded_thread_candidates,
    _bounded_worker_candidates,
    _parity_against,
)
from twelve_six.scaling_experiment import controlled_specs


def test_cpu_profile_is_execution_only_and_keeps_math_shape_explicit() -> None:
    profile = CPUTrainingProfile(
        torch_threads=4,
        interop_threads=2,
        dataloader_workers=1,
        compile_model=False,
    )
    payload = profile.to_dict()
    assert payload["schema"] == PROFILE_SCHEMA
    assert payload["profile_changes_model_identity"] is False
    assert payload["profile_changes_training_math"] is False
    assert payload["valid_targets_per_step"] == REFERENCE_TOKENS_PER_STEP
    assert payload["persistent_workers"] is True


def test_perf_profile_does_not_enter_controlled_model_identity() -> None:
    before = [spec.identity_sha256() for spec in controlled_specs()]
    _ = CPUTrainingProfile(8, 2, 2, True)
    after = [spec.identity_sha256() for spec in controlled_specs()]
    assert before == after
    assert [spec.parameter_count() for spec in controlled_specs()] == [
        95_568,
        267_912,
        467_808,
        1_037_696,
    ]


def test_equal_target_batch_shapes_are_not_equal_causal_math() -> None:
    reference = CPUTrainingProfile(1, 1, 0, False, 4, 64)
    six_by_43 = CPUTrainingProfile(1, 1, 0, False, 6, 43)
    twelve_by_22 = CPUTrainingProfile(1, 1, 0, False, 12, 22)
    assert reference.valid_targets_per_step == REFERENCE_TOKENS_PER_STEP
    assert six_by_43.valid_targets_per_step == REFERENCE_TOKENS_PER_STEP
    assert twelve_by_22.valid_targets_per_step == REFERENCE_TOKENS_PER_STEP
    assert reference.batch_shape != six_by_43.batch_shape != twelve_by_22.batch_shape


def test_exact_parity_guard_requires_loss_batch_and_every_update_trace() -> None:
    reference = {
        "status": "PASS",
        "model_identity_sha256": "a",
        "initial_parameter_sha256": "b",
        "batch_trace_sha256": "c",
        "loss_trace_sha256": "d",
        "update_trace_sha256": "e",
        "final_parameter_sha256": "f",
        "optimizer_steps": 4,
        "tokens_seen": 1008,
    }
    assert _parity_against(reference, dict(reference))["exact"] is True
    for field in (
        "model_identity_sha256",
        "initial_parameter_sha256",
        "batch_trace_sha256",
        "loss_trace_sha256",
        "update_trace_sha256",
        "final_parameter_sha256",
        "optimizer_steps",
        "tokens_seen",
    ):
        candidate = dict(reference)
        candidate[field] = "different" if isinstance(candidate[field], str) else 999
        parity = _parity_against(reference, candidate)
        assert parity["exact"] is False
        assert parity["checks"][field] is False


def test_auto_candidate_axes_are_bounded_and_always_include_safe_reference() -> None:
    threads = _bounded_thread_candidates(8)
    interop = _bounded_interop_candidates(2)
    workers = _bounded_worker_candidates(2)
    assert threads[0] == 1
    assert interop[0] == 1
    assert workers[0] == 0
    assert all(value > 0 for value in threads)
    assert all(value > 0 for value in interop)
    assert all(value >= 0 for value in workers)
