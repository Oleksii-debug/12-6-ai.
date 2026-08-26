from __future__ import annotations

from dataclasses import dataclass

import pytest

from twelve_six.experiment_failure import (
    FailureClass,
    FailurePhase,
    FailureSignal,
    RunFailureTracker,
    build_report,
    classify,
    make_signal_for_process,
    validate_report,
)


def test_missing_pytest_is_bootstrap_failure_not_model_failure() -> None:
    signal, codes = make_signal_for_process(
        phase=FailurePhase.FOCUSED_TEST,
        return_code=1,
        stderr_tail=b"/venv/bin/python: No module named pytest\n",
    )
    assert classify(signal) is FailureClass.BOOTSTRAP_DEPENDENCY_MISSING
    assert signal.experiment_started is False
    assert signal.optimizer_steps_completed == 0
    assert "PYTHON_MODULE_MISSING:pytest" in codes
    assert classify(signal) not in {
        FailureClass.NONFINITE_TRAINING,
        FailureClass.RESOURCE_OOM,
        FailureClass.CHECKPOINT_FAILURE,
        FailureClass.EVALUATION_FAILURE,
    }


def test_generic_pre_optimizer_failure_is_experiment_not_started() -> None:
    signal = FailureSignal(phase=FailurePhase.PREPARE, return_code=1)
    assert classify(signal) is FailureClass.EXPERIMENT_NOT_STARTED
    assert signal.experiment_started is False


def test_first_forward_nonfinite_remains_training_class_but_not_started() -> None:
    signal, _ = make_signal_for_process(
        phase=FailurePhase.TRAINING,
        return_code=1,
        stderr_tail=b"NonFiniteTrainingError: non-finite loss at micro_step=1\n",
    )
    assert classify(signal) is FailureClass.NONFINITE_TRAINING
    assert signal.experiment_started is False


def test_resource_oom_after_steps_records_started() -> None:
    signal, _ = make_signal_for_process(
        phase=FailurePhase.TRAINING,
        return_code=137,
        optimizer_steps_completed=3,
        stderr_tail=b"CUDA out of memory\n",
    )
    assert classify(signal) is FailureClass.RESOURCE_OOM
    assert signal.experiment_started is True


@dataclass
class _Metrics:
    optimizer_step: int
    optimizer_stepped: bool


def test_tracker_counts_only_committed_optimizer_steps_for_resume() -> None:
    tracker = RunFailureTracker(start_optimizer_step=250)
    tracker.observe_metrics(_Metrics(optimizer_step=250, optimizer_stepped=False))
    assert tracker.experiment_started is False
    tracker.observe_metrics(_Metrics(optimizer_step=251, optimizer_stepped=True))
    assert tracker.optimizer_steps_completed == 1
    assert tracker.experiment_started is True
    tracker.observe_metrics(_Metrics(optimizer_step=252, optimizer_stepped=True))
    assert tracker.optimizer_steps_completed == 2


def test_checkpoint_and_evaluation_are_phase_specific() -> None:
    checkpoint = FailureSignal(
        phase=FailurePhase.CHECKPOINT,
        return_code=1,
        optimizer_steps_completed=4,
    )
    evaluation = FailureSignal(
        phase=FailurePhase.EVALUATION,
        return_code=1,
        optimizer_steps_completed=4,
    )
    assert classify(checkpoint) is FailureClass.CHECKPOINT_FAILURE
    assert classify(evaluation) is FailureClass.EVALUATION_FAILURE


def test_report_is_machine_valid_and_retains_no_raw_output_or_environment() -> None:
    signal = FailureSignal(
        phase=FailurePhase.FOCUSED_TEST,
        return_code=1,
        missing_dependency="pytest",
    )
    report = build_report(
        signal,
        diagnostic_codes=("PYTHON_MODULE_MISSING:pytest",),
        diagnostic_summary="required test module absent",
        source_sha="abc123",
        stdout_digest={"bytes": 0, "sha256": "0" * 64},
        stderr_digest={"bytes": 17, "sha256": "1" * 64},
    )
    validate_report(report)
    assert report["failure_class"] == "BOOTSTRAP_DEPENDENCY_MISSING"
    assert report["experiment_started"] is False
    assert report["diagnostics"]["raw_output_retained"] is False
    assert report["diagnostics"]["environment_retained"] is False
    assert "No module named" not in str(report)


def test_report_hash_detects_mutation() -> None:
    report = build_report(FailureSignal(phase=FailurePhase.STATIC_CHECK, return_code=1))
    report["diagnostics"]["summary"] = "mutated"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_report(report)
