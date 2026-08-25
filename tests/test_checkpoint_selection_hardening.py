from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six.checkpoint_selection import (
    CheckpointRef,
    EvaluationPurpose,
    SelectionRule,
    SelectionValidationObservation,
    build_experiment_selection_report,
    hash_json,
    make_evaluation_purpose,
    select_checkpoint,
)


def _suite_hash(label: str) -> str:
    return hash_json({"suite": label, "version": 1})


def _purposes():
    training = make_evaluation_purpose(
        "training_metrics",
        suite_id="train-v1",
        suite_identity_sha256=_suite_hash("training-hardening"),
        metric_names=("loss",),
    )
    selection = make_evaluation_purpose(
        "selection_validation",
        suite_id="selection-v1",
        suite_identity_sha256=_suite_hash("selection-hardening"),
        metric_names=("bpb",),
    )
    final = make_evaluation_purpose(
        "final_test",
        suite_id="final-v1",
        suite_identity_sha256=_suite_hash("final-hardening"),
        metric_names=("bpb",),
    )
    diagnostic = make_evaluation_purpose(
        "diagnostic_only",
        suite_id="diagnostic-v1",
        suite_identity_sha256=_suite_hash("diagnostic-hardening"),
        metric_names=("accuracy",),
    )
    return training, selection, final, diagnostic


def test_json_style_metric_list_is_normalized_to_immutable_tuple() -> None:
    purpose = EvaluationPurpose(
        purpose="selection_validation",
        suite_id="json-suite",
        suite_identity_sha256=_suite_hash("json-suite"),
        metric_names=["bpb"],  # type: ignore[arg-type]
        selection_eligible=True,
    )
    assert purpose.metric_names == ("bpb",)
    assert isinstance(purpose.metric_names, tuple)


def test_frozen_policy_file_matches_code_defaults_and_self_hash() -> None:
    policy_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "eval139_checkpoint_selection_v1.json"
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    expected_hash = policy.pop("policy_sha256")
    assert hash_json(policy) == expected_hash

    rule = SelectionRule()
    assert policy["selector"] == {
        "metric_name": rule.metric_name,
        "direction": rule.direction,
        "smoother": rule.smoother,
        "smoothing_window": rule.smoothing_window,
        "minimum_improvement": rule.minimum_improvement,
    }


def test_report_recomputes_and_rejects_forged_selection_decision() -> None:
    training, selection, final, diagnostic = _purposes()
    checkpoints = [
        CheckpointRef("c0", 0, 10, 100),
        CheckpointRef("c1", 1, 20, 200),
        CheckpointRef("c2", 2, 30, 300),
        CheckpointRef("c3", 3, 40, 400),
    ]
    observations = [
        SelectionValidationObservation(
            checkpoint_id=checkpoint.checkpoint_id,
            purpose_identity_sha256=selection.identity_sha256,
            metric_name="bpb",
            value=value,
        )
        for checkpoint, value in zip(
            checkpoints,
            [5.0, 4.0, 3.0, 3.5],
            strict=True,
        )
    ]
    decision = select_checkpoint(
        checkpoints,
        observations,
        selection_purpose=selection,
    )
    forged_id = next(
        checkpoint.checkpoint_id
        for checkpoint in checkpoints
        if checkpoint.checkpoint_id != decision.selected_checkpoint_id
    )
    forged = replace(decision, selected_checkpoint_id=forged_id)

    with pytest.raises(ValueError, match="canonical recomputation"):
        build_experiment_selection_report(
            experiment_id="forged-decision",
            decision=forged,
            evaluation_purposes=(training, selection, final, diagnostic),
        )
