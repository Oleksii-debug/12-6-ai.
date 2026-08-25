from __future__ import annotations

import math

import pytest

from twelve_six.checkpoint_selection import (
    CheckpointRef,
    MetricObservation,
    SelectionRule,
    SelectionValidationObservation,
    build_experiment_selection_report,
    hash_json,
    make_evaluation_purpose,
    select_checkpoint,
)
from twelve_six.evaluation import BenchmarkRegistry, BenchmarkSpec


def _suite_hash(label: str) -> str:
    return hash_json({"suite": label, "version": 1})


def _purposes():
    training = make_evaluation_purpose(
        "training_metrics",
        suite_id="train-telemetry-v1",
        suite_identity_sha256=_suite_hash("training"),
        metric_names=("loss",),
    )
    selection = make_evaluation_purpose(
        "selection_validation",
        suite_id="selection-heldout-v1",
        suite_identity_sha256=_suite_hash("selection"),
        metric_names=("bpb",),
    )
    final = make_evaluation_purpose(
        "final_test",
        suite_id="final-heldout-v1",
        suite_identity_sha256=_suite_hash("final"),
        metric_names=("bpb",),
    )
    diagnostic = make_evaluation_purpose(
        "diagnostic_only",
        suite_id="diagnostic-v1",
        suite_identity_sha256=_suite_hash("diagnostic"),
        metric_names=("accuracy",),
    )
    return training, selection, final, diagnostic


def _checkpoints(prefix: str, tokens: list[int]) -> list[CheckpointRef]:
    return [
        CheckpointRef(
            checkpoint_id=f"{prefix}-{index}",
            ordinal=index,
            optimizer_step=index + 1,
            optimized_tokens=value,
        )
        for index, value in enumerate(tokens)
    ]


def _selection_obs(checkpoints, selection, values):
    return [
        SelectionValidationObservation(
            checkpoint_id=checkpoint.checkpoint_id,
            purpose_identity_sha256=selection.identity_sha256,
            metric_name="bpb",
            value=value,
        )
        for checkpoint, value in zip(checkpoints, values, strict=True)
    ]


def test_purpose_identity_binds_current_benchmark_registry_manifest() -> None:
    registry = BenchmarkRegistry(
        [
            BenchmarkSpec(
                benchmark_id="eval139-selection",
                version="1",
                source_id="project-owned-selection-split-v1",
                held_out=True,
                allowed_uses=("evaluation",),
            )
        ]
    )
    manifest_hash = registry.manifest()["manifest_sha256"]
    purpose = make_evaluation_purpose(
        "selection_validation",
        suite_id="eval139-selection@1",
        suite_identity_sha256=manifest_hash,
        metric_names=("bpb",),
    )
    assert purpose.suite_identity_sha256 == manifest_hash
    assert purpose.selection_eligible is True


def test_learn03_500k_seed1337_smoothed_selection_differs_from_raw_best_and_final() -> None:
    _, selection, _, _ = _purposes()
    checkpoints = [
        CheckpointRef(
            "effb6f1dc3b8731617fdc00ada7e2969ff3afe3fdecc43948ba396a06d115047",
            0, 17, 4284,
        ),
        CheckpointRef(
            "ce70768b7e99bf22e4c85eaaca6a4b5fb73aa29eb74f2bb4075b0bf19213cfc6",
            1, 66, 16632,
        ),
        CheckpointRef(
            "4d73bff00e44d8d42d015b2e2bf57f91bd66cbe89d559ab2223ae85a0c0ab494",
            2, 261, 65772,
        ),
        CheckpointRef(
            "46627fa96b54ee17d89c0a490f4ed6ff68fa5ed41170bf2973711c98ed04df53",
            3, 521, 131292,
        ),
        CheckpointRef(
            "21ccf005e17b80497bafc2f66187efdf71b15e903cb902f1e7d91dc3670e5753",
            4, 1041, 262332,
        ),
    ]
    values = [
        6.660433652291431,
        4.954453463478713,
        3.057124995889164,
        3.0749602234134934,
        3.798958326850685,
    ]
    decision = select_checkpoint(
        checkpoints,
        _selection_obs(checkpoints, selection, values),
        selection_purpose=selection,
    )
    assert decision.selected_checkpoint_id == checkpoints[3].checkpoint_id
    assert decision.absolute_posthoc_best_validation_checkpoint_id == checkpoints[2].checkpoint_id
    assert decision.final_checkpoint_id == checkpoints[4].checkpoint_id
    assert [item.checkpoint_id for item in decision.checkpoint_registry] == [
        item.checkpoint_id for item in checkpoints
    ]


def test_learn03_500k_seed1338_selects_131k_not_final() -> None:
    _, selection, _, _ = _purposes()
    checkpoints = _checkpoints("seed1338", [4284, 16632, 65772, 131292, 262332])
    values = [
        6.649759039928873,
        5.025608066983991,
        3.0757030249965625,
        3.0579617919507247,
        4.021088196702823,
    ]
    decision = select_checkpoint(
        checkpoints,
        _selection_obs(checkpoints, selection, values),
        selection_purpose=selection,
    )
    assert decision.selected_checkpoint_id == checkpoints[3].checkpoint_id
    assert decision.absolute_posthoc_best_validation_checkpoint_id == checkpoints[3].checkpoint_id
    assert decision.final_checkpoint_id == checkpoints[4].checkpoint_id


@pytest.mark.parametrize(
    ("parameters", "losses"),
    [
        (95_568, [5.119896444943872, 4.435370426366825, 2.686370117829578]),
        (1_037_696, [4.302876755742743, 3.014845819756536, 2.0415304297267802]),
    ],
)
def test_executed_research41_64k_prefix_is_compatible(parameters: int, losses: list[float]) -> None:
    """Historical 95K/1.04M evidence is only an executed 64K-token prefix, not a long run."""
    _, selection, _, _ = _purposes()
    checkpoints = _checkpoints(str(parameters), [4284, 16632, 65772])
    bpb = [loss / math.log(2.0) for loss in losses]
    decision = select_checkpoint(
        checkpoints,
        _selection_obs(checkpoints, selection, bpb),
        selection_purpose=selection,
    )
    assert decision.selected_checkpoint_id == checkpoints[-1].checkpoint_id
    assert decision.absolute_posthoc_best_validation_checkpoint_id == checkpoints[-1].checkpoint_id
    assert decision.final_checkpoint_id == checkpoints[-1].checkpoint_id


def test_final_test_value_cannot_change_selection_or_decision_identity() -> None:
    training, selection, final, diagnostic = _purposes()
    checkpoints = _checkpoints("c", [10, 20, 30, 40, 50])
    values = [5.0, 4.0, 3.0, 3.02, 3.6]
    decision = select_checkpoint(
        checkpoints,
        _selection_obs(checkpoints, selection, values),
        selection_purpose=selection,
    )

    def report(final_value: float):
        return build_experiment_selection_report(
            experiment_id="adversarial-final-test",
            decision=decision,
            evaluation_purposes=(training, selection, final, diagnostic),
            nonselection_observations=(
                MetricObservation(
                    checkpoint_id=decision.selected_checkpoint_id,
                    purpose_identity_sha256=final.identity_sha256,
                    metric_name="bpb",
                    value=final_value,
                ),
            ),
        )

    terrible = report(999.0)
    miraculous = report(0.001)
    assert (
        terrible["selection"]["selected_checkpoint_id"]
        == miraculous["selection"]["selected_checkpoint_id"]
    )
    assert (
        terrible["selection"]["selection_decision_sha256"]
        == miraculous["selection"]["selection_decision_sha256"]
    )
    assert terrible["report_sha256"] != miraculous["report_sha256"]


def test_selector_rejects_final_test_observation_type() -> None:
    _, selection, final, _ = _purposes()
    checkpoints = _checkpoints("c", [10, 20, 30])
    hostile = [
        MetricObservation(
            checkpoint_id=checkpoint.checkpoint_id,
            purpose_identity_sha256=final.identity_sha256,
            metric_name="bpb",
            value=float(index),
        )
        for index, checkpoint in enumerate(checkpoints)
    ]
    with pytest.raises(TypeError, match="SelectionValidationObservation only"):
        select_checkpoint(
            checkpoints, hostile, selection_purpose=selection  # type: ignore[arg-type]
        )


def test_selector_rejects_wrong_purpose_identity() -> None:
    _, selection, final, _ = _purposes()
    checkpoints = _checkpoints("c", [10, 20, 30])
    observations = [
        SelectionValidationObservation(
            checkpoint_id=checkpoint.checkpoint_id,
            purpose_identity_sha256=final.identity_sha256,
            metric_name="bpb",
            value=float(index),
        )
        for index, checkpoint in enumerate(checkpoints)
    ]
    with pytest.raises(ValueError, match="selection-validation identity"):
        select_checkpoint(checkpoints, observations, selection_purpose=selection)


def test_final_test_is_allowed_only_on_frozen_selected_checkpoint() -> None:
    training, selection, final, diagnostic = _purposes()
    checkpoints = _checkpoints("c", [10, 20, 30, 40])
    decision = select_checkpoint(
        checkpoints,
        _selection_obs(checkpoints, selection, [5.0, 4.0, 3.0, 3.5]),
        selection_purpose=selection,
    )
    unselected = next(
        checkpoint.checkpoint_id
        for checkpoint in checkpoints
        if checkpoint.checkpoint_id != decision.selected_checkpoint_id
    )
    with pytest.raises(ValueError, match="frozen selected checkpoint"):
        build_experiment_selection_report(
            experiment_id="no-test-sweep",
            decision=decision,
            evaluation_purposes=(training, selection, final, diagnostic),
            nonselection_observations=(
                MetricObservation(
                    checkpoint_id=unselected,
                    purpose_identity_sha256=final.identity_sha256,
                    metric_name="bpb",
                    value=1.0,
                ),
            ),
        )


def test_report_rejects_relabeling_same_suite_as_selection_and_final() -> None:
    training, selection, final, diagnostic = _purposes()
    relabeled_final = make_evaluation_purpose(
        "final_test",
        suite_id="dishonest-final-label",
        suite_identity_sha256=selection.suite_identity_sha256,
        metric_names=("bpb",),
    )
    checkpoints = _checkpoints("c", [10, 20, 30])
    decision = select_checkpoint(
        checkpoints,
        _selection_obs(checkpoints, selection, [3.0, 2.0, 1.0]),
        selection_purpose=selection,
    )
    with pytest.raises(ValueError, match="must be distinct"):
        build_experiment_selection_report(
            experiment_id="no-relabeling",
            decision=decision,
            evaluation_purposes=(training, selection, relabeled_final, diagnostic),
        )
    assert final.suite_identity_sha256 != selection.suite_identity_sha256


def test_minimum_improvement_blocks_noisy_plateau_cherry_pick() -> None:
    _, selection, _, _ = _purposes()
    checkpoints = _checkpoints("c", [10, 20, 30, 40, 50, 60])
    values = [4.0, 3.0, 2.0, 1.995, 1.992, 1.991]
    decision = select_checkpoint(
        checkpoints,
        _selection_obs(checkpoints, selection, values),
        selection_purpose=selection,
        rule=SelectionRule(minimum_improvement=0.01),
    )
    # The raw minimum is last, but tiny sub-threshold noise does not keep moving selection.
    assert decision.absolute_posthoc_best_validation_checkpoint_id == checkpoints[-1].checkpoint_id
    assert decision.selected_checkpoint_id != checkpoints[-1].checkpoint_id
