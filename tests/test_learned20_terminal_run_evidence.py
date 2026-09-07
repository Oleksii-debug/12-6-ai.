from __future__ import annotations

from twelve_six.learned20_terminal_run_evidence import validate_terminal_run_evidence


def _evidence() -> tuple[dict, dict]:
    pilot = {
        "loss_ledger_identity": "ledger-v1",
        "result_checkpoint_identity": "checkpoint-best-v1",
    }
    d06 = {
        "selection_trajectory": [
            {"optimizer_step": 10, "optimized_target_exposure": 40_000, "mean_nll": 5.4},
            {"optimizer_step": 20, "optimized_target_exposure": 80_000, "mean_nll": 5.1},
        ],
        "exposure_accounting": {
            "loss_ledger_identity": "ledger-v1",
            "optimized_target_exposure": 80_000,
            "unique_loss_positions_consumed": 80_000,
            "replay_exposure_count": 0,
            "padding_loss_positions": 0,
        },
        "checkpoint_selection": {
            "best_checkpoint_identity": "checkpoint-best-v1",
            "chronological_final_checkpoint_identity": "checkpoint-final-v1",
            "selection_metric_identity": "selection-bpb-v1",
            "selection_locked": True,
            "final_test_accessed_during_selection": False,
            "best_optimizer_step": 18,
            "final_optimizer_step": 20,
            "best_optimized_target_exposure": 72_000,
            "final_optimized_target_exposure": 80_000,
        },
    }
    return pilot, d06


def test_terminal_run_keeps_best_and_chronological_final_with_exact_accounting() -> None:
    pilot, d06 = _evidence()
    assert validate_terminal_run_evidence(pilot, d06) == []


def test_replay_cannot_silently_manufacture_terminal_exposure() -> None:
    pilot, d06 = _evidence()
    accounting = d06["exposure_accounting"]
    accounting["replay_exposure_count"] = 10
    accounting["optimized_target_exposure"] = 80_010
    d06["selection_trajectory"][-1]["optimized_target_exposure"] = 80_010
    d06["checkpoint_selection"]["final_optimized_target_exposure"] = 80_010
    assert (
        "bounded_pilot.d06.exposure_accounting.replay_forbidden"
        in validate_terminal_run_evidence(pilot, d06)
    )


def test_terminal_selection_trajectory_must_end_at_actual_exposure() -> None:
    pilot, d06 = _evidence()
    d06["selection_trajectory"][-1]["optimized_target_exposure"] = 79_999
    assert (
        "bounded_pilot.d06.selection_trajectory_terminal_exposure_mismatch"
        in validate_terminal_run_evidence(pilot, d06)
    )


def test_result_checkpoint_must_be_the_locked_best_checkpoint() -> None:
    pilot, d06 = _evidence()
    d06["checkpoint_selection"]["best_checkpoint_identity"] = "other-best"
    assert (
        "bounded_pilot.d06.checkpoint_selection.result_not_best_checkpoint"
        in validate_terminal_run_evidence(pilot, d06)
    )


def test_final_checkpoint_must_match_terminal_exposure_and_follow_best() -> None:
    pilot, d06 = _evidence()
    selection = d06["checkpoint_selection"]
    selection["best_optimizer_step"] = 21
    selection["final_optimized_target_exposure"] = 79_000
    blockers = validate_terminal_run_evidence(pilot, d06)
    assert "bounded_pilot.d06.checkpoint_selection.best_step_after_final" in blockers
    assert "bounded_pilot.d06.checkpoint_selection.final_exposure_mismatch" in blockers


def test_final_test_cannot_participate_in_checkpoint_selection() -> None:
    pilot, d06 = _evidence()
    d06["checkpoint_selection"]["final_test_accessed_during_selection"] = True
    assert (
        "bounded_pilot.d06.checkpoint_selection.final_test_selection_leak"
        in validate_terminal_run_evidence(pilot, d06)
    )
