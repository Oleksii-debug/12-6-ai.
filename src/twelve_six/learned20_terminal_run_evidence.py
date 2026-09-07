"""D06 terminal run accounting for learned-20M pilot evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_terminal_run_evidence(
    pilot: Mapping[str, Any], d06: Mapping[str, Any]
) -> list[str]:
    """Bind terminal D06 evidence to exact exposure and retained checkpoint identities."""

    blockers: list[str] = []
    accounting = d06.get("exposure_accounting")
    optimized: int | None = None
    if not isinstance(accounting, Mapping):
        blockers.append("bounded_pilot.d06.exposure_accounting_missing")
    else:
        optimized_value = accounting.get("optimized_target_exposure")
        unique_value = accounting.get("unique_loss_positions_consumed")
        replay_value = accounting.get("replay_exposure_count")
        padding_value = accounting.get("padding_loss_positions")
        if not _positive_int(optimized_value):
            blockers.append("bounded_pilot.d06.exposure_accounting.optimized_target_exposure_invalid")
        else:
            optimized = optimized_value
        if not _positive_int(unique_value):
            blockers.append("bounded_pilot.d06.exposure_accounting.unique_loss_positions_consumed_invalid")
        if not _nonnegative_int(replay_value):
            blockers.append("bounded_pilot.d06.exposure_accounting.replay_exposure_count_invalid")
        elif replay_value != 0:
            blockers.append("bounded_pilot.d06.exposure_accounting.replay_forbidden")
        if padding_value != 0:
            blockers.append("bounded_pilot.d06.exposure_accounting.padding_loss_positions_nonzero")
        if (
            _positive_int(optimized_value)
            and _positive_int(unique_value)
            and _nonnegative_int(replay_value)
            and optimized_value != unique_value + replay_value
        ):
            blockers.append("bounded_pilot.d06.exposure_accounting.total_mismatch")
        if accounting.get("loss_ledger_identity") != pilot.get("loss_ledger_identity"):
            blockers.append("bounded_pilot.d06.exposure_accounting.loss_ledger_identity_mismatch")

    trajectory = d06.get("selection_trajectory")
    if optimized is not None and isinstance(trajectory, Sequence) and trajectory:
        last = trajectory[-1]
        if not isinstance(last, Mapping) or last.get("optimized_target_exposure") != optimized:
            blockers.append("bounded_pilot.d06.selection_trajectory_terminal_exposure_mismatch")

    selection = d06.get("checkpoint_selection")
    if not isinstance(selection, Mapping):
        blockers.append("bounded_pilot.d06.checkpoint_selection_missing")
        return sorted(set(blockers))

    for key in (
        "best_checkpoint_identity",
        "chronological_final_checkpoint_identity",
        "selection_metric_identity",
    ):
        if not _nonempty_text(selection.get(key)):
            blockers.append(f"bounded_pilot.d06.checkpoint_selection.{key}_missing")

    if selection.get("selection_locked") is not True:
        blockers.append("bounded_pilot.d06.checkpoint_selection.selection_not_locked")
    if selection.get("final_test_accessed_during_selection") is not False:
        blockers.append("bounded_pilot.d06.checkpoint_selection.final_test_selection_leak")

    best_step = selection.get("best_optimizer_step")
    final_step = selection.get("final_optimizer_step")
    best_exposure = selection.get("best_optimized_target_exposure")
    final_exposure = selection.get("final_optimized_target_exposure")
    for key, value in (
        ("best_optimizer_step", best_step),
        ("final_optimizer_step", final_step),
        ("best_optimized_target_exposure", best_exposure),
        ("final_optimized_target_exposure", final_exposure),
    ):
        if not _positive_int(value):
            blockers.append(f"bounded_pilot.d06.checkpoint_selection.{key}_invalid")

    if _positive_int(best_step) and _positive_int(final_step) and best_step > final_step:
        blockers.append("bounded_pilot.d06.checkpoint_selection.best_step_after_final")
    if (
        _positive_int(best_exposure)
        and _positive_int(final_exposure)
        and best_exposure > final_exposure
    ):
        blockers.append("bounded_pilot.d06.checkpoint_selection.best_exposure_after_final")
    if optimized is not None and _positive_int(final_exposure) and final_exposure != optimized:
        blockers.append("bounded_pilot.d06.checkpoint_selection.final_exposure_mismatch")

    best_identity = selection.get("best_checkpoint_identity")
    if _nonempty_text(best_identity) and best_identity != pilot.get("result_checkpoint_identity"):
        blockers.append("bounded_pilot.d06.checkpoint_selection.result_not_best_checkpoint")

    return sorted(set(blockers))
