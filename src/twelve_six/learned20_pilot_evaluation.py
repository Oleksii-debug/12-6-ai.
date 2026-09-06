"""D06 fail-closed scientific evidence for a terminal learned-20M pilot."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

REQUIRED_STRATA = ("UA", "EN", "CODE")


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _unit_interval(value: Any) -> bool:
    return _nonnegative_number(value) and float(value) <= 1.0


def _sha256_digest(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value[7:]
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def _validate_heldout_metrics(d06: Mapping[str, Any], blockers: list[str]) -> None:
    heldout = d06.get("heldout_metrics")
    if not isinstance(heldout, Mapping):
        blockers.append("bounded_pilot.d06.heldout_metrics_missing")
        return

    total_targets = 0
    weighted_before = 0.0
    weighted_after = 0.0
    for stratum in REQUIRED_STRATA:
        item = heldout.get(stratum)
        prefix = f"bounded_pilot.d06.heldout_metrics.{stratum}"
        if not isinstance(item, Mapping):
            blockers.append(f"{prefix}_missing")
            continue
        targets = item.get("target_count")
        before = item.get("random_init_mean_nll")
        after = item.get("trained_mean_nll")
        if not _positive_int(targets):
            blockers.append(f"{prefix}.target_count_invalid")
            continue
        if not _nonnegative_number(before):
            blockers.append(f"{prefix}.random_init_mean_nll_invalid")
            continue
        if not _nonnegative_number(after):
            blockers.append(f"{prefix}.trained_mean_nll_invalid")
            continue
        total_targets += targets
        weighted_before += float(before) * targets
        weighted_after += float(after) * targets

    if total_targets <= 0:
        return

    claimed_before = d06.get("weighted_random_init_mean_nll")
    claimed_after = d06.get("weighted_trained_mean_nll")
    expected_before = weighted_before / total_targets
    expected_after = weighted_after / total_targets
    tolerance = 1e-12

    if not _nonnegative_number(claimed_before) or not math.isclose(
        float(claimed_before), expected_before, rel_tol=tolerance, abs_tol=tolerance
    ):
        blockers.append("bounded_pilot.d06.weighted_random_init_mean_nll_mismatch")
    if not _nonnegative_number(claimed_after) or not math.isclose(
        float(claimed_after), expected_after, rel_tol=tolerance, abs_tol=tolerance
    ):
        blockers.append("bounded_pilot.d06.weighted_trained_mean_nll_mismatch")
    if not expected_after < expected_before:
        blockers.append("bounded_pilot.d06.heldout_not_better_than_random_init")


def _validate_selection_trajectory(d06: Mapping[str, Any], blockers: list[str]) -> None:
    trajectory = d06.get("selection_trajectory")
    if (
        not isinstance(trajectory, Sequence)
        or isinstance(trajectory, (str, bytes))
        or len(trajectory) < 2
    ):
        blockers.append("bounded_pilot.d06.selection_trajectory_insufficient")
        return

    previous_step = -1
    previous_exposure = -1
    first_nll: float | None = None
    last_nll: float | None = None
    for index, point in enumerate(trajectory):
        prefix = f"bounded_pilot.d06.selection_trajectory.{index}"
        if not isinstance(point, Mapping):
            blockers.append(f"{prefix}_invalid")
            continue
        step = point.get("optimizer_step")
        exposure = point.get("optimized_target_exposure")
        nll = point.get("mean_nll")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            blockers.append(f"{prefix}.optimizer_step_invalid")
        elif step <= previous_step:
            blockers.append("bounded_pilot.d06.selection_trajectory_steps_not_strict")
        else:
            previous_step = step
        if not isinstance(exposure, int) or isinstance(exposure, bool) or exposure < 0:
            blockers.append(f"{prefix}.optimized_target_exposure_invalid")
        elif exposure <= previous_exposure:
            blockers.append("bounded_pilot.d06.selection_trajectory_exposure_not_strict")
        else:
            previous_exposure = exposure
        if not _nonnegative_number(nll):
            blockers.append(f"{prefix}.mean_nll_invalid")
        else:
            value = float(nll)
            if first_nll is None:
                first_nll = value
            last_nll = value

    if first_nll is not None and last_nll is not None and not last_nll < first_nll:
        blockers.append("bounded_pilot.d06.selection_trajectory_not_improving")


def _validate_inference_probe(
    pilot: Mapping[str, Any], d06: Mapping[str, Any], blockers: list[str]
) -> None:
    probe = d06.get("inference_probe")
    if not isinstance(probe, Mapping):
        blockers.append("bounded_pilot.d06.inference_probe_missing")
        return
    if not _nonempty_text(probe.get("prompt_suite_identity")):
        blockers.append("bounded_pilot.d06.inference_probe.prompt_suite_identity_missing")
    if not _sha256_digest(probe.get("output_fingerprint")):
        blockers.append("bounded_pilot.d06.inference_probe.output_fingerprint_invalid")
    if probe.get("fresh_process_reload") is not True:
        blockers.append("bounded_pilot.d06.inference_probe.fresh_process_reload_not_proven")
    if probe.get("checkpoint_identity") != pilot.get("result_checkpoint_identity"):
        blockers.append("bounded_pilot.d06.inference_probe.checkpoint_identity_mismatch")


def _validate_memorization(d06: Mapping[str, Any], blockers: list[str]) -> None:
    diagnostic = d06.get("memorization_diagnostic")
    if not isinstance(diagnostic, Mapping):
        blockers.append("bounded_pilot.d06.memorization_diagnostic_missing")
        return
    if not _nonempty_text(diagnostic.get("policy_identity")):
        blockers.append("bounded_pilot.d06.memorization_diagnostic.policy_identity_missing")
    for key in ("training_exact_match_rate", "heldout_exact_match_rate"):
        if not _unit_interval(diagnostic.get(key)):
            blockers.append(f"bounded_pilot.d06.memorization_diagnostic.{key}_invalid")
    if diagnostic.get("passed") is not True:
        blockers.append("bounded_pilot.d06.memorization_diagnostic_not_passed")


def validate_terminal_pilot_evaluation(evidence: Mapping[str, Any]) -> list[str]:
    """Require numeric D06 evidence before a terminal pilot can unlock long training."""

    pilot = evidence.get("bounded_pilot")
    if not isinstance(pilot, Mapping) or pilot.get("terminal") is not True:
        return []

    blockers: list[str] = []
    d06 = pilot.get("d06_evaluation")
    if not isinstance(d06, Mapping):
        return ["bounded_pilot.d06_evaluation_missing"]

    if d06.get("pilot_identity") != pilot.get("identity"):
        blockers.append("bounded_pilot.d06.pilot_identity_mismatch")
    if d06.get("evaluation_firewall_identity") != pilot.get("evaluation_firewall_identity"):
        blockers.append("bounded_pilot.d06.evaluation_firewall_identity_mismatch")
    if not _nonempty_text(d06.get("random_init_baseline_identity")):
        blockers.append("bounded_pilot.d06.random_init_baseline_identity_missing")
    if not _nonempty_text(pilot.get("result_checkpoint_identity")):
        blockers.append("bounded_pilot.result_checkpoint_identity_missing")

    _validate_heldout_metrics(d06, blockers)
    _validate_selection_trajectory(d06, blockers)
    _validate_inference_probe(pilot, d06, blockers)
    _validate_memorization(d06, blockers)

    throughput = d06.get("throughput_optimized_targets_per_second")
    if not _positive_number(throughput):
        blockers.append("bounded_pilot.d06.throughput_measurement_invalid")
    peak_memory = d06.get("peak_memory_bytes")
    if not _positive_int(peak_memory):
        blockers.append("bounded_pilot.d06.peak_memory_bytes_invalid")

    return sorted(set(blockers))
