"""D06 context rules for comparing terminal learned-20M evidence to smaller ladders."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

LEARNED20_PARAMETER_COUNT = 20_613_440


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def validate_smaller_ladder_context(d06: Mapping[str, Any]) -> list[str]:
    """Require smaller-model context without overclaiming incomparable quality ordering."""

    comparison = d06.get("smaller_ladder_comparison")
    if not isinstance(comparison, Mapping):
        return ["bounded_pilot.d06.smaller_ladder_comparison_missing"]

    blockers: list[str] = []
    if not _nonempty_text(comparison.get("authority_identity")):
        blockers.append("bounded_pilot.d06.smaller_ladder.authority_identity_missing")
    if not _nonempty_text(comparison.get("budget_caveat")):
        blockers.append("bounded_pilot.d06.smaller_ladder.budget_caveat_missing")

    mode = comparison.get("comparison_mode")
    if mode not in {"MATCHED", "CONTEXT_ONLY"}:
        blockers.append("bounded_pilot.d06.smaller_ladder.comparison_mode_invalid")
    if mode == "CONTEXT_ONLY" and comparison.get("direct_quality_ordering_claimed") is not False:
        blockers.append("bounded_pilot.d06.smaller_ladder.context_only_quality_claim_forbidden")

    rungs = comparison.get("rungs")
    if (
        not isinstance(rungs, Sequence)
        or isinstance(rungs, (str, bytes))
        or len(rungs) < 1
    ):
        blockers.append("bounded_pilot.d06.smaller_ladder.rungs_missing")
        return sorted(set(blockers))

    for index, rung in enumerate(rungs):
        prefix = f"bounded_pilot.d06.smaller_ladder.rungs.{index}"
        if not isinstance(rung, Mapping):
            blockers.append(f"{prefix}_invalid")
            continue
        for key in ("model_identity", "evaluation_identity", "data_identity", "tokenizer_identity"):
            if not _nonempty_text(rung.get(key)):
                blockers.append(f"{prefix}.{key}_missing")
        params = rung.get("parameter_count")
        if not _positive_int(params) or params >= LEARNED20_PARAMETER_COUNT:
            blockers.append(f"{prefix}.parameter_count_invalid")
        if not _positive_int(rung.get("optimized_target_exposure")):
            blockers.append(f"{prefix}.optimized_target_exposure_invalid")
        if not _nonnegative_number(rung.get("best_bpb")):
            blockers.append(f"{prefix}.best_bpb_invalid")

    if mode == "MATCHED":
        reference = comparison.get("learned20_reference")
        if not isinstance(reference, Mapping):
            blockers.append("bounded_pilot.d06.smaller_ladder.learned20_reference_missing")
        else:
            for key in ("evaluation_identity", "data_identity", "tokenizer_identity"):
                if not _nonempty_text(reference.get(key)):
                    blockers.append(
                        f"bounded_pilot.d06.smaller_ladder.learned20_reference.{key}_missing"
                    )
            for index, rung in enumerate(rungs):
                if not isinstance(rung, Mapping):
                    continue
                for key in ("evaluation_identity", "data_identity", "tokenizer_identity"):
                    expected = reference.get(key)
                    if _nonempty_text(expected) and rung.get(key) != expected:
                        blockers.append(
                            f"bounded_pilot.d06.smaller_ladder.rungs.{index}.{key}_not_matched"
                        )

    return sorted(set(blockers))
