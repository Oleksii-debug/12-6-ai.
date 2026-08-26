"""MODEL-116 execution binding to the proven RESEARCH41 controlled geometry family."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from twelve_six import model116_tokenizer_geometry as experiment
from twelve_six.model import ModelSpec
from twelve_six.scaling_experiment import controlled_specs
from twelve_six.vocabulary import rebalance_d_ff_for_vocabulary

RESEARCH41_HEAD = "9775a3432795dde9c96b3e84f6de143b2033a08c"


def incumbent_anchors() -> tuple[ModelSpec, ModelSpec, ModelSpec]:
    """Return the proven ~100K, ~500K and ~1M RESEARCH41 geometries."""
    family = controlled_specs()
    anchors = (family[0], family[2], family[3])
    counts = tuple(spec.parameter_count() for spec in anchors)
    expected = (95_568, 467_808, 1_037_696)
    if counts != expected:
        raise RuntimeError(f"MODEL-116 incumbent geometry drift: {counts!r} != {expected!r}")
    return anchors


SCALE_BUDGETS = tuple(spec.parameter_count() for spec in incumbent_anchors())


def _template_specs(_: Path) -> dict[int, ModelSpec]:
    return {spec.parameter_count(): spec for spec in incumbent_anchors()}


def _solve_not_above(spec: ModelSpec, *, cap: int, vocab_size: int) -> ModelSpec:
    allocation = rebalance_d_ff_for_vocabulary(
        spec,
        target_parameters=cap,
        vocab_size=vocab_size,
        d_ff_alignment=1,
    )
    solved = allocation.model
    if solved.parameter_count() > cap:
        if solved.d_ff <= 1:
            raise experiment.Model116Error("cannot lower d_ff while respecting monotone cap")
        solved = replace(solved, d_ff=solved.d_ff - 1)
    if solved.parameter_count() > cap:
        raise experiment.Model116Error("larger vocabulary received extra parameter capacity")
    return solved


def solve_incumbent_geometries(_: Path) -> dict[int, dict[int, ModelSpec]]:
    """Match incumbent budgets while making total capacity monotone with vocabulary.

    The first vocabulary at a scale uses MODEL37's nearest point to the exact
    RESEARCH41 budget and may lie slightly above or below it within the strict
    tolerance. Every larger vocabulary is then solved at or below the previous
    candidate's *actual* total, preventing any larger-vocabulary capacity bonus.
    """
    result: dict[int, dict[int, ModelSpec]] = {}
    for target, template in _template_specs(Path(".")).items():
        scale_rows: dict[int, ModelSpec] = {}
        previous_total: int | None = None
        for index, vocab_size in enumerate(experiment.VOCABULARIES):
            if index == 0:
                solved = rebalance_d_ff_for_vocabulary(
                    template,
                    target_parameters=target,
                    vocab_size=vocab_size,
                    d_ff_alignment=1,
                ).model
            else:
                if previous_total is None:
                    raise AssertionError("previous_total missing after first vocabulary")
                solved = _solve_not_above(
                    template,
                    cap=previous_total,
                    vocab_size=vocab_size,
                )
            total = solved.parameter_count()
            if previous_total is not None and total > previous_total:
                raise experiment.Model116Error(
                    "larger vocabulary received extra total parameter capacity"
                )
            if abs(total - target) / target > experiment.PARAMETER_TOLERANCE:
                raise experiment.Model116Error(
                    f"target={target} vocab={vocab_size} misses strict tolerance: "
                    f"{abs(total - target) / target:.6%}"
                )
            scale_rows[vocab_size] = solved
            previous_total = total
        result[target] = scale_rows
    return result


def install_incumbent_geometry() -> None:
    """Bind the generic matched experiment to exact current geometry incumbents."""
    experiment.SCALES = SCALE_BUDGETS
    experiment._template_specs = _template_specs
    experiment.solve_geometries = solve_incumbent_geometries


def main(argv: Sequence[str] | None = None) -> int:
    install_incumbent_geometry()
    return experiment.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
