"""MODEL-116 execution binding to the proven RESEARCH41 controlled geometry family."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from twelve_six import model116_tokenizer_geometry as experiment
from twelve_six.model import ModelSpec
from twelve_six.scaling_experiment import controlled_specs

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


def install_incumbent_geometry() -> None:
    """Bind the generic matched experiment to exact current geometry incumbents."""
    experiment.SCALES = SCALE_BUDGETS
    experiment._template_specs = _template_specs


def main(argv: Sequence[str] | None = None) -> int:
    install_incumbent_geometry()
    return experiment.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
