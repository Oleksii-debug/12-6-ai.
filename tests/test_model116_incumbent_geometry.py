from pathlib import Path

from twelve_six import model116_tokenizer_geometry as experiment
from twelve_six.model116_tokenizer_geometry_incumbent import (
    SCALE_BUDGETS,
    incumbent_anchors,
    install_incumbent_geometry,
)


def test_incumbent_anchors_are_exact_research41_scales():
    anchors = incumbent_anchors()
    assert SCALE_BUDGETS == (95_568, 467_808, 1_037_696)
    assert tuple(spec.parameter_count() for spec in anchors) == SCALE_BUDGETS
    assert tuple(spec.vocab_size for spec in anchors) == (256, 256, 256)
    assert tuple(spec.max_seq_len for spec in anchors) == (256, 256, 256)


def test_matched_solver_uses_incumbent_budgets_without_larger_vocab_capacity_bonus():
    install_incumbent_geometry()
    matrix = experiment.solve_geometries(Path("."))
    assert tuple(matrix) == SCALE_BUDGETS
    for scale in SCALE_BUDGETS:
        totals = [matrix[scale][v].parameter_count() for v in experiment.VOCABULARIES]
        assert totals == sorted(totals, reverse=True)
        assert all(abs(total - scale) / scale <= experiment.PARAMETER_TOLERANCE for total in totals)
        dffs = [matrix[scale][v].d_ff for v in experiment.VOCABULARIES]
        assert dffs == sorted(dffs, reverse=True)
