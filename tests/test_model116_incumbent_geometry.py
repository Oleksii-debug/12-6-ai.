from pathlib import Path

from twelve_six import model116_tokenizer_geometry as experiment
from twelve_six.model116_tokenizer_geometry_incumbent import (
    SCALE_BUDGETS,
    incumbent_anchors,
    install_incumbent_geometry,
)


EXPECTED_MATCHED = {
    95_568: {
        320: (121, 95_616),
        384: (113, 95_232),
        437: (107, 95_184),
    },
    467_808: {
        320: (251, 468_192),
        384: (245, 467_424),
        437: (240, 466_752),
    },
    1_037_696: {
        320: (348, 1_038_208),
        384: (343, 1_036_800),
        437: (339, 1_035_904),
    },
}


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
        for vocab_size in experiment.VOCABULARIES:
            spec = matrix[scale][vocab_size]
            expected_d_ff, expected_total = EXPECTED_MATCHED[scale][vocab_size]
            assert spec.d_ff == expected_d_ff
            assert spec.parameter_count() == expected_total


def test_only_vocab_and_d_ff_change_inside_each_matched_scale():
    install_incumbent_geometry()
    matrix = experiment.solve_geometries(Path("."))
    ignored = {"vocab_size", "d_ff"}
    for scale in SCALE_BUDGETS:
        specs = [matrix[scale][v].to_dict() for v in experiment.VOCABULARIES]
        structural = [
            {key: value for key, value in spec.items() if key not in ignored}
            for spec in specs
        ]
        assert structural[0] == structural[1] == structural[2]
