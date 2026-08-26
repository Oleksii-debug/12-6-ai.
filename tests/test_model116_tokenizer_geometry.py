from pathlib import Path

import torch

from twelve_six.model116_tokenizer_geometry import (
    PARAMETER_TOLERANCE,
    SCALES,
    VOCABULARIES,
    _mask_labels_to_remaining,
    solve_geometries,
)


def test_geometry_solver_keeps_larger_vocab_at_or_below_prior_capacity():
    matrix = solve_geometries(Path("."))
    assert tuple(matrix) == SCALES
    for scale in SCALES:
        totals = [matrix[scale][vocab].parameter_count() for vocab in VOCABULARIES]
        assert totals == sorted(totals, reverse=True)
        assert all(abs(total - scale) / scale <= PARAMETER_TOLERANCE for total in totals)
        specs = [matrix[scale][vocab] for vocab in VOCABULARIES]
        assert len({spec.d_model for spec in specs}) == 1
        assert len({spec.n_layers for spec in specs}) == 1
        assert len({spec.n_heads for spec in specs}) == 1
        assert len({spec.n_kv_heads for spec in specs}) == 1
        assert len({spec.head_dim for spec in specs}) == 1
        assert [spec.d_ff for spec in specs] == sorted(
            [spec.d_ff for spec in specs], reverse=True
        )


def test_final_batch_mask_hits_exact_remaining_targets():
    labels = torch.tensor(
        [
            [1, 2, 3, 4, -100],
            [5, 6, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    masked = _mask_labels_to_remaining(labels, 4)
    assert int(masked[:, 1:].ne(-100).sum().item()) == 4
    assert torch.equal(masked[:, 0], labels[:, 0])


def test_geometry_parameter_breakdown_matches_total():
    matrix = solve_geometries(Path("."))
    for scale in SCALES:
        for vocab in VOCABULARIES:
            spec = matrix[scale][vocab]
            breakdown = spec.parameter_breakdown()
            assert breakdown["total"] == spec.parameter_count()
            assert breakdown["token_embedding"] == spec.vocab_size * spec.d_model
            assert breakdown["blocks_total"] > breakdown["token_embedding"]
