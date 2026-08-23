from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six import (
    DenseScalingTemplate,
    load_stage_config,
    solve_dense_scaling_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "expected", "identity"),
    [
        (
            "s4_100m.candidate.json",
            100_384_512,
            "dc9fd9e605cbc007aa20ad29f2220c7ebade875564d68016e93d1dc2489cd693",
        ),
        (
            "s5_400m.candidate.json",
            400_598_016,
            "9abfb6d1ac2e9c28fac20aff4ae804ad54b4102ce6f1bdeeadddf5a56027f28c",
        ),
        (
            "s6_1b.candidate.json",
            999_106_560,
            "cc64cbe94a461c364f063652098e55bdcf640d4be756ee1e743a23dda3de7261",
        ),
        (
            "s7_3b.candidate.json",
            2_998_029_312,
            "1d7145ff738e61b730e918126748050d289161f5051948e99a22aa15c20873d5",
        ),
    ],
)
def test_scaling_candidate_configs_have_exact_formula_and_identity(
    filename: str,
    expected: int,
    identity: str,
) -> None:
    path = ROOT / "configs" / "stages" / filename
    stage = load_stage_config(path)
    assert stage.expected_parameters == expected
    assert stage.model.parameter_count() == expected
    assert stage.model.identity_sha256() == identity

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "engineering_candidate_not_frozen"
    assert payload["promotion_allowed"] is False
    assert payload["requires_preceding_stage_pass"] is True


def test_solver_reproduces_s4_nearest_candidate() -> None:
    template = DenseScalingTemplate(
        vocab_size=32_768,
        max_seq_len=2_048,
        d_model=768,
        n_layers=10,
        n_heads=12,
        n_kv_heads=12,
        head_dim=64,
        d_ff_multiple=64,
    )
    candidates = solve_dense_scaling_candidates(100_000_000, (template,))
    assert candidates
    best = candidates[0]
    assert best.spec.d_ff == 2_240
    assert best.exact_parameters == 100_384_512
    assert best.parameter_delta == 384_512
    assert best.relative_error < 0.004


def test_solver_reproduces_gqa_s5_candidate() -> None:
    template = DenseScalingTemplate(
        vocab_size=32_768,
        max_seq_len=4_096,
        d_model=1_024,
        n_layers=20,
        n_heads=16,
        n_kv_heads=4,
        head_dim=64,
        d_ff_multiple=64,
    )
    best = solve_dense_scaling_candidates(400_000_000, (template,))[0]
    assert best.spec.d_ff == 5_120
    assert best.spec.q_dim == 1_024
    assert best.spec.kv_dim == 256
    assert best.exact_parameters == 400_598_016


def test_solver_is_deterministic_across_template_order() -> None:
    first = DenseScalingTemplate(
        vocab_size=32_768,
        max_seq_len=4_096,
        d_model=1_024,
        n_layers=20,
        n_heads=16,
        n_kv_heads=4,
        head_dim=64,
    )
    second = DenseScalingTemplate(
        vocab_size=32_768,
        max_seq_len=4_096,
        d_model=1_280,
        n_layers=20,
        n_heads=20,
        n_kv_heads=5,
        head_dim=64,
    )
    left = solve_dense_scaling_candidates(400_000_000, (first, second))
    right = solve_dense_scaling_candidates(400_000_000, (second, first))
    assert [item.to_dict() for item in left] == [item.to_dict() for item in right]


def test_solver_rejects_invalid_search_contracts() -> None:
    template = DenseScalingTemplate(
        vocab_size=256,
        max_seq_len=128,
        d_model=20,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        head_dim=10,
        d_ff_multiple=8,
    )
    with pytest.raises(ValueError):
        solve_dense_scaling_candidates(0, (template,))
    with pytest.raises(ValueError):
        solve_dense_scaling_candidates(10_000, ())
    with pytest.raises(ValueError):
        solve_dense_scaling_candidates(10_000, (template,), max_results=0)
    with pytest.raises(ValueError):
        solve_dense_scaling_candidates(10_000, (template,), max_relative_error=1.0)


def test_search_does_not_require_model_instantiation() -> None:
    template = DenseScalingTemplate(
        vocab_size=32_768,
        max_seq_len=8_192,
        d_model=3_072,
        n_layers=24,
        n_heads=24,
        n_kv_heads=8,
        head_dim=128,
    )
    best = solve_dense_scaling_candidates(3_000_000_000, (template,))[0]
    assert best.spec.d_ff == 10_368
    assert best.exact_parameters == 2_998_029_312
    assert best.relative_error < 0.001
