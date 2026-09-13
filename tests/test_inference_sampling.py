import random

import pytest

from twelve_six.inference.sampling import greedy_token, sample_token


def test_greedy_uses_lowest_token_id_for_equal_maximum() -> None:
    assert greedy_token([1.0, 3.0, 3.0]) == 1


def test_top_k_one_is_deterministic_argmax() -> None:
    for seed in range(10):
        assert sample_token([0.0, 9.0, 1.0], rng=random.Random(seed), top_k=1) == 1


def test_sampling_seed_is_repeatable() -> None:
    first_rng = random.Random(42)
    second_rng = random.Random(42)
    first = [sample_token([0.0, 0.0, 0.0], rng=first_rng) for _ in range(20)]
    second = [sample_token([0.0, 0.0, 0.0], rng=second_rng) for _ in range(20)]
    assert first == second


def test_rejects_invalid_sampling_parameters() -> None:
    with pytest.raises(ValueError, match="temperature"):
        sample_token([0.0, 1.0], rng=random.Random(0), temperature=0)
    with pytest.raises(ValueError, match="top_p"):
        sample_token([0.0, 1.0], rng=random.Random(0), top_p=0)
