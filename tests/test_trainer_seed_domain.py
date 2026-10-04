"""Ensure D02 run seeds are in the domain accepted by torch.manual_seed.

Configuration-only regression tests; no corpus, optimizer step, or training credit.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from twelve_six.training import TrainerConfig


@pytest.mark.parametrize("seed", [0, 1, 2 ** 32, 2 ** 64 - 1])
def test_nonnegative_torch_compatible_seed_is_accepted(seed: int) -> None:
    assert TrainerConfig(seed=seed).seed == seed


@pytest.mark.parametrize("seed", [2 ** 64, 2 ** 64 + 1, 2 ** 128])
def test_oversized_seed_fails_during_pure_config_validation(seed: int) -> None:
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="seed.*2\\*\\*64"):
        TrainerConfig(seed=seed)
    assert random.getstate() == python_before
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_before[0]
    np.testing.assert_array_equal(after_numpy[1], numpy_before[1])
    assert after_numpy[2:] == numpy_before[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_before, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "703", None])
def test_invalid_seed_types_and_negative_values_remain_rejected(seed: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        TrainerConfig(seed=seed)  # type: ignore[arg-type]


def test_highest_valid_seed_is_accepted_by_installed_torch() -> None:
    # The actual configured boundary must agree with torch.manual_seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(TrainerConfig(seed=2 ** 64 - 1).seed)
