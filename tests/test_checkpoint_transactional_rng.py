from __future__ import annotations

import copy
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointError,
    capture_rng_state,
    restore_rng_state,
)
from twelve_six.checkpoint.transactional_rng import _transactional_restore


def _assert_numpy_rng_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> None:
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_transactional_rng_restore_rolls_back_late_apply_failure() -> None:
    random.seed(123)
    np.random.seed(123)
    before_python = copy.deepcopy(random.getstate())
    before_numpy = copy.deepcopy(np.random.get_state())

    random.seed(777)
    np.random.seed(777)
    target = capture_rng_state()
    random.setstate(before_python)
    np.random.set_state(before_numpy)

    real_restore = restore_rng_state

    def fail_target_then_restore_before(state: Mapping[str, Any]) -> dict[str, Any]:
        if state is target:
            random.seed(999)
            np.random.seed(999)
            raise RuntimeError("simulated late backend apply failure")
        return real_restore(state)

    with pytest.raises(CheckpointCompatibilityError, match="restored transactionally"):
        _transactional_restore(
            __import__("twelve_six.checkpoint", fromlist=["checkpoint"]),
            fail_target_then_restore_before,
            target,
        )

    assert random.getstate() == before_python
    _assert_numpy_rng_equal(np.random.get_state(), before_numpy)


def test_transactional_rng_restore_surfaces_rollback_failure() -> None:
    target = capture_rng_state()

    class FakeCore:
        CheckpointError = CheckpointError
        CheckpointCompatibilityError = CheckpointCompatibilityError

        @staticmethod
        def capture_rng_state() -> dict[str, Any]:
            return {"python": "before"}

    def always_fail(state: Mapping[str, Any]) -> dict[str, Any]:
        del state
        raise RuntimeError("backend unavailable")

    with pytest.raises(CheckpointError, match="rollback.*also failed"):
        _transactional_restore(FakeCore, always_fail, target)
