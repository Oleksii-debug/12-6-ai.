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


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_transactional_rng_interrupt_rolls_back_and_preserves_identity(
    interruption: type[BaseException],
) -> None:
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
    primary = interruption("simulated interrupted backend apply")

    def interrupt_after_partial_apply(state: Mapping[str, Any]) -> dict[str, Any]:
        if state is target:
            random.seed(999)
            np.random.seed(999)
            raise primary
        return real_restore(state)

    with pytest.raises(interruption, match="interrupted backend apply") as raised:
        _transactional_restore(
            __import__("twelve_six.checkpoint", fromlist=["checkpoint"]),
            interrupt_after_partial_apply,
            target,
        )

    assert raised.value is primary
    assert random.getstate() == before_python
    _assert_numpy_rng_equal(np.random.get_state(), before_numpy)


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_transactional_rng_rollback_failure_preserves_primary_interrupt(
    interruption: type[BaseException],
) -> None:
    class FakeCore:
        @staticmethod
        def capture_rng_state() -> dict[str, str]:
            return {"python": "before"}

    primary = interruption("primary interrupted restore")
    rollback_error = RuntimeError("secondary rollback failure")
    calls = 0

    def interrupt_then_fail(state: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal calls
        del state
        calls += 1
        if calls == 1:
            raise primary
        raise rollback_error

    with pytest.raises(interruption, match="primary interrupted restore") as raised:
        _transactional_restore(FakeCore, interrupt_then_fail, {"python": "target"})

    assert calls == 2
    assert raised.value is primary
    assert raised.value.__cause__ is rollback_error
    assert any(
        "secondary rollback failure" in note
        for note in getattr(raised.value, "__notes__", ())
    )

def test_transactional_rng_restore_rolls_back_torch_warn_only_policy() -> None:
    torch = pytest.importorskip("torch")
    ambient = capture_rng_state()
    old_policy = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.manual_seed(731)
        torch.use_deterministic_algorithms(True, warn_only=True)
        before_rng = torch.get_rng_state().clone()
        target = capture_rng_state()
        real_restore = restore_rng_state

        def fail_after_policy_drift(state: Mapping[str, Any]) -> dict[str, Any]:
            if state is target:
                torch.rand(1)
                torch.use_deterministic_algorithms(False, warn_only=False)
                raise RuntimeError("simulated Torch policy drift before restore failure")
            return real_restore(state)

        with pytest.raises(
            CheckpointCompatibilityError,
            match="restored transactionally",
        ):
            _transactional_restore(
                __import__("twelve_six.checkpoint", fromlist=["checkpoint"]),
                fail_after_policy_drift,
                target,
            )

        torch.testing.assert_close(torch.get_rng_state(), before_rng, rtol=0, atol=0)
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        restore_rng_state(ambient)
        torch.use_deterministic_algorithms(
            old_policy[0],
            warn_only=old_policy[1],
        )

def test_rng_state_roundtrip_restores_torch_warn_only_policy() -> None:
    torch = pytest.importorskip("torch")
    ambient = capture_rng_state()
    old_policy = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        state = capture_rng_state()
        assert state["torch"]["deterministic_warn_only"] is True

        torch.use_deterministic_algorithms(False, warn_only=False)
        restore_rng_state(state)

        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        restore_rng_state(ambient)
        torch.use_deterministic_algorithms(
            old_policy[0],
            warn_only=old_policy[1],
        )


def test_legacy_rng_state_preserves_live_torch_warn_only_policy() -> None:
    torch = pytest.importorskip("torch")
    ambient = capture_rng_state()
    old_policy = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        legacy = capture_rng_state()
        legacy = dict(legacy)
        legacy["torch"] = dict(legacy["torch"])
        legacy["torch"].pop("deterministic_warn_only")

        torch.use_deterministic_algorithms(True, warn_only=True)
        restore_rng_state(legacy)

        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        restore_rng_state(ambient)
        torch.use_deterministic_algorithms(
            old_policy[0],
            warn_only=old_policy[1],
        )
