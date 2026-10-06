"""Fail-closed per-family RNG recovery when D05 semantic preflight aborts.

These independent negative tests pin an unfixed preflight rollback defect in
PR #2628. Unlike the separate failed-application rollback work, this suite
only exercises detached semantic probe isolation. No real training occurs.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import core, trainer_adapter


class _FreshCanonicalTarget:
    def __init__(self) -> None:
        # Recovery markers are instance-owned in the hardened D02 contract.
        self._failure_reason: str | None = None
        self._update_incomplete = False


@pytest.mark.parametrize("failed_family", ["python", "numpy", "torch_cpu"])
@pytest.mark.parametrize("probe_rejects", [False, True])
@pytest.mark.parametrize("persistent_failure", [False, True])
@pytest.mark.parametrize("interruption", [OSError, KeyboardInterrupt, SystemExit])
def test_failed_preflight_rollback_recovers_other_rng_families(
    monkeypatch: pytest.MonkeyPatch,
    failed_family: str,
    probe_rejects: bool,
    persistent_failure: bool,
    interruption: type[BaseException],
) -> None:
    """A failed RNG setter must not strand otherwise recoverable streams."""

    initial = core.capture_rng_state()
    initial_enabled = torch.are_deterministic_algorithms_enabled()
    initial_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        expected_python = random.Random()
        expected_python.setstate(initial["python"])
        expected_numpy = np.random.RandomState()
        expected_numpy.set_state(initial["numpy"])
        expected_torch = torch.Generator(device="cpu")
        expected_torch.set_state(initial["torch"]["cpu"])
        expected = (
            expected_python.random(),
            expected_numpy.random_sample(),
            torch.rand((), generator=expected_torch).item(),
        )
        target = _FreshCanonicalTarget()
        original_error = interruption(f"injected {failed_family} RNG setter failure")
        setter_calls = 0

        def one_shot_failure(state: Any) -> None:
            nonlocal setter_calls
            setter_calls += 1
            if setter_calls == 1 or persistent_failure:
                raise original_error
            original_setter(state)

        def consuming_probe(*_args: Any, **_kwargs: Any) -> None:
            random.random()
            np.random.random_sample()
            torch.rand(())
            if probe_rejects:
                raise ValueError("injected semantic preflight rejection")

        with monkeypatch.context() as patch:
            patch.setattr(
                trainer_adapter,
                "_preflight_trainer_state_without_rng_guard",
                consuming_probe,
            )
            if failed_family == "python":
                original_setter = random.setstate
                patch.setattr(random, "setstate", one_shot_failure)
            elif failed_family == "numpy":
                original_setter = np.random.set_state
                patch.setattr(np.random, "set_state", one_shot_failure)
            else:
                original_setter = torch.set_rng_state
                patch.setattr(torch, "set_rng_state", one_shot_failure)
            expected_exception: type[BaseException]
            if issubclass(interruption, Exception):
                expected_exception = (
                    core.CheckpointError
                    if persistent_failure
                    else core.CheckpointCompatibilityError
                )
            else:
                expected_exception = interruption
            with pytest.raises(expected_exception) as raised:
                trainer_adapter._preflight_trainer_state(target, {"probe": True})

        if issubclass(interruption, Exception):
            # Production RNG restore is transactionally wrapped. Ordinary
            # backend exceptions are normalized after rollback instead of
            # leaking the raw setter failure as the public checkpoint error.
            assert raised.value is not original_error
            current: BaseException | None = raised.value
            seen: set[int] = set()
            found_original = False
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                if current is original_error:
                    found_original = True
                    break
                current = current.__cause__ or current.__context__
            assert found_original
        else:
            # KeyboardInterrupt/SystemExit are deliberately not normalized by
            # transactional_rng; the outer preflight still restores state and
            # poisons the canonical target before re-raising the same object.
            assert raised.value is original_error
        if probe_rejects:
            assert isinstance(original_error.__context__, ValueError)
            assert "semantic preflight rejection" in str(original_error.__context__)
        assert setter_calls >= 1
        assert target._failure_reason == "checkpoint_preflight_rng_rollback_failed"
        assert target._update_incomplete is True
        assert torch.are_deterministic_algorithms_enabled() == initial_enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() == initial_warn_only
        actual = (random.random(), np.random.random_sample(), torch.rand(()).item())
        if persistent_failure:
            # The permanently broken setter cannot restore its own stream.
            # It must not prevent *other* families from recovering exactly.
            if failed_family == "python":
                assert actual[1:] == expected[1:]
            elif failed_family == "numpy":
                assert actual[0] == expected[0]
                assert actual[2] == expected[2]
            else:
                assert actual[:2] == expected[:2]
            assert any(
                "rollback" in note.lower()
                for note in getattr(raised.value, "__notes__", ())
            )
        else:
            assert actual == expected, (
                "preflight rollback must independently restore all three "
                "streams when the initial setter fault is transient"
            )
    finally:
        core.restore_rng_state(initial)
        torch.use_deterministic_algorithms(
            initial_enabled, warn_only=initial_warn_only,
        )
