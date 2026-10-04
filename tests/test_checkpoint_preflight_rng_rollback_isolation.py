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
    _failure_reason: str | None = None
    _update_incomplete: bool = False


@pytest.mark.parametrize("failed_family", ["python", "numpy"])
@pytest.mark.parametrize("probe_rejects", [False, True])
def test_failed_preflight_rollback_recovers_other_rng_families(
    monkeypatch: pytest.MonkeyPatch,
    failed_family: str,
    probe_rejects: bool,
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
        original_error = OSError(f"injected {failed_family} RNG setter failure")
        setter_calls = 0

        def one_shot_failure(state: Any) -> None:
            nonlocal setter_calls
            setter_calls += 1
            if setter_calls == 1:
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
            else:
                original_setter = np.random.set_state
                patch.setattr(np.random, "set_state", one_shot_failure)
            with pytest.raises(OSError, match="injected .* RNG setter failure") as raised:
                trainer_adapter._preflight_trainer_state(target, {"probe": True})

        assert raised.value is original_error
        if probe_rejects:
            assert isinstance(raised.value.__context__, ValueError)
            assert "semantic preflight rejection" in str(raised.value.__context__)
        assert setter_calls >= 1
        assert target._failure_reason == "checkpoint_preflight_rng_rollback_failed"
        assert target._update_incomplete is True
        assert torch.are_deterministic_algorithms_enabled() == initial_enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() == initial_warn_only
        actual = (random.random(), np.random.random_sample(), torch.rand(()).item())
        assert actual == expected, (
            "preflight rollback must independently restore Python, NumPy and "
            "torch CPU streams even when one setter initially fails"
        )
    finally:
        core.restore_rng_state(initial)
        torch.use_deterministic_algorithms(
            initial_enabled, warn_only=initial_warn_only,
        )
