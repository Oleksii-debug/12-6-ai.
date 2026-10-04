"""A preflight-only torch warn-only setter fault must preserve error and retry mode.

This is a non-owning negative regression for D05 PR #2628. A successful
RNG rollback may itself clear PyTorch warn_only before mode recovery fails.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
import torch

from twelve_six.checkpoint import core, trainer_adapter


class _FreshTarget:
    _failure_reason: str | None = None
    _update_incomplete: bool = False


@pytest.mark.parametrize("interruption", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("probe_rejects", [False, True])
@pytest.mark.parametrize("persistent_failure", [False, True])
def test_preflight_warn_only_failure_retries_policy_and_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
    probe_rejects: bool,
    persistent_failure: bool,
) -> None:
    ambient = core.capture_rng_state()
    before_enabled = torch.are_deterministic_algorithms_enabled()
    before_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_use = torch.use_deterministic_algorithms
    try:
        original_use(True, warn_only=True)
        target = _FreshTarget()
        original_error = interruption("primary preflight warn-only restoration failed")
        attempts = 0

        def faulting_use(enabled: bool, *, warn_only: bool = False) -> None:
            nonlocal attempts
            if enabled and warn_only:
                attempts += 1
                if attempts == 1:
                    raise original_error
                if persistent_failure:
                    raise OSError("secondary preflight warn-only rollback failed")
            original_use(enabled, warn_only=warn_only)

        def consuming_probe(*_args: Any, **_kwargs: Any) -> None:
            random.random()
            np.random.random_sample()
            torch.rand(())
            if probe_rejects:
                raise ValueError("original semantic probe rejection")

        with monkeypatch.context() as patch:
            patch.setattr(torch, "use_deterministic_algorithms", faulting_use)
            patch.setattr(
                trainer_adapter,
                "_preflight_trainer_state_without_rng_guard",
                consuming_probe,
            )
            with pytest.raises(interruption, match="primary preflight warn-only") as got:
                trainer_adapter._preflight_trainer_state(target, {"probe": True})

        assert got.value is original_error
        if probe_rejects:
            assert isinstance(got.value.__context__, ValueError)
            assert "semantic probe rejection" in str(got.value.__context__)
        assert attempts >= 2, "mode rollback must make a best-effort second attempt"
        assert target._failure_reason == "checkpoint_preflight_rng_rollback_failed"
        assert target._update_incomplete is True
        assert torch.are_deterministic_algorithms_enabled() is True
        if persistent_failure:
            assert any(
                "secondary preflight warn-only rollback failed" in note
                for note in getattr(got.value, "__notes__", ())
            )
        else:
            assert torch.is_deterministic_algorithms_warn_only_enabled() is True
    finally:
        core.restore_rng_state(ambient)
        original_use(before_enabled, warn_only=before_warn_only)
