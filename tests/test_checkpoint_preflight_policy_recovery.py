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
    def __init__(self) -> None:
        self._failure_reason: str | None = None
        self._update_incomplete = False


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


@pytest.mark.parametrize("interruption", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("probe_rejects", [False, True])
@pytest.mark.parametrize("persistent_failure", [False, True])
def test_preflight_rng_and_policy_double_fault_recovers_mode(
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
    probe_rejects: bool,
    persistent_failure: bool,
) -> None:
    ambient = core.capture_rng_state()
    before_enabled = torch.are_deterministic_algorithms_enabled()
    before_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_use = torch.use_deterministic_algorithms
    original_restore = core.restore_rng_state
    try:
        original_use(True, warn_only=True)
        target = _FreshTarget()
        primary = interruption("primary preflight RNG rollback failed")
        mode_attempts = 0

        def interrupted_rng(state: Any) -> None:
            original_restore(state)
            raise primary

        def interrupted_policy(enabled: bool, *, warn_only: bool = False) -> None:
            nonlocal mode_attempts
            if enabled and warn_only:
                mode_attempts += 1
                if mode_attempts == 1:
                    raise RuntimeError("secondary preflight mode restore failed")
                if persistent_failure:
                    raise RuntimeError("tertiary preflight mode restore failed")
            original_use(enabled, warn_only=warn_only)

        def probe(*_args: Any, **_kwargs: Any) -> None:
            random.random()
            np.random.random_sample()
            torch.rand(())
            if probe_rejects:
                raise ValueError("semantic probe rejection")

        with monkeypatch.context() as patch:
            patch.setattr(core, "restore_rng_state", interrupted_rng)
            patch.setattr(torch, "use_deterministic_algorithms", interrupted_policy)
            patch.setattr(trainer_adapter, "_preflight_trainer_state_without_rng_guard", probe)
            with pytest.raises(interruption, match="primary preflight RNG") as got:
                trainer_adapter._preflight_trainer_state(target, {"probe": True})

        assert got.value is primary
        if probe_rejects:
            assert isinstance(got.value.__context__, ValueError)
            assert "semantic probe rejection" in str(got.value.__context__)
        assert target._update_incomplete is True
        assert target._failure_reason == "checkpoint_preflight_rng_rollback_failed"
        assert mode_attempts >= 2, "a secondary mode failure must trigger a retry"
        assert any(
            "secondary preflight mode restore failed" in note
            for note in getattr(primary, "__notes__", ())
        )
        assert torch.are_deterministic_algorithms_enabled() is True
        if persistent_failure:
            assert any(
                "tertiary preflight mode restore failed" in note
                for note in getattr(primary, "__notes__", ())
            )
        else:
            assert torch.is_deterministic_algorithms_warn_only_enabled() is True
    finally:
        original_restore(ambient)
        original_use(before_enabled, warn_only=before_warn_only)

@pytest.mark.parametrize("interruption", [OSError, KeyboardInterrupt, SystemExit])
def test_outer_preapply_rng_rollback_failure_poison_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    ambient = core.capture_rng_state()
    before_enabled = torch.are_deterministic_algorithms_enabled()
    before_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    original_restore = core.restore_rng_state
    original_use = torch.use_deterministic_algorithms
    target = _FreshTarget()
    primary = interruption("outer preapply RNG rollback interrupted")
    attempts = 0
    try:
        random.random()
        np.random.random_sample()
        torch.rand(())

        def fail_first_restore(state: Any) -> Any:
            nonlocal attempts
            attempts += 1
            result = original_restore(state)
            if attempts == 1:
                raise primary
            return result

        with monkeypatch.context() as patch:
            patch.setattr(trainer_adapter._core, "restore_rng_state", fail_first_restore)
            with pytest.raises(interruption, match="outer preapply RNG") as got:
                trainer_adapter._restore_preapply_process_state(
                    ambient,
                    (before_enabled, before_warn_only),
                    target,
                )

        assert got.value is primary
        assert attempts >= 2
        assert target._failure_reason == "checkpoint_preapply_rng_rollback_failed"
        assert target._update_incomplete is True
        assert random.getstate() == ambient["python"]
        np_after = np.random.get_state()
        np_before = ambient["numpy"]
        assert np_after[0] == np_before[0]
        np.testing.assert_array_equal(np_after[1], np_before[1])
        assert np_after[2:] == np_before[2:]
        torch_state = ambient["torch"]
        assert torch_state is not None
        torch.testing.assert_close(
            torch.get_rng_state(),
            torch_state["cpu"],
            rtol=0,
            atol=0,
        )
        assert (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
        ) == (before_enabled, before_warn_only)
    finally:
        original_restore(ambient)
        original_use(before_enabled, warn_only=before_warn_only)

@pytest.mark.parametrize(
    "marker", ["_failure_reason", "_update_incomplete"],
    ids=["failure-marker", "incomplete-marker"],
)
def test_preapply_rollback_marker_loss_still_poisoned(
    monkeypatch: pytest.MonkeyPatch,
    marker: str,
) -> None:
    ambient = core.capture_rng_state()
    policy = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    target = _FreshTarget()
    original_restore = core.restore_rng_state
    attempts = 0

    def fail_first_restore(state: Any) -> None:
        nonlocal attempts
        attempts += 1
        result = original_restore(state)
        if attempts == 1:
            del vars(target)[marker]
            raise RuntimeError("synthetic preapply rollback failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(
            trainer_adapter._core,
            "restore_rng_state",
            fail_first_restore,
        )
        with pytest.raises(
            RuntimeError,
            match="synthetic preapply rollback failure",
        ):
            trainer_adapter._restore_preapply_process_state(
                ambient,
                policy,
                target,
            )

    assert attempts >= 2
    assert vars(target)["_failure_reason"] == "checkpoint_preapply_rng_rollback_failed"
    assert vars(target)["_update_incomplete"] is True

def test_preflight_marker_loss_then_raise_preserves_primary_and_poison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = _FreshTarget()

    def fail_after_marker_loss(*_args: Any, **_kwargs: Any) -> None:
        del vars(target)["_failure_reason"]
        raise RuntimeError("synthetic semantic preflight failure")

    with monkeypatch.context() as patch:
        patch.setattr(
            trainer_adapter,
            "_preflight_trainer_state_without_rng_guard",
            fail_after_marker_loss,
        )
        with pytest.raises(
            RuntimeError,
            match="synthetic semantic preflight failure",
        ) as got:
            trainer_adapter._preflight_trainer_state(
                target,
                {"probe": True},
            )

    assert vars(target)["_failure_reason"] == "checkpoint_restore_target_drift"
    assert vars(target)["_update_incomplete"] is True
    assert any(
        "trainer restore target drift also detected" in note
        for note in getattr(got.value, "__notes__", ())
    )

def test_preapply_rollback_uses_entry_canonical_after_marker_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ambient = core.capture_rng_state()
    policy = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    target = _FreshTarget()
    del vars(target)["_failure_reason"]

    def fail_restore(_state: Any) -> None:
        raise RuntimeError("synthetic rollback failure after marker loss")

    with monkeypatch.context() as patch:
        patch.setattr(
            trainer_adapter._core,
            "restore_rng_state",
            fail_restore,
        )
        with pytest.raises(
            RuntimeError,
            match="synthetic rollback failure after marker loss",
        ):
            trainer_adapter._restore_preapply_process_state(
                ambient,
                policy,
                target,
                expected_canonical=True,
            )

    assert vars(target)["_failure_reason"] == "checkpoint_preapply_rng_rollback_failed"
    assert vars(target)["_update_incomplete"] is True

