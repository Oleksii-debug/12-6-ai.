"""D05 rollback diagnostics must never replace the primary failure."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from twelve_six.checkpoint import core, transactional_rng


@pytest.mark.parametrize(
    "broken_notes",
    [False, True],
    ids=["valid-notes", "malformed-notes"],
)
def test_save_rollback_note_cannot_mask_primary_failure(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    broken_notes: bool,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    hostile_hook_calls: list[str] = []

    class HostileNoteError(RuntimeError):
        def add_note(self, note: str) -> None:
            hostile_hook_calls.append(note)
            raise RuntimeError("hostile add_note hook executed")

    primary = HostileNoteError("primary checkpoint save failure")
    if broken_notes:
        primary.__notes__ = "attacker-controlled non-list"

    class FailingModel:
        def state_dict(self) -> dict[str, Any]:
            raise primary

    def fail_rollback(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("forced checkpoint save RNG rollback failure")

    monkeypatch.setattr(core, "_restore_checkpoint_save_rng", fail_rollback)

    with pytest.raises(
        HostileNoteError,
        match="primary checkpoint save failure",
    ) as caught:
        core.save_checkpoint(
            checkpoint,
            model=FailingModel(),
            identity=object(),
        )

    assert caught.value is primary
    assert hostile_hook_calls == []
    if not broken_notes:
        assert any(
            "checkpoint save RNG rollback also failed" in note
            for note in getattr(caught.value, "__notes__", ())
        )
    assert not checkpoint.exists()


@pytest.mark.parametrize(
    "broken_notes",
    [False, True],
    ids=["valid-notes", "malformed-notes"],
)
def test_transactional_rng_rollback_note_cannot_mask_primary_interrupt(
    broken_notes: bool,
) -> None:
    hostile_hook_calls: list[str] = []

    class HostileInterrupt(BaseException):
        def add_note(self, note: str) -> None:
            hostile_hook_calls.append(note)
            raise RuntimeError("hostile add_note hook executed")

    primary = HostileInterrupt("primary RNG restore interrupt")
    if broken_notes:
        primary.__notes__ = "attacker-controlled non-list"

    calls = 0

    def failing_restore(state: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise primary
        raise RuntimeError("forced transactional RNG rollback failure")

    core_stub = SimpleNamespace(
        capture_rng_state=lambda: {"ambient": "state"},
    )

    with pytest.raises(HostileInterrupt) as caught:
        transactional_rng._transactional_restore(
            core_stub,
            failing_restore,
            {"replacement": "state"},
        )

    assert caught.value is primary
    assert calls == 2
    assert hostile_hook_calls == []
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "forced transactional RNG rollback failure"
    if not broken_notes:
        assert any(
            "RNG rollback of the prior process state also failed" in note
            for note in getattr(caught.value, "__notes__", ())
        )
