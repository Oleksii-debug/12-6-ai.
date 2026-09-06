"""Transactional RNG restoration for checkpoint resume.

Checkpoint bytes can pass structural preflight and a later runtime/backend apply
can still fail unexpectedly.  A partial RNG restore would make a retry consume a
different random stream.  This wrapper snapshots the live RNG state and rolls it
back if the underlying restore raises.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def _transactional_restore(
    core: Any,
    original_restore: Callable[[Mapping[str, Any]], dict[str, Any]],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore RNG state or reinstate the exact pre-call state on failure."""

    before = core.capture_rng_state()
    try:
        return original_restore(state)
    except Exception as exc:
        try:
            original_restore(before)
        except Exception as rollback_exc:
            raise core.CheckpointError(
                "RNG restore failed and rollback of the prior RNG state also failed"
            ) from rollback_exc
        raise core.CheckpointCompatibilityError(
            "RNG restore failed; prior RNG state was restored transactionally"
        ) from exc


def install(core: Any) -> None:
    """Install transactional rollback around the production RNG restore API."""

    if getattr(core, "_D05_TRANSACTIONAL_RNG_INSTALLED", False):
        return

    original_restore = core.restore_rng_state

    def restore_rng_state(state: Mapping[str, Any]) -> dict[str, Any]:
        return _transactional_restore(core, original_restore, state)

    core.restore_rng_state = restore_rng_state
    core._D05_TRANSACTIONAL_RNG_INSTALLED = True
