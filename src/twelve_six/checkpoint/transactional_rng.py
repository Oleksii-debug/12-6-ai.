"""Transactional RNG restoration for checkpoint resume.

Checkpoint bytes can pass structural preflight and a later runtime/backend apply
can still fail unexpectedly.  A partial RNG restore would make a retry consume a
different random stream.  This wrapper snapshots the live RNG state and rolls it
back if the underlying restore raises.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any


def _snapshot_torch_policy() -> tuple[bool, bool] | None:
    """Capture process-global Torch deterministic policy outside RNG payload v1."""

    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return None
    return (
        bool(torch.are_deterministic_algorithms_enabled()),
        bool(torch.is_deterministic_algorithms_warn_only_enabled()),
    )


def _restore_torch_policy(policy: tuple[bool, bool] | None) -> None:
    if policy is None:
        return
    torch = importlib.import_module("torch")
    torch.use_deterministic_algorithms(policy[0], warn_only=policy[1])


def _transactional_restore(
    core: Any,
    original_restore: Callable[[Mapping[str, Any]], dict[str, Any]],
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore RNG state or reinstate the exact pre-call state on failure."""

    before = core.capture_rng_state()
    before_policy = _snapshot_torch_policy()
    try:
        return original_restore(state)
    except BaseException as exc:  # noqa: BLE001 - rollback must be interrupt-safe
        try:
            original_restore(before)
            # Checkpoint-v1 RNG payloads do not encode Torch warn-only mode.
            # Restore the exact ambient process policy alongside RNG rollback.
            _restore_torch_policy(before_policy)
        except BaseException as rollback_exc:  # noqa: BLE001
            if not isinstance(exc, Exception):
                exc.add_note(
                    "RNG rollback of the prior process state also failed: "
                    f"{rollback_exc!r}"
                )
                raise exc from rollback_exc
            raise core.CheckpointError(
                "RNG restore failed and rollback of the prior RNG state also failed"
            ) from rollback_exc
        if not isinstance(exc, Exception):
            # Preserve KeyboardInterrupt/SystemExit/GeneratorExit identity after
            # restoring the exact pre-call process state.
            raise
        if isinstance(exc, core.CheckpointCompatibilityError):
            # A fail-closed compatibility preflight can reject before mutating
            # anything. The transactional wrapper still proves rollback, but
            # must preserve the precise incompatibility for operator diagnosis.
            raise
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
