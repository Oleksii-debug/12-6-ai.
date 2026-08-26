"""Fail-closed resume binding for checkpoint progress counters.

A checksum-valid checkpoint may still carry a consistently rebound positive
``step`` or ``tokens_seen`` value.  Callers that know the intended resume
position can bind those counters explicitly so a stale or relabelled checkpoint
is rejected before any model, optimizer, scheduler, or RNG mutation.
"""

from __future__ import annotations

from typing import Any


def _validate_expected_counter(core: Any, name: str, expected: int | None) -> None:
    if expected is None:
        return
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
        raise core.CheckpointCompatibilityError(
            f"expected_{name} must be a non-negative integer or None"
        )


def _assert_progress(core: Any, manifest: Any, *, expected_step: int | None, expected_tokens_seen: int | None) -> None:
    _validate_expected_counter(core, "step", expected_step)
    _validate_expected_counter(core, "tokens_seen", expected_tokens_seen)

    identity = manifest.get("identity") if hasattr(manifest, "get") else None
    if not hasattr(identity, "get"):
        raise core.CheckpointIntegrityError("manifest identity must be a mapping")

    expected = {
        "step": expected_step,
        "tokens_seen": expected_tokens_seen,
    }
    mismatches = {
        name: {"expected": value, "actual": identity.get(name)}
        for name, value in expected.items()
        if value is not None and identity.get(name) != value
    }
    if mismatches:
        raise core.CheckpointCompatibilityError(
            f"checkpoint progress mismatch: {mismatches}"
        )


def install(core: Any) -> None:
    """Extend checkpoint load APIs with exact progress-counter binding."""

    if getattr(core, "_D05_PROGRESS_BINDING_INSTALLED", False):
        return

    original_load_verified_checkpoint = core.load_verified_checkpoint

    def load_verified_checkpoint(
        verified: Any,
        *,
        expected_step: int | None = None,
        expected_tokens_seen: int | None = None,
        **kwargs: Any,
    ) -> Any:
        _assert_progress(
            core,
            verified.manifest,
            expected_step=expected_step,
            expected_tokens_seen=expected_tokens_seen,
        )
        return original_load_verified_checkpoint(verified, **kwargs)

    def load_checkpoint(
        directory: Any,
        *,
        expected_step: int | None = None,
        expected_tokens_seen: int | None = None,
        **kwargs: Any,
    ) -> Any:
        verified = core.prepare_checkpoint_load(directory)
        return core.load_verified_checkpoint(
            verified,
            expected_step=expected_step,
            expected_tokens_seen=expected_tokens_seen,
            **kwargs,
        )

    core.load_verified_checkpoint = load_verified_checkpoint
    core.load_checkpoint = load_checkpoint
    core._D05_PROGRESS_BINDING_INSTALLED = True
