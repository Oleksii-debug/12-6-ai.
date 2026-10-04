"""Shared early validation of caller-provided checkpoint identity expectations."""

from __future__ import annotations

from . import core as _core

_HEX = frozenset("0123456789abcdef")


def _require_expected_sha256(value: str | None, *, field: str) -> None:
    if value is None:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise _core.CheckpointCompatibilityError(
            f"{field} must be an exact lowercase 64-hex SHA-256 or None"
        )


def _require_expected_nonempty_string(value: str | None, *, field: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise _core.CheckpointCompatibilityError(
            f"{field} must be a non-empty string or None"
        )


def _validate_expected_canonical_binding(
    *,
    expected_init_spec_hash: str | None,
    expected_split_identity: str | None,
    expected_packing_hash: str | None,
    expected_packing_version: str | None,
    expected_training_config_hash: str | None,
    expected_environment_lock_hash: str | None,
    expected_seed: int | None,
) -> None:
    """Reject malformed caller provenance before accepting equal malformed metadata."""

    for field, value in (
        ("expected_init_spec_hash", expected_init_spec_hash),
        ("expected_packing_hash", expected_packing_hash),
        ("expected_training_config_hash", expected_training_config_hash),
        ("expected_environment_lock_hash", expected_environment_lock_hash),
    ):
        _require_expected_sha256(value, field=field)
    _require_expected_nonempty_string(
        expected_split_identity,
        field="expected_split_identity",
    )
    _require_expected_nonempty_string(
        expected_packing_version,
        field="expected_packing_version",
    )
    if expected_seed is not None and (
        not isinstance(expected_seed, int)
        or isinstance(expected_seed, bool)
        or expected_seed < 0
    ):
        raise _core.CheckpointCompatibilityError(
            "expected_seed must be a non-negative integer or None"
        )


