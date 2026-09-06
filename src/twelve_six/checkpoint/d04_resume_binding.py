"""Fail-closed bridge from D04 ordered exposure state to D05 checkpoints.

D04 owns the ledger, replay guard, deterministic worker/shard plan, and the
order-sensitive next-exposure identity. D05 does not recreate those semantics.
It only binds the exact D04 identities into the already hash-protected checkpoint
training identity and verifies caller expectations before any live resume
mutation.

The D04 ordered-next-exposure identity transitively includes the current
self-hashed exposure state identity, ledger/materialization/packing identities,
claim sequence, ordered claim intervals, target count, plan identity, batch,
shard, and worker. Keeping that single terminal identity here avoids inventing a
second dataloader cursor format in D05.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from .core import CheckpointCompatibilityError, CheckpointIdentity, hash_json

D04_RESUME_BINDING_SCHEMA = "12-6.d04-checkpoint-resume-binding.v1"
D04_RESUME_SHA_FIELDS = (
    "ledger_identity_sha256",
    "materialization_identity_sha256",
    "packing_identity_sha256",
    "exposure_plan_identity_sha256",
    "ordered_next_exposure_identity_sha256",
)
_HEX = frozenset("0123456789abcdef")


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise CheckpointCompatibilityError(
            f"{field} must be an exact lowercase 64-hex SHA-256"
        )
    return value


def bind_d04_resume_identity(
    identity: CheckpointIdentity,
    *,
    run_manifest: Mapping[str, Any],
) -> CheckpointIdentity:
    """Add the exact D04 resume handoff to one checkpoint identity.

    The complete run manifest must already be the one transitively bound by the
    supplied identity. All D04 fields are required together so a production
    checkpoint cannot accidentally bind only a ledger while leaving the exact
    ordered next exposure unresolved.
    """

    if not isinstance(run_manifest, Mapping):
        raise CheckpointCompatibilityError("D04 resume run_manifest must be a mapping")
    if hash_json(run_manifest) != identity.run_manifest_hash:
        raise CheckpointCompatibilityError(
            "D04 resume run manifest does not match checkpoint run_manifest_hash"
        )
    manifest_data = run_manifest.get("data")
    if not isinstance(manifest_data, Mapping):
        raise CheckpointCompatibilityError("D04 resume run manifest data must be a mapping")

    binding = {
        field: _require_sha256(manifest_data.get(field), field=f"data.{field}")
        for field in D04_RESUME_SHA_FIELDS
    }

    training_config = deepcopy(dict(identity.training_config))
    bound_data = training_config.get("data")
    if not isinstance(bound_data, Mapping):
        raise CheckpointCompatibilityError(
            "checkpoint training_config.data is required for D04 resume binding"
        )
    bound_data = deepcopy(dict(bound_data))
    for field, value in binding.items():
        existing = bound_data.get(field)
        if existing is not None and existing != value:
            raise CheckpointCompatibilityError(
                f"checkpoint training_config.data.{field} conflicts with D04 handoff"
            )
        bound_data[field] = value
    bound_data["resume_binding_schema"] = D04_RESUME_BINDING_SCHEMA
    training_config["data"] = bound_data
    return replace(identity, training_config=training_config)


def assert_d04_resume_binding(
    manifest: Mapping[str, Any],
    *,
    expected_ledger_identity_sha256: str | None = None,
    expected_materialization_identity_sha256: str | None = None,
    expected_packing_identity_sha256: str | None = None,
    expected_exposure_plan_identity_sha256: str | None = None,
    expected_ordered_next_exposure_identity_sha256: str | None = None,
) -> None:
    """Reject a wrong D04 resume handoff before model/trainer/RNG mutation."""

    expected = {
        "ledger_identity_sha256": expected_ledger_identity_sha256,
        "materialization_identity_sha256": expected_materialization_identity_sha256,
        "packing_identity_sha256": expected_packing_identity_sha256,
        "exposure_plan_identity_sha256": expected_exposure_plan_identity_sha256,
        "ordered_next_exposure_identity_sha256": (
            expected_ordered_next_exposure_identity_sha256
        ),
    }
    if all(value is None for value in expected.values()):
        return
    for field, value in expected.items():
        if value is not None:
            _require_sha256(value, field=f"expected_{field}")

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise CheckpointCompatibilityError("verified checkpoint identity is missing")
    training_config = identity.get("training_config")
    if not isinstance(training_config, Mapping):
        raise CheckpointCompatibilityError("verified checkpoint training_config is missing")
    data = training_config.get("data")
    if not isinstance(data, Mapping):
        raise CheckpointCompatibilityError("verified checkpoint D04 data binding is missing")
    if data.get("resume_binding_schema") != D04_RESUME_BINDING_SCHEMA:
        raise CheckpointCompatibilityError("verified checkpoint D04 resume binding schema mismatch")

    mismatches = {
        field: {"expected": value, "actual": data.get(field)}
        for field, value in expected.items()
        if value is not None and data.get(field) != value
    }
    if mismatches:
        raise CheckpointCompatibilityError(
            f"checkpoint D04 resume binding mismatch: {mismatches}"
        )
