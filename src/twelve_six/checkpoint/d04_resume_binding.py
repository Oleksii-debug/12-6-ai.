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
_D04_RESUME_MARKERS = frozenset({"resume_binding_schema", *D04_RESUME_SHA_FIELDS})
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
    existing_schema = bound_data.get("resume_binding_schema")
    if existing_schema is not None and existing_schema != D04_RESUME_BINDING_SCHEMA:
        raise CheckpointCompatibilityError(
            "checkpoint training_config.data.resume_binding_schema conflicts with D04 handoff"
        )
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
    """Reject wrong or latent-malformed D04 resume provenance before mutation.

    Legacy checkpoints that contain no D04 resume markers remain compatible when
    the caller does not request D04 expectations. Once any D04 marker is present,
    however, the checkpoint claims this v1 contract and must carry the complete
    exact binding even when the caller only asks for a generic resume.
    """

    expected = {
        "ledger_identity_sha256": expected_ledger_identity_sha256,
        "materialization_identity_sha256": expected_materialization_identity_sha256,
        "packing_identity_sha256": expected_packing_identity_sha256,
        "exposure_plan_identity_sha256": expected_exposure_plan_identity_sha256,
        "ordered_next_exposure_identity_sha256": (
            expected_ordered_next_exposure_identity_sha256
        ),
    }
    has_expectations = any(value is not None for value in expected.values())
    for field, value in expected.items():
        if value is not None:
            _require_sha256(value, field=f"expected_{field}")

    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        if has_expectations:
            raise CheckpointCompatibilityError("verified checkpoint identity is missing")
        return
    training_config = identity.get("training_config")
    if not isinstance(training_config, Mapping):
        if has_expectations:
            raise CheckpointCompatibilityError("verified checkpoint training_config is missing")
        return
    data = training_config.get("data")
    if not isinstance(data, Mapping):
        if has_expectations:
            raise CheckpointCompatibilityError("verified checkpoint D04 data binding is missing")
        return

    present_markers = _D04_RESUME_MARKERS.intersection(data)
    if not has_expectations and not present_markers:
        return
    missing_markers = sorted(_D04_RESUME_MARKERS - set(data))
    if missing_markers:
        raise CheckpointCompatibilityError(
            f"verified checkpoint D04 resume binding is incomplete: missing={missing_markers}"
        )
    if data.get("resume_binding_schema") != D04_RESUME_BINDING_SCHEMA:
        raise CheckpointCompatibilityError("verified checkpoint D04 resume binding schema mismatch")

    actual = {
        field: _require_sha256(
            data.get(field),
            field=f"checkpoint.data.{field}",
        )
        for field in D04_RESUME_SHA_FIELDS
    }
    mismatches = {
        field: {"expected": value, "actual": actual[field]}
        for field, value in expected.items()
        if value is not None and actual[field] != value
    }
    if mismatches:
        raise CheckpointCompatibilityError(
            f"checkpoint D04 resume binding mismatch: {mismatches}"
        )
