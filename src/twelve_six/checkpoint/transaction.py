"""Versioned generation and exposure metadata inside a checkpoint-v1 identity."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from .core import CheckpointIdentity, CheckpointIntegrityError, _require_exact_hex, hash_json

TRANSACTION_KEY = "checkpoint_transaction"
TRANSACTION_SCHEMA = "12-6.checkpoint-transaction.v1"
_FIELDS = {
    "schema",
    "generation",
    "parent_generation",
    "parent_checkpoint_id",
    "training_config_sha256",
    "exposure_cursor",
    "exposure_cursor_sha256",
}


@dataclass(frozen=True, slots=True)
class CheckpointTransaction:
    """A publication generation, which may advance less often than optimizer steps."""

    generation: int
    parent_generation: int | None
    parent_checkpoint_id: str | None
    training_config_sha256: str
    exposure_cursor: dict[str, Any]
    exposure_cursor_sha256: str


def exposure_cursor_schema(cursor: Mapping[str, Any]) -> str:
    if not isinstance(cursor, Mapping):
        raise CheckpointIntegrityError("checkpoint transaction requires an exposure cursor object")
    schema = cursor.get("schema", cursor.get("schema_version"))
    if not isinstance(schema, str) or not schema.strip() or schema.strip() != schema:
        raise CheckpointIntegrityError("checkpoint exposure cursor requires a canonical schema")
    return schema


def _parse_transaction(
    config: Mapping[str, Any],
    *,
    step: int,
) -> CheckpointTransaction | None:
    if TRANSACTION_KEY not in config:
        return None
    value = config[TRANSACTION_KEY]
    if (
        not isinstance(value, dict)
        or value.keys() != _FIELDS
        or value["schema"] != TRANSACTION_SCHEMA
    ):
        raise CheckpointIntegrityError("invalid checkpoint transaction schema")
    generation = value["generation"]
    parent_generation = value["parent_generation"]
    parent_id = value["parent_checkpoint_id"]
    if type(generation) is not int or type(step) is not int or not 0 <= generation <= step:
        raise CheckpointIntegrityError("invalid checkpoint transaction generation")
    if generation == 0:
        if parent_generation is not None or parent_id is not None:
            raise CheckpointIntegrityError("initial checkpoint must have no transaction parent")
    else:
        if type(parent_generation) is not int or parent_generation != generation - 1:
            raise CheckpointIntegrityError(
                "checkpoint transaction parent generation must precede it"
            )
        try:
            _require_exact_hex(parent_id, field="parent_checkpoint_id", lengths={64})
        except ValueError as exc:
            raise CheckpointIntegrityError(str(exc)) from exc
    base_config = {key: item for key, item in config.items() if key != TRANSACTION_KEY}
    if not base_config or hash_json(base_config) != value["training_config_sha256"]:
        raise CheckpointIntegrityError("checkpoint transaction training config digest mismatch")
    cursor = value["exposure_cursor"]
    if not isinstance(cursor, dict):
        raise CheckpointIntegrityError("checkpoint transaction requires an exposure cursor object")
    exposure_cursor_schema(cursor)
    if hash_json(cursor) != value["exposure_cursor_sha256"]:
        raise CheckpointIntegrityError("checkpoint transaction exposure cursor digest mismatch")
    return CheckpointTransaction(**{key: item for key, item in value.items() if key != "schema"})


def bind_checkpoint_transaction(
    identity: CheckpointIdentity,
    *,
    generation: int,
    parent_generation: int | None,
    parent_checkpoint_id: str | None,
    exposure_cursor: Mapping[str, Any],
) -> CheckpointIdentity:
    """Bind a JSON exposure cursor and parent generation without changing run identity."""
    config = dict(identity.training_config)
    if TRANSACTION_KEY in config:
        raise CheckpointIntegrityError("checkpoint transaction is already bound")
    if not isinstance(exposure_cursor, Mapping):
        raise CheckpointIntegrityError("checkpoint exposure cursor must be a JSON object")
    raw = dict(exposure_cursor)
    try:
        cursor = json.loads(json.dumps(raw, sort_keys=True, allow_nan=False))
    except (ValueError, TypeError, RecursionError) as exc:
        raise CheckpointIntegrityError("checkpoint exposure cursor must be canonical JSON") from exc
    if cursor != raw:
        raise CheckpointIntegrityError("checkpoint exposure cursor must round-trip as JSON")
    config[TRANSACTION_KEY] = {
        "schema": TRANSACTION_SCHEMA,
        "generation": generation,
        "parent_generation": parent_generation,
        "parent_checkpoint_id": parent_checkpoint_id,
        "training_config_sha256": hash_json(identity.training_config),
        "exposure_cursor": cursor,
        "exposure_cursor_sha256": hash_json(cursor),
    }
    _parse_transaction(config, step=identity.step)
    return replace(identity, training_config=config)


def checkpoint_transaction(manifest: Mapping[str, Any]) -> CheckpointTransaction | None:
    """Read validated transaction metadata from a verified checkpoint manifest."""
    identity = manifest.get("identity")
    config = identity.get("training_config") if isinstance(identity, Mapping) else None
    return (
        _parse_transaction(config, step=identity.get("step"))
        if isinstance(config, Mapping)
        else None
    )
