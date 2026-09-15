"""Fail-closed schema validation for checkpoint-v1 provenance metadata."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

_MANIFEST_KEYS = frozenset(
    {
        "format",
        "format_version",
        "created_at_utc",
        "checkpoint_id",
        "identity",
        "files",
        "serialization",
    }
)
_IDENTITY_KEYS = frozenset(
    {
        "git_sha",
        "model_spec",
        "model_spec_hash",
        "parameter_count",
        "tokenizer_hash",
        "tokenizer_vocab_hash",
        "dataset_manifest_hash",
        "run_manifest_hash",
        "training_config",
        "training_config_hash",
        "seed",
        "optimizer",
        "optimizer_hash",
        "scheduler",
        "scheduler_hash",
        "precision",
        "step",
        "tokens_seen",
        "environment",
        "environment_hash",
        "environment_lock_hash",
    }
)
_FILE_RECORD_KEYS = frozenset({"sha256", "bytes"})
_SERIALIZATION = {
    "weights": "safetensors",
    "state_tensors": "safetensors",
    "state_tree": "canonical-json",
    "pickle": False,
}
_ENVIRONMENT_KEYS = frozenset(
    {
        "python",
        "implementation",
        "platform",
        "machine",
        "packages",
    }
)
_ENVIRONMENT_PACKAGE_KEYS = frozenset({"numpy", "safetensors", "torch"})


def _require_exact_keys(
    value: Any,
    expected: frozenset[str],
    *,
    label: str,
    error_type: type[Exception],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{label} must be a mapping")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise error_type(
            f"{label} fields differ: missing={missing}, unexpected={unexpected}"
        )
    return value


def _validate_environment(
    value: Any,
    *,
    error_type: type[Exception],
) -> None:
    environment = _require_exact_keys(
        value,
        _ENVIRONMENT_KEYS,
        label="identity.environment",
        error_type=error_type,
    )
    for field in ("python", "implementation", "platform", "machine"):
        item = environment.get(field)
        if not isinstance(item, str) or not item:
            raise error_type(f"identity.environment.{field} must be a non-empty string")
    packages = _require_exact_keys(
        environment.get("packages"),
        _ENVIRONMENT_PACKAGE_KEYS,
        label="identity.environment.packages",
        error_type=error_type,
    )
    for name, version in packages.items():
        if version is not None and (not isinstance(version, str) or not version):
            raise error_type(
                f"identity.environment.packages.{name} must be a non-empty string or null"
            )


def _validate_created_at_utc(core_module: Any, value: Any) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise core_module.CheckpointIntegrityError(
            "checkpoint-v1 created_at_utc must be an ISO-8601 UTC Z timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise core_module.CheckpointIntegrityError(
            "checkpoint-v1 created_at_utc must be an ISO-8601 UTC Z timestamp"
        ) from exc
    if parsed.tzinfo != UTC:
        raise core_module.CheckpointIntegrityError(
            "checkpoint-v1 created_at_utc must be UTC"
        )


def _validate_manifest_envelope(core_module: Any, manifest: Any) -> None:
    value = _require_exact_keys(
        manifest,
        _MANIFEST_KEYS,
        label="checkpoint-v1 manifest",
        error_type=core_module.CheckpointIntegrityError,
    )

    version = value.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise core_module.CheckpointIntegrityError(
            "checkpoint-v1 format_version must be an integer"
        )
    if version != core_module.FORMAT_VERSION:
        raise core_module.CheckpointCompatibilityError(
            f"unsupported checkpoint-v1 format version: {version!r}"
        )

    _validate_created_at_utc(core_module, value.get("created_at_utc"))

    try:
        core_module._require_exact_hex(
            value.get("checkpoint_id"),
            field="checkpoint_id",
            lengths={64},
        )
    except ValueError as exc:
        raise core_module.CheckpointIntegrityError(str(exc)) from exc

    files = _require_exact_keys(
        value.get("files"),
        core_module._PAYLOAD_NAMES,
        label="checkpoint-v1 files",
        error_type=core_module.CheckpointIntegrityError,
    )
    for name, record in files.items():
        _require_exact_keys(
            record,
            _FILE_RECORD_KEYS,
            label=f"checkpoint-v1 files.{name}",
            error_type=core_module.CheckpointIntegrityError,
        )

    serialization = _require_exact_keys(
        value.get("serialization"),
        frozenset(_SERIALIZATION),
        label="checkpoint-v1 serialization",
        error_type=core_module.CheckpointIntegrityError,
    )
    if dict(serialization) != _SERIALIZATION:
        raise core_module.CheckpointCompatibilityError(
            "checkpoint-v1 serialization contract mismatch"
        )


def install(core_module: Any) -> None:
    """Freeze v1 metadata schemas; semantic extensions require a version bump."""

    if getattr(core_module, "_D05_MANIFEST_SCHEMA_INSTALLED", False):
        return
    original_parse_manifest = core_module._parse_manifest_bytes
    original_validate_identity = core_module._validate_manifest_identity

    def parse_manifest_bytes(
        manifest_bytes: bytes,
        checksum_bytes: bytes,
    ) -> dict[str, Any]:
        manifest = original_parse_manifest(manifest_bytes, checksum_bytes)
        _validate_manifest_envelope(core_module, manifest)
        return manifest

    def validate_manifest_identity(identity: Any) -> None:
        value = _require_exact_keys(
            identity,
            _IDENTITY_KEYS,
            label="checkpoint-v1 identity",
            error_type=core_module.CheckpointIntegrityError,
        )
        _validate_environment(
            value.get("environment"),
            error_type=core_module.CheckpointIntegrityError,
        )
        original_validate_identity(identity)

    core_module._parse_manifest_bytes = parse_manifest_bytes
    core_module._validate_manifest_identity = validate_manifest_identity
    core_module._D05_MANIFEST_SCHEMA_INSTALLED = True
