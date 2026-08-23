"""Fail-closed binding from a launch/run manifest to checkpoint identity.

The low-level checkpoint serializer is intentionally framework-neutral. Canonical
training runs should construct their identity through this module so candidate,
tokenizer, dataset, and training configuration cannot drift silently between
launch and checkpoint creation.
"""

from __future__ import annotations

import string
from collections.abc import Mapping
from typing import Any

from .core import CheckpointCompatibilityError, CheckpointIdentity, hash_json

_HEX = frozenset(string.hexdigits.lower())


def _require_mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError(f"run manifest {key!r} must be a mapping")
    return value


def _require_text(parent: Mapping[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CheckpointCompatibilityError(f"run manifest {key!r} must be non-empty text")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch.lower() not in _HEX for ch in value):
        raise CheckpointCompatibilityError(f"{field} must be an exact 64-hex SHA-256")
    return value.lower()


def _require_git_sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) not in {40, 64}:
        raise CheckpointCompatibilityError(f"{field} must be a full 40- or 64-hex Git SHA")
    if any(ch.lower() not in _HEX for ch in value):
        raise CheckpointCompatibilityError(f"{field} must be a full 40- or 64-hex Git SHA")
    return value.lower()


def _metadata(value: Any, *, field: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        return {"name": value}
    raise CheckpointCompatibilityError(f"{field} must be non-empty text, mapping, or null")


def bind_checkpoint_identity(
    *,
    run_manifest: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    step: int,
    tokens_seen: int,
    environment_lock_hash: str | None = None,
) -> CheckpointIdentity:
    """Construct a checkpoint identity only when launch identities agree exactly.

    Required run-manifest structure follows the C01 S0 contract: ``candidate``,
    ``data``, and ``training`` mappings. The function deliberately rejects an
    unresolved manifest, abbreviated Git SHAs, hash drift, tokenizer version/vocab
    drift, and model/tokenizer vocabulary mismatch.
    """

    run_id = _require_text(run_manifest, "run_id")
    if run_id == "UNRESOLVED":
        raise CheckpointCompatibilityError("run_id is unresolved; checkpoint creation is blocked")
    stage = _require_text(run_manifest, "stage")
    run_kind = _require_text(run_manifest, "run_kind")

    candidate = _require_mapping(run_manifest, "candidate")
    data = _require_mapping(run_manifest, "data")
    training = _require_mapping(run_manifest, "training")

    git_sha = _require_git_sha(candidate.get("git_sha"), field="candidate.git_sha")
    declared_model_hash = _require_sha256(
        candidate.get("modelspec_sha256"),
        field="candidate.modelspec_sha256",
    )
    actual_model_hash = hash_json(model_spec)
    if declared_model_hash != actual_model_hash:
        raise CheckpointCompatibilityError(
            "run manifest ModelSpec hash does not match the supplied ModelSpec"
        )

    parameter_count = candidate.get("parameter_count")
    if not isinstance(parameter_count, int) or isinstance(parameter_count, bool) or parameter_count <= 0:
        raise CheckpointCompatibilityError("candidate.parameter_count must be a positive integer")

    tokenizer_hash = _require_sha256(
        data.get("tokenizer_sha256"),
        field="data.tokenizer_sha256",
    )
    tokenizer_config_hash = _require_sha256(
        tokenizer_identity.get("config_sha256"),
        field="tokenizer_identity.config_sha256",
    )
    if tokenizer_hash != tokenizer_config_hash:
        raise CheckpointCompatibilityError(
            "run manifest tokenizer SHA does not match supplied tokenizer identity"
        )

    tokenizer_version = tokenizer_identity.get("version")
    if not isinstance(tokenizer_version, str) or not tokenizer_version:
        raise CheckpointCompatibilityError("tokenizer identity requires a non-empty version")
    if data.get("tokenizer_version") != tokenizer_version:
        raise CheckpointCompatibilityError(
            "run manifest tokenizer version does not match supplied tokenizer identity"
        )

    tokenizer_vocab = tokenizer_identity.get("vocab_size")
    if not isinstance(tokenizer_vocab, int) or isinstance(tokenizer_vocab, bool) or tokenizer_vocab <= 0:
        raise CheckpointCompatibilityError("tokenizer identity requires a positive vocab_size")
    model_vocab = model_spec.get("vocab_size")
    if model_vocab is not None and model_vocab != tokenizer_vocab:
        raise CheckpointCompatibilityError(
            f"ModelSpec/tokenizer vocab mismatch: model={model_vocab}, tokenizer={tokenizer_vocab}"
        )

    dataset_hash = _require_sha256(
        data.get("dataset_manifest_sha256"),
        field="data.dataset_manifest_sha256",
    )
    seed = training.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise CheckpointCompatibilityError("training.seed must be a non-negative integer")
    precision = training.get("precision")
    if not isinstance(precision, str) or not precision:
        raise CheckpointCompatibilityError("training.precision must be non-empty text")

    optimizer = _metadata(training.get("optimizer"), field="training.optimizer")
    if optimizer is None:
        raise CheckpointCompatibilityError("training.optimizer must be resolved")
    scheduler = _metadata(training.get("scheduler"), field="training.scheduler")

    if step < 0 or tokens_seen < 0:
        raise CheckpointCompatibilityError("step and tokens_seen must be non-negative")
    if environment_lock_hash is not None:
        environment_lock_hash = _require_sha256(
            environment_lock_hash,
            field="environment_lock_hash",
        )

    bound_training_config = {
        "run_id": run_id,
        "stage": stage,
        "run_kind": run_kind,
        "training": dict(training),
        "data": {
            "dataset_manifest_sha256": dataset_hash,
            "split_identity": data.get("split_identity"),
            "tokenizer_sha256": tokenizer_hash,
            "tokenizer_version": tokenizer_version,
        },
    }

    return CheckpointIdentity(
        git_sha=git_sha,
        model_spec=dict(model_spec),
        parameter_count=parameter_count,
        tokenizer_hash=tokenizer_hash,
        dataset_manifest_hash=dataset_hash,
        training_config=bound_training_config,
        seed=seed,
        precision=precision,
        step=step,
        tokens_seen=tokens_seen,
        optimizer=optimizer,
        scheduler=scheduler,
        environment_lock_hash=environment_lock_hash,
    )
