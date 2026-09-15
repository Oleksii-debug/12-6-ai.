"""Fail-closed binding from a launch/run manifest to checkpoint identity.

The low-level checkpoint serializer is intentionally framework-neutral. Canonical
training runs should construct their identity through this module so candidate,
initialization, tokenizer, dataset/split, packing, environment, and training
configuration cannot drift silently between launch and checkpoint creation.
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
    if not isinstance(value, str) or not value.strip() or value == "UNRESOLVED":
        raise CheckpointCompatibilityError(f"run manifest {key!r} must be resolved non-empty text")
    return value


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(ch not in _HEX for ch in value)
    ):
        raise CheckpointCompatibilityError(f"{field} must be an exact lowercase 64-hex SHA-256")
    return value


def _require_git_sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) not in {40, 64}:
        raise CheckpointCompatibilityError(f"{field} must be a full lowercase 40- or 64-hex Git SHA")
    if value != value.lower() or any(ch not in _HEX for ch in value):
        raise CheckpointCompatibilityError(f"{field} must be a full lowercase 40- or 64-hex Git SHA")
    return value


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
    init_spec: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    packing_identity: Mapping[str, Any],
    step: int,
    tokens_seen: int,
    environment_lock_hash: str | None = None,
) -> CheckpointIdentity:
    """Construct a canonical checkpoint identity only when launch identities agree.

    Canonical S0 binding requires exact ModelSpec and InitSpec identities, tokenizer
    configuration and vocabulary identities, dataset manifest and split identity,
    packing version/config identity, the complete C01 run manifest, and the exact
    environment lock. The complete manifest hash then transitively binds the full
    training configuration and seed into the durable checkpoint identity.
    """

    run_id = _require_text(run_manifest, "run_id")
    stage = _require_text(run_manifest, "stage")
    run_kind = _require_text(run_manifest, "run_kind")

    candidate = _require_mapping(run_manifest, "candidate")
    data = _require_mapping(run_manifest, "data")
    training = _require_mapping(run_manifest, "training")
    environment = _require_mapping(run_manifest, "environment")

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

    if not isinstance(init_spec, Mapping) or not init_spec:
        raise CheckpointCompatibilityError("supplied InitSpec must be a non-empty mapping")
    declared_init_hash = _require_sha256(
        candidate.get("initspec_sha256"),
        field="candidate.initspec_sha256",
    )
    actual_init_hash = hash_json(init_spec)
    if declared_init_hash != actual_init_hash:
        raise CheckpointCompatibilityError(
            "run manifest InitSpec hash does not match the supplied InitSpec"
        )

    parameter_count = candidate.get("parameter_count")
    if (
        not isinstance(parameter_count, int)
        or isinstance(parameter_count, bool)
        or parameter_count <= 0
    ):
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

    tokenizer_vocab_hash = _require_sha256(
        data.get("tokenizer_vocab_sha256"),
        field="data.tokenizer_vocab_sha256",
    )
    supplied_vocab_hash = _require_sha256(
        tokenizer_identity.get("vocab_sha256"),
        field="tokenizer_identity.vocab_sha256",
    )
    if tokenizer_vocab_hash != supplied_vocab_hash:
        raise CheckpointCompatibilityError(
            "run manifest tokenizer vocabulary SHA does not match supplied tokenizer identity"
        )

    tokenizer_version = _require_text(tokenizer_identity, "version")
    if data.get("tokenizer_version") != tokenizer_version:
        raise CheckpointCompatibilityError(
            "run manifest tokenizer version does not match supplied tokenizer identity"
        )

    tokenizer_vocab = tokenizer_identity.get("vocab_size")
    if (
        not isinstance(tokenizer_vocab, int)
        or isinstance(tokenizer_vocab, bool)
        or tokenizer_vocab <= 0
    ):
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
    split_identity = _require_text(data, "split_identity")

    packing_hash = _require_sha256(
        data.get("packing_sha256"),
        field="data.packing_sha256",
    )
    supplied_packing_hash = _require_sha256(
        packing_identity.get("config_sha256"),
        field="packing_identity.config_sha256",
    )
    if packing_hash != supplied_packing_hash:
        raise CheckpointCompatibilityError(
            "run manifest packing SHA does not match supplied packing identity"
        )
    packing_version = _require_text(packing_identity, "version")
    if data.get("packing_version") != packing_version:
        raise CheckpointCompatibilityError(
            "run manifest packing version does not match supplied packing identity"
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

    run_environment_lock_hash = _require_sha256(
        environment.get("lock_sha256"),
        field="environment.lock_sha256",
    )
    if environment_lock_hash is not None:
        supplied_environment_lock_hash = _require_sha256(
            environment_lock_hash,
            field="environment_lock_hash",
        )
        if supplied_environment_lock_hash != run_environment_lock_hash:
            raise CheckpointCompatibilityError(
                "run manifest environment lock SHA does not match supplied environment lock"
            )
    environment_lock_hash = run_environment_lock_hash

    run_manifest_hash = hash_json(run_manifest)
    bound_training_config = {
        "run_id": run_id,
        "run_manifest_sha256": run_manifest_hash,
        "stage": stage,
        "run_kind": run_kind,
        "init_spec_sha256": actual_init_hash,
        "training": dict(training),
        "data": {
            "dataset_manifest_sha256": dataset_hash,
            "split_identity": split_identity,
            "tokenizer_sha256": tokenizer_hash,
            "tokenizer_vocab_sha256": tokenizer_vocab_hash,
            "tokenizer_version": tokenizer_version,
            "packing_sha256": packing_hash,
            "packing_version": packing_version,
        },
        "environment": {"lock_sha256": environment_lock_hash},
    }

    return CheckpointIdentity(
        git_sha=git_sha,
        model_spec=dict(model_spec),
        parameter_count=parameter_count,
        tokenizer_hash=tokenizer_hash,
        tokenizer_vocab_hash=tokenizer_vocab_hash,
        dataset_manifest_hash=dataset_hash,
        run_manifest_hash=run_manifest_hash,
        training_config=bound_training_config,
        seed=seed,
        precision=precision,
        step=step,
        tokens_seen=tokens_seen,
        optimizer=optimizer,
        scheduler=scheduler,
        environment_lock_hash=environment_lock_hash,
    )
