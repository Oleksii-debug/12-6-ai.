"""Fail-closed tokenizer/checkpoint migration decisions for later learned stages.

This module never remaps token IDs or weights. It distinguishes exact tokenizer
identity reuse from a tokenizer change that requires a fresh vocabulary-facing
parameter initialization or a separately reviewed explicit migration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

MIGRATION_DECISION_SCHEMA = "12-6.tokenizer-migration-decision.v1"
_TOKENIZER_KEYS = frozenset(
    {
        "version",
        "config_sha256",
        "vocab_sha256",
        "vocab_size",
        "normalization",
        "encoding",
        "special_tokens",
    }
)
_IDENTITY_FIELDS = (
    "version",
    "config_sha256",
    "vocab_sha256",
    "vocab_size",
    "normalization",
    "encoding",
    "special_tokens",
)
_HEX = frozenset("0123456789abcdef")


class TokenizerMigrationError(ValueError):
    """Raised when a tokenizer migration request is ambiguous or unsafe."""


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise TokenizerMigrationError(f"{label} must be an exact lowercase SHA-256")
    return value


def _require_nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TokenizerMigrationError(f"{label} must be non-empty text")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TokenizerMigrationError(f"{label} must be a positive integer")
    return value


def _normalize_special_tokens(
    value: Any, *, vocab_size: int, label: str
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TokenizerMigrationError(f"{label} must be an object")
    normalized: dict[str, int] = {}
    seen_ids: set[int] = set()
    for token, token_id in value.items():
        if not isinstance(token, str) or not token:
            raise TokenizerMigrationError(f"{label} names must be non-empty strings")
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id >= vocab_size
        ):
            raise TokenizerMigrationError(
                f"{label}.{token} must be an integer ID inside the vocabulary"
            )
        if token_id in seen_ids:
            raise TokenizerMigrationError(f"{label} IDs must be unique")
        seen_ids.add(token_id)
        normalized[token] = token_id
    return dict(sorted(normalized.items()))


def _normalize_tokenizer_identity(
    identity: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if not isinstance(identity, Mapping) or set(identity) != _TOKENIZER_KEYS:
        raise TokenizerMigrationError(f"{label} fields do not match TokenizerIdentity")
    vocab_size = _require_positive_int(identity["vocab_size"], f"{label}.vocab_size")
    return {
        "version": _require_nonempty_text(identity["version"], f"{label}.version"),
        "config_sha256": _require_sha256(
            identity["config_sha256"], f"{label}.config_sha256"
        ),
        "vocab_sha256": _require_sha256(
            identity["vocab_sha256"], f"{label}.vocab_sha256"
        ),
        "vocab_size": vocab_size,
        "normalization": _require_nonempty_text(
            identity["normalization"], f"{label}.normalization"
        ),
        "encoding": _require_nonempty_text(identity["encoding"], f"{label}.encoding"),
        "special_tokens": _normalize_special_tokens(
            identity["special_tokens"],
            vocab_size=vocab_size,
            label=f"{label}.special_tokens",
        ),
    }


def _normalize_model_spec(spec: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise TokenizerMigrationError(f"{label} must be an object")
    vocab_size = _require_positive_int(spec.get("vocab_size"), f"{label}.vocab_size")
    tied = spec.get("tie_word_embeddings")
    if not isinstance(tied, bool):
        raise TokenizerMigrationError(f"{label}.tie_word_embeddings must be boolean")
    lm_head_bias = spec.get("lm_head_bias", False)
    if not isinstance(lm_head_bias, bool):
        raise TokenizerMigrationError(f"{label}.lm_head_bias must be boolean")
    return {
        "vocab_size": vocab_size,
        "tie_word_embeddings": tied,
        "lm_head_bias": lm_head_bias,
    }


def _validate_checkpoint_source_binding(
    checkpoint_identity: Mapping[str, Any],
    source_tokenizer: Mapping[str, Any],
    source_model: Mapping[str, Any],
) -> None:
    if not isinstance(checkpoint_identity, Mapping):
        raise TokenizerMigrationError("checkpoint_identity must be an object")
    tokenizer_hash = _require_sha256(
        checkpoint_identity.get("tokenizer_hash"), "checkpoint_identity.tokenizer_hash"
    )
    vocab_hash = _require_sha256(
        checkpoint_identity.get("tokenizer_vocab_hash"),
        "checkpoint_identity.tokenizer_vocab_hash",
    )
    if tokenizer_hash != source_tokenizer["config_sha256"]:
        raise TokenizerMigrationError(
            "checkpoint tokenizer hash does not match source tokenizer identity"
        )
    if vocab_hash != source_tokenizer["vocab_sha256"]:
        raise TokenizerMigrationError(
            "checkpoint vocabulary hash does not match source tokenizer identity"
        )
    checkpoint_model = checkpoint_identity.get("model_spec")
    if not isinstance(checkpoint_model, Mapping):
        raise TokenizerMigrationError("checkpoint_identity.model_spec must be an object")
    checkpoint_vocab = _require_positive_int(
        checkpoint_model.get("vocab_size"), "checkpoint_identity.model_spec.vocab_size"
    )
    if checkpoint_vocab != source_model["vocab_size"]:
        raise TokenizerMigrationError(
            "checkpoint ModelSpec vocabulary does not match source ModelSpec"
        )


def _vocabulary_parameter_surfaces(model: Mapping[str, Any]) -> list[str]:
    surfaces = ["token_embedding.weight"]
    if not model["tie_word_embeddings"]:
        surfaces.append("lm_head.weight")
    if model["lm_head_bias"]:
        surfaces.append("lm_head.bias")
    return surfaces


def assess_tokenizer_checkpoint_migration(
    *,
    target_tokenizer_identity: Mapping[str, Any],
    target_model_spec: Mapping[str, Any],
    reuse_checkpoint_weights: bool,
    source_tokenizer_identity: Mapping[str, Any] | None = None,
    source_model_spec: Mapping[str, Any] | None = None,
    checkpoint_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic fail-closed tokenizer/checkpoint migration decision.

    Fresh stages need only a target tokenizer + target ModelSpec. Checkpoint reuse
    additionally requires an externally bound source tokenizer/model/checkpoint.
    Exact token IDs are never inferred from token strings, vocabulary size, or
    algorithm names. A changed identity is therefore not approved for weight reuse.
    """
    if not isinstance(reuse_checkpoint_weights, bool):
        raise TokenizerMigrationError("reuse_checkpoint_weights must be boolean")

    target_tokenizer = _normalize_tokenizer_identity(
        target_tokenizer_identity, label="target_tokenizer_identity"
    )
    target_model = _normalize_model_spec(target_model_spec, label="target_model_spec")
    if target_model["vocab_size"] != target_tokenizer["vocab_size"]:
        raise TokenizerMigrationError(
            "target ModelSpec vocab_size must equal target tokenizer vocab_size"
        )

    source_items = (
        source_tokenizer_identity,
        source_model_spec,
        checkpoint_identity,
    )
    source_present = any(item is not None for item in source_items)
    source_complete = all(item is not None for item in source_items)
    if source_present and not source_complete:
        raise TokenizerMigrationError(
            "source tokenizer, source ModelSpec and checkpoint identity must be supplied together"
        )
    if reuse_checkpoint_weights and not source_complete:
        raise TokenizerMigrationError(
            "checkpoint reuse requires source tokenizer, ModelSpec and checkpoint identity"
        )

    source_tokenizer: dict[str, Any] | None = None
    source_model: dict[str, Any] | None = None
    changed_fields: list[str] = []
    vocabulary_changed = False
    parameter_surfaces: list[str] = []

    if source_complete:
        assert source_tokenizer_identity is not None
        assert source_model_spec is not None
        assert checkpoint_identity is not None
        source_tokenizer = _normalize_tokenizer_identity(
            source_tokenizer_identity, label="source_tokenizer_identity"
        )
        source_model = _normalize_model_spec(source_model_spec, label="source_model_spec")
        if source_model["vocab_size"] != source_tokenizer["vocab_size"]:
            raise TokenizerMigrationError(
                "source ModelSpec vocab_size must equal source tokenizer vocab_size"
            )
        _validate_checkpoint_source_binding(
            checkpoint_identity,
            source_tokenizer,
            source_model,
        )
        changed_fields = [
            field
            for field in _IDENTITY_FIELDS
            if source_tokenizer[field] != target_tokenizer[field]
        ]
        vocabulary_changed = any(
            field in changed_fields
            for field in ("vocab_sha256", "vocab_size", "special_tokens")
        )
        if vocabulary_changed:
            parameter_surfaces = _vocabulary_parameter_surfaces(target_model)

    exact_tokenizer_identity = source_complete and not changed_fields
    if reuse_checkpoint_weights and exact_tokenizer_identity:
        status = "EXACT_TOKENIZER_IDENTITY_COMPATIBLE"
        checkpoint_weight_reuse_allowed = True
        required_action = (
            "Continue through D05 full checkpoint/ModelSpec compatibility; this decision "
            "only proves tokenizer identity compatibility."
        )
    elif reuse_checkpoint_weights:
        status = "BLOCK_CHECKPOINT_REUSE_TOKENIZER_IDENTITY_CHANGED"
        checkpoint_weight_reuse_allowed = False
        required_action = (
            "Do not load the checkpoint into the changed tokenizer lineage. Start from "
            "fresh random initialization or introduce a separately reviewed explicit "
            "token-ID/weight migration authority."
        )
    elif source_complete and changed_fields:
        status = "FRESH_INITIALIZATION_REQUIRED_TOKENIZER_CHANGED"
        checkpoint_weight_reuse_allowed = False
        required_action = (
            "Use the target tokenizer only with a target ModelSpec and fresh random "
            "initialization; do not inherit source vocabulary-facing weights."
        )
    else:
        status = "FRESH_INITIALIZATION_TARGET_TOKENIZER_BOUND"
        checkpoint_weight_reuse_allowed = False
        required_action = (
            "Target tokenizer/model vocabulary is internally compatible for a fresh "
            "random-init stage; no predecessor tokenizer is assumed."
        )

    decision: dict[str, Any] = {
        "schema_version": MIGRATION_DECISION_SCHEMA,
        "status": status,
        "source_identity_present": source_complete,
        "reuse_checkpoint_weights_requested": reuse_checkpoint_weights,
        "checkpoint_weight_reuse_allowed_by_tokenizer_contract": (
            checkpoint_weight_reuse_allowed
        ),
        "full_checkpoint_compatibility_proven": False,
        "exact_tokenizer_identity": bool(exact_tokenizer_identity),
        "changed_tokenizer_fields": changed_fields,
        "vocabulary_identity_changed": vocabulary_changed,
        "token_id_mapping_assumed": False,
        "token_strings_used_to_infer_id_equivalence": False,
        "vocabulary_parameter_surfaces_requiring_fresh_init_or_explicit_migration": (
            parameter_surfaces
        ),
        "target_tokenizer_identity": target_tokenizer,
        "target_model_vocabulary": target_model,
        "source_tokenizer_identity": source_tokenizer,
        "source_model_vocabulary": source_model,
        "required_action": required_action,
        "training_authorized_by_this_contract": False,
    }
    decision["decision_identity_sha256"] = _canonical_sha256(decision)
    return decision
