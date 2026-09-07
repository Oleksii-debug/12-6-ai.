"""Downstream bundle validation for S1 numerical preflight evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .s0_evidence_contract import (
    DATASET_IDENTITY_SHA256,
    DATASET_MANIFEST_SHA256,
    PACKING_CONFIG_SHA256,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_VOCAB_SHA256,
    validate_locked_environment_evidence,
)
from .s1_preflight import (
    FIXTURE_TOKENIZER_VOCAB,
    S1_MODEL_VOCAB,
    S1PreflightError,
    validate_s1_numerical_preflight,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S1PreflightError(message)


def validate_s1_preflight_bundle(
    evidence: Mapping[str, Any],
    locked_environment_evidence: Mapping[str, Any],
) -> None:
    """Validate preflight plus the retained exact-source D08 lock companion."""
    validate_s1_numerical_preflight(evidence)
    identity = evidence["identity"]
    _require(isinstance(identity, Mapping), "identity block missing")
    source_sha = identity.get("source_sha")
    _require(isinstance(source_sha, str), "source SHA missing")

    expected_environment = validate_locked_environment_evidence(
        locked_environment_evidence,
        source_sha=source_sha,
    )
    bound_environment = identity.get("environment")
    _require(isinstance(bound_environment, Mapping), "environment binding missing")
    _require(dict(bound_environment) == expected_environment, "locked environment binding mismatch")

    fixture = identity.get("fixture")
    _require(isinstance(fixture, Mapping), "fixture identity missing")
    expected_fixture = {
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
        "packing_config_sha256": PACKING_CONFIG_SHA256,
        "tokenizer_vocab_size": FIXTURE_TOKENIZER_VOCAB,
        "unused_model_vocab_rows": S1_MODEL_VOCAB - FIXTURE_TOKENIZER_VOCAB,
    }
    for key, expected in expected_fixture.items():
        _require(fixture.get(key) == expected, f"controlled fixture identity mismatch: {key}")
    max_token_id = fixture.get("max_emitted_token_id")
    _require(
        isinstance(max_token_id, int) and 0 <= max_token_id < FIXTURE_TOKENIZER_VOCAB,
        "controlled byte fixture emitted token outside byte vocabulary",
    )

    profiles = evidence.get("profiles")
    _require(isinstance(profiles, Mapping), "precision profiles missing")
    for precision in ("fp32", "bf16"):
        profile = profiles.get(precision)
        _require(isinstance(profile, Mapping), f"{precision} profile missing")
        _require(profile.get("precision") == precision, f"{precision} profile label mismatch")
        _require(
            profile.get("optimizer_steps") == profile.get("microbatches_consumed"),
            f"{precision} step/microbatch accounting mismatch",
        )

    seed_ordering = evidence.get("seed_ordering")
    _require(isinstance(seed_ordering, Mapping), "seed ordering missing")
    declared_seed = seed_ordering.get("seed")
    _require(isinstance(declared_seed, int), "declared seed missing")
    for precision in ("fp32", "bf16"):
        profile = profiles[precision]
        _require(profile.get("seed") == declared_seed, f"{precision} seed binding mismatch")
