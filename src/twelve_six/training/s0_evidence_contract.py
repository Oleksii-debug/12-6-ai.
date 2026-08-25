"""Fail-closed contract for candidate-bound S0 real-training evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

SCHEMA_VERSION = "12-6.s0-real-training-evidence.v2"
REPOSITORY = "Oleksii-debug/12-6-ai."
MODEL_SPEC_SHA256 = "86c75b31dff05b7b5db9f6ed068c571a6ead01ba663412fe630f5e52b09d9b6b"
INIT_SPEC_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
DATASET_MANIFEST_SHA256 = "b085a7ab56510575a11a80824fcff3a95a17f237d46d1be820e59d1289f220c2"
DATASET_IDENTITY_SHA256 = "bab60119d49e93303c972b77900fcb5553817f754cbc5d9a58019228cfa0ca89"
TOKENIZER_CONFIG_SHA256 = "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
TOKENIZER_VOCAB_SHA256 = "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
PACKING_CONFIG_SHA256 = "23a695b807f3e3f5c61d19c34968bcd88fafc6a45346dc08673d7a494219f285"
LOCK_INDEX_PATH = "requirements/locks/index.json"
LOCK_INDEX_FILE_SHA256 = "61fa31fbb5da7a4289cccce5abfcebde943664f5318b0ce3d69ae9bb3db852ac"
LOCK_INDEX_SEMANTIC_SHA256 = "5de40d40012123ccf654b3e29d9cd47df814978e4155ca9dde232b61e9cd6341"
LOCK_PROFILE_ID = "linux-x86_64"
LOCK_PROFILE_MANIFEST_SHA256 = "283ca83571e527babda700e0c66ed03fb1c2aa4674bee0dba2272f64f344e1bf"
LOCK_PROFILE_FILE_SHA256 = "0dbd74f05d2824082d378aa9031141c864f6a092f950d63ccfae0ade4d76e121"
PYTHON_VERSION = "3.11.16"
PARAMETER_COUNT = 10_140
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class S0EvidenceContractError(ValueError):
    """Raised when evidence cannot be trusted under the S0 contract."""


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S0EvidenceContractError(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} block missing")
    return value


def _finite_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field} must be numeric",
    )
    number = float(value)
    _require(math.isfinite(number), f"{field} must be finite")
    if nonnegative:
        _require(number >= 0.0, f"{field} must be non-negative")
    return number


def validate_locked_environment_evidence(
    evidence: Mapping[str, Any], *, source_sha: str
) -> dict[str, str]:
    """Validate D08 locked-environment evidence and return the binding payload."""
    _require(
        _GIT_SHA.fullmatch(source_sha) is not None,
        "source_sha must be full lowercase Git SHA",
    )
    _require(
        evidence.get("schema_version") == "12-6.locked-environment-evidence.v1",
        "wrong locked environment schema",
    )
    _require(evidence.get("source_sha") == source_sha, "locked environment source SHA mismatch")
    _require(evidence.get("profile_id") == LOCK_PROFILE_ID, "wrong locked environment profile")

    python = _mapping(evidence.get("python"), "locked environment python")
    _require(python.get("version") == PYTHON_VERSION, "locked Python version mismatch")

    lock_index = _mapping(evidence.get("lock_index"), "locked environment index")
    _require(lock_index.get("path") == LOCK_INDEX_PATH, "lock index path mismatch")
    _require(
        lock_index.get("file_sha256") == LOCK_INDEX_FILE_SHA256,
        "lock index file SHA mismatch",
    )
    _require(
        lock_index.get("index_sha256") == LOCK_INDEX_SEMANTIC_SHA256,
        "lock index semantic SHA mismatch",
    )

    profile = _mapping(evidence.get("lock_profile"), "lock profile")
    _require(
        profile.get("manifest_sha256") == LOCK_PROFILE_MANIFEST_SHA256,
        "lock profile semantic SHA mismatch",
    )
    _require(
        profile.get("file_sha256") == LOCK_PROFILE_FILE_SHA256,
        "lock profile file SHA mismatch",
    )

    verification = _mapping(evidence.get("verification"), "locked environment verification")
    required_pass = (
        "committed_lock_validation",
        "editable_install_import_cli",
        "wheel_install_import_cli",
        "repo_checks",
    )
    for key in required_pass:
        _require(verification.get(key) == "PASS", f"locked environment {key} is not PASS")

    claimed_hash = evidence.get("evidence_sha256")
    _require(isinstance(claimed_hash, str), "locked environment evidence hash missing")
    unhashed = dict(evidence)
    unhashed.pop("evidence_sha256", None)
    _require(
        _canonical_hash(unhashed) == claimed_hash,
        "locked environment evidence self-hash mismatch",
    )
    return {
        "profile_id": LOCK_PROFILE_ID,
        "python_version": PYTHON_VERSION,
        "lock_index_file_sha256": LOCK_INDEX_FILE_SHA256,
        "lock_index_sha256": LOCK_INDEX_SEMANTIC_SHA256,
        "lock_profile_manifest_sha256": LOCK_PROFILE_MANIFEST_SHA256,
        "lock_profile_file_sha256": LOCK_PROFILE_FILE_SHA256,
        "environment_evidence_sha256": claimed_hash,
    }


def validate_s0_training_evidence(
    evidence: Mapping[str, Any], *, require_locked_environment: bool = True
) -> None:
    """Validate machine evidence for D05/D06/C01/D10 consumption."""
    _require(evidence.get("schema_version") == SCHEMA_VERSION, "wrong S0 training evidence schema")
    _require(
        evidence.get("authority") == "LOCAL_FREE_OR_FREE_HOSTED_CPU_EVIDENCE_NOT_PROMOTION",
        "wrong evidence authority",
    )

    identity = _mapping(evidence.get("identity"), "identity")
    _require(identity.get("repository") == REPOSITORY, "repository identity mismatch")
    source_sha = identity.get("source_sha")
    _require(
        isinstance(source_sha, str) and _GIT_SHA.fullmatch(source_sha) is not None,
        "source SHA must be full lowercase Git SHA",
    )
    expected_identity = {
        "stage": "S0",
        "modelspec_sha256": MODEL_SPEC_SHA256,
        "initspec_sha256": INIT_SPEC_SHA256,
        "parameter_count": PARAMETER_COUNT,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
        "packing_config_sha256": PACKING_CONFIG_SHA256,
    }
    for key, expected in expected_identity.items():
        _require(identity.get(key) == expected, f"{key} identity mismatch")

    claimed_identity_hash = evidence.get("identity_sha256")
    _require(
        isinstance(claimed_identity_hash, str) and _canonical_hash(identity) == claimed_identity_hash,
        "identity self-hash mismatch",
    )

    if require_locked_environment:
        environment = _mapping(identity.get("environment"), "locked environment binding")
        expected_environment = {
            "profile_id": LOCK_PROFILE_ID,
            "python_version": PYTHON_VERSION,
            "lock_index_file_sha256": LOCK_INDEX_FILE_SHA256,
            "lock_index_sha256": LOCK_INDEX_SEMANTIC_SHA256,
            "lock_profile_manifest_sha256": LOCK_PROFILE_MANIFEST_SHA256,
            "lock_profile_file_sha256": LOCK_PROFILE_FILE_SHA256,
        }
        for key, expected in expected_environment.items():
            _require(environment.get(key) == expected, f"training {key} mismatch")
        _require(
            isinstance(environment.get("environment_evidence_sha256"), str),
            "environment evidence hash missing",
        )

    seed_ordering = _mapping(evidence.get("seed_ordering"), "seed ordering")
    _require(
        seed_ordering.get("seed_applied_before_model_construction") is True,
        "model construction seed ordering not proven",
    )
    _require(
        seed_ordering.get("trainer_reapplies_training_rng_seed") is True,
        "trainer seed ordering not proven",
    )
    _require(
        seed_ordering.get("resume_policy") == "restore_rng_from_verified_checkpoint_not_reseed",
        "resume RNG policy mismatch",
    )

    split = _mapping(evidence.get("split_isolation"), "split isolation")
    _require(split.get("optimized_split") == "train", "only train split may be optimized")
    _require(split.get("record_id_overlap") == [], "train/validation overlap detected")
    _require(split.get("validation_optimized_tokens") == 0, "validation tokens were optimized")
    _require(
        split.get("validation_optimizer_step_before_final_eval")
        == split.get("validation_optimizer_step_after_final_eval"),
        "held-out evaluation mutated optimizer step",
    )

    training = _mapping(evidence.get("training"), "training")
    _require(
        isinstance(training.get("optimizer_steps"), int) and training["optimizer_steps"] > 0,
        "optimizer step count invalid",
    )
    _require(
        isinstance(training.get("optimized_tokens"), int) and training["optimized_tokens"] > 0,
        "optimized token count invalid",
    )
    _require(
        training.get("optimized_tokens") == training.get("trainer_tokens_seen"),
        "optimized token accounting mismatch",
    )
    initial_train = _finite_number(training.get("initial_train_loss"), "initial_train_loss")
    final_train = _finite_number(training.get("final_train_loss"), "final_train_loss")
    _require(final_train < initial_train, "full-train loss did not decrease")
    _finite_number(training.get("initial_validation_loss"), "initial_validation_loss")
    _finite_number(training.get("final_validation_loss"), "final_validation_loss")
    grad_min = _finite_number(training.get("gradient_norm_min"), "gradient_norm_min", nonnegative=True)
    grad_max = _finite_number(training.get("gradient_norm_max"), "gradient_norm_max", nonnegative=True)
    _require(grad_max >= grad_min, "gradient norm range is inverted")

    delta = _mapping(training.get("weight_delta"), "weight delta")
    _require(
        isinstance(delta.get("changed_parameter_elements"), int)
        and delta["changed_parameter_elements"] > 0,
        "no trainable parameter changed",
    )
    _require(
        delta.get("trainable_parameter_elements") == PARAMETER_COUNT,
        "weight delta parameter cardinality mismatch",
    )
    _finite_number(delta.get("l2"), "weight_delta.l2", nonnegative=True)
    _finite_number(delta.get("max_abs"), "weight_delta.max_abs", nonnegative=True)

    failure = _mapping(evidence.get("failure_semantics"), "failure semantics")
    _require(
        failure.get("nan_fail_closed_and_fresh_recovery") is True,
        "NaN recovery contract not proven",
    )
    _require(
        failure.get("inf_fail_closed_and_fresh_recovery") is True,
        "Inf recovery contract not proven",
    )

    runtime = _mapping(evidence.get("runtime"), "runtime")
    _finite_number(runtime.get("wall_seconds"), "wall_seconds", nonnegative=True)
    _finite_number(runtime.get("process_cpu_seconds"), "process_cpu_seconds", nonnegative=True)
    _require(runtime.get("python") == PYTHON_VERSION, "observed training Python version mismatch")
    _require(runtime.get("device") == "cpu", "S0 LOCAL_FREE evidence must use CPU")

    claims = _mapping(evidence.get("claims"), "claims")
    forbidden_true = (
        "foreign_pretrained_weights_used",
        "instruction_or_alignment_training",
        "paid_compute_authorized_or_used",
        "candidate_or_stable_promotion",
    )
    for key in forbidden_true:
        _require(claims.get(key) is False, f"forbidden claim/input flag set: {key}")

    claimed_hash = evidence.get("evidence_sha256")
    _require(isinstance(claimed_hash, str), "evidence self-hash missing")
    unhashed = dict(evidence)
    unhashed.pop("evidence_sha256", None)
    _require(_canonical_hash(unhashed) == claimed_hash, "training evidence self-hash mismatch")
