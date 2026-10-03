"""Fail-closed learned-20M recipe authority for LEARN-345.

This module freezes recipe semantics only. It cannot authorize compute or
training, and it never executes an optimizer step.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

from twelve_six.learned20m_evaluation_firewall import PRIMARY_SELECTION_METRIC
from twelve_six.packing.core import DEFAULT_SEQUENCE_LENGTH, PACKING_CONFIG_HASH

REPOSITORY = "Oleksii-debug/12-6-ai."
SCHEMA = "12-6.learn345.learned20m-recipe-authority.v1"
SESSION_SCHEMA = "12-6.learn345.learned20m-recipe-session.v1"

MODEL341 = {
    "authority_git_sha": "e4ff486fd90802fc123bebf60eed4e59196a98df",
    "model_spec_sha256": "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441",
    "init_spec_sha256": "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5",
    "parameter_count": 20_613_440,
    "canonical_base": "random_init",
}
TRAIN344B = {
    "authority_git_sha": "630e4cbbac1ba8fbdb12a4df701793047e068f29",
    "contract_identity_sha256": "f212d051e80d65b9c731ea38dbe08b3e48d930f32caeb3da7becf68030613f21",
}
REQUESTED_TARGETS = 20_000_000
MEANINGFUL_FLOOR = 10_000_000
BOUNDARIES = ("0", "0.10", "0.25", "0.50", "0.75", "0.90", "1.00")
REQUIRED_HELD_OUT_STRATA = ("ua", "en", "code")
TRUSTED_AUTHORITY_ROLES = ("tokenizer", "d04", "d05", "d06")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class RecipeValidationError(ValueError):
    """Raised when a recipe policy or terminal binding is malformed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic canonical JSON bytes after rejecting non-finite values."""

    def walk(item: Any) -> None:
        if isinstance(item, bool) or item is None or isinstance(item, (str, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise RecipeValidationError("non-finite numeric value")
            return
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise RecipeValidationError("JSON object keys must be strings")
            for child in item.values():
                walk(child)
            return
        raise RecipeValidationError(f"unsupported JSON value type: {type(item).__name__}")

    walk(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def identity_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _json_contract_equal(value: Any, expected: Any) -> bool:
    """Compare JSON contracts without Python bool/int/float equality aliases."""
    return canonical_json_bytes(value) == canonical_json_bytes(expected)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise RecipeValidationError(f"{label} keys mismatch; missing={missing} unknown={unknown}")


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RecipeValidationError(f"{label} must be lowercase sha256")
    return value


def _git_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _GIT_SHA_RE.fullmatch(value) is None:
        raise RecipeValidationError(f"{label} must be lowercase git sha")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RecipeValidationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RecipeValidationError(f"{label} must be a non-negative integer")
    return value


def _terminal_authority(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RecipeValidationError(f"{label} must be an authority object")
    expected = {
        "repository",
        "git_sha",
        "evidence_sha256",
        "terminal",
        "workflow_run_id",
        "workflow_conclusion",
    }
    _exact_keys(value, expected, label)
    if value["repository"] != REPOSITORY:
        raise RecipeValidationError(f"{label}.repository mismatch")
    _git_sha(value["git_sha"], f"{label}.git_sha")
    _sha256(value["evidence_sha256"], f"{label}.evidence_sha256")
    _positive_int(value["workflow_run_id"], f"{label}.workflow_run_id")
    if value["terminal"] is not True:
        raise RecipeValidationError(f"{label}.terminal must be true")
    if value["workflow_conclusion"] != "success":
        raise RecipeValidationError(f"{label}.workflow_conclusion must be success")
    return dict(value)


def _validate_policy_without_identity(policy: dict[str, Any]) -> None:
    expected = {
        "schema",
        "campaign",
        "execution_profile",
        "source_authorities",
        "recipe",
        "runtime_budget",
        "checkpoint_and_evaluation",
        "failure_semantics",
        "truth_boundary",
    }
    _exact_keys(policy, expected, "policy")
    if policy["schema"] != SCHEMA:
        raise RecipeValidationError("policy schema mismatch")
    if policy["campaign"] != "LEARN-345":
        raise RecipeValidationError("campaign mismatch")
    if policy["execution_profile"] != "LOCAL_FREE":
        raise RecipeValidationError("execution_profile must be LOCAL_FREE")

    sources = policy["source_authorities"]
    if not isinstance(sources, dict):
        raise RecipeValidationError("source_authorities must be object")
    _exact_keys(sources, {"model341", "train344b"}, "source_authorities")
    if not _json_contract_equal(sources["model341"], MODEL341):
        raise RecipeValidationError("MODEL-341 authority drift")
    if not _json_contract_equal(sources["train344b"], TRAIN344B):
        raise RecipeValidationError("TRAIN-344B authority drift")

    recipe = policy["recipe"]
    if not isinstance(recipe, dict):
        raise RecipeValidationError("recipe must be object")
    _exact_keys(
        recipe,
        {
            "optimizer",
            "learning_rate",
            "learning_rate_rationale",
            "betas",
            "eps",
            "weight_decay",
            "gradient_clip_norm",
            "scheduler",
            "warmup_steps",
            "sequence_length",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "precision",
            "seed_vector",
        },
        "recipe",
    )
    frozen = {
        "optimizer": "AdamW",
        "learning_rate": 0.00022,
        "learning_rate_rationale": (
            "predeclared neutral control: middle TRAIN-344B mechanically-qualified arm; "
            "not selected from learned-corpus, validation, or final-test outcomes"
        ),
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
        "gradient_clip_norm": 1.0,
        "scheduler": "constant",
        "warmup_steps": 0,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "precision": "fp32",
        "seed_vector": {
            "model_init": 20260826,
            "data_order": 20260826,
            "dataloader": 20260826,
        },
    }
    if not _json_contract_equal(recipe, frozen):
        raise RecipeValidationError("frozen recipe drift")

    budget = policy["runtime_budget"]
    if not isinstance(budget, dict):
        raise RecipeValidationError("runtime_budget must be object")
    _exact_keys(
        budget,
        {
            "requested_unique_loss_positions",
            "meaningful_minimum_unique_loss_positions",
            "rule",
            "replay_allowed",
            "replacement_sampling_allowed",
            "padding_counts_as_capacity",
            "max_exposures_per_unique_position",
        },
        "runtime_budget",
    )
    if not _json_contract_equal(
        budget,
        {
            "requested_unique_loss_positions": REQUESTED_TARGETS,
            "meaningful_minimum_unique_loss_positions": MEANINGFUL_FLOOR,
            "rule": "min(20000000, terminal_d04_unique_nonignored_causal_loss_positions)",
            "replay_allowed": False,
            "replacement_sampling_allowed": False,
            "padding_counts_as_capacity": False,
            "max_exposures_per_unique_position": 1,
        },
    ):
        raise RecipeValidationError("runtime budget drift")

    cadence = policy["checkpoint_and_evaluation"]
    if not isinstance(cadence, dict):
        raise RecipeValidationError("checkpoint_and_evaluation must be object")
    _exact_keys(
        cadence,
        {
            "boundaries",
            "mandatory_fresh_process_resume_fraction",
            "train_trace_frozen_before_step_1",
            "next_exposure_identity_required_before_step_1",
            "chronological_final_retained",
            "best_selection_checkpoint_retained",
            "final_test_sealed_until_selection_lock",
            "final_test_may_influence_selection",
        },
        "checkpoint_and_evaluation",
    )
    if not _json_contract_equal(
        cadence,
        {
            "boundaries": list(BOUNDARIES),
            "mandatory_fresh_process_resume_fraction": "0.50",
            "train_trace_frozen_before_step_1": True,
            "next_exposure_identity_required_before_step_1": True,
            "chronological_final_retained": True,
            "best_selection_checkpoint_retained": True,
            "final_test_sealed_until_selection_lock": True,
            "final_test_may_influence_selection": False,
        },
    ):
        raise RecipeValidationError("checkpoint/evaluation cadence drift")

    failure = policy["failure_semantics"]
    if not _json_contract_equal(
        failure,
        {
            "nan_or_inf_loss": "STOP_FAIL_CLOSED_NO_COMMIT",
            "nan_or_inf_gradient": "STOP_FAIL_CLOSED_NO_COMMIT",
            "silent_replay_allowed": False,
            "in_place_counter_repair_allowed": False,
        },
    ):
        raise RecipeValidationError("failure semantics drift")

    truth = policy["truth_boundary"]
    if not _json_contract_equal(
        truth,
        {
            "training_authorized": False,
            "compute_authorized": False,
            "authorized_optimized_targets": 0,
            "optimizer_updates_executed": 0,
            "final_test_payload_accessed": False,
        },
    ):
        raise RecipeValidationError("truth boundary drift")


def validate_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise RecipeValidationError("policy must be object")
    _exact_keys(
        policy,
        {
            "schema",
            "campaign",
            "execution_profile",
            "source_authorities",
            "recipe",
            "runtime_budget",
            "checkpoint_and_evaluation",
            "failure_semantics",
            "truth_boundary",
            "policy_identity_sha256",
        },
        "policy",
    )
    body = {key: value for key, value in policy.items() if key != "policy_identity_sha256"}
    _validate_policy_without_identity(body)
    expected = identity_sha256(body)
    if policy["policy_identity_sha256"] != expected:
        raise RecipeValidationError("policy identity drift")
    return dict(policy)


def _validate_trusted_authorities(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise RecipeValidationError("trusted_authorities must be an out-of-packet role map")
    _exact_keys(value, set(TRUSTED_AUTHORITY_ROLES), "trusted_authorities")
    return {
        role: _terminal_authority(value[role], f"trusted_authorities.{role}")
        for role in TRUSTED_AUTHORITY_ROLES
    }


def _require_trusted_role(
    packet_authority: Any,
    trusted_authorities: Mapping[str, dict[str, Any]],
    role: str,
) -> dict[str, Any]:
    packet = _terminal_authority(packet_authority, f"{role}.authority")
    if packet != trusted_authorities[role]:
        raise RecipeValidationError(f"{role} authority is not the trusted role-bound authority")
    return packet


def _validate_bindings(bindings: Any, trusted_authorities: Any) -> dict[str, Any]:
    if not isinstance(bindings, dict):
        raise RecipeValidationError("bindings must be object")
    _exact_keys(bindings, {"code", "model", "tokenizer", "d04", "d05", "d06"}, "bindings")
    trusted = _validate_trusted_authorities(trusted_authorities)

    code = bindings["code"]
    if not isinstance(code, dict):
        raise RecipeValidationError("code binding must be object")
    _exact_keys(code, {"git_sha"}, "code")
    _git_sha(code["git_sha"], "code.git_sha")

    model = bindings["model"]
    if not isinstance(model, dict):
        raise RecipeValidationError("model binding must be object")
    _exact_keys(
        model,
        {
            "authority_git_sha",
            "model_spec_sha256",
            "init_spec_sha256",
            "parameter_count",
            "canonical_base",
        },
        "model",
    )
    if not _json_contract_equal(model, MODEL341):
        raise RecipeValidationError("model binding does not equal exact MODEL-341 authority")

    tok = bindings["tokenizer"]
    if not isinstance(tok, dict):
        raise RecipeValidationError("tokenizer binding must be object")
    _exact_keys(tok, {"authority", "identity_sha256", "decision"}, "tokenizer")
    _require_trusted_role(tok["authority"], trusted, "tokenizer")
    _sha256(tok["identity_sha256"], "tokenizer.identity_sha256")
    if tok["decision"] not in {"TRAINED_TOKENIZER", "BYTE_BASELINE_RETAINED"}:
        raise RecipeValidationError("tokenizer decision not terminal")

    d04 = bindings["d04"]
    if not isinstance(d04, dict):
        raise RecipeValidationError("d04 binding must be object")
    _exact_keys(
        d04,
        {
            "authority",
            "unique_nonignored_causal_loss_positions",
            "next_exposure_identity_sha256",
            "train_trace_identity_sha256",
            "corpus_manifest_sha256",
            "split_sha256",
            "packing_sha256",
            "sequence_length",
            "tokenizer_identity_sha256",
        },
        "d04",
    )
    _require_trusted_role(d04["authority"], trusted, "d04")
    _positive_int(
        d04["unique_nonignored_causal_loss_positions"],
        "d04.unique_nonignored_causal_loss_positions",
    )
    for key in (
        "next_exposure_identity_sha256",
        "train_trace_identity_sha256",
        "corpus_manifest_sha256",
        "split_sha256",
        "packing_sha256",
        "tokenizer_identity_sha256",
    ):
        _sha256(d04[key], f"d04.{key}")
    if not _json_contract_equal(d04["sequence_length"], DEFAULT_SEQUENCE_LENGTH):
        raise RecipeValidationError("d04 sequence length does not equal canonical packing contract")
    if d04["packing_sha256"] != PACKING_CONFIG_HASH:
        raise RecipeValidationError("d04 packing identity does not equal canonical packing contract")
    if d04["tokenizer_identity_sha256"] != tok["identity_sha256"]:
        raise RecipeValidationError("d04 tokenizer identity does not match terminal tokenizer")

    d05 = bindings["d05"]
    if not isinstance(d05, dict):
        raise RecipeValidationError("d05 binding must be object")
    _exact_keys(
        d05,
        {
            "authority",
            "status",
            "checkpoint_contract_identity_sha256",
            "fresh_process_resume_equivalence",
            "next_exposure_identity_sha256",
            "train_trace_identity_sha256",
        },
        "d05",
    )
    _require_trusted_role(d05["authority"], trusted, "d05")
    if d05["status"] != "PASS":
        raise RecipeValidationError("d05 status must be PASS")
    for key in (
        "checkpoint_contract_identity_sha256",
        "next_exposure_identity_sha256",
        "train_trace_identity_sha256",
    ):
        _sha256(d05[key], f"d05.{key}")
    if d05["fresh_process_resume_equivalence"] is not True:
        raise RecipeValidationError("d05 fresh-process resume equivalence required")
    if d05["next_exposure_identity_sha256"] != d04["next_exposure_identity_sha256"]:
        raise RecipeValidationError("d05 next-exposure identity does not match d04")
    if d05["train_trace_identity_sha256"] != d04["train_trace_identity_sha256"]:
        raise RecipeValidationError("d05 train-trace identity does not match d04")

    d06 = bindings["d06"]
    if not isinstance(d06, dict):
        raise RecipeValidationError("d06 binding must be object")
    _exact_keys(
        d06,
        {
            "authority",
            "status",
            "evaluation_firewall_identity_sha256",
            "selection_validation_identity_sha256",
            "primary_selection_metric",
            "required_held_out_strata",
            "final_test_sealed",
            "final_test_may_influence_selection",
        },
        "d06",
    )
    _require_trusted_role(d06["authority"], trusted, "d06")
    if d06["status"] != "PASS":
        raise RecipeValidationError("d06 status must be PASS")
    _sha256(d06["evaluation_firewall_identity_sha256"], "d06.evaluation_firewall_identity_sha256")
    _sha256(d06["selection_validation_identity_sha256"], "d06.selection_validation_identity_sha256")
    if d06["primary_selection_metric"] != PRIMARY_SELECTION_METRIC:
        raise RecipeValidationError("d06 primary selection metric does not match LEARN-345")
    if d06["required_held_out_strata"] != list(REQUIRED_HELD_OUT_STRATA):
        raise RecipeValidationError("d06 required held-out strata must be ua/en/code")
    if d06["final_test_sealed"] is not True:
        raise RecipeValidationError("d06 final test must remain sealed")
    if d06["final_test_may_influence_selection"] is not False:
        raise RecipeValidationError("d06 final test may not influence selection")

    return dict(bindings)


def blocked_template(policy: Any) -> dict[str, Any]:
    """Emit the checked-in fail-closed state without accepting any upstream authority."""
    valid = validate_policy(policy)
    return {
        "schema": SESSION_SCHEMA,
        "status": "BLOCKED_TEMPLATE",
        "policy_identity_sha256": valid["policy_identity_sha256"],
        "bindings_identity_sha256": None,
        "trusted_authorities_identity_sha256": None,
        "session_identity_sha256": None,
        "qualified_runtime_unique_loss_positions": 0,
        "training_recipe_status": "BLOCKED",
        "training_authorized": False,
        "compute_authorized": False,
        "authorized_optimized_targets": 0,
        "optimizer_updates_executed": 0,
    }


def bind_terminal_authorities(
    policy: Any,
    bindings: Any,
    *,
    trusted_authorities: Any,
    expected_trusted_authorities_identity_sha256: Any,
) -> dict[str, Any]:
    """Bind externally anchored terminal roles and qualify only the frozen recipe."""
    valid_policy = validate_policy(policy)
    validated_trusted = _validate_trusted_authorities(trusted_authorities)
    expected_trusted_identity = _sha256(
        expected_trusted_authorities_identity_sha256,
        "expected_trusted_authorities_identity_sha256",
    )
    trusted_identity = identity_sha256(validated_trusted)
    if trusted_identity != expected_trusted_identity:
        raise RecipeValidationError(
            "trusted authorities identity does not match external expectation"
        )
    valid_bindings = _validate_bindings(bindings, validated_trusted)

    available = valid_bindings["d04"]["unique_nonignored_causal_loss_positions"]
    runtime_budget = min(REQUESTED_TARGETS, available)
    if runtime_budget < MEANINGFUL_FLOOR:
        raise RecipeValidationError("terminal D04 capacity is below LEARN-345 meaningful floor")

    bindings_identity = identity_sha256(valid_bindings)
    core = {
        "schema": SESSION_SCHEMA,
        "status": "QUALIFIED_RECIPE_ONLY",
        "policy_identity_sha256": valid_policy["policy_identity_sha256"],
        "bindings_identity_sha256": bindings_identity,
        "trusted_authorities_identity_sha256": trusted_identity,
        "qualified_runtime_unique_loss_positions": runtime_budget,
        "training_recipe_status": "QUALIFIED",
        "training_authorized": False,
        "compute_authorized": False,
        "authorized_optimized_targets": 0,
        "optimizer_updates_executed": 0,
    }
    session_identity = identity_sha256(core)
    return {**core, "session_identity_sha256": session_identity}


def readiness_fragment(session: Any, authority: Any) -> dict[str, Any]:
    """Map a qualified recipe session into the existing R01 readiness evidence shape."""
    if not isinstance(session, dict):
        raise RecipeValidationError("session must be object")
    expected = {
        "schema",
        "status",
        "policy_identity_sha256",
        "bindings_identity_sha256",
        "trusted_authorities_identity_sha256",
        "session_identity_sha256",
        "qualified_runtime_unique_loss_positions",
        "training_recipe_status",
        "training_authorized",
        "compute_authorized",
        "authorized_optimized_targets",
        "optimizer_updates_executed",
    }
    _exact_keys(session, expected, "session")
    if session["schema"] != SESSION_SCHEMA or session["status"] != "QUALIFIED_RECIPE_ONLY":
        raise RecipeValidationError("session is not a qualified recipe session")
    for key in (
        "policy_identity_sha256",
        "bindings_identity_sha256",
        "trusted_authorities_identity_sha256",
        "session_identity_sha256",
    ):
        _sha256(session[key], f"session.{key}")
    session_core = {
        key: value for key, value in session.items() if key != "session_identity_sha256"
    }
    if identity_sha256(session_core) != session["session_identity_sha256"]:
        raise RecipeValidationError("session identity drift")
    if session["training_recipe_status"] != "QUALIFIED":
        raise RecipeValidationError("session training recipe not qualified")
    runtime_budget = _positive_int(
        session["qualified_runtime_unique_loss_positions"],
        "session.qualified_runtime_unique_loss_positions",
    )
    if runtime_budget < MEANINGFUL_FLOOR or runtime_budget > REQUESTED_TARGETS:
        raise RecipeValidationError("session runtime budget outside LEARN-345 bounds")
    for key in ("training_authorized", "compute_authorized"):
        if session[key] is not False:
            raise RecipeValidationError(f"{key} must remain false")
    for key in ("authorized_optimized_targets", "optimizer_updates_executed"):
        if _nonnegative_int(session[key], key) != 0:
            raise RecipeValidationError(f"{key} must remain zero")

    valid_authority = _terminal_authority(authority, "training_recipe.authority")
    stopping_policy = {
        "runtime_budget_rule": (
            "min(20000000, terminal_d04_unique_nonignored_causal_loss_positions)"
        ),
        "meaningful_floor": MEANINGFUL_FLOOR,
        "no_replay": True,
        "nan_or_inf": "STOP_FAIL_CLOSED_NO_COMMIT",
        "mandatory_fresh_process_resume_fraction": "0.50",
        "boundaries": list(BOUNDARIES),
    }
    return {
        "authority": valid_authority,
        "status": "QUALIFIED",
        "seed_count": 1,
        "config_sha256": session["session_identity_sha256"],
        "stopping_policy_sha256": identity_sha256(stopping_policy),
        "requested_unique_loss_positions": runtime_budget,
        "requested_total_training_exposures": runtime_budget,
        "max_exposures_per_unique_position": 1,
    }
