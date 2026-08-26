"""Fail-closed authorization gate for the first learned ~20M campaign."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

BLOCKED = "BLOCKED"
READY_FOR_AUTHORIZATION_REQUEST = "READY_FOR_AUTHORIZATION_REQUEST"
TRAINING_AUTHORIZED = "TRAINING_AUTHORIZED"

EXPECTED_AUTHORITY = {
    "r01_merge_main_sha": "a73ab38026cb7849f478cc13ad58b93534a76e2f",
    "r01_campaign_path": "configs/research/r01_20m_to_100m_scaling_campaign_v1.json",
    "r01_campaign_blob_sha1": "c50154db609d41eceb2ffc97912360df567bcc04",
    "r01_campaign_id": "R01-20M-TO-100M-SCALING-V1",
    "model341_sha": "e4ff486fd90802fc123bebf60eed4e59196a98df",
    "modelspec_sha256": "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441",
    "parameter_count": 20_613_440,
    "canonical_base": "random_init",
    "base_lineage": "PRETRAINING_ONLY",
}

_ROOT_KEYS = {
    "schema_version",
    "packet_id",
    "authority",
    "artifacts",
    "gates",
    "recipe",
    "resources",
    "authorizations",
}
_ARTIFACT_KEYS = {
    "code_sha",
    "run_config_sha256",
    "tokenizer_sha256",
    "corpus_sha256",
    "split_sha256",
    "packing_sha256",
    "unique_loss_ledger_sha256",
    "unique_post_pack_causal_loss_positions",
}
_GATE_KEYS = {
    "no_replay_proven",
    "no_replay_accounting_ref",
    "evaluation_decontamination_terminal",
    "evaluation_decontamination_ref",
    "checkpoint_integrity_terminal",
    "checkpoint_integrity_ref",
    "learned_3m_independent_terminal",
    "learned_3m_independent_ref",
    "learned_10m_independent_terminal",
    "learned_10m_independent_ref",
}
_RECIPE_KEYS = {
    "optimizer",
    "scheduler",
    "precision",
    "seeds",
    "target_causal_loss_positions",
    "stop_rule",
    "checkpoint_policy_ref",
}
_RESOURCE_KEYS = {
    "hardware_profile",
    "estimated_flops",
    "estimated_wall_clock_hours",
    "maximum_cost_usd",
    "output_destination",
    "cancellation_rule",
}
_AUTHORIZATION_KEYS = {
    "compute_authorization_ref",
    "training_authorization_ref",
}

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_exact_keys(
    errors: list[str],
    value: Any,
    *,
    expected: set[str],
    label: str,
) -> Mapping[str, Any] | None:
    if not _is_mapping(value):
        errors.append(f"{label}_must_be_object")
        return None
    keys = set(value)
    for key in sorted(expected - keys):
        errors.append(f"{label}_missing_{key}")
    for key in sorted(keys - expected):
        errors.append(f"{label}_unknown_{key}")
    return value


def _require_sha(
    errors: list[str], value: Any, *, length: int, label: str
) -> None:
    pattern = _HEX40 if length == 40 else _HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        errors.append(f"{label}_must_be_lowercase_hex_{length}")


def _require_text(errors: list[str], value: Any, label: str) -> None:
    if not _is_nonempty_text(value):
        errors.append(f"{label}_required")


def assess_learned_20m_launch(packet: Mapping[str, Any]) -> dict[str, Any]:
    """Derive launch state from evidence; never trust a caller-supplied state."""

    blockers: list[str] = []
    authorization_blockers: list[str] = []

    root = _validate_exact_keys(blockers, packet, expected=_ROOT_KEYS, label="packet")
    if root is None:
        return _result(blockers, authorization_blockers)

    if root.get("schema_version") != 1:
        blockers.append("schema_version_must_be_1")
    if root.get("packet_id") != "LEARNED-20M-LAUNCH-V1":
        blockers.append("packet_id_mismatch")

    authority = _validate_exact_keys(
        blockers,
        root.get("authority"),
        expected=set(EXPECTED_AUTHORITY),
        label="authority",
    )
    if authority is not None:
        for key, expected_value in EXPECTED_AUTHORITY.items():
            if authority.get(key) != expected_value:
                blockers.append(f"authority_{key}_mismatch")

    artifacts = _validate_exact_keys(
        blockers,
        root.get("artifacts"),
        expected=_ARTIFACT_KEYS,
        label="artifacts",
    )
    unique_positions: int | None = None
    if artifacts is not None:
        _require_sha(
            blockers,
            artifacts.get("code_sha"),
            length=40,
            label="artifacts_code_sha",
        )
        for key in (
            "run_config_sha256",
            "tokenizer_sha256",
            "corpus_sha256",
            "split_sha256",
            "packing_sha256",
            "unique_loss_ledger_sha256",
        ):
            _require_sha(blockers, artifacts.get(key), length=64, label=f"artifacts_{key}")
        value = artifacts.get("unique_post_pack_causal_loss_positions")
        if not _is_positive_int(value):
            blockers.append("unique_post_pack_causal_loss_positions_must_be_positive_integer")
        else:
            unique_positions = value

    gates = _validate_exact_keys(
        blockers,
        root.get("gates"),
        expected=_GATE_KEYS,
        label="gates",
    )
    if gates is not None:
        if gates.get("no_replay_proven") is not True:
            blockers.append("no_replay_accounting_not_proven")
        _require_text(
            blockers,
            gates.get("no_replay_accounting_ref"),
            "no_replay_accounting_ref",
        )
        for stem in (
            "evaluation_decontamination",
            "checkpoint_integrity",
            "learned_3m_independent",
            "learned_10m_independent",
        ):
            if gates.get(f"{stem}_terminal") is not True:
                blockers.append(f"{stem}_not_terminal")
            _require_text(blockers, gates.get(f"{stem}_ref"), f"{stem}_ref")

    recipe = _validate_exact_keys(
        blockers,
        root.get("recipe"),
        expected=_RECIPE_KEYS,
        label="recipe",
    )
    if recipe is not None:
        for key in (
            "optimizer",
            "scheduler",
            "precision",
            "stop_rule",
            "checkpoint_policy_ref",
        ):
            _require_text(blockers, recipe.get(key), f"recipe_{key}")
        seeds = recipe.get("seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(
                not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                for seed in seeds
            )
            or len(seeds) != len(set(seeds))
        ):
            blockers.append("recipe_seeds_must_be_unique_nonnegative_integers")
        target = recipe.get("target_causal_loss_positions")
        if not _is_positive_int(target):
            blockers.append("target_causal_loss_positions_must_be_positive_integer")
        elif unique_positions is not None and target > unique_positions:
            blockers.append("target_causal_loss_positions_exceed_unique_authorized_positions")

    resources = _validate_exact_keys(
        blockers,
        root.get("resources"),
        expected=_RESOURCE_KEYS,
        label="resources",
    )
    if resources is not None:
        for key in ("hardware_profile", "output_destination", "cancellation_rule"):
            _require_text(blockers, resources.get(key), f"resources_{key}")
        flops = resources.get("estimated_flops")
        if not _is_finite_number(flops) or float(flops) <= 0:
            blockers.append("estimated_flops_must_be_positive_finite")
        hours = resources.get("estimated_wall_clock_hours")
        if not _is_finite_number(hours) or float(hours) <= 0:
            blockers.append("estimated_wall_clock_hours_must_be_positive_finite")
        cost = resources.get("maximum_cost_usd")
        if not _is_finite_number(cost) or float(cost) < 0:
            blockers.append("maximum_cost_usd_must_be_nonnegative_finite")

    authorizations = _validate_exact_keys(
        blockers,
        root.get("authorizations"),
        expected=_AUTHORIZATION_KEYS,
        label="authorizations",
    )
    if authorizations is not None:
        compute_ref = authorizations.get("compute_authorization_ref")
        training_ref = authorizations.get("training_authorization_ref")
        if not _is_nonempty_text(compute_ref):
            authorization_blockers.append("compute_authorization_ref_missing")
        if not _is_nonempty_text(training_ref):
            authorization_blockers.append("training_authorization_ref_missing")
        if (
            _is_nonempty_text(compute_ref)
            and _is_nonempty_text(training_ref)
            and compute_ref.strip() == training_ref.strip()
        ):
            authorization_blockers.append(
                "compute_and_training_authorization_refs_must_be_distinct"
            )

    return _result(blockers, authorization_blockers)


def _result(blockers: list[str], authorization_blockers: list[str]) -> dict[str, Any]:
    blockers = sorted(set(blockers))
    authorization_blockers = sorted(set(authorization_blockers))
    ready = not blockers
    authorized = ready and not authorization_blockers
    if not ready:
        state = BLOCKED
    elif authorized:
        state = TRAINING_AUTHORIZED
    else:
        state = READY_FOR_AUTHORIZATION_REQUEST
    return {
        "schema_version": 1,
        "gate_id": "LEARNED-20M-LAUNCH-GATE-V1",
        "state": state,
        "ready_for_authorization_request": ready,
        "training_authorized": authorized,
        "blockers": blockers,
        "authorization_blockers": authorization_blockers,
    }
