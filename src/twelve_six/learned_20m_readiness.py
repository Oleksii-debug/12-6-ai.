from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REPOSITORY = "Oleksii-debug/12-6-ai."
MODEL341_SHA = "e4ff486fd90802fc123bebf60eed4e59196a98df"
MODEL341_MODELSPEC_SHA256 = (
    "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
)
MODEL341_PARAMETER_COUNT = 20_613_440
R01_MERGE_SHA = "a73ab38026cb7849f478cc13ad58b93534a76e2f"
R01_CAMPAIGN_ID = "R01-20M-TO-100M-SCALING-V1"
R01_CAMPAIGN_BLOB_SHA1 = "c50154db609d41eceb2ffc97912360df567bcc04"

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_AUTHORITIES = {
    "corpus": "DATA_CORPUS",
    "split": "DATA_SPLIT",
    "packing": "D04_PACKING",
    "tokenizer": "D04_TOKENIZER",
    "data_budget": "R01_DATA_BUDGET",
    "d05_checkpoint": "D05_CHECKPOINT",
    "evaluation_firewall": "D06_EVALUATION_FIREWALL",
    "selection_validation": "D06_SELECTION_VALIDATION",
}


@dataclass(frozen=True)
class ReadinessResult:
    local_pilot_ready: bool
    authorization_request_ready: bool
    material_training_authorized: bool
    local_pilot_blockers: tuple[str, ...]
    authorization_request_blockers: tuple[str, ...]
    material_training_blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_pilot_ready": self.local_pilot_ready,
            "authorization_request_ready": self.authorization_request_ready,
            "material_training_authorized": self.material_training_authorized,
            "blockers": {
                "local_pilot": list(self.local_pilot_blockers),
                "authorization_request": list(self.authorization_request_blockers),
                "material_training": list(self.material_training_blockers),
            },
        }


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def verify_r01_campaign_bytes(payload: bytes) -> list[str]:
    errors: list[str] = []
    if git_blob_sha1(payload) != R01_CAMPAIGN_BLOB_SHA1:
        errors.append("r01_campaign_blob_mismatch")
        return errors

    try:
        campaign = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return ["r01_campaign_not_valid_utf8_json"]

    if not isinstance(campaign, dict):
        return ["r01_campaign_root_not_object"]
    if campaign.get("campaign_id") != R01_CAMPAIGN_ID:
        errors.append("r01_campaign_id_mismatch")
    if campaign.get("status") != "CANDIDATE_PLANNING_ONLY":
        errors.append("r01_campaign_status_not_planning_only")
    authority = campaign.get("authority")
    if not isinstance(authority, dict):
        errors.append("r01_campaign_authority_missing")
    else:
        if authority.get("model341_sha") != MODEL341_SHA:
            errors.append("r01_campaign_model341_sha_mismatch")
        if authority.get("modelspec_sha256") != MODEL341_MODELSPEC_SHA256:
            errors.append("r01_campaign_modelspec_mismatch")
        if authority.get("parameter_count") != MODEL341_PARAMETER_COUNT:
            errors.append("r01_campaign_parameter_count_mismatch")
    boundaries = campaign.get("hard_boundaries")
    if not isinstance(boundaries, dict):
        errors.append("r01_campaign_boundaries_missing")
    else:
        for key in ("paid_compute_authorized", "long_training_authorized"):
            if boundaries.get(key) is not False:
                errors.append(f"r01_campaign_{key}_must_be_false")
    return errors


def verify_r01_campaign_path(path: Path) -> list[str]:
    try:
        payload = path.read_bytes()
    except OSError:
        return ["r01_campaign_file_unreadable"]
    return verify_r01_campaign_bytes(payload)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
    )


def _evidence_errors(value: Any, *, expected_authority: str, prefix: str) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{prefix}_missing"]

    errors: list[str] = []
    if value.get("authority") != expected_authority:
        errors.append(f"{prefix}_authority_mismatch")
    if value.get("repository") != REPOSITORY:
        errors.append(f"{prefix}_repository_mismatch")
    if not isinstance(value.get("source_ref"), str) or not value.get("source_ref"):
        errors.append(f"{prefix}_source_ref_missing")
    if not isinstance(value.get("source_sha"), str) or not _SHA1_RE.fullmatch(
        value.get("source_sha", "")
    ):
        errors.append(f"{prefix}_source_sha_invalid")
    if not isinstance(value.get("identity_sha256"), str) or not _SHA256_RE.fullmatch(
        value.get("identity_sha256", "")
    ):
        errors.append(f"{prefix}_identity_sha256_invalid")
    if value.get("terminal_state") != "PASS":
        errors.append(f"{prefix}_not_terminal_pass")
    if value.get("self_asserted") is not False:
        errors.append(f"{prefix}_self_asserted_or_unset")
    if value.get("superseded") is not False:
        errors.append(f"{prefix}_superseded_or_unset")
    return errors


def _base_errors(packet: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if packet.get("schema_version") != 1:
        errors.append("packet_schema_version_mismatch")
    if packet.get("repository") != REPOSITORY:
        errors.append("packet_repository_mismatch")

    model = packet.get("model")
    if not isinstance(model, Mapping):
        errors.append("model_authority_missing")
    else:
        if model.get("source_sha") != MODEL341_SHA:
            errors.append("model341_sha_mismatch")
        if model.get("modelspec_sha256") != MODEL341_MODELSPEC_SHA256:
            errors.append("model341_modelspec_mismatch")
        if model.get("parameter_count") != MODEL341_PARAMETER_COUNT:
            errors.append("model341_parameter_count_mismatch")
        if model.get("canonical_base") != "random_init":
            errors.append("model341_not_random_init")

    r01 = packet.get("r01")
    if not isinstance(r01, Mapping):
        errors.append("r01_binding_missing")
    else:
        if r01.get("merge_sha") != R01_MERGE_SHA:
            errors.append("r01_merge_sha_mismatch")
        if r01.get("campaign_id") != R01_CAMPAIGN_ID:
            errors.append("r01_campaign_id_mismatch")
        if r01.get("campaign_blob_sha1") != R01_CAMPAIGN_BLOB_SHA1:
            errors.append("r01_campaign_blob_mismatch")

    return errors


def _local_pilot_errors(packet: Mapping[str, Any]) -> list[str]:
    errors = _base_errors(packet)

    authorities = packet.get("authorities")
    if not isinstance(authorities, Mapping):
        errors.append("authorities_missing")
        authorities = {}
    for key, expected in _REQUIRED_AUTHORITIES.items():
        errors.extend(
            _evidence_errors(
                authorities.get(key),
                expected_authority=expected,
                prefix=f"authority_{key}",
            )
        )

    corpus = authorities.get("corpus")
    split = authorities.get("split")
    packing = authorities.get("packing")
    tokenizer = authorities.get("tokenizer")
    data_budget = authorities.get("data_budget")
    d05 = authorities.get("d05_checkpoint")

    corpus_identity = corpus.get("corpus_identity_sha256") if isinstance(corpus, Mapping) else None
    split_identity = split.get("split_identity_sha256") if isinstance(split, Mapping) else None
    packing_identity = (
        packing.get("packing_identity_sha256") if isinstance(packing, Mapping) else None
    )
    tokenizer_identity = (
        tokenizer.get("tokenizer_identity_sha256") if isinstance(tokenizer, Mapping) else None
    )
    for name, value in (
        ("corpus_identity", corpus_identity),
        ("split_identity", split_identity),
        ("packing_identity", packing_identity),
        ("tokenizer_identity", tokenizer_identity),
    ):
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            errors.append(f"{name}_invalid")
    if isinstance(split, Mapping) and split.get("corpus_identity_sha256") != corpus_identity:
        errors.append("split_not_bound_to_corpus")
    if isinstance(packing, Mapping) and packing.get("split_identity_sha256") != split_identity:
        errors.append("packing_not_bound_to_split")
    if isinstance(d05, Mapping) and d05.get("model_sha") != MODEL341_SHA:
        errors.append("d05_not_bound_to_model341")

    ledger = packet.get("unique_loss_ledger")
    errors.extend(
        _evidence_errors(
            ledger,
            expected_authority="DATA_UNIQUE_LOSS_LEDGER",
            prefix="unique_loss_ledger",
        )
    )
    ledger_positions = 0
    if isinstance(ledger, Mapping):
        ledger_positions = ledger.get("unique_causal_loss_positions", 0)
        if not _is_positive_int(ledger_positions):
            errors.append("unique_loss_positions_must_be_positive")
            ledger_positions = 0
        if ledger.get("post_pack") is not True:
            errors.append("unique_loss_ledger_must_be_post_pack")
        if ledger.get("no_replay") is not True:
            errors.append("unique_loss_ledger_must_prove_no_replay")
        if ledger.get("non_ignored_targets_only") is not True:
            errors.append("unique_loss_ledger_must_count_non_ignored_targets_only")
        for field, expected in (
            ("corpus_identity_sha256", corpus_identity),
            ("split_identity_sha256", split_identity),
            ("packing_identity_sha256", packing_identity),
            ("tokenizer_identity_sha256", tokenizer_identity),
        ):
            if ledger.get(field) != expected:
                errors.append(f"unique_loss_ledger_{field}_mismatch")
        if isinstance(data_budget, Mapping):
            if data_budget.get("qualified_unique_loss_positions") != ledger_positions:
                errors.append("data_budget_not_bound_to_unique_loss_positions")
            if data_budget.get("unique_loss_ledger_identity_sha256") != ledger.get(
                "identity_sha256"
            ):
                errors.append("data_budget_not_bound_to_unique_loss_ledger")

    recipe = packet.get("training_recipe")
    errors.extend(
        _evidence_errors(
            recipe,
            expected_authority="TRAIN_RECIPE",
            prefix="training_recipe",
        )
    )
    if isinstance(recipe, Mapping):
        for field in ("optimizer", "scheduler", "precision", "stopping_rule"):
            if not isinstance(recipe.get(field), str) or not recipe.get(field):
                errors.append(f"training_recipe_{field}_missing")
        seeds = recipe.get("seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(not _is_positive_int(seed) for seed in seeds)
            or len(seeds) != len(set(seeds))
        ):
            errors.append("training_recipe_seeds_invalid")
        if recipe.get("model_sha") != MODEL341_SHA:
            errors.append("training_recipe_not_bound_to_model341")
        if recipe.get("packing_identity_sha256") != packing_identity:
            errors.append("training_recipe_not_bound_to_packing")
        if recipe.get("tokenizer_identity_sha256") != tokenizer_identity:
            errors.append("training_recipe_not_bound_to_tokenizer")
        if isinstance(ledger, Mapping) and recipe.get(
            "unique_loss_ledger_identity_sha256"
        ) != ledger.get("identity_sha256"):
            errors.append("training_recipe_not_bound_to_unique_loss_ledger")
        recipe_positions = recipe.get("max_unique_loss_positions")
        if not _is_positive_int(recipe_positions):
            errors.append("training_recipe_budget_invalid")
        elif ledger_positions and recipe_positions > ledger_positions:
            errors.append("training_recipe_exceeds_unique_loss_ledger")

    pilot = packet.get("pilot_plan")
    if not isinstance(pilot, Mapping):
        errors.append("pilot_plan_missing")
    else:
        if pilot.get("local_free_only") is not True:
            errors.append("pilot_plan_not_local_free")
        if pilot.get("material_compute") is not False:
            errors.append("pilot_plan_material_compute_must_be_false")
        if not _is_positive_int(pilot.get("max_optimizer_updates")):
            errors.append("pilot_plan_optimizer_update_bound_invalid")
        pilot_positions = pilot.get("max_unique_loss_positions")
        if not _is_positive_int(pilot_positions):
            errors.append("pilot_plan_loss_position_bound_invalid")
        elif ledger_positions and pilot_positions > ledger_positions:
            errors.append("pilot_plan_exceeds_unique_loss_ledger")

    return errors


def _authorization_request_errors(packet: Mapping[str, Any]) -> list[str]:
    errors = _local_pilot_errors(packet)

    bounded = packet.get("bounded_pilot")
    errors.extend(
        _evidence_errors(
            bounded,
            expected_authority="TRAIN_BOUNDED_PILOT",
            prefix="bounded_pilot",
        )
    )
    if isinstance(bounded, Mapping):
        recipe = packet.get("training_recipe")
        ledger = packet.get("unique_loss_ledger")
        if bounded.get("model_sha") != MODEL341_SHA:
            errors.append("bounded_pilot_not_bound_to_model341")
        if isinstance(recipe, Mapping) and bounded.get(
            "training_recipe_identity_sha256"
        ) != recipe.get("identity_sha256"):
            errors.append("bounded_pilot_not_bound_to_training_recipe")
        if isinstance(ledger, Mapping) and bounded.get(
            "unique_loss_ledger_identity_sha256"
        ) != ledger.get("identity_sha256"):
            errors.append("bounded_pilot_not_bound_to_unique_loss_ledger")
        for field in (
            "numerics_pass",
            "checkpoint_resume_pass",
            "evaluation_firewall_pass",
            "loss_trajectory_observed",
        ):
            if bounded.get(field) is not True:
                errors.append(f"bounded_pilot_{field}_required")

    envelope = packet.get("cost_envelope")
    errors.extend(
        _evidence_errors(
            envelope,
            expected_authority="C01_COST_ENVELOPE",
            prefix="cost_envelope",
        )
    )
    if isinstance(envelope, Mapping):
        if not _is_positive_int(envelope.get("estimated_flops")):
            errors.append("cost_envelope_estimated_flops_invalid")
        if not _is_positive_int(envelope.get("estimated_wall_clock_seconds")):
            errors.append("cost_envelope_wall_clock_invalid")
        if not _is_nonnegative_number(envelope.get("max_cost_usd")):
            errors.append("cost_envelope_max_cost_invalid")
        if not isinstance(envelope.get("hardware_profile"), str) or not envelope.get(
            "hardware_profile"
        ):
            errors.append("cost_envelope_hardware_profile_missing")

    audit = packet.get("independent_audit")
    errors.extend(
        _evidence_errors(
            audit,
            expected_authority="INDEPENDENT_AUDIT",
            prefix="independent_audit",
        )
    )
    if isinstance(audit, Mapping) and audit.get("verdict") not in {
        "PASS",
        "PASS_WITH_NOTES",
    }:
        errors.append("independent_audit_verdict_not_passing")

    return errors


def _material_training_errors(packet: Mapping[str, Any]) -> list[str]:
    errors = _authorization_request_errors(packet)

    compute = packet.get("compute_authorization")
    errors.extend(
        _evidence_errors(
            compute,
            expected_authority="C01_COMPUTE_AUTHORIZATION",
            prefix="compute_authorization",
        )
    )
    training = packet.get("training_authorization")
    errors.extend(
        _evidence_errors(
            training,
            expected_authority="TRAINING_AUTHORIZATION",
            prefix="training_authorization",
        )
    )

    envelope = packet.get("cost_envelope")
    recipe = packet.get("training_recipe")
    if isinstance(compute, Mapping):
        if isinstance(envelope, Mapping) and compute.get(
            "cost_envelope_identity_sha256"
        ) != envelope.get("identity_sha256"):
            errors.append("compute_authorization_not_bound_to_cost_envelope")
        if isinstance(recipe, Mapping) and compute.get(
            "training_recipe_identity_sha256"
        ) != recipe.get("identity_sha256"):
            errors.append("compute_authorization_not_bound_to_training_recipe")
        authorized_cost = compute.get("max_cost_usd")
        if not _is_nonnegative_number(authorized_cost):
            errors.append("compute_authorization_cost_invalid")
        elif isinstance(envelope, Mapping):
            requested_cost = envelope.get("max_cost_usd")
            if _is_nonnegative_number(requested_cost) and authorized_cost < requested_cost:
                errors.append("compute_authorization_below_cost_envelope")

        authorized_positions = compute.get("max_unique_loss_positions")
        if not _is_positive_int(authorized_positions):
            errors.append("compute_authorization_loss_positions_invalid")
        elif isinstance(recipe, Mapping):
            recipe_positions = recipe.get("max_unique_loss_positions")
            if _is_positive_int(recipe_positions) and authorized_positions < recipe_positions:
                errors.append("compute_authorization_below_training_recipe")

    if isinstance(training, Mapping):
        if isinstance(recipe, Mapping) and training.get(
            "training_recipe_identity_sha256"
        ) != recipe.get("identity_sha256"):
            errors.append("training_authorization_not_bound_to_training_recipe")
        if isinstance(compute, Mapping) and training.get(
            "compute_authorization_identity_sha256"
        ) != compute.get("identity_sha256"):
            errors.append("training_authorization_not_bound_to_compute_authorization")
        authorized_positions = training.get("max_unique_loss_positions")
        if not _is_positive_int(authorized_positions):
            errors.append("training_authorization_loss_positions_invalid")
        elif isinstance(recipe, Mapping):
            recipe_positions = recipe.get("max_unique_loss_positions")
            if _is_positive_int(recipe_positions) and authorized_positions < recipe_positions:
                errors.append("training_authorization_below_training_recipe")
        if training.get("long_training_authorized") is not True:
            errors.append("training_authorization_long_training_flag_missing")

    return errors


def evaluate_learned_20m_readiness(packet: Mapping[str, Any]) -> ReadinessResult:
    local_errors = tuple(dict.fromkeys(_local_pilot_errors(packet)))
    request_errors = tuple(dict.fromkeys(_authorization_request_errors(packet)))
    material_errors = tuple(dict.fromkeys(_material_training_errors(packet)))
    return ReadinessResult(
        local_pilot_ready=not local_errors,
        authorization_request_ready=not request_errors,
        material_training_authorized=not material_errors,
        local_pilot_blockers=local_errors,
        authorization_request_blockers=request_errors,
        material_training_blockers=material_errors,
    )
