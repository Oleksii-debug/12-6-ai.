"""Fail-closed readiness evaluation for the first learned ~20M Base campaign."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

REPOSITORY = "Oleksii-debug/12-6-ai."
CAMPAIGN_ID = "R01-LEARNED-20M-LAUNCH-V1"
R01_CAMPAIGN_COMMIT_SHA = "a73ab38026cb7849f478cc13ad58b93534a76e2f"
R01_CAMPAIGN_BLOB_SHA1 = "c50154db609d41eceb2ffc97912360df567bcc04"

MODEL341_AUTHORITY = {
    "branch": "model341/20m-candidate-a-20260826",
    "git_sha": "e4ff486fd90802fc123bebf60eed4e59196a98df",
    "modelspec_sha256": "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441",
    "parameter_count": 20_613_440,
    "canonical_base": "random_init",
}

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ReadinessAssessment:
    """Three distinct launch phases; readiness never propagates implicitly."""

    ready_for_local_free_pilot: bool
    ready_for_compute_authorization_request: bool
    material_training_authorized: bool
    local_free_pilot_blockers: tuple[str, ...]
    compute_request_blockers: tuple[str, ...]
    material_training_blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ready_for_local_free_pilot": self.ready_for_local_free_pilot,
            "ready_for_compute_authorization_request": (
                self.ready_for_compute_authorization_request
            ),
            "material_training_authorized": self.material_training_authorized,
            "local_free_pilot_blockers": list(self.local_free_pilot_blockers),
            "compute_request_blockers": list(self.compute_request_blockers),
            "material_training_blockers": list(self.material_training_blockers),
        }


def _is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and _GIT_SHA_RE.fullmatch(value) is not None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _valid_authority_ref(
    value: Any,
    *,
    evidence_kind: str,
    require_workflow: bool = False,
    require_independent: bool = False,
) -> bool:
    """Require a typed, exact-head, non-self-asserted GitHub evidence reference."""
    if not isinstance(value, dict):
        return False
    if value.get("repository") != REPOSITORY:
        return False
    if value.get("evidence_kind") != evidence_kind:
        return False
    if not _is_git_sha(value.get("git_sha")):
        return False
    if value.get("observed_head_sha") != value.get("git_sha"):
        return False
    if not _is_sha256(value.get("evidence_sha256")):
        return False
    if value.get("terminal") is not True:
        return False
    if value.get("self_asserted") is not False:
        return False
    if require_workflow:
        run_id = value.get("workflow_run_id")
        if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
            return False
        if value.get("workflow_conclusion") != "success":
            return False
        if value.get("workflow_run_head_sha") != value.get("git_sha"):
            return False
    if require_independent:
        if value.get("independent") is not True:
            return False
        producer = value.get("producer_identity")
        verifier = value.get("verifier_identity")
        if not isinstance(producer, str) or not producer:
            return False
        if not isinstance(verifier, str) or not verifier or verifier == producer:
            return False
    return True


def _require_identity(blockers: list[str], value: Any, name: str) -> None:
    if not _is_sha256(value):
        blockers.append(name)


def _require_authority(
    blockers: list[str],
    value: Any,
    name: str,
    *,
    evidence_kind: str,
    require_workflow: bool = False,
    require_independent: bool = False,
) -> None:
    if not _valid_authority_ref(
        value,
        evidence_kind=evidence_kind,
        require_workflow=require_workflow,
        require_independent=require_independent,
    ):
        blockers.append(name)


def _validate_envelope(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("schema_version_mismatch")
    if data.get("campaign_id") != CAMPAIGN_ID:
        errors.append("campaign_id_mismatch")
    if data.get("r01_campaign_commit_sha") != R01_CAMPAIGN_COMMIT_SHA:
        errors.append("r01_campaign_commit_authority_drift")
    if data.get("r01_campaign_blob_sha1") != R01_CAMPAIGN_BLOB_SHA1:
        errors.append("r01_campaign_authority_drift")

    model = data.get("model_authority")
    if not isinstance(model, dict):
        errors.append("model_authority_missing")
    else:
        for key, expected in MODEL341_AUTHORITY.items():
            if model.get(key) != expected:
                errors.append(f"model_authority_{key}_mismatch")

    boundaries = data.get("truth_boundary")
    if not isinstance(boundaries, dict):
        errors.append("truth_boundary_missing")
    else:
        for key in (
            "foreign_pretrained_weights_used",
            "alignment_or_posttraining_mixed_into_base",
            "final_test_payload_consumed",
            "model_training_executed_by_this_package",
            "paid_compute_executed_by_this_package",
        ):
            if boundaries.get(key) is not False:
                errors.append(f"truth_boundary_{key}_must_be_false")
    return errors


def assess_learned20m_readiness(data: dict[str, Any]) -> ReadinessAssessment:
    """Assess launch readiness without granting authority from partial evidence."""
    envelope_errors = _validate_envelope(data)
    evidence = data.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
        envelope_errors.append("evidence_missing")

    local = list(envelope_errors)

    code = evidence.get("code") if isinstance(evidence.get("code"), dict) else {}
    code_sha = code.get("git_sha")
    if not _is_git_sha(code_sha):
        local.append("exact_code_sha_missing")
    _require_authority(
        local,
        code.get("authority"),
        "qualified_integration_head_authority_missing",
        evidence_kind="qualified_integration_head",
        require_workflow=True,
    )
    code_authority = code.get("authority")
    if isinstance(code_authority, dict) and code_authority.get("git_sha") != code_sha:
        local.append("qualified_integration_head_sha_mismatch")

    corpus = evidence.get("corpus") if isinstance(evidence.get("corpus"), dict) else {}
    _require_identity(local, corpus.get("manifest_sha256"), "corpus_manifest_missing")
    _require_identity(local, corpus.get("split_sha256"), "corpus_split_identity_missing")
    _require_identity(local, corpus.get("packing_sha256"), "packing_identity_missing")
    if corpus.get("two_clean_builds_identical") is not True:
        local.append("two_clean_builds_not_proven")
    _require_authority(
        local,
        corpus.get("authority"),
        "terminal_corpus_authority_missing",
        evidence_kind="research_corpus_v1",
        require_workflow=True,
    )

    tokenizer = (
        evidence.get("tokenizer") if isinstance(evidence.get("tokenizer"), dict) else {}
    )
    _require_identity(local, tokenizer.get("identity_sha256"), "tokenizer_identity_missing")
    if tokenizer.get("decision") not in {"TRAINED_TOKENIZER", "BYTE_BASELINE_RETAINED"}:
        local.append("tokenizer_decision_not_terminal")
    _require_authority(
        local,
        tokenizer.get("authority"),
        "terminal_tokenizer_authority_missing",
        evidence_kind="tokenizer_decision",
        require_workflow=True,
    )

    ledger = evidence.get("loss_ledger") if isinstance(evidence.get("loss_ledger"), dict) else {}
    _require_identity(local, ledger.get("identity_sha256"), "unique_loss_ledger_missing")
    positions = ledger.get("unique_causal_loss_positions")
    if isinstance(positions, bool) or not isinstance(positions, int) or positions <= 0:
        local.append("unique_loss_positions_not_positive")
    _require_authority(
        local,
        ledger.get("authority"),
        "terminal_unique_loss_ledger_authority_missing",
        evidence_kind="unique_post_pack_loss_ledger",
        require_workflow=True,
    )
    _require_authority(
        local,
        ledger.get("data_budget_authority"),
        "data_budget_authority_missing",
        evidence_kind="terminal_data_budget_qualification",
        require_workflow=True,
    )
    if ledger.get("data_budget_status") != "QUALIFIED":
        local.append("data_budget_not_qualified")

    checkpoint = (
        evidence.get("checkpoint_integrity")
        if isinstance(evidence.get("checkpoint_integrity"), dict)
        else {}
    )
    _require_authority(
        local,
        checkpoint.get("authority"),
        "checkpoint_integrity_authority_missing",
        evidence_kind="d05_checkpoint_integrity",
        require_workflow=True,
    )
    if checkpoint.get("status") != "PASS":
        local.append("checkpoint_integrity_not_terminal_pass")

    evaluation = (
        evidence.get("evaluation") if isinstance(evidence.get("evaluation"), dict) else {}
    )
    _require_authority(
        local,
        evaluation.get("firewall_authority"),
        "evaluation_firewall_authority_missing",
        evidence_kind="evaluation_firewall",
        require_workflow=True,
    )
    _require_authority(
        local,
        evaluation.get("selection_validation_authority"),
        "selection_validation_authority_missing",
        evidence_kind="selection_validation",
        require_workflow=True,
    )
    if evaluation.get("status") != "PASS":
        local.append("evaluation_boundary_not_terminal_pass")

    recipe = (
        evidence.get("training_recipe")
        if isinstance(evidence.get("training_recipe"), dict)
        else {}
    )
    _require_authority(
        local,
        recipe.get("authority"),
        "training_recipe_authority_missing",
        evidence_kind="learned_20m_training_recipe",
        require_workflow=True,
    )
    if recipe.get("status") != "QUALIFIED":
        local.append("training_recipe_not_qualified")
    if recipe.get("seed_count", 0) < 1:
        local.append("training_seed_plan_missing")
    _require_identity(local, recipe.get("config_sha256"), "training_config_identity_missing")
    _require_identity(local, recipe.get("stopping_policy_sha256"), "stopping_policy_missing")
    requested_positions = recipe.get("requested_unique_loss_positions")
    if (
        isinstance(requested_positions, bool)
        or not isinstance(requested_positions, int)
        or requested_positions <= 0
    ):
        local.append("requested_unique_loss_positions_not_positive")
    elif (
        isinstance(positions, int)
        and not isinstance(positions, bool)
        and positions > 0
        and requested_positions > positions
    ):
        local.append("requested_unique_loss_positions_exceed_ledger")

    local = sorted(set(local))

    compute = list(local)
    pilot = evidence.get("bounded_pilot") if isinstance(evidence.get("bounded_pilot"), dict) else {}
    _require_authority(
        compute,
        pilot.get("authority"),
        "bounded_pilot_authority_missing",
        evidence_kind="bounded_learned_20m_pilot",
        require_workflow=True,
    )
    if pilot.get("status") != "PASS":
        compute.append("bounded_pilot_not_terminal_pass")
    for key in ("numerics_finite", "resume_equivalent", "loss_trajectory_acceptable"):
        if pilot.get(key) is not True:
            compute.append(f"bounded_pilot_{key}_not_proven")

    cost = evidence.get("cost_envelope") if isinstance(evidence.get("cost_envelope"), dict) else {}
    _require_authority(
        compute,
        cost.get("authority"),
        "cost_envelope_authority_missing",
        evidence_kind="material_compute_cost_envelope",
    )
    if cost.get("status") != "ESTIMATED":
        compute.append("cost_envelope_not_estimated")
    maximum_cost = cost.get("maximum_cost_usd")
    if isinstance(maximum_cost, bool) or not isinstance(maximum_cost, (int, float)):
        compute.append("maximum_cost_missing")
    elif maximum_cost <= 0:
        compute.append("maximum_cost_not_positive")

    audit = (
        evidence.get("independent_audit")
        if isinstance(evidence.get("independent_audit"), dict)
        else {}
    )
    _require_authority(
        compute,
        audit.get("authority"),
        "independent_audit_authority_missing",
        evidence_kind="independent_launch_audit",
        require_workflow=True,
        require_independent=True,
    )
    if audit.get("status") not in {"PASS", "PASS_WITH_NOTES"}:
        compute.append("independent_audit_not_terminal_pass")

    compute = sorted(set(compute))

    material = list(compute)
    authorization = (
        evidence.get("compute_authorization")
        if isinstance(evidence.get("compute_authorization"), dict)
        else {}
    )
    _require_authority(
        material,
        authorization.get("authority"),
        "compute_authorization_authority_missing",
        evidence_kind="material_compute_authorization",
    )
    if authorization.get("status") != "COMPUTE_AUTHORIZED":
        material.append("compute_not_explicitly_authorized")
    if authorization.get("scope") != "LEARNED_20M_MATERIAL_COMPUTE":
        material.append("compute_authorization_scope_mismatch")
    if authorization.get("authorized_by_owner") is not True:
        material.append("compute_owner_authorization_missing")
    authorized_limit = authorization.get("maximum_cost_usd")
    maximum_cost = cost.get("maximum_cost_usd")
    if isinstance(authorized_limit, bool) or not isinstance(authorized_limit, (int, float)):
        material.append("authorized_cost_limit_missing")
    elif isinstance(maximum_cost, (int, float)) and not isinstance(maximum_cost, bool):
        if authorized_limit < maximum_cost:
            material.append("authorized_cost_below_estimated_maximum")

    training_authorization = (
        evidence.get("training_authorization")
        if isinstance(evidence.get("training_authorization"), dict)
        else {}
    )
    _require_authority(
        material,
        training_authorization.get("authority"),
        "training_authorization_authority_missing",
        evidence_kind="learned_20m_training_authorization",
    )
    if training_authorization.get("status") != "TRAINING_AUTHORIZED":
        material.append("training_not_explicitly_authorized")
    if training_authorization.get("scope") != "LEARNED_20M_MATERIAL_TRAINING":
        material.append("training_authorization_scope_mismatch")
    if training_authorization.get("authorized_by_owner") is not True:
        material.append("training_owner_authorization_missing")
    if training_authorization.get("training_config_sha256") != recipe.get("config_sha256"):
        material.append("training_authorization_config_mismatch")
    compute_ref = authorization.get("authority")
    compute_evidence_sha = (
        compute_ref.get("evidence_sha256") if isinstance(compute_ref, dict) else None
    )
    if training_authorization.get("compute_authorization_evidence_sha256") != compute_evidence_sha:
        material.append("training_authorization_compute_binding_mismatch")

    material = sorted(set(material))
    return ReadinessAssessment(
        ready_for_local_free_pilot=not local,
        ready_for_compute_authorization_request=not compute,
        material_training_authorized=not material,
        local_free_pilot_blockers=tuple(local),
        compute_request_blockers=tuple(compute),
        material_training_blockers=tuple(material),
    )
