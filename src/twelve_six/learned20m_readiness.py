"""Fail-closed readiness evaluation for the first learned ~20M Base campaign."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

REPOSITORY = "Oleksii-debug/12-6-ai."
CAMPAIGN_ID = "R01-LEARNED-20M-LAUNCH-V1"
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
_METADATA_UNSET = object()

_SCIENTIFIC_ROLE_WORKFLOW_REQUIRED = {
    "code": True,
    "corpus": True,
    "tokenizer": True,
    "loss_ledger": True,
    "data_budget": True,
    "checkpoint_integrity": True,
    "evaluation_firewall": True,
    "selection_validation": True,
    "training_recipe": True,
    "bounded_pilot": True,
    "learned_3m": True,
    "learned_10m": True,
    "cost_envelope": False,
    "independent_audit": True,
}

_SCIENTIFIC_METADATA_KEYS = {
    "code": frozenset({"git_sha"}),
    "corpus": frozenset(
        {"manifest_sha256", "split_sha256", "packing_sha256", "two_clean_builds_identical"}
    ),
    "tokenizer": frozenset({"identity_sha256", "decision"}),
    "loss_ledger": frozenset({"identity_sha256", "unique_causal_loss_positions"}),
    "data_budget": frozenset(
        {"ledger_identity_sha256", "unique_causal_loss_positions", "data_budget_status"}
    ),
    "checkpoint_integrity": frozenset({"status"}),
    "evaluation_firewall": frozenset({"status"}),
    "selection_validation": frozenset({"status"}),
    "training_recipe": frozenset(
        {
            "status",
            "seed_count",
            "config_sha256",
            "stopping_policy_sha256",
            "requested_unique_loss_positions",
            "requested_total_training_exposures",
            "max_exposures_per_unique_position",
        }
    ),
    "bounded_pilot": frozenset(
        {"status", "numerics_finite", "resume_equivalent", "loss_trajectory_acceptable"}
    ),
    "learned_3m": frozenset({"status"}),
    "learned_10m": frozenset({"status"}),
    "cost_envelope": frozenset({"status", "maximum_cost_usd"}),
    "independent_audit": frozenset({"status"}),
}


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


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, int):
        return True
    return math.isfinite(value)


def _is_finite_positive_number(value: Any) -> bool:
    return _is_finite_number(value) and value > 0


def _valid_authority_ref(
    value: Any,
    *,
    require_workflow: bool = False,
    strict: bool = False,
) -> bool:
    """Require a machine-addressable exact-head GitHub evidence reference."""
    if not isinstance(value, dict):
        return False
    if strict:
        expected = {"repository", "git_sha", "evidence_sha256", "terminal"}
        if require_workflow:
            expected.update({"workflow_run_id", "workflow_conclusion"})
        if set(value) != expected:
            return False
    if value.get("repository") != REPOSITORY:
        return False
    if not _is_git_sha(value.get("git_sha")):
        return False
    if not _is_sha256(value.get("evidence_sha256")):
        return False
    if value.get("terminal") is not True:
        return False
    if require_workflow:
        run_id = value.get("workflow_run_id")
        if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
            return False
        if value.get("workflow_conclusion") != "success":
            return False
    return True


def _canonicalize_scientific_metadata(value: Any) -> Any:
    """Normalize readiness metadata without decimal-size or bool/int ambiguity."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        sign = "-" if value < 0 else ""
        return {"__int_hex__": sign + format(abs(value), "x")}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific metadata floats must be finite")
        return {"__float_hex__": value.hex()}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("scientific metadata keys must be strings")
        return {
            key: _canonicalize_scientific_metadata(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_scientific_metadata(item) for item in value]
    raise TypeError(f"unsupported scientific metadata type: {type(value).__name__}")


def scientific_metadata_sha256(metadata: Any) -> str | None:
    """Return a deterministic digest for exact readiness-consumed metadata."""
    try:
        canonical = _canonicalize_scientific_metadata(metadata)
        encoded = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def scientific_role_metadata(role: str, evidence: Any) -> dict[str, Any] | None:
    """Project exactly the packet metadata consumed for one scientific role.

    This projection is not trust. A caller must compare a token derived from an
    independently trusted copy of this metadata against the externally verified
    token supplied to :func:`assess_learned20m_readiness`.
    """
    if not isinstance(role, str) or not role.strip() or not isinstance(evidence, dict):
        return None
    normalized_role = role.strip()

    def block(name: str) -> dict[str, Any]:
        value = evidence.get(name)
        return value if isinstance(value, dict) else {}

    if normalized_role == "code":
        item = block("code")
        return {"git_sha": item.get("git_sha")}
    if normalized_role == "corpus":
        item = block("corpus")
        return {
            "manifest_sha256": item.get("manifest_sha256"),
            "split_sha256": item.get("split_sha256"),
            "packing_sha256": item.get("packing_sha256"),
            "two_clean_builds_identical": item.get("two_clean_builds_identical"),
        }
    if normalized_role == "tokenizer":
        item = block("tokenizer")
        return {
            "identity_sha256": item.get("identity_sha256"),
            "decision": item.get("decision"),
        }
    if normalized_role == "loss_ledger":
        item = block("loss_ledger")
        return {
            "identity_sha256": item.get("identity_sha256"),
            "unique_causal_loss_positions": item.get("unique_causal_loss_positions"),
        }
    if normalized_role == "data_budget":
        item = block("loss_ledger")
        return {
            "ledger_identity_sha256": item.get("identity_sha256"),
            "unique_causal_loss_positions": item.get("unique_causal_loss_positions"),
            "data_budget_status": item.get("data_budget_status"),
        }
    if normalized_role == "checkpoint_integrity":
        item = block("checkpoint_integrity")
        return {"status": item.get("status")}
    if normalized_role in {"evaluation_firewall", "selection_validation"}:
        item = block("evaluation")
        return {"status": item.get("status")}
    if normalized_role == "training_recipe":
        item = block("training_recipe")
        return {
            "status": item.get("status"),
            "seed_count": item.get("seed_count"),
            "config_sha256": item.get("config_sha256"),
            "stopping_policy_sha256": item.get("stopping_policy_sha256"),
            "requested_unique_loss_positions": item.get("requested_unique_loss_positions"),
            "requested_total_training_exposures": item.get(
                "requested_total_training_exposures"
            ),
            "max_exposures_per_unique_position": item.get(
                "max_exposures_per_unique_position"
            ),
        }
    if normalized_role == "bounded_pilot":
        item = block("bounded_pilot")
        return {
            "status": item.get("status"),
            "numerics_finite": item.get("numerics_finite"),
            "resume_equivalent": item.get("resume_equivalent"),
            "loss_trajectory_acceptable": item.get("loss_trajectory_acceptable"),
        }
    if normalized_role in {"learned_3m", "learned_10m"}:
        scale = block("learned_scale_evidence")
        item = scale.get(normalized_role)
        item = item if isinstance(item, dict) else {}
        return {"status": item.get("status")}
    if normalized_role == "cost_envelope":
        item = block("cost_envelope")
        return {
            "status": item.get("status"),
            "maximum_cost_usd": item.get("maximum_cost_usd"),
        }
    if normalized_role == "independent_audit":
        item = block("independent_audit")
        return {"status": item.get("status")}
    return None


def scientific_authority_token(
    role: str,
    authority: Any,
    *,
    metadata: Any = _METADATA_UNSET,
    require_workflow: bool = False,
) -> str | None:
    """Return a role-bound digest a trusted live resolver may verify out of packet.

    ``metadata`` is optional only for compatibility with non-readiness consumers of
    this helper. The learned-20M readiness evaluator always supplies the exact
    role projection, so legacy authority-only tokens cannot satisfy readiness.
    """
    if not isinstance(role, str) or not role.strip():
        return None
    if not _valid_authority_ref(
        authority,
        require_workflow=require_workflow,
        strict=metadata is not _METADATA_UNSET,
    ):
        return None

    payload = {
        "role": role.strip(),
        "repository": authority["repository"],
        "git_sha": authority["git_sha"],
        "evidence_sha256": authority["evidence_sha256"],
        "terminal": True,
        "workflow_run_id": authority.get("workflow_run_id") if require_workflow else None,
        "workflow_conclusion": (
            authority.get("workflow_conclusion") if require_workflow else None
        ),
    }
    if metadata is not _METADATA_UNSET:
        metadata_sha256 = scientific_metadata_sha256(metadata)
        if metadata_sha256 is None:
            return None
        payload["metadata_sha256"] = metadata_sha256
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trusted_readiness_inputs(
    bindings: Any,
) -> tuple[set[str], set[str]] | None:
    """Resolve a separate trusted-binding bundle into evaluator inputs.

    The bundle is intentionally independent from the candidate readiness packet.
    It contains trusted copies of role authority + role metadata; the evaluator
    separately derives candidate-side tokens and therefore detects any drift.
    """
    if not isinstance(bindings, dict):
        return None
    if set(bindings) != {
        "schema_version",
        "scientific_authorities",
        "verified_authorization_refs",
    }:
        return None
    if bindings.get("schema_version") != 1:
        return None

    scientific = bindings.get("scientific_authorities")
    refs = bindings.get("verified_authorization_refs")
    if not isinstance(scientific, dict) or not isinstance(refs, list):
        return None

    verified_tokens: set[str] = set()
    for role, record in scientific.items():
        if role not in _SCIENTIFIC_ROLE_WORKFLOW_REQUIRED:
            return None
        if not isinstance(record, dict) or set(record) != {"authority", "metadata"}:
            return None
        metadata = record.get("metadata")
        if not isinstance(metadata, dict) or set(metadata) != _SCIENTIFIC_METADATA_KEYS[role]:
            return None
        token = scientific_authority_token(
            role,
            record.get("authority"),
            metadata=metadata,
            require_workflow=_SCIENTIFIC_ROLE_WORKFLOW_REQUIRED[role],
        )
        if token is None:
            return None
        verified_tokens.add(token)

    verified_refs: set[str] = set()
    for ref in refs:
        if not isinstance(ref, str) or not ref.strip():
            return None
        normalized = ref.strip()
        if normalized in verified_refs:
            return None
        verified_refs.add(normalized)

    return verified_tokens, verified_refs


def _require_identity(blockers: list[str], value: Any, name: str) -> None:
    if not _is_sha256(value):
        blockers.append(name)


def _require_authority(
    blockers: list[str], value: Any, name: str, *, require_workflow: bool = False
) -> None:
    if not _valid_authority_ref(value, require_workflow=require_workflow):
        blockers.append(name)


def _require_scientific_authority(
    blockers: list[str],
    value: Any,
    name: str,
    role: str,
    verified_authorities: set[str],
    metadata: Any,
    *,
    require_workflow: bool = False,
) -> None:
    if not _valid_authority_ref(
        value,
        require_workflow=require_workflow,
        strict=True,
    ):
        blockers.append(name)
        return
    token = scientific_authority_token(
        role,
        value,
        metadata=metadata,
        require_workflow=require_workflow,
    )
    if token is None or token not in verified_authorities:
        base = name.removesuffix("_missing")
        blockers.append(f"{base}_unverified")


def _verified_ref(
    blockers: list[str],
    value: Any,
    name: str,
    verified_refs: set[str],
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        blockers.append(f"{name}_missing")
        return None
    normalized = value.strip()
    if normalized not in verified_refs:
        blockers.append(f"{name}_unverified")
    return normalized


def _validate_envelope(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("schema_version_mismatch")
    if data.get("campaign_id") != CAMPAIGN_ID:
        errors.append("campaign_id_mismatch")
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
        for key in (
            "source_bytes_are_not_loss_positions",
            "training_exposure_is_not_unique_data",
            "replay_cannot_inflate_unique_loss_ledger",
        ):
            if boundaries.get(key) is not True:
                errors.append(f"truth_boundary_{key}_must_be_true")
    return errors


def assess_learned20m_readiness(
    data: dict[str, Any],
    *,
    verified_scientific_authorities: Collection[str] = (),
    verified_authorization_refs: Collection[str] = (),
) -> ReadinessAssessment:
    """Assess launch readiness; packet contents alone cannot grant scientific trust."""
    verified_scientific = {
        value.strip()
        for value in verified_scientific_authorities
        if isinstance(value, str) and value.strip()
    }
    envelope_errors = _validate_envelope(data)
    evidence = data.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
        envelope_errors.append("evidence_missing")

    local = list(envelope_errors)

    code = evidence.get("code") if isinstance(evidence.get("code"), dict) else {}
    if not _is_git_sha(code.get("git_sha")):
        local.append("exact_code_sha_missing")
    _require_scientific_authority(
        local,
        code.get("authority"),
        "exact_code_authority_missing",
        "code",
        verified_scientific,
        scientific_role_metadata("code", evidence),
        require_workflow=True,
    )

    corpus = evidence.get("corpus") if isinstance(evidence.get("corpus"), dict) else {}
    _require_identity(local, corpus.get("manifest_sha256"), "corpus_manifest_missing")
    _require_identity(local, corpus.get("split_sha256"), "corpus_split_identity_missing")
    _require_identity(local, corpus.get("packing_sha256"), "packing_identity_missing")
    if corpus.get("two_clean_builds_identical") is not True:
        local.append("two_clean_builds_not_proven")
    _require_scientific_authority(
        local,
        corpus.get("authority"),
        "terminal_corpus_authority_missing",
        "corpus",
        verified_scientific,
        scientific_role_metadata("corpus", evidence),
        require_workflow=True,
    )

    tokenizer = (
        evidence.get("tokenizer") if isinstance(evidence.get("tokenizer"), dict) else {}
    )
    _require_identity(local, tokenizer.get("identity_sha256"), "tokenizer_identity_missing")
    if tokenizer.get("decision") not in {"TRAINED_TOKENIZER", "BYTE_BASELINE_RETAINED"}:
        local.append("tokenizer_decision_not_terminal")
    _require_scientific_authority(
        local,
        tokenizer.get("authority"),
        "terminal_tokenizer_authority_missing",
        "tokenizer",
        verified_scientific,
        scientific_role_metadata("tokenizer", evidence),
        require_workflow=True,
    )

    ledger = evidence.get("loss_ledger") if isinstance(evidence.get("loss_ledger"), dict) else {}
    _require_identity(local, ledger.get("identity_sha256"), "unique_loss_ledger_missing")
    positions = ledger.get("unique_causal_loss_positions")
    if not _is_positive_int(positions):
        local.append("unique_loss_positions_not_positive")
    _require_scientific_authority(
        local,
        ledger.get("authority"),
        "terminal_unique_loss_ledger_authority_missing",
        "loss_ledger",
        verified_scientific,
        scientific_role_metadata("loss_ledger", evidence),
        require_workflow=True,
    )
    _require_scientific_authority(
        local,
        ledger.get("data_budget_authority"),
        "data_budget_authority_missing",
        "data_budget",
        verified_scientific,
        scientific_role_metadata("data_budget", evidence),
        require_workflow=True,
    )
    if ledger.get("data_budget_status") != "QUALIFIED":
        local.append("data_budget_not_qualified")

    checkpoint = (
        evidence.get("checkpoint_integrity")
        if isinstance(evidence.get("checkpoint_integrity"), dict)
        else {}
    )
    _require_scientific_authority(
        local,
        checkpoint.get("authority"),
        "checkpoint_integrity_authority_missing",
        "checkpoint_integrity",
        verified_scientific,
        scientific_role_metadata("checkpoint_integrity", evidence),
        require_workflow=True,
    )
    if checkpoint.get("status") != "PASS":
        local.append("checkpoint_integrity_not_terminal_pass")

    evaluation = (
        evidence.get("evaluation") if isinstance(evidence.get("evaluation"), dict) else {}
    )
    _require_scientific_authority(
        local,
        evaluation.get("firewall_authority"),
        "evaluation_firewall_authority_missing",
        "evaluation_firewall",
        verified_scientific,
        scientific_role_metadata("evaluation_firewall", evidence),
        require_workflow=True,
    )
    _require_scientific_authority(
        local,
        evaluation.get("selection_validation_authority"),
        "selection_validation_authority_missing",
        "selection_validation",
        verified_scientific,
        scientific_role_metadata("selection_validation", evidence),
        require_workflow=True,
    )
    if evaluation.get("status") != "PASS":
        local.append("evaluation_boundary_not_terminal_pass")

    recipe = (
        evidence.get("training_recipe")
        if isinstance(evidence.get("training_recipe"), dict)
        else {}
    )
    _require_scientific_authority(
        local,
        recipe.get("authority"),
        "training_recipe_authority_missing",
        "training_recipe",
        verified_scientific,
        scientific_role_metadata("training_recipe", evidence),
        require_workflow=True,
    )
    if recipe.get("status") != "QUALIFIED":
        local.append("training_recipe_not_qualified")
    if not _is_positive_int(recipe.get("seed_count")):
        local.append("training_seed_plan_missing")
    _require_identity(local, recipe.get("config_sha256"), "training_config_identity_missing")
    _require_identity(local, recipe.get("stopping_policy_sha256"), "stopping_policy_missing")

    requested_positions = recipe.get("requested_unique_loss_positions")
    if not _is_positive_int(requested_positions):
        local.append("requested_unique_loss_positions_not_positive")
    elif _is_positive_int(positions) and requested_positions > positions:
        local.append("requested_unique_loss_positions_exceed_ledger")

    total_exposures = recipe.get("requested_total_training_exposures")
    if not _is_positive_int(total_exposures):
        local.append("requested_total_training_exposures_not_positive")

    max_replay = recipe.get("max_exposures_per_unique_position")
    if not _is_positive_int(max_replay):
        local.append("max_exposures_per_unique_position_not_positive")

    if _is_positive_int(requested_positions) and _is_positive_int(total_exposures):
        if total_exposures < requested_positions:
            local.append("total_training_exposures_below_unique_requirement")
        if _is_positive_int(max_replay):
            maximum_exposures = requested_positions * max_replay
            if total_exposures > maximum_exposures:
                local.append("total_training_exposures_exceed_replay_cap")

    local = sorted(set(local))

    compute = list(local)
    pilot = evidence.get("bounded_pilot") if isinstance(evidence.get("bounded_pilot"), dict) else {}
    _require_scientific_authority(
        compute,
        pilot.get("authority"),
        "bounded_pilot_authority_missing",
        "bounded_pilot",
        verified_scientific,
        scientific_role_metadata("bounded_pilot", evidence),
        require_workflow=True,
    )
    if pilot.get("status") != "PASS":
        compute.append("bounded_pilot_not_terminal_pass")
    for key in ("numerics_finite", "resume_equivalent", "loss_trajectory_acceptable"):
        if pilot.get(key) is not True:
            compute.append(f"bounded_pilot_{key}_not_proven")

    scale = (
        evidence.get("learned_scale_evidence")
        if isinstance(evidence.get("learned_scale_evidence"), dict)
        else {}
    )
    for label in ("learned_3m", "learned_10m"):
        item = scale.get(label) if isinstance(scale.get(label), dict) else {}
        _require_scientific_authority(
            compute,
            item.get("authority"),
            f"{label}_authority_missing",
            label,
            verified_scientific,
            scientific_role_metadata(label, evidence),
            require_workflow=True,
        )
        if item.get("status") != "PASS":
            compute.append(f"{label}_not_terminal_pass")

    cost = evidence.get("cost_envelope") if isinstance(evidence.get("cost_envelope"), dict) else {}
    _require_scientific_authority(
        compute,
        cost.get("authority"),
        "cost_envelope_authority_missing",
        "cost_envelope",
        verified_scientific,
        scientific_role_metadata("cost_envelope", evidence),
    )
    if cost.get("status") != "ESTIMATED":
        compute.append("cost_envelope_not_estimated")
    maximum_cost = cost.get("maximum_cost_usd")
    if isinstance(maximum_cost, bool) or not isinstance(maximum_cost, (int, float)):
        compute.append("maximum_cost_missing")
    elif not _is_finite_number(maximum_cost):
        compute.append("maximum_cost_not_finite")
    elif maximum_cost <= 0:
        compute.append("maximum_cost_not_positive")

    audit = (
        evidence.get("independent_audit")
        if isinstance(evidence.get("independent_audit"), dict)
        else {}
    )
    _require_scientific_authority(
        compute,
        audit.get("authority"),
        "independent_audit_authority_missing",
        "independent_audit",
        verified_scientific,
        scientific_role_metadata("independent_audit", evidence),
        require_workflow=True,
    )
    if audit.get("status") not in {"PASS", "PASS_WITH_NOTES"}:
        compute.append("independent_audit_not_terminal_pass")

    compute = sorted(set(compute))

    material = list(compute)
    verified_refs = {
        value.strip()
        for value in verified_authorization_refs
        if isinstance(value, str) and value.strip()
    }

    compute_auth = (
        evidence.get("compute_authorization")
        if isinstance(evidence.get("compute_authorization"), dict)
        else {}
    )
    _require_authority(
        material,
        compute_auth.get("authority"),
        "compute_authorization_authority_missing",
    )
    if compute_auth.get("status") != "COMPUTE_AUTHORIZED":
        material.append("compute_not_explicitly_authorized")
    compute_decision_ref = _verified_ref(
        material,
        compute_auth.get("decision_ref"),
        "compute_authorization_ref",
        verified_refs,
    )
    authorized_limit = compute_auth.get("maximum_cost_usd")
    if isinstance(authorized_limit, bool) or not isinstance(authorized_limit, (int, float)):
        material.append("authorized_cost_limit_missing")
    elif not _is_finite_number(authorized_limit):
        material.append("authorized_cost_limit_not_finite")
    elif authorized_limit <= 0:
        material.append("authorized_cost_limit_not_positive")
    elif _is_finite_positive_number(maximum_cost) and authorized_limit < maximum_cost:
        material.append("authorized_cost_below_estimated_maximum")

    training_auth = (
        evidence.get("training_authorization")
        if isinstance(evidence.get("training_authorization"), dict)
        else {}
    )
    _require_authority(
        material,
        training_auth.get("authority"),
        "training_authorization_authority_missing",
    )
    if training_auth.get("status") != "TRAINING_AUTHORIZED":
        material.append("training_not_explicitly_authorized")
    training_decision_ref = _verified_ref(
        material,
        training_auth.get("decision_ref"),
        "training_authorization_ref",
        verified_refs,
    )
    if (
        compute_decision_ref is not None
        and training_decision_ref is not None
        and compute_decision_ref == training_decision_ref
    ):
        material.append("compute_and_training_authorization_refs_must_be_distinct")

    material = sorted(set(material))
    return ReadinessAssessment(
        ready_for_local_free_pilot=not local,
        ready_for_compute_authorization_request=not compute,
        material_training_authorized=not material,
        local_free_pilot_blockers=tuple(local),
        compute_request_blockers=tuple(compute),
        material_training_blockers=tuple(material),
    )
