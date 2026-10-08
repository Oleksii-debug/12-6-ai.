"""Fail-closed routing for the accelerated 20M -> 200M -> 1B scale path.

This module decides which evidence package may be prepared next.  It never grants
training or compute authorization and it deliberately keeps the learned-20M
terminal proof ahead of every larger-model activity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder

REPOSITORY = "Oleksii-debug/12-6-ai."
ROADMAP_ID = "R01-ACCELERATED-SCALING-ROADMAP-V2"
ROADMAP_SCHEMA_VERSION = 2
EXPECTED_STRATEGY_GIT_SHA = "7cec1a1693676fc8e737400890727e94e8b46bee"
EXPECTED_STRATEGY_SHA256 = (
    "5cfaf394ff850356343679061a3bf0c2288a7975ed8c9132c5008b422bff2941"
)
EXPECTED_PREVIOUS_ROADMAP_SHA256 = (
    "1a7996618e3675b66c0f65484105d93e2f952b24d48a76e8808f2925192f97e8"
)

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_ROUTE_IDS = (
    "LAB_3M_10M",
    "TERMINAL_LEARNED_20M",
    "OPTIONAL_50M_100M_PROBES",
    "PRODUCT_200M",
    "PRODUCT_1B",
)

TERMINAL_AUDIT_INDEPENDENCE_FIELDS = {
    "producer_and_audit_workflow_run_ids_must_differ",
    "producer_and_audit_evidence_sha256_must_differ",
    "same_code_git_sha_permitted",
}

TERMINAL_EVIDENCE_CROSSBINDING_FIELDS = {
    "producer_must_attest_exact_evidence_manifest_sha256",
    "audit_must_attest_exact_evidence_manifest_sha256",
    "audit_must_bind_exact_producer_authority_sha256",
}

REQUIRED_TERMINAL_20M_EVIDENCE = {
    "exact_code_model_init_identity",
    "terminal_corpus_split_packing_identity",
    "unique_post_pack_causal_loss_ledger",
    "terminal_tokenizer_identity_or_byte_baseline_decision",
    "checkpoint_corruption_and_fresh_process_resume",
    "evaluation_firewall_and_selection_validation",
    "bounded_pilot_numerics_throughput_and_loss",
    "exact_optimized_targets_without_hidden_replay",
    "best_and_chronological_final_checkpoint_identity",
    "heldout_ua_en_code_metrics",
    "first_party_inference_reload_fingerprint",
    "memorization_and_exposure_diagnostics",
    "independent_audit",
}

REQUIRED_TERMINAL_200M_EVIDENCE = {
    "exact_code_model_init_identity",
    "terminal_corpus_split_packing_identity",
    "unique_post_pack_causal_loss_ledger",
    "terminal_tokenizer_identity",
    "exact_optimized_targets_without_hidden_replay",
    "checkpoint_lineage_and_cross_provider_recovery",
    "evaluation_firewall_and_selection_validation",
    "heldout_ua_en_code_metrics",
    "measured_memory_throughput_and_cost",
    "first_party_inference_reload_fingerprint",
    "product_utility_evaluation",
    "memorization_and_exposure_diagnostics",
    "independent_audit",
}

REQUIRED_RUN_PACKET_FIELDS = {
    "source_git_sha",
    "modelspec_sha256",
    "initspec_sha256",
    "tokenizer_sha256",
    "corpus_manifest_sha256",
    "split_sha256",
    "packing_sha256",
    "unique_loss_ledger_sha256",
    "training_config_sha256",
    "optimizer_scheduler_precision",
    "seed",
    "target_unique_loss_positions",
    "maximum_total_exposures",
    "checkpoint_lineage",
    "stop_resume_policy_sha256",
    "evaluation_schedule_sha256",
    "resource_class",
    "maximum_cost",
}

REQUIRED_BACKEND_QUALIFICATION = {
    "exact_version",
    "license",
    "modelspec_parity",
    "optimizer_semantics_parity",
    "checkpoint_save_load_resume",
    "fresh_process_recovery",
    "determinism_or_seed_variance_evidence",
    "bounded_throughput_and_memory_benchmark",
    "rollback_path",
}

REQUIRED_MEASUREMENTS = {
    "tokens_per_second",
    "step_time_seconds",
    "peak_ram_bytes",
    "peak_vram_bytes_or_not_applicable",
    "checkpoint_size_bytes",
    "checkpoint_save_seconds",
    "checkpoint_load_seconds",
    "estimated_wall_clock_to_target",
    "energy_or_thermal_observation_if_available",
}

REQUIRED_200M_FEASIBILITY = {
    "candidate_architecture_and_parameter_count",
    "weights_gradients_optimizer_activations_memory",
    "gradient_accumulation_and_checkpointing_plan",
    "qualified_backend_evidence",
    "unique_data_supply_and_target_exposure",
    "measured_20m_throughput_extrapolation",
    "local_free_and_free_gpu_execution_plan",
    "checkpoint_transport_and_resume_plan",
    "evaluation_capacity",
    "paid_compute_threshold_without_authorization",
}

REQUIRED_1B_FEASIBILITY = {
    "terminal_learned_200m_evidence",
    "expanded_unique_corpus_budget",
    "measured_memory_throughput_and_cost_model",
    "distributed_or_offload_strategy",
    "cross_provider_checkpoint_transport_and_recovery",
    "scale_appropriate_evaluation_capacity",
    "architecture_re_evaluation",
    "explicit_compute_authorization_if_materially_paid",
}


@dataclass(frozen=True)
class ScalingRoadmapAssessment:
    """Deterministic routing result; no field is an authorization grant."""

    contract_valid: bool
    next_action: str
    terminal_20m_proven: bool
    ready_for_200m_feasibility: bool
    ready_to_request_200m_authorization: bool
    terminal_200m_proven: bool
    ready_for_1b_feasibility: bool
    ready_to_request_1b_authorization: bool
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_valid": self.contract_valid,
            "next_action": self.next_action,
            "terminal_20m_proven": self.terminal_20m_proven,
            "ready_for_200m_feasibility": self.ready_for_200m_feasibility,
            "ready_to_request_200m_authorization": (
                self.ready_to_request_200m_authorization
            ),
            "terminal_200m_proven": self.terminal_200m_proven,
            "ready_for_1b_feasibility": self.ready_for_1b_feasibility,
            "ready_to_request_1b_authorization": (
                self.ready_to_request_1b_authorization
            ),
            "blockers": list(self.blockers),
        }


def _is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and _GIT_SHA_RE.fullmatch(value) is not None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _canonical_sha256(value: Any) -> str | None:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _valid_terminal_authority(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        value.get("repository") == REPOSITORY
        and _is_git_sha(value.get("git_sha"))
        and _is_sha256(value.get("evidence_sha256"))
        and _is_positive_int(value.get("workflow_run_id"))
        and value.get("workflow_conclusion") == "success"
        and value.get("terminal") is True
    )


def _authority_attests_manifest(authority: Any, manifest_sha256: Any) -> bool:
    return (
        _valid_terminal_authority(authority)
        and _is_sha256(manifest_sha256)
        and authority.get("attested_evidence_manifest_sha256") == manifest_sha256
    )


def _audit_binds_producer(audit: Any, producer: Any) -> bool:
    if not (_valid_terminal_authority(audit) and _valid_terminal_authority(producer)):
        return False
    producer_sha256 = _canonical_sha256(producer)
    return (
        producer_sha256 is not None
        and _is_sha256(audit.get("audited_producer_authority_sha256"))
        and audit.get("audited_producer_authority_sha256") == producer_sha256
    )


def _terminal_authorities_are_independent(producer: Any, audit: Any) -> bool:
    if not (_valid_terminal_authority(producer) and _valid_terminal_authority(audit)):
        return False
    return (
        producer["workflow_run_id"] != audit["workflow_run_id"]
        and producer["evidence_sha256"] != audit["evidence_sha256"]
    )


def _route_by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    route = data.get("scale_route")
    if not isinstance(route, list):
        return {}
    return {
        item["id"]: item
        for item in route
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def _validate_authority(errors: list[str], data: dict[str, Any]) -> None:
    authority = data.get("authority")
    _expect(errors, isinstance(authority, dict), "authority_missing")
    if not isinstance(authority, dict):
        return

    strategy = authority.get("source_strategy")
    _expect(errors, isinstance(strategy, dict), "source_strategy_authority_missing")
    if isinstance(strategy, dict):
        _expect(
            errors,
            strategy.get("path")
            == "docs/TRAINING_COMPUTE_AND_SCALING_STRATEGY_2026-09-07.md",
            "source_strategy_path_mismatch",
        )
        _expect(
            errors,
            strategy.get("git_sha") == EXPECTED_STRATEGY_GIT_SHA,
            "source_strategy_git_sha_mismatch",
        )
        _expect(
            errors,
            strategy.get("sha256") == EXPECTED_STRATEGY_SHA256,
            "source_strategy_sha256_mismatch",
        )

    previous = authority.get("previous_roadmap")
    _expect(errors, isinstance(previous, dict), "previous_roadmap_authority_missing")
    if isinstance(previous, dict):
        _expect(
            errors,
            previous.get("path")
            == "configs/research/r01_20m_to_100m_scaling_campaign_v1.json",
            "previous_roadmap_path_mismatch",
        )
        _expect(
            errors,
            previous.get("sha256") == EXPECTED_PREVIOUS_ROADMAP_SHA256,
            "previous_roadmap_sha256_mismatch",
        )
        _expect(
            errors,
            previous.get("disposition") == "SUPERSEDED_FOR_POST_20M_ROUTING_ONLY",
            "previous_roadmap_disposition_mismatch",
        )
        _expect(
            errors,
            previous.get("scientific_gates_preserved") is True,
            "previous_roadmap_scientific_gates_not_preserved",
        )


def _validate_boundaries(errors: list[str], data: dict[str, Any]) -> None:
    boundaries = data.get("hard_boundaries")
    _expect(errors, isinstance(boundaries, dict), "hard_boundaries_missing")
    if not isinstance(boundaries, dict):
        return

    _expect(
        errors,
        boundaries.get("canonical_base") == "RANDOM_INIT_PRETRAINING_ONLY",
        "canonical_base_boundary_drift",
    )
    for key in (
        "terminal_learned_20m_required",
        "unique_data_must_scale_with_model",
        "project_owned_scientific_contracts_authoritative",
    ):
        _expect(errors, boundaries.get(key) is True, f"hard_boundaries_{key}_must_be_true")
    for key in (
        "foreign_pretrained_or_aligned_weights_allowed",
        "hidden_teacher_logits_allowed",
        "evaluation_or_final_test_training_use_allowed",
        "replay_or_padding_may_count_as_unique_exposure",
        "paid_compute_authorized",
        "material_training_authorized",
        "stage_promotion_authorized",
        "backend_promotion_authorized",
        "training_executed_by_this_package",
    ):
        _expect(errors, boundaries.get(key) is False, f"hard_boundaries_{key}_must_be_false")


def _validate_terminal_audit_independence(
    errors: list[str], data: dict[str, Any]
) -> None:
    contract = data.get("terminal_audit_independence")
    _expect(
        errors,
        isinstance(contract, dict),
        "terminal_audit_independence_missing",
    )
    if not isinstance(contract, dict):
        return

    _expect(
        errors,
        set(contract) == TERMINAL_AUDIT_INDEPENDENCE_FIELDS,
        "terminal_audit_independence_keys_mismatch",
    )
    for field in TERMINAL_AUDIT_INDEPENDENCE_FIELDS:
        _expect(
            errors,
            contract.get(field) is True,
            f"terminal_audit_independence_{field}_must_be_true",
        )


def _validate_terminal_evidence_crossbinding(
    errors: list[str], data: dict[str, Any]
) -> None:
    contract = data.get("terminal_evidence_crossbinding")
    _expect(
        errors,
        isinstance(contract, dict),
        "terminal_evidence_crossbinding_missing",
    )
    if not isinstance(contract, dict):
        return

    _expect(
        errors,
        set(contract) == TERMINAL_EVIDENCE_CROSSBINDING_FIELDS,
        "terminal_evidence_crossbinding_keys_mismatch",
    )
    for field in TERMINAL_EVIDENCE_CROSSBINDING_FIELDS:
        _expect(
            errors,
            contract.get(field) is True,
            f"terminal_evidence_crossbinding_{field}_must_be_true",
        )


def _validate_route(errors: list[str], data: dict[str, Any]) -> None:
    route = data.get("scale_route")
    _expect(errors, isinstance(route, list), "scale_route_missing")
    if not isinstance(route, list):
        return

    ids = [item.get("id") for item in route if isinstance(item, dict)]
    _expect(errors, len(ids) == len(route), "scale_route_entry_invalid")
    _expect(
        errors,
        tuple(ids) == REQUIRED_ROUTE_IDS,
        "scale_route_order_or_ids_mismatch",
    )
    _expect(errors, len(ids) == len(set(ids)), "scale_route_ids_not_unique")

    stages = _route_by_id(data)
    learned20 = stages.get("TERMINAL_LEARNED_20M", {})
    _expect(
        errors,
        learned20.get("target_parameters") == 20_613_440,
        "learned_20m_target_drift",
    )
    _expect(
        errors,
        learned20.get("role") == "TERMINAL_PIPELINE_QUALIFICATION",
        "learned_20m_role_drift",
    )
    _expect(
        errors,
        learned20.get("mandatory") is True,
        "learned_20m_must_remain_mandatory",
    )
    _expect(
        errors,
        learned20.get("authorized_now") is False,
        "learned_20m_not_authorized_now",
    )

    probes = stages.get("OPTIONAL_50M_100M_PROBES", {})
    _expect(
        errors,
        probes.get("parameter_range") == [50_000_000, 100_000_000],
        "optional_probe_range_drift",
    )
    _expect(
        errors,
        probes.get("mandatory") is False,
        "50m_100m_probes_must_remain_optional",
    )
    _expect(
        errors,
        probes.get("full_campaign_required") is False,
        "50m_100m_full_campaign_must_not_be_required",
    )
    _expect(
        errors,
        probes.get("may_block_200m_by_default") is False,
        "optional_probes_cannot_block_200m_by_default",
    )
    _expect(
        errors,
        probes.get("authorized_now") is False,
        "optional_probes_not_authorized_now",
    )

    product200 = stages.get("PRODUCT_200M", {})
    _expect(
        errors,
        product200.get("approximate_target_parameters") == 200_000_000,
        "product_200m_target_drift",
    )
    _expect(
        errors,
        product200.get("role") == "FIRST_SERIOUS_PRODUCT_BRAIN",
        "product_200m_role_drift",
    )
    _expect(
        errors,
        product200.get("requires_terminal_stage") == "TERMINAL_LEARNED_20M",
        "product_200m_must_require_terminal_20m",
    )
    _expect(
        errors,
        product200.get("modelspec_frozen") is False,
        "product_200m_modelspec_cannot_be_frozen",
    )
    _expect(
        errors,
        product200.get("authorized_now") is False,
        "product_200m_not_authorized_now",
    )

    product1b = stages.get("PRODUCT_1B", {})
    _expect(
        errors,
        product1b.get("approximate_target_parameters") == 1_000_000_000,
        "product_1b_target_drift",
    )
    _expect(
        errors,
        product1b.get("requires_terminal_stage") == "PRODUCT_200M",
        "product_1b_must_require_terminal_200m",
    )
    _expect(
        errors,
        product1b.get("modelspec_frozen") is False,
        "product_1b_modelspec_cannot_be_frozen",
    )
    _expect(
        errors,
        product1b.get("authorized_now") is False,
        "product_1b_not_authorized_now",
    )


def _validate_portability(errors: list[str], data: dict[str, Any]) -> None:
    portable = data.get("portable_training_runner")
    _expect(errors, isinstance(portable, dict), "portable_training_runner_missing")
    if not isinstance(portable, dict):
        return

    for key in (
        "provider_neutral",
        "project_manifest_authoritative",
        "cross_provider_resume_required",
        "early_checkpoint_required_for_ephemeral_sessions",
    ):
        _expect(errors, portable.get(key) is True, f"portable_runner_{key}_must_be_true")

    priorities = portable.get("resource_priority")
    _expect(
        errors,
        priorities
        == [
            "OWNER_LAPTOP_LOCAL_FREE",
            "FREE_GPU_KAGGLE",
            "FREE_GPU_OTHER",
            "PAID_ONLY_AFTER_EXPLICIT_AUTHORIZATION",
        ],
        "resource_priority_mismatch",
    )

    fields = portable.get("required_run_packet_fields")
    _expect(errors, isinstance(fields, list), "required_run_packet_fields_missing")
    if isinstance(fields, list):
        _expect(
            errors,
            REQUIRED_RUN_PACKET_FIELDS.issubset(set(fields)),
            "required_run_packet_fields_incomplete",
        )

    qualification = portable.get("backend_qualification_requirements")
    _expect(
        errors,
        isinstance(qualification, list),
        "backend_qualification_requirements_missing",
    )
    if isinstance(qualification, list):
        _expect(
            errors,
            REQUIRED_BACKEND_QUALIFICATION.issubset(set(qualification)),
            "backend_qualification_requirements_incomplete",
        )

    _expect(
        errors,
        portable.get("selected_backend") is None,
        "backend_cannot_be_selected_without_evidence",
    )
    backends = portable.get("backend_candidates")
    _expect(
        errors,
        isinstance(backends, list) and bool(backends),
        "backend_candidates_missing",
    )
    if isinstance(backends, list):
        names = []
        for candidate in backends:
            if not isinstance(candidate, dict):
                errors.append("backend_candidate_invalid")
                continue
            names.append(candidate.get("id"))
            _expect(
                errors,
                candidate.get("canonical_now") is False,
                f"backend_{candidate.get('id')}_cannot_be_canonical_now",
            )
            _expect(
                errors,
                candidate.get("qualification_status") == "UNQUALIFIED",
                f"backend_{candidate.get('id')}_must_start_unqualified",
            )
        _expect(errors, len(names) == len(set(names)), "backend_candidate_ids_not_unique")


def _validate_requirements(errors: list[str], data: dict[str, Any]) -> None:
    mappings = (
        ("terminal_20m_evidence_requirements", REQUIRED_TERMINAL_20M_EVIDENCE),
        ("terminal_200m_evidence_requirements", REQUIRED_TERMINAL_200M_EVIDENCE),
        ("measurement_contract", REQUIRED_MEASUREMENTS),
        ("feasibility_200m_requirements", REQUIRED_200M_FEASIBILITY),
        ("feasibility_1b_requirements", REQUIRED_1B_FEASIBILITY),
    )
    for key, required in mappings:
        values = data.get(key)
        _expect(errors, isinstance(values, list), f"{key}_missing")
        if isinstance(values, list):
            _expect(errors, required.issubset(set(values)), f"{key}_incomplete")


def _validate_evidence_state(errors: list[str], data: dict[str, Any]) -> None:
    state = data.get("evidence_state")
    _expect(errors, isinstance(state, dict), "evidence_state_missing")
    if not isinstance(state, dict):
        return

    learned20 = state.get("learned_20m")
    learned200 = state.get("learned_200m")
    feasibility200 = state.get("feasibility_200m")
    feasibility1b = state.get("feasibility_1b")
    for key, value in (
        ("learned_20m", learned20),
        ("feasibility_200m", feasibility200),
        ("learned_200m", learned200),
        ("feasibility_1b", feasibility1b),
    ):
        _expect(errors, isinstance(value, dict), f"evidence_state_{key}_missing")

    if not all(
        isinstance(value, dict)
        for value in (learned20, learned200, feasibility200, feasibility1b)
    ):
        return

    learned_requirements = {
        "learned_20m": REQUIRED_TERMINAL_20M_EVIDENCE,
        "learned_200m": REQUIRED_TERMINAL_200M_EVIDENCE,
    }
    for name, learned in (("learned_20m", learned20), ("learned_200m", learned200)):
        _expect(
            errors,
            learned.get("status") in {"NOT_TERMINAL", "PASS"},
            f"{name}_status_invalid",
        )
        if learned.get("status") == "PASS":
            manifest_sha256 = learned.get("evidence_manifest_sha256")
            manifest_valid = _is_sha256(manifest_sha256)
            _expect(
                errors,
                manifest_valid,
                f"{name}_evidence_manifest_sha256_invalid",
            )
            satisfied = learned.get("requirements_satisfied")
            _expect(
                errors,
                isinstance(satisfied, list)
                and learned_requirements[name].issubset(set(satisfied)),
                f"{name}_requirements_incomplete",
            )
            terminal_authority = learned.get("terminal_authority")
            audit_authority = learned.get("independent_audit_authority")
            terminal_valid = _valid_terminal_authority(terminal_authority)
            audit_valid = _valid_terminal_authority(audit_authority)
            _expect(
                errors,
                terminal_valid,
                f"{name}_terminal_authority_invalid",
            )
            _expect(
                errors,
                audit_valid,
                f"{name}_independent_audit_authority_invalid",
            )
            if terminal_valid and manifest_valid:
                _expect(
                    errors,
                    _authority_attests_manifest(
                        terminal_authority, manifest_sha256
                    ),
                    f"{name}_terminal_authority_manifest_attestation_invalid",
                )
            if audit_valid and manifest_valid:
                _expect(
                    errors,
                    _authority_attests_manifest(audit_authority, manifest_sha256),
                    f"{name}_independent_audit_manifest_attestation_invalid",
                )
            if terminal_valid and audit_valid:
                _expect(
                    errors,
                    _terminal_authorities_are_independent(
                        terminal_authority, audit_authority
                    ),
                    f"{name}_independent_audit_not_distinct",
                )
                _expect(
                    errors,
                    _audit_binds_producer(audit_authority, terminal_authority),
                    f"{name}_independent_audit_producer_binding_invalid",
                )

    feasibility_requirements = {
        "feasibility_200m": REQUIRED_200M_FEASIBILITY,
        "feasibility_1b": REQUIRED_1B_FEASIBILITY,
    }
    for name, feasibility in (
        ("feasibility_200m", feasibility200),
        ("feasibility_1b", feasibility1b),
    ):
        _expect(
            errors,
            feasibility.get("status") in {"NOT_PREPARED", "PASS"},
            f"{name}_status_invalid",
        )
        _expect(
            errors,
            feasibility.get("decision")
            in {"NOT_EVALUATED", "GO", "HOLD", "NO_GO"},
            f"{name}_decision_invalid",
        )
        if feasibility.get("status") == "PASS":
            _expect(
                errors,
                _is_sha256(feasibility.get("packet_sha256")),
                f"{name}_packet_sha256_invalid",
            )
            satisfied = feasibility.get("requirements_satisfied")
            _expect(
                errors,
                isinstance(satisfied, list)
                and feasibility_requirements[name].issubset(set(satisfied)),
                f"{name}_requirements_incomplete",
            )
            _expect(
                errors,
                _valid_terminal_authority(feasibility.get("terminal_authority")),
                f"{name}_terminal_authority_invalid",
            )
            _expect(
                errors,
                feasibility.get("decision") != "NOT_EVALUATED",
                f"{name}_decision_missing",
            )

    terminal20 = learned20.get("status") == "PASS"
    terminal200 = learned200.get("status") == "PASS"
    feasibility200_pass = feasibility200.get("status") == "PASS"
    feasibility1b_pass = feasibility1b.get("status") == "PASS"
    _expect(
        errors,
        not feasibility200_pass or terminal20,
        "200m_feasibility_cannot_precede_terminal_20m",
    )
    _expect(
        errors,
        not terminal200
        or (feasibility200_pass and feasibility200.get("decision") == "GO"),
        "terminal_200m_cannot_precede_go_feasibility",
    )
    _expect(
        errors,
        not feasibility1b_pass or terminal200,
        "1b_feasibility_cannot_precede_terminal_200m",
    )


def validate_roadmap(data: dict[str, Any]) -> list[str]:
    """Return invariant violations; an empty list means structurally valid, not ready."""
    errors: list[str] = []
    _expect(
        errors,
        data.get("schema_version") == ROADMAP_SCHEMA_VERSION,
        "schema_version_mismatch",
    )
    _expect(errors, data.get("roadmap_id") == ROADMAP_ID, "roadmap_id_mismatch")
    _expect(
        errors,
        data.get("status") == "BLOCKED_PENDING_TERMINAL_LEARNED_20M",
        "snapshot_status_must_remain_blocked_pending_terminal_learned_20m",
    )
    _validate_authority(errors, data)
    _validate_boundaries(errors, data)
    _validate_terminal_audit_independence(errors, data)
    _validate_terminal_evidence_crossbinding(errors, data)
    _validate_route(errors, data)
    _validate_portability(errors, data)
    _validate_requirements(errors, data)
    _validate_evidence_state(errors, data)
    return sorted(set(errors))


def _terminal_evidence(
    blockers: list[str], value: Any, name: str, required: set[str]
) -> bool:
    if not isinstance(value, dict) or value.get("status") != "PASS":
        blockers.append(f"{name}_not_terminal_pass")
        return False
    valid = True
    manifest_sha256 = value.get("evidence_manifest_sha256")
    manifest_valid = _is_sha256(manifest_sha256)
    if not manifest_valid:
        blockers.append(f"{name}_evidence_manifest_sha256_invalid")
        valid = False
    satisfied = value.get("requirements_satisfied")
    if not isinstance(satisfied, list) or not required.issubset(set(satisfied)):
        blockers.append(f"{name}_requirements_incomplete")
        valid = False
    terminal_authority = value.get("terminal_authority")
    audit_authority = value.get("independent_audit_authority")
    terminal_valid = _valid_terminal_authority(terminal_authority)
    audit_valid = _valid_terminal_authority(audit_authority)
    if not terminal_valid:
        blockers.append(f"{name}_terminal_authority_invalid")
        valid = False
    if not audit_valid:
        blockers.append(f"{name}_independent_audit_authority_invalid")
        valid = False
    if (
        terminal_valid
        and manifest_valid
        and not _authority_attests_manifest(terminal_authority, manifest_sha256)
    ):
        blockers.append(f"{name}_terminal_authority_manifest_attestation_invalid")
        valid = False
    if (
        audit_valid
        and manifest_valid
        and not _authority_attests_manifest(audit_authority, manifest_sha256)
    ):
        blockers.append(f"{name}_independent_audit_manifest_attestation_invalid")
        valid = False
    if (
        terminal_valid
        and audit_valid
        and not _terminal_authorities_are_independent(
            terminal_authority, audit_authority
        )
    ):
        blockers.append(f"{name}_independent_audit_not_distinct")
        valid = False
    if (
        terminal_valid
        and audit_valid
        and not _audit_binds_producer(audit_authority, terminal_authority)
    ):
        blockers.append(f"{name}_independent_audit_producer_binding_invalid")
        valid = False
    return valid


def _feasibility_evidence(
    blockers: list[str], value: Any, name: str, required: set[str]
) -> tuple[bool, str]:
    if not isinstance(value, dict) or value.get("status") != "PASS":
        blockers.append(f"{name}_not_terminal_pass")
        return False, "NOT_EVALUATED"
    valid = True
    if not _is_sha256(value.get("packet_sha256")):
        blockers.append(f"{name}_packet_sha256_invalid")
        valid = False
    satisfied = value.get("requirements_satisfied")
    if not isinstance(satisfied, list) or not required.issubset(set(satisfied)):
        blockers.append(f"{name}_requirements_incomplete")
        valid = False
    if not _valid_terminal_authority(value.get("terminal_authority")):
        blockers.append(f"{name}_terminal_authority_invalid")
        valid = False
    decision = value.get("decision")
    if decision not in {"GO", "HOLD", "NO_GO"}:
        blockers.append(f"{name}_decision_invalid")
        valid = False
        decision = "NOT_EVALUATED"
    return valid, decision


def assess_roadmap(data: dict[str, Any]) -> ScalingRoadmapAssessment:
    """Choose the next scale action while preserving all scientific gates."""
    contract_errors = validate_roadmap(data)
    if contract_errors:
        return ScalingRoadmapAssessment(
            contract_valid=False,
            next_action="REPAIR_INVALID_ROADMAP_CONTRACT",
            terminal_20m_proven=False,
            ready_for_200m_feasibility=False,
            ready_to_request_200m_authorization=False,
            terminal_200m_proven=False,
            ready_for_1b_feasibility=False,
            ready_to_request_1b_authorization=False,
            blockers=tuple(contract_errors),
        )

    state = data["evidence_state"]
    blockers: list[str] = []
    terminal20 = _terminal_evidence(
        blockers,
        state["learned_20m"],
        "learned_20m",
        REQUIRED_TERMINAL_20M_EVIDENCE,
    )
    if not terminal20:
        return ScalingRoadmapAssessment(
            contract_valid=True,
            next_action="FINISH_TERMINAL_LEARNED_20M_CRITICAL_PATH",
            terminal_20m_proven=False,
            ready_for_200m_feasibility=False,
            ready_to_request_200m_authorization=False,
            terminal_200m_proven=False,
            ready_for_1b_feasibility=False,
            ready_to_request_1b_authorization=False,
            blockers=tuple(sorted(set(blockers))),
        )

    feasibility200_ok, decision200 = _feasibility_evidence(
        blockers,
        state["feasibility_200m"],
        "feasibility_200m",
        REQUIRED_200M_FEASIBILITY,
    )
    if not feasibility200_ok:
        return ScalingRoadmapAssessment(
            contract_valid=True,
            next_action="PREPARE_200M_FEASIBILITY_PACKET",
            terminal_20m_proven=True,
            ready_for_200m_feasibility=True,
            ready_to_request_200m_authorization=False,
            terminal_200m_proven=False,
            ready_for_1b_feasibility=False,
            ready_to_request_1b_authorization=False,
            blockers=tuple(sorted(set(blockers))),
        )
    if decision200 != "GO":
        blockers.append(f"feasibility_200m_decision_{decision200.lower()}")
        return ScalingRoadmapAssessment(
            contract_valid=True,
            next_action="HOLD_200M_AND_REASSESS_EVIDENCE",
            terminal_20m_proven=True,
            ready_for_200m_feasibility=True,
            ready_to_request_200m_authorization=False,
            terminal_200m_proven=False,
            ready_for_1b_feasibility=False,
            ready_to_request_1b_authorization=False,
            blockers=tuple(sorted(set(blockers))),
        )

    terminal200 = _terminal_evidence(
        blockers,
        state["learned_200m"],
        "learned_200m",
        REQUIRED_TERMINAL_200M_EVIDENCE,
    )
    if not terminal200:
        return ScalingRoadmapAssessment(
            contract_valid=True,
            next_action="REQUEST_EXPLICIT_200M_COMPUTE_AND_TRAINING_AUTHORIZATION",
            terminal_20m_proven=True,
            ready_for_200m_feasibility=True,
            ready_to_request_200m_authorization=True,
            terminal_200m_proven=False,
            ready_for_1b_feasibility=False,
            ready_to_request_1b_authorization=False,
            blockers=tuple(sorted(set(blockers))),
        )

    feasibility1b_ok, decision1b = _feasibility_evidence(
        blockers,
        state["feasibility_1b"],
        "feasibility_1b",
        REQUIRED_1B_FEASIBILITY,
    )
    if not feasibility1b_ok:
        return ScalingRoadmapAssessment(
            contract_valid=True,
            next_action="PREPARE_1B_FEASIBILITY_PACKET",
            terminal_20m_proven=True,
            ready_for_200m_feasibility=True,
            ready_to_request_200m_authorization=True,
            terminal_200m_proven=True,
            ready_for_1b_feasibility=True,
            ready_to_request_1b_authorization=False,
            blockers=tuple(sorted(set(blockers))),
        )
    if decision1b != "GO":
        blockers.append(f"feasibility_1b_decision_{decision1b.lower()}")
        next_action = "HOLD_1B_AND_REASSESS_EVIDENCE"
        ready1b = False
    else:
        next_action = "REQUEST_EXPLICIT_1B_COMPUTE_AND_TRAINING_AUTHORIZATION"
        ready1b = True

    return ScalingRoadmapAssessment(
        contract_valid=True,
        next_action=next_action,
        terminal_20m_proven=True,
        ready_for_200m_feasibility=True,
        ready_to_request_200m_authorization=True,
        terminal_200m_proven=True,
        ready_for_1b_feasibility=True,
        ready_to_request_1b_authorization=ready1b,
        blockers=tuple(sorted(set(blockers))),
    )


# Plan 7 / Section 1 -- identity-bound, bounded experimental scale proxies.
SCHEMA_VERSION = 1
ALLOWED_VARIANTS = frozenset(
    {"baseline", "mlp_width", "depth", "gqa", "heads", "context"}
)


class ScaleAdmissionError(ValueError):
    """Candidate or experiment exceeds a fail-closed LOCAL_FREE budget."""


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False,
            allow_nan=False, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ProxyProtocol:
    """Identity of one fixed-control, synthetic-data comparison protocol."""

    version: int
    fixture_sha256: str
    seed: int
    batch: int
    sequence: int
    repeats: int = 2
    dtype: str = "float32"
    resource_class: str = "LOCAL_FREE"

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError("unsupported proxy protocol version")
        if len(self.fixture_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.fixture_sha256
        ):
            raise ValueError("fixture identity must be a lowercase SHA-256")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("batch", "sequence", "repeats"):
            _positive_int(name, getattr(self, name))
        if self.dtype != "float32" or self.resource_class != "LOCAL_FREE":
            raise ScaleAdmissionError("only LOCAL_FREE CPU float32 proxies are supported")

    def identity(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True, slots=True)
class ProxyBudget:
    max_parameters: int = 100_000
    max_flops: int = 200_000_000
    max_memory_bytes: int = 128_000_000
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _positive_int(name, value)


@dataclass(frozen=True, slots=True)
class ArchitectureHypothesis:
    """Non-promoted ModelSpec delta; cannot mutate canonical architecture."""

    version: int
    hypothesis_id: str
    variant: str
    field: str
    value: int

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION or not self.hypothesis_id.strip():
            raise ValueError("invalid hypothesis identity/version")
        if self.variant not in ALLOWED_VARIANTS:
            raise ValueError("unsupported experimental variant")
        field_by_variant = {
            "baseline": "",
            "mlp_width": "d_ff",
            "depth": "n_layers",
            "gqa": "n_kv_heads",
            "heads": "n_heads",
            "context": "max_seq_len",
        }
        if self.field != field_by_variant[self.variant]:
            raise ValueError("variant/field mismatch")
        if self.variant != "baseline":
            _positive_int("hypothesis value", self.value)
        elif self.value != 0:
            raise ValueError("baseline must have value 0")

    def candidate(self, parent: ModelSpec) -> ModelSpec:
        if not isinstance(parent, ModelSpec):
            raise TypeError("parent must be a canonical ModelSpec")
        return parent if self.variant == "baseline" else replace(parent, **{self.field: self.value})


def estimate_proxy_resources(
    spec: ModelSpec, protocol: ProxyProtocol,
) -> dict[str, int]:
    """Conservative score-equivalent planning model; not observed RAM or GPU peak."""
    batch, seq = protocol.batch, protocol.sequence
    if seq > spec.max_seq_len:
        raise ScaleAdmissionError("sequence exceeds candidate context")
    q, kv, d, ff, layers = (
        spec.q_dim, spec.kv_dim, spec.d_model, spec.d_ff, spec.n_layers
    )
    tokens = batch * seq
    # 2 FLOPs per multiply-add; causal S^2 attention/softmax planning bound.
    forward_flops = (
        2 * tokens * layers * (d * (2 * q + 2 * kv) + 3 * d * ff)
        + 4 * batch * layers * spec.n_heads * seq * seq * spec.head_dim
        + 2 * tokens * d * spec.vocab_size
    )
    parameter_bytes = spec.parameter_count() * 4
    activation_bytes = 4 * (
        tokens * (d + spec.vocab_size + layers * (d + q + 2 * kv + 3 * ff))
        + batch * layers * spec.n_heads * seq * seq
    )
    return {
        "parameters": spec.parameter_count(),
        "parameter_bytes": parameter_bytes,
        "score_equivalent_activation_bytes": activation_bytes,
        "planning_memory_bytes": parameter_bytes + activation_bytes,
        "forward_flops": forward_flops,
        "total_proxy_flops": forward_flops * protocol.repeats,
        "total_proxy_tokens": tokens * protocol.repeats,
    }


def admit_proxy(
    spec: ModelSpec, protocol: ProxyProtocol, budget: ProxyBudget,
) -> dict[str, int]:
    resources = estimate_proxy_resources(spec, protocol)
    if resources["parameters"] > budget.max_parameters:
        raise ScaleAdmissionError("parameter budget exceeded")
    if resources["total_proxy_flops"] > budget.max_flops:
        raise ScaleAdmissionError("FLOP budget exceeded")
    if resources["planning_memory_bytes"] > budget.max_memory_bytes:
        raise ScaleAdmissionError("memory budget exceeded")
    if resources["total_proxy_tokens"] > budget.max_tokens:
        raise ScaleAdmissionError("token budget exceeded")
    return resources


def run_proxy(
    parent: ModelSpec,
    hypothesis: ArchitectureHypothesis,
    protocol: ProxyProtocol,
    budget: ProxyBudget,
    *,
    init: InitSpec | None = None,
) -> dict[str, Any]:
    """Execute an admitted tiny decoder on deterministic synthetic tokens only."""
    candidate = hypothesis.candidate(parent)
    resources = admit_proxy(candidate, protocol, budget)  # BEFORE allocation
    if not isinstance(parent, ModelSpec):
        raise TypeError("parent must be ModelSpec")
    init_spec = InitSpec() if init is None else init
    if not isinstance(init_spec, InitSpec):
        raise TypeError("init must be InitSpec")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(protocol.seed)
        model = TwelveSixDecoder(candidate, init_spec).cpu().eval()
        generator = torch.Generator(device="cpu").manual_seed(protocol.seed + 1)
        samples = torch.randint(
            0, candidate.vocab_size,
            (protocol.batch, protocol.sequence),
            generator=generator,
        )
        losses = []
        output_sha256 = []
        with torch.no_grad():
            for _ in range(protocol.repeats):
                logits = model(samples).logits.float()
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("nonfinite proxy logits")
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, candidate.vocab_size),
                    samples[:, 1:].reshape(-1),
                ) if protocol.sequence > 1 else logits.square().mean()
                if not math.isfinite(float(loss)):
                    raise RuntimeError("nonfinite proxy loss")
                losses.append(float(loss))
                output_sha256.append(hashlib.sha256(
                    logits.contiguous().numpy().tobytes()
                ).hexdigest())
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": _digest({
            "parent": parent.identity_sha256(),
            "hypothesis": asdict(hypothesis),
            "protocol": protocol.identity(),
            "init": init_spec.identity_sha256(),
        }),
        "hypothesis": asdict(hypothesis),
        "parent_modelspec_sha256": parent.identity_sha256(),
        "candidate_modelspec_sha256": candidate.identity_sha256(),
        "init_sha256": init_spec.identity_sha256(),
        "protocol_sha256": protocol.identity(),
        "resources": resources,
        "losses": losses,
        "output_sha256": output_sha256,
        "promotion_authorized": False,
        "training_executed": False,
        "paid_compute_authorized": False,
        "evidence_class": "LOCAL_FREE_SYNTHETIC_PROXY_ONLY",
    }


def compare_proxies(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Only same-protocol, same-parent/init records are comparable."""
    required = (
        "schema_version", "parent_modelspec_sha256", "init_sha256",
        "protocol_sha256", "resources", "losses", "evidence_class",
        "promotion_authorized", "training_executed",
    )
    if any(k not in left or k not in right for k in required):
        raise ValueError("incomplete experiment receipt")
    if any(
        left[k] != right[k] for k in (
            "schema_version", "parent_modelspec_sha256",
            "init_sha256", "protocol_sha256",
        )
    ) or left["schema_version"] != SCHEMA_VERSION:
        raise ValueError("non-comparable experiment protocol/identity")
    for receipt in (left, right):
        if (
            receipt["evidence_class"] != "LOCAL_FREE_SYNTHETIC_PROXY_ONLY"
            or receipt["promotion_authorized"] is not False
            or receipt["training_executed"] is not False
            or not receipt["losses"]
            or any(not math.isfinite(float(v)) for v in receipt["losses"])
        ):
            raise ValueError("invalid or promoted proxy receipt")
    return {
        "left_experiment_id": left.get("experiment_id"),
        "right_experiment_id": right.get("experiment_id"),
        "left_loss_mean": sum(left["losses"]) / len(left["losses"]),
        "right_loss_mean": sum(right["losses"]) / len(right["losses"]),
        "promotion_authorized": False,
        "interpretation": "synthetic mechanics comparison, not model quality",
    }


# Plan 7 / Section 2 -- bounded, audited function-preserving growth.
# This API does not alter ModelSpec v1, the training/checkpoint runtime or
# canonical stages.  Only FFN widening and identity residual depth extension
# have a function-preserving policy; all other changes fail closed.


class GrowthRejected(ValueError):
    """Requested growth lacks a supported semantics-preserving transform."""


def _growth_target(parent: ModelSpec, descendant: ModelSpec) -> tuple[int, int]:
    """Return FFN/depth increments only when all other ModelSpec semantics match."""
    if not isinstance(parent, ModelSpec) or not isinstance(descendant, ModelSpec):
        raise TypeError("growth requires ModelSpec parent and descendant")
    old = parent.to_dict()
    new = descendant.to_dict()
    for key in ("d_ff", "n_layers"):
        old.pop(key)
        new.pop(key)
    if old != new:
        raise GrowthRejected("unsupported geometry or vocabulary changed")
    if descendant.d_ff < parent.d_ff or descendant.n_layers < parent.n_layers:
        raise GrowthRejected("function-preserving growth cannot shrink dimensions")
    if descendant.d_ff == parent.d_ff and descendant.n_layers == parent.n_layers:
        raise GrowthRejected("descendant must increase FFN width or layer count")
    return descendant.d_ff - parent.d_ff, descendant.n_layers - parent.n_layers


def _growth_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise GrowthRejected("seed must be a nonnegative integer")
    return seed


def _growth_validate_parent(parent: TwelveSixDecoder) -> None:
    if not isinstance(parent, TwelveSixDecoder):
        raise TypeError("parent must be TwelveSixDecoder")
    if parent.training:
        raise GrowthRejected("growth parity requires parent.eval()")
    if any(parameter.device.type != "cpu" for parameter in parent.parameters()):
        raise ScaleAdmissionError("growth admission only supports local CPU models")
    if any(
        not bool(torch.isfinite(tensor).all())
        for tensor in parent.state_dict().values()
        if tensor.is_floating_point()
    ):
        raise GrowthRejected("nonfinite parent model state")


def _growth_state_sha256(model: TwelveSixDecoder) -> str:
    """Hash concrete tensor values, names, shapes, and dtypes, not only the schema."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        raw_name = name.encode("utf-8")
        descriptor = json.dumps(
            [list(tensor.shape), str(tensor.dtype)],
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(raw_name).to_bytes(4, "big"))
        digest.update(raw_name)
        digest.update(len(descriptor).to_bytes(4, "big"))
        digest.update(descriptor)
        contiguous = tensor.detach().cpu().contiguous().numpy().tobytes()
        digest.update(len(contiguous).to_bytes(8, "big"))
        digest.update(contiguous)
    return digest.hexdigest()


def _copy_growth_weights(
    parent: TwelveSixDecoder, descendant: TwelveSixDecoder,
) -> list[dict[str, Any]]:
    """Copy all incumbent tensors; zero widened downstream channels and new residuals."""
    original = parent.state_dict()
    target = descendant.state_dict()
    previous_layers = parent.spec.n_layers
    copied: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, dest in target.items():
            if name not in original:
                if not name.startswith("blocks."):
                    raise GrowthRejected(f"unrecognized descendant tensor: {name}")
                index = int(name.split(".")[1])
                if index < previous_layers:
                    raise GrowthRejected(f"missing incumbent tensor: {name}")
                # A genuinely new block is made an identity residual below.
                copied.append({"tensor": name, "policy": "NEW_RESIDUAL_LAYER"})
                continue

            old = original[name]
            if dest.shape == old.shape:
                dest.copy_(old)
                copied.append({"tensor": name, "policy": "EXACT_COPY"})
                continue
            allowed_prefix = (
                ".mlp.gate_proj.weight",
                ".mlp.up_proj.weight",
                ".mlp.gate_proj.bias",
                ".mlp.up_proj.bias",
            )
            if name.startswith("blocks.") and name.endswith(allowed_prefix):
                if dest.ndim != old.ndim or dest.shape[0] < old.shape[0]:
                    raise GrowthRejected(f"unsupported FFN tensor width: {name}")
                if dest.shape[1:] != old.shape[1:]:
                    raise GrowthRejected(f"unsupported FFN tensor input: {name}")
                dest.zero_()
                dest[:old.shape[0]].copy_(old)
                copied.append({
                    "tensor": name, "policy": "COPY_ROWS_ZERO_APPEND",
                    "old_shape": list(old.shape), "new_shape": list(dest.shape),
                })
            elif name.startswith("blocks.") and name.endswith(".mlp.down_proj.weight"):
                if dest.ndim != 2 or old.ndim != 2:
                    raise GrowthRejected("invalid FFN output projection rank")
                if dest.shape[0] != old.shape[0] or dest.shape[1] < old.shape[1]:
                    raise GrowthRejected(f"unsupported FFN output tensor: {name}")
                dest.zero_()
                dest[:, :old.shape[1]].copy_(old)
                copied.append({
                    "tensor": name, "policy": "COPY_COLS_ZERO_APPEND",
                    "old_shape": list(old.shape), "new_shape": list(dest.shape),
                })
            else:
                raise GrowthRejected(f"unsupported tensor transform: {name}")

        for index in range(previous_layers, descendant.spec.n_layers):
            block = descendant.blocks[index]
            block.attn.out_proj.weight.zero_()
            block.mlp.down_proj.weight.zero_()
            if block.attn.out_proj.bias is not None:
                block.attn.out_proj.bias.zero_()
            if block.mlp.down_proj.bias is not None:
                block.mlp.down_proj.bias.zero_()
    return copied


def _growth_inputs(spec: ModelSpec, protocol: ProxyProtocol) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(protocol.seed + 1)
    return torch.randint(
        0, spec.vocab_size, (protocol.batch, protocol.sequence),
        generator=generator,
    )


def grow_function_preserving(
    parent: TwelveSixDecoder,
    target: ModelSpec,
    protocol: ProxyProtocol,
    budget: ProxyBudget,
    *,
    seed: int = 0,
    parity_atol: float = 1e-5,
) -> tuple[TwelveSixDecoder, dict[str, Any]]:
    """Produce a certified LOCAL_FREE descendant, or return no descendant.

    No optimizer state or checkpoint lineage is transferred. The caller must
    start a new run with a new ModelSpec identity and may not promote this result.
    """
    _growth_seed(seed)
    if isinstance(parity_atol, bool) or not isinstance(parity_atol, (float, int)):
        raise GrowthRejected("parity tolerance must be finite and nonnegative")
    if not math.isfinite(parity_atol) or parity_atol < 0 or parity_atol > 1e-3:
        raise GrowthRejected("parity tolerance exceeds the bounded acceptance envelope")
    _growth_validate_parent(parent)
    width_added, layers_added = _growth_target(parent.spec, target)
    resources = admit_proxy(target, protocol, budget)  # Before allocating descendant
    admit_proxy(parent.spec, protocol, budget)
    parent_modelspec_sha = parent.spec.identity_sha256()
    parent_state_sha = _growth_state_sha256(parent)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        descendant = TwelveSixDecoder(target, parent.init_spec).cpu().eval()
        mapping = _copy_growth_weights(parent, descendant)
        inputs = _growth_inputs(parent.spec, protocol)
        with torch.no_grad():
            before = parent(inputs).logits.float()
            after = descendant(inputs).logits.float()
            if not bool(torch.isfinite(before).all() and torch.isfinite(after).all()):
                raise GrowthRejected("nonfinite parity logits")
            max_error = float((before - after).abs().max())
            if max_error > parity_atol:
                raise GrowthRejected(f"function parity failed: max error={max_error}")
    if _growth_state_sha256(parent) != parent_state_sha:
        raise GrowthRejected("parent weights mutated by growth operation")
    record = {
        "schema_version": 1,
        "policy": "COPY_PREFIX_ZERO_NEW_CHANNELS_AND_IDENTITY_RESIDUALS",
        "mode": "FUNCTION_PRESERVING_GROWTH",
        "parent_modelspec_sha256": parent_modelspec_sha,
        "parent_state_sha256": parent_state_sha,
        "descendant_modelspec_sha256": target.identity_sha256(),
        "descendant_state_sha256": _growth_state_sha256(descendant),
        "init_sha256": parent.init_spec.identity_sha256(),
        "protocol_sha256": protocol.identity(),
        "seed": seed,
        "added_ffn_width": width_added,
        "added_layers": layers_added,
        "tensor_mapping": mapping,
        "resource_admission": resources,
        "parity_max_abs_error": max_error,
        "parity_atol": parity_atol,
        "function_preserved_on_fixture": True,
        "optimizer_state_transferred": False,
        "checkpoint_resume_authorized": False,
        "stage_promotion_authorized": False,
        "training_authorized": False,
        "paid_compute_authorized": False,
        "evidence_class": "LOCAL_FREE_SYNTHETIC_GROWTH_PARITY_ONLY",
    }
    record["receipt_sha256"] = _digest(record)
    return descendant, record


def fresh_init_scale_up(
    parent: TwelveSixDecoder,
    target: ModelSpec,
    protocol: ProxyProtocol,
    budget: ProxyBudget,
    *,
    seed: int = 0,
) -> tuple[TwelveSixDecoder, dict[str, Any]]:
    """Explicitly non-preserving fallback, never described as weight transfer."""
    _growth_seed(seed)
    _growth_validate_parent(parent)
    if target.identity_sha256() == parent.spec.identity_sha256():
        raise GrowthRejected("fresh-init scale-up must change model identity")
    resources = admit_proxy(target, protocol, budget)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        descendant = TwelveSixDecoder(target, parent.init_spec).cpu().eval()
    record = {
        "schema_version": 1,
        "mode": "FRESH_INIT_SCALE_UP",
        "policy": "NO_WEIGHT_TRANSFER",
        "parent_modelspec_sha256": parent.spec.identity_sha256(),
        "parent_state_sha256": _growth_state_sha256(parent),
        "descendant_modelspec_sha256": target.identity_sha256(),
        "descendant_state_sha256": _growth_state_sha256(descendant),
        "init_sha256": parent.init_spec.identity_sha256(),
        "protocol_sha256": protocol.identity(),
        "seed": seed,
        "tensor_mapping": [],
        "resource_admission": resources,
        "function_preserved_on_fixture": False,
        "parity_max_abs_error": None,
        "optimizer_state_transferred": False,
        "checkpoint_resume_authorized": False,
        "stage_promotion_authorized": False,
        "training_authorized": False,
        "paid_compute_authorized": False,
        "evidence_class": "LOCAL_FREE_SYNTHETIC_FRESH_INIT_ONLY",
    }
    record["receipt_sha256"] = _digest(record)
    return descendant, record
