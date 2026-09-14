"""Fail-closed routing for the accelerated 20M -> 200M -> 1B scale path.

This module decides which evidence package may be prepared next.  It never grants
training or compute authorization and it deliberately keeps the learned-20M
terminal proof ahead of every larger-model activity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

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
    _expect(errors, isinstance(qualification, list), "backend_qualification_requirements_missing")
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
    _expect(errors, isinstance(backends, list) and bool(backends), "backend_candidates_missing")
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
            _expect(
                errors,
                _is_sha256(learned.get("evidence_manifest_sha256")),
                f"{name}_evidence_manifest_sha256_invalid",
            )
            satisfied = learned.get("requirements_satisfied")
            _expect(
                errors,
                isinstance(satisfied, list)
                and learned_requirements[name].issubset(set(satisfied)),
                f"{name}_requirements_incomplete",
            )
            _expect(
                errors,
                _valid_terminal_authority(learned.get("terminal_authority")),
                f"{name}_terminal_authority_invalid",
            )
            _expect(
                errors,
                _valid_terminal_authority(learned.get("independent_audit_authority")),
                f"{name}_independent_audit_authority_invalid",
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
            feasibility.get("decision") in {"NOT_EVALUATED", "GO", "HOLD", "NO_GO"},
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
    if not _is_sha256(value.get("evidence_manifest_sha256")):
        blockers.append(f"{name}_evidence_manifest_sha256_invalid")
        valid = False
    satisfied = value.get("requirements_satisfied")
    if not isinstance(satisfied, list) or not required.issubset(set(satisfied)):
        blockers.append(f"{name}_requirements_incomplete")
        valid = False
    for field in ("terminal_authority", "independent_audit_authority"):
        if not _valid_terminal_authority(value.get(field)):
            blockers.append(f"{name}_{field}_invalid")
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
