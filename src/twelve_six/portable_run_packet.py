"""Validate provider-neutral LOCAL_FREE/free-GPU training run packets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from twelve_six.accelerated_scaling import REPOSITORY, REQUIRED_RUN_PACKET_FIELDS

PACKET_ID = "R01-LEARNED20M-PORTABLE-RUN-PACKET-V1"
PACKET_SCHEMA_VERSION = 1
MODEL341_PARAMETER_COUNT = 20_613_440

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_KEY_RE = re.compile(
    r"(^|_)(api_key|secret|password|access_token|refresh_token|credential)(_|$)",
    re.IGNORECASE,
)

CONTRACT_FIELD_LOCATIONS = {
    "source_git_sha": ("identities", "source_git_sha"),
    "modelspec_sha256": ("identities", "modelspec_sha256"),
    "initspec_sha256": ("identities", "initspec_sha256"),
    "tokenizer_sha256": ("identities", "tokenizer_sha256"),
    "corpus_manifest_sha256": ("identities", "corpus_manifest_sha256"),
    "split_sha256": ("identities", "split_sha256"),
    "packing_sha256": ("identities", "packing_sha256"),
    "unique_loss_ledger_sha256": ("identities", "unique_loss_ledger_sha256"),
    "training_config_sha256": ("recipe", "training_config_sha256"),
    "optimizer_scheduler_precision": ("recipe", "optimizer_scheduler_precision"),
    "seed": ("recipe", "seed"),
    "target_unique_loss_positions": ("recipe", "target_unique_loss_positions"),
    "maximum_total_exposures": ("recipe", "maximum_total_exposures"),
    "checkpoint_lineage": ("checkpoint", "lineage"),
    "stop_resume_policy_sha256": ("checkpoint", "stop_resume_policy_sha256"),
    "evaluation_schedule_sha256": ("evaluation", "evaluation_schedule_sha256"),
    "resource_class": ("resource", "resource_class"),
    "maximum_cost": ("resource", "maximum_cost_usd"),
}

REQUIRED_AUTHORITIES = {
    "code",
    "model",
    "tokenizer",
    "data",
    "loss_ledger",
    "checkpoint_integrity",
    "evaluation_firewall",
    "backend",
}

ALLOWED_BACKENDS = {
    "PROJECT_NATIVE_PYTORCH",
    "LITGPT",
    "HF_ACCELERATE_OR_TRAINER",
    "PYTORCH_FSDP",
    "DEEPSPEED",
}


@dataclass(frozen=True)
class PortableRunAssessment:
    """Readiness is split between a fresh launch and a transferred resume."""

    contract_valid: bool
    ready_for_initial_local_free_launch: bool
    ready_for_cross_provider_resume: bool
    contract_errors: tuple[str, ...]
    launch_blockers: tuple[str, ...]
    resume_blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_valid": self.contract_valid,
            "ready_for_initial_local_free_launch": (
                self.ready_for_initial_local_free_launch
            ),
            "ready_for_cross_provider_resume": self.ready_for_cross_provider_resume,
            "contract_errors": list(self.contract_errors),
            "launch_blockers": list(self.launch_blockers),
            "resume_blockers": list(self.resume_blockers),
        }


def _is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and _GIT_SHA_RE.fullmatch(value) is not None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_zero_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == 0
    )


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _valid_terminal_authority(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    run_id = value.get("workflow_run_id")
    return (
        value.get("repository") == REPOSITORY
        and _is_git_sha(value.get("git_sha"))
        and _is_sha256(value.get("evidence_sha256"))
        and _is_positive_int(run_id)
        and value.get("workflow_conclusion") == "success"
        and value.get("terminal") is True
    )


def _find_embedded_secrets(value: Any, path: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _SECRET_KEY_RE.search(str(key)) and child not in (None, ""):
                errors.append(f"embedded_secret_forbidden:{child_path}")
            errors.extend(_find_embedded_secrets(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_find_embedded_secrets(child, f"{path}[{index}]"))
    return errors


def _get_mapping(data: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key}_missing")
        return {}
    return value


def validate_portable_run_contract(data: dict[str, Any]) -> list[str]:
    """Validate immutable safety/shape rules without claiming launch readiness."""
    errors: list[str] = []
    _expect(errors, data.get("schema_version") == PACKET_SCHEMA_VERSION, "schema_version_mismatch")
    _expect(errors, data.get("packet_id") == PACKET_ID, "packet_id_mismatch")
    _expect(
        errors,
        data.get("status") in {"BLOCKED_TEMPLATE", "READY_CANDIDATE"},
        "status_invalid",
    )

    declared = data.get("contract_fields")
    _expect(errors, isinstance(declared, list), "contract_fields_missing")
    if isinstance(declared, list):
        _expect(
            errors,
            set(declared) == REQUIRED_RUN_PACKET_FIELDS,
            "contract_fields_drift_from_accelerated_roadmap",
        )
        _expect(errors, len(declared) == len(set(declared)), "contract_fields_not_unique")

    boundaries = _get_mapping(data, "truth_boundary", errors)
    _expect(
        errors,
        boundaries.get("canonical_base") == "RANDOM_INIT_PRETRAINING_ONLY",
        "canonical_base_boundary_drift",
    )
    for key in (
        "foreign_pretrained_or_aligned_weights_used",
        "hidden_teacher_logits_used",
        "selection_validation_used_for_training",
        "final_test_payload_accessed",
        "replay_or_padding_counted_as_unique_exposure",
        "materially_paid_compute_authorized",
        "materially_paid_compute_requested",
    ):
        _expect(errors, boundaries.get(key) is False, f"truth_boundary_{key}_must_be_false")

    resource = _get_mapping(data, "resource", errors)
    _expect(
        errors,
        resource.get("resource_class") in {"LOCAL_FREE", "FREE_GPU"},
        "resource_class_must_be_local_free_or_free_gpu",
    )
    _expect(
        errors,
        resource.get("provider")
        in {"UNBOUND", "OWNER_LAPTOP", "KAGGLE", "COLAB", "OTHER_FREE"},
        "resource_provider_invalid",
    )
    _expect(
        errors,
        _is_zero_number(resource.get("maximum_cost_usd")),
        "maximum_cost_usd_must_be_zero",
    )
    _expect(
        errors,
        resource.get("materially_paid") is False,
        "materially_paid_must_be_false",
    )
    _expect(
        errors,
        resource.get("paid_authorization_ref") is None,
        "paid_authorization_ref_must_be_absent",
    )

    checkpoint = _get_mapping(data, "checkpoint", errors)
    _expect(
        errors,
        checkpoint.get("mode") in {"FRESH_START", "RESUME"},
        "checkpoint_mode_invalid",
    )
    for key in (
        "atomic_publish_required",
        "fresh_process_resume_required",
        "cross_provider_resume_required",
        "checkpoint_early_in_ephemeral_session",
    ):
        _expect(errors, checkpoint.get(key) is True, f"checkpoint_{key}_must_be_true")

    evaluation = _get_mapping(data, "evaluation", errors)
    _expect(
        errors,
        evaluation.get("selection_validation_only") is True,
        "selection_validation_only_must_be_true",
    )
    _expect(
        errors,
        evaluation.get("final_test_payload_access") is False,
        "final_test_payload_access_must_be_false",
    )

    identities = _get_mapping(data, "identities", errors)
    _expect(
        errors,
        identities.get("canonical_base") == "random_init",
        "identity_canonical_base_must_be_random_init",
    )
    _expect(
        errors,
        identities.get("parameter_count") == MODEL341_PARAMETER_COUNT,
        "model341_parameter_count_mismatch",
    )

    authorities = _get_mapping(data, "authorities", errors)
    _expect(
        errors,
        REQUIRED_AUTHORITIES.issubset(authorities),
        "required_authority_slots_missing",
    )

    for key in ("recipe", "runtime", "output"):
        _get_mapping(data, key, errors)

    errors.extend(_find_embedded_secrets(data))
    return sorted(set(errors))


def _require_sha256(blockers: list[str], value: Any, name: str) -> None:
    if not _is_sha256(value):
        blockers.append(f"{name}_invalid")


def _require_authority(blockers: list[str], value: Any, name: str) -> None:
    if not _valid_terminal_authority(value):
        blockers.append(f"{name}_authority_invalid")


def _artifact_uri_valid(value: Any) -> bool:
    if not _is_nonempty_string(value):
        return False
    parsed = urlsplit(value)
    if parsed.scheme not in {"file", "https", "s3", "gs", "hf"}:
        return False
    return parsed.username is None and parsed.password is None and not parsed.query


def _launch_blockers(data: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    identities = data.get("identities", {})
    recipe = data.get("recipe", {})
    checkpoint = data.get("checkpoint", {})
    evaluation = data.get("evaluation", {})
    resource = data.get("resource", {})
    runtime = data.get("runtime", {})
    output = data.get("output", {})
    authorities = data.get("authorities", {})

    if not _is_git_sha(identities.get("source_git_sha")):
        blockers.append("source_git_sha_invalid")
    for field in (
        "modelspec_sha256",
        "initspec_sha256",
        "tokenizer_sha256",
        "corpus_manifest_sha256",
        "split_sha256",
        "packing_sha256",
        "unique_loss_ledger_sha256",
    ):
        _require_sha256(blockers, identities.get(field), field)

    for name in REQUIRED_AUTHORITIES:
        _require_authority(blockers, authorities.get(name), name)

    _require_sha256(
        blockers,
        recipe.get("training_config_sha256"),
        "training_config_sha256",
    )
    optimizer = recipe.get("optimizer_scheduler_precision")
    if not isinstance(optimizer, dict):
        blockers.append("optimizer_scheduler_precision_invalid")
    else:
        for name in ("optimizer", "scheduler", "precision"):
            if not _is_nonempty_string(optimizer.get(name)):
                blockers.append(f"{name}_missing")
    if not _is_nonnegative_int(recipe.get("seed")):
        blockers.append("seed_invalid")

    target = recipe.get("target_unique_loss_positions")
    total = recipe.get("maximum_total_exposures")
    ledger = recipe.get("available_unique_loss_positions")
    replay_cap = recipe.get("max_exposures_per_unique_position")
    if not _is_positive_int(target):
        blockers.append("target_unique_loss_positions_invalid")
    if not _is_positive_int(total):
        blockers.append("maximum_total_exposures_invalid")
    if not _is_positive_int(ledger):
        blockers.append("available_unique_loss_positions_invalid")
    if replay_cap != 1:
        blockers.append("max_exposures_per_unique_position_must_be_one")
    if _is_positive_int(target) and _is_positive_int(ledger) and target > ledger:
        blockers.append("target_unique_loss_positions_exceed_ledger")
    if _is_positive_int(target) and _is_positive_int(total) and total != target:
        blockers.append("maximum_total_exposures_must_equal_unique_target")

    _require_sha256(
        blockers,
        checkpoint.get("stop_resume_policy_sha256"),
        "stop_resume_policy_sha256",
    )
    session_limit = checkpoint.get("session_time_limit_minutes")
    first_deadline = checkpoint.get("first_checkpoint_deadline_minutes")
    if not _is_positive_int(session_limit):
        blockers.append("session_time_limit_minutes_invalid")
    if not _is_positive_int(first_deadline):
        blockers.append("first_checkpoint_deadline_minutes_invalid")
    if (
        _is_positive_int(session_limit)
        and _is_positive_int(first_deadline)
        and first_deadline >= session_limit
    ):
        blockers.append("first_checkpoint_deadline_must_precede_session_limit")
    if not _is_positive_int(checkpoint.get("checkpoint_every_steps")):
        blockers.append("checkpoint_every_steps_invalid")

    _require_sha256(
        blockers,
        evaluation.get("evaluation_schedule_sha256"),
        "evaluation_schedule_sha256",
    )
    if runtime.get("backend_id") not in ALLOWED_BACKENDS:
        blockers.append("backend_id_not_qualified_candidate")
    for name in ("python_version", "framework_version", "device_type"):
        if not _is_nonempty_string(runtime.get(name)):
            blockers.append(f"runtime_{name}_missing")
    _require_sha256(
        blockers,
        runtime.get("environment_lock_sha256"),
        "environment_lock_sha256",
    )

    if resource.get("provider") == "UNBOUND":
        blockers.append("resource_provider_unbound")
    if not _artifact_uri_valid(output.get("artifact_store_uri")):
        blockers.append("artifact_store_uri_invalid_or_credential_bearing")
    if output.get("content_addressed") is not True:
        blockers.append("output_must_be_content_addressed")

    return sorted(set(blockers))


def _resume_blockers(data: dict[str, Any]) -> list[str]:
    blockers = _launch_blockers(data)
    checkpoint = data.get("checkpoint", {})
    resource = data.get("resource", {})
    authorities = data.get("authorities", {})
    if checkpoint.get("mode") != "RESUME":
        blockers.append("checkpoint_mode_is_not_resume")
        return sorted(set(blockers))

    lineage = checkpoint.get("lineage")
    if not isinstance(lineage, dict):
        blockers.append("checkpoint_lineage_invalid")
        return sorted(set(blockers))
    for name in ("parent_checkpoint_sha256", "parent_manifest_sha256"):
        _require_sha256(blockers, lineage.get(name), name)
    if not _is_nonempty_string(lineage.get("previous_run_id")):
        blockers.append("previous_run_id_missing")
    if lineage.get("resume_validated") is not True:
        blockers.append("parent_checkpoint_resume_not_validated")
    if lineage.get("cross_provider_transfer") is not True:
        blockers.append("cross_provider_transfer_not_declared")
    source_provider = lineage.get("source_provider")
    target_provider = resource.get("provider")
    if not _is_nonempty_string(source_provider):
        blockers.append("source_provider_missing")
    elif source_provider == target_provider:
        blockers.append("cross_provider_source_and_target_must_differ")
    _require_authority(
        blockers,
        authorities.get("parent_checkpoint"),
        "parent_checkpoint",
    )
    return sorted(set(blockers))


def assess_portable_run_packet(data: dict[str, Any]) -> PortableRunAssessment:
    """Assess a packet without turning readiness into execution authority."""
    contract_errors = validate_portable_run_contract(data)
    if contract_errors:
        errors = tuple(contract_errors)
        return PortableRunAssessment(
            contract_valid=False,
            ready_for_initial_local_free_launch=False,
            ready_for_cross_provider_resume=False,
            contract_errors=errors,
            launch_blockers=errors,
            resume_blockers=errors,
        )

    launch = _launch_blockers(data)
    resume = _resume_blockers(data)
    checkpoint = data["checkpoint"]
    ready_initial = not launch and checkpoint.get("mode") == "FRESH_START"
    ready_resume = not resume and checkpoint.get("mode") == "RESUME"
    return PortableRunAssessment(
        contract_valid=True,
        ready_for_initial_local_free_launch=ready_initial,
        ready_for_cross_provider_resume=ready_resume,
        contract_errors=(),
        launch_blockers=tuple(launch),
        resume_blockers=tuple(resume),
    )
