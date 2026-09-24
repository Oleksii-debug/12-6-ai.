"""Bind terminal learned-20M readiness evidence into a portable run packet."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from twelve_six.learned20m_readiness import assess_learned20m_readiness
from twelve_six.portable_run_packet import (
    PortableRunAssessment,
    assess_portable_run_packet,
    validate_portable_run_contract,
)

OVERLAY_ID = "R01-LEARNED20M-PORTABLE-SESSION-OVERLAY-V1"
OVERLAY_SCHEMA_VERSION = 1

_MODEL341_MODELSPEC_SHA256 = "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
_MODEL341_INITSPEC_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
_MODEL341_PARAMETER_COUNT = 20_613_440
_LEARN345_POLICY_IDENTITY_SHA256 = "84152a673c4ed8fd34f4b81b03a96b4a3f5b40a22d961f436cbed11af23000e7"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_ROOT_KEYS = {
    "schema_version",
    "overlay_id",
    "status",
    "scientific_bindings",
    "checkpoint",
    "evaluation",
    "runtime",
    "resource",
    "output",
}
_SECTION_KEYS = {
    "scientific_bindings": {
        "initspec_sha256",
        "seed",
        "optimizer_scheduler_precision",
        "authorities",
    },
    "optimizer_scheduler_precision": {"optimizer", "scheduler", "precision"},
    "authorities": {"code", "model", "backend"},
    "checkpoint": {
        "mode",
        "lineage",
        "parent_checkpoint_authority",
        "session_time_limit_minutes",
        "first_checkpoint_deadline_minutes",
        "checkpoint_every_steps",
    },
    "lineage": {
        "parent_checkpoint_sha256",
        "parent_manifest_sha256",
        "previous_run_id",
        "source_provider",
        "cross_provider_transfer",
        "resume_validated",
    },
    "evaluation": {"evaluation_schedule_sha256"},
    "runtime": {
        "backend_id",
        "python_version",
        "framework_version",
        "environment_lock_sha256",
        "device_type",
    },
    "resource": {
        "resource_class",
        "provider",
        "maximum_cost_usd",
        "materially_paid",
        "paid_authorization_ref",
    },
    "output": {"artifact_store_uri", "content_addressed"},
}
_EXPECTED_EXECUTION_ROOT_KEYS = {"model", "training", "session"}
_EXPECTED_MODEL_KEYS = {
    "modelspec_sha256",
    "initspec_sha256",
    "parameter_count",
    "canonical_base",
}
_EXPECTED_TRAINING_KEYS = {
    "training_config_sha256",
    "stopping_policy_sha256",
    "policy_identity_sha256",
    "optimizer",
    "learning_rate",
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
}
_EXPECTED_SEED_KEYS = {"model_init", "data_order", "dataloader"}
_EXPECTED_SESSION_KEYS = {
    "scientific_bindings",
    "checkpoint",
    "evaluation",
    "runtime",
    "resource",
    "output",
}
_LEARN345_FIXED_TRAINING = {
    "policy_identity_sha256": _LEARN345_POLICY_IDENTITY_SHA256,
    "optimizer": "AdamW",
    "learning_rate": 0.00022,
    "betas": [0.9, 0.95],
    "eps": 1e-08,
    "weight_decay": 0.1,
    "gradient_clip_norm": 1.0,
    "scheduler": "constant",
    "warmup_steps": 0,
    "sequence_length": 128,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 1,
    "precision": "fp32",
    "seed_vector": {
        "model_init": 20260826,
        "data_order": 20260826,
        "dataloader": 20260826,
    },
}


@dataclass(frozen=True)
class PortableRunBinding:
    """A packet is exposed only when every upstream and session gate passes."""

    binding_ready: bool
    mode: str | None
    readiness_ready: bool
    overlay_contract_valid: bool
    packet_contract_valid: bool
    blockers: tuple[str, ...]
    readiness_sha256: str | None
    overlay_sha256: str | None
    portable_execution_sha256: str | None
    packet_sha256: str | None
    packet: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "binding_ready": self.binding_ready,
            "mode": self.mode,
            "readiness_ready": self.readiness_ready,
            "overlay_contract_valid": self.overlay_contract_valid,
            "packet_contract_valid": self.packet_contract_valid,
            "blockers": list(self.blockers),
            "readiness_sha256": self.readiness_sha256,
            "overlay_sha256": self.overlay_sha256,
            "portable_execution_sha256": self.portable_execution_sha256,
            "packet_sha256": self.packet_sha256,
        }


def canonical_sha256(value: Any) -> str:
    """Hash canonical JSON so the binding records exact input identities."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _exact_mapping(
    errors: list[str],
    parent: dict[str, Any],
    key: str,
    expected_keys: set[str],
) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        errors.append(f"overlay_{key}_missing")
        return {}
    missing = expected_keys - set(value)
    unexpected = set(value) - expected_keys
    errors.extend(f"overlay_{key}_{name}_missing" for name in sorted(missing))
    errors.extend(f"overlay_{key}_{name}_unexpected" for name in sorted(unexpected))
    return value


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_zero_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == 0
    )


def _validate_ready_candidate_scalars(
    errors: list[str],
    scientific: dict[str, Any],
    checkpoint: dict[str, Any],
    lineage: dict[str, Any],
    resource: dict[str, Any],
    output: dict[str, Any],
) -> None:
    if not _is_nonnegative_int(scientific.get("seed")):
        errors.append("overlay_seed_invalid")

    for name in (
        "session_time_limit_minutes",
        "first_checkpoint_deadline_minutes",
        "checkpoint_every_steps",
    ):
        if not _is_positive_int(checkpoint.get(name)):
            errors.append(f"overlay_{name}_invalid")

    for name in ("cross_provider_transfer", "resume_validated"):
        if not isinstance(lineage.get(name), bool):
            errors.append(f"overlay_lineage_{name}_must_be_boolean")

    if not _is_zero_number(resource.get("maximum_cost_usd")):
        errors.append("overlay_resource_maximum_cost_usd_must_be_zero")
    if resource.get("materially_paid") is not False:
        errors.append("overlay_resource_materially_paid_must_be_false")
    if output.get("content_addressed") is not True:
        errors.append("overlay_output_content_addressed_must_be_true")


def _exact_keys(
    errors: list[str],
    value: Any,
    expected_keys: set[str],
    prefix: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{prefix}_must_be_object")
        return {}
    missing = expected_keys - set(value)
    unexpected = set(value) - expected_keys
    errors.extend(f"{prefix}_{name}_missing" for name in sorted(missing))
    errors.extend(f"{prefix}_{name}_unexpected" for name in sorted(unexpected))
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def validate_session_overlay_contract(value: Any) -> list[str]:
    """Reject drift and scalar coercions before they can influence a run packet."""
    if not isinstance(value, dict):
        return ["overlay_root_must_be_object"]

    errors: list[str] = []
    missing = _ROOT_KEYS - set(value)
    unexpected = set(value) - _ROOT_KEYS
    errors.extend(f"overlay_root_{name}_missing" for name in sorted(missing))
    errors.extend(f"overlay_root_{name}_unexpected" for name in sorted(unexpected))

    schema_version = value.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != OVERLAY_SCHEMA_VERSION
    ):
        errors.append("overlay_schema_version_mismatch")
    if value.get("overlay_id") != OVERLAY_ID:
        errors.append("overlay_id_mismatch")
    if value.get("status") not in {"BLOCKED_TEMPLATE", "READY_CANDIDATE"}:
        errors.append("overlay_status_invalid")

    scientific = _exact_mapping(
        errors,
        value,
        "scientific_bindings",
        _SECTION_KEYS["scientific_bindings"],
    )
    _exact_mapping(
        errors,
        scientific,
        "optimizer_scheduler_precision",
        _SECTION_KEYS["optimizer_scheduler_precision"],
    )
    _exact_mapping(
        errors,
        scientific,
        "authorities",
        _SECTION_KEYS["authorities"],
    )
    checkpoint = _exact_mapping(
        errors,
        value,
        "checkpoint",
        _SECTION_KEYS["checkpoint"],
    )
    lineage = _exact_mapping(errors, checkpoint, "lineage", _SECTION_KEYS["lineage"])
    resource = _exact_mapping(errors, value, "resource", _SECTION_KEYS["resource"])
    output = _exact_mapping(errors, value, "output", _SECTION_KEYS["output"])
    for section in ("evaluation", "runtime"):
        _exact_mapping(errors, value, section, _SECTION_KEYS[section])

    if value.get("status") == "READY_CANDIDATE":
        _validate_ready_candidate_scalars(
            errors,
            scientific,
            checkpoint,
            lineage,
            resource,
            output,
        )
    return sorted(set(errors))


def _session_projection(overlay: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(overlay.get(key))
        for key in (
            "scientific_bindings",
            "checkpoint",
            "evaluation",
            "runtime",
            "resource",
            "output",
        )
    }


def validate_authenticated_portable_execution(value: Any) -> list[str]:
    """Validate the root-authenticated MODEL-341/LEARN-345 run projection."""
    if value is None:
        return ["authenticated_portable_execution_missing"]
    errors: list[str] = []
    root = _exact_keys(
        errors,
        value,
        _EXPECTED_EXECUTION_ROOT_KEYS,
        "authenticated_portable_execution",
    )
    model = _exact_keys(
        errors,
        root.get("model"),
        _EXPECTED_MODEL_KEYS,
        "authenticated_portable_execution_model",
    )
    training = _exact_keys(
        errors,
        root.get("training"),
        _EXPECTED_TRAINING_KEYS,
        "authenticated_portable_execution_training",
    )
    session = _exact_keys(
        errors,
        root.get("session"),
        _EXPECTED_SESSION_KEYS,
        "authenticated_portable_execution_session",
    )

    expected_model = {
        "modelspec_sha256": _MODEL341_MODELSPEC_SHA256,
        "initspec_sha256": _MODEL341_INITSPEC_SHA256,
        "parameter_count": _MODEL341_PARAMETER_COUNT,
        "canonical_base": "random_init",
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            errors.append(f"authenticated_portable_execution_model_{key}_mismatch")

    if not _is_sha256(training.get("training_config_sha256")):
        errors.append("authenticated_portable_execution_training_config_sha256_invalid")
    if not _is_sha256(training.get("stopping_policy_sha256")):
        errors.append("authenticated_portable_execution_stopping_policy_sha256_invalid")
    for key, expected in _LEARN345_FIXED_TRAINING.items():
        if training.get(key) != expected:
            errors.append(f"authenticated_portable_execution_training_{key}_mismatch")

    seed_vector = _exact_keys(
        errors,
        training.get("seed_vector"),
        _EXPECTED_SEED_KEYS,
        "authenticated_portable_execution_seed_vector",
    )
    if seed_vector and any(seed_vector.get(key) != 20260826 for key in _EXPECTED_SEED_KEYS):
        errors.append("authenticated_portable_execution_seed_vector_mismatch")

    synthetic_overlay = {
        "schema_version": OVERLAY_SCHEMA_VERSION,
        "overlay_id": OVERLAY_ID,
        "status": "READY_CANDIDATE",
        **copy.deepcopy(session),
    }
    session_errors = validate_session_overlay_contract(synthetic_overlay)
    errors.extend(
        f"authenticated_portable_execution_session:{item}" for item in session_errors
    )

    bindings = _mapping(session.get("scientific_bindings"))
    summary = _mapping(bindings.get("optimizer_scheduler_precision"))
    if bindings.get("initspec_sha256") != _MODEL341_INITSPEC_SHA256:
        errors.append("authenticated_portable_execution_session_initspec_mismatch")
    if bindings.get("seed") != 20260826:
        errors.append("authenticated_portable_execution_session_seed_mismatch")
    expected_summary = {
        "optimizer": "AdamW",
        "scheduler": "constant",
        "precision": "fp32",
    }
    if summary != expected_summary:
        errors.append("authenticated_portable_execution_session_recipe_summary_mismatch")

    return sorted(set(errors))


def _coherence_blockers(
    readiness: dict[str, Any],
    overlay: dict[str, Any],
    expected_execution: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    evidence = _mapping(readiness.get("evidence"))
    code = _mapping(evidence.get("code"))
    model = _mapping(readiness.get("model_authority"))
    recipe_evidence = _mapping(evidence.get("training_recipe"))
    bindings = _mapping(overlay.get("scientific_bindings"))
    authorities = _mapping(bindings.get("authorities"))
    runtime = _mapping(overlay.get("runtime"))
    expected_model = _mapping(expected_execution.get("model"))
    expected_training = _mapping(expected_execution.get("training"))
    expected_session = _mapping(expected_execution.get("session"))

    if _session_projection(overlay) != expected_session:
        blockers.append("authenticated_session_projection_mismatch")

    for key in ("modelspec_sha256", "parameter_count", "canonical_base"):
        if expected_model.get(key) != model.get(key):
            blockers.append(f"authenticated_model_{key}_mismatch")
    if expected_training.get("training_config_sha256") != recipe_evidence.get("config_sha256"):
        blockers.append("authenticated_training_config_sha256_mismatch")
    if expected_training.get("stopping_policy_sha256") != recipe_evidence.get(
        "stopping_policy_sha256"
    ):
        blockers.append("authenticated_stopping_policy_sha256_mismatch")

    code_authority = _mapping(authorities.get("code"))
    if code_authority and code_authority.get("git_sha") != code.get("git_sha"):
        blockers.append("code_authority_git_sha_mismatch")

    model_authority = _mapping(authorities.get("model"))
    if model_authority:
        if model_authority.get("git_sha") != model.get("git_sha"):
            blockers.append("model_authority_git_sha_mismatch")
        if model_authority.get("modelspec_sha256") != model.get("modelspec_sha256"):
            blockers.append("model_authority_modelspec_sha256_mismatch")

    backend_authority = _mapping(authorities.get("backend"))
    if backend_authority:
        if backend_authority.get("backend_id") != runtime.get("backend_id"):
            blockers.append("backend_authority_backend_id_mismatch")
        if (
            backend_authority.get("environment_lock_sha256")
            != runtime.get("environment_lock_sha256")
        ):
            blockers.append("backend_authority_environment_lock_sha256_mismatch")
    return sorted(set(blockers))


def _build_candidate(
    readiness: dict[str, Any],
    template: dict[str, Any],
    overlay: dict[str, Any],
    expected_execution: dict[str, Any],
    *,
    readiness_sha256: str,
    overlay_sha256: str,
    portable_execution_sha256: str,
) -> dict[str, Any]:
    packet = copy.deepcopy(template)
    evidence = _mapping(readiness.get("evidence"))
    model = _mapping(readiness.get("model_authority"))
    code = _mapping(evidence.get("code"))
    corpus = _mapping(evidence.get("corpus"))
    tokenizer = _mapping(evidence.get("tokenizer"))
    ledger = _mapping(evidence.get("loss_ledger"))
    checkpoint_evidence = _mapping(evidence.get("checkpoint_integrity"))
    evaluation_evidence = _mapping(evidence.get("evaluation"))
    recipe_evidence = _mapping(evidence.get("training_recipe"))

    expected_training = _mapping(expected_execution.get("training"))
    expected_session = _mapping(expected_execution.get("session"))
    bindings = _mapping(expected_session.get("scientific_bindings"))
    binding_authorities = _mapping(bindings.get("authorities"))
    checkpoint_overlay = _mapping(expected_session.get("checkpoint"))

    packet["status"] = "READY_CANDIDATE"
    packet["identities"].update(
        {
            "source_git_sha": code.get("git_sha"),
            "modelspec_sha256": model.get("modelspec_sha256"),
            "initspec_sha256": bindings.get("initspec_sha256"),
            "tokenizer_sha256": tokenizer.get("identity_sha256"),
            "corpus_manifest_sha256": corpus.get("manifest_sha256"),
            "split_sha256": corpus.get("split_sha256"),
            "packing_sha256": corpus.get("packing_sha256"),
            "unique_loss_ledger_sha256": ledger.get("identity_sha256"),
            "canonical_base": model.get("canonical_base"),
            "parameter_count": model.get("parameter_count"),
        }
    )
    packet["authorities"].update(
        {
            "code": copy.deepcopy(binding_authorities.get("code")),
            "model": copy.deepcopy(binding_authorities.get("model")),
            "tokenizer": copy.deepcopy(tokenizer.get("authority")),
            "data": copy.deepcopy(corpus.get("authority")),
            "loss_ledger": copy.deepcopy(ledger.get("authority")),
            "checkpoint_integrity": copy.deepcopy(checkpoint_evidence.get("authority")),
            "evaluation_firewall": copy.deepcopy(
                evaluation_evidence.get("firewall_authority")
            ),
            "backend": copy.deepcopy(binding_authorities.get("backend")),
            "parent_checkpoint": copy.deepcopy(
                checkpoint_overlay.get("parent_checkpoint_authority")
            ),
        }
    )
    packet["recipe"].update(
        {
            "training_config_sha256": recipe_evidence.get("config_sha256"),
            "optimizer_scheduler_precision": copy.deepcopy(
                bindings.get("optimizer_scheduler_precision")
            ),
            "seed": bindings.get("seed"),
            "target_unique_loss_positions": recipe_evidence.get(
                "requested_unique_loss_positions"
            ),
            "maximum_total_exposures": recipe_evidence.get(
                "requested_total_training_exposures"
            ),
            "available_unique_loss_positions": ledger.get(
                "unique_causal_loss_positions"
            ),
            "max_exposures_per_unique_position": recipe_evidence.get(
                "max_exposures_per_unique_position"
            ),
            "execution_projection": copy.deepcopy(expected_training),
        }
    )
    packet["checkpoint"].update(
        {
            "mode": checkpoint_overlay.get("mode"),
            "lineage": copy.deepcopy(checkpoint_overlay.get("lineage")),
            "stop_resume_policy_sha256": recipe_evidence.get(
                "stopping_policy_sha256"
            ),
            "session_time_limit_minutes": checkpoint_overlay.get(
                "session_time_limit_minutes"
            ),
            "first_checkpoint_deadline_minutes": checkpoint_overlay.get(
                "first_checkpoint_deadline_minutes"
            ),
            "checkpoint_every_steps": checkpoint_overlay.get(
                "checkpoint_every_steps"
            ),
        }
    )
    packet["evaluation"].update(
        copy.deepcopy(_mapping(expected_session.get("evaluation")))
    )
    packet["runtime"].update(copy.deepcopy(_mapping(expected_session.get("runtime"))))
    packet["resource"].update(copy.deepcopy(_mapping(expected_session.get("resource"))))
    packet["output"].update(copy.deepcopy(_mapping(expected_session.get("output"))))
    packet["binding"] = {
        "readiness_campaign_id": readiness.get("campaign_id"),
        "readiness_sha256": readiness_sha256,
        "session_overlay_id": overlay.get("overlay_id"),
        "session_overlay_sha256": overlay_sha256,
        "portable_execution_sha256": portable_execution_sha256,
    }
    return packet


def bind_portable_run_packet(
    readiness: Any,
    template: Any,
    overlay: Any,
    *,
    expected_portable_execution: Any = None,
    verified_scientific_authorities: Collection[str] = (),
    verified_authorization_refs: Collection[str] = (),
) -> PortableRunBinding:
    """Return a runnable packet only when upstream and external execution roots pass."""
    blockers: list[str] = []
    readiness_data = _mapping(readiness)
    template_data = _mapping(template)
    overlay_data = _mapping(overlay)
    expected_execution_data = _mapping(expected_portable_execution)

    readiness_hash = canonical_sha256(readiness) if isinstance(readiness, dict) else None
    overlay_hash = canonical_sha256(overlay) if isinstance(overlay, dict) else None
    execution_errors = validate_authenticated_portable_execution(
        expected_portable_execution
    )
    execution_hash = (
        canonical_sha256(expected_portable_execution)
        if isinstance(expected_portable_execution, dict) and not execution_errors
        else None
    )
    blockers.extend(f"binding:{item}" for item in execution_errors)

    readiness_result = assess_learned20m_readiness(
        readiness_data,
        verified_scientific_authorities=verified_scientific_authorities,
        verified_authorization_refs=verified_authorization_refs,
    )
    if not readiness_result.ready_for_local_free_pilot:
        blockers.extend(
            f"readiness:{item}" for item in readiness_result.local_free_pilot_blockers
        )

    template_errors = (
        validate_portable_run_contract(template_data)
        if isinstance(template, dict)
        else ["template_root_must_be_object"]
    )
    blockers.extend(f"template:{item}" for item in template_errors)

    overlay_errors = validate_session_overlay_contract(overlay)
    blockers.extend(f"overlay:{item}" for item in overlay_errors)
    if isinstance(overlay, dict) and overlay.get("status") != "READY_CANDIDATE":
        blockers.append("overlay:status_not_ready_candidate")

    mode = _mapping(overlay_data.get("checkpoint")).get("mode")
    packet_assessment: PortableRunAssessment | None = None
    candidate: dict[str, Any] | None = None
    coherence: list[str] = []
    if (
        not template_errors
        and not overlay_errors
        and not execution_errors
        and readiness_hash
        and overlay_hash
        and execution_hash
    ):
        coherence = _coherence_blockers(
            readiness_data,
            overlay_data,
            expected_execution_data,
        )
        blockers.extend(f"binding:{item}" for item in coherence)
        candidate = _build_candidate(
            readiness_data,
            template_data,
            overlay_data,
            expected_execution_data,
            readiness_sha256=readiness_hash,
            overlay_sha256=overlay_hash,
            portable_execution_sha256=execution_hash,
        )
        packet_assessment = assess_portable_run_packet(candidate)
        relevant = (
            packet_assessment.resume_blockers
            if mode == "RESUME"
            else packet_assessment.launch_blockers
        )
        blockers.extend(f"packet:{item}" for item in relevant)

    packet_contract_valid = bool(
        packet_assessment is not None and packet_assessment.contract_valid
    )
    desired_packet_ready = bool(
        packet_assessment is not None
        and (
            (mode == "FRESH_START" and packet_assessment.ready_for_initial_local_free_launch)
            or (mode == "RESUME" and packet_assessment.ready_for_cross_provider_resume)
        )
    )
    ready = bool(
        readiness_result.ready_for_local_free_pilot
        and not overlay_errors
        and not execution_errors
        and overlay_data.get("status") == "READY_CANDIDATE"
        and not coherence
        and desired_packet_ready
    )
    exposed_packet = candidate if ready else None
    return PortableRunBinding(
        binding_ready=ready,
        mode=mode if isinstance(mode, str) else None,
        readiness_ready=readiness_result.ready_for_local_free_pilot,
        overlay_contract_valid=not overlay_errors,
        packet_contract_valid=packet_contract_valid,
        blockers=tuple(sorted(set(blockers))),
        readiness_sha256=readiness_hash,
        overlay_sha256=overlay_hash,
        portable_execution_sha256=execution_hash,
        packet_sha256=canonical_sha256(candidate) if ready and candidate else None,
        packet=exposed_packet,
    )
