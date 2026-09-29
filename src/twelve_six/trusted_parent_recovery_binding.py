"""Operational same-provider resume binding over canonical D05 recovery evidence.

The provider-neutral portable packet intentionally remains fail-closed for same-provider
resume. This module is the thin composition boundary that can discharge exactly the
``trusted_parent_recovery_binding_missing`` blocker after canonical D05 recovery has
been resolved against an externally fixed reference and a terminal recovery authority
is bound to the exact trusted projection. It does not create a second checkpoint or
session framework.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from twelve_six.accelerated_scaling import REPOSITORY
from twelve_six.checkpoint import LoadResult, load_trainer_checkpoint
from twelve_six.portable_run_binding import (
    PortableRunBinding,
    _build_candidate,
    bind_portable_run_packet,
    canonical_sha256,
)
from twelve_six.portable_run_packet import assess_portable_run_packet
from twelve_six.scale141_recovery import (
    RecoveryLifecycleError,
    RecoveryResolution,
    resolve_recovery_generation,
)
from twelve_six.scale141_resume_sidecar import SIDECAR_SCHEMA

TRUSTED_PARENT_RECOVERY_BINDING_SCHEMA = (
    "12-6.trusted-parent-recovery-session-binding.v1"
)
TRUSTED_RECOVERY_AUTHORITY_ROLE = "trusted_parent_recovery"

_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA64_RE = re.compile(r"^[0-9a-f]{64}$")
_COARSE_PROVIDERS = frozenset({"OWNER_LAPTOP", "KAGGLE", "COLAB", "OTHER_FREE"})
_REFERENCE_KEYS = frozenset(
    {
        "generation",
        "object_key",
        "checkpoint_id",
        "manifest_sha256",
        "pointer_sha256",
        "source_sha",
        "run_manifest_hash",
        "optimizer_step",
        "tokens_seen",
    }
)
_AUTHORITY_KEYS = frozenset(
    {
        "repository",
        "git_sha",
        "evidence_sha256",
        "workflow_run_id",
        "workflow_conclusion",
        "terminal",
    }
)
_BINDING_KEYS_WITHOUT_HASH = frozenset(
    {
        "schema",
        "checkpoint_id",
        "checkpoint_manifest_sha256",
        "recovery_pointer_sha256",
        "recovery_generation",
        "recovery_object_key",
        "source_git_sha",
        "run_manifest_sha256",
        "previous_run_id",
        "optimizer_step",
        "tokens_seen",
        "d04_state_identity_sha256",
        "ordered_next_exposure_identity_sha256",
        "ledger_identity_sha256",
        "materialization_identity_sha256",
        "packing_identity_sha256",
        "exposure_plan_identity_sha256",
        "provider_class",
        "provider_id",
        "provider_session_id",
        "previous_provider_session_id",
        "terminal_recovery_authority_identity",
    }
)
_BINDING_KEYS = _BINDING_KEYS_WITHOUT_HASH | {"binding_sha256"}


class TrustedParentRecoveryBindingError(ValueError):
    """The D05/session projection cannot establish an exact trusted binding."""


def _is_sha40(value: Any) -> bool:
    return isinstance(value, str) and _SHA40_RE.fullmatch(value) is not None


def _is_sha64(value: Any) -> bool:
    return isinstance(value, str) and _SHA64_RE.fullmatch(value) is not None


def _require_sha64(value: Any, name: str) -> str:
    if not _is_sha64(value):
        raise TrustedParentRecoveryBindingError(f"{name}_invalid")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrustedParentRecoveryBindingError(f"{name}_invalid")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TrustedParentRecoveryBindingError(f"{name}_invalid")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    observed = _require_nonnegative_int(value, name)
    if observed == 0:
        raise TrustedParentRecoveryBindingError(f"{name}_invalid")
    return observed


def _exact_authority(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrustedParentRecoveryBindingError("terminal_recovery_authority_invalid")
    authority = dict(value)
    if set(authority) != _AUTHORITY_KEYS:
        raise TrustedParentRecoveryBindingError("terminal_recovery_authority_fields_invalid")
    if authority.get("repository") != REPOSITORY:
        raise TrustedParentRecoveryBindingError("terminal_recovery_authority_repository_invalid")
    if not _is_sha40(authority.get("git_sha")):
        raise TrustedParentRecoveryBindingError("terminal_recovery_authority_git_sha_invalid")
    _require_sha64(
        authority.get("evidence_sha256"),
        "terminal_recovery_authority_evidence_sha256",
    )
    _require_positive_int(
        authority.get("workflow_run_id"),
        "terminal_recovery_authority_workflow_run_id",
    )
    if authority.get("workflow_conclusion") != "success":
        raise TrustedParentRecoveryBindingError(
            "terminal_recovery_authority_workflow_conclusion_invalid"
        )
    if authority.get("terminal") is not True:
        raise TrustedParentRecoveryBindingError("terminal_recovery_authority_not_terminal")
    return authority


def _terminal_authority_identity(authority: Mapping[str, Any]) -> str:
    """Identify the terminal envelope without creating a hash cycle through evidence."""
    exact = _exact_authority(authority)
    core = {key: value for key, value in exact.items() if key != "evidence_sha256"}
    return canonical_sha256(
        {"role": TRUSTED_RECOVERY_AUTHORITY_ROLE, "terminal_authority": core}
    )


def trusted_recovery_authority_token(authority: Mapping[str, Any]) -> str:
    """Return the verification token consumed only after external authority checking."""
    exact = _exact_authority(authority)
    return canonical_sha256(
        {"role": TRUSTED_RECOVERY_AUTHORITY_ROLE, "authority": exact}
    )


def _exact_recovery_reference(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrustedParentRecoveryBindingError("recovery_reference_invalid")
    reference = dict(value)
    keys = set(reference)
    if not _REFERENCE_KEYS.issubset(keys) or keys - (_REFERENCE_KEYS | {"resume_state"}):
        raise TrustedParentRecoveryBindingError("recovery_reference_fields_invalid")
    generation = _require_positive_int(reference.get("generation"), "recovery_generation")
    checkpoint_id = _require_sha64(reference.get("checkpoint_id"), "checkpoint_id")
    if reference.get("object_key") != f"checkpoints/{checkpoint_id}":
        raise TrustedParentRecoveryBindingError("recovery_object_key_mismatch")
    _require_sha64(reference.get("manifest_sha256"), "checkpoint_manifest_sha256")
    _require_sha64(reference.get("pointer_sha256"), "recovery_pointer_sha256")
    if not _is_sha40(reference.get("source_sha")):
        raise TrustedParentRecoveryBindingError("source_git_sha_invalid")
    _require_sha64(reference.get("run_manifest_hash"), "run_manifest_sha256")
    _require_nonnegative_int(reference.get("optimizer_step"), "optimizer_step")
    _require_nonnegative_int(reference.get("tokens_seen"), "tokens_seen")
    if "resume_state" not in reference or not isinstance(reference["resume_state"], Mapping):
        raise TrustedParentRecoveryBindingError("recovery_resume_state_reference_missing")
    if generation <= 0:
        raise TrustedParentRecoveryBindingError("recovery_generation_invalid")
    return reference


def _manifest_previous_run_id(manifest: Any) -> str:
    if not isinstance(manifest, Mapping):
        raise TrustedParentRecoveryBindingError("recovery_manifest_invalid")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise TrustedParentRecoveryBindingError("recovery_manifest_identity_missing")
    training_config = identity.get("training_config")
    if not isinstance(training_config, Mapping):
        raise TrustedParentRecoveryBindingError("recovery_training_config_missing")
    return _require_nonempty_string(training_config.get("run_id"), "previous_run_id")


def trusted_parent_recovery_binding_from_resolution(
    resolution: RecoveryResolution,
    *,
    provider_class: str,
    provider_id: str,
    provider_session_id: str,
    previous_provider_session_id: str,
    terminal_recovery_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Project already-verified D05/D04 state into one content-addressed trust object.

    This projection is not itself authority. Operational consumption additionally requires
    an externally fixed expected binding SHA and a separately verified terminal authority.
    """
    if not isinstance(resolution, RecoveryResolution):
        raise TrustedParentRecoveryBindingError("recovery_resolution_type_invalid")
    reference = _exact_recovery_reference(resolution.reference)
    resume_state = resolution.resume_state
    if not isinstance(resume_state, Mapping):
        raise TrustedParentRecoveryBindingError("validated_d04_resume_state_missing")
    sidecar = dict(resume_state)
    if sidecar.get("schema") != SIDECAR_SCHEMA:
        raise TrustedParentRecoveryBindingError("validated_d04_resume_state_schema_invalid")

    checkpoint_id = _require_sha64(reference["checkpoint_id"], "checkpoint_id")
    manifest_sha = _require_sha64(
        reference["manifest_sha256"], "checkpoint_manifest_sha256"
    )
    source_sha = reference["source_sha"]
    run_manifest_sha = _require_sha64(
        reference["run_manifest_hash"], "run_manifest_sha256"
    )
    optimizer_step = _require_nonnegative_int(
        reference["optimizer_step"], "optimizer_step"
    )
    tokens_seen = _require_nonnegative_int(reference["tokens_seen"], "tokens_seen")

    exact_sidecar_checks = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_manifest_sha256": manifest_sha,
        "source_sha": source_sha,
        "run_manifest_hash": run_manifest_sha,
        "optimizer_step": optimizer_step,
        "tokens_seen": tokens_seen,
    }
    for name, expected in exact_sidecar_checks.items():
        if sidecar.get(name) != expected:
            raise TrustedParentRecoveryBindingError(f"d04_{name}_mismatch")

    if provider_class not in _COARSE_PROVIDERS:
        raise TrustedParentRecoveryBindingError("provider_class_invalid")
    concrete_provider = _require_nonempty_string(provider_id, "provider_id")
    if concrete_provider in _COARSE_PROVIDERS or concrete_provider == "UNBOUND":
        raise TrustedParentRecoveryBindingError("provider_id_not_concrete")
    current_session = _require_nonempty_string(provider_session_id, "provider_session_id")
    previous_session = _require_nonempty_string(
        previous_provider_session_id, "previous_provider_session_id"
    )
    if current_session == previous_session:
        raise TrustedParentRecoveryBindingError("provider_session_must_be_fresh_process")

    value: dict[str, Any] = {
        "schema": TRUSTED_PARENT_RECOVERY_BINDING_SCHEMA,
        "checkpoint_id": checkpoint_id,
        "checkpoint_manifest_sha256": manifest_sha,
        "recovery_pointer_sha256": _require_sha64(
            reference["pointer_sha256"], "recovery_pointer_sha256"
        ),
        "recovery_generation": _require_positive_int(
            reference["generation"], "recovery_generation"
        ),
        "recovery_object_key": reference["object_key"],
        "source_git_sha": source_sha,
        "run_manifest_sha256": run_manifest_sha,
        "previous_run_id": _manifest_previous_run_id(resolution.manifest),
        "optimizer_step": optimizer_step,
        "tokens_seen": tokens_seen,
        "d04_state_identity_sha256": _require_sha64(
            sidecar.get("state_identity_sha256"), "d04_state_identity_sha256"
        ),
        "ordered_next_exposure_identity_sha256": _require_sha64(
            sidecar.get("ordered_next_exposure_identity_sha256"),
            "ordered_next_exposure_identity_sha256",
        ),
        "ledger_identity_sha256": _require_sha64(
            sidecar.get("ledger_identity_sha256"), "ledger_identity_sha256"
        ),
        "materialization_identity_sha256": _require_sha64(
            sidecar.get("materialization_identity_sha256"),
            "materialization_identity_sha256",
        ),
        "packing_identity_sha256": _require_sha64(
            sidecar.get("packing_identity_sha256"), "packing_identity_sha256"
        ),
        "exposure_plan_identity_sha256": _require_sha64(
            sidecar.get("exposure_plan_identity_sha256"),
            "exposure_plan_identity_sha256",
        ),
        "provider_class": provider_class,
        "provider_id": concrete_provider,
        "provider_session_id": current_session,
        "previous_provider_session_id": previous_session,
        "terminal_recovery_authority_identity": _terminal_authority_identity(
            terminal_recovery_authority
        ),
    }
    if set(value) != _BINDING_KEYS_WITHOUT_HASH:
        raise AssertionError("trusted parent recovery binding schema drift")
    value["binding_sha256"] = canonical_sha256(value)
    return value


def _blocked(base: PortableRunBinding, *extra: str) -> PortableRunBinding:
    return PortableRunBinding(
        binding_ready=False,
        mode=base.mode,
        readiness_ready=base.readiness_ready,
        overlay_contract_valid=base.overlay_contract_valid,
        packet_contract_valid=base.packet_contract_valid,
        blockers=tuple(sorted(set(base.blockers + tuple(extra)))),
        readiness_sha256=base.readiness_sha256,
        overlay_sha256=base.overlay_sha256,
        packet_sha256=None,
        packet=None,
    )


def bind_trusted_same_provider_resume(
    readiness: Any,
    template: Any,
    overlay: Any,
    *,
    recovery_root: str | Path,
    expected_recovery_reference: Mapping[str, Any],
    provider_class: str,
    provider_id: str,
    provider_session_id: str,
    previous_provider_session_id: str,
    terminal_recovery_authority: Mapping[str, Any],
    expected_trusted_parent_binding_sha256: str,
    verified_trusted_recovery_authorities: Collection[str] = (),
    verified_scientific_authorities: Collection[str] = (),
    verified_authorization_refs: Collection[str] = (),
) -> PortableRunBinding:
    """Discharge exactly one same-provider blocker after canonical D05 verification."""
    base = bind_portable_run_packet(
        readiness,
        template,
        overlay,
        verified_scientific_authorities=verified_scientific_authorities,
        verified_authorization_refs=verified_authorization_refs,
    )
    if base.mode != "RESUME":
        return _blocked(base, "trusted:same_provider_resume_mode_required")
    expected_only_blocker = ("packet:trusted_parent_recovery_binding_missing",)
    if base.blockers != expected_only_blocker:
        return base
    if not _is_sha64(expected_trusted_parent_binding_sha256):
        return _blocked(base, "trusted:expected_binding_sha256_invalid")

    try:
        reference = _exact_recovery_reference(expected_recovery_reference)
        authority = _exact_authority(terminal_recovery_authority)
    except TrustedParentRecoveryBindingError as exc:
        return _blocked(base, f"trusted:{exc}")

    overlay_data = overlay if isinstance(overlay, dict) else {}
    checkpoint = overlay_data.get("checkpoint")
    checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
    lineage = checkpoint.get("lineage")
    lineage = lineage if isinstance(lineage, dict) else {}
    resource = overlay_data.get("resource")
    resource = resource if isinstance(resource, dict) else {}

    if checkpoint.get("parent_checkpoint_authority") != authority:
        return _blocked(base, "trusted:terminal_authority_not_overlay_bound")
    if authority.get("evidence_sha256") != expected_trusted_parent_binding_sha256:
        return _blocked(base, "trusted:terminal_authority_evidence_binding_mismatch")
    authority_token = trusted_recovery_authority_token(authority)
    if authority_token not in set(verified_trusted_recovery_authorities):
        return _blocked(base, "trusted:terminal_recovery_authority_not_verified")

    try:
        resolution = resolve_recovery_generation(
            recovery_root,
            expected_reference=reference,
            expected_source_sha=reference["source_sha"],
            expected_run_manifest_hash=reference["run_manifest_hash"],
            expected_step=reference["optimizer_step"],
            expected_tokens_seen=reference["tokens_seen"],
        )
        trusted = trusted_parent_recovery_binding_from_resolution(
            resolution,
            provider_class=provider_class,
            provider_id=provider_id,
            provider_session_id=provider_session_id,
            previous_provider_session_id=previous_provider_session_id,
            terminal_recovery_authority=authority,
        )
    except (RecoveryLifecycleError, TrustedParentRecoveryBindingError, OSError) as exc:
        return _blocked(base, f"trusted:recovery_resolution_failed:{type(exc).__name__}")

    if trusted["binding_sha256"] != expected_trusted_parent_binding_sha256:
        return _blocked(base, "trusted:binding_sha256_mismatch")

    exact_candidate_checks = {
        "parent_checkpoint_sha256": trusted["checkpoint_id"],
        "parent_manifest_sha256": trusted["checkpoint_manifest_sha256"],
        "previous_run_id": trusted["previous_run_id"],
        "source_provider": trusted["provider_class"],
        "cross_provider_transfer": False,
        "resume_validated": True,
    }
    for field, expected in exact_candidate_checks.items():
        if lineage.get(field) != expected:
            return _blocked(base, f"trusted:candidate_{field}_mismatch")
    if resource.get("provider") != trusted["provider_class"]:
        return _blocked(base, "trusted:target_provider_class_mismatch")

    if not isinstance(readiness, dict) or not isinstance(template, dict) or not isinstance(overlay, dict):
        return _blocked(base, "trusted:binding_inputs_not_objects")
    if base.readiness_sha256 is None or base.overlay_sha256 is None:
        return _blocked(base, "trusted:incumbent_binding_hash_missing")

    candidate = _build_candidate(
        readiness,
        template,
        overlay,
        readiness_sha256=base.readiness_sha256,
        overlay_sha256=base.overlay_sha256,
    )
    assessment = assess_portable_run_packet(candidate)
    if not assessment.contract_valid:
        return _blocked(base, "trusted:candidate_contract_invalid")
    if assessment.same_provider_resume_blockers != (
        "trusted_parent_recovery_binding_missing",
    ):
        return _blocked(base, "trusted:incumbent_same_provider_blockers_changed")
    identities = candidate.get("identities")
    identities = identities if isinstance(identities, dict) else {}
    if identities.get("source_git_sha") != trusted["source_git_sha"]:
        return _blocked(base, "trusted:candidate_source_git_sha_mismatch")

    binding_section = candidate.get("binding")
    if not isinstance(binding_section, dict):
        return _blocked(base, "trusted:candidate_binding_section_missing")
    binding_section["trusted_parent_recovery"] = trusted
    binding_section["trusted_parent_recovery_authority_token"] = authority_token

    return PortableRunBinding(
        binding_ready=True,
        mode="RESUME",
        readiness_ready=base.readiness_ready,
        overlay_contract_valid=base.overlay_contract_valid,
        packet_contract_valid=True,
        blockers=(),
        readiness_sha256=base.readiness_sha256,
        overlay_sha256=base.overlay_sha256,
        packet_sha256=canonical_sha256(candidate),
        packet=candidate,
    )


def _exact_trusted_parent_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrustedParentRecoveryBindingError("trusted_parent_binding_invalid")
    trusted = dict(value)
    if set(trusted) != _BINDING_KEYS:
        raise TrustedParentRecoveryBindingError("trusted_parent_binding_fields_invalid")
    claimed = _require_sha64(
        trusted.get("binding_sha256"),
        "trusted_parent_binding_sha256",
    )
    payload = dict(trusted)
    payload.pop("binding_sha256")
    if canonical_sha256(payload) != claimed:
        raise TrustedParentRecoveryBindingError("trusted_parent_binding_sha256_mismatch")
    return trusted


def restore_trusted_same_provider_resume(
    binding: PortableRunBinding,
    resolution: RecoveryResolution,
    *,
    model: Any,
    trainer: Any,
    terminal_recovery_authority: Mapping[str, Any],
    expected_trusted_parent_binding_sha256: str,
    verified_trusted_recovery_authorities: Collection[str],
    strict_model: bool = True,
    restore_rng: bool = True,
) -> LoadResult:
    """Restore exactly the D05 checkpoint authenticated by one READY resume binding.

    This is a thin operational consumer of the existing D05/D04 checkpoint stack.
    It does not resolve a second checkpoint path or define a new recovery format.
    Every exact checkpoint, manifest, progress and D04 identity is checked before
    the canonical loader mutates model, trainer or RNG state.
    """

    if not isinstance(binding, PortableRunBinding):
        raise TrustedParentRecoveryBindingError("portable_binding_type_invalid")
    if not binding.binding_ready or binding.mode != "RESUME" or binding.blockers:
        raise TrustedParentRecoveryBindingError("portable_binding_not_ready")
    if not isinstance(binding.packet, Mapping):
        raise TrustedParentRecoveryBindingError("portable_binding_packet_missing")
    packet = dict(binding.packet)
    if binding.packet_sha256 != canonical_sha256(packet):
        raise TrustedParentRecoveryBindingError("portable_binding_packet_sha256_mismatch")
    if not _is_sha64(expected_trusted_parent_binding_sha256):
        raise TrustedParentRecoveryBindingError("expected_binding_sha256_invalid")

    binding_section = packet.get("binding")
    if not isinstance(binding_section, Mapping):
        raise TrustedParentRecoveryBindingError("portable_binding_section_missing")
    trusted = _exact_trusted_parent_binding(
        binding_section.get("trusted_parent_recovery")
    )
    if trusted["binding_sha256"] != expected_trusted_parent_binding_sha256:
        raise TrustedParentRecoveryBindingError("trusted_parent_binding_expected_mismatch")

    authority = _exact_authority(terminal_recovery_authority)
    if authority["evidence_sha256"] != expected_trusted_parent_binding_sha256:
        raise TrustedParentRecoveryBindingError(
            "terminal_recovery_authority_evidence_binding_mismatch"
        )
    expected_authority_token = trusted_recovery_authority_token(authority)
    authority_token = binding_section.get("trusted_parent_recovery_authority_token")
    if authority_token != expected_authority_token:
        raise TrustedParentRecoveryBindingError(
            "trusted_parent_recovery_authority_token_mismatch"
        )
    if expected_authority_token not in set(verified_trusted_recovery_authorities):
        raise TrustedParentRecoveryBindingError(
            "trusted_parent_recovery_authority_not_verified"
        )

    if not isinstance(resolution, RecoveryResolution):
        raise TrustedParentRecoveryBindingError("recovery_resolution_type_invalid")
    reference = _exact_recovery_reference(resolution.reference)
    if resolution.content_path is None:
        raise TrustedParentRecoveryBindingError(
            "recovery_resolution_content_path_missing"
        )
    sidecar = resolution.resume_state
    if not isinstance(sidecar, Mapping):
        raise TrustedParentRecoveryBindingError("validated_d04_resume_state_missing")
    if sidecar.get("schema") != SIDECAR_SCHEMA:
        raise TrustedParentRecoveryBindingError("validated_d04_resume_state_schema_invalid")

    reprojection = trusted_parent_recovery_binding_from_resolution(
        resolution,
        provider_class=trusted["provider_class"],
        provider_id=trusted["provider_id"],
        provider_session_id=trusted["provider_session_id"],
        previous_provider_session_id=trusted["previous_provider_session_id"],
        terminal_recovery_authority=authority,
    )
    if reprojection != trusted:
        raise TrustedParentRecoveryBindingError(
            "trusted_parent_binding_resolution_mismatch"
        )

    return load_trainer_checkpoint(
        resolution.content_path,
        model=model,
        trainer=trainer,
        strict_model=strict_model,
        restore_rng=restore_rng,
        expected_checkpoint_id=trusted["checkpoint_id"],
        expected_manifest_sha256=trusted["checkpoint_manifest_sha256"],
        expected_git_sha=trusted["source_git_sha"],
        expected_run_manifest_hash=trusted["run_manifest_sha256"],
        expected_step=trusted["optimizer_step"],
        expected_tokens_seen=trusted["tokens_seen"],
        expected_ledger_identity_sha256=trusted["ledger_identity_sha256"],
        expected_materialization_identity_sha256=trusted[
            "materialization_identity_sha256"
        ],
        expected_packing_identity_sha256=trusted["packing_identity_sha256"],
        expected_exposure_plan_identity_sha256=trusted[
            "exposure_plan_identity_sha256"
        ],
        expected_ordered_next_exposure_identity_sha256=trusted[
            "ordered_next_exposure_identity_sha256"
        ],
    )
