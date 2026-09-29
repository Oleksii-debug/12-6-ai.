from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import twelve_six.trusted_parent_recovery_binding as trusted_module
from twelve_six.portable_run_binding import PortableRunBinding
from twelve_six.portable_run_packet import PortableRunAssessment
from twelve_six.scale141_recovery import RecoveryResolution
from twelve_six.scale141_resume_sidecar import SIDECAR_SCHEMA
from twelve_six.trusted_parent_recovery_binding import (
    bind_trusted_same_provider_resume,
    restore_trusted_same_provider_resume,
    trusted_parent_recovery_binding_from_resolution,
    trusted_recovery_authority_token,
)

SHA40 = "a" * 40
CHECKPOINT = "b" * 64
MANIFEST = "c" * 64
POINTER = "d" * 64
RUN_MANIFEST = "e" * 64
STATE = "1" * 64
ORDERED = "2" * 64
LEDGER = "3" * 64
MATERIALIZATION = "4" * 64
PACKING = "5" * 64
PLAN = "6" * 64
PREVIOUS_RUN = "R01-GITHUB-SESSION-001"
PROVIDER_CLASS = "OTHER_FREE"
PROVIDER_ID = "GITHUB_ACTIONS_STANDARD_LINUX_X64"
CURRENT_SESSION = "gha-run-200-job-2"
PREVIOUS_SESSION = "gha-run-199-job-1"


def _authority(evidence_sha256: str) -> dict:
    return {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": SHA40,
        "evidence_sha256": evidence_sha256,
        "workflow_run_id": 987654,
        "workflow_conclusion": "success",
        "terminal": True,
    }


def _resolution() -> RecoveryResolution:
    reference = {
        "generation": 2,
        "object_key": f"checkpoints/{CHECKPOINT}",
        "checkpoint_id": CHECKPOINT,
        "manifest_sha256": MANIFEST,
        "pointer_sha256": POINTER,
        "source_sha": SHA40,
        "run_manifest_hash": RUN_MANIFEST,
        "optimizer_step": 17,
        "tokens_seen": 4096,
        "resume_state": {"file_sha256": "7" * 64},
    }
    manifest = {
        "identity": {
            "training_config": {"run_id": PREVIOUS_RUN},
        }
    }
    resume_state = {
        "schema": SIDECAR_SCHEMA,
        "checkpoint_id": CHECKPOINT,
        "checkpoint_manifest_sha256": MANIFEST,
        "source_sha": SHA40,
        "run_manifest_hash": RUN_MANIFEST,
        "optimizer_step": 17,
        "tokens_seen": 4096,
        "ledger_identity_sha256": LEDGER,
        "materialization_identity_sha256": MATERIALIZATION,
        "packing_identity_sha256": PACKING,
        "exposure_plan_identity_sha256": PLAN,
        "ordered_next_exposure_identity_sha256": ORDERED,
        "state_identity_sha256": STATE,
    }
    return RecoveryResolution(
        path=Path("/verified/generation-00000002"),
        content_path=Path(f"/verified/checkpoints/{CHECKPOINT}"),
        reference=reference,
        manifest=manifest,
        resume_state=resume_state,
    )


def _binding_material(resolution: RecoveryResolution) -> tuple[dict, str, str]:
    placeholder = _authority("0" * 64)
    projected = trusted_parent_recovery_binding_from_resolution(
        resolution,
        provider_class=PROVIDER_CLASS,
        provider_id=PROVIDER_ID,
        provider_session_id=CURRENT_SESSION,
        previous_provider_session_id=PREVIOUS_SESSION,
        terminal_recovery_authority=placeholder,
    )
    expected = projected["binding_sha256"]
    authority = _authority(expected)
    final_projection = trusted_parent_recovery_binding_from_resolution(
        resolution,
        provider_class=PROVIDER_CLASS,
        provider_id=PROVIDER_ID,
        provider_session_id=CURRENT_SESSION,
        previous_provider_session_id=PREVIOUS_SESSION,
        terminal_recovery_authority=authority,
    )
    assert final_projection["binding_sha256"] == expected
    return authority, expected, trusted_recovery_authority_token(authority)


def _overlay(authority: dict) -> dict:
    return {
        "checkpoint": {
            "mode": "RESUME",
            "lineage": {
                "parent_checkpoint_sha256": CHECKPOINT,
                "parent_manifest_sha256": MANIFEST,
                "previous_run_id": PREVIOUS_RUN,
                "source_provider": PROVIDER_CLASS,
                "cross_provider_transfer": False,
                "resume_validated": True,
            },
            "parent_checkpoint_authority": authority,
        },
        "resource": {"provider": PROVIDER_CLASS},
    }


def _patch_incumbent(
    monkeypatch: pytest.MonkeyPatch,
    resolution: RecoveryResolution,
) -> None:
    base = PortableRunBinding(
        binding_ready=False,
        mode="RESUME",
        readiness_ready=True,
        overlay_contract_valid=True,
        packet_contract_valid=True,
        blockers=("packet:trusted_parent_recovery_binding_missing",),
        readiness_sha256="8" * 64,
        overlay_sha256="9" * 64,
        packet_sha256=None,
        packet=None,
    )
    monkeypatch.setattr(
        trusted_module,
        "bind_portable_run_packet",
        lambda *args, **kwargs: base,
    )

    def fake_resolver(root: str | Path, **kwargs: object) -> RecoveryResolution:
        assert Path(root) == Path("/verified")
        assert kwargs["expected_reference"] == resolution.reference
        assert kwargs["expected_source_sha"] == resolution.reference["source_sha"]
        assert kwargs["expected_run_manifest_hash"] == resolution.reference["run_manifest_hash"]
        assert kwargs["expected_step"] == resolution.reference["optimizer_step"]
        assert kwargs["expected_tokens_seen"] == resolution.reference["tokens_seen"]
        return resolution

    monkeypatch.setattr(trusted_module, "resolve_recovery_generation", fake_resolver)
    monkeypatch.setattr(
        trusted_module,
        "_build_candidate",
        lambda *args, **kwargs: {
            "identities": {"source_git_sha": SHA40},
            "binding": {},
        },
    )
    monkeypatch.setattr(
        trusted_module,
        "assess_portable_run_packet",
        lambda candidate: PortableRunAssessment(
            contract_valid=True,
            ready_for_initial_local_free_launch=False,
            ready_for_same_provider_fresh_process_resume=False,
            ready_for_cross_provider_resume=False,
            contract_errors=(),
            launch_blockers=(),
            same_provider_resume_blockers=("trusted_parent_recovery_binding_missing",),
            resume_blockers=(),
        ),
    )


def _bind(
    monkeypatch: pytest.MonkeyPatch,
    *,
    resolution: RecoveryResolution | None = None,
    overlay: dict | None = None,
    provider_class: str = PROVIDER_CLASS,
    provider_id: str = PROVIDER_ID,
    provider_session_id: str = CURRENT_SESSION,
    previous_provider_session_id: str = PREVIOUS_SESSION,
    verified: bool = True,
) -> PortableRunBinding:
    resolution = resolution or _resolution()
    authority, expected, token = _binding_material(_resolution())
    _patch_incumbent(monkeypatch, resolution)
    selected_overlay = overlay or _overlay(authority)
    return bind_trusted_same_provider_resume(
        {},
        {},
        selected_overlay,
        recovery_root="/verified",
        expected_recovery_reference=resolution.reference,
        provider_class=provider_class,
        provider_id=provider_id,
        provider_session_id=provider_session_id,
        previous_provider_session_id=previous_provider_session_id,
        terminal_recovery_authority=authority,
        expected_trusted_parent_binding_sha256=expected,
        verified_trusted_recovery_authorities=(token,) if verified else (),
    )


def test_trusted_same_provider_resume_clears_only_incumbent_missing_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolution = _resolution()
    authority, expected, token = _binding_material(resolution)
    _patch_incumbent(monkeypatch, resolution)

    result = bind_trusted_same_provider_resume(
        {},
        {},
        _overlay(authority),
        recovery_root="/verified",
        expected_recovery_reference=resolution.reference,
        provider_class=PROVIDER_CLASS,
        provider_id=PROVIDER_ID,
        provider_session_id=CURRENT_SESSION,
        previous_provider_session_id=PREVIOUS_SESSION,
        terminal_recovery_authority=authority,
        expected_trusted_parent_binding_sha256=expected,
        verified_trusted_recovery_authorities=(token,),
    )

    assert result.binding_ready
    assert result.mode == "RESUME"
    assert result.blockers == ()
    assert result.packet is not None
    trusted = result.packet["binding"]["trusted_parent_recovery"]
    assert trusted["binding_sha256"] == expected
    assert trusted["checkpoint_id"] == CHECKPOINT
    assert trusted["checkpoint_manifest_sha256"] == MANIFEST
    assert trusted["previous_run_id"] == PREVIOUS_RUN
    assert trusted["provider_id"] == PROVIDER_ID
    assert trusted["provider_session_id"] == CURRENT_SESSION
    assert trusted["previous_provider_session_id"] == PREVIOUS_SESSION
    assert result.packet["binding"]["trusted_parent_recovery_authority_token"] == token


def test_fixed_trusted_root_rejects_well_formed_candidate_lineage_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolution = _resolution()
    authority, expected, token = _binding_material(resolution)
    mutations = (
        ("parent_checkpoint_sha256", "f" * 64, "trusted:candidate_parent_checkpoint_sha256_mismatch"),
        ("parent_manifest_sha256", "f" * 64, "trusted:candidate_parent_manifest_sha256_mismatch"),
        ("previous_run_id", "R01-RESEALED-999", "trusted:candidate_previous_run_id_mismatch"),
        ("resume_validated", False, "trusted:candidate_resume_validated_mismatch"),
        ("cross_provider_transfer", True, "trusted:candidate_cross_provider_transfer_mismatch"),
    )
    for field, replacement, blocker in mutations:
        _patch_incumbent(monkeypatch, resolution)
        overlay = _overlay(authority)
        overlay["checkpoint"]["lineage"][field] = replacement
        result = bind_trusted_same_provider_resume(
            {},
            {},
            overlay,
            recovery_root="/verified",
            expected_recovery_reference=resolution.reference,
            provider_class=PROVIDER_CLASS,
            provider_id=PROVIDER_ID,
            provider_session_id=CURRENT_SESSION,
            previous_provider_session_id=PREVIOUS_SESSION,
            terminal_recovery_authority=authority,
            expected_trusted_parent_binding_sha256=expected,
            verified_trusted_recovery_authorities=(token,),
        )
        assert not result.binding_ready
        assert result.packet is None
        assert blocker in result.blockers


def test_fixed_root_rejects_provider_and_session_reseal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for kwargs in (
        {"provider_id": "GITHUB_ACTIONS_DIFFERENT_POOL"},
        {"provider_session_id": "gha-run-201-job-9"},
        {"previous_provider_session_id": "gha-run-198-job-4"},
        {"provider_class": "OWNER_LAPTOP"},
    ):
        result = _bind(monkeypatch, **kwargs)
        assert not result.binding_ready
        assert result.packet is None
        assert "trusted:binding_sha256_mismatch" in result.blockers


def test_generic_or_unverified_terminal_authority_cannot_mint_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _bind(monkeypatch, verified=False)
    assert not result.binding_ready
    assert "trusted:terminal_recovery_authority_not_verified" in result.blockers

    resolution = _resolution()
    authority, expected, token = _binding_material(resolution)
    generic = _authority("f" * 64)
    _patch_incumbent(monkeypatch, resolution)
    overlay = _overlay(generic)
    result = bind_trusted_same_provider_resume(
        {},
        {},
        overlay,
        recovery_root="/verified",
        expected_recovery_reference=resolution.reference,
        provider_class=PROVIDER_CLASS,
        provider_id=PROVIDER_ID,
        provider_session_id=CURRENT_SESSION,
        previous_provider_session_id=PREVIOUS_SESSION,
        terminal_recovery_authority=generic,
        expected_trusted_parent_binding_sha256=expected,
        verified_trusted_recovery_authorities=(token,),
    )
    assert not result.binding_ready
    assert "trusted:terminal_authority_evidence_binding_mismatch" in result.blockers

    _patch_incumbent(monkeypatch, resolution)
    mismatched_overlay = _overlay(authority)
    mismatched_overlay["checkpoint"]["parent_checkpoint_authority"] = generic
    result = bind_trusted_same_provider_resume(
        {},
        {},
        mismatched_overlay,
        recovery_root="/verified",
        expected_recovery_reference=resolution.reference,
        provider_class=PROVIDER_CLASS,
        provider_id=PROVIDER_ID,
        provider_session_id=CURRENT_SESSION,
        previous_provider_session_id=PREVIOUS_SESSION,
        terminal_recovery_authority=authority,
        expected_trusted_parent_binding_sha256=expected,
        verified_trusted_recovery_authorities=(token,),
    )
    assert not result.binding_ready
    assert "trusted:terminal_authority_not_overlay_bound" in result.blockers


def test_d04_and_counter_type_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_step = _resolution()
    bad_step.reference["optimizer_step"] = True
    _patch_incumbent(monkeypatch, bad_step)
    authority, expected, token = _binding_material(_resolution())
    result = bind_trusted_same_provider_resume(
        {},
        {},
        _overlay(authority),
        recovery_root="/verified",
        expected_recovery_reference=bad_step.reference,
        provider_class=PROVIDER_CLASS,
        provider_id=PROVIDER_ID,
        provider_session_id=CURRENT_SESSION,
        previous_provider_session_id=PREVIOUS_SESSION,
        terminal_recovery_authority=authority,
        expected_trusted_parent_binding_sha256=expected,
        verified_trusted_recovery_authorities=(token,),
    )
    assert not result.binding_ready
    assert any("optimizer_step_invalid" in item for item in result.blockers)

    missing_state = _resolution()
    assert missing_state.resume_state is not None
    missing_state.resume_state.pop("state_identity_sha256")
    _patch_incumbent(monkeypatch, missing_state)
    result = bind_trusted_same_provider_resume(
        {},
        {},
        _overlay(authority),
        recovery_root="/verified",
        expected_recovery_reference=missing_state.reference,
        provider_class=PROVIDER_CLASS,
        provider_id=PROVIDER_ID,
        provider_session_id=CURRENT_SESSION,
        previous_provider_session_id=PREVIOUS_SESSION,
        terminal_recovery_authority=authority,
        expected_trusted_parent_binding_sha256=expected,
        verified_trusted_recovery_authorities=(token,),
    )
    assert not result.binding_ready
    assert any("TrustedParentRecoveryBindingError" in item for item in result.blockers)


def test_projection_rejects_coarse_provider_as_concrete_identity() -> None:
    resolution = _resolution()
    with pytest.raises(ValueError, match="provider_id_not_concrete"):
        trusted_parent_recovery_binding_from_resolution(
            resolution,
            provider_class="OTHER_FREE",
            provider_id="OTHER_FREE",
            provider_session_id=CURRENT_SESSION,
            previous_provider_session_id=PREVIOUS_SESSION,
            terminal_recovery_authority=_authority("0" * 64),
        )


def test_projection_requires_fresh_process_session_identity() -> None:
    resolution = _resolution()
    with pytest.raises(ValueError, match="provider_session_must_be_fresh_process"):
        trusted_parent_recovery_binding_from_resolution(
            resolution,
            provider_class=PROVIDER_CLASS,
            provider_id=PROVIDER_ID,
            provider_session_id=PREVIOUS_SESSION,
            previous_provider_session_id=PREVIOUS_SESSION,
            terminal_recovery_authority=_authority("0" * 64),
        )


def test_operational_restore_rejects_unrelated_verified_terminal_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolution = _resolution()
    authority, expected, _ = _binding_material(resolution)
    trusted = trusted_parent_recovery_binding_from_resolution(
        resolution,
        provider_class=PROVIDER_CLASS,
        provider_id=PROVIDER_ID,
        provider_session_id=CURRENT_SESSION,
        previous_provider_session_id=PREVIOUS_SESSION,
        terminal_recovery_authority=authority,
    )

    unrelated_authority = dict(authority)
    unrelated_authority["workflow_run_id"] = authority["workflow_run_id"] + 1
    unrelated_token = trusted_recovery_authority_token(unrelated_authority)
    packet = {
        "binding": {
            "trusted_parent_recovery": trusted,
            "trusted_parent_recovery_authority_token": unrelated_token,
        }
    }
    binding = PortableRunBinding(
        binding_ready=True,
        mode="RESUME",
        readiness_ready=True,
        overlay_contract_valid=True,
        packet_contract_valid=True,
        blockers=(),
        readiness_sha256="8" * 64,
        overlay_sha256="9" * 64,
        packet_sha256=trusted_module.canonical_sha256(packet),
        packet=packet,
    )

    restore_called = False

    def fail_if_restored(*args: object, **kwargs: object) -> None:
        nonlocal restore_called
        restore_called = True
        raise AssertionError("checkpoint restore must not run")

    monkeypatch.setattr(trusted_module, "load_trainer_checkpoint", fail_if_restored)

    with pytest.raises(ValueError, match="trusted_parent_binding_resolution_mismatch"):
        restore_trusted_same_provider_resume(
            binding,
            resolution,
            model=object(),
            trainer=object(),
            terminal_recovery_authority=unrelated_authority,
            expected_trusted_parent_binding_sha256=expected,
            verified_trusted_recovery_authorities=(unrelated_token,),
        )

    assert not restore_called


@pytest.mark.parametrize(
    "field",
    ["readiness_ready", "overlay_contract_valid", "packet_contract_valid"],
)
def test_operational_restore_requires_all_incumbent_ready_gates(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    resolution = _resolution()
    authority, expected, token = _binding_material(resolution)
    trusted = trusted_parent_recovery_binding_from_resolution(
        resolution,
        provider_class=PROVIDER_CLASS,
        provider_id=PROVIDER_ID,
        provider_session_id=CURRENT_SESSION,
        previous_provider_session_id=PREVIOUS_SESSION,
        terminal_recovery_authority=authority,
    )
    packet = {
        "binding": {
            "trusted_parent_recovery": trusted,
            "trusted_parent_recovery_authority_token": token,
        }
    }
    binding = PortableRunBinding(
        binding_ready=True,
        mode="RESUME",
        readiness_ready=True,
        overlay_contract_valid=True,
        packet_contract_valid=True,
        blockers=(),
        readiness_sha256="8" * 64,
        overlay_sha256="9" * 64,
        packet_sha256=trusted_module.canonical_sha256(packet),
        packet=packet,
    )
    binding = replace(binding, **{field: False})

    restore_called = False

    def fail_if_restored(*args: object, **kwargs: object) -> None:
        nonlocal restore_called
        restore_called = True
        raise AssertionError("checkpoint restore must not run")

    monkeypatch.setattr(trusted_module, "load_trainer_checkpoint", fail_if_restored)

    with pytest.raises(ValueError, match="portable_binding_not_ready"):
        restore_trusted_same_provider_resume(
            binding,
            resolution,
            model=object(),
            trainer=object(),
            terminal_recovery_authority=authority,
            expected_trusted_parent_binding_sha256=expected,
            verified_trusted_recovery_authorities=(token,),
        )

    assert not restore_called
