from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import twelve_six.learned20m_current_run_authority as current_run
from twelve_six.learned20m_current_run_authority import (
    CURRENT_RUN_POINTER_REF,
    CurrentRunAuthorityInspection,
    activate_current_run_authority,
    build_current_run_identity,
    build_current_run_pointer_state,
    decode_current_run_pointer_state,
    inspect_current_run_authority,
    refresh_current_run_authority,
    retire_current_run_authority,
    validate_current_run_identity,
    validate_current_run_pointer_state,
    verify_candidate_against_current_run,
    verify_terminal_authority_current_run_binding,
)
from twelve_six.learned20m_global_training_lease import (
    GlobalLeaseInspection,
    acquire_global_training_run_lease,
    build_global_lease_state,
    global_training_run_lease_ref,
    renew_global_training_run_lease,
)
from twelve_six.learned20m_training_lease import (
    TERMINAL_AUTHORITY_SCHEMA,
    base_launch_manifest_sha256,
    build_authorized_training_run_lease,
    build_training_run_lease,
    canonical_json_bytes,
    finalize_launch_manifest,
    launch_manifest_sha256,
    terminal_authority_sha256,
)

NOW = datetime(2026, 9, 27, 13, 0, tzinfo=UTC)


def _manifest(*, source_git_sha: str = "b" * 40, binding: str = "a" * 64) -> dict:
    hashes = {
        "modelspec_sha256": "1" * 64,
        "initspec_sha256": "2" * 64,
        "tokenizer_sha256": "3" * 64,
        "corpus_manifest_sha256": "4" * 64,
        "split_sha256": "5" * 64,
        "packing_sha256": "6" * 64,
        "unique_loss_ledger_sha256": "7" * 64,
        "training_config_sha256": "8" * 64,
        "portable_run_packet_sha256": "9" * 64,
        "portable_run_binding_sha256": binding,
    }
    return {
        "schema_version": 1,
        "manifest_id": "R01-LEARNED20M-LAUNCH-MANIFEST-V1",
        "stage": "LEARNED_20M",
        "identities": {"source_git_sha": source_git_sha, **hashes},
        "recipe": {
            "optimizer_scheduler_precision": "AdamW|cosine|fp32",
            "seed": 1337,
            "target_unique_loss_positions": 20_000_000,
            "maximum_total_exposures": 24_000_000,
        },
        "checkpoint": {
            "lineage": "learned20m-v1",
            "checkpoint_contract_sha256": "c" * 64,
        },
        "evaluation": {
            "firewall_sha256": "d" * 64,
            "final_test_payload_access": False,
        },
        "resource": {
            "resource_class": "LOCAL_FREE",
            "maximum_cost_usd": 0,
            "materially_paid": False,
        },
        "authorities": {
            "training": {
                "reference": "github:issue/653#comment-training",
                "evidence_sha256": "e" * 64,
            },
            "compute": {
                "reference": "github:issue/653#comment-compute",
                "evidence_sha256": "f" * 64,
            },
        },
        "execution_backend": "PROJECT_NATIVE_PYTORCH",
    }


def _identity(
    *,
    manifest: dict | None = None,
    run_id: str = "run-a",
    recovery_manifest: str = "1" * 64,
    binding: str = "a" * 64,
    source_git_sha: str = "b" * 40,
) -> dict:
    if manifest is None:
        manifest = _manifest(source_git_sha=source_git_sha, binding=binding)
    return build_current_run_identity(
        run_id=run_id,
        base_launch_manifest_sha256=current_run._base_manifest_digest(manifest),
        recovery_run_manifest_sha256=recovery_manifest,
        recovery_attempt_authority_sha256="2" * 64,
        portable_run_binding_sha256=binding,
        source_git_sha=source_git_sha,
    )


def _terminal_authority(
    manifest: dict,
    identity: dict,
    *,
    exposure: int = 1_000,
) -> dict:
    identities = manifest["identities"]
    authority = {
        "schema": TERMINAL_AUTHORITY_SCHEMA,
        "authority_identity_sha256": "0" * 64,
        "base_manifest_sha256": base_launch_manifest_sha256(manifest),
        "source_git_sha": identities["source_git_sha"],
        "carrier_authority_sha256": "0" * 64,
        "modelspec_sha256": identities["modelspec_sha256"],
        "initspec_sha256": identities["initspec_sha256"],
        "random_init": True,
        "foreign_pretrained_weights_used": False,
        "launch_input_authority_sha256": "1" * 64,
        "corpus_manifest_sha256": identities["corpus_manifest_sha256"],
        "split_sha256": identities["split_sha256"],
        "tokenizer_decision_sha256": "2" * 64,
        "tokenizer_sha256": identities["tokenizer_sha256"],
        "packing_sha256": identities["packing_sha256"],
        "loss_bearing_manifest_sha256": "3" * 64,
        "unique_loss_ledger_sha256": identities["unique_loss_ledger_sha256"],
        "exposure_plan_sha256": "4" * 64,
        "portable_run_packet_sha256": identities["portable_run_packet_sha256"],
        "portable_run_binding_sha256": identities["portable_run_binding_sha256"],
        "recipe_authority_sha256": "5" * 64,
        "training_config_sha256": identities["training_config_sha256"],
        "seed_vector_sha256": "6" * 64,
        "target_unique_loss_positions": manifest["recipe"]["target_unique_loss_positions"],
        "maximum_total_exposures": manifest["recipe"]["maximum_total_exposures"],
        "replay_cap": 4_000_000,
        "recovery_run_id": identity["run_id"],
        "recovery_run_manifest_sha256": identity["recovery_run_manifest_sha256"],
        "recovery_attempt_authority_sha256": identity[
            "recovery_attempt_authority_sha256"
        ],
        "checkpoint_contract_sha256": manifest["checkpoint"][
            "checkpoint_contract_sha256"
        ],
        "checkpoint_cadence_sha256": "9" * 64,
        "resume_rules_sha256": "a" * 64,
        "safe_stop_current_run_sha256": identity["identity_sha256"],
        "evaluation_schedule_sha256": "c" * 64,
        "evaluation_firewall_sha256": manifest["evaluation"]["firewall_sha256"],
        "poison_stop_semantics_sha256": "d" * 64,
        "resource_evidence_sha256": "e" * 64,
        "execution_target_sha256": "f" * 64,
        "measured_resource_envelope_sha256": "0" * 64,
        "resource_class": manifest["resource"]["resource_class"],
        "maximum_cost_usd": 0,
        "materially_paid": False,
        "final_test_payload_access": False,
        "training_authority_ref": manifest["authorities"]["training"]["reference"],
        "training_authority_sha256": manifest["authorities"]["training"][
            "evidence_sha256"
        ],
        "compute_authority_ref": manifest["authorities"]["compute"]["reference"],
        "compute_authority_sha256": manifest["authorities"]["compute"][
            "evidence_sha256"
        ],
        "execution_backend": manifest["execution_backend"],
        "authorized_optimized_target_exposure": exposure,
    }
    authority["authority_identity_sha256"] = terminal_authority_sha256(authority)
    return authority


def _authorized_run(
    base_manifest: dict | None = None,
    *,
    run_id: str = "run-a",
    holder_id: str = "runner-a",
    recovery_manifest: str = "1" * 64,
    ttl_seconds: int = 3600,
    now: datetime = NOW,
) -> tuple[dict, dict, object, str]:
    base = deepcopy(_manifest() if base_manifest is None else base_manifest)
    identity = _identity(
        manifest=base,
        run_id=run_id,
        recovery_manifest=recovery_manifest,
        binding=base["identities"]["portable_run_binding_sha256"],
        source_git_sha=base["identities"]["source_git_sha"],
    )
    authority = _terminal_authority(base, identity)
    manifest = finalize_launch_manifest(base, authority)
    expected_authority = authority["authority_identity_sha256"]
    lease = build_authorized_training_run_lease(
        manifest,
        expected_terminal_authority_sha256=expected_authority,
        run_id=run_id,
        holder_id=holder_id,
        ttl_seconds=ttl_seconds,
        now=now,
    )
    return manifest, identity, lease, expected_authority


def _global_inspection(
    manifest: dict,
    *,
    run_id: str = "run-a",
    remote_tip: str = "c" * 40,
) -> GlobalLeaseInspection:
    return GlobalLeaseInspection(
        present=True,
        valid=True,
        ref=global_training_run_lease_ref(manifest),
        remote_tip=remote_tip,
        launch_manifest_sha256=launch_manifest_sha256(manifest),
        run_id=run_id,
        lease_status="RUNNING",
        renewal_sequence=0,
        blockers=(),
    )


def _global_state_sha256(
    manifest: dict,
    *,
    run_id: str = "run-a",
    acquired_at: datetime = NOW,
) -> str:
    lease = build_training_run_lease(
        manifest,
        run_id=run_id,
        holder_id="runner-a",
        ttl_seconds=3600,
        now=acquired_at,
    )
    state = build_global_lease_state(manifest, lease.as_dict())
    return hashlib.sha256(canonical_json_bytes(state)).hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def git_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    if shutil.which("git") is None:
        pytest.skip("git executable unavailable")
    remote = tmp_path / "remote.git"
    writer_a = tmp_path / "writer-a"
    writer_b = tmp_path / "writer-b"
    _git("init", "--bare", str(remote))
    _git("init", str(writer_a))
    _git("init", str(writer_b))
    return remote, writer_a, writer_b


def test_current_run_identity_is_closed_world_and_self_hashed() -> None:
    identity = _identity()
    assert validate_current_run_identity(identity) == ()
    assert identity["identity_sha256"] == current_run._identity_digest(identity)

    substituted = deepcopy(identity)
    substituted["recovery_run_manifest_sha256"] = "3" * 64
    assert "current_run_identity_self_hash_mismatch" in validate_current_run_identity(
        substituted
    )

    widened = deepcopy(identity)
    widened["training_authorized"] = True
    assert "current_run_identity_fields_mismatch" in validate_current_run_identity(
        widened
    )


def test_terminal_authority_crossbind_matches_safe_stop_identity_without_cycle() -> None:
    identity = _identity()
    terminal = {
        "base_manifest_sha256": identity["base_launch_manifest_sha256"],
        "recovery_run_id": identity["run_id"],
        "recovery_run_manifest_sha256": identity["recovery_run_manifest_sha256"],
        "recovery_attempt_authority_sha256": identity[
            "recovery_attempt_authority_sha256"
        ],
        "safe_stop_current_run_sha256": identity["identity_sha256"],
        "source_git_sha": identity["source_git_sha"],
        "portable_run_binding_sha256": identity["portable_run_binding_sha256"],
    }
    assert verify_terminal_authority_current_run_binding(identity, terminal) == ()

    for field in tuple(terminal):
        substituted = deepcopy(terminal)
        if field == "recovery_run_id":
            substituted[field] = "run-b"
        elif field == "source_git_sha":
            substituted[field] = "c" * 40
        else:
            substituted[field] = "f" * 64
        blockers = verify_terminal_authority_current_run_binding(
            identity,
            substituted,
        )
        assert blockers == (f"terminal_current_run_binding_mismatch:{field}",)


def test_pointer_state_binds_incumbent_manifest_global_lease_and_run() -> None:
    manifest = _manifest()
    identity = _identity()
    global_lease = _global_inspection(manifest)
    state = build_current_run_pointer_state(
        manifest, global_lease, identity, generation=1,
        global_lease_state_sha256=_global_state_sha256(manifest),
        global_lease_expires_at_utc="2026-09-27T14:00:00Z",
    )

    assert validate_current_run_pointer_state(state) == ()
    assert state["launch_manifest_sha256"] == launch_manifest_sha256(manifest)
    assert state["global_lease_ref"] == global_training_run_lease_ref(manifest)
    assert state["current_run_identity"]["run_id"] == "run-a"

    wrong_run = _global_inspection(manifest, run_id="run-b")
    with pytest.raises(ValueError, match="global_lease_run_id_mismatch"):
        build_current_run_pointer_state(
            manifest,
            wrong_run,
            identity,
            generation=1,
            global_lease_state_sha256=_global_state_sha256(manifest),
            global_lease_expires_at_utc="2026-09-27T14:00:00Z",
        )

    wrong_ref = deepcopy(global_lease)
    object.__setattr__(wrong_ref, "ref", "refs/heads/attacker")
    with pytest.raises(ValueError, match="global_lease_ref_mismatch"):
        build_current_run_pointer_state(
            manifest,
            wrong_ref,
            identity,
            generation=1,
            global_lease_state_sha256=_global_state_sha256(manifest),
            global_lease_expires_at_utc="2026-09-27T14:00:00Z",
        )


def test_base_manifest_substitution_fails_under_fixed_run_identity() -> None:
    manifest = _manifest()
    identity = _identity(manifest=manifest)
    substituted = deepcopy(manifest)
    substituted["recipe"]["seed"] += 1

    with pytest.raises(ValueError, match="current_run_base_manifest_mismatch"):
        build_current_run_pointer_state(
            substituted,
            _global_inspection(substituted),
            identity,
            generation=1,
            global_lease_state_sha256=_global_state_sha256(substituted),
            global_lease_expires_at_utc="2026-09-27T14:00:00Z",
        )


def test_pointer_decoder_rejects_noncanonical_and_unknown_fields() -> None:
    manifest = _manifest()
    state = build_current_run_pointer_state(
        manifest, _global_inspection(manifest), _identity(), generation=1,
        global_lease_state_sha256=_global_state_sha256(manifest),
        global_lease_expires_at_utc="2026-09-27T14:00:00Z",
    )
    assert decode_current_run_pointer_state(canonical_json_bytes(state)) == state

    pretty = json.dumps(state, indent=2, sort_keys=True).encode()
    with pytest.raises(ValueError, match="current_run_pointer_not_canonical"):
        decode_current_run_pointer_state(pretty)

    widened = deepcopy(state)
    widened["optimizer_start_permitted"] = True
    widened["pointer_identity_sha256"] = current_run._pointer_digest(widened)
    with pytest.raises(ValueError, match="current_run_pointer_fields_mismatch"):
        decode_current_run_pointer_state(canonical_json_bytes(widened))


def test_candidate_b_cannot_self_select_namespace_when_a_is_current() -> None:
    manifest_a = _manifest()
    identity_a = _identity()
    state_a = build_current_run_pointer_state(
        manifest_a, _global_inspection(manifest_a), identity_a, generation=1,
        global_lease_state_sha256=_global_state_sha256(manifest_a),
        global_lease_expires_at_utc="2026-09-27T14:00:00Z",
    )
    inspection = CurrentRunAuthorityInspection(
        present=True,
        valid=True,
        active=True,
        ref=CURRENT_RUN_POINTER_REF,
        remote_tip="d" * 40,
        generation=1,
        pointer_identity_sha256=state_a["pointer_identity_sha256"],
        launch_manifest_sha256=state_a["launch_manifest_sha256"],
        global_lease_ref=state_a["global_lease_ref"],
        global_lease_remote_tip=state_a["global_lease_remote_tip"],
        global_lease_state_sha256=state_a["global_lease_state_sha256"],
        global_lease_expires_at_utc=state_a["global_lease_expires_at_utc"],
        run_id="run-a",
        recovery_run_manifest_sha256="1" * 64,
        current_run_identity_sha256=identity_a["identity_sha256"],
        blockers=(),
    )

    assert (
        verify_candidate_against_current_run(
            inspection,
            launch_manifest_sha256_value=launch_manifest_sha256(manifest_a),
            run_id="run-a",
            recovery_run_manifest_sha256="1" * 64,
            current_run_identity_sha256=identity_a["identity_sha256"],
        )
        == ()
    )

    manifest_b = _manifest(binding="f" * 64)
    identity_b = _identity(
        run_id="run-b", recovery_manifest="4" * 64, binding="f" * 64
    )
    blockers = verify_candidate_against_current_run(
        inspection,
        launch_manifest_sha256_value=launch_manifest_sha256(manifest_b),
        run_id="run-b",
        recovery_run_manifest_sha256="4" * 64,
        current_run_identity_sha256=identity_b["identity_sha256"],
    )
    assert set(blockers) == {
        "current_run_launch_manifest_mismatch",
        "current_run_id_mismatch",
        "current_run_recovery_manifest_mismatch",
        "current_run_identity_mismatch",
    }


def test_fixed_pointer_activation_retirement_and_generation(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest_a, identity_a, lease_a, expected_authority_a = _authorized_run()
    global_acquire = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest_a,
        lease_a.as_dict(),
        expected_terminal_authority_sha256=expected_authority_a,
        now=NOW,
    )
    assert global_acquire.committed is True

    first = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest_a,
        identity_a,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity_a["identity_sha256"],
        now=NOW,
    )
    assert first.committed is True
    assert first.post_write_reread_verified is True
    assert first.generation == 1

    current = inspect_current_run_authority(
        writer_b, str(remote), manifest=manifest_a, now=NOW,
    )
    assert current.present is True
    assert current.valid is True
    assert current.active is True
    assert current.ref == CURRENT_RUN_POINTER_REF
    assert current.run_id == "run-a"

    without_manifest = inspect_current_run_authority(writer_b, str(remote), now=NOW)
    assert without_manifest.valid is False
    assert without_manifest.active is False
    assert without_manifest.blockers == ("current_run_trusted_launch_manifest_required",)

    duplicate = activate_current_run_authority(
        writer_b,
        str(remote),
        manifest_a,
        identity_a,
        expected_pointer_tip=first.written_remote_tip,
        expected_current_run_identity_sha256=identity_a["identity_sha256"],
        now=NOW,
    )
    assert duplicate.committed is False
    assert duplicate.blockers == ("current_run_already_active",)

    retired = retire_current_run_authority(
        writer_a,
        str(remote),
        expected_pointer_tip=first.written_remote_tip,
        expected_current_run_identity_sha256=identity_a["identity_sha256"],
    )
    assert retired.committed is True
    assert retired.post_write_reread_verified is True
    assert inspect_current_run_authority(writer_b, str(remote), now=NOW).active is False

    stale = activate_current_run_authority(
        writer_b,
        str(remote),
        manifest_a,
        identity_a,
        expected_pointer_tip=first.written_remote_tip,
        expected_current_run_identity_sha256=identity_a["identity_sha256"],
        now=NOW,
    )
    assert stale.committed is False
    assert stale.blockers == ("current_run_pointer_expected_tip_mismatch",)


def test_retired_pointer_can_advance_only_from_exact_latest_tip(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest_a, identity_a, lease_a, expected_authority_a = _authorized_run()
    manifest_b, identity_b, lease_b, expected_authority_b = _authorized_run(
        _manifest(source_git_sha="c" * 40, binding="f" * 64),
        run_id="run-b",
        holder_id="runner-b",
        recovery_manifest="4" * 64,
    )
    assert acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest_a,
        lease_a.as_dict(),
        expected_terminal_authority_sha256=expected_authority_a,
        now=NOW,
    ).committed
    assert acquire_global_training_run_lease(
        writer_b,
        str(remote),
        manifest_b,
        lease_b.as_dict(),
        expected_terminal_authority_sha256=expected_authority_b,
        now=NOW,
    ).committed

    first = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest_a,
        identity_a,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity_a["identity_sha256"],
        now=NOW,
    )
    retired = retire_current_run_authority(
        writer_a,
        str(remote),
        expected_pointer_tip=first.written_remote_tip,
        expected_current_run_identity_sha256=identity_a["identity_sha256"],
    )
    assert retired.committed is True

    second = activate_current_run_authority(
        writer_b,
        str(remote),
        manifest_b,
        identity_b,
        expected_pointer_tip=retired.written_remote_tip,
        expected_current_run_identity_sha256=identity_b["identity_sha256"],
        now=NOW,
    )
    assert second.committed is True
    assert second.generation == 2
    current = inspect_current_run_authority(
        writer_a, str(remote), manifest=manifest_b, now=NOW,
    )
    assert current.active is True
    assert current.generation == 2
    assert current.run_id == "run-b"
    assert current.launch_manifest_sha256 == launch_manifest_sha256(manifest_b)


def test_boolean_generation_and_source_or_binding_substitution_fail_closed() -> None:
    manifest = _manifest()
    identity = _identity()
    with pytest.raises(ValueError, match="generation_must_be_positive_integer"):
        build_current_run_pointer_state(
            manifest, _global_inspection(manifest), identity, generation=True,
            global_lease_state_sha256=_global_state_sha256(manifest),
            global_lease_expires_at_utc="2026-09-27T14:00:00Z",
        )

    wrong_source = _identity(source_git_sha="c" * 40)
    with pytest.raises(ValueError, match="current_run_source_git_sha_mismatch"):
        build_current_run_pointer_state(
            manifest,
            _global_inspection(manifest),
            wrong_source,
            generation=1,
            global_lease_state_sha256=_global_state_sha256(manifest),
            global_lease_expires_at_utc="2026-09-27T14:00:00Z",
        )

    wrong_binding = _identity(manifest=manifest, binding="f" * 64)
    with pytest.raises(ValueError, match="current_run_portable_binding_mismatch"):
        build_current_run_pointer_state(
            manifest,
            _global_inspection(manifest),
            wrong_binding,
            generation=1,
            global_lease_state_sha256=_global_state_sha256(manifest),
            global_lease_expires_at_utc="2026-09-27T14:00:00Z",
        )


def test_activation_rejects_candidate_selected_current_run_identity_root(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, _ = git_pair
    manifest = _manifest()
    identity = _identity(manifest=manifest)

    result = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256="f" * 64,
        now=NOW,
    )

    assert result.committed is False
    assert result.blockers == ("expected_current_run_identity_sha256_mismatch",)
    assert inspect_current_run_authority(writer_a, str(remote)).present is False


def test_mutations_fail_closed_on_uncanonicalizable_manifest(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, _ = git_pair
    manifest = _manifest()
    identity = _identity(manifest=manifest)
    malformed = deepcopy(manifest)
    malformed["recipe"]["opaque"] = object()

    activated = activate_current_run_authority(
        writer_a,
        str(remote),
        malformed,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert activated.committed is False
    assert activated.blockers[0].startswith("launch_manifest_snapshot_invalid:")

    refreshed = refresh_current_run_authority(
        writer_a,
        str(remote),
        malformed,
        expected_pointer_tip="a" * 40,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert refreshed.committed is False
    assert refreshed.blockers[0].startswith("launch_manifest_snapshot_invalid:")


def test_activation_fail_closes_on_uncanonicalizable_identity(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, _ = git_pair
    manifest = _manifest()
    identity = _identity(manifest=manifest)
    identity["opaque"] = object()

    result = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256="a" * 64,
        now=NOW,
    )

    assert result.committed is False
    assert result.blockers[0].startswith("current_run_identity_snapshot_invalid:")


def test_activation_rejects_expired_running_global_lease(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, _ = git_pair
    acquired_at = datetime(2026, 9, 27, 11, 0, tzinfo=UTC)
    manifest, identity, lease, expected_authority = _authorized_run(now=acquired_at)
    assert acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=acquired_at,
    ).committed

    result = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )

    assert result.committed is False
    assert result.blockers == ("training_run_lease_expired",)
    assert inspect_current_run_authority(writer_a, str(remote), now=NOW).present is False


def test_active_pointer_invalidates_on_global_lease_tip_change_or_expiry(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True
    pointer = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert pointer.committed is True

    expired = inspect_current_run_authority(
        writer_b,
        str(remote),
        manifest=manifest,
        now=NOW + timedelta(hours=2),
    )
    assert expired.valid is False
    assert expired.active is False
    assert "current_run_global_lease_expired" in expired.blockers

    renewed = renew_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        expected_remote_tip=acquired.written_remote_tip,
        ttl_seconds=3600,
        now=NOW + timedelta(minutes=10),
    )
    assert renewed.committed is True
    drifted = inspect_current_run_authority(
        writer_b,
        str(remote),
        manifest=manifest,
        now=NOW + timedelta(minutes=10),
    )
    assert drifted.valid is False
    assert drifted.active is False
    assert drifted.blockers == ("current_run_global_lease_tip_changed",)

def test_same_run_refresh_after_global_lease_renewal(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True

    pointer = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert pointer.committed is True
    assert pointer.written_remote_tip is not None

    renewed = renew_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        expected_remote_tip=acquired.written_remote_tip,
        ttl_seconds=3600,
        now=NOW + timedelta(minutes=10),
    )
    assert renewed.committed is True
    assert renewed.written_remote_tip is not None

    before_refresh = inspect_current_run_authority(
        writer_b,
        str(remote),
        manifest=manifest,
        now=NOW + timedelta(minutes=10),
    )
    assert before_refresh.valid is False
    assert before_refresh.active is False
    assert before_refresh.blockers == ("current_run_global_lease_tip_changed",)

    refreshed = refresh_current_run_authority(
        writer_b,
        str(remote),
        manifest,
        expected_pointer_tip=pointer.written_remote_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW + timedelta(minutes=10),
    )
    assert refreshed.committed is True
    assert refreshed.post_write_reread_verified is True
    assert refreshed.generation == pointer.generation
    assert refreshed.written_remote_tip is not None
    assert refreshed.written_remote_tip != pointer.written_remote_tip

    current = inspect_current_run_authority(
        writer_a,
        str(remote),
        manifest=manifest,
        now=NOW + timedelta(minutes=10),
    )
    assert current.valid is True
    assert current.active is True
    assert current.generation == 1
    assert current.run_id == "run-a"
    assert current.current_run_identity_sha256 == identity["identity_sha256"]
    assert current.global_lease_remote_tip == renewed.written_remote_tip

    stale = refresh_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        expected_pointer_tip=pointer.written_remote_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW + timedelta(minutes=10),
    )
    assert stale.committed is False
    assert stale.blockers == ("current_run_pointer_expected_tip_mismatch",)


def test_refresh_rejects_identity_substitution_and_unrenewed_lease(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, _ = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True
    pointer = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert pointer.committed is True
    assert pointer.written_remote_tip is not None

    no_renewal = refresh_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        expected_pointer_tip=pointer.written_remote_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert no_renewal.committed is False
    assert no_renewal.blockers == ("global_training_run_lease_not_renewed",)

    renewed = renew_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        expected_remote_tip=acquired.written_remote_tip,
        ttl_seconds=3600,
        now=NOW + timedelta(minutes=10),
    )
    assert renewed.committed is True

    candidate_b = _identity(
        manifest=manifest,
        run_id="run-b",
        recovery_manifest="4" * 64,
    )
    substitution = refresh_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        expected_pointer_tip=pointer.written_remote_tip,
        expected_current_run_identity_sha256=candidate_b["identity_sha256"],
        now=NOW + timedelta(minutes=10),
    )
    assert substitution.committed is False
    assert substitution.blockers == ("current_run_identity_mismatch",)


def test_refresh_rejects_candidate_manifest_substitution(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, _ = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True
    pointer = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert pointer.committed is True
    assert pointer.written_remote_tip is not None

    renewed = renew_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        expected_remote_tip=acquired.written_remote_tip,
        ttl_seconds=3600,
        now=NOW + timedelta(minutes=10),
    )
    assert renewed.committed is True

    manifest_b = _manifest(source_git_sha="c" * 40)
    substituted = refresh_current_run_authority(
        writer_a,
        str(remote),
        manifest_b,
        expected_pointer_tip=pointer.written_remote_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW + timedelta(minutes=10),
    )
    assert substituted.committed is False
    assert substituted.blockers == ("current_run_launch_manifest_mismatch",)


def test_refresh_second_renewal_race_commits_no_active_authority(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True
    pointer = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert pointer.committed is True
    assert pointer.written_remote_tip is not None

    first_renewal = renew_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        expected_remote_tip=acquired.written_remote_tip,
        ttl_seconds=3600,
        now=NOW + timedelta(minutes=10),
    )
    assert first_renewal.committed is True
    assert first_renewal.written_remote_tip is not None

    original_read_snapshot = current_run._read_snapshot
    raced = False

    def read_then_renew_again(
        repo_root: str | Path,
        remote_name: str,
        manifest_value: dict,
    ):
        nonlocal raced
        snapshot = original_read_snapshot(repo_root, remote_name, manifest_value)
        if not raced:
            raced = True
            second = renew_global_training_run_lease(
                writer_a,
                str(remote),
                manifest,
                expected_remote_tip=first_renewal.written_remote_tip,
                ttl_seconds=3600,
                now=NOW + timedelta(minutes=20),
            )
            assert second.committed is True
        return snapshot

    monkeypatch.setattr(current_run, "_read_snapshot", read_then_renew_again)
    refreshed = refresh_current_run_authority(
        writer_b,
        str(remote),
        manifest,
        expected_pointer_tip=pointer.written_remote_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW + timedelta(minutes=10),
    )

    assert raced is True
    assert refreshed.committed is True
    assert refreshed.post_write_reread_verified is False
    assert refreshed.blockers == ("current_run_pointer_post_write_reread_mismatch",)

    inspection = inspect_current_run_authority(
        writer_a,
        str(remote),
        manifest=manifest,
        now=NOW + timedelta(minutes=20),
    )
    assert inspection.valid is False
    assert inspection.active is False
    assert inspection.blockers == ("current_run_global_lease_tip_changed",)


def test_refresh_rejects_retired_pointer_and_manifest_substitution(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, _ = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True

    pointer = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert pointer.committed is True
    assert pointer.written_remote_tip is not None

    renewed = renew_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        expected_remote_tip=acquired.written_remote_tip,
        ttl_seconds=3600,
        now=NOW + timedelta(minutes=10),
    )
    assert renewed.committed is True

    substituted_manifest = deepcopy(manifest)
    substituted_manifest["recipe"]["seed"] += 1
    substituted = refresh_current_run_authority(
        writer_a,
        str(remote),
        substituted_manifest,
        expected_pointer_tip=pointer.written_remote_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW + timedelta(minutes=10),
    )
    assert substituted.committed is False
    assert substituted.blockers == ("current_run_launch_manifest_mismatch",)

    retired = retire_current_run_authority(
        writer_a,
        str(remote),
        expected_pointer_tip=pointer.written_remote_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
    )
    assert retired.committed is True
    assert retired.written_remote_tip is not None

    rejected = refresh_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        expected_pointer_tip=retired.written_remote_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW + timedelta(minutes=10),
    )
    assert rejected.committed is False
    assert rejected.blockers == ("current_run_pointer_not_active",)


def test_inspection_rechecks_global_lease_tip_after_blob_read(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True
    assert acquired.written_remote_tip is not None

    activated = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert activated.committed is True

    original_fetch = current_run._fetch_remote_commit
    raced = False

    def read_then_renew(
        repo_root: str | Path,
        remote_name: str,
        ref: str,
        expected_tip: str,
    ) -> bytes:
        nonlocal raced
        raw = original_fetch(repo_root, remote_name, ref, expected_tip)
        if not raced:
            raced = True
            renewed = renew_global_training_run_lease(
                writer_a,
                str(remote),
                manifest,
                expected_remote_tip=acquired.written_remote_tip,
                ttl_seconds=3600,
                now=NOW + timedelta(minutes=10),
            )
            assert renewed.committed is True
        return raw

    monkeypatch.setattr(current_run, "_fetch_remote_commit", read_then_renew)
    inspection = inspect_current_run_authority(
        writer_b,
        str(remote),
        manifest=manifest,
        now=NOW + timedelta(minutes=10),
    )

    assert raced is True
    assert inspection.valid is False
    assert inspection.active is False
    assert inspection.blockers == ("current_run_global_lease_tip_changed_during_read",)


def test_pointer_read_rechecks_fixed_ref_after_blob_read(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True
    pointer = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert pointer.committed is True
    assert pointer.written_remote_tip is not None

    calls = 0

    def drifting_tip(
        repo_root: str | Path,
        remote_name: str,
        ref: str,
    ) -> str:
        nonlocal calls
        calls += 1
        assert repo_root == writer_b
        assert remote_name == str(remote)
        assert ref == CURRENT_RUN_POINTER_REF
        if calls == 1:
            return pointer.written_remote_tip
        return "f" * 40

    monkeypatch.setattr(current_run, "_remote_tip", drifting_tip)
    with pytest.raises(
        current_run.CurrentRunAuthorityError,
        match="current_run_pointer_changed_during_read",
    ):
        current_run._fetch_pointer_bytes(
            writer_b,
            str(remote),
            pointer.written_remote_tip,
        )
    assert calls == 2


def test_inspection_fail_closes_on_pointer_type_error(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, _ = git_pair

    def malformed_pointer(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("current_run_pointer_not_object")

    monkeypatch.setattr(current_run, "_read_pointer_state", malformed_pointer)
    inspection = inspect_current_run_authority(writer_a, str(remote), now=NOW)

    assert inspection.present is True
    assert inspection.valid is False
    assert inspection.active is False
    assert inspection.blockers == ("current_run_pointer_not_object",)


def test_retire_contains_pointer_commit_failure(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, _ = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True
    activated = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert activated.committed is True
    assert activated.written_remote_tip is not None

    def fail_write(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise current_run.CurrentRunAuthorityError("synthetic_pointer_commit_failure")

    monkeypatch.setattr(current_run, "_write_pointer_commit", fail_write)
    retired = retire_current_run_authority(
        writer_a,
        str(remote),
        expected_pointer_tip=activated.written_remote_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
    )

    assert retired.committed is False
    assert retired.blockers == ("synthetic_pointer_commit_failure",)
    assert retired.expected_remote_tip == activated.written_remote_tip
    assert retired.observed_remote_tip == activated.written_remote_tip


def test_mutations_fail_closed_on_pointer_type_error(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, _ = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True

    def malformed_pointer(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("current_run_pointer_not_object")

    monkeypatch.setattr(current_run, "_read_pointer_state", malformed_pointer)

    activated = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert activated.committed is False
    assert activated.blockers == ("current_run_pointer_not_object",)

    refreshed = refresh_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        expected_pointer_tip="a" * 40,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert refreshed.committed is False
    assert refreshed.blockers == ("current_run_pointer_not_object",)

    retired = retire_current_run_authority(
        writer_a,
        str(remote),
        expected_pointer_tip="a" * 40,
        expected_current_run_identity_sha256=identity["identity_sha256"],
    )
    assert retired.committed is False
    assert retired.blockers == ("current_run_pointer_not_object",)


def test_activate_uses_absent_ref_force_with_lease(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, _ = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True

    original_run_git = current_run._run_git
    pointer_push_args: list[str] | None = None

    def capture_pointer_push(
        repo_root: str | Path,
        args: list[str],
        *,
        input_bytes: bytes | None = None,
    ):
        nonlocal pointer_push_args
        if args and args[0] == "push" and args[-1].endswith(
            f":{CURRENT_RUN_POINTER_REF}"
        ):
            pointer_push_args = list(args)
        return original_run_git(repo_root, args, input_bytes=input_bytes)

    monkeypatch.setattr(current_run, "_run_git", capture_pointer_push)
    activated = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )

    assert activated.committed is True
    assert activated.post_write_reread_verified is True
    assert pointer_push_args is not None
    assert (
        f"--force-with-lease={CURRENT_RUN_POINTER_REF}:"
        in pointer_push_args
    )


def test_refresh_pointer_deletion_race_fails_closed_without_recreation(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True
    pointer = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert pointer.committed is True
    assert pointer.written_remote_tip is not None
    renewed = renew_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        expected_remote_tip=acquired.written_remote_tip,
        ttl_seconds=3600,
        now=NOW + timedelta(minutes=10),
    )
    assert renewed.committed is True

    original_run_git = current_run._run_git
    deleted = False
    observed_push_args: list[str] | None = None

    def delete_pointer_before_push(
        repo_root: str | Path,
        args: list[str],
        *,
        input_bytes: bytes | None = None,
    ):
        nonlocal deleted, observed_push_args
        if (
            not deleted
            and args
            and args[0] == "push"
            and args[-1].endswith(f":{CURRENT_RUN_POINTER_REF}")
        ):
            deleted = True
            observed_push_args = list(args)
            _git(
                "--git-dir",
                str(remote),
                "update-ref",
                "-d",
                CURRENT_RUN_POINTER_REF,
            )
        return original_run_git(repo_root, args, input_bytes=input_bytes)

    monkeypatch.setattr(current_run, "_run_git", delete_pointer_before_push)
    refreshed = refresh_current_run_authority(
        writer_b,
        str(remote),
        manifest,
        expected_pointer_tip=pointer.written_remote_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW + timedelta(minutes=10),
    )

    assert deleted is True
    assert observed_push_args is not None
    assert (
        f"--force-with-lease={CURRENT_RUN_POINTER_REF}:"
        f"{pointer.written_remote_tip}"
        in observed_push_args
    )
    assert refreshed.committed is False
    assert refreshed.blockers == ("current_run_pointer_cas_conflict",)
    assert refreshed.observed_remote_tip is None
    assert (
        _git("ls-remote", "--refs", str(remote), CURRENT_RUN_POINTER_REF)
        == ""
    )


def test_retire_pointer_deletion_race_fails_closed_without_recreation(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, _ = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True
    pointer = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert pointer.committed is True
    assert pointer.written_remote_tip is not None

    original_run_git = current_run._run_git
    deleted = False
    observed_push_args: list[str] | None = None

    def delete_pointer_before_push(
        repo_root: str | Path,
        args: list[str],
        *,
        input_bytes: bytes | None = None,
    ):
        nonlocal deleted, observed_push_args
        if (
            not deleted
            and args
            and args[0] == "push"
            and args[-1].endswith(f":{CURRENT_RUN_POINTER_REF}")
        ):
            deleted = True
            observed_push_args = list(args)
            _git(
                "--git-dir",
                str(remote),
                "update-ref",
                "-d",
                CURRENT_RUN_POINTER_REF,
            )
        return original_run_git(repo_root, args, input_bytes=input_bytes)

    monkeypatch.setattr(current_run, "_run_git", delete_pointer_before_push)
    retired = retire_current_run_authority(
        writer_a,
        str(remote),
        expected_pointer_tip=pointer.written_remote_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
    )

    assert deleted is True
    assert observed_push_args is not None
    assert (
        f"--force-with-lease={CURRENT_RUN_POINTER_REF}:"
        f"{pointer.written_remote_tip}"
        in observed_push_args
    )
    assert retired.committed is False
    assert retired.blockers == ("current_run_pointer_cas_conflict",)
    assert retired.observed_remote_tip is None
    assert (
        _git("ls-remote", "--refs", str(remote), CURRENT_RUN_POINTER_REF)
        == ""
    )


def test_replacement_activate_pointer_deletion_race_fails_closed(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, _ = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True
    first = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert first.committed is True
    assert first.written_remote_tip is not None
    retired = retire_current_run_authority(
        writer_a,
        str(remote),
        expected_pointer_tip=first.written_remote_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
    )
    assert retired.committed is True
    assert retired.written_remote_tip is not None

    original_run_git = current_run._run_git
    deleted = False
    observed_push_args: list[str] | None = None

    def delete_pointer_before_push(
        repo_root: str | Path,
        args: list[str],
        *,
        input_bytes: bytes | None = None,
    ):
        nonlocal deleted, observed_push_args
        if (
            not deleted
            and args
            and args[0] == "push"
            and args[-1].endswith(f":{CURRENT_RUN_POINTER_REF}")
        ):
            deleted = True
            observed_push_args = list(args)
            _git(
                "--git-dir",
                str(remote),
                "update-ref",
                "-d",
                CURRENT_RUN_POINTER_REF,
            )
        return original_run_git(repo_root, args, input_bytes=input_bytes)

    monkeypatch.setattr(current_run, "_run_git", delete_pointer_before_push)
    replacement = activate_current_run_authority(
        writer_a,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=retired.written_remote_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )

    assert deleted is True
    assert observed_push_args is not None
    assert (
        f"--force-with-lease={CURRENT_RUN_POINTER_REF}:"
        f"{retired.written_remote_tip}"
        in observed_push_args
    )
    assert replacement.committed is False
    assert replacement.blockers == ("current_run_pointer_cas_conflict",)
    assert replacement.observed_remote_tip is None
    assert (
        _git("ls-remote", "--refs", str(remote), CURRENT_RUN_POINTER_REF)
        == ""
    )


def test_active_pointer_rejects_untrusted_manifest_variants(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest, identity, lease, authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer_a, str(remote), manifest, lease.as_dict(),
        expected_terminal_authority_sha256=authority, now=NOW,
    )
    assert acquired.committed is True
    pointer = activate_current_run_authority(
        writer_a, str(remote), manifest, identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert pointer.committed is True

    alternate, _, _, _ = _authorized_run(
        base_manifest=_manifest(source_git_sha="c" * 40),
    )
    wrong = inspect_current_run_authority(
        writer_b, str(remote), manifest=alternate, now=NOW,
    )
    assert wrong.present is True
    assert wrong.valid is False
    assert wrong.active is False
    assert wrong.blockers == ("current_run_trusted_launch_manifest_mismatch",)

    malformed = deepcopy(manifest)
    malformed["resource"]["maximum_cost_usd"] = float("nan")
    rejected = inspect_current_run_authority(
        writer_b, str(remote), manifest=malformed, now=NOW,
    )
    assert rejected.present is True
    assert rejected.valid is False
    assert rejected.active is False
    assert len(rejected.blockers) == 1
    assert rejected.blockers[0].startswith("current_run_trusted_launch_manifest_invalid:")


def test_oversized_remote_pointer_denied_before_git_blob_capture(
    git_pair: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    remote, writer, reader = git_pair
    manifest, identity, lease, expected_authority = _authorized_run()
    acquired = acquire_global_training_run_lease(
        writer,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True
    activated = activate_current_run_authority(
        writer,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert activated.committed is True
    assert activated.written_remote_tip is not None

    def write_object(data: bytes, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=writer, input=data, capture_output=True, check=True
        )
        return result.stdout.decode("ascii").strip()

    oversized = b"y" * (current_run.MAX_CURRENT_RUN_POINTER_BYTES + 1)
    blob_sha = write_object(oversized, "hash-object", "-w", "--stdin")
    tree_sha = write_object(
        f"100644 blob {blob_sha}\t{current_run.CURRENT_RUN_POINTER_PATH}\n"
        .encode("ascii"),
        "mktree",
    )
    corrupt_tip = write_object(
        b"oversized current-run pointer\n",
        "-c", "user.name=R01 test",
        "-c", "user.email=r01-test@example.invalid",
        "commit-tree", tree_sha, "-p", activated.written_remote_tip,
    )
    _git("push", str(remote), f"{corrupt_tip}:{CURRENT_RUN_POINTER_REF}", cwd=writer)

    original_run_git = current_run._run_git

    def reject_blob_capture(repo_root, args, **kwargs):
        if args[:2] == ["cat-file", "blob"]:
            raise AssertionError("oversized pointer must not be captured")
        return original_run_git(repo_root, args, **kwargs)

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("invalid pointer must not write or push")

    monkeypatch.setattr(current_run, "_run_git", reject_blob_capture)
    monkeypatch.setattr(current_run, "_write_pointer_commit", unexpected_write)
    inspection = inspect_current_run_authority(
        reader, str(remote), manifest=manifest, now=NOW
    )
    assert inspection.present is True
    assert inspection.valid is False
    assert inspection.active is False
    assert inspection.blockers == ("current_run_pointer_exceeds_byte_limit",)

    rejected = activate_current_run_authority(
        reader,
        str(remote),
        manifest,
        identity,
        expected_pointer_tip=corrupt_tip,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert rejected.committed is False
    assert rejected.blockers == ("current_run_pointer_exceeds_byte_limit",)
    assert rejected.training_authority_granted_by_this_module is False
    assert rejected.optimizer_start_permitted_by_this_module is False
    assert _git("ls-remote", str(remote), CURRENT_RUN_POINTER_REF).split()[0] == corrupt_tip


def test_pointer_decoder_bounds_remote_bytes_and_recursion() -> None:
    oversized = (
        b'{"padding":"' + b"a" * current_run.MAX_CURRENT_RUN_POINTER_BYTES
    )
    with pytest.raises(ValueError, match="current_run_pointer_exceeds_byte_limit"):
        decode_current_run_pointer_state(oversized)

    deep_json = b'{"nested":' + b"[" * 10_000 + b"0" + b"]" * 10_000 + b"}"
    with pytest.raises(ValueError, match="current_run_pointer_json_invalid"):
        decode_current_run_pointer_state(deep_json)


def test_deep_current_run_caller_mappings_fail_closed_before_remote_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspection, activation and refresh cannot leak recursive input faults."""
    deep: dict = {}
    child = deep
    for _ in range(3_000):
        nested: dict = {}
        child["nested"] = nested
        child = nested

    def unexpected_git(*_args, **_kwargs):
        raise AssertionError("invalid caller mapping must not access Git")

    manifest, identity, _lease, _authority = _authorized_run()
    # ACTIVE pointer state is only needed to reach the inspection manifest path.
    global_view = _global_inspection(manifest)
    pointer = build_current_run_pointer_state(
        manifest, global_view, identity, generation=1,
        global_lease_state_sha256=_global_state_sha256(manifest),
        global_lease_expires_at_utc=(NOW + timedelta(hours=1)).isoformat(),
    )
    monkeypatch.setattr(
        current_run, "_read_pointer_state",
        lambda *_args: ("a" * 40, pointer),
    )
    monkeypatch.setattr(current_run, "_run_git", unexpected_git)

    inspected = inspect_current_run_authority(".", "origin", manifest=deep, now=NOW)
    assert inspected.present is True
    assert inspected.valid is False
    assert inspected.active is False
    assert inspected.blockers == (
        "current_run_trusted_launch_manifest_invalid:maximum recursion depth exceeded",
    )

    denied = activate_current_run_authority(
        ".", "origin", deep, identity,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert denied.committed is False
    assert denied.blockers[0].startswith("launch_manifest_snapshot_invalid:")
    assert denied.optimizer_start_permitted_by_this_module is False
    assert denied.training_authority_granted_by_this_module is False

    denied_identity = activate_current_run_authority(
        ".", "origin", manifest, deep,
        expected_pointer_tip=None,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert denied_identity.committed is False
    assert denied_identity.blockers[0].startswith("current_run_identity_snapshot_invalid:")
    assert denied_identity.optimizer_start_permitted_by_this_module is False
    assert denied_identity.training_authority_granted_by_this_module is False

    refresh_denied = refresh_current_run_authority(
        ".", "origin", deep,
        expected_pointer_tip="a" * 40,
        expected_current_run_identity_sha256=identity["identity_sha256"],
        now=NOW,
    )
    assert refresh_denied.committed is False
    assert refresh_denied.blockers[0].startswith("launch_manifest_snapshot_invalid:")
    assert refresh_denied.optimizer_start_permitted_by_this_module is False
    assert refresh_denied.training_authority_granted_by_this_module is False
