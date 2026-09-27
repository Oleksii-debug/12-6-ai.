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
    build_training_run_lease,
    canonical_json_bytes,
    launch_manifest_sha256,
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
    manifest_a = _manifest()
    identity_a = _identity(manifest=manifest_a)
    lease_a = build_training_run_lease(
        manifest_a,
        run_id="run-a",
        holder_id="runner-a",
        ttl_seconds=3600,
        now=NOW,
    )
    global_acquire = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest_a,
        lease_a.as_dict(),
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

    current = inspect_current_run_authority(writer_b, str(remote), now=NOW)
    assert current.present is True
    assert current.valid is True
    assert current.active is True
    assert current.ref == CURRENT_RUN_POINTER_REF
    assert current.run_id == "run-a"

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
    manifest_a = _manifest()
    identity_a = _identity(manifest=manifest_a)
    manifest_b = _manifest(source_git_sha="c" * 40, binding="f" * 64)
    identity_b = _identity(
        manifest=manifest_b,
        run_id="run-b",
        recovery_manifest="4" * 64,
        binding="f" * 64,
        source_git_sha="c" * 40,
    )

    lease_a = build_training_run_lease(
        manifest_a,
        run_id="run-a",
        holder_id="runner-a",
        ttl_seconds=3600,
        now=NOW,
    )
    lease_b = build_training_run_lease(
        manifest_b,
        run_id="run-b",
        holder_id="runner-b",
        ttl_seconds=3600,
        now=NOW,
    )
    assert acquire_global_training_run_lease(
        writer_a, str(remote), manifest_a, lease_a.as_dict(), now=NOW
    ).committed
    assert acquire_global_training_run_lease(
        writer_b, str(remote), manifest_b, lease_b.as_dict(), now=NOW
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
    current = inspect_current_run_authority(writer_a, str(remote), now=NOW)
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

    wrong_binding = _identity(binding="f" * 64)
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

def test_activation_rejects_expired_running_global_lease(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, _ = git_pair
    manifest = _manifest()
    identity = _identity(manifest=manifest)
    acquired_at = datetime(2026, 9, 27, 11, 0, tzinfo=UTC)
    lease = build_training_run_lease(
        manifest,
        run_id="run-a",
        holder_id="runner-a",
        ttl_seconds=3600,
        now=acquired_at,
    )
    assert acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
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
    manifest = _manifest()
    identity = _identity(manifest=manifest)
    lease = build_training_run_lease(
        manifest,
        run_id="run-a",
        holder_id="runner-a",
        ttl_seconds=3600,
        now=NOW,
    )
    acquired = acquire_global_training_run_lease(
        writer_a, str(remote), manifest, lease.as_dict(), now=NOW
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
        now=NOW + timedelta(minutes=10),
    )
    assert drifted.valid is False
    assert drifted.active is False
    assert drifted.blockers == ("current_run_global_lease_tip_changed",)

