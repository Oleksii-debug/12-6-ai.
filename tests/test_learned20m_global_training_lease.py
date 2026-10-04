from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import twelve_six.learned20m_global_training_lease as global_lease_module
from twelve_six.learned20m_global_training_lease import (
    CANONICAL_LOCK_DOMAIN,
    CANONICAL_REPOSITORY,
    GLOBAL_LEASE_CONTRACT_ID,
    GlobalLeaseOperation,
    acquire_global_training_run_lease,
    build_global_lease_state,
    decode_global_lease_state,
    global_training_run_lease_ref,
    inspect_global_training_run_lease,
    renew_global_training_run_lease,
    terminate_global_training_run_lease,
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

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)


def _manifest() -> dict:
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
        "portable_run_binding_sha256": "a" * 64,
    }
    return {
        "schema_version": 1,
        "manifest_id": "R01-LEARNED20M-LAUNCH-MANIFEST-V1",
        "stage": "LEARNED_20M",
        "identities": {"source_git_sha": "b" * 40, **hashes},
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


def _terminal_authority(manifest: dict, *, exposure: int) -> dict:
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
        "recovery_run_id": "learned20m-run-001",
        "recovery_run_manifest_sha256": "7" * 64,
        "recovery_attempt_authority_sha256": "8" * 64,
        "checkpoint_contract_sha256": manifest["checkpoint"]["checkpoint_contract_sha256"],
        "checkpoint_cadence_sha256": "9" * 64,
        "resume_rules_sha256": "a" * 64,
        "safe_stop_current_run_sha256": "b" * 64,
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
        "training_authority_sha256": manifest["authorities"]["training"]["evidence_sha256"],
        "compute_authority_ref": manifest["authorities"]["compute"]["reference"],
        "compute_authority_sha256": manifest["authorities"]["compute"]["evidence_sha256"],
        "execution_backend": manifest["execution_backend"],
        "authorized_optimized_target_exposure": exposure,
    }
    authority["authority_identity_sha256"] = terminal_authority_sha256(authority)
    return authority


def _lease(manifest: dict, *, run_id: str = "run-a"):
    return build_training_run_lease(
        manifest,
        run_id=run_id,
        holder_id="runner-a",
        ttl_seconds=3600,
        now=NOW,
    )


def _authorized_manifest() -> tuple[dict, str]:
    base = _manifest()
    authority = _terminal_authority(base, exposure=1_000)
    return (
        finalize_launch_manifest(base, authority),
        authority["authority_identity_sha256"],
    )


def _authorized_lease(
    manifest: dict,
    expected_terminal_authority_sha256: str,
    *,
    run_id: str = "run-a",
    holder_id: str = "runner-a",
    ttl_seconds: int = 3600,
    now: datetime = NOW,
):
    return build_authorized_training_run_lease(
        manifest,
        expected_terminal_authority_sha256=expected_terminal_authority_sha256,
        run_id=run_id,
        holder_id=holder_id,
        ttl_seconds=ttl_seconds,
        now=now,
    )


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


def _assert_no_authority_widening(result: GlobalLeaseOperation) -> None:
    assert result.remote_write_outcome_unknown is False
    assert result.provider_backend_global_exclusivity_proven is False
    assert result.global_exclusivity_proven is False
    assert result.renewal_authority_granted is False
    assert result.resume_relaunch_authority_granted is False
    assert result.optimizer_start_permitted_by_this_module is False
    assert result.training_authority_granted_by_this_module is False
    assert result.scientific_truth_changed is False


def test_terminal_authority_lease_composes_with_global_cas_without_authority_widening(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, writer_b = git_pair
    base = _manifest()
    authority = _terminal_authority(base, exposure=1_000)
    manifest = finalize_launch_manifest(base, authority)
    expected_authority = authority["authority_identity_sha256"]

    lease = build_authorized_training_run_lease(
        manifest,
        expected_terminal_authority_sha256=expected_authority,
        run_id="run-terminal-authority",
        holder_id="runner-terminal-authority",
        ttl_seconds=3600,
        now=NOW,
    )

    expected_manifest = launch_manifest_sha256(manifest)
    assert lease.manifest_sha256 == expected_manifest
    assert global_training_run_lease_ref(manifest).endswith(f"/{expected_manifest}")

    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True
    assert acquired.post_write_reread_verified is True
    assert acquired.launch_manifest_sha256 == expected_manifest
    assert acquired.run_id == lease.run_id
    _assert_no_authority_widening(acquired)

    inspection = inspect_global_training_run_lease(writer_b, str(remote), manifest)
    assert inspection.present is True
    assert inspection.valid is True
    assert inspection.launch_manifest_sha256 == expected_manifest
    assert inspection.run_id == lease.run_id
    assert inspection.optimizer_start_permitted_by_this_module is False


def test_ref_and_state_bind_fixed_repository_lock_domain_and_manifest() -> None:
    manifest = _manifest()
    lease = _lease(manifest)
    digest = launch_manifest_sha256(manifest)

    assert global_training_run_lease_ref(manifest).endswith(f"/{digest}")
    state = build_global_lease_state(manifest, lease.as_dict())
    assert state["global_lease_contract_id"] == GLOBAL_LEASE_CONTRACT_ID
    assert state["repository"] == CANONICAL_REPOSITORY
    assert state["lock_domain"] == CANONICAL_LOCK_DOMAIN
    assert state["launch_manifest_sha256"] == digest


def test_raw_state_decoder_rejects_duplicate_nonfinite_and_noncanonical_json() -> None:
    manifest, expected_authority = _authorized_manifest()
    state = build_global_lease_state(
        manifest,
        _authorized_lease(manifest, expected_authority).as_dict(),
    )

    with pytest.raises(ValueError, match="duplicate_json_key"):
        decode_global_lease_state(b'{"schema_version":1,"schema_version":1}', manifest)
    with pytest.raises(ValueError, match="global_lease_state_json_invalid"):
        decode_global_lease_state(b'{"x":NaN}', manifest)
    pretty = json.dumps(state, indent=2, sort_keys=True).encode("utf-8")
    with pytest.raises(ValueError, match="global_lease_state_not_canonical"):
        decode_global_lease_state(pretty, manifest)


def test_raw_state_decoder_rejects_lock_domain_substitution_and_extra_fields() -> None:
    manifest, expected_authority = _authorized_manifest()
    state = build_global_lease_state(
        manifest,
        _authorized_lease(manifest, expected_authority).as_dict(),
    )

    substituted = deepcopy(state)
    substituted["lock_domain"] = "github.com/attacker/repository"
    with pytest.raises(ValueError, match="global_lease_lock_domain_mismatch"):
        decode_global_lease_state(canonical_json_bytes(substituted), manifest)

    extra = deepcopy(state)
    extra["grant_training"] = True
    with pytest.raises(ValueError, match="global_lease_state_fields_mismatch"):
        decode_global_lease_state(canonical_json_bytes(extra), manifest)



@pytest.mark.parametrize("payload", [b"[]", b"null", b'"text"', b"7"])
def test_non_object_remote_lease_blocks_inspect_and_renew(
    monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    manifest, _ = _authorized_manifest()
    expected_tip = "a" * 40

    with pytest.raises(TypeError, match="global_lease_state_not_object"):
        decode_global_lease_state(payload, manifest)

    monkeypatch.setattr(global_lease_module, "_remote_tip", lambda *_: expected_tip)
    monkeypatch.setattr(global_lease_module, "_fetch_remote_commit", lambda *_: payload)

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("invalid remote state must not write or push")

    monkeypatch.setattr(global_lease_module, "_write_state_commit", unexpected_write)
    monkeypatch.setattr(global_lease_module, "_push_candidate", unexpected_write)

    inspected = inspect_global_training_run_lease(".", "origin", manifest)
    assert inspected.present is True
    assert inspected.valid is False
    assert inspected.blockers == ("global_lease_remote_state_invalid",)

    renewed = renew_global_training_run_lease(
        ".",
        "origin",
        manifest,
        expected_remote_tip=expected_tip,
        ttl_seconds=3600,
        now=NOW,
    )
    assert renewed.committed is False
    assert renewed.blockers == ("global_lease_remote_state_invalid",)
    _assert_no_authority_widening(renewed)


def test_non_object_remote_lease_via_real_git_ref_is_fail_closed(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer, reader = git_pair
    manifest, expected_authority = _authorized_manifest()
    lease = _authorized_lease(manifest, expected_authority)
    acquired = acquire_global_training_run_lease(
        writer,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.committed is True
    assert acquired.post_write_reread_verified is True
    assert acquired.written_remote_tip is not None

    def write_git_object(data: bytes, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=writer, input=data, capture_output=True, check=True
        )
        return result.stdout.decode("ascii").strip()

    blob = write_git_object(b"[]", "hash-object", "-w", "--stdin")
    tree = write_git_object(
        f"100644 blob {blob}\t{global_lease_module.GLOBAL_LEASE_STATE_PATH}\n".encode(
            "ascii"
        ),
        "mktree",
    )
    corrupt_tip = write_git_object(
        b"malformed remote lease object\n",
        "-c", "user.name=R01 test",
        "-c", "user.email=r01-test@example.invalid",
        "commit-tree", tree, "-p", acquired.written_remote_tip,
    )
    ref = global_training_run_lease_ref(manifest)
    _git("push", str(remote), f"{corrupt_tip}:{ref}", cwd=writer)

    inspected = inspect_global_training_run_lease(reader, str(remote), manifest)
    assert inspected.present is True
    assert inspected.valid is False
    assert inspected.blockers == ("global_lease_remote_state_invalid",)

    renewed = renew_global_training_run_lease(
        reader, str(remote), manifest,
        expected_remote_tip=corrupt_tip, ttl_seconds=3600, now=NOW,
    )
    assert renewed.committed is False
    assert renewed.blockers == ("global_lease_remote_state_invalid",)
    _assert_no_authority_widening(renewed)
    assert _git("ls-remote", str(remote), ref).split()[0] == corrupt_tip


def test_acquire_is_single_winner_and_reread_verified(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest, expected_authority = _authorized_manifest()
    first_lease = _authorized_lease(
        manifest, expected_authority, run_id="run-a"
    )
    second_lease = _authorized_lease(
        manifest,
        expected_authority,
        run_id="run-b",
        holder_id="runner-b",
    )

    first = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        first_lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    second = acquire_global_training_run_lease(
        writer_b,
        str(remote),
        manifest,
        second_lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )

    assert first.committed is True
    assert first.post_write_reread_verified is True
    assert first.cooperative_git_ref_cas_mechanics_verified is True
    assert second.committed is False
    assert second.blockers == ("global_training_run_lease_already_exists",)
    assert second.observed_remote_tip == first.written_remote_tip
    _assert_no_authority_widening(first)
    _assert_no_authority_widening(second)

    inspection = inspect_global_training_run_lease(writer_b, str(remote), manifest)
    assert inspection.present is True
    assert inspection.valid is True
    assert inspection.remote_tip == first.written_remote_tip
    assert inspection.run_id == "run-a"
    assert inspection.global_exclusivity_proven is False
    assert inspection.optimizer_start_permitted_by_this_module is False


def test_acquire_does_not_overwrite_ref_created_after_empty_read(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, _ = git_pair
    manifest, expected_authority = _authorized_manifest()
    lease = _authorized_lease(manifest, expected_authority)
    state = build_global_lease_state(manifest, lease.as_dict())
    ref = global_training_run_lease_ref(manifest)
    real_push = global_lease_module._push_candidate
    competing_tip: str | None = None

    def install_competitor_before_push(
        repo_root,
        remote_arg,
        candidate_tip,
        ref_arg,
        *,
        expected_remote_tip,
    ):
        nonlocal competing_tip
        assert ref_arg == ref
        assert expected_remote_tip is None
        competing_tip = global_lease_module._write_state_commit(
            repo_root,
            state,
            parent_tip=None,
            operation="competing-acquire",
        )
        _git("push", str(remote), f"{competing_tip}:{ref}", cwd=writer_a)
        return real_push(
            repo_root,
            remote_arg,
            candidate_tip,
            ref_arg,
            expected_remote_tip=expected_remote_tip,
        )

    monkeypatch.setattr(
        global_lease_module,
        "_push_candidate",
        install_competitor_before_push,
    )
    denied = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )

    assert competing_tip is not None
    assert denied.committed is False
    assert denied.post_write_reread_verified is False
    assert denied.remote_write_outcome_unknown is False
    assert denied.written_remote_tip is None
    assert denied.blockers == ("global_lease_ref_create_rejected",)
    assert denied.observed_remote_tip == competing_tip
    assert _git("--git-dir", str(remote), "rev-parse", ref) == competing_tip


def test_acquire_freezes_manifest_and_lease_before_assessment(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, _ = git_pair
    manifest, expected_authority = _authorized_manifest()
    original_manifest = deepcopy(manifest)
    lease = _authorized_lease(manifest, expected_authority).as_dict()
    original_lease = deepcopy(lease)
    real_assess = global_lease_module.assess_training_run_lease

    def mutating_assess(manifest_arg, lease_arg, *, now):
        result = real_assess(manifest_arg, lease_arg, now=now)
        manifest["recipe"]["seed"] = 9999
        lease["run_id"] = "mutated-run"
        lease["holder_id"] = "mutated-holder"
        return result

    monkeypatch.setattr(
        global_lease_module,
        "assess_training_run_lease",
        mutating_assess,
    )
    result = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease,
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )

    assert result.committed is True
    assert result.post_write_reread_verified is True
    assert result.launch_manifest_sha256 == launch_manifest_sha256(original_manifest)
    assert result.run_id == original_lease["run_id"]
    inspection = inspect_global_training_run_lease(
        writer_a,
        str(remote),
        original_manifest,
    )
    assert inspection.valid is True
    assert inspection.run_id == original_lease["run_id"]
    assert launch_manifest_sha256(manifest) != launch_manifest_sha256(original_manifest)


def test_acquire_recovers_when_push_reports_failure_after_remote_commit(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, _ = git_pair
    manifest, expected_authority = _authorized_manifest()
    real_push = global_lease_module._push_candidate

    def push_then_report_failure(
        repo_root,
        remote_arg,
        candidate_tip,
        ref,
        *,
        expected_remote_tip,
    ):
        assert (
            real_push(
                repo_root,
                remote_arg,
                candidate_tip,
                ref,
                expected_remote_tip=expected_remote_tip,
            )
            is True
        )
        return False

    monkeypatch.setattr(global_lease_module, "_push_candidate", push_then_report_failure)
    result = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        _authorized_lease(manifest, expected_authority).as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )

    assert result.committed is True
    assert result.post_write_reread_verified is True
    assert result.remote_write_outcome_unknown is False
    assert result.written_remote_tip is not None
    assert (
        _git("--git-dir", str(remote), "rev-parse", global_training_run_lease_ref(manifest))
        == result.written_remote_tip
    )


def test_acquire_reports_committed_unverified_after_post_write_transport_failure(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, _ = git_pair
    manifest, expected_authority = _authorized_manifest()
    real_remote_tip = global_lease_module._remote_tip
    calls = 0

    def fail_after_write(repo_root, remote_arg, ref):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise global_lease_module._GlobalLeaseFailure("simulated_post_write_failure")
        return real_remote_tip(repo_root, remote_arg, ref)

    monkeypatch.setattr(global_lease_module, "_remote_tip", fail_after_write)
    result = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        _authorized_lease(manifest, expected_authority).as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )

    assert result.committed is True
    assert result.post_write_reread_verified is False
    assert result.remote_write_outcome_unknown is False
    assert result.written_remote_tip is not None
    assert result.blockers == ("simulated_post_write_failure",)
    assert (
        _git("--git-dir", str(remote), "rev-parse", global_training_run_lease_ref(manifest))
        == result.written_remote_tip
    )


def test_global_acquire_rejects_self_consistent_manifest_under_wrong_expected_root(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, _ = git_pair
    base_a = _manifest()
    authority_a = _terminal_authority(base_a, exposure=1_000)
    expected_authority_a = authority_a["authority_identity_sha256"]

    base_b = deepcopy(base_a)
    base_b["recipe"]["seed"] += 1
    authority_b = _terminal_authority(base_b, exposure=1_000)
    manifest_b = finalize_launch_manifest(base_b, authority_b)
    ordinary_lease_b = build_training_run_lease(
        manifest_b,
        run_id="bypass-run",
        holder_id="bypass-holder",
        ttl_seconds=3600,
        now=NOW,
    )

    denied = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest_b,
        ordinary_lease_b.as_dict(),
        expected_terminal_authority_sha256=expected_authority_a,
        now=NOW,
    )

    assert denied.committed is False
    assert "terminal_authority_not_independently_expected" in denied.blockers
    assert denied.written_remote_tip is None
    assert denied.training_authority_granted_by_this_module is False
    assert denied.optimizer_start_permitted_by_this_module is False


def test_renew_freezes_manifest_before_post_write_reread(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, writer_b = git_pair
    original_manifest, expected_authority = _authorized_manifest()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        original_manifest,
        _authorized_lease(original_manifest, expected_authority).as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.written_remote_tip is not None

    mutable_manifest = deepcopy(original_manifest)
    real_push = global_lease_module._push_candidate

    def push_then_mutate(
        repo_root,
        remote_arg,
        candidate_tip,
        ref,
        *,
        expected_remote_tip,
    ):
        pushed = real_push(
            repo_root,
            remote_arg,
            candidate_tip,
            ref,
            expected_remote_tip=expected_remote_tip,
        )
        mutable_manifest["recipe"]["seed"] = 424242
        return pushed

    monkeypatch.setattr(global_lease_module, "_push_candidate", push_then_mutate)
    renewed = renew_global_training_run_lease(
        writer_b,
        str(remote),
        mutable_manifest,
        expected_remote_tip=acquired.written_remote_tip,
        ttl_seconds=3600,
        now=NOW + timedelta(minutes=10),
    )

    assert renewed.committed is True
    assert renewed.post_write_reread_verified is True
    assert renewed.launch_manifest_sha256 == launch_manifest_sha256(original_manifest)
    inspection = inspect_global_training_run_lease(
        writer_b,
        str(remote),
        original_manifest,
    )
    assert inspection.valid is True
    assert inspection.renewal_sequence == 1


def test_transition_does_not_recreate_ref_deleted_after_authenticated_read(
    git_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest, expected_authority = _authorized_manifest()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        _authorized_lease(manifest, expected_authority).as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.written_remote_tip is not None

    ref = global_training_run_lease_ref(manifest)
    real_push = global_lease_module._push_candidate

    def delete_ref_before_push(
        repo_root,
        remote_arg,
        candidate_tip,
        ref_arg,
        *,
        expected_remote_tip,
    ):
        assert ref_arg == ref
        assert expected_remote_tip == acquired.written_remote_tip
        _git("--git-dir", str(remote), "update-ref", "-d", ref)
        return real_push(
            repo_root,
            remote_arg,
            candidate_tip,
            ref_arg,
            expected_remote_tip=expected_remote_tip,
        )

    monkeypatch.setattr(
        global_lease_module,
        "_push_candidate",
        delete_ref_before_push,
    )
    renewed = renew_global_training_run_lease(
        writer_b,
        str(remote),
        manifest,
        expected_remote_tip=acquired.written_remote_tip,
        ttl_seconds=3600,
        now=NOW + timedelta(minutes=10),
    )

    assert renewed.committed is False
    assert renewed.post_write_reread_verified is False
    assert renewed.remote_write_outcome_unknown is False
    assert renewed.written_remote_tip is None
    assert renewed.blockers == ("global_lease_transition_push_rejected",)
    assert (
        _git(
            "--git-dir",
            str(remote),
            "for-each-ref",
            "--format=%(refname)",
            ref,
        )
        == ""
    )


def test_renew_is_fast_forward_and_stale_tip_cannot_retry_itself_into_authority(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest, expected_authority = _authorized_manifest()
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        _authorized_lease(manifest, expected_authority).as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.written_remote_tip is not None

    renewed = renew_global_training_run_lease(
        writer_b,
        str(remote),
        manifest,
        expected_remote_tip=acquired.written_remote_tip,
        ttl_seconds=3600,
        now=NOW + timedelta(minutes=10),
    )
    stale = renew_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        expected_remote_tip=acquired.written_remote_tip,
        ttl_seconds=3600,
        now=NOW + timedelta(minutes=20),
    )

    assert renewed.committed is True
    assert renewed.written_remote_tip is not None
    assert stale.committed is False
    assert stale.blockers == ("global_lease_expected_tip_mismatch",)
    assert stale.observed_remote_tip == renewed.written_remote_tip
    _assert_no_authority_widening(renewed)
    _assert_no_authority_widening(stale)

    parents = _git(
        "--git-dir",
        str(remote),
        "rev-list",
        "--parents",
        "-n",
        "1",
        renewed.written_remote_tip,
    ).split()
    assert parents == [renewed.written_remote_tip, acquired.written_remote_tip]


def test_terminal_lineage_remains_immutable_and_cannot_be_freshly_reacquired(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest, expected_authority = _authorized_manifest()
    lease = _authorized_lease(manifest, expected_authority)
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        lease.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.written_remote_tip is not None

    terminal = terminate_global_training_run_lease(
        writer_b,
        str(remote),
        manifest,
        expected_remote_tip=acquired.written_remote_tip,
        status="ABORTED",
        now=NOW + timedelta(minutes=10),
    )
    assert terminal.committed is True
    assert terminal.lease_status == "ABORTED"
    _assert_no_authority_widening(terminal)

    replacement = _authorized_lease(
        manifest,
        expected_authority,
        run_id="replacement",
        holder_id="runner-new",
        now=NOW + timedelta(minutes=20),
    )
    denied = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        replacement.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW + timedelta(minutes=20),
    )
    assert denied.committed is False
    assert denied.blockers == ("global_training_run_lease_already_exists",)
    assert denied.observed_remote_tip == terminal.written_remote_tip


def test_expired_running_lease_cannot_be_renewed_by_backdated_retry(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, _ = git_pair
    manifest, expected_authority = _authorized_manifest()
    short = _authorized_lease(
        manifest,
        expected_authority,
        run_id="short",
        ttl_seconds=60,
    )
    acquired = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        short.as_dict(),
        expected_terminal_authority_sha256=expected_authority,
        now=NOW,
    )
    assert acquired.written_remote_tip is not None

    denied = renew_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        expected_remote_tip=acquired.written_remote_tip,
        ttl_seconds=3600,
        now=NOW + timedelta(seconds=61),
    )
    assert denied.committed is False
    assert denied.blockers
    assert denied.blockers[0].startswith("global_lease_transition_invalid:expired_lease")


def test_remote_global_lease_decoder_bounds_size_and_nesting() -> None:
    manifest, _ = _authorized_manifest()
    oversized = b'{"padding":"' + b"a" * global_lease_module.MAX_GLOBAL_LEASE_STATE_BYTES
    with pytest.raises(ValueError, match="global_lease_state_exceeds_byte_limit"):
        decode_global_lease_state(oversized, manifest)

    deeply_nested = b'{"nested":' + b"[" * 10_000 + b"0" + b"]" * 10_000 + b"}"
    with pytest.raises(ValueError, match="global_lease_state_json_invalid"):
        decode_global_lease_state(deeply_nested, manifest)
