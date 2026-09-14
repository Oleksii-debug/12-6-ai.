from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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
    build_training_run_lease,
    canonical_json_bytes,
    launch_manifest_sha256,
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


def _lease(manifest: dict, *, run_id: str = "run-a"):
    return build_training_run_lease(
        manifest,
        run_id=run_id,
        holder_id="runner-a",
        ttl_seconds=3600,
        now=NOW,
    )


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
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
    assert result.provider_backend_global_exclusivity_proven is False
    assert result.global_exclusivity_proven is False
    assert result.renewal_authority_granted is False
    assert result.resume_relaunch_authority_granted is False
    assert result.optimizer_start_permitted_by_this_module is False
    assert result.training_authority_granted_by_this_module is False
    assert result.scientific_truth_changed is False


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
    manifest = _manifest()
    state = build_global_lease_state(manifest, _lease(manifest).as_dict())

    with pytest.raises(ValueError, match="duplicate_json_key"):
        decode_global_lease_state(b'{"schema_version":1,"schema_version":1}', manifest)
    with pytest.raises(ValueError, match="global_lease_state_json_invalid"):
        decode_global_lease_state(b'{"x":NaN}', manifest)
    pretty = json.dumps(state, indent=2, sort_keys=True).encode("utf-8")
    with pytest.raises(ValueError, match="global_lease_state_not_canonical"):
        decode_global_lease_state(pretty, manifest)


def test_raw_state_decoder_rejects_lock_domain_substitution_and_extra_fields() -> None:
    manifest = _manifest()
    state = build_global_lease_state(manifest, _lease(manifest).as_dict())

    substituted = deepcopy(state)
    substituted["lock_domain"] = "github.com/attacker/repository"
    with pytest.raises(ValueError, match="global_lease_lock_domain_mismatch"):
        decode_global_lease_state(canonical_json_bytes(substituted), manifest)

    extra = deepcopy(state)
    extra["grant_training"] = True
    with pytest.raises(ValueError, match="global_lease_state_fields_mismatch"):
        decode_global_lease_state(canonical_json_bytes(extra), manifest)


def test_acquire_is_single_winner_and_reread_verified(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest = _manifest()
    first_lease = _lease(manifest, run_id="run-a")
    second_lease = build_training_run_lease(
        manifest,
        run_id="run-b",
        holder_id="runner-b",
        ttl_seconds=3600,
        now=NOW,
    )

    first = acquire_global_training_run_lease(
        writer_a, str(remote), manifest, first_lease.as_dict(), now=NOW
    )
    second = acquire_global_training_run_lease(
        writer_b, str(remote), manifest, second_lease.as_dict(), now=NOW
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


def test_renew_is_fast_forward_and_stale_tip_cannot_retry_itself_into_authority(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, writer_b = git_pair
    manifest = _manifest()
    acquired = acquire_global_training_run_lease(
        writer_a, str(remote), manifest, _lease(manifest).as_dict(), now=NOW
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
    manifest = _manifest()
    lease = _lease(manifest)
    acquired = acquire_global_training_run_lease(
        writer_a, str(remote), manifest, lease.as_dict(), now=NOW
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

    replacement = build_training_run_lease(
        manifest,
        run_id="replacement",
        holder_id="runner-new",
        ttl_seconds=3600,
        now=NOW + timedelta(minutes=20),
    )
    denied = acquire_global_training_run_lease(
        writer_a,
        str(remote),
        manifest,
        replacement.as_dict(),
        now=NOW + timedelta(minutes=20),
    )
    assert denied.committed is False
    assert denied.blockers == ("global_training_run_lease_already_exists",)
    assert denied.observed_remote_tip == terminal.written_remote_tip


def test_expired_running_lease_cannot_be_renewed_by_backdated_retry(
    git_pair: tuple[Path, Path, Path],
) -> None:
    remote, writer_a, _ = git_pair
    manifest = _manifest()
    short = build_training_run_lease(
        manifest,
        run_id="short",
        holder_id="runner-a",
        ttl_seconds=60,
        now=NOW,
    )
    acquired = acquire_global_training_run_lease(
        writer_a, str(remote), manifest, short.as_dict(), now=NOW
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
