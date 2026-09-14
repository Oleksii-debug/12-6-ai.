"""Cross-runner learned-20M TRAINING_RUN lease using cooperative Git ref CAS.

This module composes the canonical learned-20M launch manifest and local lease
contract.  It proves only compare-and-swap mechanics on the selected Git
transport.  It never turns a successful Git write into training, renewal,
resume/relaunch, optimizer, scientific, or provider-backend authority.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from twelve_six.learned20m_training_lease import (
    TrainingLease,
    assess_training_run_lease,
    canonical_json_bytes,
    launch_manifest_sha256,
    renew_training_run_lease,
    terminate_training_run_lease,
    validate_launch_manifest,
    validate_lease_transition,
    validate_training_run_lease,
)

GLOBAL_LEASE_CONTRACT_ID = "C01-LEARNED20M-GLOBAL-GIT-REF-LEASE-V1"
GLOBAL_LEASE_SCHEMA_VERSION = 1
CANONICAL_REPOSITORY = "Oleksii-debug/12-6-ai."
CANONICAL_LOCK_DOMAIN = "github.com/Oleksii-debug/12-6-ai."
GLOBAL_LEASE_REF_PREFIX = "refs/heads/ts6-training-run-lease-v1"
GLOBAL_LEASE_STATE_PATH = "training-run-lease-v1.json"
MECHANICS_SCOPE = "COOPERATIVE_GIT_REF_CAS_ON_SELECTED_TRANSPORT_ONLY"

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_STATE_FIELDS = {
    "schema_version",
    "global_lease_contract_id",
    "repository",
    "lock_domain",
    "launch_manifest_sha256",
    "lease",
}


class _GlobalLeaseFailure(RuntimeError):
    def __init__(self, blocker: str):
        super().__init__(blocker)
        self.blocker = blocker


class _DuplicateKey(ValueError):
    pass


@dataclass(frozen=True)
class GlobalLeaseSnapshot:
    ref: str
    remote_tip: str
    launch_manifest_sha256: str
    lease: dict[str, Any]


@dataclass(frozen=True)
class GlobalLeaseInspection:
    present: bool
    valid: bool
    ref: str
    remote_tip: str | None
    launch_manifest_sha256: str | None
    run_id: str | None
    lease_status: str | None
    renewal_sequence: int | None
    blockers: tuple[str, ...]
    mechanics_scope: str = MECHANICS_SCOPE
    provider_backend_global_exclusivity_proven: bool = False
    global_exclusivity_proven: bool = False
    renewal_authority_granted: bool = False
    resume_relaunch_authority_granted: bool = False
    optimizer_start_permitted_by_this_module: bool = False
    training_authority_granted_by_this_module: bool = False
    scientific_truth_changed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "valid": self.valid,
            "ref": self.ref,
            "remote_tip": self.remote_tip,
            "launch_manifest_sha256": self.launch_manifest_sha256,
            "run_id": self.run_id,
            "lease_status": self.lease_status,
            "renewal_sequence": self.renewal_sequence,
            "blockers": list(self.blockers),
            "mechanics_scope": self.mechanics_scope,
            "provider_backend_global_exclusivity_proven": (
                self.provider_backend_global_exclusivity_proven
            ),
            "global_exclusivity_proven": self.global_exclusivity_proven,
            "renewal_authority_granted": self.renewal_authority_granted,
            "resume_relaunch_authority_granted": self.resume_relaunch_authority_granted,
            "optimizer_start_permitted_by_this_module": (
                self.optimizer_start_permitted_by_this_module
            ),
            "training_authority_granted_by_this_module": (
                self.training_authority_granted_by_this_module
            ),
            "scientific_truth_changed": self.scientific_truth_changed,
        }


@dataclass(frozen=True)
class GlobalLeaseOperation:
    operation: str
    committed: bool
    post_write_reread_verified: bool
    ref: str
    launch_manifest_sha256: str | None
    expected_remote_tip: str | None
    observed_remote_tip: str | None
    written_remote_tip: str | None
    run_id: str | None
    lease_status: str | None
    blockers: tuple[str, ...]
    mechanics_scope: str = MECHANICS_SCOPE
    cooperative_git_ref_cas_mechanics_verified: bool = False
    provider_backend_global_exclusivity_proven: bool = False
    global_exclusivity_proven: bool = False
    renewal_authority_granted: bool = False
    resume_relaunch_authority_granted: bool = False
    optimizer_start_permitted_by_this_module: bool = False
    training_authority_granted_by_this_module: bool = False
    scientific_truth_changed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "committed": self.committed,
            "post_write_reread_verified": self.post_write_reread_verified,
            "ref": self.ref,
            "launch_manifest_sha256": self.launch_manifest_sha256,
            "expected_remote_tip": self.expected_remote_tip,
            "observed_remote_tip": self.observed_remote_tip,
            "written_remote_tip": self.written_remote_tip,
            "run_id": self.run_id,
            "lease_status": self.lease_status,
            "blockers": list(self.blockers),
            "mechanics_scope": self.mechanics_scope,
            "cooperative_git_ref_cas_mechanics_verified": (
                self.cooperative_git_ref_cas_mechanics_verified
            ),
            "provider_backend_global_exclusivity_proven": (
                self.provider_backend_global_exclusivity_proven
            ),
            "global_exclusivity_proven": self.global_exclusivity_proven,
            "renewal_authority_granted": self.renewal_authority_granted,
            "resume_relaunch_authority_granted": self.resume_relaunch_authority_granted,
            "optimizer_start_permitted_by_this_module": (
                self.optimizer_start_permitted_by_this_module
            ),
            "training_authority_granted_by_this_module": (
                self.training_authority_granted_by_this_module
            ),
            "scientific_truth_changed": self.scientific_truth_changed,
        }


def _exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _normalize_now(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC).replace(microsecond=0)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now_must_be_timezone_aware")
    return value.astimezone(UTC).replace(microsecond=0)


def _validate_transport(remote: str) -> None:
    if not isinstance(remote, str) or not remote or len(remote) > 2048:
        raise ValueError("git_remote_invalid")
    if any(character in remote for character in ("\x00", "\r", "\n")):
        raise ValueError("git_remote_invalid")


def global_training_run_lease_ref(manifest: Mapping[str, Any]) -> str:
    errors = validate_launch_manifest(manifest)
    if errors:
        raise ValueError("invalid_launch_manifest:" + ";".join(errors))
    return f"{GLOBAL_LEASE_REF_PREFIX}/{launch_manifest_sha256(manifest)}"


def build_global_lease_state(
    manifest: Mapping[str, Any], lease: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_errors = validate_launch_manifest(manifest)
    if manifest_errors:
        raise ValueError("invalid_launch_manifest:" + ";".join(manifest_errors))
    lease_errors = validate_training_run_lease(lease, manifest)
    if lease_errors:
        raise ValueError("invalid_training_run_lease:" + ";".join(lease_errors))
    return {
        "schema_version": GLOBAL_LEASE_SCHEMA_VERSION,
        "global_lease_contract_id": GLOBAL_LEASE_CONTRACT_ID,
        "repository": CANONICAL_REPOSITORY,
        "lock_domain": CANONICAL_LOCK_DOMAIN,
        "launch_manifest_sha256": launch_manifest_sha256(manifest),
        "lease": dict(lease),
    }


def validate_global_lease_state(
    state: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[str, ...]:
    errors: list[str] = []
    if set(state) != _STATE_FIELDS:
        errors.append("global_lease_state_fields_mismatch")
    if not _exact_int(state.get("schema_version"), GLOBAL_LEASE_SCHEMA_VERSION):
        errors.append("global_lease_schema_version_mismatch")
    if state.get("global_lease_contract_id") != GLOBAL_LEASE_CONTRACT_ID:
        errors.append("global_lease_contract_id_mismatch")
    if state.get("repository") != CANONICAL_REPOSITORY:
        errors.append("global_lease_repository_mismatch")
    if state.get("lock_domain") != CANONICAL_LOCK_DOMAIN:
        errors.append("global_lease_lock_domain_mismatch")

    manifest_errors = validate_launch_manifest(manifest)
    if manifest_errors:
        errors.extend(f"manifest:{error}" for error in manifest_errors)
        expected_manifest_sha256 = None
    else:
        expected_manifest_sha256 = launch_manifest_sha256(manifest)
        if state.get("launch_manifest_sha256") != expected_manifest_sha256:
            errors.append("global_lease_manifest_sha256_mismatch")

    lease = state.get("lease")
    if not isinstance(lease, Mapping):
        errors.append("global_lease_embedded_lease_missing_or_not_object")
    else:
        errors.extend(
            f"lease:{error}" for error in validate_training_run_lease(lease, manifest)
        )
        if (
            expected_manifest_sha256 is not None
            and lease.get("manifest_sha256") != expected_manifest_sha256
        ):
            errors.append("global_lease_embedded_manifest_sha256_mismatch")
    return tuple(dict.fromkeys(errors))


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise _DuplicateKey(f"duplicate_json_key:{key}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non_finite_json_constant:{value}")


def decode_global_lease_state(
    raw: bytes, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("global_lease_state_utf8_invalid") from exc
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateKey:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("global_lease_state_json_invalid") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("global_lease_state_not_object")
    canonical = canonical_json_bytes(parsed)
    if canonical != raw:
        raise ValueError("global_lease_state_not_canonical")
    errors = validate_global_lease_state(parsed, manifest)
    if errors:
        raise ValueError("invalid_global_lease_state:" + ";".join(errors))
    return dict(parsed)


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_NAME": "Twelve Six Global Lease",
            "GIT_AUTHOR_EMAIL": "global-lease@invalid.local",
            "GIT_COMMITTER_NAME": "Twelve Six Global Lease",
            "GIT_COMMITTER_EMAIL": "global-lease@invalid.local",
        }
    )
    return env


def _run_git(
    repo_root: str | Path,
    args: list[str],
    *,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=os.fspath(repo_root),
            env=_git_env(),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise _GlobalLeaseFailure("git_command_unavailable_or_timed_out") from exc


def _git_ascii(repo_root: str | Path, args: list[str], *, blocker: str) -> str:
    result = _run_git(repo_root, args)
    if result.returncode != 0:
        raise _GlobalLeaseFailure(blocker)
    try:
        return result.stdout.decode("ascii")
    except UnicodeDecodeError as exc:
        raise _GlobalLeaseFailure("git_output_non_ascii") from exc


def _remote_tip(repo_root: str | Path, remote: str, ref: str) -> str | None:
    output = _git_ascii(
        repo_root,
        ["ls-remote", "--refs", "--", remote, ref],
        blocker="git_ls_remote_failed",
    )
    if output == "":
        return None
    lines = output.splitlines()
    if len(lines) != 1:
        raise _GlobalLeaseFailure("git_ls_remote_ambiguous")
    pieces = lines[0].split("\t")
    if len(pieces) != 2 or pieces[1] != ref or _GIT_SHA.fullmatch(pieces[0]) is None:
        raise _GlobalLeaseFailure("git_ls_remote_malformed")
    return pieces[0]


def _delete_local_ref(repo_root: str | Path, ref: str) -> None:
    _run_git(repo_root, ["update-ref", "-d", ref])


def _fetch_remote_commit(
    repo_root: str | Path, remote: str, ref: str, expected_tip: str
) -> bytes:
    temporary_ref = f"refs/ts6-global-lease-read/{secrets.token_hex(16)}"
    try:
        fetch = _run_git(
            repo_root,
            ["fetch", "--quiet", "--no-tags", "--", remote, f"{ref}:{temporary_ref}"],
        )
        if fetch.returncode != 0:
            raise _GlobalLeaseFailure("git_fetch_global_lease_failed")
        if _remote_tip(repo_root, remote, ref) != expected_tip:
            raise _GlobalLeaseFailure("remote_tip_changed_during_read")

        parents_output = _git_ascii(
            repo_root,
            ["rev-list", "--parents", "-n", "1", expected_tip],
            blocker="git_commit_parent_read_failed",
        ).strip()
        parent_parts = parents_output.split()
        if not parent_parts or parent_parts[0] != expected_tip or len(parent_parts) > 2:
            raise _GlobalLeaseFailure("global_lease_commit_parent_shape_invalid")

        tree = _run_git(repo_root, ["ls-tree", "-z", "--full-tree", expected_tip])
        if tree.returncode != 0:
            raise _GlobalLeaseFailure("global_lease_tree_read_failed")
        entries = [entry for entry in tree.stdout.split(b"\x00") if entry]
        if len(entries) != 1:
            raise _GlobalLeaseFailure("global_lease_tree_not_closed_world")
        prefix = b"100644 blob "
        suffix = b"\t" + GLOBAL_LEASE_STATE_PATH.encode("ascii")
        if not entries[0].startswith(prefix) or not entries[0].endswith(suffix):
            raise _GlobalLeaseFailure("global_lease_tree_not_closed_world")

        blob = _run_git(
            repo_root,
            ["cat-file", "blob", f"{expected_tip}:{GLOBAL_LEASE_STATE_PATH}"],
        )
        if blob.returncode != 0:
            raise _GlobalLeaseFailure("global_lease_state_blob_missing")
        return blob.stdout
    finally:
        _delete_local_ref(repo_root, temporary_ref)


def _read_snapshot(
    repo_root: str | Path,
    remote: str,
    manifest: Mapping[str, Any],
) -> GlobalLeaseSnapshot | None:
    ref = global_training_run_lease_ref(manifest)
    tip = _remote_tip(repo_root, remote, ref)
    if tip is None:
        return None
    raw = _fetch_remote_commit(repo_root, remote, ref, tip)
    try:
        state = decode_global_lease_state(raw, manifest)
    except ValueError as exc:
        raise _GlobalLeaseFailure("global_lease_remote_state_invalid") from exc
    return GlobalLeaseSnapshot(
        ref=ref,
        remote_tip=tip,
        launch_manifest_sha256=str(state["launch_manifest_sha256"]),
        lease=dict(state["lease"]),
    )


def _write_state_commit(
    repo_root: str | Path,
    state: Mapping[str, Any],
    *,
    parent_tip: str | None,
    operation: str,
) -> str:
    state_bytes = canonical_json_bytes(state)
    blob = _run_git(repo_root, ["hash-object", "-w", "--stdin"], input_bytes=state_bytes)
    if blob.returncode != 0:
        raise _GlobalLeaseFailure("git_hash_object_failed")
    try:
        blob_sha = blob.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise _GlobalLeaseFailure("git_hash_object_output_invalid") from exc
    if _GIT_SHA.fullmatch(blob_sha) is None:
        raise _GlobalLeaseFailure("git_hash_object_output_invalid")

    tree_input = f"100644 blob {blob_sha}\t{GLOBAL_LEASE_STATE_PATH}\n".encode("ascii")
    tree = _run_git(repo_root, ["mktree"], input_bytes=tree_input)
    if tree.returncode != 0:
        raise _GlobalLeaseFailure("git_mktree_failed")
    try:
        tree_sha = tree.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise _GlobalLeaseFailure("git_mktree_output_invalid") from exc
    if _GIT_SHA.fullmatch(tree_sha) is None:
        raise _GlobalLeaseFailure("git_mktree_output_invalid")

    message = (
        f"ts6 global TRAINING_RUN lease {operation}\n\n"
        f"attempt-nonce: {secrets.token_hex(16)}\n"
    )
    args = ["commit-tree", tree_sha]
    if parent_tip is not None:
        args.extend(["-p", parent_tip])
    commit = _run_git(repo_root, args, input_bytes=message.encode("ascii"))
    if commit.returncode != 0:
        raise _GlobalLeaseFailure("git_commit_tree_failed")
    try:
        commit_sha = commit.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise _GlobalLeaseFailure("git_commit_tree_output_invalid") from exc
    if _GIT_SHA.fullmatch(commit_sha) is None:
        raise _GlobalLeaseFailure("git_commit_tree_output_invalid")
    return commit_sha


def _push_candidate(
    repo_root: str | Path,
    remote: str,
    candidate_tip: str,
    ref: str,
) -> bool:
    pushed = _run_git(
        repo_root,
        ["push", "--porcelain", "--", remote, f"{candidate_tip}:{ref}"],
    )
    return pushed.returncode == 0


def _operation_failure(
    operation: str,
    ref: str,
    manifest_sha256: str | None,
    *,
    blocker: str,
    expected_remote_tip: str | None = None,
    observed_remote_tip: str | None = None,
    run_id: str | None = None,
    lease_status: str | None = None,
) -> GlobalLeaseOperation:
    return GlobalLeaseOperation(
        operation=operation,
        committed=False,
        post_write_reread_verified=False,
        ref=ref,
        launch_manifest_sha256=manifest_sha256,
        expected_remote_tip=expected_remote_tip,
        observed_remote_tip=observed_remote_tip,
        written_remote_tip=None,
        run_id=run_id,
        lease_status=lease_status,
        blockers=(blocker,),
    )


def inspect_global_training_run_lease(
    repo_root: str | Path,
    remote: str,
    manifest: Mapping[str, Any],
) -> GlobalLeaseInspection:
    _validate_transport(remote)
    try:
        ref = global_training_run_lease_ref(manifest)
        digest = launch_manifest_sha256(manifest)
    except ValueError as exc:
        return GlobalLeaseInspection(
            present=False,
            valid=False,
            ref="",
            remote_tip=None,
            launch_manifest_sha256=None,
            run_id=None,
            lease_status=None,
            renewal_sequence=None,
            blockers=(str(exc),),
        )
    try:
        snapshot = _read_snapshot(repo_root, remote, manifest)
    except _GlobalLeaseFailure as exc:
        return GlobalLeaseInspection(
            present=True,
            valid=False,
            ref=ref,
            remote_tip=None,
            launch_manifest_sha256=digest,
            run_id=None,
            lease_status=None,
            renewal_sequence=None,
            blockers=(exc.blocker,),
        )
    if snapshot is None:
        return GlobalLeaseInspection(
            present=False,
            valid=True,
            ref=ref,
            remote_tip=None,
            launch_manifest_sha256=digest,
            run_id=None,
            lease_status=None,
            renewal_sequence=None,
            blockers=("global_training_run_lease_missing",),
        )
    return GlobalLeaseInspection(
        present=True,
        valid=True,
        ref=ref,
        remote_tip=snapshot.remote_tip,
        launch_manifest_sha256=snapshot.launch_manifest_sha256,
        run_id=str(snapshot.lease["run_id"]),
        lease_status=str(snapshot.lease["status"]),
        renewal_sequence=int(snapshot.lease["renewal_sequence"]),
        blockers=(),
    )


def acquire_global_training_run_lease(
    repo_root: str | Path,
    remote: str,
    manifest: Mapping[str, Any],
    lease: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> GlobalLeaseOperation:
    """Atomically create the manifest-derived remote ref once, never overwrite it."""
    _validate_transport(remote)
    try:
        ref = global_training_run_lease_ref(manifest)
        digest = launch_manifest_sha256(manifest)
    except ValueError as exc:
        return _operation_failure("ACQUIRE", "", None, blocker=str(exc))
    run_id = str(lease.get("run_id", "")) if isinstance(lease, Mapping) else None
    status = str(lease.get("status", "")) if isinstance(lease, Mapping) else None
    assessment = assess_training_run_lease(manifest, lease, now=_normalize_now(now))
    blockers = tuple(dict.fromkeys((*assessment.contract_errors, *assessment.blockers)))
    if blockers:
        return _operation_failure(
            "ACQUIRE", ref, digest, blocker=blockers[0], run_id=run_id, lease_status=status
        )
    if (
        lease.get("status") != "RUNNING"
        or lease.get("renewal_sequence") != 0
        or lease.get("terminal_at_utc") is not None
        or lease.get("acquired_at_utc") != lease.get("renewed_at_utc")
    ):
        return _operation_failure(
            "ACQUIRE",
            ref,
            digest,
            blocker="global_lease_initial_state_invalid",
            run_id=run_id,
            lease_status=status,
        )
    try:
        existing_tip = _remote_tip(repo_root, remote, ref)
        if existing_tip is not None:
            return _operation_failure(
                "ACQUIRE",
                ref,
                digest,
                blocker="global_training_run_lease_already_exists",
                observed_remote_tip=existing_tip,
                run_id=run_id,
                lease_status=status,
            )
        state = build_global_lease_state(manifest, lease)
        candidate_tip = _write_state_commit(
            repo_root, state, parent_tip=None, operation="acquire"
        )
        if not _push_candidate(repo_root, remote, candidate_tip, ref):
            observed = _remote_tip(repo_root, remote, ref)
            return _operation_failure(
                "ACQUIRE",
                ref,
                digest,
                blocker="global_lease_ref_create_rejected",
                observed_remote_tip=observed,
                run_id=run_id,
                lease_status=status,
            )
        observed_after = _remote_tip(repo_root, remote, ref)
        if observed_after != candidate_tip:
            return _operation_failure(
                "ACQUIRE",
                ref,
                digest,
                blocker="global_lease_post_write_tip_mismatch",
                observed_remote_tip=observed_after,
                run_id=run_id,
                lease_status=status,
            )
        reread = _read_snapshot(repo_root, remote, manifest)
        if reread is None or reread.remote_tip != candidate_tip or reread.lease != dict(lease):
            return _operation_failure(
                "ACQUIRE",
                ref,
                digest,
                blocker="global_lease_post_write_reread_mismatch",
                observed_remote_tip=None if reread is None else reread.remote_tip,
                run_id=run_id,
                lease_status=status,
            )
    except _GlobalLeaseFailure as exc:
        return _operation_failure(
            "ACQUIRE", ref, digest, blocker=exc.blocker, run_id=run_id, lease_status=status
        )
    return GlobalLeaseOperation(
        operation="ACQUIRE",
        committed=True,
        post_write_reread_verified=True,
        ref=ref,
        launch_manifest_sha256=digest,
        expected_remote_tip=None,
        observed_remote_tip=candidate_tip,
        written_remote_tip=candidate_tip,
        run_id=run_id,
        lease_status=status,
        blockers=(),
        cooperative_git_ref_cas_mechanics_verified=True,
    )


def _transition_global_training_run_lease(
    repo_root: str | Path,
    remote: str,
    manifest: Mapping[str, Any],
    *,
    expected_remote_tip: str,
    operation: str,
    ttl_seconds: int | None = None,
    terminal_status: str | None = None,
    now: datetime | None = None,
) -> GlobalLeaseOperation:
    _validate_transport(remote)
    try:
        ref = global_training_run_lease_ref(manifest)
        digest = launch_manifest_sha256(manifest)
    except ValueError as exc:
        return _operation_failure(operation, "", None, blocker=str(exc))
    if _GIT_SHA.fullmatch(expected_remote_tip) is None:
        return _operation_failure(
            operation, ref, digest, blocker="expected_remote_tip_invalid"
        )
    try:
        snapshot = _read_snapshot(repo_root, remote, manifest)
        if snapshot is None:
            return _operation_failure(
                operation,
                ref,
                digest,
                blocker="global_training_run_lease_missing",
                expected_remote_tip=expected_remote_tip,
            )
        if snapshot.remote_tip != expected_remote_tip:
            return _operation_failure(
                operation,
                ref,
                digest,
                blocker="global_lease_expected_tip_mismatch",
                expected_remote_tip=expected_remote_tip,
                observed_remote_tip=snapshot.remote_tip,
                run_id=str(snapshot.lease["run_id"]),
                lease_status=str(snapshot.lease["status"]),
            )
        previous = TrainingLease(**snapshot.lease)
        current = _normalize_now(now)
        if operation == "RENEW":
            if ttl_seconds is None:
                return _operation_failure(
                    operation,
                    ref,
                    digest,
                    blocker="ttl_seconds_missing",
                    expected_remote_tip=expected_remote_tip,
                    observed_remote_tip=snapshot.remote_tip,
                    run_id=previous.run_id,
                    lease_status=previous.status,
                )
            candidate = renew_training_run_lease(previous, ttl_seconds=ttl_seconds, now=current)
        elif operation == "TERMINATE":
            if terminal_status is None:
                return _operation_failure(
                    operation,
                    ref,
                    digest,
                    blocker="terminal_status_missing",
                    expected_remote_tip=expected_remote_tip,
                    observed_remote_tip=snapshot.remote_tip,
                    run_id=previous.run_id,
                    lease_status=previous.status,
                )
            candidate = terminate_training_run_lease(previous, status=terminal_status, now=current)
        else:
            raise AssertionError("unreachable operation")
        transition_errors = validate_lease_transition(
            previous.as_dict(), candidate.as_dict(), manifest
        )
        if transition_errors:
            return _operation_failure(
                operation,
                ref,
                digest,
                blocker=transition_errors[0],
                expected_remote_tip=expected_remote_tip,
                observed_remote_tip=snapshot.remote_tip,
                run_id=previous.run_id,
                lease_status=previous.status,
            )
        state = build_global_lease_state(manifest, candidate.as_dict())
        candidate_tip = _write_state_commit(
            repo_root,
            state,
            parent_tip=snapshot.remote_tip,
            operation=operation.lower(),
        )
        if not _push_candidate(repo_root, remote, candidate_tip, ref):
            observed = _remote_tip(repo_root, remote, ref)
            return _operation_failure(
                operation,
                ref,
                digest,
                blocker="global_lease_transition_push_rejected",
                expected_remote_tip=expected_remote_tip,
                observed_remote_tip=observed,
                run_id=previous.run_id,
                lease_status=previous.status,
            )
        observed_after = _remote_tip(repo_root, remote, ref)
        if observed_after != candidate_tip:
            return _operation_failure(
                operation,
                ref,
                digest,
                blocker="global_lease_post_write_tip_mismatch",
                expected_remote_tip=expected_remote_tip,
                observed_remote_tip=observed_after,
                run_id=candidate.run_id,
                lease_status=candidate.status,
            )
        reread = _read_snapshot(repo_root, remote, manifest)
        if (
            reread is None
            or reread.remote_tip != candidate_tip
            or reread.lease != candidate.as_dict()
        ):
            return _operation_failure(
                operation,
                ref,
                digest,
                blocker="global_lease_post_write_reread_mismatch",
                expected_remote_tip=expected_remote_tip,
                observed_remote_tip=None if reread is None else reread.remote_tip,
                run_id=candidate.run_id,
                lease_status=candidate.status,
            )
    except (ValueError, TypeError) as exc:
        return _operation_failure(
            operation,
            ref,
            digest,
            blocker=f"global_lease_transition_invalid:{exc}",
            expected_remote_tip=expected_remote_tip,
        )
    except _GlobalLeaseFailure as exc:
        return _operation_failure(
            operation,
            ref,
            digest,
            blocker=exc.blocker,
            expected_remote_tip=expected_remote_tip,
        )
    return GlobalLeaseOperation(
        operation=operation,
        committed=True,
        post_write_reread_verified=True,
        ref=ref,
        launch_manifest_sha256=digest,
        expected_remote_tip=expected_remote_tip,
        observed_remote_tip=candidate_tip,
        written_remote_tip=candidate_tip,
        run_id=candidate.run_id,
        lease_status=candidate.status,
        blockers=(),
        cooperative_git_ref_cas_mechanics_verified=True,
    )


def renew_global_training_run_lease(
    repo_root: str | Path,
    remote: str,
    manifest: Mapping[str, Any],
    *,
    expected_remote_tip: str,
    ttl_seconds: int,
    now: datetime | None = None,
) -> GlobalLeaseOperation:
    return _transition_global_training_run_lease(
        repo_root,
        remote,
        manifest,
        expected_remote_tip=expected_remote_tip,
        operation="RENEW",
        ttl_seconds=ttl_seconds,
        now=now,
    )


def terminate_global_training_run_lease(
    repo_root: str | Path,
    remote: str,
    manifest: Mapping[str, Any],
    *,
    expected_remote_tip: str,
    status: str,
    now: datetime | None = None,
) -> GlobalLeaseOperation:
    return _transition_global_training_run_lease(
        repo_root,
        remote,
        manifest,
        expected_remote_tip=expected_remote_tip,
        operation="TERMINATE",
        terminal_status=status,
        now=now,
    )
