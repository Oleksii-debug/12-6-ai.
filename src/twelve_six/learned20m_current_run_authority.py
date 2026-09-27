"""Fixed current-run authority pointer for learned-20M launch safety.

This module does not create training, optimizer, corpus, exposure, or global
exclusivity authority.  It selects exactly one already-running canonical
TRAINING_RUN through a fixed repository ref so consumers such as Windows
safe-stop do not let candidate bytes select their own trust namespace.
"""

from __future__ import annotations

import json
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from twelve_six.learned20m_global_training_lease import (
    CANONICAL_LOCK_DOMAIN,
    CANONICAL_REPOSITORY,
    GlobalLeaseInspection,
    _GlobalLeaseFailure,
    _delete_local_ref,
    _read_snapshot,
    _remote_tip,
    _run_git,
    _validate_transport,
    global_training_run_lease_ref,
    inspect_global_training_run_lease,
)
from twelve_six.learned20m_training_lease import (
    assess_training_run_lease,
    canonical_json_bytes,
    launch_manifest_sha256,
    validate_launch_manifest,
)

CURRENT_RUN_IDENTITY_SCHEMA = "R01-LEARNED20M-CURRENT-RUN-IDENTITY-V1"
CURRENT_RUN_POINTER_SCHEMA = "R01-LEARNED20M-CURRENT-RUN-POINTER-V1"
CURRENT_RUN_POINTER_REF = "refs/heads/ts6-current-training-run-v1"
CURRENT_RUN_POINTER_PATH = "current-training-run-v1.json"
MECHANICS_SCOPE = "FIXED_REPOSITORY_REF_CURRENT_RUN_SELECTION_ONLY"

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_IDENTITY_FIELDS = {
    "schema",
    "identity_sha256",
    "run_id",
    "recovery_run_manifest_sha256",
    "recovery_attempt_authority_sha256",
    "portable_run_binding_sha256",
    "source_git_sha",
}
_POINTER_FIELDS = {
    "schema",
    "pointer_identity_sha256",
    "repository",
    "lock_domain",
    "generation",
    "status",
    "launch_manifest_sha256",
    "global_lease_ref",
    "global_lease_remote_tip",
    "current_run_identity",
}
_POINTER_STATUSES = {"ACTIVE", "RETIRED"}


class CurrentRunAuthorityError(RuntimeError):
    """Fail-closed current-run authority error."""


@dataclass(frozen=True)
class CurrentRunAuthorityInspection:
    present: bool
    valid: bool
    active: bool
    ref: str
    remote_tip: str | None
    generation: int | None
    pointer_identity_sha256: str | None
    launch_manifest_sha256: str | None
    global_lease_ref: str | None
    global_lease_remote_tip: str | None
    run_id: str | None
    recovery_run_manifest_sha256: str | None
    current_run_identity_sha256: str | None
    blockers: tuple[str, ...]
    mechanics_scope: str = MECHANICS_SCOPE
    optimizer_start_permitted_by_this_module: bool = False
    training_authority_granted_by_this_module: bool = False
    scientific_truth_changed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class CurrentRunAuthorityOperation:
    operation: str
    committed: bool
    post_write_reread_verified: bool
    ref: str
    expected_remote_tip: str | None
    observed_remote_tip: str | None
    written_remote_tip: str | None
    generation: int | None
    run_id: str | None
    current_run_identity_sha256: str | None
    blockers: tuple[str, ...]
    mechanics_scope: str = MECHANICS_SCOPE
    optimizer_start_permitted_by_this_module: bool = False
    training_authority_granted_by_this_module: bool = False
    scientific_truth_changed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _git_sha(value: Any) -> bool:
    return isinstance(value, str) and _GIT_SHA.fullmatch(value) is not None


def _token(value: Any) -> bool:
    return isinstance(value, str) and _TOKEN.fullmatch(value) is not None


def _exact_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _identity_digest(identity: Mapping[str, Any]) -> str:
    unsigned = dict(identity)
    unsigned.pop("identity_sha256", None)
    import hashlib

    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def _pointer_digest(state: Mapping[str, Any]) -> str:
    unsigned = dict(state)
    unsigned.pop("pointer_identity_sha256", None)
    import hashlib

    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


def build_current_run_identity(
    *,
    run_id: str,
    recovery_run_manifest_sha256: str,
    recovery_attempt_authority_sha256: str,
    portable_run_binding_sha256: str,
    source_git_sha: str,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "schema": CURRENT_RUN_IDENTITY_SCHEMA,
        "identity_sha256": "0" * 64,
        "run_id": run_id,
        "recovery_run_manifest_sha256": recovery_run_manifest_sha256,
        "recovery_attempt_authority_sha256": recovery_attempt_authority_sha256,
        "portable_run_binding_sha256": portable_run_binding_sha256,
        "source_git_sha": source_git_sha,
    }
    errors = validate_current_run_identity(identity, verify_self_hash=False)
    if errors:
        raise ValueError("invalid_current_run_identity:" + ";".join(errors))
    identity["identity_sha256"] = _identity_digest(identity)
    return identity


def validate_current_run_identity(
    identity: Mapping[str, Any],
    *,
    verify_self_hash: bool = True,
) -> tuple[str, ...]:
    errors: list[str] = []
    if set(identity) != _IDENTITY_FIELDS:
        errors.append("current_run_identity_fields_mismatch")
    if identity.get("schema") != CURRENT_RUN_IDENTITY_SCHEMA:
        errors.append("current_run_identity_schema_mismatch")
    if not _token(identity.get("run_id")):
        errors.append("current_run_id_invalid")
    for field in (
        "recovery_run_manifest_sha256",
        "recovery_attempt_authority_sha256",
        "portable_run_binding_sha256",
    ):
        if not _sha256(identity.get(field)):
            errors.append(f"{field}_invalid")
    if not _git_sha(identity.get("source_git_sha")):
        errors.append("source_git_sha_invalid")
    if not _sha256(identity.get("identity_sha256")):
        errors.append("current_run_identity_sha256_invalid")
    elif verify_self_hash and _identity_digest(identity) != identity.get("identity_sha256"):
        errors.append("current_run_identity_self_hash_mismatch")
    return tuple(dict.fromkeys(errors))


def build_current_run_pointer_state(
    manifest: Mapping[str, Any],
    global_lease: GlobalLeaseInspection,
    current_run_identity: Mapping[str, Any],
    *,
    generation: int,
    status: str = "ACTIVE",
) -> dict[str, Any]:
    manifest_errors = validate_launch_manifest(manifest)
    if manifest_errors:
        raise ValueError("invalid_launch_manifest:" + ";".join(manifest_errors))
    identity_errors = validate_current_run_identity(current_run_identity)
    if identity_errors:
        raise ValueError("invalid_current_run_identity:" + ";".join(identity_errors))
    if not _exact_positive_int(generation):
        raise ValueError("generation_must_be_positive_integer")
    if status not in _POINTER_STATUSES:
        raise ValueError("current_run_pointer_status_invalid")
    if not global_lease.present or not global_lease.valid:
        raise ValueError("global_training_run_lease_not_valid")
    if global_lease.lease_status != "RUNNING":
        raise ValueError("global_training_run_lease_not_running")
    digest = launch_manifest_sha256(manifest)
    expected_ref = global_training_run_lease_ref(manifest)
    if global_lease.launch_manifest_sha256 != digest:
        raise ValueError("global_lease_manifest_sha256_mismatch")
    if global_lease.ref != expected_ref:
        raise ValueError("global_lease_ref_mismatch")
    if not _git_sha(global_lease.remote_tip):
        raise ValueError("global_lease_remote_tip_invalid")
    if global_lease.run_id != current_run_identity.get("run_id"):
        raise ValueError("global_lease_run_id_mismatch")
    identities = manifest.get("identities")
    if not isinstance(identities, Mapping):
        raise ValueError("manifest_identities_missing")
    if identities.get("source_git_sha") != current_run_identity.get("source_git_sha"):
        raise ValueError("current_run_source_git_sha_mismatch")
    if identities.get("portable_run_binding_sha256") != current_run_identity.get(
        "portable_run_binding_sha256"
    ):
        raise ValueError("current_run_portable_binding_mismatch")

    state: dict[str, Any] = {
        "schema": CURRENT_RUN_POINTER_SCHEMA,
        "pointer_identity_sha256": "0" * 64,
        "repository": CANONICAL_REPOSITORY,
        "lock_domain": CANONICAL_LOCK_DOMAIN,
        "generation": generation,
        "status": status,
        "launch_manifest_sha256": digest,
        "global_lease_ref": expected_ref,
        "global_lease_remote_tip": global_lease.remote_tip,
        "current_run_identity": dict(current_run_identity),
    }
    state["pointer_identity_sha256"] = _pointer_digest(state)
    errors = validate_current_run_pointer_state(state)
    if errors:
        raise ValueError("invalid_current_run_pointer_state:" + ";".join(errors))
    return state


def validate_current_run_pointer_state(state: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if set(state) != _POINTER_FIELDS:
        errors.append("current_run_pointer_fields_mismatch")
    if state.get("schema") != CURRENT_RUN_POINTER_SCHEMA:
        errors.append("current_run_pointer_schema_mismatch")
    if state.get("repository") != CANONICAL_REPOSITORY:
        errors.append("current_run_pointer_repository_mismatch")
    if state.get("lock_domain") != CANONICAL_LOCK_DOMAIN:
        errors.append("current_run_pointer_lock_domain_mismatch")
    if not _exact_positive_int(state.get("generation")):
        errors.append("current_run_pointer_generation_invalid")
    if state.get("status") not in _POINTER_STATUSES:
        errors.append("current_run_pointer_status_invalid")
    if not _sha256(state.get("launch_manifest_sha256")):
        errors.append("current_run_pointer_manifest_sha256_invalid")
    ref = state.get("global_lease_ref")
    if not isinstance(ref, str) or not ref.startswith(
        "refs/heads/ts6-training-run-lease-v1/"
    ):
        errors.append("current_run_pointer_global_lease_ref_invalid")
    if not _git_sha(state.get("global_lease_remote_tip")):
        errors.append("current_run_pointer_global_lease_tip_invalid")
    identity = state.get("current_run_identity")
    if not isinstance(identity, Mapping):
        errors.append("current_run_identity_missing_or_not_object")
    else:
        errors.extend(validate_current_run_identity(identity))
    if not _sha256(state.get("pointer_identity_sha256")):
        errors.append("current_run_pointer_identity_sha256_invalid")
    elif _pointer_digest(state) != state.get("pointer_identity_sha256"):
        errors.append("current_run_pointer_self_hash_mismatch")
    return tuple(dict.fromkeys(errors))


def decode_current_run_pointer_state(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        parsed = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("current_run_pointer_json_invalid") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError("current_run_pointer_not_object")
    canonical = canonical_json_bytes(parsed)
    if canonical != raw:
        raise ValueError("current_run_pointer_not_canonical")
    errors = validate_current_run_pointer_state(parsed)
    if errors:
        raise ValueError("invalid_current_run_pointer_state:" + ";".join(errors))
    return dict(parsed)


def _fetch_pointer_bytes(
    repo_root: str | Path,
    remote: str,
    expected_tip: str,
) -> bytes:
    temporary_ref = f"refs/ts6-current-run-read/{secrets.token_hex(16)}"
    try:
        result = _run_git(
            repo_root,
            [
                "fetch",
                "--quiet",
                "--no-tags",
                "--",
                remote,
                f"{CURRENT_RUN_POINTER_REF}:{temporary_ref}",
            ],
        )
        if result.returncode != 0:
            raise CurrentRunAuthorityError("current_run_pointer_fetch_failed")
        if _remote_tip(repo_root, remote, CURRENT_RUN_POINTER_REF) != expected_tip:
            raise CurrentRunAuthorityError("current_run_pointer_changed_during_read")
        parents = _run_git(repo_root, ["rev-list", "--parents", "-n", "1", expected_tip])
        if parents.returncode != 0:
            raise CurrentRunAuthorityError("current_run_pointer_parent_read_failed")
        parts = parents.stdout.decode("ascii").strip().split()
        if not parts or parts[0] != expected_tip or len(parts) > 2:
            raise CurrentRunAuthorityError("current_run_pointer_parent_shape_invalid")
        tree = _run_git(repo_root, ["ls-tree", "-z", "--full-tree", expected_tip])
        if tree.returncode != 0:
            raise CurrentRunAuthorityError("current_run_pointer_tree_read_failed")
        entries = [entry for entry in tree.stdout.split(b"\x00") if entry]
        expected_suffix = b"\t" + CURRENT_RUN_POINTER_PATH.encode("ascii")
        if (
            len(entries) != 1
            or not entries[0].startswith(b"100644 blob ")
            or not entries[0].endswith(expected_suffix)
        ):
            raise CurrentRunAuthorityError("current_run_pointer_tree_not_closed_world")
        blob = _run_git(
            repo_root,
            ["cat-file", "blob", f"{expected_tip}:{CURRENT_RUN_POINTER_PATH}"],
        )
        if blob.returncode != 0:
            raise CurrentRunAuthorityError("current_run_pointer_blob_missing")
        return blob.stdout
    finally:
        _delete_local_ref(repo_root, temporary_ref)


def _read_pointer_state(
    repo_root: str | Path,
    remote: str,
) -> tuple[str, dict[str, Any]] | None:
    tip = _remote_tip(repo_root, remote, CURRENT_RUN_POINTER_REF)
    if tip is None:
        return None
    raw = _fetch_pointer_bytes(repo_root, remote, tip)
    return tip, decode_current_run_pointer_state(raw)


def inspect_current_run_authority(
    repo_root: str | Path,
    remote: str,
) -> CurrentRunAuthorityInspection:
    _validate_transport(remote)
    try:
        snapshot = _read_pointer_state(repo_root, remote)
    except (CurrentRunAuthorityError, ValueError) as exc:
        return CurrentRunAuthorityInspection(
            present=True,
            valid=False,
            active=False,
            ref=CURRENT_RUN_POINTER_REF,
            remote_tip=None,
            generation=None,
            pointer_identity_sha256=None,
            launch_manifest_sha256=None,
            global_lease_ref=None,
            global_lease_remote_tip=None,
            run_id=None,
            recovery_run_manifest_sha256=None,
            current_run_identity_sha256=None,
            blockers=(str(exc),),
        )
    if snapshot is None:
        return CurrentRunAuthorityInspection(
            present=False,
            valid=False,
            active=False,
            ref=CURRENT_RUN_POINTER_REF,
            remote_tip=None,
            generation=None,
            pointer_identity_sha256=None,
            launch_manifest_sha256=None,
            global_lease_ref=None,
            global_lease_remote_tip=None,
            run_id=None,
            recovery_run_manifest_sha256=None,
            current_run_identity_sha256=None,
            blockers=("current_run_pointer_missing",),
        )
    tip, state = snapshot
    identity = state["current_run_identity"]
    return CurrentRunAuthorityInspection(
        present=True,
        valid=True,
        active=state["status"] == "ACTIVE",
        ref=CURRENT_RUN_POINTER_REF,
        remote_tip=tip,
        generation=int(state["generation"]),
        pointer_identity_sha256=str(state["pointer_identity_sha256"]),
        launch_manifest_sha256=str(state["launch_manifest_sha256"]),
        global_lease_ref=str(state["global_lease_ref"]),
        global_lease_remote_tip=str(state["global_lease_remote_tip"]),
        run_id=str(identity["run_id"]),
        recovery_run_manifest_sha256=str(identity["recovery_run_manifest_sha256"]),
        current_run_identity_sha256=str(identity["identity_sha256"]),
        blockers=(),
    )


def verify_candidate_against_current_run(
    inspection: CurrentRunAuthorityInspection,
    *,
    launch_manifest_sha256_value: str,
    run_id: str,
    recovery_run_manifest_sha256: str,
    current_run_identity_sha256: str,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if not inspection.present or not inspection.valid or not inspection.active:
        blockers.append("current_run_authority_not_active")
        return tuple(blockers)
    pairs = (
        (
            "current_run_launch_manifest_mismatch",
            launch_manifest_sha256_value,
            inspection.launch_manifest_sha256,
        ),
        ("current_run_id_mismatch", run_id, inspection.run_id),
        (
            "current_run_recovery_manifest_mismatch",
            recovery_run_manifest_sha256,
            inspection.recovery_run_manifest_sha256,
        ),
        (
            "current_run_identity_mismatch",
            current_run_identity_sha256,
            inspection.current_run_identity_sha256,
        ),
    )
    for blocker, candidate, expected in pairs:
        if candidate != expected:
            blockers.append(blocker)
    return tuple(blockers)


def _write_pointer_commit(
    repo_root: str | Path,
    state: Mapping[str, Any],
    *,
    parent_tip: str | None,
    operation: str,
) -> str:
    state_bytes = canonical_json_bytes(state)
    blob = _run_git(repo_root, ["hash-object", "-w", "--stdin"], input_bytes=state_bytes)
    if blob.returncode != 0:
        raise CurrentRunAuthorityError("current_run_pointer_hash_object_failed")
    blob_sha = blob.stdout.decode("ascii").strip()
    if not _git_sha(blob_sha):
        raise CurrentRunAuthorityError("current_run_pointer_blob_sha_invalid")
    tree_input = f"100644 blob {blob_sha}\t{CURRENT_RUN_POINTER_PATH}\n".encode("ascii")
    tree = _run_git(repo_root, ["mktree"], input_bytes=tree_input)
    if tree.returncode != 0:
        raise CurrentRunAuthorityError("current_run_pointer_mktree_failed")
    tree_sha = tree.stdout.decode("ascii").strip()
    if not _git_sha(tree_sha):
        raise CurrentRunAuthorityError("current_run_pointer_tree_sha_invalid")
    args = ["commit-tree", tree_sha]
    if parent_tip is not None:
        args.extend(["-p", parent_tip])
    message = (
        f"ts6 current learned20m run {operation}\n\n"
        f"attempt-nonce: {secrets.token_hex(16)}\n"
    ).encode("ascii")
    commit = _run_git(repo_root, args, input_bytes=message)
    if commit.returncode != 0:
        raise CurrentRunAuthorityError("current_run_pointer_commit_tree_failed")
    commit_sha = commit.stdout.decode("ascii").strip()
    if not _git_sha(commit_sha):
        raise CurrentRunAuthorityError("current_run_pointer_commit_sha_invalid")
    return commit_sha


def _operation_failure(
    operation: str,
    *,
    blocker: str,
    expected_remote_tip: str | None = None,
    observed_remote_tip: str | None = None,
    generation: int | None = None,
    run_id: str | None = None,
    identity_sha256: str | None = None,
) -> CurrentRunAuthorityOperation:
    return CurrentRunAuthorityOperation(
        operation=operation,
        committed=False,
        post_write_reread_verified=False,
        ref=CURRENT_RUN_POINTER_REF,
        expected_remote_tip=expected_remote_tip,
        observed_remote_tip=observed_remote_tip,
        written_remote_tip=None,
        generation=generation,
        run_id=run_id,
        current_run_identity_sha256=identity_sha256,
        blockers=(blocker,),
    )


def activate_current_run_authority(
    repo_root: str | Path,
    remote: str,
    manifest: Mapping[str, Any],
    current_run_identity: Mapping[str, Any],
    *,
    expected_pointer_tip: str | None,
    now: datetime | None = None,
) -> CurrentRunAuthorityOperation:
    _validate_transport(remote)
    manifest_snapshot = json.loads(canonical_json_bytes(manifest))
    identity_snapshot = json.loads(canonical_json_bytes(current_run_identity))
    identity_errors = validate_current_run_identity(identity_snapshot)
    if identity_errors:
        return _operation_failure(
            "ACTIVATE",
            blocker="invalid_current_run_identity:" + ";".join(identity_errors),
        )
    global_lease = inspect_global_training_run_lease(
        repo_root, remote, manifest_snapshot
    )
    if not global_lease.present or not global_lease.valid:
        return _operation_failure("ACTIVATE", blocker="global_training_run_lease_not_valid")
    if global_lease.lease_status != "RUNNING":
        return _operation_failure("ACTIVATE", blocker="global_training_run_lease_not_running")
    try:
        global_snapshot = _read_snapshot(repo_root, remote, manifest_snapshot)
    except _GlobalLeaseFailure as exc:
        return _operation_failure("ACTIVATE", blocker=exc.blocker)
    if global_snapshot is None:
        return _operation_failure("ACTIVATE", blocker="global_training_run_lease_missing")
    lease_assessment = assess_training_run_lease(
        manifest_snapshot,
        global_snapshot.lease,
        now=now,
    )
    lease_blockers = tuple(
        dict.fromkeys((*lease_assessment.contract_errors, *lease_assessment.blockers))
    )
    if lease_blockers:
        return _operation_failure("ACTIVATE", blocker=lease_blockers[0])
    if global_snapshot.remote_tip != global_lease.remote_tip:
        return _operation_failure("ACTIVATE", blocker="global_training_run_lease_tip_mismatch")

    try:
        current = _read_pointer_state(repo_root, remote)
    except (CurrentRunAuthorityError, ValueError) as exc:
        return _operation_failure("ACTIVATE", blocker=str(exc))
    observed_tip = None if current is None else current[0]
    if observed_tip != expected_pointer_tip:
        return _operation_failure(
            "ACTIVATE",
            blocker="current_run_pointer_expected_tip_mismatch",
            expected_remote_tip=expected_pointer_tip,
            observed_remote_tip=observed_tip,
        )
    if current is not None and current[1]["status"] == "ACTIVE":
        return _operation_failure(
            "ACTIVATE",
            blocker="current_run_already_active",
            expected_remote_tip=expected_pointer_tip,
            observed_remote_tip=observed_tip,
            generation=int(current[1]["generation"]),
            run_id=str(current[1]["current_run_identity"]["run_id"]),
            identity_sha256=str(current[1]["current_run_identity"]["identity_sha256"]),
        )

    generation = 1 if current is None else int(current[1]["generation"]) + 1
    try:
        state = build_current_run_pointer_state(
            manifest_snapshot,
            global_lease,
            identity_snapshot,
            generation=generation,
        )
        candidate_tip = _write_pointer_commit(
            repo_root,
            state,
            parent_tip=observed_tip,
            operation="ACTIVATE",
        )
    except (CurrentRunAuthorityError, TypeError, ValueError) as exc:
        return _operation_failure(
            "ACTIVATE",
            blocker=str(exc),
            expected_remote_tip=expected_pointer_tip,
            observed_remote_tip=observed_tip,
            generation=generation,
            run_id=str(identity_snapshot.get("run_id", "")),
            identity_sha256=str(identity_snapshot.get("identity_sha256", "")),
        )

    pushed = _run_git(
        repo_root,
        [
            "push",
            "--porcelain",
            "--",
            remote,
            f"{candidate_tip}:{CURRENT_RUN_POINTER_REF}",
        ],
    )
    observed_after = _remote_tip(repo_root, remote, CURRENT_RUN_POINTER_REF)
    if pushed.returncode != 0 or observed_after != candidate_tip:
        return _operation_failure(
            "ACTIVATE",
            blocker="current_run_pointer_cas_conflict",
            expected_remote_tip=expected_pointer_tip,
            observed_remote_tip=observed_after,
            generation=generation,
            run_id=str(identity_snapshot["run_id"]),
            identity_sha256=str(identity_snapshot["identity_sha256"]),
        )
    reread = inspect_current_run_authority(repo_root, remote)
    verified = (
        reread.valid
        and reread.active
        and reread.remote_tip == candidate_tip
        and reread.generation == generation
        and reread.current_run_identity_sha256 == identity_snapshot["identity_sha256"]
    )
    return CurrentRunAuthorityOperation(
        operation="ACTIVATE",
        committed=True,
        post_write_reread_verified=verified,
        ref=CURRENT_RUN_POINTER_REF,
        expected_remote_tip=expected_pointer_tip,
        observed_remote_tip=observed_after,
        written_remote_tip=candidate_tip,
        generation=generation,
        run_id=str(identity_snapshot["run_id"]),
        current_run_identity_sha256=str(identity_snapshot["identity_sha256"]),
        blockers=() if verified else ("current_run_pointer_post_write_reread_mismatch",),
    )


def retire_current_run_authority(
    repo_root: str | Path,
    remote: str,
    *,
    expected_pointer_tip: str,
    expected_current_run_identity_sha256: str,
) -> CurrentRunAuthorityOperation:
    _validate_transport(remote)
    if not _git_sha(expected_pointer_tip):
        return _operation_failure("RETIRE", blocker="expected_pointer_tip_invalid")
    if not _sha256(expected_current_run_identity_sha256):
        return _operation_failure("RETIRE", blocker="expected_current_run_identity_invalid")
    try:
        current = _read_pointer_state(repo_root, remote)
    except (CurrentRunAuthorityError, ValueError) as exc:
        return _operation_failure("RETIRE", blocker=str(exc))
    if current is None:
        return _operation_failure("RETIRE", blocker="current_run_pointer_missing")
    observed_tip, state = current
    identity = state["current_run_identity"]
    if observed_tip != expected_pointer_tip:
        return _operation_failure(
            "RETIRE",
            blocker="current_run_pointer_expected_tip_mismatch",
            expected_remote_tip=expected_pointer_tip,
            observed_remote_tip=observed_tip,
        )
    if state["status"] != "ACTIVE":
        return _operation_failure(
            "RETIRE",
            blocker="current_run_pointer_not_active",
            expected_remote_tip=expected_pointer_tip,
            observed_remote_tip=observed_tip,
        )
    if identity["identity_sha256"] != expected_current_run_identity_sha256:
        return _operation_failure(
            "RETIRE",
            blocker="current_run_identity_mismatch",
            expected_remote_tip=expected_pointer_tip,
            observed_remote_tip=observed_tip,
        )
    retired = dict(state)
    retired["status"] = "RETIRED"
    retired["pointer_identity_sha256"] = _pointer_digest(retired)
    candidate_tip = _write_pointer_commit(
        repo_root,
        retired,
        parent_tip=observed_tip,
        operation="RETIRE",
    )
    pushed = _run_git(
        repo_root,
        [
            "push",
            "--porcelain",
            "--",
            remote,
            f"{candidate_tip}:{CURRENT_RUN_POINTER_REF}",
        ],
    )
    observed_after = _remote_tip(repo_root, remote, CURRENT_RUN_POINTER_REF)
    if pushed.returncode != 0 or observed_after != candidate_tip:
        return _operation_failure(
            "RETIRE",
            blocker="current_run_pointer_cas_conflict",
            expected_remote_tip=expected_pointer_tip,
            observed_remote_tip=observed_after,
            generation=int(state["generation"]),
            run_id=str(identity["run_id"]),
            identity_sha256=str(identity["identity_sha256"]),
        )
    reread = inspect_current_run_authority(repo_root, remote)
    verified = (
        reread.valid
        and not reread.active
        and reread.remote_tip == candidate_tip
        and reread.current_run_identity_sha256 == expected_current_run_identity_sha256
    )
    return CurrentRunAuthorityOperation(
        operation="RETIRE",
        committed=True,
        post_write_reread_verified=verified,
        ref=CURRENT_RUN_POINTER_REF,
        expected_remote_tip=expected_pointer_tip,
        observed_remote_tip=observed_after,
        written_remote_tip=candidate_tip,
        generation=int(state["generation"]),
        run_id=str(identity["run_id"]),
        current_run_identity_sha256=str(identity["identity_sha256"]),
        blockers=() if verified else ("current_run_pointer_post_write_reread_mismatch",),
    )
