"""Fail-closed learned-20M launch-manifest and local TRAINING_RUN lease contract.

The local lease primitive is atomic only for processes sharing one filesystem. It is
not a distributed/global lock and this module never grants optimizer-start authority.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

MANIFEST_ID = "R01-LEARNED20M-LAUNCH-MANIFEST-V1"
LEASE_CONTRACT_ID = "R01-LEARNED20M-TRAINING-RUN-LEASE-V1"
SCHEMA_VERSION = 1
STAGE = "LEARNED_20M"
LOCAL_ATOMICITY_SCOPE = "SINGLE_SHARED_FILESYSTEM_ONLY"
MAX_LEASE_SECONDS = 6 * 60 * 60

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_IDENTITY_HASHES = (
    "modelspec_sha256",
    "initspec_sha256",
    "tokenizer_sha256",
    "corpus_manifest_sha256",
    "split_sha256",
    "packing_sha256",
    "unique_loss_ledger_sha256",
    "training_config_sha256",
    "portable_run_packet_sha256",
    "portable_run_binding_sha256",
)
_FREE_RESOURCE_CLASSES = {"LOCAL_FREE", "GITHUB_HOSTED_FREE", "FREE_GPU"}
_TERMINAL = {"COMPLETED", "FAILED", "ABORTED"}
_STATUSES = {"RUNNING", *_TERMINAL}


@dataclass(frozen=True)
class TrainingLease:
    schema_version: int
    lease_contract_id: str
    manifest_sha256: str
    run_id: str
    holder_id: str
    status: str
    training_authority_ref: str
    training_authority_sha256: str
    compute_authority_ref: str
    compute_authority_sha256: str
    acquired_at_utc: str
    renewed_at_utc: str
    expires_at_utc: str
    renewal_sequence: int
    terminal_at_utc: str | None

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class TrainingLeaseAssessment:
    manifest_valid: bool
    lease_valid: bool
    contract_valid: bool
    active_lease_matches_manifest: bool
    local_duplicate_guard_open: bool
    manifest_sha256: str | None
    contract_errors: tuple[str, ...]
    blockers: tuple[str, ...]
    local_atomicity_scope: str = LOCAL_ATOMICITY_SCOPE
    global_exclusivity_proven: bool = False
    external_authority_verified: bool = False
    optimizer_start_permitted_by_this_module: bool = False
    scientific_truth_changed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_valid": self.manifest_valid,
            "lease_valid": self.lease_valid,
            "contract_valid": self.contract_valid,
            "active_lease_matches_manifest": self.active_lease_matches_manifest,
            "local_duplicate_guard_open": self.local_duplicate_guard_open,
            "manifest_sha256": self.manifest_sha256,
            "contract_errors": list(self.contract_errors),
            "blockers": list(self.blockers),
            "local_atomicity_scope": self.local_atomicity_scope,
            "global_exclusivity_proven": self.global_exclusivity_proven,
            "external_authority_verified": self.external_authority_verified,
            "optimizer_start_permitted_by_this_module": self.optimizer_start_permitted_by_this_module,
            "scientific_truth_changed": self.scientific_truth_changed,
        }


@dataclass(frozen=True)
class LocalLeaseAcquisition:
    acquired: bool
    path: str
    manifest_sha256: str
    run_id: str
    blockers: tuple[str, ...]
    local_atomicity_scope: str = LOCAL_ATOMICITY_SCOPE
    global_exclusivity_proven: bool = False
    scientific_truth_changed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "acquired": self.acquired,
            "path": self.path,
            "manifest_sha256": self.manifest_sha256,
            "run_id": self.run_id,
            "blockers": list(self.blockers),
            "local_atomicity_scope": self.local_atomicity_scope,
            "global_exclusivity_proven": self.global_exclusivity_proven,
            "scientific_truth_changed": self.scientific_truth_changed,
        }


def _exact_int(value: Any, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _integer(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (value > 0 if positive else value >= 0)
    )


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _text(value: Any, maximum: int = 512) -> bool:
    return isinstance(value, str) and 0 < len(value.strip()) <= maximum


def _token(value: Any) -> bool:
    return isinstance(value, str) and _TOKEN.fullmatch(value) is not None


def _enum_member(value: Any, allowed: set[str]) -> bool:
    """Return membership for untrusted JSON values without raising on lists/objects."""
    return isinstance(value, str) and value in allowed


def _expect(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def _object(value: Any, name: str, errors: list[str]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    errors.append(f"{name}_missing_or_not_object")
    return {}


def _canonical_type_errors(value: Any, path: str = "root") -> list[str]:
    if value is None or isinstance(value, (str, bool)):
        return []
    if isinstance(value, int) and not isinstance(value, bool):
        return []
    if isinstance(value, float):
        return [f"floating_point_forbidden:{path}"]
    if isinstance(value, list):
        return [
            error
            for index, child in enumerate(value)
            for error in _canonical_type_errors(child, f"{path}[{index}]")
        ]
    if isinstance(value, Mapping):
        errors: list[str] = []
        for key, child in value.items():
            if not isinstance(key, str):
                errors.append(f"non_string_key_forbidden:{path}")
            else:
                errors.extend(_canonical_type_errors(child, f"{path}.{key}"))
        return errors
    return [f"non_json_type_forbidden:{path}"]


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    errors = _canonical_type_errors(value)
    if errors:
        raise ValueError(";".join(errors))
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def launch_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def validate_launch_manifest(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    """Validate shape/bindings only; authority references are not self-authenticating."""
    errors = _canonical_type_errors(manifest)
    _expect(
        errors,
        set(manifest)
        == {
            "schema_version",
            "manifest_id",
            "stage",
            "identities",
            "recipe",
            "checkpoint",
            "evaluation",
            "resource",
            "authorities",
            "execution_backend",
        },
        "manifest_top_level_fields_mismatch",
    )
    _expect(errors, _exact_int(manifest.get("schema_version"), 1), "schema_version_mismatch")
    _expect(errors, manifest.get("manifest_id") == MANIFEST_ID, "manifest_id_mismatch")
    _expect(errors, manifest.get("stage") == STAGE, "stage_mismatch")

    identities = _object(manifest.get("identities"), "identities", errors)
    _expect(
        errors,
        set(identities) == {"source_git_sha", *_IDENTITY_HASHES},
        "identity_fields_mismatch",
    )
    _expect(
        errors,
        isinstance(identities.get("source_git_sha"), str)
        and _GIT_SHA.fullmatch(identities["source_git_sha"]) is not None,
        "source_git_sha_invalid",
    )
    for field in _IDENTITY_HASHES:
        _expect(errors, _sha256(identities.get(field)), f"{field}_invalid")

    recipe = _object(manifest.get("recipe"), "recipe", errors)
    _expect(
        errors,
        set(recipe)
        == {
            "optimizer_scheduler_precision",
            "seed",
            "target_unique_loss_positions",
            "maximum_total_exposures",
        },
        "recipe_fields_mismatch",
    )
    _expect(
        errors,
        _text(recipe.get("optimizer_scheduler_precision")),
        "optimizer_scheduler_precision_invalid",
    )
    _expect(errors, _integer(recipe.get("seed")), "seed_invalid")
    target = recipe.get("target_unique_loss_positions")
    maximum = recipe.get("maximum_total_exposures")
    _expect(errors, _integer(target, positive=True), "target_unique_loss_positions_invalid")
    _expect(errors, _integer(maximum, positive=True), "maximum_total_exposures_invalid")
    if _integer(target, positive=True) and _integer(maximum, positive=True):
        _expect(errors, maximum >= target, "maximum_total_exposures_below_target")

    checkpoint = _object(manifest.get("checkpoint"), "checkpoint", errors)
    _expect(
        errors,
        set(checkpoint) == {"lineage", "checkpoint_contract_sha256"},
        "checkpoint_fields_mismatch",
    )
    _expect(errors, _text(checkpoint.get("lineage")), "checkpoint_lineage_invalid")
    _expect(
        errors,
        _sha256(checkpoint.get("checkpoint_contract_sha256")),
        "checkpoint_contract_sha256_invalid",
    )

    evaluation = _object(manifest.get("evaluation"), "evaluation", errors)
    _expect(
        errors,
        set(evaluation) == {"firewall_sha256", "final_test_payload_access"},
        "evaluation_fields_mismatch",
    )
    _expect(
        errors,
        _sha256(evaluation.get("firewall_sha256")),
        "evaluation_firewall_sha256_invalid",
    )
    _expect(
        errors,
        evaluation.get("final_test_payload_access") is False,
        "final_test_payload_access_must_be_false",
    )

    resource = _object(manifest.get("resource"), "resource", errors)
    _expect(
        errors,
        set(resource) == {"resource_class", "maximum_cost_usd", "materially_paid"},
        "resource_fields_mismatch",
    )
    _expect(
        errors,
        _enum_member(resource.get("resource_class"), _FREE_RESOURCE_CLASSES),
        "resource_class_not_free_only",
    )
    _expect(
        errors,
        _exact_int(resource.get("maximum_cost_usd"), 0),
        "maximum_cost_usd_must_be_zero",
    )
    _expect(errors, resource.get("materially_paid") is False, "materially_paid_must_be_false")

    authorities = _object(manifest.get("authorities"), "authorities", errors)
    _expect(errors, set(authorities) == {"training", "compute"}, "authority_fields_mismatch")
    for name in ("training", "compute"):
        authority = _object(authorities.get(name), f"{name}_authority", errors)
        _expect(
            errors,
            set(authority) == {"reference", "evidence_sha256"},
            f"{name}_authority_fields_mismatch",
        )
        _expect(
            errors,
            _text(authority.get("reference")),
            f"{name}_authority_reference_invalid",
        )
        _expect(
            errors,
            _sha256(authority.get("evidence_sha256")),
            f"{name}_authority_evidence_sha256_invalid",
        )

    _expect(errors, _text(manifest.get("execution_backend")), "execution_backend_invalid")
    return tuple(dict.fromkeys(errors))


def _normalize_time(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC).replace(microsecond=0)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(UTC).replace(microsecond=0)


def _format_time(value: datetime) -> str:
    return _normalize_time(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time(value: Any, field: str, errors: list[str]) -> datetime | None:
    if not isinstance(value, str):
        errors.append(f"{field}_invalid")
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        errors.append(f"{field}_invalid")
        return None


def _authority(manifest: Mapping[str, Any], name: str) -> tuple[str, str]:
    value = manifest["authorities"][name]
    return str(value["reference"]), str(value["evidence_sha256"])


def build_training_run_lease(
    manifest: Mapping[str, Any],
    *,
    run_id: str,
    holder_id: str,
    ttl_seconds: int,
    now: datetime | None = None,
) -> TrainingLease:
    errors = validate_launch_manifest(manifest)
    if errors:
        raise ValueError("invalid launch manifest:" + ";".join(errors))
    if not _token(run_id) or not _token(holder_id):
        raise ValueError("run_or_holder_id_invalid")
    if not _integer(ttl_seconds, positive=True) or ttl_seconds > MAX_LEASE_SECONDS:
        raise ValueError("ttl_seconds_out_of_range")
    current = _normalize_time(now)
    training_ref, training_sha = _authority(manifest, "training")
    compute_ref, compute_sha = _authority(manifest, "compute")
    return TrainingLease(
        schema_version=1,
        lease_contract_id=LEASE_CONTRACT_ID,
        manifest_sha256=launch_manifest_sha256(manifest),
        run_id=run_id,
        holder_id=holder_id,
        status="RUNNING",
        training_authority_ref=training_ref,
        training_authority_sha256=training_sha,
        compute_authority_ref=compute_ref,
        compute_authority_sha256=compute_sha,
        acquired_at_utc=_format_time(current),
        renewed_at_utc=_format_time(current),
        expires_at_utc=_format_time(current + timedelta(seconds=ttl_seconds)),
        renewal_sequence=0,
        terminal_at_utc=None,
    )


def validate_training_run_lease(
    lease: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[str, ...]:
    errors: list[str] = []
    _expect(errors, set(lease) == set(TrainingLease.__dataclass_fields__), "lease_fields_mismatch")
    _expect(errors, _exact_int(lease.get("schema_version"), 1), "lease_schema_version_mismatch")
    _expect(
        errors,
        lease.get("lease_contract_id") == LEASE_CONTRACT_ID,
        "lease_contract_id_mismatch",
    )
    manifest_errors = validate_launch_manifest(manifest)
    if manifest_errors:
        errors.extend(f"manifest:{error}" for error in manifest_errors)
    else:
        _expect(
            errors,
            lease.get("manifest_sha256") == launch_manifest_sha256(manifest),
            "lease_manifest_sha256_mismatch",
        )
        for name in ("training", "compute"):
            reference, digest = _authority(manifest, name)
            _expect(
                errors,
                lease.get(f"{name}_authority_ref") == reference,
                f"{name}_authority_ref_mismatch",
            )
            _expect(
                errors,
                lease.get(f"{name}_authority_sha256") == digest,
                f"{name}_authority_sha256_mismatch",
            )
    _expect(errors, _token(lease.get("run_id")), "run_id_invalid")
    _expect(errors, _token(lease.get("holder_id")), "holder_id_invalid")
    status = lease.get("status")
    _expect(errors, _enum_member(status, _STATUSES), "lease_status_invalid")
    _expect(errors, _integer(lease.get("renewal_sequence")), "renewal_sequence_invalid")

    acquired = _parse_time(lease.get("acquired_at_utc"), "acquired_at_utc", errors)
    renewed = _parse_time(lease.get("renewed_at_utc"), "renewed_at_utc", errors)
    expires = _parse_time(lease.get("expires_at_utc"), "expires_at_utc", errors)
    if acquired and renewed:
        _expect(errors, acquired <= renewed, "renewed_before_acquired")
    if renewed and expires:
        _expect(errors, renewed < expires, "expires_not_after_renewed")
        _expect(
            errors,
            expires - renewed <= timedelta(seconds=MAX_LEASE_SECONDS),
            "lease_ttl_exceeds_maximum",
        )

    terminal = lease.get("terminal_at_utc")
    if status == "RUNNING":
        _expect(errors, terminal is None, "running_lease_has_terminal_time")
    elif _enum_member(status, _TERMINAL):
        parsed_terminal = _parse_time(terminal, "terminal_at_utc", errors)
        if parsed_terminal and renewed:
            _expect(errors, parsed_terminal >= renewed, "terminal_before_renewal")
    return tuple(dict.fromkeys(errors))


def assess_training_run_lease(
    manifest: Mapping[str, Any],
    lease: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> TrainingLeaseAssessment:
    current = _normalize_time(now)
    manifest_errors = validate_launch_manifest(manifest)
    digest = launch_manifest_sha256(manifest) if not manifest_errors else None
    lease_errors: tuple[str, ...] = ()
    blockers: list[str] = []
    if lease is None:
        blockers.append("training_run_lease_missing")
    else:
        lease_errors = validate_training_run_lease(lease, manifest)
        if not lease_errors:
            time_errors: list[str] = []
            acquired = _parse_time(lease.get("acquired_at_utc"), "acquired_at_utc", time_errors)
            renewed = _parse_time(lease.get("renewed_at_utc"), "renewed_at_utc", time_errors)
            expires = _parse_time(lease.get("expires_at_utc"), "expires_at_utc", time_errors)
            if lease.get("status") != "RUNNING":
                blockers.append("training_run_lease_not_running")
            elif acquired is not None and current < acquired:
                blockers.append("training_run_lease_not_yet_active")
            elif renewed is not None and current < renewed:
                blockers.append("training_run_lease_renewed_in_future")
            elif expires is not None and current >= expires:
                blockers.append("training_run_lease_expired")
    errors = tuple(dict.fromkeys((*manifest_errors, *lease_errors)))
    lease_valid = lease is not None and not lease_errors
    active = not errors and lease_valid and not blockers
    return TrainingLeaseAssessment(
        manifest_valid=not manifest_errors,
        lease_valid=lease_valid,
        contract_valid=not errors,
        active_lease_matches_manifest=active,
        local_duplicate_guard_open=active,
        manifest_sha256=digest,
        contract_errors=errors,
        blockers=tuple(blockers),
    )


def renew_training_run_lease(
    previous: TrainingLease, *, ttl_seconds: int, now: datetime | None = None
) -> TrainingLease:
    if previous.status != "RUNNING":
        raise ValueError("only RUNNING leases can be renewed")
    if not _integer(ttl_seconds, positive=True) or ttl_seconds > MAX_LEASE_SECONDS:
        raise ValueError("ttl_seconds_out_of_range")
    current = _normalize_time(now)
    renewed = datetime.strptime(previous.renewed_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    expires = datetime.strptime(previous.expires_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    if current <= renewed:
        raise ValueError("renewal_time_must_advance")
    if current >= expires:
        raise ValueError("expired_lease_cannot_be_renewed")
    return replace(
        previous,
        renewed_at_utc=_format_time(current),
        expires_at_utc=_format_time(current + timedelta(seconds=ttl_seconds)),
        renewal_sequence=previous.renewal_sequence + 1,
    )


def terminate_training_run_lease(
    previous: TrainingLease, *, status: str, now: datetime | None = None
) -> TrainingLease:
    if previous.status != "RUNNING" or not _enum_member(status, _TERMINAL):
        raise ValueError("invalid_terminal_transition")
    current = _normalize_time(now)
    renewed = datetime.strptime(previous.renewed_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    expires = datetime.strptime(previous.expires_at_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    if current < renewed:
        raise ValueError("terminal_time_before_last_renewal")
    if status == "COMPLETED" and current >= expires:
        raise ValueError("completed_after_lease_expiry")
    return replace(previous, status=status, terminal_at_utc=_format_time(current))


def validate_lease_transition(
    previous: Mapping[str, Any],
    candidate: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    errors = [
        *validate_training_run_lease(previous, manifest),
        *validate_training_run_lease(candidate, manifest),
    ]
    immutable = (
        "schema_version",
        "lease_contract_id",
        "manifest_sha256",
        "run_id",
        "holder_id",
        "training_authority_ref",
        "training_authority_sha256",
        "compute_authority_ref",
        "compute_authority_sha256",
        "acquired_at_utc",
    )
    for field in immutable:
        _expect(
            errors,
            previous.get(field) == candidate.get(field),
            f"lease_transition_changes_immutable_field:{field}",
        )
    previous_status = previous.get("status")
    candidate_status = candidate.get("status")
    if _enum_member(previous_status, _TERMINAL):
        errors.append("terminal_lease_cannot_transition")
    elif previous_status != "RUNNING":
        errors.append("previous_lease_not_running")
    elif candidate_status == "RUNNING":
        _expect(
            errors,
            _integer(previous.get("renewal_sequence"))
            and candidate.get("renewal_sequence") == previous["renewal_sequence"] + 1,
            "renewal_sequence_must_increment_by_one",
        )
        old = _parse_time(previous.get("renewed_at_utc"), "old_renewed_at_utc", errors)
        previous_expires = _parse_time(
            previous.get("expires_at_utc"), "old_expires_at_utc", errors
        )
        new = _parse_time(candidate.get("renewed_at_utc"), "new_renewed_at_utc", errors)
        if old and new:
            _expect(errors, new > old, "renewal_time_must_advance")
        if previous_expires and new:
            _expect(errors, new < previous_expires, "expired_lease_cannot_be_renewed")
    elif _enum_member(candidate_status, _TERMINAL):
        for field in ("renewal_sequence", "renewed_at_utc", "expires_at_utc"):
            _expect(
                errors,
                candidate.get(field) == previous.get(field),
                f"terminal_transition_changes:{field}",
            )
        if candidate_status == "COMPLETED":
            previous_expires = _parse_time(
                previous.get("expires_at_utc"), "old_expires_at_utc", errors
            )
            terminal = _parse_time(candidate.get("terminal_at_utc"), "terminal_at_utc", errors)
            if previous_expires and terminal:
                _expect(errors, terminal < previous_expires, "completed_after_lease_expiry")
    else:
        errors.append("lease_transition_status_invalid")
    return tuple(dict.fromkeys(errors))


def training_side_effect_idempotency_key(
    *, manifest_sha256: str, run_id: str, effect_kind: str, logical_step: int
) -> str:
    if not _sha256(manifest_sha256) or not _token(run_id) or not _token(effect_kind):
        raise ValueError("idempotency_identity_invalid")
    if not _integer(logical_step):
        raise ValueError("logical_step_invalid")
    payload = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "run_id": run_id,
        "effect_kind": effect_kind,
        "logical_step": logical_step,
    }
    return "ts6:v1:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def local_training_run_lease_path(root: str | Path, manifest_sha256: str) -> Path:
    if not _sha256(manifest_sha256):
        raise ValueError("manifest_sha256_invalid")
    return Path(root) / f"{manifest_sha256}.training-run-lease-v1.json"


def acquire_local_training_run_lease(
    root: str | Path,
    manifest: Mapping[str, Any],
    lease: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> LocalLeaseAcquisition:
    """Exclusive-create one canonical record; never overwrite even stale/terminal evidence."""
    assessment = assess_training_run_lease(manifest, lease, now=now)
    digest = assessment.manifest_sha256 or ""
    run_id = str(lease.get("run_id", ""))
    path = local_training_run_lease_path(root, digest) if digest else Path("")
    if not assessment.local_duplicate_guard_open:
        blockers = tuple(dict.fromkeys((*assessment.contract_errors, *assessment.blockers)))
        return LocalLeaseAcquisition(False, str(path), digest, run_id, blockers)

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        blocker = (
            "training_run_lease_already_exists"
            if exc.errno == errno.EEXIST
            else f"local_lease_create_failed:{exc.errno}"
        )
        return LocalLeaseAcquisition(False, str(path), digest, run_id, (blocker,))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(dict(lease)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return LocalLeaseAcquisition(True, str(path), digest, run_id, ())
