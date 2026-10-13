"""Versioned, fixture-safe fault campaigns with durable one-shot effects."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FAULTS = frozenset({
    "process_kill", "network_loss", "disk_corruption",
    "checkpoint_interruption", "stale_state", "provider_failure",
    "node_failure", "interrupted_publication",
})
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")


def parse_campaign_bytes(raw: bytes) -> FaultCampaign:
    """Load a bounded versioned fault matrix without duplicate JSON fields."""
    if type(raw) is not bytes or not 0 < len(raw) <= 65_536:
        raise ValueError("campaign bytes exceed bound")
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate campaign JSON field")
            value[key] = item
        return value
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid campaign JSON") from exc
    if type(value) is not dict or set(value) != {
        "schema_version", "campaign_id", "source_sha", "cases"
    } or type(value["cases"]) is not list or not 0 < len(value["cases"]) <= 32:
        raise ValueError("noncanonical campaign manifest")
    cases = []
    for row in value["cases"]:
        if type(row) is not dict or set(row) != {
            "case_id", "fault", "expected_recovery_sha256"
        }:
            raise ValueError("noncanonical campaign case")
        cases.append(FaultCase(**row))
    return FaultCampaign(value["campaign_id"], value["source_sha"], tuple(cases),
                         value["schema_version"])


def _bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_exact(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65_536:
        raise ValueError("unsafe campaign journal")
    return path.read_bytes()


def _write_once(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("unsafe campaign journal path")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("effect already claimed; never replay") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True, slots=True)
class FaultCase:
    case_id: str
    fault: str
    expected_recovery_sha256: str

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not _ID.fullmatch(self.case_id):
            raise ValueError("invalid fault case identifier")
        if type(self.fault) is not str or self.fault not in FAULTS:
            raise ValueError("unsupported fault")
        if type(self.expected_recovery_sha256) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", self.expected_recovery_sha256
        ):
            raise ValueError("invalid recovery witness SHA-256")


@dataclass(frozen=True, slots=True)
class FaultCampaign:
    campaign_id: str
    source_sha: str
    cases: tuple[FaultCase, ...]
    schema_version: str = "12-6.plan8-chaos-campaign.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "12-6.plan8-chaos-campaign.v1":
            raise ValueError("unsupported campaign version")
        if type(self.campaign_id) is not str or not _ID.fullmatch(self.campaign_id):
            raise ValueError("invalid campaign identifier")
        if type(self.source_sha) is not str or not _SHA40.fullmatch(self.source_sha):
            raise ValueError("exact source SHA required")
        if type(self.cases) is not tuple or not self.cases:
            raise ValueError("campaign needs cases")
        ids = [case.case_id for case in self.cases]
        if len(set(ids)) != len(ids) or ids != sorted(ids):
            raise ValueError("cases must be unique and sorted")
        for case in self.cases:
            if type(case) is not FaultCase:
                raise ValueError("invalid fault case")
            case.__post_init__()

    def identity_sha256(self) -> str:
        self.__post_init__()
        return _digest(_bytes({
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "source_sha": self.source_sha,
            "cases": [
                (case.case_id, case.fault, case.expected_recovery_sha256)
                for case in self.cases
            ],
        }))


def run_fault(
    root: Path,
    campaign: FaultCampaign,
    case: FaultCase,
    *,
    inject: Callable[[], Any],
    recover: Callable[[], Any],
) -> dict[str, object]:
    """Fence an effect before injection; recover via observation, never rerun it."""
    campaign.__post_init__()
    if type(case) is not FaultCase or case not in campaign.cases:
        raise ValueError("case is outside campaign")
    if not isinstance(root, Path) or root.is_symlink():
        raise ValueError("unsafe campaign root")
    identity = campaign.identity_sha256()
    directory = root / identity
    claim = directory / (case.case_id + ".claim")
    receipt = directory / (case.case_id + ".receipt")
    if receipt.exists():
        raise ValueError("case already has receipt; never replay")
    _write_once(claim, _bytes({"campaign": identity, "case": case.case_id}))
    try:
        inject()
    except (OSError, RuntimeError):
        # A fault is expected; the independent recovery witness decides success.
        return recover_fault(root, campaign, case, recover=recover)
    return recover_fault(root, campaign, case, recover=recover)


def recover_fault(
    root: Path,
    campaign: FaultCampaign,
    case: FaultCase,
    *,
    recover: Callable[[], Any],
) -> dict[str, object]:
    """Recovery is repeatable only when the original witness bytes agree."""
    campaign.__post_init__()
    if type(case) is not FaultCase or case not in campaign.cases:
        raise ValueError("case is outside campaign")
    if not isinstance(root, Path) or root.is_symlink():
        raise ValueError("unsafe campaign root")
    identity = campaign.identity_sha256()
    directory = root / identity
    if directory.is_symlink():
        raise ValueError("unsafe campaign journal path")
    if _read_exact(directory / (case.case_id + ".claim")) != _bytes({
        "campaign": identity, "case": case.case_id
    }):
        raise ValueError("claim was corrupted")
    observed = _digest(_bytes(recover()))
    if observed != case.expected_recovery_sha256:
        failure = _bytes({
            "schema_version": "12-6.plan8-chaos-failure.v1",
            "campaign_sha256": identity,
            "case_id": case.case_id,
            "observed_sha256": observed,
            "expected_sha256": case.expected_recovery_sha256,
        }) + b"\n"
        failure_path = directory / (case.case_id + ".failure")
        if failure_path.exists():
            if _read_exact(failure_path) != failure:
                raise ValueError("conflicting durable failure receipt")
        else:
            _write_once(failure_path, failure)
        raise ValueError("recovery witness mismatch")
    payload: dict[str, object] = {
        "schema_version": "12-6.plan8-chaos-receipt.v1",
        "campaign_sha256": identity,
        "source_sha": campaign.source_sha,
        "case_id": case.case_id,
        "fault": case.fault,
        "recovery_sha256": observed,
        "effect_replayed": False,
        "evidence_level": "Q0",
    }
    raw = _bytes(payload) + b"\n"
    receipt = directory / (case.case_id + ".receipt")
    if receipt.exists():
        if _read_exact(receipt) != raw:
            raise ValueError("conflicting durable recovery receipt")
    else:
        try:
            _write_once(receipt, raw)
        except ValueError:
            if _read_exact(receipt) != raw:
                raise
    return payload


def campaign_readback(root: Path, campaign: FaultCampaign) -> dict[str, object]:
    """Verify durable Q0 receipts and expose missing/failing fault cases."""
    campaign.__post_init__()
    if not isinstance(root, Path) or root.is_symlink():
        raise ValueError("unsafe campaign root")
    identity = campaign.identity_sha256()
    directory = root / identity
    if directory.is_symlink():
        raise ValueError("unsafe campaign journal path")
    passed: list[str] = []
    missing: list[str] = []
    failed: list[str] = []
    for case in campaign.cases:
        receipt = directory / (case.case_id + ".receipt")
        failure = directory / (case.case_id + ".failure")
        claim = directory / (case.case_id + ".claim")
        if failure.exists():
            _read_exact(failure)
            failed.append(case.case_id)
            continue
        if not receipt.exists():
            if claim.exists():
                _read_exact(claim)
            missing.append(case.case_id)
            continue
        if _read_exact(claim) != _bytes({"campaign": identity, "case": case.case_id}):
            raise ValueError("claim was corrupted")
        expected = {
            "schema_version": "12-6.plan8-chaos-receipt.v1",
            "campaign_sha256": identity,
            "source_sha": campaign.source_sha,
            "case_id": case.case_id,
            "fault": case.fault,
            "recovery_sha256": case.expected_recovery_sha256,
            "effect_replayed": False,
            "evidence_level": "Q0",
        }
        if _read_exact(receipt) != _bytes(expected) + b"\n":
            raise ValueError("campaign receipt was corrupted")
        passed.append(case.case_id)
    return {"campaign_sha256": identity, "evidence_level": "Q0",
            "passed": passed, "missing": missing, "failed": failed,
            "complete": len(passed) == len(campaign.cases) and not failed}
