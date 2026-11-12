"""Versioned, fixture-safe fault campaigns with durable one-shot effects."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from twelve_six.checkpoint.durability import fsync_parent_directory
from twelve_six.checkpoint.pinned_directory import pinned_real_directory
from twelve_six.checkpoint.recovery_lock import exclusive_recovery_lock
from twelve_six.evidence_control_bridge import _reject_bridge_redirects

FAULTS = frozenset({
    "process_kill", "network_loss", "disk_corruption",
    "checkpoint_interruption", "stale_state", "provider_failure",
    "node_failure", "interrupted_publication",
})
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_MAX_JOURNAL_BYTES = 65_536


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
    try:
        visible = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError("missing campaign journal record") from exc
    if not stat.S_ISREG(visible.st_mode) or (
        getattr(visible, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise ValueError("unsafe campaign journal")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (visible.st_dev, visible.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("campaign journal binding changed")
        raw = stream.read(_MAX_JOURNAL_BYTES + 1)
    if not raw or len(raw) > _MAX_JOURNAL_BYTES:
        raise ValueError("unsafe campaign journal")
    return raw


def _present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _write_once(path: Path, value: bytes) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".staged-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        fsync_parent_directory(path)
    except FileExistsError as exc:
        raise ValueError("effect already claimed; never replay") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _journal_directory(root: Path, identity: str) -> Path:
    if not isinstance(root, Path) or root.is_symlink():
        raise ValueError("unsafe campaign root")
    directory = root / identity
    _reject_bridge_redirects(directory)
    return directory


@contextmanager
def _locked_journal(root: Path, identity: str, *, create: bool = True) -> Iterator[Path]:
    directory = _journal_directory(root, identity)
    lock = root / ".chaos-locks" / identity
    _reject_bridge_redirects(lock)
    with exclusive_recovery_lock(lock):
        _reject_bridge_redirects(directory)
        with pinned_real_directory(directory, create=create) as pinned:
            yield pinned.path


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
        if type(self.expected_recovery_sha256) is not str or not _SHA256.fullmatch(
            self.expected_recovery_sha256
        ):
            raise ValueError("invalid recovery witness SHA-256")


@dataclass(frozen=True, slots=True)
class FaultCampaign:
    campaign_id: str
    source_sha: str
    cases: tuple[FaultCase, ...]
    schema_version: str = "12-6.plan8-chaos-campaign.v1"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or self.schema_version != "12-6.plan8-chaos-campaign.v1":
            raise ValueError("unsupported campaign version")
        if type(self.campaign_id) is not str or not _ID.fullmatch(self.campaign_id):
            raise ValueError("invalid campaign identifier")
        if type(self.source_sha) is not str or not _SHA40.fullmatch(self.source_sha):
            raise ValueError("exact source SHA required")
        if type(self.cases) is not tuple or not 1 <= len(self.cases) <= 32:
            raise ValueError("campaign must contain 1..32 cases")
        for case in self.cases:
            if type(case) is not FaultCase:
                raise ValueError("invalid fault case")
            case.__post_init__()
        ids = [case.case_id for case in self.cases]
        if len(set(ids)) != len(ids) or ids != sorted(ids):
            raise ValueError("cases must be unique and sorted")

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


def _case_identity(campaign: FaultCampaign, case: FaultCase) -> str:
    if type(campaign) is not FaultCampaign:
        raise ValueError("campaign must have exact type")
    identity = campaign.identity_sha256()
    if type(case) is not FaultCase or case not in campaign.cases:
        raise ValueError("case is outside campaign")
    return identity


def _claim_bytes(identity: str, case: FaultCase) -> bytes:
    return _bytes({"campaign": identity, "case": case.case_id})


def _receipt(
    campaign: FaultCampaign,
    identity: str,
    case: FaultCase,
) -> dict[str, object]:
    return {
        "schema_version": "12-6.plan8-chaos-receipt.v1",
        "campaign_sha256": identity,
        "source_sha": campaign.source_sha,
        "case_id": case.case_id,
        "fault": case.fault,
        "recovery_sha256": case.expected_recovery_sha256,
        "effect_replayed": False,
        "evidence_level": "Q0",
    }


def _failure_bytes(identity: str, case: FaultCase, observed: str) -> bytes:
    return _bytes({
        "schema_version": "12-6.plan8-chaos-failure.v1",
        "campaign_sha256": identity,
        "case_id": case.case_id,
        "observed_sha256": observed,
        "expected_sha256": case.expected_recovery_sha256,
    }) + b"\n"


def _validate_failure(path: Path, identity: str, case: FaultCase) -> None:
    raw = _read_exact(path)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("campaign failure receipt was corrupted") from exc
    observed = value.get("observed_sha256") if type(value) is dict else None
    if (
        type(observed) is not str
        or not _SHA256.fullmatch(observed)
        or observed == case.expected_recovery_sha256
        or raw != _failure_bytes(identity, case, observed)
    ):
        raise ValueError("campaign failure receipt was corrupted")


def _recover_locked(
    directory: Path,
    campaign: FaultCampaign,
    identity: str,
    case: FaultCase,
    recover: Callable[[], Any],
) -> dict[str, object]:
    if _read_exact(directory / (case.case_id + ".claim")) != _claim_bytes(identity, case):
        raise ValueError("claim was corrupted")
    failure = directory / (case.case_id + ".failure")
    if _present(failure):
        _validate_failure(failure, identity, case)
        raise ValueError("conflicting durable failure receipt")
    payload = _receipt(campaign, identity, case)
    raw = _bytes(payload) + b"\n"
    receipt = directory / (case.case_id + ".receipt")
    existing = _present(receipt)
    if existing and _read_exact(receipt) != raw:
        raise ValueError("conflicting durable recovery receipt")
    observed = _digest(_bytes(recover()))
    if observed != case.expected_recovery_sha256:
        _write_once(failure, _failure_bytes(identity, case, observed))
        raise ValueError("recovery witness mismatch")
    if not existing:
        _write_once(receipt, raw)
    return payload


def run_fault(
    root: Path,
    campaign: FaultCampaign,
    case: FaultCase,
    *,
    inject: Callable[[], Any],
    recover: Callable[[], Any],
) -> dict[str, object]:
    """Publish a one-shot claim before injection and serialize recovery."""
    identity = _case_identity(campaign, case)
    with _locked_journal(root, identity) as directory:
        claim = directory / (case.case_id + ".claim")
        receipt = directory / (case.case_id + ".receipt")
        failure = directory / (case.case_id + ".failure")
        if _present(receipt) or _present(failure):
            raise ValueError("case already has a durable outcome; never replay")
        _write_once(claim, _claim_bytes(identity, case))
        try:
            inject()
        except (OSError, RuntimeError):
            # The independent recovery witness decides whether the fault was contained.
            pass
        return _recover_locked(directory, campaign, identity, case, recover)


def recover_fault(
    root: Path,
    campaign: FaultCampaign,
    case: FaultCase,
    *,
    recover: Callable[[], Any],
) -> dict[str, object]:
    """Observe an already claimed case without executing its injection again."""
    identity = _case_identity(campaign, case)
    with _locked_journal(root, identity) as directory:
        return _recover_locked(directory, campaign, identity, case, recover)


def _readback(directory: Path | None, campaign: FaultCampaign, identity: str) -> dict[str, object]:
    passed: list[str] = []
    missing: list[str] = []
    failed: list[str] = []
    claimed_pending: list[str] = []
    for case in campaign.cases:
        if directory is None:
            missing.append(case.case_id)
            continue
        receipt = directory / (case.case_id + ".receipt")
        failure = directory / (case.case_id + ".failure")
        claim = directory / (case.case_id + ".claim")
        claimed = _present(claim)
        recorded = _present(receipt)
        failed_case = _present(failure)
        if (claimed or recorded or failed_case) and _read_exact(claim) != _claim_bytes(identity, case):
            raise ValueError("claim was corrupted")
        if recorded and _read_exact(receipt) != _bytes(_receipt(campaign, identity, case)) + b"\n":
            raise ValueError("campaign receipt was corrupted")
        if failed_case:
            _validate_failure(failure, identity, case)
            failed.append(case.case_id)
        elif recorded:
            passed.append(case.case_id)
        else:
            missing.append(case.case_id)
            if claimed:
                claimed_pending.append(case.case_id)
    return {
        "campaign_sha256": identity,
        "source_sha": campaign.source_sha,
        "evidence_level": "Q0",
        "passed": passed,
        "missing": missing,
        "failed": failed,
        "claimed_pending": claimed_pending,
        "complete": len(passed) == len(campaign.cases),
    }


def campaign_readback(root: Path, campaign: FaultCampaign) -> dict[str, object]:
    """Verify journal records and distinguish unfinished claims from unstarted cases."""
    if type(campaign) is not FaultCampaign:
        raise ValueError("campaign must have exact type")
    identity = campaign.identity_sha256()
    directory = _journal_directory(root, identity)
    if not _present(directory):
        return _readback(None, campaign, identity)
    with _locked_journal(root, identity, create=False) as journal:
        return _readback(journal, campaign, identity)
