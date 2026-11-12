"""Immutable qualification records and their retained evidence."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from twelve_six.ai_qa_control import (
    AIQAPolicy,
    FailureClass,
    FailurePacket,
    FailureSource,
    GateKind,
    GateReceipt,
    PhysicalScope,
    RepairCandidate,
    _canonical_json_bytes,
    _receipt_from_dict,
    evaluate_promotion,
)
from twelve_six.checkpoint.durability import fsync_parent_directory
from twelve_six.checkpoint.recovery_lock import exclusive_recovery_lock
from twelve_six.evidence_control_bridge import _reject_bridge_redirects
from twelve_six.evidence_policy import (
    EvidenceLevel,
    EvidencePolicy,
    EvidenceTarget,
    VerifiedEvidence,
    evaluate_qualified_repair,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_KINDS = frozenset({"FAILURE", "CANDIDATE", "RETEST", "PROMOTION_DECISION"})
_SCHEMA = "12-6.plan8-evidence-lineage-record.v1"


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: object, name: str) -> None:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError(f"invalid {name} SHA-256")


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    """A retained local artifact or a hash reference to an artifact held at its source."""

    evidence_sha256: str
    content_sha256: str
    size: int
    provenance: str
    content: bytes | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        _digest(self.evidence_sha256, "artifact evidence")
        _digest(self.content_sha256, "artifact content")
        if type(self.size) is not int or self.size < 0:
            raise ValueError("invalid artifact size")
        if type(self.provenance) is not str or not self.provenance.strip():
            raise ValueError("artifact provenance is required")
        if self.content is not None and (
            type(self.content) is not bytes or len(self.content) != self.size
            or _hash(self.content) != self.content_sha256
        ):
            raise ValueError("retained artifact content does not match its identity")
        return {
            "evidence_sha256": self.evidence_sha256,
            "content_sha256": self.content_sha256,
            "size": self.size,
            "provenance": self.provenance,
            "retention": "SOURCE_REFERENCE" if self.content is None else "LOCAL_PINNED",
        }


def defect_identity(failure: FailurePacket) -> str:
    """Group repeated observations of a defect on the same failing revision and gate."""
    if type(failure) is not FailurePacket:
        raise ValueError("exact failure packet required")
    packet = failure.to_dict()
    key = {
        "schema_version": "12-6.plan8-defect-identity.v1",
        "failure_summary": " ".join(packet["failure_summary"].split()),
        **{
            field: packet[field] for field in (
                "failing_git_sha", "failure_class", "reproducer_argv",
                "physical_scope", "physical_gate_id",
            )
        },
    }
    return _hash(_canonical_json_bytes(key))


def _failure_from_record(payload: dict[str, Any]) -> FailurePacket:
    return FailurePacket(**{
        **payload,
        "source": FailureSource(payload["source"]),
        "failure_class": FailureClass(payload["failure_class"]),
        "physical_scope": PhysicalScope(payload["physical_scope"]),
        "reproducer_argv": tuple(payload["reproducer_argv"]),
    })


def _bind_candidate(candidate: RepairCandidate, packet: dict[str, Any]) -> None:
    if (
        candidate.failure_packet_identity_sha256 != _hash(_canonical_json_bytes(packet))
        or candidate.base_git_sha != packet["failing_git_sha"]
        or candidate.defect_id != packet["defect_id"]
    ):
        raise ValueError("candidate does not bind the recorded failure")


def _validate_recorded_promotion(
    payload: dict[str, Any], parents: list[dict[str, Any]], failure: FailurePacket,
) -> None:
    _digest(payload["evidence_policy_sha256"], "promotion policy")
    _digest(payload["registry_sha256"], "promotion registry")
    policy = payload["repair_policy"]
    repair_policy = AIQAPolicy(**{
        **policy,
        "automated_gate_order": tuple(GateKind(item) for item in policy["automated_gate_order"]),
        "promotion_gate_order": tuple(GateKind(item) for item in policy["promotion_gate_order"]),
    })
    frozen = parents[0]["payload"]
    candidate = RepairCandidate(**frozen["candidate"])
    receipts = tuple(_receipt_from_dict(item["payload"]["receipt"]) for item in parents[1:])
    decision = payload["decision"]
    repair = evaluate_promotion(
        failure, candidate, receipts,
        certifier_actor_id=decision["repair_gate"]["certifier_actor_id"], policy=repair_policy,
    )
    evidence = decision["capability_evidence"]
    required = EvidenceLevel(evidence["required_level"]).value
    levels = {level.value for level in EvidenceLevel}
    admitted = {
        item["kind"] for item in payload["qualification_evidence"]
        if item["kind"] in levels and item["subject"] == evidence["capability_id"]
        and item["registry_sha256"] == payload["registry_sha256"]
        and item["git_sha"] == candidate.candidate_git_sha
        and item["package_sha256"] == (None if item["kind"] == "Q0" else frozen["package_sha256"])
    }
    package_levels = admitted - {"Q0"}
    qualification = {
        "capability_id": evidence["capability_id"], "required_level": required,
        "admitted_levels": sorted(admitted), "package_bound_levels": sorted(package_levels),
        "evidence_complete": required in admitted,
    }
    ready = repair.decision == "READY_FOR_INDEPENDENT_PROMOTION" and required in admitted and bool(package_levels)
    expected = {
        "decision": "READY_FOR_INDEPENDENT_PROMOTION" if ready else "BLOCK",
        "repair_gate": repair.to_dict(), "capability_evidence": qualification,
    }
    if decision != expected:
        raise ValueError("promotion decision does not match the recorded candidate or gates")


def _validate_causal_record(record: dict[str, Any], earlier: dict[str, dict[str, Any]]) -> None:
    identities = record["parents"]
    if any(type(identity) is not str or identity not in earlier for identity in identities):
        raise ValueError("lineage record has a missing or out-of-order parent")
    parents = [earlier[identity] for identity in identities]
    kinds = [parent["kind"] for parent in parents]
    payload = record["payload"]
    kind = record["kind"]
    try:
        if kind == "FAILURE":
            if parents or set(payload) != {"packet"}:
                raise ValueError("failure record has a parent or invalid fields")
            failure = _failure_from_record(payload["packet"])
            if failure.to_dict() != payload["packet"] or defect_identity(failure) != record["defect_sha256"]:
                raise ValueError("failure record does not match its defect")
        elif kind == "CANDIDATE":
            if kinds not in (["FAILURE"], ["FAILURE", "CANDIDATE"]) or set(payload) != {
                "candidate", "package_sha256",
            }:
                raise ValueError("candidate freeze has invalid parents or fields")
            candidate = RepairCandidate(**payload["candidate"])
            _digest(payload["package_sha256"], "candidate package")
            _bind_candidate(candidate, parents[0]["payload"]["packet"])
        elif kind == "RETEST":
            if kinds != ["CANDIDATE"] or set(payload) != {"receipt"}:
                raise ValueError("retest does not reference a candidate freeze")
            receipt = _receipt_from_dict(payload["receipt"])
            if receipt.git_sha != parents[0]["payload"]["candidate"]["candidate_git_sha"]:
                raise ValueError("retest receipt belongs to a different candidate")
        else:
            if (
                not kinds or kinds[0] != "CANDIDATE"
                or any(parent["kind"] != "RETEST" or parent["parents"] != identities[:1] for parent in parents[1:])
                or identities[1:] != sorted(set(identities[1:]))
                or set(payload) != {
                    "decision", "evidence_policy_sha256", "registry_sha256", "qualification_evidence",
                    "repair_policy",
                }
            ):
                raise ValueError("promotion record has invalid parents or fields")
            observed = earlier[parents[0]["parents"][0]]
            _validate_recorded_promotion(payload, parents, _failure_from_record(observed["payload"]["packet"]))
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed lineage causal record") from exc


class EvidenceLineage:
    """Record lifecycle edges in one durable local journal without changing a candidate."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).absolute()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        _reject_bridge_redirects(self.root)
        with exclusive_recovery_lock(self.root) as root:
            path = root / "evidence-lineage.sqlite3"
            # The lock can supply a trusted directory-fd alias on POSIX.
            # Inspect SQLite's files without rechecking that alias as a path.
            existed = False
            for suffix in ("", "-journal", "-wal", "-shm"):
                leaf = path.with_name(path.name + suffix)
                try:
                    info = leaf.lstat()
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(info.st_mode) or (
                    getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                ):
                    raise ValueError("unsafe evidence journal file")
                if not suffix:
                    existed = True
            with closing(sqlite3.connect(path, timeout=10)) as db:
                db.execute("PRAGMA synchronous=FULL")
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS records ("
                        "identity TEXT PRIMARY KEY, defect TEXT NOT NULL, "
                        "dedup_key TEXT UNIQUE, body BLOB NOT NULL)"
                    )
                    db.execute(
                        "CREATE TABLE IF NOT EXISTS retained ("
                        "digest TEXT PRIMARY KEY, content BLOB NOT NULL)"
                    )
                    yield db
            if not existed:
                fsync_parent_directory(path)

    @staticmethod
    def _decode(identity: str, defect: str, raw: bytes) -> dict[str, Any]:
        _digest(identity, "record")
        if type(raw) is not bytes or _hash(raw) != identity:
            raise ValueError("lineage record content identity mismatch")
        record = json.loads(raw)
        if (
            type(record) is not dict
            or set(record) != {
                "schema_version", "defect_sha256", "kind", "parents", "payload", "artifacts",
            }
            or record["schema_version"] != _SCHEMA
            or record["defect_sha256"] != defect
            or type(record["kind"]) is not str
            or record["kind"] not in _KINDS
            or type(record["parents"]) is not list
            or type(record["payload"]) is not dict
            or type(record["artifacts"]) is not list
            or _canonical_json_bytes(record) != raw
        ):
            raise ValueError("invalid lineage record")
        return record

    @classmethod
    def _record(cls, db: sqlite3.Connection, identity: str, kind: str) -> dict[str, Any]:
        _digest(identity, "parent record")
        row = db.execute("SELECT defect, body FROM records WHERE identity = ?", (identity,)).fetchone()
        if row is None:
            raise ValueError("lineage parent record is missing")
        record = cls._decode(identity, *row)
        if record["kind"] != kind:
            raise ValueError("lineage parent has the wrong record kind")
        return record

    @staticmethod
    def _append(
        db: sqlite3.Connection, *, defect: str, kind: str, parents: tuple[str, ...],
        payload: dict[str, Any], artifacts: tuple[EvidenceArtifact, ...] = (),
        dedup_key: str | None = None, expected_evidence_sha256: str | None = None,
    ) -> str:
        if type(artifacts) is not tuple or any(type(item) is not EvidenceArtifact for item in artifacts):
            raise ValueError("artifacts must be a tuple of exact evidence artifacts")
        refs = [item.to_dict() for item in artifacts]
        refs.sort(key=_canonical_json_bytes)
        if len({_canonical_json_bytes(ref) for ref in refs}) != len(refs):
            raise ValueError("duplicate artifact references")
        if expected_evidence_sha256 is not None and refs and not any(
            ref["evidence_sha256"] == expected_evidence_sha256 for ref in refs
        ):
            raise ValueError("artifacts do not bind the primary evidence identity")
        record = {
            "schema_version": _SCHEMA, "defect_sha256": defect, "kind": kind,
            "parents": list(parents), "payload": payload, "artifacts": refs,
        }
        raw = _canonical_json_bytes(record)
        identity = _hash(raw)
        if dedup_key is not None:
            previous = db.execute(
                "SELECT identity, defect, body FROM records WHERE dedup_key = ?", (dedup_key,),
            ).fetchone()
            if previous is not None:
                previous_id, previous_defect, previous_raw = previous
                existing = EvidenceLineage._decode(previous_id, previous_defect, previous_raw)
                if kind == "FAILURE":
                    # Observers can use different tracker labels for the same exact failure.
                    existing["payload"]["packet"]["defect_id"] = payload["packet"]["defect_id"]
                if _canonical_json_bytes(existing) != raw:
                    raise ValueError("existing record has conflicting evidence or provenance")
                EvidenceLineage._read_artifacts(db, refs)
                return previous_id
        for item in artifacts:
            if item.content is None:
                continue
            db.execute("INSERT OR IGNORE INTO retained VALUES (?, ?)", (item.content_sha256, item.content))
        EvidenceLineage._read_artifacts(db, refs)
        db.execute("INSERT OR IGNORE INTO records VALUES (?, ?, ?, ?)", (identity, defect, dedup_key, raw))
        stored = db.execute("SELECT defect, dedup_key, body FROM records WHERE identity = ?", (identity,)).fetchone()
        if stored != (defect, dedup_key, raw):
            raise ValueError("existing lineage record differs from exact bytes")
        return identity

    @staticmethod
    def _read_artifacts(db: sqlite3.Connection, refs: list[dict[str, Any]]) -> dict[str, bytes | None]:
        results = {}
        for ref in refs:
            if ref["retention"] == "SOURCE_REFERENCE":
                content = None
            elif ref["retention"] == "LOCAL_PINNED":
                row = db.execute("SELECT content FROM retained WHERE digest = ?", (ref["content_sha256"],)).fetchone()
                if row is None:
                    raise ValueError("retained artifact is missing")
                content = row[0]
            else:
                raise ValueError("unknown evidence artifact retention")
            checked = EvidenceArtifact(
                ref["evidence_sha256"], ref["content_sha256"], ref["size"], ref["provenance"], content,
            )
            if checked.to_dict() != ref:
                raise ValueError("evidence artifact metadata is not canonical")
            if ref["content_sha256"] not in results or content is not None:
                results[ref["content_sha256"]] = content
        return results

    def record_failure(self, failure: FailurePacket, *, artifacts: tuple[EvidenceArtifact, ...] = ()) -> str:
        defect = defect_identity(failure)
        packet = failure.to_dict()
        observation = _hash(_canonical_json_bytes([
            defect, packet["source"], packet["source_evidence_identity_sha256"],
        ]))
        with self._transaction() as db:
            return self._append(
                db, defect=defect, kind="FAILURE", parents=(), payload={"packet": packet},
                artifacts=artifacts, dedup_key=f"failure:{observation}",
                expected_evidence_sha256=failure.source_evidence_identity_sha256,
            )

    def freeze_candidate(
        self, failure_record: str, candidate: RepairCandidate, *, package_sha256: str,
        repair_parent: str | None = None, artifacts: tuple[EvidenceArtifact, ...] = (),
    ) -> str:
        if type(candidate) is not RepairCandidate:
            raise ValueError("exact repair candidate required")
        _digest(package_sha256, "candidate package")
        body = candidate.to_dict()
        with self._transaction() as db:
            failure = self._record(db, failure_record, "FAILURE")
            packet = failure["payload"]["packet"]
            defect = failure["defect_sha256"]
            _bind_candidate(candidate, packet)
            parents = (failure_record,)
            if repair_parent is not None:
                previous = self._record(db, repair_parent, "CANDIDATE")
                if previous["defect_sha256"] != defect:
                    raise ValueError("repair parent belongs to a different defect")
                parents += (repair_parent,)
            return self._append(
                db, defect=defect, kind="CANDIDATE", parents=parents,
                payload={"candidate": body, "package_sha256": package_sha256}, artifacts=artifacts,
                dedup_key=f"candidate:{defect}:{candidate.candidate_git_sha}",
                expected_evidence_sha256=candidate.patch_sha256,
            )

    def record_retest(
        self, candidate_record: str, receipt: GateReceipt, *, artifacts: tuple[EvidenceArtifact, ...] = (),
    ) -> str:
        if type(receipt) is not GateReceipt:
            raise ValueError("exact gate receipt required")
        body = receipt.to_dict()
        with self._transaction() as db:
            frozen = self._record(db, candidate_record, "CANDIDATE")
            if receipt.git_sha != frozen["payload"]["candidate"]["candidate_git_sha"]:
                raise ValueError("retest receipt belongs to a different candidate")
            return self._append(
                db, defect=frozen["defect_sha256"], kind="RETEST", parents=(candidate_record,),
                payload={"receipt": body}, artifacts=artifacts,
                dedup_key=f"retest:{candidate_record}:{body['gate']}:{receipt.evidence_identity_sha256}",
                expected_evidence_sha256=receipt.evidence_identity_sha256,
            )

    def record_promotion_decision(
        self, candidate_record: str, retests: tuple[str, ...], *,
        failure: FailurePacket, candidate: RepairCandidate, certifier_actor_id: str,
        repair_policy: AIQAPolicy, evidence_policy: EvidencePolicy, capability_id: str,
        evidence: tuple[VerifiedEvidence, ...],
    ) -> str:
        if type(failure) is not FailurePacket or type(candidate) is not RepairCandidate:
            raise ValueError("exact failure packet and repair candidate required")
        if (
            type(retests) is not tuple or any(type(item) is not str for item in retests)
            or len(set(retests)) != len(retests)
        ):
            raise ValueError("retests must be a tuple of unique record identities")
        retests = tuple(sorted(retests))
        with self._transaction() as db:
            frozen = self._record(db, candidate_record, "CANDIDATE")
            observed = self._record(db, frozen["parents"][0], "FAILURE")
            if (
                frozen["payload"]["candidate"] != candidate.to_dict()
                or observed["payload"]["packet"] != failure.to_dict()
            ):
                raise ValueError("promotion inputs differ from the frozen failure or candidate")
            recorded = [self._record(db, identity, "RETEST") for identity in retests]
            if any(item["parents"] != [candidate_record] for item in recorded):
                raise ValueError("promotion retests belong to a different candidate freeze")
            receipts = tuple(_receipt_from_dict(item["payload"]["receipt"]) for item in recorded)
            target = EvidenceTarget(candidate.candidate_git_sha, frozen["payload"]["package_sha256"])
            decision = evaluate_qualified_repair(
                failure, candidate, receipts, certifier_actor_id=certifier_actor_id,
                repair_policy=repair_policy, evidence_policy=evidence_policy,
                capability_id=capability_id, target=target, evidence=evidence,
            )
            return self._append(
                db, defect=frozen["defect_sha256"], kind="PROMOTION_DECISION",
                parents=(candidate_record, *retests),
                payload={
                    "decision": decision,
                    "evidence_policy_sha256": evidence_policy.policy_sha256,
                    "registry_sha256": evidence_policy.registry_sha256,
                    "repair_policy": asdict(repair_policy),
                    "qualification_evidence": sorted(
                        ({
                            "kind": item.kind, "subject": item.subject,
                            "git_sha": item.target.git_sha, "package_sha256": item.target.package_sha256,
                            "artifact_sha256": item.artifact_sha256, "registry_sha256": item.registry_sha256,
                            "config_sha256": item.config_sha256, "backend": item.backend,
                        } for item in evidence), key=_canonical_json_bytes,
                    ),
                },
            )

    def readback(self, defect: str, *, expected_snapshot_sha256: str | None = None) -> dict[str, Any]:
        """Rehash immutable records, retained artifacts and causal references."""
        _digest(defect, "defect")
        if expected_snapshot_sha256 is not None:
            _digest(expected_snapshot_sha256, "trusted snapshot")
        with self._transaction() as db:
            records: dict[str, dict[str, Any]] = {}
            rows = db.execute("SELECT identity, body FROM records WHERE defect = ? ORDER BY rowid", (defect,))
            for identity, raw in rows:
                record = self._decode(identity, defect, raw)
                _validate_causal_record(record, records)
                self._read_artifacts(db, record["artifacts"])
                records[identity] = record
            snapshot = {"defect_sha256": defect, "records": records}
            identity = _hash(_canonical_json_bytes(snapshot))
            if expected_snapshot_sha256 is not None and identity != expected_snapshot_sha256:
                raise ValueError("lineage does not match the trusted snapshot")
            return {**snapshot, "snapshot_sha256": identity}

    def retained_artifact(self, record_sha256: str, content_sha256: str) -> bytes | None:
        _digest(record_sha256, "record")
        _digest(content_sha256, "artifact content")
        with self._transaction() as db:
            row = db.execute("SELECT defect, body FROM records WHERE identity = ?", (record_sha256,)).fetchone()
            if row is None:
                raise ValueError("lineage record is missing")
            record = self._decode(record_sha256, *row)
            artifacts = self._read_artifacts(db, record["artifacts"])
            if content_sha256 not in artifacts:
                raise ValueError("artifact is not referenced by the record")
            return artifacts[content_sha256]
