from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    DATASET_CLASSIFICATION,
    BaseCorpusBoundaryError,
    DatasetRecord,
    FactoryResult,
    canonical_json,
    sha256_json,
)

LINEAGE_WORKER_ID = "NEXT100-091-TEACHER-DATA-LINEAGE"
LINEAGE_SCHEMA = "12-6.postbase-synthetic-dataset-lineage.v1"
LINEAGE_TRAINING_USE = "POSTBASE_SYNTHETIC_EXPERIMENTAL_ONLY"
_GENESIS_MANIFEST_SHA256 = "0" * 64


class LineageError(RuntimeError):
    """Base error for immutable synthetic-dataset lineage operations."""


class LineageIntegrityError(LineageError):
    """Raised when stored lineage evidence fails deterministic verification."""


class LineageMutationError(LineageError):
    """Raised when a caller attempts to rewrite immutable lineage history."""


class UnknownDatasetVersionError(LineageError):
    """Raised when a requested dataset version does not exist."""


def _require_version(value: str, *, field: str = "dataset_version") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


@dataclass(frozen=True)
class AcceptedRecordRef:
    record_id: str
    record_sha256: str
    source_dataset_version: str
    source_dataset_revision: int
    critic_identity: str
    verifier_identities: tuple[str, ...]
    source_proposal_ids: tuple[str, ...]
    teacher_identities: tuple[str, ...]

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "record_sha256": self.record_sha256,
            "source_dataset_version": self.source_dataset_version,
            "source_dataset_revision": self.source_dataset_revision,
            "critic_identity": self.critic_identity,
            "verifier_identities": self.verifier_identities,
            "source_proposal_ids": self.source_proposal_ids,
            "teacher_identities": self.teacher_identities,
        }


@dataclass(frozen=True)
class RejectionEntry:
    rejection_sha256: str
    task_id: str
    reason: str
    decision_id: str
    judge_identity: str
    critic_identity: str
    verifier_identities: tuple[str, ...]
    verification_ids: tuple[str, ...]
    source_proposal_ids: tuple[str, ...]
    teacher_identities: tuple[str, ...]

    def body(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "reason": self.reason,
            "decision_id": self.decision_id,
            "judge_identity": self.judge_identity,
            "critic_identity": self.critic_identity,
            "verifier_identities": self.verifier_identities,
            "verification_ids": self.verification_ids,
            "source_proposal_ids": self.source_proposal_ids,
            "teacher_identities": self.teacher_identities,
        }

    def manifest_entry(self) -> dict[str, Any]:
        return {**self.body(), "rejection_sha256": self.rejection_sha256}

    @classmethod
    def from_factory_result(cls, result: FactoryResult) -> RejectionEntry:
        if result.accepted or result.record is not None:
            raise ValueError("rejection log accepts rejected factory results only")
        body = {
            "task_id": result.task.task_id,
            "reason": result.reason,
            "decision_id": result.judge_decision.contribution_id,
            "judge_identity": result.judge_decision.provenance.actor_id,
            "critic_identity": result.critic_review.provenance.actor_id,
            "verifier_identities": tuple(
                sorted({item.verifier_id for item in result.verifications})
            ),
            "verification_ids": tuple(
                sorted(item.contribution_id for item in result.verifications)
            ),
            "source_proposal_ids": tuple(
                sorted(item.contribution_id for item in result.teacher_proposals)
            ),
            "teacher_identities": tuple(
                sorted({item.teacher_id for item in result.teacher_proposals})
            ),
        }
        return cls(sha256_json(body), **body)


@dataclass(frozen=True)
class LineageOperation:
    kind: str
    added_record_ids: tuple[str, ...] = ()
    deleted_record_ids: tuple[str, ...] = ()
    superseded_records: tuple[tuple[str, str], ...] = ()
    rollback_target_version: str | None = None
    rollback_target_manifest_sha256: str | None = None
    rejection_ids_added: tuple[str, ...] = ()

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "added_record_ids": self.added_record_ids,
            "deleted_record_ids": self.deleted_record_ids,
            "superseded_records": self.superseded_records,
            "rollback_target_version": self.rollback_target_version,
            "rollback_target_manifest_sha256": self.rollback_target_manifest_sha256,
            "rejection_ids_added": self.rejection_ids_added,
        }


@dataclass(frozen=True)
class DatasetVersionSnapshot:
    dataset_name: str
    dataset_version: str
    parent_version: str | None
    parent_manifest_sha256: str
    accepted_records: tuple[AcceptedRecordRef, ...]
    rejection_log: tuple[RejectionEntry, ...]
    operation: LineageOperation
    manifest_sha256: str
    schema: str = LINEAGE_SCHEMA
    worker_id: str = LINEAGE_WORKER_ID
    classification: str = DATASET_CLASSIFICATION
    base_corpus_evidence: bool = False
    canonical_base_training_eligible: bool = False
    training_use: str = LINEAGE_TRAINING_USE

    @property
    def accepted_record_hashes(self) -> tuple[str, ...]:
        return tuple(item.record_sha256 for item in self.accepted_records)

    @property
    def accepted_record_ids(self) -> tuple[str, ...]:
        return tuple(item.record_id for item in self.accepted_records)

    @property
    def critic_identities(self) -> tuple[str, ...]:
        return tuple(sorted({item.critic_identity for item in self.accepted_records}))

    @property
    def verifier_identities(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    verifier
                    for item in self.accepted_records
                    for verifier in item.verifier_identities
                }
            )
        )

    @property
    def source_proposal_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    proposal_id
                    for item in self.accepted_records
                    for proposal_id in item.source_proposal_ids
                }
            )
        )

    @property
    def teacher_identities(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    teacher
                    for item in self.accepted_records
                    for teacher in item.teacher_identities
                }
            )
        )

    def manifest_body(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "worker_id": self.worker_id,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "parent_version": self.parent_version,
            "parent_manifest_sha256": self.parent_manifest_sha256,
            "classification": self.classification,
            "base_corpus_evidence": self.base_corpus_evidence,
            "canonical_base_training_eligible": self.canonical_base_training_eligible,
            "training_use": self.training_use,
            "accepted_record_ids": self.accepted_record_ids,
            "accepted_record_hashes": self.accepted_record_hashes,
            "accepted_records": tuple(item.manifest_entry() for item in self.accepted_records),
            "critic_identities": self.critic_identities,
            "verifier_identities": self.verifier_identities,
            "source_proposal_ids": self.source_proposal_ids,
            "teacher_identities": self.teacher_identities,
            "rejection_log": tuple(item.manifest_entry() for item in self.rejection_log),
            "operation": self.operation.manifest_entry(),
            "history_policy": "APPEND_ONLY_VERSION_CHAIN",
        }

    def manifest(self) -> dict[str, Any]:
        return copy.deepcopy({**self.manifest_body(), "manifest_sha256": self.manifest_sha256})


class SyntheticDatasetLineage:
    """Append-only lineage store for accepted and rejected synthetic teacher evidence.

    Dataset versions are immutable snapshots. Removal, supersession, and rollback always
    create a new version linked to an existing parent manifest. No API can turn these
    POSTBASE datasets into canonical Base-training evidence.
    """

    def __init__(self, dataset_name: str) -> None:
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError("dataset_name must be a non-empty string")
        self.dataset_name = dataset_name
        self._versions: dict[str, DatasetVersionSnapshot] = {}
        self._record_objects: dict[str, DatasetRecord] = {}
        self._record_canonical_json: dict[str, str] = {}
        self._active_version: str | None = None

    @property
    def active_version(self) -> str | None:
        return self._active_version

    @property
    def versions(self) -> tuple[str, ...]:
        return tuple(self._versions)

    def _record_ref(self, record: DatasetRecord) -> AcceptedRecordRef:
        if record.dataset_name != self.dataset_name:
            raise LineageIntegrityError(
                "accepted record dataset_name does not match lineage dataset_name"
            )
        if record.classification != DATASET_CLASSIFICATION:
            raise LineageIntegrityError("accepted record has non-POSTBASE classification")
        if record.base_corpus_evidence:
            raise BaseCorpusBoundaryError("synthetic accepted record claims Base corpus evidence")
        if record.canonical_base_training_eligible:
            raise BaseCorpusBoundaryError(
                "synthetic accepted record claims canonical Base-training eligibility"
            )
        if record.training_use != LINEAGE_TRAINING_USE:
            raise LineageIntegrityError("accepted record has unexpected training_use")
        record_sha256 = sha256_json(record)
        if not _is_sha256(record_sha256):
            raise LineageIntegrityError("accepted record hash is invalid")
        canonical = canonical_json(record)
        previous = self._record_canonical_json.get(record_sha256)
        if previous is not None and previous != canonical:
            raise LineageIntegrityError("record SHA-256 collision or payload substitution detected")
        self._record_objects.setdefault(record_sha256, record)
        self._record_canonical_json.setdefault(record_sha256, canonical)
        return AcceptedRecordRef(
            record_id=record.record_id,
            record_sha256=record_sha256,
            source_dataset_version=record.dataset_version,
            source_dataset_revision=record.dataset_revision,
            critic_identity=record.critic_review.provenance.actor_id,
            verifier_identities=tuple(sorted({item.verifier_id for item in record.verifications})),
            source_proposal_ids=tuple(
                sorted(item.contribution_id for item in record.teacher_proposals)
            ),
            teacher_identities=tuple(sorted({item.teacher_id for item in record.teacher_proposals})),
        )

    @staticmethod
    def _rejection_entry(result: FactoryResult) -> RejectionEntry:
        return RejectionEntry.from_factory_result(result)

    def _new_rejections(
        self,
        existing: Sequence[RejectionEntry],
        results: Sequence[FactoryResult],
    ) -> tuple[RejectionEntry, ...]:
        seen = {item.rejection_sha256 for item in existing}
        output: list[RejectionEntry] = []
        for result in results:
            entry = self._rejection_entry(result)
            if entry.rejection_sha256 in seen:
                raise LineageMutationError("duplicate rejection evidence cannot be appended twice")
            seen.add(entry.rejection_sha256)
            output.append(entry)
        return tuple(output)

    def _parent(self, parent_version: str | None) -> DatasetVersionSnapshot:
        chosen = parent_version or self._active_version
        if chosen is None:
            raise UnknownDatasetVersionError("lineage has no parent version")
        return self.read(chosen)

    def _commit(
        self,
        *,
        dataset_version: str,
        parent_version: str | None,
        accepted_records: Sequence[AcceptedRecordRef],
        rejection_log: Sequence[RejectionEntry],
        operation: LineageOperation,
    ) -> DatasetVersionSnapshot:
        version = _require_version(dataset_version)
        if version in self._versions:
            raise LineageMutationError(
                f"dataset version {version!r} already exists; immutable history cannot be rewritten"
            )
        if parent_version is None:
            parent_manifest_sha256 = _GENESIS_MANIFEST_SHA256
        else:
            parent = self.read(parent_version)
            parent_manifest_sha256 = parent.manifest_sha256
        record_ids = [item.record_id for item in accepted_records]
        record_hashes = [item.record_sha256 for item in accepted_records]
        if len(record_ids) != len(set(record_ids)):
            raise LineageIntegrityError("active version contains duplicate accepted record IDs")
        if len(record_hashes) != len(set(record_hashes)):
            raise LineageIntegrityError("active version contains duplicate accepted record hashes")
        rejection_ids = [item.rejection_sha256 for item in rejection_log]
        if len(rejection_ids) != len(set(rejection_ids)):
            raise LineageIntegrityError("version contains duplicate rejection-log entries")
        provisional = DatasetVersionSnapshot(
            dataset_name=self.dataset_name,
            dataset_version=version,
            parent_version=parent_version,
            parent_manifest_sha256=parent_manifest_sha256,
            accepted_records=tuple(accepted_records),
            rejection_log=tuple(rejection_log),
            operation=operation,
            manifest_sha256="",
        )
        manifest_sha256 = sha256_json(provisional.manifest_body())
        snapshot = DatasetVersionSnapshot(
            dataset_name=provisional.dataset_name,
            dataset_version=provisional.dataset_version,
            parent_version=provisional.parent_version,
            parent_manifest_sha256=provisional.parent_manifest_sha256,
            accepted_records=provisional.accepted_records,
            rejection_log=provisional.rejection_log,
            operation=provisional.operation,
            manifest_sha256=manifest_sha256,
        )
        self._versions[version] = snapshot
        self._active_version = version
        self.verify_version(version)
        return snapshot

    def create_version(
        self,
        dataset_version: str,
        *,
        accepted_records: Sequence[DatasetRecord] = (),
        rejected_results: Sequence[FactoryResult] = (),
    ) -> DatasetVersionSnapshot:
        if self._versions:
            raise LineageMutationError(
                "genesis already exists; derive a new immutable dataset version instead"
            )
        refs = tuple(self._record_ref(record) for record in accepted_records)
        rejections = self._new_rejections((), rejected_results)
        operation = LineageOperation(
            kind="CREATE",
            added_record_ids=tuple(item.record_id for item in refs),
            rejection_ids_added=tuple(item.rejection_sha256 for item in rejections),
        )
        return self._commit(
            dataset_version=dataset_version,
            parent_version=None,
            accepted_records=refs,
            rejection_log=rejections,
            operation=operation,
        )

    def derive_version(
        self,
        dataset_version: str,
        *,
        parent_version: str | None = None,
        accepted_records: Sequence[DatasetRecord] = (),
        rejected_results: Sequence[FactoryResult] = (),
    ) -> DatasetVersionSnapshot:
        parent = self._parent(parent_version)
        new_refs = tuple(self._record_ref(record) for record in accepted_records)
        existing_ids = set(parent.accepted_record_ids)
        existing_hashes = set(parent.accepted_record_hashes)
        for item in new_refs:
            if item.record_id in existing_ids or item.record_sha256 in existing_hashes:
                raise LineageMutationError("accepted record already exists in parent version")
        new_rejections = self._new_rejections(parent.rejection_log, rejected_results)
        operation = LineageOperation(
            kind="DERIVE",
            added_record_ids=tuple(item.record_id for item in new_refs),
            rejection_ids_added=tuple(item.rejection_sha256 for item in new_rejections),
        )
        return self._commit(
            dataset_version=dataset_version,
            parent_version=parent.dataset_version,
            accepted_records=(*parent.accepted_records, *new_refs),
            rejection_log=(*parent.rejection_log, *new_rejections),
            operation=operation,
        )

    def delete_records(
        self,
        dataset_version: str,
        record_ids: Sequence[str],
        *,
        parent_version: str | None = None,
    ) -> DatasetVersionSnapshot:
        parent = self._parent(parent_version)
        requested = tuple(record_ids)
        if not requested or any(not item for item in requested):
            raise ValueError("record_ids must contain at least one non-empty record ID")
        if len(requested) != len(set(requested)):
            raise ValueError("record_ids must be unique")
        active = set(parent.accepted_record_ids)
        missing = sorted(set(requested) - active)
        if missing:
            raise UnknownDatasetVersionError(
                "cannot delete record IDs absent from parent version: " + ", ".join(missing)
            )
        retained = tuple(
            item for item in parent.accepted_records if item.record_id not in set(requested)
        )
        operation = LineageOperation(kind="DELETE", deleted_record_ids=tuple(sorted(requested)))
        return self._commit(
            dataset_version=dataset_version,
            parent_version=parent.dataset_version,
            accepted_records=retained,
            rejection_log=parent.rejection_log,
            operation=operation,
        )

    def supersede_records(
        self,
        dataset_version: str,
        replacements: Mapping[str, DatasetRecord],
        *,
        parent_version: str | None = None,
    ) -> DatasetVersionSnapshot:
        parent = self._parent(parent_version)
        if not replacements:
            raise ValueError("replacements must not be empty")
        active = set(parent.accepted_record_ids)
        missing = sorted(set(replacements) - active)
        if missing:
            raise UnknownDatasetVersionError(
                "cannot supersede record IDs absent from parent version: " + ", ".join(missing)
            )
        replacement_refs = {old: self._record_ref(new) for old, new in replacements.items()}
        retained_ids = active - set(replacements)
        replacement_ids = [item.record_id for item in replacement_refs.values()]
        replacement_hashes = [item.record_sha256 for item in replacement_refs.values()]
        if any(old == item.record_id for old, item in replacement_refs.items()):
            raise LineageMutationError("supersession must replace a record with a distinct record")
        if len(replacement_ids) != len(set(replacement_ids)):
            raise LineageIntegrityError("supersession introduces duplicate replacement record IDs")
        if len(replacement_hashes) != len(set(replacement_hashes)):
            raise LineageIntegrityError("supersession introduces duplicate replacement hashes")
        if any(item in retained_ids for item in replacement_ids):
            raise LineageMutationError("replacement record already exists in parent active set")
        result: list[AcceptedRecordRef] = []
        for item in parent.accepted_records:
            result.append(replacement_refs.get(item.record_id, item))
        operation = LineageOperation(
            kind="SUPERSEDE",
            superseded_records=tuple(
                sorted((old, ref.record_id) for old, ref in replacement_refs.items())
            ),
        )
        return self._commit(
            dataset_version=dataset_version,
            parent_version=parent.dataset_version,
            accepted_records=tuple(result),
            rejection_log=parent.rejection_log,
            operation=operation,
        )

    def rollback(
        self,
        dataset_version: str,
        target_version: str,
        *,
        parent_version: str | None = None,
    ) -> DatasetVersionSnapshot:
        parent = self._parent(parent_version)
        target = self.read(target_version)
        rejection_by_id = {item.rejection_sha256: item for item in parent.rejection_log}
        for item in target.rejection_log:
            rejection_by_id.setdefault(item.rejection_sha256, item)
        operation = LineageOperation(
            kind="ROLLBACK",
            rollback_target_version=target.dataset_version,
            rollback_target_manifest_sha256=target.manifest_sha256,
        )
        return self._commit(
            dataset_version=dataset_version,
            parent_version=parent.dataset_version,
            accepted_records=target.accepted_records,
            rejection_log=tuple(rejection_by_id.values()),
            operation=operation,
        )

    def read(
        self,
        dataset_version: str,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> DatasetVersionSnapshot:
        version = _require_version(dataset_version)
        snapshot = self._versions.get(version)
        if snapshot is None:
            raise UnknownDatasetVersionError(f"unknown dataset version: {version}")
        self.verify_version(version)
        if expected_manifest_sha256 is not None:
            if not _is_sha256(expected_manifest_sha256):
                raise ValueError("expected_manifest_sha256 must be a lowercase SHA-256 digest")
            if snapshot.manifest_sha256 != expected_manifest_sha256:
                raise LineageIntegrityError("read verification failed: manifest SHA-256 mismatch")
        return snapshot

    def manifest(self, dataset_version: str | None = None) -> dict[str, Any]:
        chosen = dataset_version or self._active_version
        if chosen is None:
            raise UnknownDatasetVersionError("lineage contains no dataset versions")
        return self.read(chosen).manifest()

    def verify_version(self, dataset_version: str) -> bool:
        snapshot = self._versions.get(dataset_version)
        if snapshot is None:
            raise UnknownDatasetVersionError(f"unknown dataset version: {dataset_version}")
        if snapshot.dataset_name != self.dataset_name:
            raise LineageIntegrityError("snapshot dataset_name mismatch")
        if snapshot.schema != LINEAGE_SCHEMA or snapshot.worker_id != LINEAGE_WORKER_ID:
            raise LineageIntegrityError("snapshot lineage schema or worker identity mismatch")
        if snapshot.classification != DATASET_CLASSIFICATION:
            raise LineageIntegrityError("snapshot classification mismatch")
        if snapshot.base_corpus_evidence or snapshot.canonical_base_training_eligible:
            raise BaseCorpusBoundaryError("lineage snapshot violates permanent Base ineligibility")
        if snapshot.training_use != LINEAGE_TRAINING_USE:
            raise LineageIntegrityError("snapshot training_use mismatch")
        if snapshot.parent_version is None:
            if snapshot.parent_manifest_sha256 != _GENESIS_MANIFEST_SHA256:
                raise LineageIntegrityError("genesis parent manifest hash mismatch")
        else:
            parent = self._versions.get(snapshot.parent_version)
            if parent is None:
                raise LineageIntegrityError("parent dataset version is missing")
            if parent.manifest_sha256 != snapshot.parent_manifest_sha256:
                raise LineageIntegrityError("parent dataset manifest hash mismatch")
        expected_manifest_sha256 = sha256_json(snapshot.manifest_body())
        if snapshot.manifest_sha256 != expected_manifest_sha256:
            raise LineageIntegrityError("dataset manifest SHA-256 verification failed")
        if not _is_sha256(snapshot.manifest_sha256):
            raise LineageIntegrityError("stored dataset manifest hash is invalid")
        for entry in snapshot.rejection_log:
            if entry.rejection_sha256 != sha256_json(entry.body()):
                raise LineageIntegrityError("rejection log hash verification failed")
        seen_record_ids: set[str] = set()
        seen_record_hashes: set[str] = set()
        for ref in snapshot.accepted_records:
            if ref.record_id in seen_record_ids or ref.record_sha256 in seen_record_hashes:
                raise LineageIntegrityError("duplicate accepted record detected during read verification")
            seen_record_ids.add(ref.record_id)
            seen_record_hashes.add(ref.record_sha256)
            record = self._record_objects.get(ref.record_sha256)
            canonical = self._record_canonical_json.get(ref.record_sha256)
            if record is None or canonical is None:
                raise LineageIntegrityError("accepted record payload is unavailable for verification")
            if canonical_json(record) != canonical or sha256_json(record) != ref.record_sha256:
                raise LineageIntegrityError("accepted record payload hash verification failed")
            expected_ref = self._record_ref(record)
            if expected_ref != ref:
                raise LineageIntegrityError("accepted record identity metadata mismatch")
        return True

    def verify_history(self) -> bool:
        for version in self._versions:
            self.verify_version(version)
        for version, snapshot in self._versions.items():
            seen: set[str] = set()
            current: DatasetVersionSnapshot | None = snapshot
            while current is not None:
                if current.dataset_version in seen:
                    raise LineageIntegrityError(f"dataset lineage cycle detected at {version}")
                seen.add(current.dataset_version)
                if current.parent_version is None:
                    break
                current = self._versions.get(current.parent_version)
                if current is None:
                    raise LineageIntegrityError("dataset lineage parent disappeared during verification")
        return True

    def as_base_corpus_evidence(self, dataset_version: str | None = None) -> None:
        if dataset_version is not None:
            self.read(dataset_version)
        raise BaseCorpusBoundaryError(
            "versioned synthetic teacher datasets are permanently POSTBASE/EXPERIMENTAL, "
            "canonical Base eligibility is always false"
        )
