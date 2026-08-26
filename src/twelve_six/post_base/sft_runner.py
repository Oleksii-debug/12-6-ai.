"""Mechanics-only supervised fine-tuning runner for the post-Base communication layer.

The runner is deliberately framework-neutral. It exercises the orchestration boundary
around a backend supplied by a later model-adapter worker, while enforcing that this
worker can run only tiny local/project-owned fixtures. It never receives the canonical
Base path as a mutable training target.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from twelve_six.post_base.contract import (
    POST_BASE_ARTIFACT_NAMESPACE,
    POST_BASE_EVIDENCE_NAMESPACE,
    DatasetProvenance,
    PostBaseConsumptionContract,
    PostBaseStage,
    PreparedPostBaseWorkspace,
    TokenizerCompatibility,
    prepare_post_base_workspace,
    snapshot_directory,
)
from twelve_six.posttraining.contracts import CheckpointRef, ComputeClass

SFT_MECHANICS_SCHEMA = "12-6.post-base.sft-mechanics.v1"
SFT_CHECKPOINT_NAMESPACE = f"{POST_BASE_ARTIFACT_NAMESPACE}/sft/checkpoints"
SFT_EVALUATION_NAMESPACE = f"{POST_BASE_EVIDENCE_NAMESPACE}/sft/evaluations"
MAX_MECHANICS_STEPS = 32


def _require_text(value: str, *, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_bytes(_canonical_bytes(dict(payload)))
    os.replace(temporary, path)


def _normalized_metrics(metrics: Mapping[str, float], *, field: str) -> dict[str, float]:
    if not metrics:
        raise ValueError(f"{field} metrics must be non-empty")
    normalized: dict[str, float] = {}
    for name, value in metrics.items():
        _require_text(name, field=f"{field} metric name")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} metric {name!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{field} metric {name!r} must be finite")
        normalized[name] = numeric
    return normalized


class FixtureSourceKind(StrEnum):
    PROJECT_OWNED = "project_owned"
    SYNTHETIC_LOCAL = "synthetic_local"


class MechanicsSplit(StrEnum):
    TRAIN = "train"
    EVALUATION = "evaluation"


class SFTExecutionMode(StrEnum):
    MECHANICS_FIXTURE_ONLY = "mechanics_fixture_only"


@dataclass(frozen=True, slots=True)
class FixtureProvenance:
    source_id: str
    source_kind: FixtureSourceKind
    source_revision: str
    generator_id: str | None = None
    foreign_model_output: bool = False

    def __post_init__(self) -> None:
        _require_text(self.source_id, field="source_id")
        _require_text(self.source_revision, field="source_revision")
        if self.foreign_model_output:
            raise ValueError("POSTBASE-353 forbids foreign-model output in mechanics fixtures")
        if self.source_kind is FixtureSourceKind.SYNTHETIC_LOCAL:
            if not (self.generator_id and self.generator_id.strip()):
                raise ValueError("synthetic_local fixtures require a local generator_id")
        elif self.generator_id is not None:
            raise ValueError("project_owned fixtures must not claim a synthetic generator")


@dataclass(frozen=True, slots=True)
class SFTMechanicsExample:
    record_id: str
    split: MechanicsSplit
    user_text: str
    assistant_text: str
    provenance: FixtureProvenance

    def __post_init__(self) -> None:
        _require_text(self.record_id, field="record_id")
        _require_text(self.user_text, field="user_text")
        _require_text(self.assistant_text, field="assistant_text")

    def canonical_record(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "split": self.split.value,
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "provenance": {
                "source_id": self.provenance.source_id,
                "source_kind": self.provenance.source_kind.value,
                "source_revision": self.provenance.source_revision,
                "generator_id": self.provenance.generator_id,
                "foreign_model_output": self.provenance.foreign_model_output,
            },
        }


@dataclass(frozen=True, slots=True)
class SFTMechanicsDataset:
    """Tiny local/project-owned fixture bundle used only to prove runner mechanics."""

    dataset_id: str
    train: tuple[SFTMechanicsExample, ...]
    evaluation: tuple[SFTMechanicsExample, ...]
    fixture_only: bool = True

    def __post_init__(self) -> None:
        _require_text(self.dataset_id, field="dataset_id")
        if not self.fixture_only:
            raise ValueError("POSTBASE-353 accepts mechanics fixtures only")
        if not self.train or not self.evaluation:
            raise ValueError("mechanics fixtures require non-empty train and evaluation splits")
        if any(record.split is not MechanicsSplit.TRAIN for record in self.train):
            raise ValueError("train fixtures must declare split=train")
        if any(record.split is not MechanicsSplit.EVALUATION for record in self.evaluation):
            raise ValueError("evaluation fixtures must declare split=evaluation")
        record_ids = [record.record_id for record in (*self.train, *self.evaluation)]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("mechanics fixture record_id values must be globally unique")
        if self.train_split_sha256 == self.evaluation_split_sha256:
            raise ValueError("training and evaluation fixture identities must be distinct")

    @staticmethod
    def _split_identity(records: tuple[SFTMechanicsExample, ...]) -> str:
        return _sha256([record.canonical_record() for record in records])

    @property
    def train_split_sha256(self) -> str:
        return self._split_identity(self.train)

    @property
    def evaluation_split_sha256(self) -> str:
        return self._split_identity(self.evaluation)

    @property
    def source_registry_sha256(self) -> str:
        sources = {
            (
                record.provenance.source_id,
                record.provenance.source_kind.value,
                record.provenance.source_revision,
                record.provenance.generator_id,
            )
            for record in (*self.train, *self.evaluation)
        }
        payload = [
            {
                "source_id": source_id,
                "source_kind": source_kind,
                "source_revision": source_revision,
                "generator_id": generator_id,
                "foreign_model_output": False,
            }
            for source_id, source_kind, source_revision, generator_id in sorted(
                sources,
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2],
                    item[3] or "",
                ),
            )
        ]
        return _sha256(payload)

    @property
    def manifest_sha256(self) -> str:
        return _sha256(
            {
                "schema": SFT_MECHANICS_SCHEMA,
                "dataset_id": self.dataset_id,
                "fixture_only": self.fixture_only,
                "train_count": len(self.train),
                "evaluation_count": len(self.evaluation),
                "train_split_sha256": self.train_split_sha256,
                "evaluation_split_sha256": self.evaluation_split_sha256,
                "source_registry_sha256": self.source_registry_sha256,
            }
        )

    def to_contract_provenance(self) -> DatasetProvenance:
        return DatasetProvenance(
            dataset_id=self.dataset_id,
            manifest_sha256=self.manifest_sha256,
            source_registry_sha256=self.source_registry_sha256,
            train_split_sha256=self.train_split_sha256,
            evaluation_split_sha256=self.evaluation_split_sha256,
        )

    def require_contract_match(self, contract: PostBaseConsumptionContract) -> None:
        expected = self.to_contract_provenance()
        if contract.dataset != expected:
            raise ValueError("mechanics fixture identities do not match the communication contract")


@dataclass(frozen=True, slots=True)
class SFTMechanicsPlan:
    run_id: str
    backend_id: str
    contract: PostBaseConsumptionContract
    tokenizer: TokenizerCompatibility
    max_steps: int
    seed: int
    execution_mode: SFTExecutionMode = SFTExecutionMode.MECHANICS_FIXTURE_ONLY
    compute_class: ComputeClass = ComputeClass.LOCAL_FREE
    checkpoint_namespace: str = SFT_CHECKPOINT_NAMESPACE
    evaluation_namespace: str = SFT_EVALUATION_NAMESPACE
    real_campaign_authorization_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.run_id, field="run_id")
        _require_text(self.backend_id, field="backend_id")
        if self.contract.stage is not PostBaseStage.COMMUNICATION_SUPERVISION:
            raise ValueError("SFT mechanics require the communication supervision stage")
        self.contract.require_tokenizer(self.tokenizer)
        if not 1 <= self.max_steps <= MAX_MECHANICS_STEPS:
            raise ValueError(f"max_steps must be between 1 and {MAX_MECHANICS_STEPS}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.execution_mode is not SFTExecutionMode.MECHANICS_FIXTURE_ONLY:
            raise ValueError("POSTBASE-353 authorizes mechanics_fixture_only mode only")
        if self.compute_class is not ComputeClass.LOCAL_FREE:
            raise ValueError("POSTBASE-353 is LOCAL_FREE only")
        if self.checkpoint_namespace != SFT_CHECKPOINT_NAMESPACE:
            raise ValueError("SFT checkpoints must stay in the dedicated post-Base namespace")
        if self.evaluation_namespace != SFT_EVALUATION_NAMESPACE:
            raise ValueError(
                "SFT evaluation must stay in the dedicated post-Base evidence namespace"
            )
        if self.real_campaign_authorization_id is not None:
            raise ValueError("POSTBASE-353 does not authorize a real communication campaign")
        if self.contract.execution_authorized:
            raise ValueError("communication-consumption v1 must remain execution_authorized=false")


@runtime_checkable
class SFTMechanicsBackend(Protocol):
    """Minimal backend contract used by the mechanics runner.

    A real model adapter may implement this later. POSTBASE-353 tests use a tiny
    deterministic scalar fixture backend only.
    """

    backend_id: str

    def load_input_checkpoint(self, checkpoint_root: Path) -> object:
        ...

    def train_step(
        self,
        state: object,
        example: SFTMechanicsExample,
        *,
        step: int,
        seed: int,
    ) -> Mapping[str, float]:
        ...

    def evaluate(
        self,
        state: object,
        examples: Sequence[SFTMechanicsExample],
    ) -> Mapping[str, float]:
        ...

    def save_checkpoint(self, state: object, checkpoint_root: Path) -> None:
        ...

    def load_checkpoint(self, checkpoint_root: Path) -> object:
        ...


@dataclass(frozen=True, slots=True)
class CheckpointGeneration:
    generation: int
    path: Path
    manifest_sha256: str
    backend_snapshot_sha256: str


class SFTCheckpointStore:
    """Immutable generation store with a mutable active-generation pointer."""

    def __init__(self, experiment_root: Path, run_id: str, *, create: bool) -> None:
        _require_text(run_id, field="run_id")
        self.run_id = run_id
        self.experiment_root = experiment_root.resolve()
        self.root = (self.experiment_root / SFT_CHECKPOINT_NAMESPACE).resolve()
        self.active_pointer = self.root / "active.json"
        if create:
            if self.root.exists():
                raise FileExistsError("SFT checkpoint namespace already exists")
            self.root.mkdir(parents=True)
        elif not self.root.is_dir():
            raise FileNotFoundError("SFT checkpoint namespace does not exist")

    @staticmethod
    def _generation_name(generation: int) -> str:
        if generation < 0:
            raise ValueError("generation must be non-negative")
        return f"generation_{generation:06d}"

    def generation_path(self, generation: int) -> Path:
        return self.root / self._generation_name(generation)

    def _read_manifest(self, generation: int) -> dict[str, object]:
        path = self.generation_path(generation) / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint generation {generation} is missing")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != SFT_MECHANICS_SCHEMA:
            raise ValueError("checkpoint manifest schema mismatch")
        if payload.get("run_id") != self.run_id:
            raise ValueError("checkpoint generation belongs to a different run")
        if payload.get("generation") != generation:
            raise ValueError("checkpoint manifest generation mismatch")
        return payload

    def active_generation(self) -> int:
        if not self.active_pointer.is_file():
            raise FileNotFoundError("active checkpoint pointer is missing")
        payload = json.loads(self.active_pointer.read_text(encoding="utf-8"))
        if payload.get("run_id") != self.run_id:
            raise ValueError("active checkpoint pointer belongs to a different run")
        generation = payload.get("generation")
        if not isinstance(generation, int) or generation < 0:
            raise ValueError("active checkpoint pointer contains an invalid generation")
        self._read_manifest(generation)
        return generation

    def _activate(self, generation: int, *, reason: str) -> None:
        manifest = self._read_manifest(generation)
        manifest_sha256 = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
        _atomic_json(
            self.active_pointer,
            {
                "schema": SFT_MECHANICS_SCHEMA,
                "run_id": self.run_id,
                "generation": generation,
                "manifest_sha256": manifest_sha256,
                "reason": reason,
            },
        )

    def publish(
        self,
        backend: SFTMechanicsBackend,
        state: object,
        *,
        generation: int,
        parent_generation: int | None,
        input_snapshot_sha256: str,
        base_checkpoint: CheckpointRef,
        dataset_manifest_sha256: str,
    ) -> CheckpointGeneration:
        destination = self.generation_path(generation)
        if destination.exists():
            raise FileExistsError(f"checkpoint generation {generation} already exists")
        if generation == 0:
            if parent_generation is not None:
                raise ValueError("generation zero cannot have a parent generation")
            if self.active_pointer.exists():
                raise ValueError("generation zero must be the first published checkpoint")
        else:
            if parent_generation is None:
                raise ValueError("nonzero checkpoint generations require a parent")
            if self.active_generation() != parent_generation:
                raise ValueError("checkpoint parent must equal the active generation")

        temporary = self.root / f".{self._generation_name(generation)}.tmp"
        if temporary.exists():
            raise FileExistsError("checkpoint publication temporary path already exists")
        try:
            backend_root = temporary / "backend"
            backend_root.mkdir(parents=True)
            backend.save_checkpoint(state, backend_root)
            backend_snapshot = snapshot_directory(backend_root)
            manifest: dict[str, object] = {
                "schema": SFT_MECHANICS_SCHEMA,
                "run_id": self.run_id,
                "generation": generation,
                "parent_generation": parent_generation,
                "backend_id": backend.backend_id,
                "backend_snapshot_sha256": backend_snapshot.identity_sha256,
                "input_snapshot_sha256": input_snapshot_sha256,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "base_checkpoint": {
                    "checkpoint_id": base_checkpoint.checkpoint_id,
                    "sha256": base_checkpoint.sha256,
                    "git_sha": base_checkpoint.git_sha,
                    "stage": base_checkpoint.stage,
                    "lineage": base_checkpoint.lineage.value,
                },
            }
            manifest_bytes = _canonical_bytes(manifest)
            (temporary / "manifest.json").write_bytes(manifest_bytes)
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            temporary.rename(destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        self._activate(generation, reason="publish")
        return CheckpointGeneration(
            generation=generation,
            path=destination,
            manifest_sha256=manifest_sha256,
            backend_snapshot_sha256=backend_snapshot.identity_sha256,
        )

    def rollback_to(self, generation: int) -> Path:
        current = self.active_generation()
        if generation > current:
            raise ValueError("rollback target cannot be newer than the active generation")
        self._activate(generation, reason=f"rollback_from_{current}")
        return self.generation_path(generation)

    def load_state(self, backend: SFTMechanicsBackend, generation: int) -> object:
        manifest = self._read_manifest(generation)
        if manifest.get("backend_id") != backend.backend_id:
            raise ValueError("checkpoint backend_id does not match the requested backend")
        backend_root = self.generation_path(generation) / "backend"
        current_snapshot = snapshot_directory(backend_root)
        expected = manifest.get("backend_snapshot_sha256")
        if current_snapshot.identity_sha256 != expected:
            raise RuntimeError("checkpoint generation bytes do not match its immutable manifest")
        return backend.load_checkpoint(backend_root)


class SFTEvaluationStore:
    """Immutable evaluation evidence separated from checkpoint artifacts."""

    def __init__(self, experiment_root: Path, run_id: str, *, create: bool) -> None:
        _require_text(run_id, field="run_id")
        self.run_id = run_id
        self.experiment_root = experiment_root.resolve()
        self.root = (self.experiment_root / SFT_EVALUATION_NAMESPACE).resolve()
        if create:
            if self.root.exists():
                raise FileExistsError("SFT evaluation namespace already exists")
            self.root.mkdir(parents=True)
        elif not self.root.is_dir():
            raise FileNotFoundError("SFT evaluation namespace does not exist")

    def publish(
        self,
        *,
        generation: int,
        phase: str,
        metrics: Mapping[str, float],
        dataset: SFTMechanicsDataset,
    ) -> Path:
        _require_text(phase, field="evaluation phase")
        normalized = _normalized_metrics(metrics, field=f"evaluation/{phase}")
        path = self.root / f"generation_{generation:06d}_{phase}.json"
        if path.exists():
            raise FileExistsError("evaluation evidence is immutable")
        payload: dict[str, object] = {
            "schema": SFT_MECHANICS_SCHEMA,
            "run_id": self.run_id,
            "generation": generation,
            "phase": phase,
            "evaluation_split_sha256": dataset.evaluation_split_sha256,
            "metrics": normalized,
        }
        path.write_bytes(_canonical_bytes(payload))
        return path


@dataclass(frozen=True, slots=True)
class SFTStepReceipt:
    step: int
    record_id: str
    metrics: Mapping[str, float]
    checkpoint: CheckpointGeneration


@dataclass(frozen=True, slots=True)
class SFTMechanicsReceipt:
    run_id: str
    workspace: PreparedPostBaseWorkspace
    checkpoint_namespace: Path
    evaluation_namespace: Path
    baseline_checkpoint: CheckpointGeneration
    final_checkpoint: CheckpointGeneration
    baseline_evaluation: Path
    final_evaluation: Path
    baseline_metrics: Mapping[str, float]
    final_metrics: Mapping[str, float]
    steps: tuple[SFTStepReceipt, ...]
    rollback_generation: int
    rollback_base_checkpoint: CheckpointRef
    canonical_snapshot_before: str
    canonical_snapshot_after: str

    @property
    def training_steps(self) -> int:
        return len(self.steps)


def run_sft_mechanics(
    *,
    plan: SFTMechanicsPlan,
    dataset: SFTMechanicsDataset,
    canonical_base_root: Path,
    experiment_root: Path,
    backend: SFTMechanicsBackend,
) -> SFTMechanicsReceipt:
    """Exercise SFT orchestration on a disjoint cloned fixture checkpoint.

    This function is intentionally unavailable for real communication campaigns in
    POSTBASE-353. The plan and dataset contracts both fail closed outside fixture-only,
    LOCAL_FREE mechanics qualification.
    """

    if backend.backend_id != plan.backend_id:
        raise ValueError("backend_id does not match the mechanics plan")
    dataset.require_contract_match(plan.contract)

    canonical_before = snapshot_directory(canonical_base_root)
    workspace = prepare_post_base_workspace(canonical_base_root, experiment_root)
    checkpoint_store = SFTCheckpointStore(workspace.experiment_root, plan.run_id, create=True)
    evaluation_store = SFTEvaluationStore(workspace.experiment_root, plan.run_id, create=True)

    try:
        state = backend.load_input_checkpoint(workspace.cloned_checkpoint_root)
        baseline_checkpoint = checkpoint_store.publish(
            backend,
            state,
            generation=0,
            parent_generation=None,
            input_snapshot_sha256=workspace.cloned_snapshot_sha256,
            base_checkpoint=plan.contract.base_checkpoint,
            dataset_manifest_sha256=dataset.manifest_sha256,
        )
        baseline_metrics = _normalized_metrics(
            backend.evaluate(state, dataset.evaluation),
            field="baseline evaluation",
        )
        baseline_evaluation = evaluation_store.publish(
            generation=0,
            phase="baseline",
            metrics=baseline_metrics,
            dataset=dataset,
        )

        step_receipts: list[SFTStepReceipt] = []
        parent_generation = 0
        for step in range(1, plan.max_steps + 1):
            example = dataset.train[(step - 1) % len(dataset.train)]
            metrics = _normalized_metrics(
                backend.train_step(state, example, step=step, seed=plan.seed),
                field=f"train step {step}",
            )
            checkpoint = checkpoint_store.publish(
                backend,
                state,
                generation=step,
                parent_generation=parent_generation,
                input_snapshot_sha256=workspace.cloned_snapshot_sha256,
                base_checkpoint=plan.contract.base_checkpoint,
                dataset_manifest_sha256=dataset.manifest_sha256,
            )
            step_receipts.append(
                SFTStepReceipt(
                    step=step,
                    record_id=example.record_id,
                    metrics=metrics,
                    checkpoint=checkpoint,
                )
            )
            parent_generation = step

        final_checkpoint = step_receipts[-1].checkpoint
        final_metrics = _normalized_metrics(
            backend.evaluate(state, dataset.evaluation),
            field="final evaluation",
        )
        final_evaluation = evaluation_store.publish(
            generation=final_checkpoint.generation,
            phase="final",
            metrics=final_metrics,
            dataset=dataset,
        )
    finally:
        canonical_after = snapshot_directory(canonical_base_root)
        cloned_after = snapshot_directory(workspace.cloned_checkpoint_root)
        if (
            canonical_before.identity_sha256 != canonical_after.identity_sha256
            or canonical_before.files != canonical_after.files
        ):
            raise RuntimeError("canonical Base mutated during post-Base SFT mechanics")
        if (
            workspace.cloned_snapshot_sha256 != cloned_after.identity_sha256
            or canonical_before.files != cloned_after.files
        ):
            raise RuntimeError("post-Base input checkpoint clone mutated during SFT mechanics")

    return SFTMechanicsReceipt(
        run_id=plan.run_id,
        workspace=workspace,
        checkpoint_namespace=checkpoint_store.root,
        evaluation_namespace=evaluation_store.root,
        baseline_checkpoint=baseline_checkpoint,
        final_checkpoint=final_checkpoint,
        baseline_evaluation=baseline_evaluation,
        final_evaluation=final_evaluation,
        baseline_metrics=baseline_metrics,
        final_metrics=final_metrics,
        steps=tuple(step_receipts),
        rollback_generation=0,
        rollback_base_checkpoint=plan.contract.rollback_checkpoint,
        canonical_snapshot_before=canonical_before.identity_sha256,
        canonical_snapshot_after=canonical_after.identity_sha256,
    )
