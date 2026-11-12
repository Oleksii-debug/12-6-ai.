"""Bind one portable training packet to the native ordered training runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import torch
from torch import Tensor

from ..checkpoint import CheckpointIdentity, bind_checkpoint_identity, hash_json
from ..checkpoint.core import _require_exact_hex, _strict_json_object
from ..model import BaseInitManifest, InitSpec, ModelSpec, build_canonical_base
from .config import TrainingRecipe
from .exposure_accounting import OrderedExposureRuntime
from .resource_envelope import (
    DISK_FLOOR_DENIAL,
    ResourceAdmissionError,
    assess_resource_envelope,
    observe_host_resources,
)
from .run_control import RunBoundary, RunControlError, RunController, RunPhase, RunReceipt
from .trainer import Batch, Trainer

PACKET_SCHEMA = "12-6.portable-training-packet.v1"
EXECUTION_SCHEMA = "12-6.portable-training-execution.v1"
_COMMANDS = ("prepare", "run", "status")
_EVIDENCE = ("run_receipt", "checkpoint_manifest", "ordered_exposure_cursor")
_CHECKPOINT_POLICY = "atomic-per-optimizer-step-with-verified-resume"
_PROCESS_TRAINER_LOCK = Lock()


@contextmanager
def _exclusive_trainer() -> Iterator[None]:
    # Trainer configuration mutates process-global RNG and deterministic settings.
    if not _PROCESS_TRAINER_LOCK.acquire(blocking=False):
        raise RunControlError("another portable training operation is active in this process")
    try:
        yield
    finally:
        _PROCESS_TRAINER_LOCK.release()


@dataclass(frozen=True, slots=True)
class TrainingDataReferences:
    """Provider-independent identities for the tokenizer and ordered training shards."""

    tokenizer_version: str
    tokenizer_config_sha256: str
    tokenizer_vocab_sha256: str
    tokenizer_vocab_size: int
    dataset_manifest_sha256: str
    split_identity: str
    packing_version: str
    packing_sha256: str
    ledger_sha256: str
    exposure_plan_sha256: str
    shard_contract_sha256: str

    def __post_init__(self) -> None:
        for field in fields(self):
            if field.name.endswith("_sha256"):
                _require_exact_hex(getattr(self, field.name), field=field.name, lengths={64})


@dataclass(frozen=True, slots=True)
class PortableTrainingPacket:
    """The scientific inputs shared by every execution environment."""

    run_id: str
    stage: str
    source_git_sha: str
    environment_lock_sha256: str
    model_spec: ModelSpec
    init_spec: InitSpec
    initialization: BaseInitManifest
    recipe: TrainingRecipe
    data: TrainingDataReferences

    def __post_init__(self) -> None:
        if self.initialization.model_spec_sha256 != self.model_spec.identity_sha256():
            raise ValueError("initialization differs from the packet ModelSpec")
        if self.initialization.init_spec_sha256 != self.init_spec.identity_sha256():
            raise ValueError("initialization differs from the packet InitSpec")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PACKET_SCHEMA,
            "run_id": self.run_id,
            "stage": self.stage,
            "source_git_sha": self.source_git_sha,
            "environment_lock_sha256": self.environment_lock_sha256,
            "model_spec": self.model_spec.to_dict(),
            "init_spec": self.init_spec.to_dict(),
            "initialization": self.initialization.to_dict(),
            "recipe": self.recipe.to_dict(),
            "data": asdict(self.data),
            "checkpoint_policy": _CHECKPOINT_POLICY,
            "commands": list(_COMMANDS),
            "expected_evidence": list(_EVIDENCE),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PortableTrainingPacket:
        keys = {field.name for field in fields(cls)}
        if type(value) is not dict or value.keys() != keys | {
            "schema",
            "checkpoint_policy",
            "commands",
            "expected_evidence",
        }:
            raise ValueError("portable packet fields must match schema v1 exactly")
        if (
            value["schema"] != PACKET_SCHEMA
            or value["checkpoint_policy"] != _CHECKPOINT_POLICY
            or value["commands"] != list(_COMMANDS)
            or value["expected_evidence"] != list(_EVIDENCE)
        ):
            raise ValueError("portable packet schema, policy, commands or evidence differs")
        packet = cls(
            run_id=value["run_id"],
            stage=value["stage"],
            source_git_sha=value["source_git_sha"],
            environment_lock_sha256=value["environment_lock_sha256"],
            model_spec=ModelSpec.from_dict(value["model_spec"]),
            init_spec=InitSpec.from_dict(value["init_spec"]),
            initialization=BaseInitManifest.from_dict(value["initialization"]),
            recipe=TrainingRecipe.from_dict(value["recipe"]),
            data=TrainingDataReferences(**value["data"]),
        )
        if hash_json(value) != packet.identity_sha256():
            raise ValueError("portable packet must contain complete canonical records")
        packet.checkpoint_identity()
        return packet

    @classmethod
    def from_json(cls, payload: bytes) -> PortableTrainingPacket:
        return cls.from_dict(
            _strict_json_object(
                payload,
                artifact="portable training packet",
                invalid_message="invalid portable packet JSON",
            )
        )

    def identity_sha256(self) -> str:
        return hash_json(self.to_dict())

    def checkpoint_identity(self) -> CheckpointIdentity:
        data = self.data
        model = self.model_spec.to_dict()
        init = self.init_spec.to_dict()
        return bind_checkpoint_identity(
            run_manifest={
                "run_id": self.run_id,
                "stage": self.stage,
                "run_kind": PACKET_SCHEMA,
                "candidate": {
                    "git_sha": self.source_git_sha,
                    "modelspec_sha256": self.model_spec.identity_sha256(),
                    "initspec_sha256": self.init_spec.identity_sha256(),
                    "parameter_count": self.model_spec.parameter_count(),
                },
                "data": {
                    "dataset_manifest_sha256": data.dataset_manifest_sha256,
                    "split_identity": data.split_identity,
                    "tokenizer_sha256": data.tokenizer_config_sha256,
                    "tokenizer_vocab_sha256": data.tokenizer_vocab_sha256,
                    "tokenizer_version": data.tokenizer_version,
                    "packing_sha256": data.packing_sha256,
                    "packing_version": data.packing_version,
                },
                "training": self.recipe.run_training_fields(),
                "environment": {"lock_sha256": self.environment_lock_sha256},
                "portable_packet_sha256": self.identity_sha256(),
            },
            model_spec=model,
            init_spec=init,
            tokenizer_identity={
                "config_sha256": data.tokenizer_config_sha256,
                "vocab_sha256": data.tokenizer_vocab_sha256,
                "vocab_size": data.tokenizer_vocab_size,
                "version": data.tokenizer_version,
            },
            packing_identity={
                "config_sha256": data.packing_sha256,
                "version": data.packing_version,
                "sequence_length": self.recipe.sequence_length,
            },
            step=0,
            tokens_seen=0,
        )


AdapterKind = Literal["local", "free", "server"]


@dataclass(frozen=True, slots=True)
class TrainingExecutionSettings:
    """Host settings used by the selected adapter outside the packet identity."""

    adapter: AdapterKind
    provider: str
    device: str
    run_root: Path
    environment_lock_sha256: str

    def __post_init__(self) -> None:
        if self.adapter not in ("local", "free", "server"):
            raise ValueError("unsupported portable training adapter")
        for name in ("provider", "device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"execution {name} must be non-empty text")
        if not isinstance(self.run_root, Path):
            raise TypeError("execution run_root must be a path")
        _require_exact_hex(
            self.environment_lock_sha256,
            field="environment_lock_sha256",
            lengths={64},
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "schema": EXECUTION_SCHEMA, "run_root": str(self.run_root)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainingExecutionSettings:
        keys = {field.name for field in fields(cls)}
        if type(value) is not dict or value.keys() != keys | {"schema"}:
            raise ValueError("execution fields must match schema v1 exactly")
        if value["schema"] != EXECUTION_SCHEMA:
            raise ValueError("unsupported portable training execution schema")
        path = value["run_root"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError("execution run_root must be non-empty path text")
        return cls(
            adapter=value["adapter"],
            provider=value["provider"],
            device=value["device"],
            run_root=Path(path),
            environment_lock_sha256=value["environment_lock_sha256"],
        )


BatchSource = Callable[[TrainingExecutionSettings, int], Iterable[Batch]]


class PortableTrainingRun:
    """Open a trusted packet on a caller-supplied batch source for one environment."""

    def __init__(
        self,
        packet: PortableTrainingPacket,
        settings: TrainingExecutionSettings,
        *,
        expected_packet_sha256: str,
        sources: Mapping[AdapterKind, BatchSource],
        ledger: Mapping[str, Any],
        exposure_plan: Mapping[str, Any],
        shard_contract: Mapping[str, Any],
    ) -> None:
        packet = PortableTrainingPacket.from_dict(packet.to_dict())
        settings = TrainingExecutionSettings.from_dict(settings.to_dict())
        if packet.identity_sha256() != expected_packet_sha256:
            raise ValueError("portable packet differs from its expected SHA-256")
        if settings.environment_lock_sha256 != packet.environment_lock_sha256:
            raise ValueError("execution environment differs from the packet runtime lock")
        source = sources.get(settings.adapter)
        if source is None:
            raise ValueError("no batch source is registered for the selected adapter")
        if not callable(source):
            raise TypeError("portable training batch source must be callable")
        data = packet.data
        if ledger.get("packing_identity_sha256") != data.packing_sha256:
            raise ValueError("exposure ledger differs from the packet packing contract")
        self._packet = packet
        self._settings = settings
        self._source = source
        self._identity = packet.checkpoint_identity()
        self._controller = RunController(settings.run_root, self._identity.run_manifest_hash)
        self._runtime = OrderedExposureRuntime(
            ledger=ledger,
            plan=exposure_plan,
            content_manifest=shard_contract,
            expected_ledger_sha256=data.ledger_sha256,
            expected_plan_sha256=data.exposure_plan_sha256,
            expected_content_sha256=data.shard_contract_sha256,
            unique_target_budget=packet.recipe.target_unique_loss_positions,
        )

    def _batches(self, index: int) -> Iterable[Batch]:
        recipe = self._packet.recipe
        for batch in self._source(self._settings, index):
            inputs = batch.get("input_ids")
            if (
                not isinstance(inputs, Tensor)
                or inputs.ndim != 2
                or not 0 < inputs.shape[0] <= recipe.microbatch_size
                or not 0 < inputs.shape[1] <= recipe.sequence_length
            ):
                raise ValueError("batch dimensions exceed the portable training recipe")
            yield batch

    @property
    def run_manifest_sha256(self) -> str:
        return self._identity.run_manifest_hash

    def _resource_report(
        self,
        *,
        remaining_updates: int,
        initial_checkpoint_needed: bool,
    ) -> dict[str, Any]:
        checkpoint_root = self._controller.namespace / "checkpoints"
        observed = observe_host_resources(checkpoint_root, self._settings.device)
        return assess_resource_envelope(
            self._packet.model_spec,
            observed,
            remaining_updates=remaining_updates,
            initial_checkpoint_needed=initial_checkpoint_needed,
        )

    def _trainer(
        self,
        *,
        remaining_updates: int,
        initial_checkpoint_needed: bool,
        defer_storage_check: bool = False,
    ) -> Trainer:
        packet = self._packet
        if torch.get_default_dtype() != torch.float32 or torch.get_default_device().type != "cpu":
            raise ResourceAdmissionError("native model allocation requires default FP32 on CPU")
        resources = self._resource_report(
            remaining_updates=remaining_updates,
            initial_checkpoint_needed=initial_checkpoint_needed,
        )
        if packet.recipe.trainer.precision == "fp16" and not resources["resources"][
            "device"
        ].startswith("cuda"):
            raise ResourceAdmissionError("FP16 training requires a CUDA device")
        denials = [
            reason
            for reason in resources["denials"]
            if not (defer_storage_check and reason == DISK_FLOOR_DENIAL)
        ]
        if denials:
            raise ResourceAdmissionError(
                f"native training resource admission failed for {packet.stage}: "
                + ", ".join(denials)
            )
        model = build_canonical_base(packet.model_spec, packet.init_spec, packet.initialization)
        return Trainer(model, packet.recipe.trainer, device=self._settings.device)

    def _admit_recovered_storage(self, current: RunReceipt) -> None:
        resources = self._resource_report(
            remaining_updates=self._packet.recipe.trainer.max_steps - current.committed_step,
            initial_checkpoint_needed=False,
        )
        if DISK_FLOOR_DENIAL in resources["denials"]:
            raise ResourceAdmissionError(
                f"native training resource admission failed for {self._packet.stage}: "
                f"{DISK_FLOOR_DENIAL}"
            )

    def prepare(self) -> RunReceipt:
        with _exclusive_trainer():
            trainer = self._trainer(
                remaining_updates=self._packet.recipe.trainer.max_steps,
                initial_checkpoint_needed=True,
            )
            return self._runtime.prepare(self._controller, trainer, self._identity)

    def run(
        self,
        *,
        expected_receipt_sha256: str,
        recover: bool = False,
        control: Callable[[RunBoundary], RunPhase | None] | None = None,
    ) -> RunReceipt:
        with _exclusive_trainer():
            current = self._controller.status()
            if current.receipt_sha256 != expected_receipt_sha256:
                raise RunControlError("stale run receipt")
            remaining = self._packet.recipe.trainer.max_steps - current.committed_step
            pending_recovery = (
                recover and current.pending_step is not None and current.pending_tokens is not None
            )
            return self._runtime.run(
                self._controller,
                self._trainer(
                    remaining_updates=remaining,
                    initial_checkpoint_needed=False,
                    defer_storage_check=pending_recovery,
                ),
                self._identity,
                self._batches,
                expected_receipt_sha256=expected_receipt_sha256,
                recover=recover,
                control=control,
                admit_progress=self._admit_recovered_storage if pending_recovery else None,
            )

    def status(self) -> RunReceipt:
        return self._controller.status()
