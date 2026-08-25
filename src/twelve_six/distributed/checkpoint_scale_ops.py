"""Operational scale layer for the D18 distributed checkpoint incumbent.

D18 owns the DCP model/optimizer data plane. This module deliberately does not
implement a second DCP writer. It adds training-control continuation, v1 migration
guards, retention, and stage-triggered adoption policy around that incumbent.
"""

from __future__ import annotations

import copy
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from twelve_six.checkpoint.core import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    sha256_file,
    verify_checkpoint,
)

from .contracts import ParallelPlan
from .dcp_checkpoint import (
    COMMITTED,
    LoadResult as DcpLoadResult,
    ResumeMode,
    ScaleCheckpointIdentity,
    load_scale_checkpoint,
    save_scale_checkpoint,
    verify_scale_checkpoint,
)

CONTROL_SCHEMA = "12-6.scale-resume-control.v1"
POLICY_SCHEMA = "12-6.checkpoint-scale-policy.v1"


@dataclass(frozen=True, slots=True)
class CheckpointScalePolicy:
    schema: str
    parameter_count: int
    distributed: bool
    training_resume_format: str
    checkpoint_v1_status: str
    d18_status: str
    async_status: str
    object_storage_status: str
    rationale: str


@dataclass(frozen=True, slots=True)
class AsyncCheckpointGate:
    supported: bool
    max_in_flight: int
    reason: str
    required_evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainerScaleLoadResult:
    dcp: DcpLoadResult
    trainer_control_restored: bool
    rng_restored: bool
    exact_training_continuation_claim_allowed: bool


@dataclass(frozen=True, slots=True)
class V1ScaleMigrationProvenance:
    """External identities needed because checkpoint-v1 did not split these fields."""

    source_run_manifest_sha256: str
    init_spec_sha256: str
    packing_sha256: str

    def validate(self) -> None:
        _require_sha256(self.source_run_manifest_sha256, "source_run_manifest_sha256")
        _require_sha256(self.init_spec_sha256, "init_spec_sha256")
        _require_sha256(self.packing_sha256, "packing_sha256")


@dataclass(frozen=True, slots=True)
class ScaleRetentionPolicy:
    keep_last: int = 2
    keep_every_n_steps: int | None = None

    def validate(self) -> None:
        if not isinstance(self.keep_last, int) or isinstance(self.keep_last, bool):
            raise TypeError("keep_last must be an integer")
        if self.keep_last < 1:
            raise ValueError("keep_last must be positive")
        if self.keep_every_n_steps is not None:
            if not isinstance(self.keep_every_n_steps, int) or isinstance(
                self.keep_every_n_steps, bool
            ):
                raise TypeError("keep_every_n_steps must be an integer or None")
            if self.keep_every_n_steps < 1:
                raise ValueError("keep_every_n_steps must be positive")


@dataclass(frozen=True, slots=True)
class ScaleGeneration:
    path: Path
    step: int
    tokens_seen: int
    aggregate_checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class ScaleRetentionPlan:
    keep: tuple[ScaleGeneration, ...]
    delete: tuple[ScaleGeneration, ...]
    uncommitted_staging: tuple[Path, ...]


def checkpoint_scale_policy(
    parameter_count: int,
    *,
    distributed: bool = False,
) -> CheckpointScalePolicy:
    """Choose the training-resume path by scale instead of replacing v1 globally."""

    if not isinstance(parameter_count, int) or isinstance(parameter_count, bool):
        raise TypeError("parameter_count must be an integer")
    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    if not isinstance(distributed, bool):
        raise TypeError("distributed must be bool")

    if distributed:
        training_format = "D18_DCP_REQUIRED"
        v1_status = "NOT_A_DISTRIBUTED_TRAINING_RESUME_FORMAT"
        rationale = "distributed writers/topology require the D18 DCP successor"
    elif parameter_count < 500_000:
        training_format = "CHECKPOINT_V1"
        v1_status = "PREFERRED_FOR_SMALL_SINGLE_PROCESS"
        rationale = "small checkpoints still benefit from v1 immutable-byte preflight"
    elif parameter_count < 5_000_000:
        training_format = "DUAL_QUALIFICATION_PREFER_D18_FOR_SCALE_READINESS"
        v1_status = "ALLOWED_WITH_MEASURED_HOST_RAM"
        rationale = "~1M is the qualification crossover rather than a forced format cutover"
    else:
        training_format = "D18_DCP_REQUIRED_FOR_TRAINING_RESUME"
        v1_status = "COMPATIBILITY_ONLY_NOT_DEFAULT_TRAINING_RESUME"
        rationale = "10M+ should not retain whole-checkpoint verified payload bytes in host RAM"

    return CheckpointScalePolicy(
        schema=POLICY_SCHEMA,
        parameter_count=parameter_count,
        distributed=distributed,
        training_resume_format=training_format,
        checkpoint_v1_status=v1_status,
        d18_status="INCUMBENT_DCP_DATA_PLANE",
        async_status="BLOCKED_PENDING_D18_ASYNC_STAGING_AND_OVERLAP_EVIDENCE",
        object_storage_status="BLOCKED_PENDING_STORAGE_ADAPTER_AND_COMMIT_PROTOCOL",
        rationale=rationale,
    )


def async_checkpoint_gate() -> AsyncCheckpointGate:
    """Fail closed instead of wrapping synchronous D18 save in an unsafe background thread."""

    return AsyncCheckpointGate(
        supported=False,
        max_in_flight=1,
        reason="D18 currently exposes a synchronous committed-generation writer only",
        required_evidence=(
            "DCP async_save integration without live-model mutation races",
            "measured peak host staging bytes on target accelerator topology",
            "measured training-step overlap and checkpoint completion latency",
            "one-in-flight backpressure and failure propagation",
            "exact resume after interrupted async save",
        ),
    )


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be exact lowercase SHA-256")
    return value


def _json_safe(value: Any, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains non-finite float")
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{field} keys must be non-empty strings")
            output[key] = _json_safe(item, f"{field}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{field}[]") for item in value]
    raise TypeError(f"{field} contains unsupported type {type(value).__name__}")


def _encode_state(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "$kind": "numpy.ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data_hex": array.tobytes(order="C").hex(),
        }
    cls = value.__class__
    if cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}:
        tensor = value.detach().cpu().contiguous()
        if str(tensor.dtype) != "torch.uint8":
            raise TypeError("scale resume RNG codec only accepts torch.uint8 tensors")
        return {
            "$kind": "torch.uint8",
            "shape": list(tensor.shape),
            "data_hex": tensor.numpy().tobytes().hex(),
        }
    if isinstance(value, tuple):
        return {"$kind": "tuple", "items": [_encode_state(item) for item in value]}
    if isinstance(value, list):
        return [_encode_state(item) for item in value]
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("scale resume state mapping keys must be non-empty strings")
            output[key] = _encode_state(item)
        return output
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scale resume state contains non-finite float")
        return value
    raise TypeError(f"unsupported scale resume state type {type(value).__name__}")


def _decode_state(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_state(item) for item in value]
    if isinstance(value, Mapping):
        kind = value.get("$kind")
        if kind == "tuple":
            if set(value) != {"$kind", "items"} or not isinstance(value["items"], list):
                raise ValueError("invalid tuple state encoding")
            return tuple(_decode_state(item) for item in value["items"])
        if kind == "numpy.ndarray":
            if set(value) != {"$kind", "dtype", "shape", "data_hex"}:
                raise ValueError("invalid NumPy state encoding")
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(item) for item in value["shape"])
            raw = bytes.fromhex(value["data_hex"])
            array = np.frombuffer(raw, dtype=dtype).copy()
            if array.size != math.prod(shape):
                raise ValueError("NumPy state byte length does not match shape")
            return array.reshape(shape)
        if kind == "torch.uint8":
            if set(value) != {"$kind", "shape", "data_hex"}:
                raise ValueError("invalid torch RNG state encoding")
            import torch

            shape = tuple(int(item) for item in value["shape"])
            raw = bytes.fromhex(value["data_hex"])
            tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
            if tensor.numel() != math.prod(shape):
                raise ValueError("torch RNG byte length does not match shape")
            return tensor.reshape(shape)
        if kind is not None:
            raise ValueError(f"unknown scale resume state encoding: {kind!r}")
        return {str(key): _decode_state(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"invalid encoded scale resume state type {type(value).__name__}")


def capture_trainer_resume_control(trainer: Any) -> dict[str, Any]:
    """Capture D02 control state while leaving model/optimizer tensors to D18 DCP."""

    if not hasattr(trainer, "assert_checkpoint_safe") or not hasattr(trainer, "state_dict"):
        raise TypeError("trainer must provide D02 checkpoint-safe state interfaces")
    trainer.assert_checkpoint_safe()
    state = trainer.state_dict()
    required = ("micro_step", "optimizer_step", "tokens_seen", "scheduler", "scaler", "config")
    if any(not hasattr(state, name) for name in required):
        raise TypeError("trainer state does not provide the expected D02 control fields")
    control = {
        "schema": CONTROL_SCHEMA,
        "trainer": {
            "micro_step": int(state.micro_step),
            "optimizer_step": int(state.optimizer_step),
            "tokens_seen": int(state.tokens_seen),
            "scheduler": _json_safe(state.scheduler, "trainer.scheduler"),
            "scaler": _json_safe(state.scaler, "trainer.scaler"),
            "config": _json_safe(state.config, "trainer.config"),
        },
        "rng": _encode_state(capture_rng_state()),
    }
    return control


def restore_trainer_resume_control(
    trainer: Any,
    control: Mapping[str, Any],
    *,
    restore_rng: bool = True,
) -> tuple[bool, bool]:
    """Restore D02 counters/scheduler/scaler around the optimizer already loaded by D18."""

    if not isinstance(control, Mapping) or control.get("schema") != CONTROL_SCHEMA:
        raise ValueError("unsupported or missing scale resume control state")
    trainer_state = control.get("trainer")
    if not isinstance(trainer_state, Mapping):
        raise TypeError("scale resume trainer control must be a mapping")
    expected_keys = {
        "micro_step",
        "optimizer_step",
        "tokens_seen",
        "scheduler",
        "scaler",
        "config",
    }
    if set(trainer_state) != expected_keys:
        raise ValueError("scale resume trainer control schema mismatch")
    if not hasattr(trainer, "optimizer") or not hasattr(trainer, "load_state_dict"):
        raise TypeError("trainer must expose optimizer and load_state_dict")
    full_state = {
        "micro_step": trainer_state["micro_step"],
        "optimizer_step": trainer_state["optimizer_step"],
        "tokens_seen": trainer_state["tokens_seen"],
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "scheduler": copy.deepcopy(trainer_state["scheduler"]),
        "scaler": copy.deepcopy(trainer_state["scaler"]),
        "config": copy.deepcopy(trainer_state["config"]),
    }
    trainer.load_state_dict(full_state)
    rng_restored = False
    if restore_rng:
        rng_raw = control.get("rng")
        rng_state = _decode_state(rng_raw)
        if not isinstance(rng_state, Mapping):
            raise TypeError("decoded scale resume RNG state must be a mapping")
        restore_rng_state(rng_state)
        rng_restored = True
    return True, rng_restored


def save_trainer_scale_checkpoint(
    checkpoint_dir: str | Path,
    *,
    trainer: Any,
    plan: ParallelPlan,
    identity: ScaleCheckpointIdentity,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save exact D02 control state through D18 without duplicating its DCP data plane."""

    control = capture_trainer_resume_control(trainer)
    metadata_value = dict(metadata or {})
    metadata_value["resume_control_schema"] = CONTROL_SCHEMA
    return save_scale_checkpoint(
        checkpoint_dir,
        model=trainer.model,
        optimizer=trainer.optimizer,
        plan=plan,
        identity=identity,
        metadata=metadata_value,
        rank_state=control,
    )


def load_trainer_scale_checkpoint(
    checkpoint_dir: str | Path,
    *,
    trainer: Any,
    target_plan: ParallelPlan,
    expected_identity_sha256: str | None = None,
    restore_rng: bool = True,
) -> TrainerScaleLoadResult:
    """Exact-topology D02 resume. Topology-changing Trainer continuation remains blocked."""

    result = load_scale_checkpoint(
        checkpoint_dir,
        model=trainer.model,
        optimizer=trainer.optimizer,
        target_plan=target_plan,
        mode=ResumeMode.EXACT_TOPOLOGY,
        expected_identity_sha256=expected_identity_sha256,
    )
    if result.rank_state is None:
        raise ValueError("scale checkpoint lacks rank-local D02 resume control state")
    restored, rng_restored = restore_trainer_resume_control(
        trainer, result.rank_state, restore_rng=restore_rng
    )
    return TrainerScaleLoadResult(
        dcp=result,
        trainer_control_restored=restored,
        rng_restored=rng_restored,
        exact_training_continuation_claim_allowed=bool(restored and rng_restored),
    )


def assert_v1_scale_migration_compatible(
    v1_manifest: Mapping[str, Any],
    *,
    scale_identity: ScaleCheckpointIdentity,
    provenance: V1ScaleMigrationProvenance,
) -> None:
    """Refuse migration that would relabel any identity carried by checkpoint-v1."""

    scale_identity.validate()
    provenance.validate()
    if v1_manifest.get("format") != "12-6-checkpoint" or v1_manifest.get("format_version") != 1:
        raise ValueError("source is not checkpoint-v1")
    identity = v1_manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise TypeError("checkpoint-v1 identity must be a mapping")
    expected = {
        "git_sha": scale_identity.git_sha,
        "model_spec_hash": scale_identity.model_spec_sha256,
        "tokenizer_hash": scale_identity.tokenizer_config_sha256,
        "tokenizer_vocab_hash": scale_identity.tokenizer_vocab_sha256,
        "dataset_manifest_hash": scale_identity.data_manifest_sha256,
        "training_config_hash": scale_identity.training_config_sha256,
        "environment_lock_hash": scale_identity.environment_lock_sha256,
        "seed": scale_identity.seed,
        "step": scale_identity.step,
        "tokens_seen": scale_identity.tokens_seen,
        "run_manifest_hash": provenance.source_run_manifest_sha256,
    }
    mismatches = {
        key: {"v1": identity.get(key), "scale": value}
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if provenance.init_spec_sha256 != scale_identity.init_spec_sha256:
        mismatches["init_spec_sha256"] = {
            "migration": provenance.init_spec_sha256,
            "scale": scale_identity.init_spec_sha256,
        }
    if provenance.packing_sha256 != scale_identity.packing_sha256:
        mismatches["packing_sha256"] = {
            "migration": provenance.packing_sha256,
            "scale": scale_identity.packing_sha256,
        }
    if mismatches:
        raise ValueError(f"checkpoint-v1 -> D18 migration identity mismatch: {mismatches}")


def migrate_v1_trainer_checkpoint_to_scale(
    source_v1: str | Path,
    destination_scale: str | Path,
    *,
    trainer: Any,
    plan: ParallelPlan,
    scale_identity: ScaleCheckpointIdentity,
    provenance: V1ScaleMigrationProvenance,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One-time v1 bridge. The old whole-snapshot RAM cost is paid only during migration."""

    plan.validate()
    if plan.world_size != 1:
        raise ValueError("checkpoint-v1 migration must first materialize a one-rank D18 generation")
    source_manifest = verify_checkpoint(source_v1)
    assert_v1_scale_migration_compatible(
        source_manifest,
        scale_identity=scale_identity,
        provenance=provenance,
    )
    loaded = load_checkpoint(
        source_v1,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        restore_rng=True,
        expected_git_sha=scale_identity.git_sha,
        expected_model_spec_hash=scale_identity.model_spec_sha256,
        expected_tokenizer_hash=scale_identity.tokenizer_config_sha256,
        expected_tokenizer_vocab_hash=scale_identity.tokenizer_vocab_sha256,
        expected_dataset_manifest_hash=scale_identity.data_manifest_sha256,
    )
    if not loaded.trainer_state:
        raise ValueError("checkpoint-v1 migration requires canonical D02 trainer state")
    trainer.load_state_dict(loaded.trainer_state)
    migration_metadata = dict(metadata or {})
    migration_metadata["migration"] = {
        "from_format": "12-6-checkpoint-v1",
        "source_checkpoint_id": source_manifest["checkpoint_id"],
        "source_manifest_sha256": sha256_file(Path(source_v1) / "manifest.json"),
        "source_run_manifest_sha256": provenance.source_run_manifest_sha256,
        "init_spec_sha256": provenance.init_spec_sha256,
        "packing_sha256": provenance.packing_sha256,
        "whole_snapshot_ram_paid_once": True,
    }
    return save_trainer_scale_checkpoint(
        destination_scale,
        trainer=trainer,
        plan=plan,
        identity=scale_identity,
        metadata=migration_metadata,
    )


def plan_scale_checkpoint_retention(
    parent: str | Path,
    *,
    policy: ScaleRetentionPolicy,
) -> ScaleRetentionPlan:
    """Retain only verified committed generations; never auto-delete staging attempts."""

    policy.validate()
    root = Path(parent)
    generations: list[ScaleGeneration] = []
    staging: list[Path] = []
    for path in sorted(root.iterdir()):
        if path.is_symlink() or not path.is_dir():
            continue
        if path.name.startswith(".") and ".dcp-staging-" in path.name:
            staging.append(path)
            continue
        if not (path / COMMITTED).is_file():
            continue
        manifest = verify_scale_checkpoint(path)
        identity = manifest.get("identity")
        if not isinstance(identity, Mapping):
            raise ValueError(f"committed checkpoint lacks identity: {path}")
        step = identity.get("step")
        tokens_seen = identity.get("tokens_seen")
        aggregate = manifest.get("aggregate_checkpoint_sha256")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError(f"invalid checkpoint step: {path}")
        if not isinstance(tokens_seen, int) or isinstance(tokens_seen, bool) or tokens_seen < 0:
            raise ValueError(f"invalid checkpoint tokens_seen: {path}")
        _require_sha256(aggregate, "aggregate_checkpoint_sha256")
        generations.append(
            ScaleGeneration(
                path=path,
                step=step,
                tokens_seen=tokens_seen,
                aggregate_checkpoint_sha256=aggregate,
            )
        )
    generations.sort(key=lambda item: (item.step, item.tokens_seen, item.path.name))
    keep_paths = {item.path for item in generations[-policy.keep_last :]}
    if policy.keep_every_n_steps is not None:
        keep_paths.update(
            item.path
            for item in generations
            if item.step % policy.keep_every_n_steps == 0
        )
    return ScaleRetentionPlan(
        keep=tuple(item for item in generations if item.path in keep_paths),
        delete=tuple(item for item in generations if item.path not in keep_paths),
        uncommitted_staging=tuple(staging),
    )


def apply_scale_checkpoint_retention(plan: ScaleRetentionPlan) -> None:
    """Reverify checkpoint identity immediately before each destructive deletion."""

    for generation in plan.delete:
        manifest = verify_scale_checkpoint(generation.path)
        if manifest.get("aggregate_checkpoint_sha256") != generation.aggregate_checkpoint_sha256:
            raise ValueError(f"checkpoint changed after retention planning: {generation.path}")
        shutil.rmtree(generation.path)
