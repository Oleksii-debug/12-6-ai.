"""Scale checkpoint v2 built on PyTorch Distributed Checkpoint.

Checkpoint-v1 remains the S0 portability format. This module is an additive
training-resume format for model/optimizer states that no longer fit the v1
single-file, whole-snapshot-in-RAM design.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import stat
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from safetensors.numpy import load_file as load_safetensors
from safetensors.numpy import save_file as save_safetensors
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from .core import (
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointIdentity,
    CheckpointIntegrityError,
    canonical_json_bytes,
    capture_rng_state,
    environment_snapshot,
    hash_json,
    load_checkpoint,
    restore_rng_state,
    sha256_file,
    verify_checkpoint,
)
from .state_tree import StateTreeError, pack_state_tree, unpack_state_tree

FORMAT_NAME = "12-6-checkpoint-v2"
FORMAT_VERSION = 2
MANIFEST_NAME = "manifest.json"
MANIFEST_CHECKSUM_NAME = "MANIFEST.sha256"
DCP_DIR = "dcp"
CONTROL_DIR = "control"

_MANIFEST_KEYS = frozenset(
    {
        "format",
        "format_version",
        "status",
        "created_at_utc",
        "checkpoint_id",
        "semantic_identity",
        "semantic_identity_sha256",
        "source_topology",
        "source_topology_sha256",
        "writers",
        "storage",
        "files",
        "migration",
    }
)
_SEMANTIC_KEYS = frozenset(
    {
        "git_sha",
        "model_spec",
        "model_spec_hash",
        "parameter_count",
        "tokenizer_hash",
        "tokenizer_vocab_hash",
        "dataset_manifest_hash",
        "run_manifest_hash",
        "training_config",
        "training_config_hash",
        "seed",
        "optimizer",
        "optimizer_hash",
        "scheduler",
        "scheduler_hash",
        "precision",
        "step",
        "tokens_seen",
        "environment_lock_hash",
    }
)
_HEX = frozenset("0123456789abcdef")
_ASYNC_LOCK = threading.Lock()
_ASYNC_ACTIVE = False


@dataclass(frozen=True, slots=True)
class ResumeTopology:
    """Semantic training topology recorded for exact resume / reshard decisions."""

    world_size: int
    parallelism: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.world_size, int) or isinstance(self.world_size, bool):
            raise ValueError("world_size must be an integer")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not isinstance(self.parallelism, Mapping):
            raise ValueError("parallelism must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "parallelism": _jsonable(self.parallelism),
        }

    def identity_sha256(self) -> str:
        return hash_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class CheckpointV2LoadResult:
    manifest: dict[str, Any]
    trainer_state: Mapping[str, Any]
    rng_restored: bool
    resharded: bool


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    keep: tuple[Path, ...]
    delete: tuple[Path, ...]


class AsyncCheckpointV2:
    """One in-flight DCP save whose COMPLETE manifest is published by ``wait``."""

    def __init__(self, *, future: Any, context: dict[str, Any]) -> None:
        self._future = future
        self._context = context
        self._done = False

    def wait(self) -> dict[str, Any]:
        global _ASYNC_ACTIVE
        if self._done:
            return copy.deepcopy(self._context["manifest"])
        try:
            self._future.result()
            manifest = _finish_save(self._context, save_mode="async")
            self._context["manifest"] = manifest
            self._done = True
            return copy.deepcopy(manifest)
        finally:
            with _ASYNC_LOCK:
                _ASYNC_ACTIVE = False


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    raise TypeError(f"value cannot be encoded as canonical JSON: {type(value)!r}")


def _semantic_identity(identity: CheckpointIdentity) -> dict[str, Any]:
    identity.validate()
    model_spec = _jsonable(identity.model_spec)
    training_config = _jsonable(identity.training_config)
    optimizer = _jsonable(identity.optimizer)
    scheduler = _jsonable(identity.scheduler)
    return {
        "git_sha": identity.git_sha,
        "model_spec": model_spec,
        "model_spec_hash": hash_json(model_spec),
        "parameter_count": identity.parameter_count,
        "tokenizer_hash": identity.tokenizer_hash,
        "tokenizer_vocab_hash": identity.tokenizer_vocab_hash,
        "dataset_manifest_hash": identity.dataset_manifest_hash,
        "run_manifest_hash": identity.run_manifest_hash,
        "training_config": training_config,
        "training_config_hash": hash_json(training_config),
        "seed": identity.seed,
        "optimizer": optimizer,
        "optimizer_hash": hash_json(optimizer),
        "scheduler": scheduler,
        "scheduler_hash": hash_json(scheduler),
        "precision": identity.precision,
        "step": identity.step,
        "tokens_seen": identity.tokens_seen,
        "environment_lock_hash": identity.environment_lock_hash,
    }


def current_topology(parallelism: Mapping[str, Any] | None = None) -> ResumeTopology:
    return ResumeTopology(
        world_size=dist.get_world_size() if _distributed() else 1,
        parallelism=dict(parallelism or {}),
    )


def _distributed() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


def _rank() -> int:
    return dist.get_rank() if _distributed() else 0


def _barrier() -> None:
    if _distributed():
        dist.barrier()


def _all_gather_object(value: Any) -> list[Any]:
    if not _distributed():
        return [value]
    output: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(output, value)
    return output


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _state_dict_or_none(obj: Any | None) -> Any | None:
    if obj is None:
        return None
    if not hasattr(obj, "state_dict"):
        raise TypeError(f"{type(obj).__name__} must provide state_dict()")
    return obj.state_dict()


def _control_paths(root: Path, rank: int) -> tuple[Path, Path]:
    stem = f"rank-{rank:05d}"
    return root / CONTROL_DIR / f"{stem}.json", root / CONTROL_DIR / f"{stem}.safetensors"


def _write_rank_control(
    root: Path,
    *,
    trainer_state: Mapping[str, Any] | None,
    scheduler: Any | None,
) -> None:
    tree_path, tensor_path = _control_paths(root, _rank())
    packed = pack_state_tree(
        {
            "trainer": dict(trainer_state or {}),
            "scheduler": _state_dict_or_none(scheduler),
            "rng": capture_rng_state(),
        }
    )
    save_safetensors(packed.tensors, str(tensor_path))
    _write_json(tree_path, packed.tree)


def _read_rank_control(root: Path, rank: int) -> Mapping[str, Any]:
    tree_path, tensor_path = _control_paths(root, rank)
    try:
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
        tensors = load_safetensors(str(tensor_path))
        value = unpack_state_tree(tree, tensors)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, StateTreeError, ValueError) as exc:
        raise CheckpointIntegrityError(f"cannot decode v2 control state for rank {rank}") from exc
    if not isinstance(value, Mapping):
        raise CheckpointIntegrityError("v2 control state must be a mapping")
    return value


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _payload_paths(root: Path) -> list[Path]:
    output: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in list(dirnames):
            candidate = base / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise CheckpointIntegrityError(f"v2 directory is not a real directory: {candidate}")
        for name in filenames:
            candidate = base / name
            if candidate.name in {MANIFEST_NAME, MANIFEST_CHECKSUM_NAME} and base == root:
                continue
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise CheckpointIntegrityError(f"v2 payload is not a regular file: {candidate}")
            output.append(candidate)
    return sorted(output)


def _parallel_file_records(root: Path) -> list[dict[str, Any]]:
    paths = _payload_paths(root)
    rank = _rank()
    world_size = dist.get_world_size() if _distributed() else 1
    local = [
        _file_record(root, path)
        for index, path in enumerate(paths)
        if index % world_size == rank
    ]
    gathered = _all_gather_object(local)
    merged = [item for partition in gathered for item in partition]
    return sorted(merged, key=lambda item: item["path"])


def _writer_records() -> list[dict[str, Any]]:
    backend = dist.get_backend() if _distributed() else "none"
    local_environment = environment_snapshot()
    local = {
        "rank": _rank(),
        "backend": str(backend),
        "environment": local_environment,
        "environment_sha256": hash_json(local_environment),
    }
    writers = _all_gather_object(local)
    return sorted(writers, key=lambda item: item["rank"])


def _prepare_save(
    directory: str | Path,
    *,
    model: Any,
    optimizer: Any | None,
    scheduler: Any | None,
    trainer_state: Mapping[str, Any] | None,
    identity: CheckpointIdentity,
    topology: ResumeTopology | None,
    migration: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(directory)
    topology = current_topology() if topology is None else topology
    actual_world = dist.get_world_size() if _distributed() else 1
    if topology.world_size != actual_world:
        raise CheckpointCompatibilityError(
            "declared topology world_size="
            f"{topology.world_size} but runtime world_size={actual_world}"
        )

    if _rank() == 0:
        if root.exists() or root.is_symlink():
            raise FileExistsError(f"checkpoint-v2 destination already exists: {root}")
        root.mkdir(parents=True)
        (root / DCP_DIR).mkdir()
        (root / CONTROL_DIR).mkdir()
    _barrier()

    try:
        _write_rank_control(root, trainer_state=trainer_state, scheduler=scheduler)
        _barrier()
        optimizers: Any = [] if optimizer is None else optimizer
        model_state, optimizer_state = get_state_dict(model, optimizers)
        dcp_state: dict[str, Any] = {"model": model_state}
        if optimizer is not None:
            dcp_state["optimizer"] = optimizer_state
        context = {
            "root": root,
            "identity": _semantic_identity(identity),
            "topology": topology.to_dict(),
            "has_optimizer": optimizer is not None,
            "has_scheduler": scheduler is not None,
            "migration": _jsonable(migration) if migration is not None else None,
        }
        return dcp_state, context
    except Exception:
        _barrier()
        if _rank() == 0 and root.exists():
            shutil.rmtree(root, ignore_errors=True)
        _barrier()
        raise


def _finish_save(context: dict[str, Any], *, save_mode: str) -> dict[str, Any]:
    root: Path = context["root"]
    _barrier()
    files = _parallel_file_records(root)
    writers = _writer_records()
    semantic_identity = context["identity"]
    topology = context["topology"]
    storage = {
        "training_state": "torch-distributed-checkpoint",
        "control_tensors": "safetensors",
        "control_tree": "canonical-json",
        "filesystem_commit": "manifest-last",
        "save_mode": save_mode,
        "has_optimizer": context["has_optimizer"],
        "has_scheduler": context["has_scheduler"],
        "control_pickle": False,
        "dcp_metadata": "pytorch-managed-trusted-project-artifact",
        "object_store": "NOT_IMPLEMENTED_REQUIRES_STORE_ADAPTER",
    }
    checkpoint_core = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "semantic_identity": semantic_identity,
        "source_topology": topology,
        "writers": writers,
        "storage": storage,
        "files": files,
        "migration": context["migration"],
    }
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "status": "COMPLETE",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "checkpoint_id": hash_json(checkpoint_core),
        "semantic_identity": semantic_identity,
        "semantic_identity_sha256": hash_json(semantic_identity),
        "source_topology": topology,
        "source_topology_sha256": hash_json(topology),
        "writers": writers,
        "storage": storage,
        "files": files,
        "migration": context["migration"],
    }
    if _rank() == 0:
        _write_json(root / MANIFEST_NAME, manifest)
        manifest_sha = sha256_file(root / MANIFEST_NAME)
        (root / MANIFEST_CHECKSUM_NAME).write_text(
            f"{manifest_sha}  {MANIFEST_NAME}\n", encoding="ascii"
        )
    _barrier()
    verified = verify_checkpoint_v2(root)
    if verified["checkpoint_id"] != manifest["checkpoint_id"]:
        raise CheckpointIntegrityError(
            "published v2 checkpoint identity changed during verification"
        )
    return manifest


def save_checkpoint_v2(
    directory: str | Path,
    *,
    model: Any,
    identity: CheckpointIdentity,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    trainer_state: Mapping[str, Any] | None = None,
    topology: ResumeTopology | None = None,
    migration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Synchronously save one immutable v2 training-resume checkpoint."""

    state, context = _prepare_save(
        directory,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state=trainer_state,
        identity=identity,
        topology=topology,
        migration=migration,
    )
    try:
        dcp.save(
            state,
            checkpoint_id=context["root"] / DCP_DIR,
            no_dist=not _distributed(),
        )
        return _finish_save(context, save_mode="sync")
    except Exception:
        _barrier()
        if _rank() == 0 and not (context["root"] / MANIFEST_NAME).exists():
            shutil.rmtree(context["root"], ignore_errors=True)
        _barrier()
        raise


def begin_async_checkpoint_v2(
    directory: str | Path,
    *,
    model: Any,
    identity: CheckpointIdentity,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    trainer_state: Mapping[str, Any] | None = None,
    topology: ResumeTopology | None = None,
    migration: Mapping[str, Any] | None = None,
) -> AsyncCheckpointV2:
    """Start one async DCP save; every rank must later call ``wait()``."""

    global _ASYNC_ACTIVE
    with _ASYNC_LOCK:
        if _ASYNC_ACTIVE:
            raise CheckpointError("only one checkpoint-v2 async save may be in flight per process")
        _ASYNC_ACTIVE = True
    try:
        state, context = _prepare_save(
            directory,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            trainer_state=trainer_state,
            identity=identity,
            topology=topology,
            migration=migration,
        )
        future = dcp.async_save(
            state,
            checkpoint_id=context["root"] / DCP_DIR,
            no_dist=not _distributed(),
        )
        return AsyncCheckpointV2(future=future, context=context)
    except Exception:
        with _ASYNC_LOCK:
            _ASYNC_ACTIVE = False
        raise


def _validate_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise CheckpointIntegrityError("v2 file path must be non-empty text")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise CheckpointIntegrityError(f"unsafe v2 file path: {value!r}")
    return value


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    checksum_path = root / MANIFEST_CHECKSUM_NAME
    for path in (manifest_path, checksum_path):
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as exc:
            raise CheckpointIntegrityError(
                "checkpoint-v2 is incomplete: missing commit manifest"
            ) from exc
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise CheckpointIntegrityError(
                f"checkpoint-v2 metadata is not a regular file: {path.name}"
            )
    checksum = checksum_path.read_text(encoding="ascii").strip().split()
    if len(checksum) != 2 or checksum[1] != MANIFEST_NAME:
        raise CheckpointIntegrityError("invalid checkpoint-v2 MANIFEST.sha256")
    if checksum[0] != sha256_file(manifest_path):
        raise CheckpointIntegrityError("checkpoint-v2 manifest checksum mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError("checkpoint-v2 manifest is not valid JSON") from exc
    return manifest


def _require_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(ch not in _HEX for ch in value)
    ):
        raise CheckpointIntegrityError(f"{field} must be exact lowercase SHA-256")
    return value


def _validate_semantic_identity(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _SEMANTIC_KEYS:
        raise CheckpointIntegrityError("checkpoint-v2 semantic identity schema mismatch")
    for field in (
        "model_spec_hash",
        "tokenizer_hash",
        "tokenizer_vocab_hash",
        "dataset_manifest_hash",
        "run_manifest_hash",
        "training_config_hash",
        "optimizer_hash",
        "scheduler_hash",
    ):
        _require_sha256(value.get(field), field=f"semantic_identity.{field}")
    lock_hash = value.get("environment_lock_hash")
    if lock_hash is not None:
        _require_sha256(lock_hash, field="semantic_identity.environment_lock_hash")
    if hash_json(value.get("model_spec")) != value["model_spec_hash"]:
        raise CheckpointIntegrityError("checkpoint-v2 model_spec hash mismatch")
    if hash_json(value.get("training_config")) != value["training_config_hash"]:
        raise CheckpointIntegrityError("checkpoint-v2 training_config hash mismatch")
    if hash_json(value.get("optimizer")) != value["optimizer_hash"]:
        raise CheckpointIntegrityError("checkpoint-v2 optimizer descriptor hash mismatch")
    if hash_json(value.get("scheduler")) != value["scheduler_hash"]:
        raise CheckpointIntegrityError("checkpoint-v2 scheduler descriptor hash mismatch")
    try:
        reconstructed = CheckpointIdentity(
            git_sha=value["git_sha"],
            model_spec=value["model_spec"],
            parameter_count=value["parameter_count"],
            tokenizer_hash=value["tokenizer_hash"],
            tokenizer_vocab_hash=value["tokenizer_vocab_hash"],
            dataset_manifest_hash=value["dataset_manifest_hash"],
            run_manifest_hash=value["run_manifest_hash"],
            training_config=value["training_config"],
            seed=value["seed"],
            precision=value["precision"],
            step=value["step"],
            tokens_seen=value["tokens_seen"],
            optimizer=value["optimizer"],
            scheduler=value["scheduler"],
            environment_lock_hash=value["environment_lock_hash"],
        )
        reconstructed.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointIntegrityError("checkpoint-v2 semantic identity is invalid") from exc


def verify_checkpoint_v2(directory: str | Path) -> dict[str, Any]:
    """Streaming verification: no checkpoint-sized payload snapshot is retained in RAM."""

    root = Path(directory)
    try:
        mode = root.lstat().st_mode
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(f"checkpoint-v2 does not exist: {root}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise CheckpointIntegrityError("checkpoint-v2 root must be a real directory")

    manifest = _read_manifest(root)
    if set(manifest) != _MANIFEST_KEYS:
        raise CheckpointIntegrityError("checkpoint-v2 manifest schema mismatch")
    if manifest.get("format") != FORMAT_NAME or manifest.get("format_version") != FORMAT_VERSION:
        raise CheckpointCompatibilityError("unsupported checkpoint-v2 format")
    if manifest.get("status") != "COMPLETE":
        raise CheckpointIntegrityError("checkpoint-v2 is not COMPLETE")

    semantic = manifest.get("semantic_identity")
    topology = manifest.get("source_topology")
    writers = manifest.get("writers")
    storage = manifest.get("storage")
    files = manifest.get("files")
    if not isinstance(semantic, Mapping) or not isinstance(topology, Mapping):
        raise CheckpointIntegrityError("checkpoint-v2 identity/topology must be mappings")
    _validate_semantic_identity(semantic)
    if hash_json(semantic) != manifest.get("semantic_identity_sha256"):
        raise CheckpointIntegrityError("checkpoint-v2 semantic identity hash mismatch")
    if hash_json(topology) != manifest.get("source_topology_sha256"):
        raise CheckpointIntegrityError("checkpoint-v2 topology hash mismatch")
    try:
        ResumeTopology(
            world_size=topology["world_size"],
            parallelism=topology["parallelism"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointIntegrityError("checkpoint-v2 source topology is invalid") from exc
    if set(topology) != {"world_size", "parallelism"}:
        raise CheckpointIntegrityError("checkpoint-v2 source topology schema mismatch")
    if not isinstance(writers, list) or not writers:
        raise CheckpointIntegrityError("checkpoint-v2 writer records are missing")
    expected_ranks = list(range(topology["world_size"]))
    actual_ranks: list[int] = []
    for writer in writers:
        if not isinstance(writer, Mapping) or set(writer) != {
            "rank",
            "backend",
            "environment",
            "environment_sha256",
        }:
            raise CheckpointIntegrityError("checkpoint-v2 writer record schema mismatch")
        rank = writer["rank"]
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise CheckpointIntegrityError("checkpoint-v2 writer rank is invalid")
        if not isinstance(writer["backend"], str) or not writer["backend"]:
            raise CheckpointIntegrityError("checkpoint-v2 writer backend is invalid")
        if not isinstance(writer["environment"], Mapping):
            raise CheckpointIntegrityError("checkpoint-v2 writer environment is invalid")
        _require_sha256(
            writer["environment_sha256"],
            field=f"writers.{rank}.environment_sha256",
        )
        if hash_json(writer["environment"]) != writer["environment_sha256"]:
            raise CheckpointIntegrityError("checkpoint-v2 writer environment hash mismatch")
        actual_ranks.append(rank)
    if sorted(actual_ranks) != expected_ranks:
        raise CheckpointIntegrityError("checkpoint-v2 writer ranks do not match source topology")
    if (
        not isinstance(storage, Mapping)
        or storage.get("training_state") != "torch-distributed-checkpoint"
        or storage.get("control_tensors") != "safetensors"
        or storage.get("control_tree") != "canonical-json"
        or storage.get("filesystem_commit") != "manifest-last"
        or storage.get("control_pickle") is not False
    ):
        raise CheckpointIntegrityError("checkpoint-v2 storage declaration is invalid")
    if set(storage) != {
        "training_state",
        "control_tensors",
        "control_tree",
        "filesystem_commit",
        "save_mode",
        "has_optimizer",
        "has_scheduler",
        "control_pickle",
        "dcp_metadata",
        "object_store",
    }:
        raise CheckpointIntegrityError("checkpoint-v2 storage schema mismatch")
    if storage.get("save_mode") not in {"sync", "async"}:
        raise CheckpointIntegrityError("checkpoint-v2 save mode is invalid")
    if not isinstance(storage.get("has_optimizer"), bool) or not isinstance(
        storage.get("has_scheduler"), bool
    ):
        raise CheckpointIntegrityError("checkpoint-v2 optimizer/scheduler flags are invalid")
    if storage.get("dcp_metadata") != "pytorch-managed-trusted-project-artifact":
        raise CheckpointIntegrityError("checkpoint-v2 DCP metadata trust boundary is invalid")
    if storage.get("object_store") != "NOT_IMPLEMENTED_REQUIRES_STORE_ADAPTER":
        raise CheckpointIntegrityError("checkpoint-v2 object-store boundary is invalid")
    if not isinstance(files, list) or not files:
        raise CheckpointIntegrityError("checkpoint-v2 file inventory is empty")

    expected_paths: set[str] = set()
    for record in files:
        if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
            raise CheckpointIntegrityError("invalid checkpoint-v2 file record")
        relative = _validate_relative_path(record.get("path"))
        if relative in expected_paths:
            raise CheckpointIntegrityError(f"duplicate checkpoint-v2 file: {relative}")
        expected_paths.add(relative)
        expected_bytes = record.get("bytes")
        expected_hash = record.get("sha256")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            raise CheckpointIntegrityError(f"invalid checkpoint-v2 record for {relative}")
        _require_sha256(expected_hash, field=f"files.{relative}.sha256")
        path = root / relative
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as exc:
            raise CheckpointIntegrityError(f"missing checkpoint-v2 payload: {relative}") from exc
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise CheckpointIntegrityError(f"invalid checkpoint-v2 payload type: {relative}")
        if path.stat().st_size != expected_bytes:
            raise CheckpointIntegrityError(f"checkpoint-v2 size mismatch: {relative}")
        if sha256_file(path) != expected_hash:
            raise CheckpointIntegrityError(f"checkpoint-v2 checksum mismatch: {relative}")

    actual_paths = {path.relative_to(root).as_posix() for path in _payload_paths(root)}
    if actual_paths != expected_paths:
        raise CheckpointIntegrityError("checkpoint-v2 payload inventory mismatch")

    checkpoint_core = {
        "format": manifest["format"],
        "format_version": manifest["format_version"],
        "semantic_identity": semantic,
        "source_topology": topology,
        "writers": writers,
        "storage": storage,
        "files": files,
        "migration": manifest.get("migration"),
    }
    if hash_json(checkpoint_core) != manifest.get("checkpoint_id"):
        raise CheckpointIntegrityError("checkpoint-v2 checkpoint_id mismatch")
    return manifest


def _topology_matches(manifest: Mapping[str, Any], topology: ResumeTopology) -> bool:
    return manifest["source_topology_sha256"] == topology.identity_sha256()


def load_checkpoint_v2(
    directory: str | Path,
    *,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    expected_identity: CheckpointIdentity | None = None,
    topology: ResumeTopology | None = None,
    allow_reshard: bool = False,
    restore_rng: bool = True,
) -> CheckpointV2LoadResult:
    """Load DCP state, supporting explicit topology-changing reshard resume.

    Unlike v1, payload bytes are not snapshotted in RAM. Integrity is checked
    before and after DCP reads. If either check fails, the target process state
    is invalid and must not continue training.
    """

    root = Path(directory)
    before = verify_checkpoint_v2(root)
    if expected_identity is not None:
        expected_semantic = _semantic_identity(expected_identity)
        if hash_json(expected_semantic) != before["semantic_identity_sha256"]:
            raise CheckpointCompatibilityError("checkpoint-v2 semantic identity mismatch")

    topology = current_topology() if topology is None else topology
    exact_topology = _topology_matches(before, topology)
    if not exact_topology and not allow_reshard:
        raise CheckpointCompatibilityError(
            "checkpoint-v2 topology differs; set allow_reshard=True for an explicit reshard resume"
        )
    if not exact_topology and restore_rng:
        raise CheckpointCompatibilityError(
            "rank-local RNG cannot be restored across topology changes; use restore_rng=False"
        )

    has_optimizer = bool(before["storage"].get("has_optimizer"))
    has_scheduler = bool(before["storage"].get("has_scheduler"))
    if optimizer is not None and not has_optimizer:
        raise CheckpointCompatibilityError("optimizer requested but checkpoint-v2 has none")
    if scheduler is not None and not has_scheduler:
        raise CheckpointCompatibilityError("scheduler requested but checkpoint-v2 has none")

    optimizers: Any = [] if optimizer is None else optimizer
    model_state, optimizer_state = get_state_dict(model, optimizers)
    dcp_state: dict[str, Any] = {"model": model_state}
    if optimizer is not None:
        dcp_state["optimizer"] = optimizer_state
    dcp.load(
        dcp_state,
        checkpoint_id=root / DCP_DIR,
        no_dist=not _distributed(),
    )

    after = verify_checkpoint_v2(root)
    if after["checkpoint_id"] != before["checkpoint_id"]:
        raise CheckpointIntegrityError(
            "checkpoint-v2 changed while loading; target process is invalid"
        )

    set_state_dict(
        model,
        [] if optimizer is None else optimizer,
        model_state_dict=dcp_state["model"],
        optim_state_dict=dcp_state.get("optimizer", {}),
    )

    control_rank = _rank() if exact_topology else 0
    control = _read_rank_control(root, control_rank)
    if scheduler is not None:
        scheduler_state = control.get("scheduler")
        if scheduler_state is None:
            raise CheckpointCompatibilityError(
                "scheduler requested but control state is missing it"
            )
        scheduler.load_state_dict(scheduler_state)
    rng_restored = False
    if restore_rng:
        rng_state = control.get("rng")
        if not isinstance(rng_state, Mapping):
            raise CheckpointIntegrityError("checkpoint-v2 rank control is missing RNG state")
        restore_rng_state(rng_state)
        rng_restored = True
    trainer_state = control.get("trainer", {})
    if not isinstance(trainer_state, Mapping):
        raise CheckpointIntegrityError("checkpoint-v2 trainer state must be a mapping")
    return CheckpointV2LoadResult(
        manifest=copy.deepcopy(after),
        trainer_state=trainer_state,
        rng_restored=rng_restored,
        resharded=not exact_topology,
    )


def migrate_checkpoint_v1_to_v2(
    source_v1: str | Path,
    destination_v2: str | Path,
    *,
    model: Any,
    identity: CheckpointIdentity,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    topology: ResumeTopology | None = None,
) -> dict[str, Any]:
    """One-time legacy bridge; v1's whole-snapshot RAM cost is paid only here."""

    source_manifest = verify_checkpoint(source_v1)
    requested = _semantic_identity(identity)
    source_identity = source_manifest["identity"]
    required_equal = {
        "git_sha": requested["git_sha"],
        "model_spec_hash": requested["model_spec_hash"],
        "parameter_count": requested["parameter_count"],
        "tokenizer_hash": requested["tokenizer_hash"],
        "tokenizer_vocab_hash": requested["tokenizer_vocab_hash"],
        "dataset_manifest_hash": requested["dataset_manifest_hash"],
        "run_manifest_hash": requested["run_manifest_hash"],
        "training_config_hash": requested["training_config_hash"],
        "seed": requested["seed"],
        "optimizer_hash": requested["optimizer_hash"],
        "scheduler_hash": requested["scheduler_hash"],
        "precision": requested["precision"],
        "step": requested["step"],
        "tokens_seen": requested["tokens_seen"],
    }
    mismatches = {
        key: {"v1": source_identity.get(key), "v2": value}
        for key, value in required_equal.items()
        if source_identity.get(key) != value
    }
    if mismatches:
        raise CheckpointCompatibilityError(f"v1 migration identity mismatch: {mismatches}")
    loaded = load_checkpoint(
        source_v1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        restore_rng=True,
    )
    migration = {
        "from_format": source_manifest["format"],
        "from_format_version": source_manifest["format_version"],
        "source_checkpoint_id": source_manifest["checkpoint_id"],
        "source_manifest_sha256": sha256_file(Path(source_v1) / "manifest.json"),
    }
    return save_checkpoint_v2(
        destination_v2,
        model=model,
        identity=identity,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state=loaded.trainer_state,
        topology=topology,
        migration=migration,
    )


def plan_retention_v2(
    parent: str | Path,
    *,
    keep_last: int,
    keep_every_n_steps: int | None = None,
) -> RetentionPlan:
    """Plan deletion only for fully verified COMPLETE v2 checkpoint directories."""

    if not isinstance(keep_last, int) or isinstance(keep_last, bool) or keep_last < 1:
        raise ValueError("keep_last must be a positive integer")
    if keep_every_n_steps is not None and (
        not isinstance(keep_every_n_steps, int)
        or isinstance(keep_every_n_steps, bool)
        or keep_every_n_steps < 1
    ):
        raise ValueError("keep_every_n_steps must be a positive integer or None")

    candidates: list[tuple[int, Path]] = []
    for path in Path(parent).iterdir():
        if not path.is_dir() or path.is_symlink():
            continue
        if not (path / MANIFEST_NAME).is_file():
            continue
        manifest = verify_checkpoint_v2(path)
        step = manifest["semantic_identity"].get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise CheckpointIntegrityError(f"invalid checkpoint-v2 step in {path}")
        candidates.append((step, path))
    candidates.sort(key=lambda item: (item[0], item[1].name))
    keep_paths = {path for _, path in candidates[-keep_last:]}
    if keep_every_n_steps is not None:
        keep_paths.update(path for step, path in candidates if step % keep_every_n_steps == 0)
    return RetentionPlan(
        keep=tuple(path for _, path in candidates if path in keep_paths),
        delete=tuple(path for _, path in candidates if path not in keep_paths),
    )


def apply_retention_v2(plan: RetentionPlan) -> None:
    """Delete only directories that still verify to the planned COMPLETE checkpoint."""

    for path in plan.delete:
        verify_checkpoint_v2(path)
        shutil.rmtree(path)
