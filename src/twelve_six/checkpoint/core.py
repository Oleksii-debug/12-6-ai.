"""12-6 AI checkpoint format v1.

The checkpoint is a self-contained directory with SafeTensors payloads, a
JSON state tree, a manifest, and SHA-256 integrity records. Loading snapshots
and verifies every recorded byte before mutating model or trainer state.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import platform
import random
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from safetensors.numpy import load as load_safetensors_bytes
from safetensors.numpy import save_file as save_safetensors

from .state_tree import StateTreeError, pack_state_tree, unpack_state_tree

FORMAT_NAME = "12-6-checkpoint"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
MANIFEST_CHECKSUM_NAME = "MANIFEST.sha256"
WEIGHTS_NAME = "weights.safetensors"
STATE_TENSORS_NAME = "state.safetensors"
STATE_TREE_NAME = "state.json"
_PAYLOAD_NAMES = frozenset({WEIGHTS_NAME, STATE_TENSORS_NAME, STATE_TREE_NAME})
_DIRECTORY_NAMES = frozenset({MANIFEST_NAME, MANIFEST_CHECKSUM_NAME, *_PAYLOAD_NAMES})
_HEX = frozenset("0123456789abcdef")


class CheckpointError(RuntimeError):
    """Base checkpoint failure."""


class CheckpointIntegrityError(CheckpointError):
    """Raised when a checkpoint checksum or identity check fails."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint cannot be applied to the requested target."""


def _require_exact_hex(value: Any, *, field: str, lengths: set[int]) -> str:
    if not isinstance(value, str) or len(value) not in lengths:
        expected = "/".join(str(length) for length in sorted(lengths))
        raise ValueError(f"{field} must be exact lowercase {expected}-hex")
    if value != value.lower() or any(ch not in _HEX for ch in value):
        expected = "/".join(str(length) for length in sorted(lengths))
        raise ValueError(f"{field} must be exact lowercase {expected}-hex")
    return value


@dataclass(frozen=True)
class CheckpointIdentity:
    """Inputs that define the training/artifact lineage of a checkpoint."""

    git_sha: str
    model_spec: Mapping[str, Any]
    parameter_count: int
    tokenizer_hash: str
    tokenizer_vocab_hash: str
    dataset_manifest_hash: str
    run_manifest_hash: str
    training_config: Mapping[str, Any]
    seed: int
    precision: str
    step: int
    tokens_seen: int
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any] | None
    environment_lock_hash: str | None = None

    def validate(self) -> None:
        _require_exact_hex(self.git_sha, field="git_sha", lengths={40, 64})
        _require_exact_hex(self.tokenizer_hash, field="tokenizer_hash", lengths={64})
        _require_exact_hex(self.tokenizer_vocab_hash, field="tokenizer_vocab_hash", lengths={64})
        _require_exact_hex(self.dataset_manifest_hash, field="dataset_manifest_hash", lengths={64})
        _require_exact_hex(self.run_manifest_hash, field="run_manifest_hash", lengths={64})
        if self.environment_lock_hash is not None:
            _require_exact_hex(
                self.environment_lock_hash,
                field="environment_lock_hash",
                lengths={64},
            )
        if not isinstance(self.model_spec, Mapping) or not self.model_spec:
            raise ValueError("model_spec must be a non-empty mapping")
        if not isinstance(self.training_config, Mapping) or not self.training_config:
            raise ValueError("training_config must be a non-empty mapping")
        if not isinstance(self.optimizer, Mapping) or not self.optimizer:
            raise ValueError("optimizer must be a non-empty mapping")
        if self.scheduler is not None and (
            not isinstance(self.scheduler, Mapping) or not self.scheduler
        ):
            raise ValueError("scheduler must be a non-empty mapping or None")
        if (
            not isinstance(self.parameter_count, int)
            or isinstance(self.parameter_count, bool)
            or self.parameter_count <= 0
        ):
            raise ValueError("parameter_count must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.precision, str) or not self.precision.strip():
            raise ValueError("precision must be a non-empty string")
        if (
            not isinstance(self.step, int)
            or isinstance(self.step, bool)
            or not isinstance(self.tokens_seen, int)
            or isinstance(self.tokens_seen, bool)
            or self.step < 0
            or self.tokens_seen < 0
        ):
            raise ValueError("step and tokens_seen must be non-negative integers")


@dataclass(frozen=True)
class LoadResult:
    """Verified checkpoint metadata and decoded trainer state."""

    manifest: dict[str, Any]
    trainer_state: Mapping[str, Any]
    rng_state: Mapping[str, Any]


@dataclass(frozen=True)
class VerifiedCheckpoint:
    """Immutable in-memory byte snapshot verified before any target mutation.

    Manifest and payload bytes are stored privately. ``manifest`` reparses the
    exact verified manifest bytes on each access, so callers cannot mutate the
    snapshot's provenance before :func:`load_verified_checkpoint` consumes it.
    """

    _manifest_bytes: bytes
    _artifacts: Mapping[str, bytes]

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self._manifest_bytes.decode("utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical JSON encoding used for all identity hashes."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
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


def detect_git_sha(cwd: str | Path | None = None) -> str | None:
    """Return HEAD when running in a Git checkout; never fabricate a SHA."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha if sha else None


def environment_snapshot() -> dict[str, Any]:
    """Capture dependency/environment facts relevant to resume evidence."""

    package_names = ("numpy", "safetensors", "torch")
    packages: dict[str, str | None] = {}
    for name in package_names:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
    }


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and available PyTorch RNG state."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        state["torch"] = None
        return state
    torch_state: dict[str, Any] = {
        "cpu": torch.get_rng_state(),
        "cuda": [],
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    if torch.cuda.is_available():
        torch_state["cuda"] = torch.cuda.get_rng_state_all()
    state["torch"] = torch_state
    return state


def _preflight_rng_state(state: Mapping[str, Any]) -> None:
    """Validate supported RNG state without touching global RNG streams."""

    if not isinstance(state, Mapping):
        raise CheckpointCompatibilityError("checkpoint RNG state must be a mapping")
    if "python" in state:
        try:
            probe = random.Random()
            probe.setstate(state["python"])
        except (TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError("checkpoint Python RNG state is invalid") from exc
    if "numpy" in state:
        try:
            probe_np = np.random.RandomState()
            probe_np.set_state(state["numpy"])
        except (TypeError, ValueError) as exc:
            raise CheckpointCompatibilityError("checkpoint NumPy RNG state is invalid") from exc
    torch_state = state.get("torch")
    if not torch_state:
        return
    if not isinstance(torch_state, Mapping) or "cpu" not in torch_state:
        raise CheckpointCompatibilityError("checkpoint torch RNG state is invalid")
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise CheckpointCompatibilityError(
            "checkpoint contains torch RNG state but torch is unavailable"
        ) from exc
    try:
        generator = torch.Generator(device="cpu")
        generator.set_state(torch_state["cpu"].cpu())
    except (AttributeError, RuntimeError, TypeError) as exc:
        raise CheckpointCompatibilityError("checkpoint torch CPU RNG state is invalid") from exc
    cuda_states = torch_state.get("cuda", [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise CheckpointCompatibilityError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable; "
                "load with restore_rng=False"
            )
        if len(cuda_states) != torch.cuda.device_count():
            raise CheckpointCompatibilityError(
                "CUDA device count differs from the checkpoint; load with restore_rng=False"
            )


def restore_rng_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Restore captured RNG streams and report the exact restored scope."""

    _preflight_rng_state(state)
    scope = {"python": False, "numpy": False, "torch_cpu": False, "torch_cuda_devices": 0}
    if "python" in state:
        random.setstate(state["python"])
        scope["python"] = True
    if "numpy" in state:
        np.random.set_state(state["numpy"])
        scope["numpy"] = True
    torch_state = state.get("torch")
    if torch_state:
        torch = importlib.import_module("torch")
        torch.set_rng_state(torch_state["cpu"].cpu())
        scope["torch_cpu"] = True
        cuda_states = torch_state.get("cuda", [])
        if cuda_states:
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
            scope["torch_cuda_devices"] = len(cuda_states)
        torch.use_deterministic_algorithms(bool(torch_state.get("deterministic_algorithms", False)))
    return scope


def _model_state_to_numpy(model: Any) -> dict[str, np.ndarray]:
    if not hasattr(model, "state_dict"):
        raise TypeError("model must provide state_dict()")
    state = model.state_dict()
    if not isinstance(state, Mapping) or not state:
        raise ValueError("model.state_dict() must be a non-empty mapping")
    output: dict[str, np.ndarray] = {}
    for name, value in state.items():
        if isinstance(value, np.ndarray):
            output[str(name)] = np.ascontiguousarray(value)
            continue
        cls = value.__class__
        if cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}:
            tensor = value.detach().cpu().contiguous()
            if str(tensor.dtype) == "torch.bfloat16":
                torch = importlib.import_module("torch")
                output[str(name)] = tensor.view(torch.uint16).numpy().copy()
            else:
                output[str(name)] = tensor.numpy().copy()
            continue
        raise TypeError(f"unsupported model state tensor {name!r}: {type(value)!r}")
    return output


def _materialize_for_target(array: np.ndarray, target: Any) -> Any:
    if isinstance(target, np.ndarray):
        if tuple(target.shape) != tuple(array.shape):
            raise CheckpointCompatibilityError(
                f"shape mismatch: checkpoint {tuple(array.shape)} vs target {tuple(target.shape)}"
            )
        return array.astype(target.dtype, copy=True)
    cls = target.__class__
    if cls.__module__.startswith("torch") and cls.__name__ in {"Tensor", "Parameter"}:
        torch = importlib.import_module("torch")
        if str(target.dtype) == "torch.bfloat16" and array.dtype == np.uint16:
            tensor = torch.from_numpy(array.copy()).view(torch.bfloat16)
        else:
            tensor = torch.from_numpy(array.copy()).to(dtype=target.dtype)
        if tuple(target.shape) != tuple(tensor.shape):
            raise CheckpointCompatibilityError(
                f"shape mismatch: checkpoint {tuple(tensor.shape)} vs target {tuple(target.shape)}"
            )
        return tensor.to(device=target.device)
    raise CheckpointCompatibilityError(f"unsupported target tensor type {type(target)!r}")


def _prepare_model_weights(
    model: Any, arrays: Mapping[str, np.ndarray], strict: bool
) -> dict[str, Any]:
    """Materialize and validate all model tensors without mutating the model."""

    target_state = model.state_dict()
    target_keys = set(target_state)
    source_keys = set(arrays)
    if strict and target_keys != source_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise CheckpointCompatibilityError(
            f"state_dict keys differ: missing={missing}, unexpected={unexpected}"
        )
    return {
        name: _materialize_for_target(arrays[name], target_state[name])
        for name in target_state.keys() & arrays.keys()
    }


def _apply_model_weights(model: Any, materialized: Mapping[str, Any], strict: bool) -> None:
    if hasattr(model, "load_state_dict"):
        try:
            model.load_state_dict(materialized, strict=strict)
        except TypeError:
            model.load_state_dict(materialized)
        return
    raise TypeError("model must provide load_state_dict()")


def _state_dict_or_none(obj: Any | None) -> Any | None:
    if obj is None:
        return None
    if not hasattr(obj, "state_dict"):
        raise TypeError(f"{type(obj).__name__} must provide state_dict()")
    return obj.state_dict()


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _build_identity(identity: CheckpointIdentity, environment: Mapping[str, Any]) -> dict[str, Any]:
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
        "environment": _jsonable(environment),
        "environment_hash": hash_json(environment),
        "environment_lock_hash": identity.environment_lock_hash,
    }


def save_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    identity: CheckpointIdentity,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    trainer_state: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically publish one immutable verified checkpoint directory.

    Checkpoint-v1 directories are immutable once published. ``overwrite=True``
    is retained for API compatibility only when the destination does not exist;
    replacing an existing non-empty directory cannot be made crash-atomic across
    supported filesystems and therefore fails closed instead of deleting a prior
    valid checkpoint.
    """

    destination = Path(directory)
    if destination.exists() or destination.is_symlink():
        if overwrite:
            raise FileExistsError(
                "checkpoint-v1 is immutable and cannot overwrite existing destination: "
                f"{destination}"
            )
        raise FileExistsError(f"checkpoint already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        save_safetensors(_model_state_to_numpy(model), str(temp_dir / WEIGHTS_NAME))
        combined_state = {
            "optimizer": _state_dict_or_none(optimizer),
            "scheduler": _state_dict_or_none(scheduler),
            "trainer": dict(trainer_state or {}),
            "rng": capture_rng_state(),
        }
        packed = pack_state_tree(combined_state)
        save_safetensors(packed.tensors, str(temp_dir / STATE_TENSORS_NAME))
        _write_json(temp_dir / STATE_TREE_NAME, packed.tree)

        environment = environment_snapshot()
        identity_record = _build_identity(identity, environment)
        files = {
            WEIGHTS_NAME: _artifact_record(temp_dir / WEIGHTS_NAME),
            STATE_TENSORS_NAME: _artifact_record(temp_dir / STATE_TENSORS_NAME),
            STATE_TREE_NAME: _artifact_record(temp_dir / STATE_TREE_NAME),
        }
        checkpoint_id = hash_json({"identity": identity_record, "files": files})
        manifest = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "checkpoint_id": checkpoint_id,
            "identity": identity_record,
            "files": files,
            "serialization": {
                "weights": "safetensors",
                "state_tensors": "safetensors",
                "state_tree": "canonical-json",
                "pickle": False,
            },
        }
        _write_json(temp_dir / MANIFEST_NAME, manifest)
        manifest_sha = sha256_file(temp_dir / MANIFEST_NAME)
        (temp_dir / MANIFEST_CHECKSUM_NAME).write_text(
            f"{manifest_sha}  {MANIFEST_NAME}\n", encoding="ascii"
        )
        verify_checkpoint(temp_dir)
        try:
            os.replace(temp_dir, destination)
        except OSError:
            # The destination may have appeared after the initial existence check.
            # A valid winning checkpoint is non-empty, so rename fails instead of replacing it.
            raise
        return manifest
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _require_checkpoint_directory(root: Path) -> None:
    try:
        root_stat = root.lstat()
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(f"checkpoint directory does not exist: {root}") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise CheckpointIntegrityError("checkpoint root must be a real directory, not a symlink")
    names = {entry.name for entry in root.iterdir()}
    if names != _DIRECTORY_NAMES:
        missing = sorted(_DIRECTORY_NAMES - names)
        unexpected = sorted(names - _DIRECTORY_NAMES)
        raise CheckpointIntegrityError(
            f"checkpoint directory inventory mismatch: missing={missing}, unexpected={unexpected}"
        )


def _read_regular_bytes(root: Path, name: str) -> bytes:
    path = root / name
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(f"missing checkpoint artifact: {name}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CheckpointIntegrityError(
            f"checkpoint artifact must be a regular non-symlink file: {name}"
        )

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CheckpointIntegrityError(f"cannot safely open checkpoint artifact: {name}") from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise CheckpointIntegrityError(
                f"checkpoint artifact changed type while opening: {name}"
            )
        before_identity = (before.st_dev, before.st_ino)
        opened_identity = (opened.st_dev, opened.st_ino)
        if before_identity != opened_identity:
            raise CheckpointIntegrityError(f"checkpoint artifact changed while opening: {name}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(fd)


def _parse_manifest_bytes(manifest_bytes: bytes, checksum_bytes: bytes) -> dict[str, Any]:
    try:
        checksum_line = checksum_bytes.decode("ascii").strip().split()
    except UnicodeDecodeError as exc:
        raise CheckpointIntegrityError("MANIFEST.sha256 must be ASCII") from exc
    if len(checksum_line) != 2 or checksum_line[1] != MANIFEST_NAME:
        raise CheckpointIntegrityError("invalid MANIFEST.sha256 format")
    actual_manifest_hash = sha256_bytes(manifest_bytes)
    if checksum_line[0] != actual_manifest_hash:
        raise CheckpointIntegrityError("manifest checksum mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError("manifest is not valid UTF-8 JSON") from exc
    if manifest.get("format") != FORMAT_NAME or manifest.get("format_version") != FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            "unsupported checkpoint format: "
            f"{manifest.get('format')!r} v{manifest.get('format_version')!r}"
        )
    return manifest


def _validate_manifest_identity(identity: Any) -> None:
    if not isinstance(identity, Mapping):
        raise CheckpointIntegrityError("manifest identity must be a mapping")
    try:
        _require_exact_hex(identity.get("git_sha"), field="identity.git_sha", lengths={40, 64})
        for field in (
            "model_spec_hash",
            "tokenizer_hash",
            "tokenizer_vocab_hash",
            "dataset_manifest_hash",
            "run_manifest_hash",
            "training_config_hash",
            "optimizer_hash",
            "scheduler_hash",
            "environment_hash",
        ):
            _require_exact_hex(identity.get(field), field=f"identity.{field}", lengths={64})
        lock_hash = identity.get("environment_lock_hash")
        if lock_hash is not None:
            _require_exact_hex(lock_hash, field="identity.environment_lock_hash", lengths={64})
    except ValueError as exc:
        raise CheckpointIntegrityError(str(exc)) from exc

    hash_pairs = (
        ("model_spec", "model_spec_hash"),
        ("training_config", "training_config_hash"),
        ("optimizer", "optimizer_hash"),
        ("scheduler", "scheduler_hash"),
        ("environment", "environment_hash"),
    )
    for payload_key, hash_key in hash_pairs:
        if hash_json(identity.get(payload_key)) != identity.get(hash_key):
            raise CheckpointIntegrityError(f"{hash_key} does not match {payload_key}")


def prepare_checkpoint_load(directory: str | Path) -> VerifiedCheckpoint:
    """Read each checkpoint byte once and verify that exact immutable snapshot.

    This closes the check/use gap in the original loader: later mutations of the
    source directory cannot change the bytes consumed by
    :func:`load_verified_checkpoint`.
    """

    root = Path(directory)
    _require_checkpoint_directory(root)
    manifest_bytes = _read_regular_bytes(root, MANIFEST_NAME)
    checksum_bytes = _read_regular_bytes(root, MANIFEST_CHECKSUM_NAME)
    manifest = _parse_manifest_bytes(manifest_bytes, checksum_bytes)
    _validate_manifest_identity(manifest.get("identity"))

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != _PAYLOAD_NAMES:
        raise CheckpointIntegrityError(
            f"manifest file inventory must be exactly {sorted(_PAYLOAD_NAMES)}"
        )

    payloads: dict[str, bytes] = {}
    for name in sorted(_PAYLOAD_NAMES):
        record = files.get(name)
        if not isinstance(record, Mapping):
            raise CheckpointIntegrityError(f"invalid file record for {name}")
        try:
            expected_hash = _require_exact_hex(
                record.get("sha256"), field=f"files.{name}.sha256", lengths={64}
            )
        except ValueError as exc:
            raise CheckpointIntegrityError(str(exc)) from exc
        expected_bytes = record.get("bytes")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
        ):
            raise CheckpointIntegrityError(f"invalid byte length for {name}")
        data = _read_regular_bytes(root, name)
        if len(data) != expected_bytes:
            raise CheckpointIntegrityError(f"size mismatch for {name}")
        if sha256_bytes(data) != expected_hash:
            raise CheckpointIntegrityError(f"checksum mismatch for {name}")
        payloads[name] = data

    expected_id = hash_json({"identity": manifest["identity"], "files": manifest["files"]})
    if expected_id != manifest.get("checkpoint_id"):
        raise CheckpointIntegrityError("checkpoint_id does not match identity and artifact records")
    return VerifiedCheckpoint(
        _manifest_bytes=manifest_bytes,
        _artifacts=MappingProxyType(payloads),
    )


def verify_checkpoint(directory: str | Path) -> dict[str, Any]:
    """Verify exact inventory, lineage identities, and every payload checksum."""

    return prepare_checkpoint_load(directory).manifest


def assert_identity(
    manifest: Mapping[str, Any],
    *,
    git_sha: str | None = None,
    model_spec_hash: str | None = None,
    tokenizer_hash: str | None = None,
    tokenizer_vocab_hash: str | None = None,
    dataset_manifest_hash: str | None = None,
    run_manifest_hash: str | None = None,
) -> None:
    """Fail closed when a requested lineage constraint differs from the checkpoint."""

    identity = manifest["identity"]
    expected = {
        "git_sha": git_sha,
        "model_spec_hash": model_spec_hash,
        "tokenizer_hash": tokenizer_hash,
        "tokenizer_vocab_hash": tokenizer_vocab_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "run_manifest_hash": run_manifest_hash,
    }
    mismatches = {
        key: {"expected": value, "actual": identity.get(key)}
        for key, value in expected.items()
        if value is not None and identity.get(key) != value
    }
    if mismatches:
        raise CheckpointCompatibilityError(f"checkpoint identity mismatch: {mismatches}")


def _decode_verified_state(
    verified: VerifiedCheckpoint,
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    try:
        arrays = load_safetensors_bytes(verified._artifacts[WEIGHTS_NAME])
    except Exception as exc:
        raise CheckpointIntegrityError("weights.safetensors cannot be decoded") from exc
    try:
        state_arrays = load_safetensors_bytes(verified._artifacts[STATE_TENSORS_NAME])
    except Exception as exc:
        raise CheckpointIntegrityError("state.safetensors cannot be decoded") from exc
    try:
        state_tree = json.loads(verified._artifacts[STATE_TREE_NAME].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError("state.json cannot be decoded") from exc
    try:
        combined_state = unpack_state_tree(state_tree, state_arrays)
    except (StateTreeError, KeyError, TypeError, ValueError) as exc:
        raise CheckpointIntegrityError("checkpoint state tree is structurally invalid") from exc
    if not isinstance(combined_state, Mapping):
        raise CheckpointIntegrityError("checkpoint combined state must be a mapping")
    if "rng" not in combined_state:
        raise CheckpointIntegrityError("checkpoint combined state is missing RNG state")
    return arrays, combined_state


def load_verified_checkpoint(
    verified: VerifiedCheckpoint,
    *,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    strict_model: bool = True,
    restore_rng: bool = True,
    expected_git_sha: str | None = None,
    expected_model_spec_hash: str | None = None,
    expected_tokenizer_hash: str | None = None,
    expected_tokenizer_vocab_hash: str | None = None,
    expected_dataset_manifest_hash: str | None = None,
    expected_run_manifest_hash: str | None = None,
) -> LoadResult:
    """Preflight/decode a verified byte snapshot, then mutate requested targets."""

    manifest = verified.manifest
    assert_identity(
        manifest,
        git_sha=expected_git_sha,
        model_spec_hash=expected_model_spec_hash,
        tokenizer_hash=expected_tokenizer_hash,
        tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
        dataset_manifest_hash=expected_dataset_manifest_hash,
        run_manifest_hash=expected_run_manifest_hash,
    )
    arrays, combined_state = _decode_verified_state(verified)
    materialized = _prepare_model_weights(model, arrays, strict_model)
    if optimizer is not None and combined_state.get("optimizer") is None:
        raise CheckpointCompatibilityError(
            "optimizer was requested but checkpoint has no optimizer state"
        )
    if scheduler is not None and combined_state.get("scheduler") is None:
        raise CheckpointCompatibilityError(
            "scheduler was requested but checkpoint has no scheduler state"
        )
    if restore_rng:
        _preflight_rng_state(combined_state["rng"])

    # No checkpoint byte is reopened after this point. All integrity, identity,
    # payload decoding, model-shape and supported RNG compatibility checks above
    # completed before the first mutation.
    _apply_model_weights(model, materialized, strict_model)
    if optimizer is not None:
        optimizer.load_state_dict(combined_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(combined_state["scheduler"])
    if restore_rng:
        restore_rng_state(combined_state["rng"])
    return LoadResult(
        manifest=copy.deepcopy(manifest),
        trainer_state=combined_state.get("trainer", {}),
        rng_state=combined_state["rng"],
    )


def load_checkpoint(
    directory: str | Path,
    *,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    strict_model: bool = True,
    restore_rng: bool = True,
    expected_git_sha: str | None = None,
    expected_model_spec_hash: str | None = None,
    expected_tokenizer_hash: str | None = None,
    expected_tokenizer_vocab_hash: str | None = None,
    expected_dataset_manifest_hash: str | None = None,
    expected_run_manifest_hash: str | None = None,
) -> LoadResult:
    """Snapshot+verify exact checkpoint bytes, then restore requested targets."""

    verified = prepare_checkpoint_load(directory)
    return load_verified_checkpoint(
        verified,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        strict_model=strict_model,
        restore_rng=restore_rng,
        expected_git_sha=expected_git_sha,
        expected_model_spec_hash=expected_model_spec_hash,
        expected_tokenizer_hash=expected_tokenizer_hash,
        expected_tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
        expected_dataset_manifest_hash=expected_dataset_manifest_hash,
        expected_run_manifest_hash=expected_run_manifest_hash,
    )
