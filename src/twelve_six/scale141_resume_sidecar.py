"""Immutable D04 exposure-state sidecars for SCALE-141 recovery generations.

Checkpoint-v1 directories stay byte-for-byte immutable.  D04 owns exposure
semantics; D05 only persists the already-built D04 state after the checkpoint
manifest exists, binds it to that exact manifest and ordered next exposure, and
verifies the sidecar before recovery can use it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twelve_six.checkpoint import D04_RESUME_BINDING_SCHEMA, hash_json, sha256_file
from twelve_six.checkpoint.durability import _atomic_publish_directory_noreplace

SIDECAR_SCHEMA = "12-6.scale141-d04-resume-sidecar.v1"
SIDECAR_ROOT = "resume-states"
SIDECAR_FILE = "state.json"
_HEX = frozenset("0123456789abcdef")


class ResumeSidecarError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ResumeSidecarContext:
    generation: str
    checkpoint_manifest_sha256: str
    checkpoint_id: str
    source_sha: str
    run_manifest_hash: str
    optimizer_step: int
    tokens_seen: int


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise ResumeSidecarError(f"{label} must be exact lowercase 64-hex SHA-256")
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ResumeSidecarError(f"{label} must be a non-negative integer")
    return value


def _generation_name(number: int) -> str:
    if not isinstance(number, int) or isinstance(number, bool) or not 0 < number <= 99_999_999:
        raise ResumeSidecarError("resume sidecar generation is outside supported range")
    return f"generation-{number:08d}"


def _checkpoint_d04_data(manifest: Mapping[str, Any]) -> dict[str, str]:
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ResumeSidecarError("checkpoint identity is missing for D04 sidecar")
    training_config = identity.get("training_config")
    if not isinstance(training_config, Mapping):
        raise ResumeSidecarError("checkpoint training_config is missing for D04 sidecar")
    data = training_config.get("data")
    if not isinstance(data, Mapping):
        raise ResumeSidecarError("checkpoint D04 data binding is missing")
    if data.get("resume_binding_schema") != D04_RESUME_BINDING_SCHEMA:
        raise ResumeSidecarError("checkpoint D04 resume binding schema mismatch")
    fields = (
        "ledger_identity_sha256",
        "materialization_identity_sha256",
        "packing_identity_sha256",
        "exposure_plan_identity_sha256",
        "ordered_next_exposure_identity_sha256",
    )
    return {field: _require_sha256(data.get(field), f"checkpoint.data.{field}") for field in fields}


def _context(
    *, generation: int, checkpoint_path: Path, manifest: Mapping[str, Any]
) -> ResumeSidecarContext:
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ResumeSidecarError("checkpoint identity is missing for D04 sidecar")
    checkpoint_id = manifest.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ResumeSidecarError("checkpoint_id is missing for D04 sidecar")
    source_sha = identity.get("git_sha")
    if not isinstance(source_sha, str) or len(source_sha) != 40:
        raise ResumeSidecarError("checkpoint source SHA is invalid for D04 sidecar")
    run_manifest_hash = _require_sha256(
        identity.get("run_manifest_hash"), "checkpoint run_manifest_hash"
    )
    optimizer_step = _require_nonnegative_int(identity.get("step"), "checkpoint optimizer_step")
    tokens_seen = _require_nonnegative_int(identity.get("tokens_seen"), "checkpoint tokens_seen")
    return ResumeSidecarContext(
        generation=_generation_name(generation),
        checkpoint_manifest_sha256=sha256_file(checkpoint_path / "manifest.json"),
        checkpoint_id=checkpoint_id,
        source_sha=source_sha,
        run_manifest_hash=run_manifest_hash,
        optimizer_step=optimizer_step,
        tokens_seen=tokens_seen,
    )


def _validate_exposure_state(
    state: Mapping[str, Any],
    *,
    context: ResumeSidecarContext,
    d04_data: Mapping[str, str],
) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ResumeSidecarError("D04 exposure state builder must return a mapping")
    value = deepcopy(dict(state))
    for field in (
        "ledger_identity_sha256",
        "materialization_identity_sha256",
        "packing_identity_sha256",
    ):
        actual = _require_sha256(value.get(field), f"D04 exposure state {field}")
        if actual != d04_data[field]:
            raise ResumeSidecarError(f"D04 exposure state {field} mismatches checkpoint binding")
    _require_sha256(value.get("state_identity_sha256"), "D04 exposure state identity")
    consumed = _require_nonnegative_int(
        value.get("consumed_loss_positions"), "D04 exposure consumed_loss_positions"
    )
    binding = value.get("trainer_state_binding")
    if not isinstance(binding, Mapping):
        raise ResumeSidecarError("D04 exposure trainer_state_binding is missing")
    expected_binding = {
        "checkpoint_generation": context.generation,
        "checkpoint_manifest_sha256": context.checkpoint_manifest_sha256,
        "optimizer_step": context.optimizer_step,
    }
    mismatches = {
        field: {"expected": expected, "actual": binding.get(field)}
        for field, expected in expected_binding.items()
        if binding.get(field) != expected
    }
    target_count = _require_nonnegative_int(
        binding.get("trainer_nonignored_target_count"),
        "D04 exposure trainer_nonignored_target_count",
    )
    if target_count != consumed:
        mismatches["trainer_nonignored_target_count"] = {
            "expected": consumed,
            "actual": target_count,
        }
    if mismatches:
        raise ResumeSidecarError(f"D04 exposure/checkpoint binding mismatch: {mismatches}")
    return value


def _payload(
    *,
    context: ResumeSidecarContext,
    d04_data: Mapping[str, str],
    exposure_state: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": SIDECAR_SCHEMA,
        "generation": context.generation,
        "checkpoint_manifest_sha256": context.checkpoint_manifest_sha256,
        "checkpoint_id": context.checkpoint_id,
        "source_sha": context.source_sha,
        "run_manifest_hash": context.run_manifest_hash,
        "optimizer_step": context.optimizer_step,
        "tokens_seen": context.tokens_seen,
        **dict(d04_data),
        "state_identity_sha256": exposure_state["state_identity_sha256"],
        "exposure_state": deepcopy(dict(exposure_state)),
    }
    value["payload_sha256"] = hash_json(value)
    return value


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_payload_directory(root: Path, generation: int, payload: Mapping[str, Any]) -> Path:
    sidecar_root = root / SIDECAR_ROOT
    sidecar_root.mkdir(parents=True, exist_ok=True)
    destination = sidecar_root / _generation_name(generation)
    if destination.exists() or destination.is_symlink():
        raise ResumeSidecarError("resume sidecar generation unexpectedly already exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=sidecar_root))
    try:
        state_path = staging / SIDECAR_FILE
        with state_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(staging)
        try:
            _atomic_publish_directory_noreplace(staging, destination)
        except FileExistsError as exc:
            raise ResumeSidecarError("resume sidecar destination appeared before publication") from exc
        _fsync_directory(sidecar_root)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def publish_resume_sidecar(
    root: str | Path,
    *,
    generation: int,
    checkpoint_path: str | Path,
    manifest: Mapping[str, Any],
    build_exposure_state: Callable[[ResumeSidecarContext], Mapping[str, Any]],
) -> dict[str, Any]:
    recovery_root = Path(root)
    checkpoint = Path(checkpoint_path)
    d04_data = _checkpoint_d04_data(manifest)
    context = _context(generation=generation, checkpoint_path=checkpoint, manifest=manifest)
    state = _validate_exposure_state(
        build_exposure_state(context), context=context, d04_data=d04_data
    )
    payload = _payload(context=context, d04_data=d04_data, exposure_state=state)
    directory = _write_payload_directory(recovery_root, generation, payload)
    file_path = directory / SIDECAR_FILE
    return {
        "schema": SIDECAR_SCHEMA,
        "directory": f"{SIDECAR_ROOT}/{context.generation}",
        "file": SIDECAR_FILE,
        "file_sha256": sha256_file(file_path),
        "payload_sha256": payload["payload_sha256"],
        "checkpoint_manifest_sha256": context.checkpoint_manifest_sha256,
        "state_identity_sha256": state["state_identity_sha256"],
        "ordered_next_exposure_identity_sha256": d04_data[
            "ordered_next_exposure_identity_sha256"
        ],
    }


def validate_resume_reference(value: Mapping[str, Any], *, generation: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResumeSidecarError("recovery resume sidecar reference must be a mapping")
    reference = dict(value)
    generation_name = _generation_name(generation)
    if reference.get("schema") != SIDECAR_SCHEMA:
        raise ResumeSidecarError("recovery resume sidecar schema mismatch")
    if reference.get("directory") != f"{SIDECAR_ROOT}/{generation_name}":
        raise ResumeSidecarError("recovery resume sidecar directory/generation mismatch")
    if reference.get("file") != SIDECAR_FILE:
        raise ResumeSidecarError("recovery resume sidecar file mismatch")
    for field in (
        "file_sha256",
        "payload_sha256",
        "checkpoint_manifest_sha256",
        "state_identity_sha256",
        "ordered_next_exposure_identity_sha256",
    ):
        _require_sha256(reference.get(field), f"recovery resume sidecar {field}")
    return reference


def _read_payload(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    expected_hash = _require_sha256(expected_file_sha256, "recovery resume sidecar file hash")
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ResumeSidecarError("recovery resume sidecar is missing") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ResumeSidecarError("recovery resume sidecar must be a regular non-symlink file")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ResumeSidecarError("recovery resume sidecar cannot be opened safely") from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ResumeSidecarError("recovery resume sidecar changed type while opening")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ResumeSidecarError("recovery resume sidecar changed while opening")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            data = handle.read()
    except OSError as exc:
        raise ResumeSidecarError("recovery resume sidecar is unreadable") from exc
    finally:
        os.close(fd)

    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ResumeSidecarError("recovery resume sidecar file hash mismatch")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeSidecarError("recovery resume sidecar is unreadable") from exc
    if not isinstance(value, dict):
        raise ResumeSidecarError("recovery resume sidecar payload must be a JSON object")
    return value


def load_resume_sidecar(
    root: str | Path,
    *,
    generation: int,
    checkpoint_path: str | Path,
    manifest: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    recovery_root = Path(root)
    checked_reference = validate_resume_reference(reference, generation=generation)
    directory = recovery_root / checked_reference["directory"]
    if directory.is_symlink() or not directory.is_dir():
        raise ResumeSidecarError("recovery resume sidecar directory is missing or unsafe")
    payload = _read_payload(directory / SIDECAR_FILE, checked_reference["file_sha256"])
    supplied_hash = payload.pop("payload_sha256", None)
    if supplied_hash != hash_json(payload):
        raise ResumeSidecarError("recovery resume sidecar payload self-hash mismatch")
    payload["payload_sha256"] = supplied_hash
    if supplied_hash != checked_reference["payload_sha256"]:
        raise ResumeSidecarError("recovery resume sidecar pointer/payload hash mismatch")

    d04_data = _checkpoint_d04_data(manifest)
    context = _context(
        generation=generation,
        checkpoint_path=Path(checkpoint_path),
        manifest=manifest,
    )
    checks = {
        "schema": (SIDECAR_SCHEMA, payload.get("schema")),
        "generation": (context.generation, payload.get("generation")),
        "checkpoint_manifest_sha256": (
            context.checkpoint_manifest_sha256,
            payload.get("checkpoint_manifest_sha256"),
        ),
        "checkpoint_id": (context.checkpoint_id, payload.get("checkpoint_id")),
        "source_sha": (context.source_sha, payload.get("source_sha")),
        "run_manifest_hash": (context.run_manifest_hash, payload.get("run_manifest_hash")),
        "optimizer_step": (context.optimizer_step, payload.get("optimizer_step")),
        "tokens_seen": (context.tokens_seen, payload.get("tokens_seen")),
        "reference_checkpoint_manifest_sha256": (
            checked_reference["checkpoint_manifest_sha256"],
            payload.get("checkpoint_manifest_sha256"),
        ),
        "reference_state_identity_sha256": (
            checked_reference["state_identity_sha256"],
            payload.get("state_identity_sha256"),
        ),
        "reference_ordered_next_exposure_identity_sha256": (
            checked_reference["ordered_next_exposure_identity_sha256"],
            payload.get("ordered_next_exposure_identity_sha256"),
        ),
    }
    for field, expected in d04_data.items():
        checks[f"checkpoint_{field}"] = (expected, payload.get(field))
    mismatches = {
        name: {"expected": expected, "actual": actual}
        for name, (expected, actual) in checks.items()
        if expected != actual
    }
    if mismatches:
        raise ResumeSidecarError(f"recovery resume sidecar binding mismatch: {mismatches}")

    state = payload.get("exposure_state")
    if not isinstance(state, Mapping):
        raise ResumeSidecarError("recovery resume sidecar exposure state is missing")
    return _validate_exposure_state(state, context=context, d04_data=d04_data)


def remove_resume_sidecar(root: str | Path, *, generation: int) -> None:
    path = Path(root) / SIDECAR_ROOT / _generation_name(generation)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        raise ResumeSidecarError("refusing cleanup through resume-sidecar symlink")
    shutil.rmtree(path)


def cleanup_orphan_resume_sidecars(
    root: str | Path, *, retained_generations: set[int]
) -> list[str]:
    sidecar_root = Path(root) / SIDECAR_ROOT
    if not sidecar_root.exists():
        return []
    if sidecar_root.is_symlink() or not sidecar_root.is_dir():
        raise ResumeSidecarError("resume sidecar root must be a real directory")
    removed: list[str] = []
    for path in sidecar_root.iterdir():
        if path.is_symlink() or not path.is_dir():
            raise ResumeSidecarError("resume sidecar root contains an unsafe entry")
        if not path.name.startswith("generation-") or len(path.name) != len("generation-00000000"):
            raise ResumeSidecarError("resume sidecar root contains an unknown entry")
        suffix = path.name.removeprefix("generation-")
        if not suffix.isdigit():
            raise ResumeSidecarError("resume sidecar root contains an invalid generation")
        number = int(suffix)
        if number not in retained_generations:
            shutil.rmtree(path)
            removed.append(path.name)
    return sorted(removed)
