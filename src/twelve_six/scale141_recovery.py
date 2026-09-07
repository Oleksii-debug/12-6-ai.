"""Immutable-generation recovery lifecycle for SCALE-141.

D05 checkpoint-v1 directories remain immutable.  This module owns only the
small mutable recovery index that selects one already-verified generation.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twelve_six.checkpoint import hash_json, verify_checkpoint

POINTER_SCHEMA = "12-6.scale141-recovery-pointer.v1"
_GENERATION = re.compile(r"^generation-(\d{8})$")
CURRENT_NAME = "current.json"


class RecoveryLifecycleError(RuntimeError):
    pass


class RecoveryPointerUpdateInterrupted(RecoveryLifecycleError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryResolution:
    path: Path
    reference: dict[str, Any]
    manifest: dict[str, Any]


def _generation_name(number: int) -> str:
    if number <= 0 or number > 99_999_999:
        raise RecoveryLifecycleError("recovery generation is outside supported range")
    return f"generation-{number:08d}"


def _generation_numbers(root: Path) -> list[int]:
    generations = root / "generations"
    if not generations.exists():
        return []
    if generations.is_symlink() or not generations.is_dir():
        raise RecoveryLifecycleError("recovery generations root must be a real directory")
    values: list[int] = []
    for entry in generations.iterdir():
        match = _GENERATION.fullmatch(entry.name)
        if match is None:
            continue
        if entry.is_symlink() or not entry.is_dir():
            raise RecoveryLifecycleError(
                f"recovery generation must be a real directory: {entry.name}"
            )
        values.append(int(match.group(1)))
    return sorted(values)


def _pointer_payload(
    *,
    generation: int,
    checkpoint_id: str,
    source_sha: str,
    run_manifest_hash: str,
    optimizer_step: int,
    tokens_seen: int,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": POINTER_SCHEMA,
        "generation": generation,
        "directory": f"generations/{_generation_name(generation)}",
        "checkpoint_id": checkpoint_id,
        "source_sha": source_sha,
        "run_manifest_hash": run_manifest_hash,
        "optimizer_step": optimizer_step,
        "tokens_seen": tokens_seen,
    }
    value["pointer_sha256"] = hash_json(value)
    return value


def _validate_pointer(value: Mapping[str, Any]) -> dict[str, Any]:
    pointer = dict(value)
    supplied_hash = pointer.pop("pointer_sha256", None)
    if supplied_hash != hash_json(pointer):
        raise RecoveryLifecycleError("recovery pointer self-hash mismatch")
    pointer["pointer_sha256"] = supplied_hash
    if pointer.get("schema") != POINTER_SCHEMA:
        raise RecoveryLifecycleError("recovery pointer schema mismatch")
    generation = pointer.get("generation")
    if not isinstance(generation, int) or isinstance(generation, bool) or generation <= 0:
        raise RecoveryLifecycleError("recovery pointer generation is invalid")
    expected_directory = f"generations/{_generation_name(generation)}"
    if pointer.get("directory") != expected_directory:
        raise RecoveryLifecycleError("recovery pointer directory/generation mismatch")
    if not isinstance(pointer.get("checkpoint_id"), str) or not pointer["checkpoint_id"]:
        raise RecoveryLifecycleError("recovery pointer checkpoint_id is missing")
    if not isinstance(pointer.get("source_sha"), str) or len(pointer["source_sha"]) != 40:
        raise RecoveryLifecycleError("recovery pointer source SHA is invalid")
    for key in ("run_manifest_hash",):
        value_at_key = pointer.get(key)
        if not isinstance(value_at_key, str) or len(value_at_key) != 64:
            raise RecoveryLifecycleError(f"recovery pointer {key} is invalid")
    for key in ("optimizer_step", "tokens_seen"):
        value_at_key = pointer.get(key)
        if not isinstance(value_at_key, int) or isinstance(value_at_key, bool) or value_at_key < 0:
            raise RecoveryLifecycleError(f"recovery pointer {key} is invalid")
    return pointer


def _read_pointer(root: Path) -> dict[str, Any]:
    path = root / CURRENT_NAME
    if path.is_symlink():
        raise RecoveryLifecycleError("recovery pointer must be a real file, not a symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RecoveryLifecycleError("recovery pointer does not exist") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryLifecycleError("recovery pointer is unreadable") from exc
    if not isinstance(value, dict):
        raise RecoveryLifecycleError("recovery pointer must be a JSON object")
    return _validate_pointer(value)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_publish_pointer(
    root: Path,
    value: Mapping[str, Any],
    *,
    failpoint: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=".current.", suffix=".tmp", dir=root)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if failpoint == "before_pointer_replace":
            raise RecoveryPointerUpdateInterrupted(
                "injected interruption before atomic recovery-pointer replacement"
            )
        if failpoint is not None:
            raise RecoveryLifecycleError(f"unknown recovery pointer failpoint: {failpoint}")
        os.replace(temp, root / CURRENT_NAME)
        _fsync_directory(root)
    finally:
        if temp.exists():
            temp.unlink()


def _assert_manifest_binding(
    manifest: Mapping[str, Any],
    pointer: Mapping[str, Any],
    *,
    expected_source_sha: str | None = None,
    expected_run_manifest_hash: str | None = None,
    expected_step: int | None = None,
    expected_tokens_seen: int | None = None,
) -> None:
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise RecoveryLifecycleError("verified recovery checkpoint identity is missing")
    checks = {
        "checkpoint_id": (pointer["checkpoint_id"], manifest.get("checkpoint_id")),
        "source_sha": (pointer["source_sha"], identity.get("git_sha")),
        "run_manifest_hash": (pointer["run_manifest_hash"], identity.get("run_manifest_hash")),
        "optimizer_step": (pointer["optimizer_step"], identity.get("step")),
        "tokens_seen": (pointer["tokens_seen"], identity.get("tokens_seen")),
        "expected_source_sha": (expected_source_sha, identity.get("git_sha")),
        "expected_run_manifest_hash": (
            expected_run_manifest_hash,
            identity.get("run_manifest_hash"),
        ),
        "expected_step": (expected_step, identity.get("step")),
        "expected_tokens_seen": (expected_tokens_seen, identity.get("tokens_seen")),
    }
    mismatches = {
        name: {"expected": expected, "actual": actual}
        for name, (expected, actual) in checks.items()
        if expected is not None and expected != actual
    }
    if mismatches:
        raise RecoveryLifecycleError(f"recovery checkpoint binding mismatch: {mismatches}")


def _assert_expected_reference(
    pointer: Mapping[str, Any], expected_reference: Mapping[str, Any] | None
) -> None:
    if expected_reference is None:
        return
    required = (
        "generation",
        "checkpoint_id",
        "pointer_sha256",
        "source_sha",
        "run_manifest_hash",
        "optimizer_step",
        "tokens_seen",
    )
    mismatches = {
        key: {"expected": expected_reference.get(key), "actual": pointer.get(key)}
        for key in required
        if expected_reference.get(key) != pointer.get(key)
    }
    if mismatches:
        raise RecoveryLifecycleError(
            f"recovery pointer does not match phase boundary reference: {mismatches}"
        )


def recovery_reference(pointer: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_pointer(pointer)
    return {
        key: value[key]
        for key in (
            "generation",
            "checkpoint_id",
            "pointer_sha256",
            "source_sha",
            "run_manifest_hash",
            "optimizer_step",
            "tokens_seen",
        )
    }


def resolve_recovery_generation(
    root: str | Path,
    *,
    expected_reference: Mapping[str, Any] | None = None,
    expected_source_sha: str | None = None,
    expected_run_manifest_hash: str | None = None,
    expected_step: int | None = None,
    expected_tokens_seen: int | None = None,
) -> RecoveryResolution:
    recovery_root = Path(root)
    pointer = _read_pointer(recovery_root)
    _assert_expected_reference(pointer, expected_reference)
    generation_path = recovery_root / "generations" / _generation_name(pointer["generation"])
    manifest = verify_checkpoint(generation_path)
    _assert_manifest_binding(
        manifest,
        pointer,
        expected_source_sha=expected_source_sha,
        expected_run_manifest_hash=expected_run_manifest_hash,
        expected_step=expected_step,
        expected_tokens_seen=expected_tokens_seen,
    )
    return RecoveryResolution(
        path=generation_path,
        reference=recovery_reference(pointer),
        manifest=manifest,
    )


def publish_recovery_generation(
    root: str | Path,
    *,
    save_generation: Callable[[Path], Mapping[str, Any] | None],
    expected_source_sha: str,
    expected_run_manifest_hash: str,
    expected_step: int,
    expected_tokens_seen: int,
    failpoint: str | None = None,
) -> dict[str, Any]:
    recovery_root = Path(root)
    recovery_root.mkdir(parents=True, exist_ok=True)

    pointer_path = recovery_root / CURRENT_NAME
    if pointer_path.exists() or pointer_path.is_symlink():
        # Never advance over an invalid last-known-good pointer.
        resolve_recovery_generation(recovery_root)

    generations_root = recovery_root / "generations"
    generations_root.mkdir(parents=True, exist_ok=True)
    existing = _generation_numbers(recovery_root)
    generation = (existing[-1] + 1) if existing else 1
    destination = generations_root / _generation_name(generation)
    if destination.exists() or destination.is_symlink():
        raise RecoveryLifecycleError("next recovery generation unexpectedly already exists")

    save_generation(destination)
    manifest = verify_checkpoint(destination)
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise RecoveryLifecycleError("saved recovery checkpoint identity is missing")
    pointer = _pointer_payload(
        generation=generation,
        checkpoint_id=str(manifest["checkpoint_id"]),
        source_sha=str(identity.get("git_sha")),
        run_manifest_hash=str(identity.get("run_manifest_hash")),
        optimizer_step=int(identity.get("step")),
        tokens_seen=int(identity.get("tokens_seen")),
    )
    _assert_manifest_binding(
        manifest,
        pointer,
        expected_source_sha=expected_source_sha,
        expected_run_manifest_hash=expected_run_manifest_hash,
        expected_step=expected_step,
        expected_tokens_seen=expected_tokens_seen,
    )
    _atomic_publish_pointer(recovery_root, pointer, failpoint=failpoint)
    return recovery_reference(pointer)


def cleanup_recovery_generations(root: str | Path, *, keep: int = 2) -> dict[str, Any]:
    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 1:
        raise ValueError("recovery cleanup keep must be >= 1")
    recovery_root = Path(root)
    current = resolve_recovery_generation(recovery_root)
    numbers = _generation_numbers(recovery_root)
    current_number = current.reference["generation"]
    others = [number for number in numbers if number != current_number]
    protected_others = set(others[-max(keep - 1, 0) :]) if keep > 1 else set()
    removed: list[str] = []
    for number in others:
        if number in protected_others:
            continue
        path = recovery_root / "generations" / _generation_name(number)
        if path.is_symlink():
            raise RecoveryLifecycleError("refusing cleanup through recovery-generation symlink")
        shutil.rmtree(path)
        removed.append(path.name)
    # Prove cleanup did not remove or corrupt the only authoritative generation.
    after = resolve_recovery_generation(
        recovery_root, expected_reference=current.reference
    )
    return {
        "current_generation": after.reference["generation"],
        "current_checkpoint_id": after.reference["checkpoint_id"],
        "removed": removed,
        "retained_generation_count": len(_generation_numbers(recovery_root)),
    }
