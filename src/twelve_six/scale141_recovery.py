"""Immutable-generation recovery lifecycle for SCALE-141.

D05 checkpoint-v1 directories remain immutable. This module owns a small mutable
selection index plus a content-addressed checkpoint object store. Ordinal
``generation-N`` directories are retained as immutable local history snapshots;
portable recovery authority is bound to ``checkpoints/<checkpoint_id>``.
Optional D04 replay state is published as an immutable sidecar only after the
checkpoint manifest exists and before the current pointer advances.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twelve_six.checkpoint import hash_json, sha256_file, verify_checkpoint
from twelve_six.checkpoint.durability import (
    _atomic_publish_directory_noreplace,
    fsync_checkpoint_tree,
    fsync_parent_directory,
)
from twelve_six.checkpoint.recovery_lock import exclusive_recovery_lock
from twelve_six.scale141_resume_sidecar import (
    ResumeSidecarContext,
    ResumeSidecarError,
    cleanup_orphan_resume_sidecars,
    load_resume_sidecar,
    publish_resume_sidecar,
    remove_resume_sidecar,
    validate_resume_reference,
)

POINTER_SCHEMA = "12-6.scale141-recovery-pointer.v1"
_GENERATION = re.compile(r"^generation-(\d{8})$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
CURRENT_NAME = "current.json"
CHECKPOINTS_DIR = "checkpoints"
MANIFEST_NAME = "manifest.json"
MAX_POINTER_BYTES = 64 * 1024


class RecoveryLifecycleError(RuntimeError):
    pass


class RecoveryPointerUpdateInterrupted(RecoveryLifecycleError):
    pass


@dataclass(frozen=True, slots=True)
class RecoveryResolution:
    path: Path
    reference: dict[str, Any]
    manifest: dict[str, Any]
    resume_state: dict[str, Any] | None = None
    content_path: Path | None = None


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


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise RecoveryLifecycleError(f"recovery pointer {field} is invalid")
    return value


def _content_key(checkpoint_id: str) -> str:
    checkpoint_id = _require_sha256(checkpoint_id, field="checkpoint_id")
    return f"{CHECKPOINTS_DIR}/{checkpoint_id}"


def _pointer_payload(
    *,
    generation: int,
    checkpoint_id: str,
    manifest_sha256: str,
    source_sha: str,
    run_manifest_hash: str,
    optimizer_step: int,
    tokens_seen: int,
    resume_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_id = _require_sha256(checkpoint_id, field="checkpoint_id")
    manifest_sha256 = _require_sha256(manifest_sha256, field="manifest_sha256")
    value: dict[str, Any] = {
        "schema": POINTER_SCHEMA,
        "generation": generation,
        "directory": f"generations/{_generation_name(generation)}",
        "object_key": _content_key(checkpoint_id),
        "checkpoint_id": checkpoint_id,
        "manifest_sha256": manifest_sha256,
        "source_sha": source_sha,
        "run_manifest_hash": run_manifest_hash,
        "optimizer_step": optimizer_step,
        "tokens_seen": tokens_seen,
    }
    if resume_state is not None:
        try:
            value["resume_state"] = validate_resume_reference(
                resume_state, generation=generation
            )
        except ResumeSidecarError as exc:
            raise RecoveryLifecycleError("recovery resume sidecar reference is invalid") from exc
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
    checkpoint_id = _require_sha256(pointer.get("checkpoint_id"), field="checkpoint_id")
    if pointer.get("object_key") != _content_key(checkpoint_id):
        raise RecoveryLifecycleError("recovery pointer object_key/checkpoint_id mismatch")
    _require_sha256(pointer.get("manifest_sha256"), field="manifest_sha256")
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
    if "resume_state" in pointer:
        try:
            pointer["resume_state"] = validate_resume_reference(
                pointer["resume_state"], generation=generation
            )
        except ResumeSidecarError as exc:
            raise RecoveryLifecycleError("recovery pointer resume sidecar is invalid") from exc
    return pointer


def _read_pointer_snapshot(path: Path) -> bytes:
    """Read one immutable pointer snapshot without reopening its pathname."""

    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise RecoveryLifecycleError("recovery pointer does not exist") from exc
    except OSError as exc:
        raise RecoveryLifecycleError("recovery pointer is unreadable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RecoveryLifecycleError("recovery pointer must be a regular non-symlink file")

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError as exc:
        raise RecoveryLifecycleError("recovery pointer changed while opening") from exc
    except OSError as exc:
        raise RecoveryLifecycleError("recovery pointer is unreadable") from exc

    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise RecoveryLifecycleError("recovery pointer must be a regular non-symlink file")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise RecoveryLifecycleError("recovery pointer changed while opening")
        if opened.st_size > MAX_POINTER_BYTES:
            raise RecoveryLifecycleError("recovery pointer exceeds maximum supported size")

        remaining = MAX_POINTER_BYTES + 1
        chunks: list[bytes] = []
        while remaining > 0:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        snapshot = b"".join(chunks)
        if len(snapshot) > MAX_POINTER_BYTES:
            raise RecoveryLifecycleError("recovery pointer exceeds maximum supported size")
        return snapshot
    except OSError as exc:
        raise RecoveryLifecycleError("recovery pointer is unreadable") from exc
    finally:
        os.close(fd)


def _read_pointer(root: Path) -> dict[str, Any]:
    path = root / CURRENT_NAME
    try:
        value = json.loads(_read_pointer_snapshot(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
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


def _ensure_real_directory(path: Path) -> None:
    created = False
    try:
        observed = path.lstat()
    except FileNotFoundError:
        path.mkdir()
        created = True
        observed = path.lstat()
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise RecoveryLifecycleError(f"recovery directory must be a real directory: {path.name}")
    if created:
        _fsync_directory(path.parent)


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


def _assert_manifest_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path / MANIFEST_NAME)
    if actual != expected:
        raise RecoveryLifecycleError(
            "recovery checkpoint manifest SHA mismatch: "
            f"expected={expected}, actual={actual}"
        )


def _assert_expected_reference(
    pointer: Mapping[str, Any], expected_reference: Mapping[str, Any] | None
) -> None:
    if expected_reference is None:
        return
    required = (
        "generation",
        "object_key",
        "checkpoint_id",
        "manifest_sha256",
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
    if "resume_state" in expected_reference and expected_reference.get("resume_state") != pointer.get(
        "resume_state"
    ):
        mismatches["resume_state"] = {
            "expected": expected_reference.get("resume_state"),
            "actual": pointer.get("resume_state"),
        }
    if mismatches:
        raise RecoveryLifecycleError(
            f"recovery pointer does not match phase boundary reference: {mismatches}"
        )


def recovery_reference(pointer: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_pointer(pointer)
    reference = {
        key: value[key]
        for key in (
            "generation",
            "object_key",
            "checkpoint_id",
            "manifest_sha256",
            "pointer_sha256",
            "source_sha",
            "run_manifest_hash",
            "optimizer_step",
            "tokens_seen",
        )
    }
    if "resume_state" in value:
        reference["resume_state"] = dict(value["resume_state"])
    return reference


def _content_path(root: Path, checkpoint_id: str) -> Path:
    return root / _content_key(checkpoint_id)


def _verify_pointer_checkpoint(
    path: Path,
    pointer: Mapping[str, Any],
    *,
    expected_source_sha: str | None = None,
    expected_run_manifest_hash: str | None = None,
    expected_step: int | None = None,
    expected_tokens_seen: int | None = None,
) -> dict[str, Any]:
    manifest = verify_checkpoint(path)
    _assert_manifest_binding(
        manifest,
        pointer,
        expected_source_sha=expected_source_sha,
        expected_run_manifest_hash=expected_run_manifest_hash,
        expected_step=expected_step,
        expected_tokens_seen=expected_tokens_seen,
    )
    _assert_manifest_sha256(path, str(pointer["manifest_sha256"]))
    return manifest


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

    content_path = _content_path(recovery_root, str(pointer["checkpoint_id"]))
    manifest = _verify_pointer_checkpoint(
        content_path,
        pointer,
        expected_source_sha=expected_source_sha,
        expected_run_manifest_hash=expected_run_manifest_hash,
        expected_step=expected_step,
        expected_tokens_seen=expected_tokens_seen,
    )

    # Keep verifying the ordinal immutable snapshot too. It is local retention
    # metadata, not the portable content address, but divergence is corruption.
    generation_path = recovery_root / "generations" / _generation_name(pointer["generation"])
    generation_manifest = _verify_pointer_checkpoint(generation_path, pointer)
    if generation_manifest != manifest:
        raise RecoveryLifecycleError("ordinal generation diverges from content-addressed checkpoint")

    resume_state = None
    if "resume_state" in pointer:
        try:
            resume_state = load_resume_sidecar(
                recovery_root,
                generation=pointer["generation"],
                checkpoint_path=content_path,
                manifest=manifest,
                reference=pointer["resume_state"],
            )
        except ResumeSidecarError as exc:
            raise RecoveryLifecycleError("recovery D04 resume sidecar failed validation") from exc
    return RecoveryResolution(
        path=generation_path,
        content_path=content_path,
        reference=recovery_reference(pointer),
        manifest=manifest,
        resume_state=resume_state,
    )


def _publish_checkpoint_clone_noreplace(source: Path, destination: Path) -> None:
    """Publish a verified immutable clone without a clobber-capable rename."""

    if destination.exists() or destination.is_symlink():
        raise RecoveryLifecycleError("next recovery generation unexpectedly already exists")
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.clone-", dir=destination.parent)
    )
    staging = staging_root / "checkpoint"
    try:
        shutil.copytree(source, staging, symlinks=True)
        verify_checkpoint(staging)
        expected_names = frozenset(entry.name for entry in staging.iterdir())
        fsync_checkpoint_tree(staging, expected_names=expected_names)
        try:
            _atomic_publish_directory_noreplace(staging, destination)
        except FileExistsError as exc:
            raise RecoveryLifecycleError(
                "next recovery generation appeared before publication"
            ) from exc
        fsync_parent_directory(destination)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _publish_content_object_from_staging(
    recovery_root: Path,
    staging: Path,
    manifest: Mapping[str, Any],
) -> tuple[Path, str, str, dict[str, Any]]:
    """Publish or authenticate one immutable content object.

    ``checkpoint_id`` intentionally excludes non-identity manifest metadata such
    as ``created_at_utc``. Therefore a repeated save can produce the same content
    address with different manifest bytes. Once an object already exists, that
    verified object's manifest and manifest SHA are canonical for every downstream
    pointer/sidecar binding; the private staged manifest is not an authority.
    """

    checkpoint_id = _require_sha256(manifest.get("checkpoint_id"), field="checkpoint_id")
    object_key = _content_key(checkpoint_id)
    object_root = recovery_root / CHECKPOINTS_DIR
    _ensure_real_directory(object_root)
    destination = recovery_root / object_key

    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise RecoveryLifecycleError("content-addressed checkpoint object is not a real directory")
        existing = verify_checkpoint(destination)
        if existing.get("checkpoint_id") != checkpoint_id:
            raise RecoveryLifecycleError("content-addressed checkpoint object identity mismatch")
        canonical_manifest_sha256 = sha256_file(destination / MANIFEST_NAME)
        _assert_manifest_sha256(destination, canonical_manifest_sha256)
        return destination, object_key, canonical_manifest_sha256, existing

    try:
        _atomic_publish_directory_noreplace(staging, destination)
    except FileExistsError:
        published = verify_checkpoint(destination)
        if published.get("checkpoint_id") != checkpoint_id:
            raise RecoveryLifecycleError("content-addressed checkpoint publication collision")
    else:
        fsync_parent_directory(destination)
        published = verify_checkpoint(destination)

    if published.get("checkpoint_id") != checkpoint_id:
        raise RecoveryLifecycleError("published content-addressed checkpoint identity mismatch")
    canonical_manifest_sha256 = sha256_file(destination / MANIFEST_NAME)
    _assert_manifest_sha256(destination, canonical_manifest_sha256)
    return destination, object_key, canonical_manifest_sha256, published


def _publish_recovery_generation_unlocked(
    root: str | Path,
    *,
    save_generation: Callable[[Path], Mapping[str, Any] | None],
    expected_source_sha: str,
    expected_run_manifest_hash: str,
    expected_step: int,
    expected_tokens_seen: int,
    build_resume_state: Callable[[ResumeSidecarContext], Mapping[str, Any]] | None = None,
    failpoint: str | None = None,
) -> dict[str, Any]:
    if failpoint not in (None, "after_sidecar_before_pointer", "before_pointer_replace"):
        raise RecoveryLifecycleError(f"unknown recovery publication failpoint: {failpoint}")
    if failpoint == "after_sidecar_before_pointer" and build_resume_state is None:
        raise RecoveryLifecycleError("after_sidecar_before_pointer requires a resume sidecar")

    recovery_root = Path(root)
    recovery_root.mkdir(parents=True, exist_ok=True)

    pointer_path = recovery_root / CURRENT_NAME
    if pointer_path.exists() or pointer_path.is_symlink():
        # Never advance over an invalid last-known-good pointer.
        resolve_recovery_generation(recovery_root)

    generations_root = recovery_root / "generations"
    if not generations_root.exists():
        generations_root.mkdir()
        _fsync_directory(recovery_root)
    _ensure_real_directory(generations_root)
    existing = _generation_numbers(recovery_root)
    generation = (existing[-1] + 1) if existing else 1
    generation_path = generations_root / _generation_name(generation)
    if generation_path.exists() or generation_path.is_symlink():
        raise RecoveryLifecycleError("next recovery generation unexpectedly already exists")

    # The callback receives only a private staging location. The stable portable
    # child locator is derived after checkpoint_id has been computed and verified.
    staging_root = Path(tempfile.mkdtemp(prefix=".checkpoint-stage-", dir=recovery_root))
    staging = staging_root / "checkpoint"
    try:
        save_generation(staging)
        staged_manifest = verify_checkpoint(staging)
        checkpoint_id = _require_sha256(
            staged_manifest.get("checkpoint_id"), field="checkpoint_id"
        )
        content_path, object_key, manifest_sha256, manifest = (
            _publish_content_object_from_staging(
                recovery_root, staging, staged_manifest
            )
        )
        identity = manifest.get("identity")
        if not isinstance(identity, Mapping):
            raise RecoveryLifecycleError("canonical recovery checkpoint identity is missing")
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

    pointer = _pointer_payload(
        generation=generation,
        checkpoint_id=checkpoint_id,
        manifest_sha256=manifest_sha256,
        source_sha=str(identity.get("git_sha")),
        run_manifest_hash=str(identity.get("run_manifest_hash")),
        optimizer_step=int(identity.get("step")),
        tokens_seen=int(identity.get("tokens_seen")),
    )
    if pointer["object_key"] != object_key:
        raise RecoveryLifecycleError("derived content-addressed object key mismatch")
    _assert_manifest_binding(
        manifest,
        pointer,
        expected_source_sha=expected_source_sha,
        expected_run_manifest_hash=expected_run_manifest_hash,
        expected_step=expected_step,
        expected_tokens_seen=expected_tokens_seen,
    )
    _assert_manifest_sha256(content_path, manifest_sha256)

    # Retain an immutable ordinal snapshot for local history/cleanup. It is never
    # advertised as the content address and must equal the authoritative object.
    _publish_checkpoint_clone_noreplace(content_path, generation_path)
    _verify_pointer_checkpoint(generation_path, pointer)

    if build_resume_state is not None:
        try:
            resume_reference = publish_resume_sidecar(
                recovery_root,
                generation=generation,
                checkpoint_path=content_path,
                manifest=manifest,
                build_exposure_state=build_resume_state,
            )
        except ResumeSidecarError as exc:
            raise RecoveryLifecycleError("D04 resume sidecar publication failed") from exc
        pointer = _pointer_payload(
            generation=generation,
            checkpoint_id=checkpoint_id,
            manifest_sha256=manifest_sha256,
            source_sha=str(identity.get("git_sha")),
            run_manifest_hash=str(identity.get("run_manifest_hash")),
            optimizer_step=int(identity.get("step")),
            tokens_seen=int(identity.get("tokens_seen")),
            resume_state=resume_reference,
        )
        if failpoint == "after_sidecar_before_pointer":
            raise RecoveryPointerUpdateInterrupted(
                "injected interruption after D04 sidecar publication and before pointer update"
            )

    pointer_failpoint = failpoint if failpoint == "before_pointer_replace" else None
    _atomic_publish_pointer(recovery_root, pointer, failpoint=pointer_failpoint)
    return recovery_reference(pointer)


def _retained_checkpoint_ids(root: Path, numbers: set[int]) -> set[str]:
    retained: set[str] = set()
    for number in numbers:
        path = root / "generations" / _generation_name(number)
        manifest = verify_checkpoint(path)
        retained.add(_require_sha256(manifest.get("checkpoint_id"), field="checkpoint_id"))
    return retained


def _cleanup_content_objects(root: Path, *, retained_ids: set[str]) -> list[str]:
    object_root = root / CHECKPOINTS_DIR
    if not object_root.exists():
        return []
    _ensure_real_directory(object_root)
    removed: list[str] = []
    for entry in object_root.iterdir():
        if entry.name in retained_ids:
            continue
        if _HEX64.fullmatch(entry.name) is None:
            raise RecoveryLifecycleError(
                f"unexpected content-addressed checkpoint entry: {entry.name}"
            )
        if entry.is_symlink() or not entry.is_dir():
            raise RecoveryLifecycleError("refusing cleanup through checkpoint-object symlink")
        shutil.rmtree(entry)
        removed.append(entry.name)
    if removed:
        _fsync_directory(object_root)
    return removed


def _cleanup_recovery_generations_unlocked(root: str | Path, *, keep: int = 2) -> dict[str, Any]:
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
        try:
            remove_resume_sidecar(recovery_root, generation=number)
        except ResumeSidecarError as exc:
            raise RecoveryLifecycleError("resume sidecar cleanup failed") from exc
        removed.append(path.name)

    retained = set(_generation_numbers(recovery_root))
    try:
        removed_sidecars = cleanup_orphan_resume_sidecars(
            recovery_root, retained_generations=retained
        )
    except ResumeSidecarError as exc:
        raise RecoveryLifecycleError("orphan resume sidecar cleanup failed") from exc

    retained_ids = _retained_checkpoint_ids(recovery_root, retained)
    removed_content_objects = _cleanup_content_objects(
        recovery_root, retained_ids=retained_ids
    )

    # Prove cleanup did not remove or corrupt the only authoritative generation.
    after = resolve_recovery_generation(
        recovery_root, expected_reference=current.reference
    )
    return {
        "current_generation": after.reference["generation"],
        "current_checkpoint_id": after.reference["checkpoint_id"],
        "removed": removed,
        "removed_resume_sidecars": removed_sidecars,
        "removed_content_objects": removed_content_objects,
        "retained_generation_count": len(_generation_numbers(recovery_root)),
    }


def publish_recovery_generation(
    root: str | Path,
    *,
    save_generation: Callable[[Path], Mapping[str, Any] | None],
    expected_source_sha: str,
    expected_run_manifest_hash: str,
    expected_step: int,
    expected_tokens_seen: int,
    build_resume_state: Callable[[ResumeSidecarContext], Mapping[str, Any]] | None = None,
    failpoint: str | None = None,
) -> dict[str, Any]:
    """Publish one generation under a crash-releasing cross-process lock."""

    with exclusive_recovery_lock(root) as locked_root:
        return _publish_recovery_generation_unlocked(
            locked_root,
            save_generation=save_generation,
            expected_source_sha=expected_source_sha,
            expected_run_manifest_hash=expected_run_manifest_hash,
            expected_step=expected_step,
            expected_tokens_seen=expected_tokens_seen,
            build_resume_state=build_resume_state,
            failpoint=failpoint,
        )


def cleanup_recovery_generations(root: str | Path, *, keep: int = 2) -> dict[str, Any]:
    """Clean immutable generations without racing an active publisher."""

    with exclusive_recovery_lock(root) as locked_root:
        return _cleanup_recovery_generations_unlocked(locked_root, keep=keep)
