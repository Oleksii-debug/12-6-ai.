"""Conservative, transaction-safe Hugging Face-style export for verified checkpoints."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .core import (
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST_CHECKSUM_NAME,
    MANIFEST_NAME,
    STATE_TENSORS_NAME,
    STATE_TREE_NAME,
    WEIGHTS_NAME,
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    _add_failure_note_preserving_primary,
    _validate_manifest_identity,
    hash_json,
    prepare_checkpoint_load,
    sha256_bytes,
)

EXPORT_ATTESTATION_NAME = "12-6-export.json"
EXPORT_CHECKSUM_NAME = "12-6-export.sha256"
PARITY_REQUEST_NAME = "12-6-parity-request.json"
EXPORTED_WEIGHTS_NAME = "model.safetensors"
EXPORTED_CONFIG_NAME = "config.json"
EXPORTED_SOURCE_MANIFEST_NAME = "12-6-checkpoint-manifest.json"
_EXPORT_FILES = frozenset(
    {
        EXPORTED_WEIGHTS_NAME,
        EXPORTED_CONFIG_NAME,
        EXPORTED_SOURCE_MANIFEST_NAME,
        EXPORT_ATTESTATION_NAME,
        EXPORT_CHECKSUM_NAME,
        PARITY_REQUEST_NAME,
    }
)
_MAX_EXPORT_METADATA_BYTES = 8 * 1024 * 1024
_MAX_EXPORT_CHECKSUM_BYTES = 256
_STREAM_HASH_CHUNK_BYTES = 1024 * 1024
_REQUIRED_PARITY_CHECKS = [
    "prompt_token_identity",
    "next_token_logit_parity",
    "greedy_generation_parity",
]
_COMPATIBILITY = {
    "layout": "HF_STYLE_SAFETENSORS_DIRECTORY",
    "weights": "EXACT_CANONICAL_BYTE_COPY",
    "transformers_architecture": "NOT_CLAIMED",
    "runtime_logit_generation_parity": "NOT_TESTED",
}
_PARITY_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "status",
        "checkpoint_id",
        "reference_weights_sha256",
        "candidate_weights_sha256",
        "candidate_config_sha256",
        "required_checks",
        "authority",
        "hook_result",
    }
)
_ATTESTATION_FIELDS = frozenset(
    {
        "schema",
        "checkpoint_id",
        "source_manifest_sha256",
        "model_safetensors_sha256",
        "config_sha256",
        "parity_request_sha256",
        "compatibility",
    }
)
_SOURCE_MANIFEST_FIELDS = frozenset(
    {
        "format",
        "format_version",
        "created_at_utc",
        "checkpoint_id",
        "identity",
        "files",
        "serialization",
    }
)
_SOURCE_FILE_NAMES = frozenset({WEIGHTS_NAME, STATE_TENSORS_NAME, STATE_TREE_NAME})
_SOURCE_FILE_RECORD_FIELDS = frozenset({"sha256", "bytes"})
_SOURCE_SERIALIZATION = {
    "weights": "safetensors",
    "state_tensors": "safetensors",
    "state_tree": "canonical-json",
    "pickle": False,
}
_SOURCE_IDENTITY_FIELDS = frozenset(
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
        "environment",
        "environment_hash",
        "environment_lock_hash",
    }
)
ParityHook = Callable[[Path, Path], Mapping[str, Any]]


def _read_regular_bytes(
    root: Path,
    name: str,
    *,
    max_bytes: int | None = None,
) -> bytes:
    path = root / name
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(f"missing HF-style export artifact: {name}") from exc
    except OSError as exc:
        raise CheckpointIntegrityError(
            f"cannot inspect HF-style export artifact: {name}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CheckpointIntegrityError(
            f"HF-style export artifact must be a regular non-symlink file: {name}"
        )

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CheckpointIntegrityError(
            f"cannot safely open HF-style export artifact: {name}"
        ) from exc
    primary_exc: BaseException | None = None
    try:
        try:
            opened = os.fstat(fd)
        except OSError as exc:
            raise CheckpointIntegrityError(
                f"cannot inspect HF-style export artifact: {name}"
            ) from exc
        if not stat.S_ISREG(opened.st_mode):
            raise CheckpointIntegrityError(f"HF-style export artifact changed type: {name}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise CheckpointIntegrityError(
                f"HF-style export artifact changed while opening: {name}"
            )
        if max_bytes is not None and opened.st_size > max_bytes:
            raise CheckpointIntegrityError(
                f"HF-style export artifact exceeds read limit: {name}"
            )
        try:
            with os.fdopen(fd, "rb", closefd=False) as handle:
                if max_bytes is None:
                    data = handle.read()
                else:
                    data = handle.read(max_bytes + 1)
        except OSError as exc:
            raise CheckpointIntegrityError(
                f"cannot read HF-style export artifact: {name}"
            ) from exc
        if max_bytes is not None and len(data) > max_bytes:
            raise CheckpointIntegrityError(
                f"HF-style export artifact exceeds read limit: {name}"
            )
        return data
    except BaseException as exc:
        primary_exc = exc
        raise
    finally:
        try:
            os.close(fd)
        except OSError as close_exc:
            if primary_exc is not None:
                _add_failure_note_preserving_primary(
                    primary_exc,
                    f"HF-style export artifact close also failed: {close_exc!r}",
                )
            else:
                raise CheckpointIntegrityError(
                    f"cannot close HF-style export artifact: {name}"
                ) from close_exc


def _stream_regular_sha256(
    root: Path,
    name: str,
) -> tuple[str, int]:
    """Hash one regular artifact through a bounded-memory held descriptor."""

    path = root / name
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(f"missing HF-style export artifact: {name}") from exc
    except OSError as exc:
        raise CheckpointIntegrityError(
            f"cannot inspect HF-style export artifact: {name}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CheckpointIntegrityError(
            f"HF-style export artifact must be a regular non-symlink file: {name}"
        )

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CheckpointIntegrityError(
            f"cannot safely open HF-style export artifact: {name}"
        ) from exc
    primary_exc: BaseException | None = None
    try:
        try:
            opened = os.fstat(fd)
        except OSError as exc:
            raise CheckpointIntegrityError(
                f"cannot inspect HF-style export artifact: {name}"
            ) from exc
        if not stat.S_ISREG(opened.st_mode):
            raise CheckpointIntegrityError(f"HF-style export artifact changed type: {name}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise CheckpointIntegrityError(
                f"HF-style export artifact changed while opening: {name}"
            )

        digest = hashlib.sha256()
        byte_count = 0
        try:
            with os.fdopen(fd, "rb", closefd=False) as handle:
                while True:
                    chunk = handle.read(_STREAM_HASH_CHUNK_BYTES)
                    if not chunk:
                        break
                    digest.update(chunk)
                    byte_count += len(chunk)
        except OSError as exc:
            raise CheckpointIntegrityError(
                f"cannot read HF-style export artifact: {name}"
            ) from exc
        return digest.hexdigest(), byte_count
    except BaseException as exc:
        primary_exc = exc
        raise
    finally:
        try:
            os.close(fd)
        except OSError as close_exc:
            if primary_exc is not None:
                _add_failure_note_preserving_primary(
                    primary_exc,
                    f"HF-style export artifact close also failed: {close_exc!r}",
                )
            else:
                raise CheckpointIntegrityError(
                    f"cannot close HF-style export artifact: {name}"
                ) from close_exc


def _require_export_root_identity(
    root: Path,
    expected_identity: tuple[int, int],
) -> None:
    try:
        observed = root.lstat()
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(
            f"HF-style export directory disappeared while reading: {root}"
        ) from exc
    except OSError as exc:
        raise CheckpointIntegrityError(
            f"cannot inspect HF-style export directory while reading: {root}"
        ) from exc
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != expected_identity
    ):
        raise CheckpointIntegrityError(
            f"HF-style export directory identity changed while reading: {root}"
        )


def _read_export_snapshot(root: Path) -> tuple[dict[str, bytes], tuple[int, int]]:
    try:
        root_stat = root.lstat()
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(f"HF-style export directory does not exist: {root}") from exc
    except OSError as exc:
        raise CheckpointIntegrityError(
            f"cannot inspect HF-style export directory: {root}"
        ) from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise CheckpointIntegrityError(
            "HF-style export root must be a real directory, not a symlink"
        )
    root_identity = (root_stat.st_dev, root_stat.st_ino)
    names: set[str] = set()
    try:
        for entry in root.iterdir():
            names.add(entry.name)
            if len(names) > len(_EXPORT_FILES):
                raise CheckpointIntegrityError(
                    "HF-style export inventory exceeds expected size"
                )
    except OSError as exc:
        raise CheckpointIntegrityError(
            f"cannot enumerate HF-style export directory: {root}"
        ) from exc
    _require_export_root_identity(root, root_identity)
    if names != _EXPORT_FILES:
        missing = sorted(_EXPORT_FILES - names)
        unexpected = sorted(names - _EXPORT_FILES)
        raise CheckpointIntegrityError(
            f"HF-style export inventory mismatch: missing={missing}, unexpected={unexpected}"
        )
    payloads: dict[str, bytes] = {}
    for name in sorted(_EXPORT_FILES):
        if name == EXPORTED_WEIGHTS_NAME:
            continue
        _require_export_root_identity(root, root_identity)
        max_bytes = (
            _MAX_EXPORT_CHECKSUM_BYTES
            if name == EXPORT_CHECKSUM_NAME
            else _MAX_EXPORT_METADATA_BYTES
        )
        payloads[name] = _read_regular_bytes(
            root,
            name,
            max_bytes=max_bytes,
        )
        _require_export_root_identity(root, root_identity)
    return payloads, root_identity


def _strict_json_bytes(value: Any, *, artifact: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise CheckpointIntegrityError(
            f"{artifact} is not strict finite JSON"
        ) from exc


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number")
    return parsed


def _object_without_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _json_object(data: bytes, *, artifact: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise CheckpointIntegrityError(f"{artifact} is not valid strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CheckpointIntegrityError(f"{artifact} must contain a JSON object")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    artifact: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise CheckpointIntegrityError(
            f"{artifact} fields mismatch: missing={missing}, unexpected={unexpected}"
        )


def _validate_source_manifest_structure(
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    _require_exact_fields(
        source_manifest,
        _SOURCE_MANIFEST_FIELDS,
        artifact=EXPORTED_SOURCE_MANIFEST_NAME,
    )
    if type(source_manifest.get("format_version")) is not int:
        raise CheckpointIntegrityError(
            "exported source manifest format_version must be an integer"
        )
    created_at = source_manifest.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at.endswith("Z"):
        raise CheckpointIntegrityError(
            "exported source manifest created_at_utc is not canonical UTC"
        )
    try:
        parsed_created_at = datetime.fromisoformat(
            f"{created_at[:-1]}+00:00"
        )
    except ValueError as exc:
        raise CheckpointIntegrityError(
            "exported source manifest created_at_utc is not canonical UTC"
        ) from exc
    canonical_created_at = parsed_created_at.isoformat().replace("+00:00", "Z")
    if parsed_created_at.tzinfo != UTC or canonical_created_at != created_at:
        raise CheckpointIntegrityError(
            "exported source manifest created_at_utc is not canonical UTC"
        )

    files = source_manifest.get("files")
    if not isinstance(files, dict):
        raise CheckpointIntegrityError(
            "exported source manifest files must be a mapping"
        )
    actual_files = set(files)
    if actual_files != _SOURCE_FILE_NAMES:
        missing = sorted(_SOURCE_FILE_NAMES - actual_files)
        unexpected = sorted(actual_files - _SOURCE_FILE_NAMES)
        raise CheckpointIntegrityError(
            "exported source manifest file inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name in sorted(_SOURCE_FILE_NAMES):
        record = files.get(name)
        if not isinstance(record, dict):
            raise CheckpointIntegrityError(
                f"exported source manifest file record is invalid: {name}"
            )
        _require_exact_fields(
            record,
            _SOURCE_FILE_RECORD_FIELDS,
            artifact=f"{EXPORTED_SOURCE_MANIFEST_NAME}:{name}",
        )
        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise CheckpointIntegrityError(
                f"exported source manifest file hash is invalid: {name}"
            )
        byte_count = record.get("bytes")
        if (
            not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count < 0
        ):
            raise CheckpointIntegrityError(
                f"exported source manifest file byte length is invalid: {name}"
            )

    serialization = source_manifest.get("serialization")
    if serialization != _SOURCE_SERIALIZATION:
        raise CheckpointIntegrityError(
            "exported source manifest serialization declaration is noncanonical"
        )
    return files


def _validate_source_manifest_identity(identity: dict[str, Any]) -> None:
    _require_exact_fields(
        identity,
        _SOURCE_IDENTITY_FIELDS,
        artifact=f"{EXPORTED_SOURCE_MANIFEST_NAME}:identity",
    )
    _validate_manifest_identity(identity)


def verify_hf_directory(directory: str | Path) -> dict[str, Any]:
    """Verify one exact HF-style export directory without trusting path metadata."""

    root = Path(directory)
    payloads, root_identity = _read_export_snapshot(root)
    try:
        checksum_text = payloads[EXPORT_CHECKSUM_NAME].decode("ascii")
    except UnicodeDecodeError as exc:
        raise CheckpointIntegrityError(f"{EXPORT_CHECKSUM_NAME} must be ASCII") from exc
    checksum_suffix = f"  {EXPORT_ATTESTATION_NAME}\n"
    if not checksum_text.endswith(checksum_suffix):
        raise CheckpointIntegrityError(f"invalid {EXPORT_CHECKSUM_NAME} format")
    checksum_digest = checksum_text[: -len(checksum_suffix)]
    if (
        len(checksum_digest) != 64
        or checksum_digest != checksum_digest.lower()
        or any(char not in "0123456789abcdef" for char in checksum_digest)
    ):
        raise CheckpointIntegrityError(f"invalid {EXPORT_CHECKSUM_NAME} format")
    if checksum_digest != sha256_bytes(payloads[EXPORT_ATTESTATION_NAME]):
        raise CheckpointIntegrityError("HF-style export attestation checksum mismatch")

    source_manifest = _json_object(
        payloads[EXPORTED_SOURCE_MANIFEST_NAME],
        artifact=EXPORTED_SOURCE_MANIFEST_NAME,
    )
    if (
        source_manifest.get("format") != FORMAT_NAME
        or source_manifest.get("format_version") != FORMAT_VERSION
    ):
        raise CheckpointCompatibilityError(
            "exported source manifest has unsupported checkpoint format"
        )
    files = _validate_source_manifest_structure(source_manifest)
    identity = source_manifest.get("identity")
    if not isinstance(identity, dict):
        raise CheckpointIntegrityError(
            "exported source manifest is missing identity mapping"
        )
    _validate_source_manifest_identity(identity)
    checkpoint_id = hash_json({"identity": identity, "files": files})
    if source_manifest.get("checkpoint_id") != checkpoint_id:
        raise CheckpointIntegrityError(
            "exported source manifest checkpoint_id is self-inconsistent"
        )
    weights_record = files.get(WEIGHTS_NAME)
    if not isinstance(weights_record, dict):
        raise CheckpointIntegrityError("source manifest is missing canonical weights record")

    _json_object(payloads[EXPORTED_CONFIG_NAME], artifact=EXPORTED_CONFIG_NAME)

    _require_export_root_identity(root, root_identity)
    weights_sha, weights_bytes = _stream_regular_sha256(
        root,
        EXPORTED_WEIGHTS_NAME,
    )
    _require_export_root_identity(root, root_identity)

    config_sha = sha256_bytes(payloads[EXPORTED_CONFIG_NAME])
    source_manifest_sha = sha256_bytes(payloads[EXPORTED_SOURCE_MANIFEST_NAME])
    parity_sha = sha256_bytes(payloads[PARITY_REQUEST_NAME])
    if weights_record.get("sha256") != weights_sha:
        raise CheckpointIntegrityError(
            "exported model.safetensors differs from canonical weights hash"
        )
    if weights_record.get("bytes") != weights_bytes:
        raise CheckpointIntegrityError(
            "exported model.safetensors differs from canonical byte length"
        )

    parity = _json_object(payloads[PARITY_REQUEST_NAME], artifact=PARITY_REQUEST_NAME)
    _require_exact_fields(
        parity,
        _PARITY_REQUEST_FIELDS,
        artifact=PARITY_REQUEST_NAME,
    )
    if parity.get("schema") != "12-6.export-parity-request.v2":
        raise CheckpointCompatibilityError("unsupported export parity request schema")
    expected_parity = {
        "checkpoint_id": checkpoint_id,
        "reference_weights_sha256": weights_sha,
        "candidate_weights_sha256": weights_sha,
        "candidate_config_sha256": config_sha,
        "required_checks": _REQUIRED_PARITY_CHECKS,
        "authority": "D07_or_independent_parity_harness",
    }
    for field, expected in expected_parity.items():
        if parity.get(field) != expected:
            raise CheckpointIntegrityError(f"export parity request {field} mismatch")
    status = parity.get("status")
    hook_result = parity.get("hook_result")
    if status == "NOT_TESTED":
        if hook_result is not None:
            raise CheckpointIntegrityError("NOT_TESTED parity request cannot attach hook evidence")
    elif status == "EXTERNAL_EVIDENCE_ATTACHED":
        if not isinstance(hook_result, dict):
            raise CheckpointIntegrityError(
                "EXTERNAL_EVIDENCE_ATTACHED parity request requires mapping evidence"
            )
    else:
        raise CheckpointIntegrityError(f"unsupported export parity status: {status!r}")

    attestation = _json_object(
        payloads[EXPORT_ATTESTATION_NAME], artifact=EXPORT_ATTESTATION_NAME
    )
    _require_exact_fields(
        attestation,
        _ATTESTATION_FIELDS,
        artifact=EXPORT_ATTESTATION_NAME,
    )
    if attestation.get("schema") != "12-6.hf-style-export.v2":
        raise CheckpointCompatibilityError("unsupported HF-style export attestation schema")
    if attestation.get("compatibility") != _COMPATIBILITY:
        raise CheckpointIntegrityError("HF-style export compatibility claims changed unexpectedly")
    expected_attestation = {
        "checkpoint_id": checkpoint_id,
        "source_manifest_sha256": source_manifest_sha,
        "model_safetensors_sha256": weights_sha,
        "config_sha256": config_sha,
        "parity_request_sha256": parity_sha,
    }
    for field, expected in expected_attestation.items():
        if attestation.get(field) != expected:
            raise CheckpointIntegrityError(f"HF-style export attestation {field} mismatch")
    return attestation


def _temporary_directory_identity(path: Path) -> tuple[int, int]:
    """Pin the created private root before exposing its pathname to a parity hook."""

    try:
        observed = path.lstat()
    except OSError as exc:
        raise CheckpointIntegrityError(
            f"cannot inspect private temporary root: {path}"
        ) from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise CheckpointIntegrityError(f"private temporary root changed type: {path}")
    if not observed.st_ino:
        raise CheckpointIntegrityError(
            f"private temporary root inode identity unavailable: {path}"
        )
    return observed.st_dev, observed.st_ino


def _create_private_temp_directory(
    *,
    prefix: str,
    parent: Path,
    label: str,
) -> tuple[Path, tuple[int, int]]:
    """Create and pin an empty private root before any recursive cleanup is allowed."""

    path = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    try:
        identity = _temporary_directory_identity(path)
    except BaseException as exc:  # noqa: BLE001 - preserve process interrupts
        try:
            os.rmdir(path)
        except BaseException as cleanup_exc:  # noqa: BLE001 - preserve primary
            _add_failure_note_preserving_primary(
                exc,
                f"{label} empty-root cleanup also failed: {cleanup_exc!r}",
            )
        raise
    return path, identity


def _remove_temp_path_strict(
    path: Path, *, label: str, expected_identity: tuple[int, int]
) -> None:
    """Never recursively remove a substituted private-root pathname.

    This closes synchronous parity-hook root replacement. As with publication,
    the last identity check does not defend against arbitrary concurrent
    same-user filesystem mutation after the check.
    """

    try:
        current = path.lstat()
    except FileNotFoundError as exc:
        raise CheckpointIntegrityError(
            f"{label} root disappeared before cleanup: {path}"
        ) from exc
    if (current.st_dev, current.st_ino) != expected_identity:
        raise CheckpointIntegrityError(
            f"{label} root identity changed before cleanup: {path}"
        )
    if not stat.S_ISDIR(current.st_mode) or stat.S_ISLNK(current.st_mode):
        raise CheckpointIntegrityError(f"{label} root changed type: {path}")

    shutil.rmtree(path)
    if os.path.lexists(path):
        raise CheckpointIntegrityError(f"{label} remained after cleanup: {path}")


def _cleanup_temp_paths_strict(
    paths: tuple[tuple[Path | None, str, tuple[int, int] | None], ...],
    *,
    primary_exc: BaseException | None = None,
) -> None:
    failures: list[tuple[str, BaseException]] = []
    for path, label, expected_identity in paths:
        if path is None:
            continue
        if expected_identity is None:
            failures.append(
                (label, CheckpointIntegrityError(f"{label} is missing its creation identity"))
            )
            continue
        try:
            _remove_temp_path_strict(
                path, label=label, expected_identity=expected_identity
            )
        except BaseException as exc:  # noqa: BLE001 - cleanup covers interrupts
            failures.append((label, exc))
    if failures:
        labels = ", ".join(label for label, _ in failures)
        if primary_exc is not None:
            for label, cleanup_exc in failures:
                _add_failure_note_preserving_primary(
                    primary_exc,
                    f"{label} cleanup also failed: {cleanup_exc!r}",
                )
            return
        first_interrupt = next(
            (
                (label, cleanup_exc)
                for label, cleanup_exc in failures
                if isinstance(
                    cleanup_exc,
                    (KeyboardInterrupt, SystemExit, GeneratorExit),
                )
            ),
            None,
        )
        if first_interrupt is not None:
            interrupt_label, interrupt_exc = first_interrupt
            for label, cleanup_exc in failures:
                if cleanup_exc is interrupt_exc:
                    continue
                _add_failure_note_preserving_primary(
                    interrupt_exc,
                    (
                        f"{label} cleanup also failed while "
                        f"{interrupt_label} raised: {cleanup_exc!r}"
                    ),
                )
            raise interrupt_exc
        first_failure = failures[0][1]
        raise CheckpointIntegrityError(
            f"temporary cleanup failed for: {labels}"
        ) from first_failure


def _materialize_verified_reference_owned(
    verified: Any,
    parent: Path,
    name: str,
) -> tuple[Path, tuple[int, int]]:
    reference, reference_identity = _create_private_temp_directory(
        prefix=f".{name}.reference-",
        parent=parent,
        label="verified checkpoint reference",
    )
    try:
        manifest_bytes = verified._manifest_bytes
        (reference / MANIFEST_NAME).write_bytes(manifest_bytes)
        (reference / MANIFEST_CHECKSUM_NAME).write_text(
            f"{sha256_bytes(manifest_bytes)}  {MANIFEST_NAME}\n",
            encoding="ascii",
        )
        for artifact in (WEIGHTS_NAME, STATE_TENSORS_NAME, STATE_TREE_NAME):
            (reference / artifact).write_bytes(verified._artifacts[artifact])
        prepare_checkpoint_load(reference)
        return reference, reference_identity
    except BaseException as exc:
        _cleanup_temp_paths_strict(
            ((reference, "verified checkpoint reference", reference_identity),),
            primary_exc=exc,
        )
        raise


def _materialize_verified_reference(verified: Any, parent: Path, name: str) -> Path:
    reference, _ = _materialize_verified_reference_owned(verified, parent, name)
    return reference


def _materialize_hook_candidate_owned(
    *,
    parent: Path,
    name: str,
    weights: bytes,
    config: bytes,
    source_manifest: bytes,
) -> tuple[Path, tuple[int, int]]:
    candidate, candidate_identity = _create_private_temp_directory(
        prefix=f".{name}.hook-candidate-",
        parent=parent,
        label="HF parity hook candidate",
    )
    try:
        (candidate / EXPORTED_WEIGHTS_NAME).write_bytes(weights)
        (candidate / EXPORTED_CONFIG_NAME).write_bytes(config)
        (candidate / EXPORTED_SOURCE_MANIFEST_NAME).write_bytes(source_manifest)
        return candidate, candidate_identity
    except BaseException as exc:
        _cleanup_temp_paths_strict(
            ((candidate, "HF parity hook candidate", candidate_identity),),
            primary_exc=exc,
        )
        raise


def _materialize_hook_candidate(
    *,
    parent: Path,
    name: str,
    weights: bytes,
    config: bytes,
    source_manifest: bytes,
) -> Path:
    candidate, _ = _materialize_hook_candidate_owned(
        parent=parent,
        name=name,
        weights=weights,
        config=config,
        source_manifest=source_manifest,
    )
    return candidate


def _publish_directory_noreplace(staging: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a concurrent destination."""

    if os.name == "nt":
        try:
            os.rename(staging, destination)
        except FileExistsError:
            raise FileExistsError(
                f"export destination appeared during publish: {destination}"
            ) from None
        return

    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("atomic no-replace directory publish requires libc renameat2")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            os.fsencode(staging),
            at_fdcwd,
            os.fsencode(destination),
            rename_noreplace,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                f"export destination appeared during publish: {destination}"
            )
        raise OSError(error_number, os.strerror(error_number), destination)

    raise RuntimeError(
        "atomic no-replace HF-style export publication is unsupported on this platform"
    )


def export_hf_directory(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    hf_config: Mapping[str, Any],
    overwrite: bool = False,
    parity_hook: ParityHook | None = None,
) -> Path:
    """Create an immutable, verified HF-style SafeTensors directory.

    Source checkpoint bytes are snapshotted and verified once through D05's
    transactional loader. Any external parity hook sees only disposable
    snapshot-derived reference/candidate trees, which are strictly cleaned before
    final publication staging is created. The final export is then built from
    immutable in-memory bytes, verified, and atomically published with no
    intervening untrusted callback.

    Existing destinations are immutable. ``overwrite=True`` is retained only for
    API compatibility and still fails closed rather than deleting prior evidence.

    The output is HF-*style* only. It does not claim Transformers architecture
    compatibility or runtime logit/generation parity. An optional external parity
    hook may attach evidence while those compatibility claims remain unchanged.
    """

    source = Path(checkpoint_dir)
    verified = prepare_checkpoint_load(source)
    source_manifest = verified.manifest
    source_manifest_bytes = verified._manifest_bytes
    source_weights_bytes = verified._artifacts[WEIGHTS_NAME]

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        suffix = (
            " (overwrite=True does not permit destructive replacement)"
            if overwrite
            else ""
        )
        raise FileExistsError(f"export destination already exists: {destination}{suffix}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    config_bytes = _strict_json_bytes(
        dict(hf_config),
        artifact=EXPORTED_CONFIG_NAME,
    ) + b"\n"
    weights_sha = sha256_bytes(source_weights_bytes)
    config_sha = sha256_bytes(config_bytes)
    parity_request: dict[str, Any] = {
        "schema": "12-6.export-parity-request.v2",
        "status": "NOT_TESTED",
        "checkpoint_id": source_manifest["checkpoint_id"],
        "reference_weights_sha256": weights_sha,
        "candidate_weights_sha256": weights_sha,
        "candidate_config_sha256": config_sha,
        "required_checks": list(_REQUIRED_PARITY_CHECKS),
        "authority": "D07_or_independent_parity_harness",
        "hook_result": None,
    }

    if parity_hook is not None:
        reference: Path | None = None
        candidate: Path | None = None
        reference_identity: tuple[int, int] | None = None
        candidate_identity: tuple[int, int] | None = None
        parity_primary_exc: BaseException | None = None
        try:
            reference, reference_identity = _materialize_verified_reference_owned(
                verified,
                destination.parent,
                destination.name,
            )
            candidate, candidate_identity = _materialize_hook_candidate_owned(
                parent=destination.parent,
                name=destination.name,
                weights=source_weights_bytes,
                config=config_bytes,
                source_manifest=source_manifest_bytes,
            )
            result = parity_hook(reference, candidate)
            if not isinstance(result, Mapping):
                raise TypeError("parity_hook must return a mapping")
            parity_request["hook_result"] = dict(result)
            parity_request["status"] = "EXTERNAL_EVIDENCE_ATTACHED"
            parity_bytes = _strict_json_bytes(
                parity_request,
                artifact=PARITY_REQUEST_NAME,
            ) + b"\n"
        except BaseException as exc:
            parity_primary_exc = exc
            raise
        finally:
            _cleanup_temp_paths_strict(
                (
                    (candidate, "HF parity hook candidate", candidate_identity),
                    (reference, "verified checkpoint reference", reference_identity),
                ),
                primary_exc=parity_primary_exc,
            )
    else:
        parity_bytes = _strict_json_bytes(
            parity_request,
            artifact=PARITY_REQUEST_NAME,
        ) + b"\n"

    attestation = {
        "schema": "12-6.hf-style-export.v2",
        "checkpoint_id": source_manifest["checkpoint_id"],
        "source_manifest_sha256": sha256_bytes(source_manifest_bytes),
        "model_safetensors_sha256": weights_sha,
        "config_sha256": config_sha,
        "parity_request_sha256": sha256_bytes(parity_bytes),
        "compatibility": dict(_COMPATIBILITY),
    }
    attestation_bytes = _strict_json_bytes(
        attestation,
        artifact=EXPORT_ATTESTATION_NAME,
    ) + b"\n"

    staging, staging_identity = _create_private_temp_directory(
        prefix=f".{destination.name}.staging-",
        parent=destination.parent,
        label="HF export staging",
    )
    staging_primary_exc: BaseException | None = None
    try:
        (staging / EXPORTED_WEIGHTS_NAME).write_bytes(source_weights_bytes)
        (staging / EXPORTED_CONFIG_NAME).write_bytes(config_bytes)
        (staging / EXPORTED_SOURCE_MANIFEST_NAME).write_bytes(source_manifest_bytes)
        (staging / PARITY_REQUEST_NAME).write_bytes(parity_bytes)
        (staging / EXPORT_ATTESTATION_NAME).write_bytes(attestation_bytes)
        (staging / EXPORT_CHECKSUM_NAME).write_text(
            f"{sha256_bytes(attestation_bytes)}  {EXPORT_ATTESTATION_NAME}\n",
            encoding="ascii",
        )

        verify_hf_directory(staging)
        _publish_directory_noreplace(staging, destination)
        staging = None
        return destination
    except BaseException as exc:
        staging_primary_exc = exc
        raise
    finally:
        if staging is not None:
            _cleanup_temp_paths_strict(
                ((staging, "HF export staging", staging_identity),),
                primary_exc=staging_primary_exc,
            )
