"""Adapt an independently identified D03 family vector for NEXT100-106."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import BinaryIO

from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError
from twelve_six.data.postdecontam_next100_adapter_v1 import adapt_family_vector_to_next100_106


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-vector", type=Path, required=True)
    parser.add_argument("--expected-family-vector-identity-sha256", required=True)
    parser.add_argument("--dedup-authority", type=Path, required=True)
    parser.add_argument("--expected-dedup-worker-id", required=True)
    parser.add_argument("--expected-dedup-head-sha", required=True)
    parser.add_argument("--expected-dedup-evidence-identity-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


MAX_AUTHORITY_JSON_BYTES = 1_048_576
MAX_AUTHORITY_JSON_DEPTH = 64
MAX_AUTHORITY_JSON_NODES = 10_000


MAX_AUTHORITY_INT_DIGITS = 64


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProjectionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_constant(value: str) -> object:
    raise ProjectionError(f"nonstandard JSON constant is forbidden: {value}")


def _strict_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ProjectionError(f"nonfinite JSON number is forbidden: {value}")
    significand = value.split("e", 1)[0].split("E", 1)[0]
    if parsed == 0.0 and any(digit in "123456789" for digit in significand):
        raise ProjectionError("nonzero JSON number underflowed to zero")
    return parsed


def _strict_int(value: str) -> int:
    if len(value.lstrip("-")) > MAX_AUTHORITY_INT_DIGITS:
        raise ProjectionError("adapter authority JSON integer exceeds digit limit")
    return int(value)


def _load_strict_object(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        decoded = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProjectionError(f"{label} is not strict UTF-8") from exc
    try:
        document = json.loads(
            decoded,
            object_pairs_hook=_strict_pairs,
            parse_constant=_strict_constant,
            parse_float=_strict_float,
            parse_int=_strict_int,
        )
    except ProjectionError:
        raise
    except (ValueError, RecursionError, OverflowError) as exc:
        raise ProjectionError(f"{label} is not strict JSON") from exc
    if not isinstance(document, dict):
        raise ProjectionError(f"{label} must contain a top-level JSON object")
    return document


def _load_authority_json(path: Path) -> dict[str, object]:
    """Read a bounded, unambiguous source before verifying its semantic identity."""
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_AUTHORITY_JSON_BYTES + 1)
    except OSError as exc:
        raise ProjectionError(f"cannot read adapter authority: {path}") from exc
    if len(raw) > MAX_AUTHORITY_JSON_BYTES:
        raise ProjectionError("adapter authority JSON exceeds byte limit")

    value = _load_strict_object(raw, label=f"adapter authority {path}")
    pending: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > MAX_AUTHORITY_JSON_NODES or depth > MAX_AUTHORITY_JSON_DEPTH:
            raise ProjectionError("adapter authority JSON structure limit exceeded")
        if isinstance(item, dict):
            for key, child in item.items():
                try:
                    key.encode("utf-8")
                except UnicodeError as exc:
                    raise ProjectionError("adapter authority has invalid Unicode") from exc
                pending.append((child, depth + 1))
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeError as exc:
                raise ProjectionError("adapter authority has invalid Unicode") from exc
    return value


def _write_staged_bytes(destination: BinaryIO, payload: bytes) -> None:
    """Do not publish unless the entire staged payload reached durable storage."""
    if destination.write(payload) != len(payload):
        raise OSError("incomplete staged adapter output write")
    destination.flush()
    os.fsync(destination.fileno())


def _same_inode(path: Path, identity: tuple[int, int]) -> bool:
    """Never clean up a competitor's replacement of our staged/final pathname."""
    try:
        info = path.stat(follow_symlinks=False)
    except OSError:
        return False
    return (info.st_dev, info.st_ino) == identity


def _staged_payload_matches(
    path: Path, identity: tuple[int, int], payload: bytes,
) -> bool:
    if not _same_inode(path, identity):
        return False
    try:
        with path.open("rb") as source:
            info = os.fstat(source.fileno())
            if (info.st_dev, info.st_ino) != identity:
                return False
            return source.read(len(payload) + 1) == payload
    except OSError:
        return False


def _write_new_output(
    path: Path, payload: bytes, *, family_vector: Path, dedup_authority: Path,
) -> None:
    """Create only; require trusted stable parents and validate final bytes.

    Cooperative filesystem faults are fail-closed; hostile same-user writers or
    concurrent parent-directory replacement cannot be made atomic portably.
    """
    if path.exists() or path.is_symlink():
        raise ProjectionError(f"refusing to overwrite existing adapter output: {path}")
    try:
        parent = path.parent.absolute()
        # Do not rely on a mutable ancestor symlink when publishing or cleaning up.
        if parent != parent.resolve(strict=True):
            raise ProjectionError("adapter output parent must have no symlink aliases")
        final = parent / path.name
        protected = {family_vector.resolve(), dedup_authority.resolve()}
        if final in protected:
            raise ProjectionError("adapter output must not alias an input authority")
    except (OSError, RuntimeError) as exc:
        raise ProjectionError("adapter output path cannot be resolved safely") from exc

    staged_path: Path | None = None
    identity: tuple[int, int] | None = None
    linked = False
    verified = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
            dir=parent, delete=False,
        ) as destination:
            staged_path = Path(destination.name)
            stage_stat = os.fstat(destination.fileno())
            identity = (stage_stat.st_dev, stage_stat.st_ino)
            if not identity[1]:
                raise OSError("filesystem does not expose a stable staged file identity")
            _write_staged_bytes(destination, payload)
        if not _staged_payload_matches(staged_path, identity, payload):
            raise ProjectionError("staged adapter bytes changed before publication")
        # Hard link is create-only, never a replacement of existing final evidence.
        os.link(staged_path, final)
        linked = True
        if (
            not _staged_payload_matches(final, identity, payload)
            or not _staged_payload_matches(staged_path, identity, payload)
            or parent != path.parent.absolute()
            or parent != path.parent.resolve(strict=True)
        ):
            raise ProjectionError("published adapter output failed byte/path verification")
        verified = True
    except FileExistsError as exc:
        raise ProjectionError(f"refusing to overwrite existing adapter output: {path}") from exc
    except OSError as exc:
        raise ProjectionError(f"cannot publish adapter output safely: {path}: {exc}") from exc
    finally:
        rollback_error: OSError | None = None
        cleanup_error: OSError | None = None
        if linked and not verified and identity is not None and _same_inode(final, identity):
            try:
                final.unlink()
            except OSError as exc:
                rollback_error = exc
        if staged_path is not None and identity is not None and _same_inode(staged_path, identity):
            try:
                staged_path.unlink()
            except OSError as exc:
                cleanup_error = exc
        if rollback_error is not None:
            raise ProjectionError(
                f"ROLLBACK_INCOMPLETE: invalid adapter output may remain: {final}"
            ) from rollback_error
        if cleanup_error is not None:
            if verified:
                # The fully verified final file is committed. Never report
                # non-publication merely because its staging alias remains.
                print(
                    "OUTPUT_COMMITTED_CLEANUP_PENDING: "
                    + json.dumps(
                        {"output": str(final), "stage": str(staged_path)},
                        ensure_ascii=True, sort_keys=True,
                    ),
                    file=sys.stderr,
                )
            else:
                raise ProjectionError(
                    f"STAGING_CLEANUP_INCOMPLETE: unpublished stage may remain: "
                    f"{staged_path}"
                ) from cleanup_error


def main() -> int:
    args = _parser().parse_args()
    try:
        result = adapt_family_vector_to_next100_106(
            _load_authority_json(args.family_vector),
            expected_family_vector_identity_sha256=(
                args.expected_family_vector_identity_sha256
            ),
            dedup_authority=_load_authority_json(args.dedup_authority),
            expected_dedup_worker_id=args.expected_dedup_worker_id,
            expected_dedup_head_sha=args.expected_dedup_head_sha,
            expected_dedup_evidence_identity_sha256=(
                args.expected_dedup_evidence_identity_sha256
            ),
        )
        try:
            payload = (
                json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as exc:
            raise ProjectionError("adapter result cannot be encoded as strict UTF-8 JSON") from exc
        _write_new_output(
            args.output, payload,
            family_vector=args.family_vector,
            dedup_authority=args.dedup_authority,
        )
    except ProjectionError as exc:
        raise SystemExit(f"FAIL_CLOSED: {exc}") from exc
    # The report is already committed. A narrow Windows console code page must
    # not turn successful publication into an apparent CLI failure.
    try:
        print(args.output)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        escaped = str(args.output).encode(encoding, errors="backslashreplace").decode(
            encoding
        )
        print(escaped)
        print(
            "OUTPUT_COMMITTED_STDOUT_ENCODING_UNAVAILABLE: "
            + json.dumps({"output": str(args.output)}, ensure_ascii=True),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
