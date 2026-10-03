"""Adapt an independently identified D03 family vector for NEXT100-106."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import BinaryIO

from twelve_six.data.postdecontam_balance_projection_v1 import (
    ProjectionError,
)
from twelve_six.data.postmaterialization_balance_projection_v1 import load_strict_json_object
from twelve_six.data.postdecontam_next100_adapter_v1 import (
    adapt_family_vector_to_next100_106,
)


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


def _load_authority_json(path: Path) -> dict[str, object]:
    """Read a bounded, unambiguous source before verifying its semantic identity."""
    try:
        with path.open("rb") as source:
            raw = source.read(MAX_AUTHORITY_JSON_BYTES + 1)
    except OSError as exc:
        raise ProjectionError(f"cannot read adapter authority: {path}") from exc
    if len(raw) > MAX_AUTHORITY_JSON_BYTES:
        raise ProjectionError("adapter authority JSON exceeds byte limit")

    value = load_strict_json_object(raw, label=f"adapter authority {path}")
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


def _write_new_output(
    path: Path, payload: bytes, *, family_vector: Path, dedup_authority: Path,
) -> None:
    """Stage completely; atomically publish a new name without replacing an incumbent."""
    if path.exists() or path.is_symlink():
        raise ProjectionError(f"refusing to overwrite existing adapter output: {path}")
    try:
        resolved = path.resolve()
        protected = {family_vector.resolve(), dedup_authority.resolve()}
    except (OSError, RuntimeError) as exc:
        raise ProjectionError("adapter output path cannot be resolved safely") from exc
    if resolved in protected:
        raise ProjectionError("adapter output must not alias an input authority")
    staged_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as destination:
            staged_path = Path(destination.name)
            _write_staged_bytes(destination, payload)
        # os.link is create-only, unlike os.replace / POSIX os.rename.
        # Same-directory staging avoids cross-volume publication.
        os.link(staged_path, path)
    except FileExistsError as exc:
        raise ProjectionError(f"refusing to overwrite existing adapter output: {path}") from exc
    except OSError as exc:
        raise ProjectionError(f"cannot publish adapter output safely: {path}: {exc}") from exc
    finally:
        if staged_path is not None:
            try:
                staged_path.unlink(missing_ok=True)
            except OSError as exc:
                raise ProjectionError(
                    f"cannot clean up staged adapter output: {staged_path}"
                ) from exc


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
        payload = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        _write_new_output(
            args.output, payload.encode("utf-8"),
            family_vector=args.family_vector,
            dedup_authority=args.dedup_authority,
        )
    except ProjectionError as exc:
        raise SystemExit(f"FAIL_CLOSED: {exc}") from exc
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
