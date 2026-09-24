#!/usr/bin/env python3
"""Machine-readable operator for the learned-20M global Git-ref lease."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.learned20m_global_training_lease import (
    acquire_global_training_run_lease,
    inspect_global_training_run_lease,
    renew_global_training_run_lease,
    terminate_global_training_run_lease,
)
from twelve_six.learned20m_training_lease import build_training_run_lease


class _DuplicateKey(ValueError):
    pass


def _pairs_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non_finite_json_constant:{value}")


def _load_mapping(path: Path) -> Mapping[str, Any]:
    raw = path.read_text(encoding="utf-8")
    value = json.loads(
        raw,
        object_pairs_hook=_pairs_without_duplicates,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, Mapping):
        raise ValueError("manifest_not_object")
    return value


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--manifest", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    subparsers.add_parser("inspect")

    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--run-id", required=True)
    acquire.add_argument("--holder-id", required=True)
    acquire.add_argument("--ttl-seconds", type=int, required=True)

    renew = subparsers.add_parser("renew")
    renew.add_argument("--expected-remote-tip", required=True)
    renew.add_argument("--ttl-seconds", type=int, required=True)

    terminate = subparsers.add_parser("terminate")
    terminate.add_argument("--expected-remote-tip", required=True)
    terminate.add_argument(
        "--status",
        choices=("COMPLETED", "FAILED", "ABORTED"),
        required=True,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        manifest = _load_mapping(args.manifest)
        if args.operation == "inspect":
            result = inspect_global_training_run_lease(
                args.repo_root, args.remote, manifest
            )
            _emit(result.as_dict())
            return 0 if result.present and result.valid else 3
        if args.operation == "acquire":
            lease = build_training_run_lease(
                manifest,
                run_id=args.run_id,
                holder_id=args.holder_id,
                ttl_seconds=args.ttl_seconds,
            )
            result = acquire_global_training_run_lease(
                args.repo_root, args.remote, manifest, lease.as_dict()
            )
        elif args.operation == "renew":
            result = renew_global_training_run_lease(
                args.repo_root,
                args.remote,
                manifest,
                expected_remote_tip=args.expected_remote_tip,
                ttl_seconds=args.ttl_seconds,
            )
        else:
            result = terminate_global_training_run_lease(
                args.repo_root,
                args.remote,
                manifest,
                expected_remote_tip=args.expected_remote_tip,
                status=args.status,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _emit({"operation": args.operation, "ok": False, "blockers": [str(exc)]})
        return 2
    _emit(result.as_dict())
    return 0 if result.committed and result.post_write_reread_verified else 3


if __name__ == "__main__":
    raise SystemExit(main())
