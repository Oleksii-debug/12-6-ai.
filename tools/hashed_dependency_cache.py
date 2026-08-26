#!/usr/bin/env python3
"""CLI for CI-164 exact-identity wheelhouse caching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.integration.hashed_dependency_cache import (
    DependencyCacheError,
    build_manifest,
    validate_manifest_files,
    verify_wheelhouse,
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DependencyCacheError(f"JSON must contain an object: {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    key = sub.add_parser("key", help="derive an exact cache key and manifest")
    key.add_argument("--root", type=Path, default=Path("."))
    key.add_argument("--profile-path", required=True)
    key.add_argument("--out", type=Path, required=True)
    key.add_argument("--allow-cuda-closure", action="store_true")

    validate = sub.add_parser("validate", help="re-hash profile/locks after cache restore")
    validate.add_argument("--root", type=Path, default=Path("."))
    validate.add_argument("--manifest", type=Path, required=True)

    verify = sub.add_parser("verify", help="verify cached wheel hashes against selected locks")
    verify.add_argument("--root", type=Path, default=Path("."))
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--wheelhouse", type=Path, required=True)
    verify.add_argument("--out", type=Path)

    args = parser.parse_args()
    if args.command == "key":
        manifest = build_manifest(
            args.root,
            args.profile_path,
            reject_accidental_cuda=not args.allow_cuda_closure,
        )
        destination = args.out if args.out.is_absolute() else args.root / args.out
        _write_json(destination, manifest)
        print(f"cache_key={manifest['cache_key']}")
        print(f"identity_sha256={manifest['identity_sha256']}")
        for record in manifest["component_locks"]:
            print(f"lock={record['path']} sha256={record['file_sha256']}")
        return 0

    manifest_path = args.manifest if args.manifest.is_absolute() else args.root / args.manifest
    manifest = _read_json(manifest_path)
    if args.command == "validate":
        validate_manifest_files(args.root, manifest)
        print("manifest_files=VERIFIED")
        return 0

    result = verify_wheelhouse(args.root, manifest, args.wheelhouse)
    if args.out is not None:
        destination = args.out if args.out.is_absolute() else args.root / args.out
        _write_json(destination, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
