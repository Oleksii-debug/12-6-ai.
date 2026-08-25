"""Text-only Windows product launcher for an installed 12-6 application wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROFILE_ID = "windows-x86_64"
STATUS_SCHEMA = "12-6.windows-product-status.v1"
EXIT_OK = 0
EXIT_USAGE = 2
EXIT_RUNTIME = 10
EXIT_CHECKPOINT = 20
_LOCK_LINE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*==[^\s;@/\\]+(?: --hash=sha256:[0-9a-f]{64})+$"
)
_NAME_NORMALIZER = re.compile(r"[-_.]+")


def _canonical_name(name: str) -> str:
    return _NAME_NORMALIZER.sub("-", name.strip()).lower()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lock_dir() -> Path:
    configured = os.environ.get("TWELVE_SIX_LOCK_DIR")
    if configured:
        return Path(configured)
    return Path(sys.executable).resolve().parent.parent / "12-6-lock"


def _load_profile(lock_dir: Path) -> dict[str, Any]:
    profile_path = lock_dir / "profile.json"
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot read installed D08 Windows lock profile") from exc
    if not isinstance(profile, dict):
        raise RuntimeError("installed D08 Windows lock profile must be an object")
    claimed = profile.get("manifest_sha256")
    payload = dict(profile)
    payload.pop("manifest_sha256", None)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise RuntimeError("installed D08 Windows lock profile has invalid self-hash")
    actual = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if actual != claimed:
        raise RuntimeError("installed D08 Windows lock profile self-hash mismatch")
    if profile.get("profile_id") != PROFILE_ID:
        raise RuntimeError("installed dependency profile is not windows-x86_64")
    return profile


def _runtime_versions(lock_path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = lock_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError("cannot read installed D08 runtime lock") from exc
    for number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _LOCK_LINE.fullmatch(line) is None:
            raise RuntimeError(f"installed runtime lock line {number} is not exact and hashed")
        requirement = line.split(" --hash=", 1)[0]
        name, version = requirement.split("==", 1)
        canonical = _canonical_name(name)
        if canonical in result:
            raise RuntimeError(f"duplicate installed runtime lock distribution: {canonical}")
        result[canonical] = version
    if not result:
        raise RuntimeError("installed D08 runtime lock is empty")
    return result


def _runtime_report() -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    lock_dir = _lock_dir()
    profile: dict[str, Any] | None = None
    versions: dict[str, str] = {}
    try:
        profile = _load_profile(lock_dir)
        python_record = profile.get("python")
        if not isinstance(python_record, dict):
            raise RuntimeError("installed D08 profile Python record is missing")
        expected_python = str(python_record.get("version", ""))
        if platform.system() != "Windows":
            errors.append(f"platform must be Windows; got {platform.system()}")
        machine = platform.machine().lower()
        if machine not in {"amd64", "x86_64"}:
            errors.append(f"machine must be x86_64/AMD64; got {platform.machine()}")
        actual_python = platform.python_version()
        if sys.implementation.name.lower() != "cpython" or actual_python != expected_python:
            errors.append(f"Python must be CPython {expected_python}; got {actual_python}")
        locks = profile.get("locks")
        if not isinstance(locks, dict) or not isinstance(locks.get("runtime"), dict):
            raise RuntimeError("installed D08 profile runtime lock record is missing")
        runtime_record = locks["runtime"]
        runtime_path = lock_dir / "runtime.lock.txt"
        expected_hash = runtime_record.get("sha256")
        if _sha256_file(runtime_path) != expected_hash:
            raise RuntimeError("installed D08 runtime lock SHA-256 mismatch")
        versions = _runtime_versions(runtime_path)
    except RuntimeError as exc:
        errors.append(str(exc))
        expected_python = "UNKNOWN"
        actual_python = platform.python_version()

    observed: dict[str, str | None] = {}
    for name, expected in sorted(versions.items()):
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        observed[name] = actual
        if actual != expected:
            errors.append(f"distribution {name} must be {expected}; got {actual or 'MISSING'}")

    imports: dict[str, str] = {}
    for module_name in ("numpy", "safetensors", "torch"):
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            imports[module_name] = f"ERROR:{type(exc).__name__}"
            errors.append(f"import {module_name} failed: {type(exc).__name__}")
        else:
            imports[module_name] = "PASS"

    try:
        app_version = importlib.metadata.version("twelve-six-ai")
    except importlib.metadata.PackageNotFoundError:
        app_version = None
        errors.append("application distribution twelve-six-ai is not installed")

    runtime: dict[str, Any] = {
        "profile_id": profile.get("profile_id") if profile else PROFILE_ID,
        "profile_manifest_sha256": profile.get("manifest_sha256") if profile else None,
        "lock_directory": str(lock_dir),
        "python_expected": expected_python,
        "python_actual": actual_python,
        "python_implementation": sys.implementation.name.lower(),
        "system": platform.system(),
        "machine": platform.machine(),
        "runtime_distributions": observed,
        "imports": imports,
        "application_version": app_version,
    }
    return runtime, errors


def _checkpoint_report(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"checkpoint does not exist: {path}"
    try:
        from twelve_six.checkpoint.core import verify_checkpoint

        manifest = verify_checkpoint(path)
        identity = manifest["identity"]
    except Exception as exc:
        return None, f"checkpoint verification failed: {type(exc).__name__}: {exc}"
    return (
        {
            "path": str(path),
            "checkpoint_id": manifest.get("checkpoint_id"),
            "git_sha": identity.get("git_sha"),
            "model_spec_hash": identity.get("model_spec_hash"),
            "step": identity.get("step"),
            "tokens_seen": identity.get("tokens_seen"),
        },
        None,
    )


def _print_status(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    print(f"status: {'READY' if payload['ready'] else 'ERROR'}")
    runtime = payload["runtime"]
    print(
        "runtime: "
        f"profile={runtime['profile_id']} python={runtime['python_actual']} "
        f"system={runtime['system']} machine={runtime['machine']}"
    )
    print(f"application: version={runtime['application_version']}")
    checkpoint = payload.get("checkpoint")
    if checkpoint is None:
        print("checkpoint: NOT_SELECTED")
    else:
        print(
            "checkpoint: "
            f"id={checkpoint['checkpoint_id']} step={checkpoint['step']} "
            f"tokens_seen={checkpoint['tokens_seen']}"
        )
    for message in payload["errors"]:
        print(f"error: {message}", file=sys.stderr)


def _status(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="12-6 status")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    runtime, errors = _runtime_report()
    checkpoint = None
    checkpoint_error = None
    if args.checkpoint is not None:
        checkpoint, checkpoint_error = _checkpoint_report(args.checkpoint)
        if checkpoint_error is not None:
            errors.append(checkpoint_error)
    payload = {
        "schema_version": STATUS_SCHEMA,
        "ready": not errors,
        "runtime": runtime,
        "checkpoint": checkpoint,
        "errors": errors,
    }
    _print_status(payload, as_json=args.json)
    if checkpoint_error is not None:
        return EXIT_CHECKPOINT
    return EXIT_OK if not errors else EXIT_RUNTIME


def _delegate(module: str, argv: list[str]) -> int:
    completed = subprocess.run([sys.executable, "-m", module, *argv], check=False)
    return int(completed.returncode)


def _usage() -> None:
    print("usage: 12-6 status [--checkpoint PATH] [--json]", file=sys.stderr)
    print("       12-6 generate --checkpoint PATH [generation options]", file=sys.stderr)
    print("       12-6 serve --checkpoint PATH [server options]", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        _usage()
        return EXIT_OK if arguments else EXIT_USAGE
    command, *rest = arguments
    if command == "status":
        return _status(rest)
    if command == "generate":
        return _delegate("twelve_six.inference.cli", rest)
    if command == "serve":
        return _delegate("twelve_six.inference.server", rest)
    print(f"error: unknown command {command!r}", file=sys.stderr)
    _usage()
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
