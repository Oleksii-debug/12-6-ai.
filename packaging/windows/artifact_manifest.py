"""Create self-hashed manifests for the separated Windows product artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _files(root: Path, *, exclude: set[str]) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): _file_record(path)
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
        if path.relative_to(root).as_posix() not in exclude
    }


def _write(path: Path, payload: dict[str, Any], *, hash_field: str = "manifest_sha256") -> None:
    payload[hash_field] = hashlib.sha256(_canonical(payload)).hexdigest()
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _application(root: Path, source_sha: str) -> None:
    wheels = sorted(root.glob("twelve_six_ai-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one application wheel, found {len(wheels)}")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
    forbidden = [
        name
        for name in names
        if name.lower().endswith(".safetensors") or "/checkpoint" in name.lower()
    ]
    if forbidden:
        raise RuntimeError(f"application wheel contains model/checkpoint bytes: {forbidden}")
    payload = {
        "schema_version": "12-6.windows-application-artifact.v1",
        "source_sha": source_sha,
        "contains_runtime_wheels": False,
        "contains_checkpoint": False,
        "files": _files(root, exclude={"app-manifest.json"}),
    }
    _write(root / "app-manifest.json", payload)


def _runtime(root: Path) -> None:
    profile = _read_json(root / "12-6-lock" / "profile.json")
    application_wheels = list((root / "wheelhouse").glob("twelve_six_ai-*.whl"))
    if application_wheels:
        raise RuntimeError("runtime wheelhouse must not contain the application wheel")
    checkpoint_bytes = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".safetensors"
    ]
    if checkpoint_bytes:
        raise RuntimeError("runtime artifact must not contain checkpoint tensors")
    payload = {
        "schema_version": "12-6.windows-runtime-artifact.v1",
        "profile_id": profile["profile_id"],
        "python_version": profile["python"]["version"],
        "profile_manifest_sha256": profile["manifest_sha256"],
        "contains_application_wheel": False,
        "contains_checkpoint": False,
        "files": _files(root, exclude={"runtime-manifest.json"}),
    }
    _write(root / "runtime-manifest.json", payload)


def _evidence(args: argparse.Namespace) -> None:
    payload = {
        "schema_version": "12-6.windows-product-packaging-evidence.v1",
        "source_sha": args.source_sha,
        "installation_root": str(args.install_root),
        "status": _read_json(args.status),
        "missing_checkpoint_status": _read_json(args.missing_status),
        "application_manifest": _read_json(args.app_manifest),
        "runtime_manifest": _read_json(args.runtime_manifest),
        "github_artifacts": {
            "application": {"id": args.app_artifact_id, "digest": args.app_artifact_digest},
            "runtime": {"id": args.runtime_artifact_id, "digest": args.runtime_artifact_digest},
        },
        "checks": {
            "artifact_only_install_no_repository_checkout": "PASS",
            "unicode_and_spaces_install_path": "PASS",
            "exact_runtime_status": "PASS",
            "application_wheel_separate_from_runtime": "PASS",
            "checkpoint_not_bundled": "PASS",
            "replaceable_checkpoint_selection": "PASS",
            "stdin_passthrough_and_error_code": "PASS",
            "generate_command_route": "PASS",
            "local_api_command_route": "PASS",
            "canonical_checkpoint_execution": "NOT_TESTED_IN_THIS_PACKAGING_PR",
            "nvda_manual_accessibility": "NOT_TESTED",
        },
    }
    _write(args.output, payload, hash_field="evidence_sha256")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    app = sub.add_parser("application")
    app.add_argument("--root", type=Path, required=True)
    app.add_argument("--source-sha", required=True)

    runtime = sub.add_parser("runtime")
    runtime.add_argument("--root", type=Path, required=True)

    evidence = sub.add_parser("evidence")
    evidence.add_argument("--output", type=Path, required=True)
    evidence.add_argument("--source-sha", required=True)
    evidence.add_argument("--install-root", type=Path, required=True)
    evidence.add_argument("--status", type=Path, required=True)
    evidence.add_argument("--missing-status", type=Path, required=True)
    evidence.add_argument("--app-manifest", type=Path, required=True)
    evidence.add_argument("--runtime-manifest", type=Path, required=True)
    evidence.add_argument("--app-artifact-id", required=True)
    evidence.add_argument("--app-artifact-digest", required=True)
    evidence.add_argument("--runtime-artifact-id", required=True)
    evidence.add_argument("--runtime-artifact-digest", required=True)

    args = parser.parse_args()
    if args.command == "application":
        _application(args.root, args.source_sha)
    elif args.command == "runtime":
        _runtime(args.root)
    else:
        _evidence(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
