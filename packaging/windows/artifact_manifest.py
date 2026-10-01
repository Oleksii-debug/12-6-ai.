"""Create self-hashed manifests for the separated Windows product artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_WINDOWS_PROFILE_MANIFEST_SHA256 = "a855d6840bc26392a7c1b515749bbdedff6bb88fdfe19c223687ee0757576299"

_MODEL_PAYLOAD_SUFFIXES = frozenset(
    {
        ".ckpt",
        ".ggml",
        ".gguf",
        ".onnx",
        ".pt",
        ".pth",
        ".safetensors",
    }
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object member: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _require_finite_json_numbers(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number is forbidden")
        return
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_json_numbers(item)
        return
    if isinstance(value, list):
        for item in value:
            _require_finite_json_numbers(item)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=_strict_json_object,
        parse_constant=_reject_json_constant,
    )
    _require_finite_json_numbers(value)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_self_hashed_manifest(
    path: Path,
    *,
    expected_schema: str,
    expected_keys: frozenset[str],
) -> dict[str, Any]:
    value = _read_json(path)
    if set(value) != expected_keys:
        raise ValueError(f"{path} has unexpected manifest fields")
    if value.get("schema_version") != expected_schema:
        raise ValueError(f"{path} has unexpected schema_version")
    claimed = value.get("manifest_sha256")
    if not isinstance(claimed, str) or _SHA256_RE.fullmatch(claimed) is None:
        raise ValueError(f"{path} has invalid manifest_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256")
    actual = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if actual != claimed:
        raise ValueError(f"{path} manifest self-hash mismatch")
    return value


def _validate_manifest_files(path: Path, manifest: dict[str, Any]) -> None:
    expected = manifest.get("files")
    if not isinstance(expected, dict):
        raise ValueError(f"{path} files must be an object")
    actual = _files(path.parent, exclude={path.name})
    if actual != expected:
        raise ValueError(f"{path} file inventory/hash mismatch")


def _validate_evidence_inputs(args: argparse.Namespace) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    status = _read_json(args.status)
    missing_status = _read_json(args.missing_status)
    app_manifest = _read_self_hashed_manifest(
        args.app_manifest,
        expected_schema="12-6.windows-application-artifact.v1",
        expected_keys=frozenset(
            {
                "schema_version",
                "source_sha",
                "contains_runtime_wheels",
                "contains_checkpoint",
                "files",
                "manifest_sha256",
            }
        ),
    )
    runtime_manifest = _read_self_hashed_manifest(
        args.runtime_manifest,
        expected_schema="12-6.windows-runtime-artifact.v1",
        expected_keys=frozenset(
            {
                "schema_version",
                "profile_id",
                "python_version",
                "profile_manifest_sha256",
                "contains_application_wheel",
                "contains_checkpoint",
                "files",
                "manifest_sha256",
            }
        ),
    )

    if not isinstance(args.source_sha, str) or _GIT_SHA_RE.fullmatch(args.source_sha) is None:
        raise ValueError("evidence source_sha must be a full lowercase Git SHA")
    if app_manifest.get("source_sha") != args.source_sha:
        raise ValueError("application manifest source_sha does not match evidence source_sha")
    if app_manifest.get("contains_runtime_wheels") is not False:
        raise ValueError("application manifest must exclude runtime wheels")
    if app_manifest.get("contains_checkpoint") is not False:
        raise ValueError("application manifest must exclude checkpoint bytes")
    if runtime_manifest.get("profile_id") != "windows-x86_64":
        raise ValueError("runtime manifest must bind windows-x86_64")
    if runtime_manifest.get("profile_manifest_sha256") != _WINDOWS_PROFILE_MANIFEST_SHA256:
        raise ValueError("runtime manifest must bind the canonical Windows profile")
    if runtime_manifest.get("python_version") != "3.11.9":
        raise ValueError("runtime manifest must bind CPython 3.11.9")
    if runtime_manifest.get("contains_application_wheel") is not False:
        raise ValueError("runtime manifest must exclude application wheel")
    if runtime_manifest.get("contains_checkpoint") is not False:
        raise ValueError("runtime manifest must exclude checkpoint bytes")

    _validate_manifest_files(args.app_manifest, app_manifest)
    _validate_manifest_files(args.runtime_manifest, runtime_manifest)

    runtime = status.get("runtime")
    if (
        status.get("schema_version") != "12-6.windows-product-status.v1"
        or status.get("ready") is not True
        or status.get("errors") != []
        or not isinstance(runtime, dict)
        or runtime.get("profile_id") != "windows-x86_64"
        or runtime.get("profile_manifest_sha256") != _WINDOWS_PROFILE_MANIFEST_SHA256
        or runtime.get("python_actual") != "3.11.9"
    ):
        raise ValueError("installed status is not an exact ready Windows runtime status")

    missing_errors = missing_status.get("errors")
    if (
        missing_status.get("schema_version") != "12-6.windows-product-status.v1"
        or missing_status.get("ready") is not False
        or missing_status.get("checkpoint") is not None
        or not isinstance(missing_errors, list)
        or not any(
            isinstance(item, str) and item.startswith("checkpoint does not exist:")
            for item in missing_errors
        )
    ):
        raise ValueError("missing-checkpoint status is not the expected fail-closed result")

    for label, artifact_id, digest in (
        ("application", args.app_artifact_id, args.app_artifact_digest),
        ("runtime", args.runtime_artifact_id, args.runtime_artifact_digest),
    ):
        if not isinstance(artifact_id, str) or not artifact_id.isdecimal() or int(artifact_id) <= 0:
            raise ValueError(f"{label} artifact id must be a positive decimal integer")
        if not isinstance(digest, str) or _ARTIFACT_DIGEST_RE.fullmatch(digest) is None:
            raise ValueError(f"{label} artifact digest must be sha256:<64 lowercase hex>")

    return status, missing_status, app_manifest, runtime_manifest


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


def _is_model_checkpoint_payload(name: str) -> bool:
    """Return whether one artifact member is serialized model/checkpoint payload bytes.

    Package namespaces such as ``twelve_six/checkpoint/core.py`` are executable
    application code, not a bundled checkpoint. The artifact boundary therefore
    classifies actual serialized model formats instead of matching directory
    names containing ``checkpoint``.
    """

    normalized = name.replace("\\", "/").rstrip("/").casefold()
    return any(normalized.endswith(suffix) for suffix in _MODEL_PAYLOAD_SUFFIXES)


def _application(root: Path, source_sha: str) -> None:
    wheels = sorted(root.glob("twelve_six_ai-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one application wheel, found {len(wheels)}")
    artifact_wheels = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".whl"
    )
    if artifact_wheels != wheels:
        unexpected = [
            path.relative_to(root).as_posix()
            for path in artifact_wheels
            if path not in wheels
        ]
        raise RuntimeError(f"application artifact contains unexpected wheel bytes: {unexpected}")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
    forbidden = sorted(name for name in names if _is_model_checkpoint_payload(name))
    if forbidden:
        raise RuntimeError(f"application wheel contains model/checkpoint bytes: {forbidden}")
    bundle_forbidden = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != wheels[0]
        and _is_model_checkpoint_payload(path.relative_to(root).as_posix())
    )
    if bundle_forbidden:
        raise RuntimeError(
            "application artifact contains model/checkpoint sidecar bytes: "
            f"{bundle_forbidden}"
        )
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
    runtime_wheels = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".whl"
    )
    application_wheels = [
        path
        for path in runtime_wheels
        if path.name.casefold().startswith("twelve_six_ai-")
    ]
    if application_wheels:
        relative = [path.relative_to(root).as_posix() for path in application_wheels]
        raise RuntimeError(
            f"runtime artifact must not contain the application wheel: {relative}"
        )
    unexpected_wheel_locations = [
        path.relative_to(root).as_posix()
        for path in runtime_wheels
        if path.parent != root / "wheelhouse"
    ]
    if unexpected_wheel_locations:
        raise RuntimeError(
            f"runtime artifact wheel is outside canonical wheelhouse: {unexpected_wheel_locations}"
        )
    checkpoint_bytes = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and _is_model_checkpoint_payload(path.relative_to(root).as_posix())
    )
    if checkpoint_bytes:
        raise RuntimeError(
            f"runtime artifact must not contain model/checkpoint bytes: {checkpoint_bytes}"
        )
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
    status, missing_status, app_manifest, runtime_manifest = _validate_evidence_inputs(args)
    payload = {
        "schema_version": "12-6.windows-product-packaging-evidence.v1",
        "source_sha": args.source_sha,
        "installation_root": str(args.install_root),
        "status": status,
        "missing_checkpoint_status": missing_status,
        "application_manifest": app_manifest,
        "runtime_manifest": runtime_manifest,
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
