"""Bootstrap exact purpose-specific dependency locks from pip reports.

This is resolver provenance only. Generated files become authoritative only after they
are committed and the exact-head clean-install verifier succeeds without resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

PROFILE_SCHEMA = "12-6.purpose-environment-profile.v1"
INDEX_SCHEMA = "12-6.purpose-environment-index.v1"
SPECS_SCHEMA = "12-6.purpose-environment-specs.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOCK_LINE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*==([^\s;@/\\]+)(?: --hash=sha256:[0-9a-f]{64})+$"
)
_NAME_NORMALIZER = re.compile(r"[-_.]+")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_name(name: str) -> str:
    value = _NAME_NORMALIZER.sub("-", name.strip()).lower()
    if not value:
        raise ValueError("distribution name must not be empty")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _read_lock(path: Path) -> dict[str, tuple[str, tuple[str, ...]]]:
    packages: dict[str, tuple[str, tuple[str, ...]]] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LOCK_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid exact lock line {path}:{number}")
        name, tail = line.split("==", 1)
        version = tail.split(" ", 1)[0]
        hashes = tuple(sorted(part.removeprefix("--hash=sha256:") for part in tail.split()[1:]))
        canonical = _canonical_name(name)
        if canonical in packages:
            raise ValueError(f"duplicate distribution {canonical} in {path}")
        packages[canonical] = (version, hashes)
    return packages


def _write_constraints(path: Path, packages: dict[str, tuple[str, tuple[str, ...]]]) -> None:
    lines = [f"{name}=={version}" for name, (version, _hashes) in sorted(packages.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_report(requirements: list[str], *, constraints: Path | None = None) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        report_path = Path(handle.name)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--dry-run",
        "--ignore-installed",
        "--report",
        str(report_path),
    ]
    if constraints is not None:
        command.extend(["--constraint", str(constraints)])
    command.extend(requirements)
    try:
        subprocess.run(command, check=True)
        report = _load_json(report_path)
    finally:
        report_path.unlink(missing_ok=True)
    return report


def _packages(report: dict[str, Any]) -> dict[str, tuple[str, tuple[str, ...]]]:
    installs = report.get("install")
    if not isinstance(installs, list):
        raise TypeError("pip report is missing install list")
    result: dict[str, tuple[str, tuple[str, ...]]] = {}
    for item in installs:
        if not isinstance(item, dict):
            raise TypeError("pip report install entry must be an object")
        metadata = item.get("metadata")
        download = item.get("download_info")
        if not isinstance(metadata, dict) or not isinstance(download, dict):
            raise TypeError("pip report entry lacks metadata/download_info")
        name = _canonical_name(str(metadata.get("name", "")))
        version = str(metadata.get("version", "")).strip()
        archive = download.get("archive_info")
        if not version or not isinstance(archive, dict):
            raise RuntimeError(f"pip report lacks exact artifact evidence for {name}")
        hashes = archive.get("hashes")
        if not isinstance(hashes, dict) or not hashes.get("sha256"):
            raise RuntimeError(f"pip report lacks SHA-256 for {name}=={version}")
        raw_hashes = hashes["sha256"]
        if isinstance(raw_hashes, str):
            values = (raw_hashes,)
        elif isinstance(raw_hashes, list):
            values = tuple(str(value) for value in raw_hashes)
        else:
            raise TypeError(f"pip report SHA-256 is malformed for {name}")
        normalized = tuple(sorted(set(values)))
        if not normalized or any(_SHA256.fullmatch(value) is None for value in normalized):
            raise RuntimeError(f"pip report has invalid SHA-256 evidence for {name}")
        current = (version, normalized)
        previous = result.get(name)
        if previous is not None and previous != current:
            raise RuntimeError(f"conflicting resolver evidence for {name}")
        result[name] = current
    return result


def _write_lock(
    path: Path,
    packages: dict[str, tuple[str, tuple[str, ...]]],
    *,
    profile_id: str,
    group: str,
) -> None:
    lines = [
        f"# 12-6 AI {profile_id} {group}; resolver bootstrap only until committed CI passes.",
        "# Authority install: python -m pip install --require-hashes --no-deps -r <this-file>",
    ]
    for name, (version, hashes) in sorted(packages.items()):
        line = f"{name}=={version}"
        for digest in hashes:
            line += f" --hash=sha256:{digest}"
        lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _canonical_lock(repo_root: Path) -> tuple[dict[str, Any], str]:
    path = repo_root / "requirements/locks/index.json"
    index = _load_json(path)
    claimed = index.get("index_sha256")
    payload = dict(index)
    payload.pop("index_sha256", None)
    if _SHA256.fullmatch(str(claimed or "")) is None:
        raise ValueError("canonical lock index has invalid semantic hash")
    if _sha256_bytes(_canonical_bytes(payload)) != claimed:
        raise ValueError("canonical lock index semantic hash mismatch")
    return index, _sha256_file(path)


def _base_profile_ref(repo_root: Path, index: dict[str, Any], profile_id: str) -> dict[str, str]:
    record = index.get("profiles", {}).get(profile_id)
    if not isinstance(record, dict):
        raise KeyError(f"canonical base profile {profile_id!r} missing")
    relative = str(record["path"])
    path = repo_root / relative
    if _sha256_file(path) != record.get("sha256"):
        raise ValueError("canonical base profile file hash mismatch")
    manifest = _load_json(path)
    if manifest.get("manifest_sha256") != record.get("manifest_sha256"):
        raise ValueError("canonical base profile semantic hash mismatch")
    return {
        "profile_id": profile_id,
        "path": relative,
        "file_sha256": str(record["sha256"]),
        "manifest_sha256": str(record["manifest_sha256"]),
    }


def _canonical_ref(repo_root: Path, index: dict[str, Any], index_file_sha: str) -> dict[str, str]:
    return {
        "path": "requirements/locks/index.json",
        "file_sha256": index_file_sha,
        "index_sha256": str(index["index_sha256"]),
    }


def _profile_hash(payload: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(payload))


def _write_profile(path: Path, payload: dict[str, Any]) -> None:
    profile = dict(payload)
    profile["profile_sha256"] = _profile_hash(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _lock_record(relative: Path, physical: Path, count: int) -> dict[str, Any]:
    return {
        "path": relative.as_posix(),
        "sha256": _sha256_file(physical),
        "package_count": count,
    }


def _linux_profiles(repo_root: Path, output_root: Path, specs: dict[str, Any]) -> None:
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("linux profile bootstrap requires Linux x86_64")
    if platform.python_version() != "3.11.16":
        raise RuntimeError(f"linux profile bootstrap requires CPython 3.11.16, got {platform.python_version()}")

    index, index_file_sha = _canonical_lock(repo_root)
    canonical_ref = _canonical_ref(repo_root, index, index_file_sha)
    base_ref = _base_profile_ref(repo_root, index, "linux-x86_64")
    base_runtime = _read_lock(repo_root / "requirements/locks/linux-x86_64/runtime.lock.txt")

    with tempfile.TemporaryDirectory(prefix="twelve-six-purpose-constraints-") as temp_name:
        constraints = Path(temp_name) / "base-runtime.txt"
        _write_constraints(constraints, base_runtime)

        for profile_id, spec in sorted(specs["profiles"].items()):
            kind = spec.get("kind")
            if kind == "linux-overlay":
                report = _run_report([str(item) for item in spec["requirements"]], constraints=constraints)
                resolved = _packages(report)
                overlay: dict[str, tuple[str, tuple[str, ...]]] = {}
                reused_base: dict[str, str] = {}
                for name, record in resolved.items():
                    base = base_runtime.get(name)
                    if base is None:
                        overlay[name] = record
                        continue
                    if base[0] != record[0]:
                        raise RuntimeError(
                            f"{profile_id} would drift canonical base {name}: {base[0]} -> {record[0]}"
                        )
                    reused_base[name] = record[0]

                relative_lock = Path("requirements/profiles") / profile_id / "overlay.lock.txt"
                physical_lock = output_root / relative_lock
                _write_lock(
                    physical_lock,
                    overlay,
                    profile_id=profile_id,
                    group="overlay",
                )
                profile = {
                    "schema_version": PROFILE_SCHEMA,
                    "profile_id": profile_id,
                    "kind": kind,
                    "purpose": spec["purpose"],
                    "python": {"implementation": "cpython", "version": "3.11.16"},
                    "platform": {"system": "Linux", "machine": "x86_64"},
                    "canonical_lock": canonical_ref,
                    "base_profile": base_ref,
                    "direct_requirements": list(spec["requirements"]),
                    "reused_base_distributions": dict(sorted(reused_base.items())),
                    "locks": {
                        "overlay": _lock_record(relative_lock, physical_lock, len(overlay))
                    },
                }
                _write_profile(
                    output_root / "requirements/profiles" / profile_id / "profile.json",
                    profile,
                )
            elif kind == "linux-base-role":
                profile = {
                    "schema_version": PROFILE_SCHEMA,
                    "profile_id": profile_id,
                    "kind": kind,
                    "purpose": spec["purpose"],
                    "python": {"implementation": "cpython", "version": "3.11.16"},
                    "platform": {"system": "Linux", "machine": "x86_64"},
                    "canonical_lock": canonical_ref,
                    "base_profile": base_ref,
                    "direct_requirements": [],
                    "locks": {},
                    "runtime_expectations": spec.get("runtime_expectations", {}),
                }
                _write_profile(
                    output_root / "requirements/profiles" / profile_id / "profile.json",
                    profile,
                )


def _windows_profile(repo_root: Path, output_root: Path, specs: dict[str, Any]) -> None:
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("windows profile bootstrap requires Windows x86_64")
    if platform.python_version() != "3.11.9":
        raise RuntimeError(
            f"windows profile bootstrap requires CPython 3.11.9, got {platform.python_version()}"
        )

    index, index_file_sha = _canonical_lock(repo_root)
    canonical_ref = _canonical_ref(repo_root, index, index_file_sha)
    linux_base_ref = _base_profile_ref(repo_root, index, "linux-x86_64")
    linux_runtime = _read_lock(repo_root / "requirements/locks/linux-x86_64/runtime.lock.txt")
    linux_toolchain = _read_lock(repo_root / "requirements/locks/linux-x86_64/toolchain.lock.txt")

    profile_id = "windows-x86_64-runtime"
    spec = specs["profiles"][profile_id]
    direct_names = [_canonical_name(str(name)) for name in spec["direct_versions_from_linux_base"]]
    direct_requirements: list[str] = []
    direct_versions: dict[str, str] = {}
    for name in direct_names:
        if name not in linux_runtime:
            raise KeyError(f"{name} missing from canonical Linux runtime lock")
        version = linux_runtime[name][0]
        direct_versions[name] = version
        direct_requirements.append(f"{name}=={version}")

    runtime = _packages(_run_report(direct_requirements))
    toolchain_requirements = [
        f"{name}=={version}" for name, (version, _hashes) in sorted(linux_toolchain.items())
    ]
    toolchain = _packages(_run_report(toolchain_requirements))

    runtime_relative = Path("requirements/profiles") / profile_id / "runtime.lock.txt"
    toolchain_relative = Path("requirements/profiles") / profile_id / "toolchain.lock.txt"
    runtime_physical = output_root / runtime_relative
    toolchain_physical = output_root / toolchain_relative
    _write_lock(runtime_physical, runtime, profile_id=profile_id, group="runtime")
    _write_lock(toolchain_physical, toolchain, profile_id=profile_id, group="toolchain")

    profile = {
        "schema_version": PROFILE_SCHEMA,
        "profile_id": profile_id,
        "kind": "windows-runtime",
        "purpose": spec["purpose"],
        "python": {"implementation": "cpython", "version": "3.11.9"},
        "platform": {"system": "Windows", "machine": "x86_64"},
        "canonical_lock": canonical_ref,
        "version_source_profile": linux_base_ref,
        "direct_requirements": direct_requirements,
        "direct_versions": dict(sorted(direct_versions.items())),
        "locks": {
            "runtime": _lock_record(runtime_relative, runtime_physical, len(runtime)),
            "toolchain": _lock_record(toolchain_relative, toolchain_physical, len(toolchain)),
        },
    }
    _write_profile(
        output_root / "requirements/profiles" / profile_id / "profile.json",
        profile,
    )


def _build_index(repo_root: Path, profile_root: Path) -> None:
    canonical_index, canonical_file_sha = _canonical_lock(repo_root)
    profiles: dict[str, dict[str, str]] = {}
    for profile_path in sorted(profile_root.glob("*/profile.json")):
        profile = _load_json(profile_path)
        claimed = profile.get("profile_sha256")
        payload = dict(profile)
        payload.pop("profile_sha256", None)
        if profile.get("schema_version") != PROFILE_SCHEMA:
            raise ValueError(f"unsupported profile schema: {profile_path}")
        if _SHA256.fullmatch(str(claimed or "")) is None or _profile_hash(payload) != claimed:
            raise ValueError(f"purpose profile self-hash mismatch: {profile_path}")
        profile_id = str(profile["profile_id"])
        if profile_path.parent.name != profile_id:
            raise ValueError("purpose profile directory/id mismatch")
        profiles[profile_id] = {
            "path": f"requirements/profiles/{profile_id}/profile.json",
            "sha256": _sha256_file(profile_path),
            "profile_sha256": str(claimed),
        }
    expected = set(_load_json(repo_root / "requirements/profiles/specs.json")["profiles"])
    if set(profiles) != expected:
        raise ValueError(f"purpose profile set mismatch: expected={sorted(expected)} actual={sorted(profiles)}")
    payload: dict[str, Any] = {
        "schema_version": INDEX_SCHEMA,
        "canonical_lock": _canonical_ref(repo_root, canonical_index, canonical_file_sha),
        "profiles": profiles,
    }
    payload["index_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    (profile_root / "index.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("linux", "windows", "index"), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("generated"))
    parser.add_argument("--profile-root", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    specs = _load_json(repo_root / "requirements/profiles/specs.json")
    if specs.get("schema_version") != SPECS_SCHEMA:
        raise ValueError("unsupported purpose environment specs schema")

    if args.mode == "linux":
        _linux_profiles(repo_root, args.output_root.resolve(), specs)
    elif args.mode == "windows":
        _windows_profile(repo_root, args.output_root.resolve(), specs)
    else:
        if args.profile_root is None:
            parser.error("--profile-root is required for --mode index")
        _build_index(repo_root, args.profile_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
