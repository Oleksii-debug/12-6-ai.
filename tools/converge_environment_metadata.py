"""Deterministically derive D08 aggregate/profile metadata from exact lock bytes.

This tool never resolves or edits dependency lock contents. It only derives the
metadata that binds pyproject.toml, canonical component locks, purpose overlays,
and the aggregate indices. ``--check`` is fail-closed; ``--write`` is the single
repair/migration path after an intentional lock or project-metadata change.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCK_MODULE = ROOT / "src" / "twelve_six" / "integration" / "dependency_lock.py"
PROFILE_SCHEMA = "12-6.purpose-environment-profile.v1"
INDEX_SCHEMA = "12-6.purpose-environment-index.v1"
SPECS_SCHEMA = "12-6.purpose-environment-specs.v1"
_LOCK_LINE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*==[^\s;@/\\]+(?: --hash=sha256:[0-9a-f]{64})+$"
)


def _load_contract() -> Any:
    spec = importlib.util.spec_from_file_location("twelve_six_dependency_lock_env154", LOCK_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load dependency lock contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOCK = _load_contract()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _count_exact_lock(path: Path) -> int:
    seen: set[str] = set()
    count = 0
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _LOCK_LINE.fullmatch(line) is None:
            raise ValueError(f"non-exact or unhashed lock line {path}:{number}")
        name = LOCK.canonical_distribution_name(line.split("==", 1)[0])
        if name in seen:
            raise ValueError(f"duplicate distribution {name} in {path}")
        seen.add(name)
        count += 1
    return count


def _base_platform(profile_id: str) -> tuple[str, str]:
    if profile_id == "linux-x86_64":
        return "Linux", "x86_64"
    if profile_id == "linux-aarch64":
        return "Linux", "aarch64"
    raise ValueError(f"no deterministic platform mapping for {profile_id}")


def _base_profile(root: Path, profile_id: str) -> dict[str, Any]:
    lock_files = {
        group: Path("requirements/locks") / profile_id / f"{group}.lock.txt"
        for group in ("toolchain", "runtime", "dev")
    }
    counts = {group: _count_exact_lock(root / relative) for group, relative in lock_files.items()}
    system, machine = _base_platform(profile_id)
    return LOCK.build_profile_manifest(
        root=root,
        profile_id=profile_id,
        lock_files=lock_files,
        package_counts=counts,
        platform_system=system,
        platform_machine=machine,
    )


def _base_index(base_profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    profiles: dict[str, dict[str, str]] = {}
    for profile_id, profile in sorted(base_profiles.items()):
        relative = f"requirements/locks/{profile_id}/profile.json"
        profiles[profile_id] = {
            "path": relative,
            "sha256": _sha_bytes(_pretty(profile)),
            "manifest_sha256": str(profile["manifest_sha256"]),
        }
    payload: dict[str, Any] = {
        "schema_version": LOCK.INDEX_SCHEMA_VERSION,
        "project": LOCK.PROJECT_DISTRIBUTION,
        "python_version": LOCK.EXACT_PYTHON_VERSION,
        "profiles": profiles,
    }
    payload["index_sha256"] = _sha_bytes(_canonical(payload))
    return payload


def _canonical_ref(index: dict[str, Any]) -> dict[str, str]:
    return {
        "path": "requirements/locks/index.json",
        "file_sha256": _sha_bytes(_pretty(index)),
        "index_sha256": str(index["index_sha256"]),
    }


def _base_ref(index: dict[str, Any], profile_id: str) -> dict[str, str]:
    record = index["profiles"][profile_id]
    return {
        "profile_id": profile_id,
        "path": str(record["path"]),
        "file_sha256": str(record["sha256"]),
        "manifest_sha256": str(record["manifest_sha256"]),
    }


def _purpose_profile(
    root: Path,
    profile_id: str,
    spec: dict[str, Any],
    canonical_index: dict[str, Any],
) -> dict[str, Any]:
    path = root / "requirements/profiles" / profile_id / "profile.json"
    current = _load_json(path)
    if current.get("schema_version") != PROFILE_SCHEMA or current.get("profile_id") != profile_id:
        raise ValueError(f"purpose profile identity/schema mismatch: {profile_id}")
    if current.get("kind") != spec.get("kind") or current.get("purpose") != spec.get("purpose"):
        raise ValueError(f"non-derived purpose semantics drift: {profile_id}")
    if current.get("python", {}).get("version") != spec.get("python_version"):
        raise ValueError(f"purpose Python policy drift: {profile_id}")

    profile = dict(current)
    profile.pop("profile_sha256", None)
    profile["canonical_lock"] = _canonical_ref(canonical_index)
    kind = str(profile["kind"])
    if kind in {"linux-overlay", "linux-base-role"}:
        base_id = str(spec.get("base_profile", ""))
        profile["base_profile"] = _base_ref(canonical_index, base_id)
    elif kind == "windows-runtime":
        source = current.get("version_source_profile")
        if not isinstance(source, dict):
            raise ValueError(f"Windows purpose profile lacks version source: {profile_id}")
        profile["version_source_profile"] = _base_ref(
            canonical_index, str(source.get("profile_id", "linux-x86_64"))
        )
    else:
        raise ValueError(f"unsupported purpose profile kind: {kind}")

    locks = profile.get("locks")
    if not isinstance(locks, dict):
        raise ValueError(f"purpose profile locks malformed: {profile_id}")
    derived_locks: dict[str, dict[str, Any]] = {}
    for group, record in sorted(locks.items()):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"purpose {profile_id} {group} lock record malformed")
        relative = str(record["path"])
        lock_path = root / relative
        derived_locks[group] = {
            "path": relative,
            "sha256": _sha_file(lock_path),
            "package_count": _count_exact_lock(lock_path),
        }
    profile["locks"] = derived_locks
    profile["profile_sha256"] = _sha_bytes(_canonical(profile))
    return profile


def derive(root: Path) -> dict[Path, bytes]:
    root = root.resolve()
    specs = _load_json(root / "requirements/profiles/specs.json")
    if specs.get("schema_version") != SPECS_SCHEMA:
        raise ValueError("purpose environment specs schema mismatch")
    spec_profiles = specs.get("profiles")
    if not isinstance(spec_profiles, dict) or not spec_profiles:
        raise ValueError("purpose environment specs are empty")

    base_profiles = {
        profile_id: _base_profile(root, profile_id)
        for profile_id in sorted(LOCK.SUPPORTED_PROFILES)
    }
    canonical_index = _base_index(base_profiles)
    expected: dict[Path, bytes] = {
        Path(f"requirements/locks/{profile_id}/profile.json"): _pretty(profile)
        for profile_id, profile in base_profiles.items()
    }
    expected[Path("requirements/locks/index.json")] = _pretty(canonical_index)

    purpose_profiles = {
        profile_id: _purpose_profile(root, profile_id, spec, canonical_index)
        for profile_id, spec in sorted(spec_profiles.items())
    }
    for profile_id, profile in purpose_profiles.items():
        expected[Path(f"requirements/profiles/{profile_id}/profile.json")] = _pretty(profile)

    purpose_records = {
        profile_id: {
            "path": f"requirements/profiles/{profile_id}/profile.json",
            "sha256": _sha_bytes(_pretty(profile)),
            "profile_sha256": str(profile["profile_sha256"]),
        }
        for profile_id, profile in sorted(purpose_profiles.items())
    }
    purpose_index: dict[str, Any] = {
        "schema_version": INDEX_SCHEMA,
        "canonical_lock": _canonical_ref(canonical_index),
        "profiles": purpose_records,
    }
    purpose_index["index_sha256"] = _sha_bytes(_canonical(purpose_index))
    expected[Path("requirements/profiles/index.json")] = _pretty(purpose_index)
    return expected


def converge(root: Path, *, write: bool) -> dict[str, Any]:
    expected = derive(root)
    stale: list[dict[str, str]] = []
    for relative, wanted in sorted(expected.items(), key=lambda item: item[0].as_posix()):
        path = root / relative
        actual = path.read_bytes() if path.exists() else b""
        if actual != wanted:
            stale.append(
                {
                    "path": relative.as_posix(),
                    "expected_sha256": _sha_bytes(wanted),
                    "actual_sha256": _sha_bytes(actual) if actual else "MISSING",
                }
            )
            if write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(wanted)
    if write:
        remaining = converge(root, write=False)["stale"]
        if remaining:
            raise RuntimeError(f"metadata convergence was not idempotent: {remaining}")
    return {"status": "REPAIRED" if write and stale else "PASS", "stale": stale}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = converge(args.repo_root.resolve(), write=args.write)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.check and result["stale"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
