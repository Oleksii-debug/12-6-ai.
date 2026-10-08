"""Verify committed dependency locks and clean wheel/editable installs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
import venv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = Path("requirements/locks/index.json")
LOCK_MODULE = ROOT / "src" / "twelve_six" / "integration" / "dependency_lock.py"
_LOCK_LINE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*==[^\s;@/\\]+(?: --hash=sha256:[0-9a-f]{64})+$"
)
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _load_contract() -> Any:
    spec = importlib.util.spec_from_file_location("twelve_six_dependency_lock", LOCK_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load dependency lock contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOCK = _load_contract()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _safe_relative_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise LOCK.DependencyLockError(f"unsafe lock path: {relative}")
    resolved = (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise LOCK.DependencyLockError(f"lock path escapes repository: {relative}") from exc
    return resolved


def _validate_lock_text(path: Path, expected_count: int) -> None:
    seen: set[str] = set()
    count = 0
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _LOCK_LINE.fullmatch(line) is None:
            raise LOCK.DependencyLockError(f"non-exact or unhashed lock line {path}:{number}")
        name = LOCK.canonical_distribution_name(line.split("==", 1)[0])
        if name in seen:
            raise LOCK.DependencyLockError(f"duplicate locked distribution {name} in {path}")
        seen.add(name)
        count += 1
    if count != expected_count:
        raise LOCK.DependencyLockError(
            f"lock package count mismatch for {path}: manifest={expected_count} actual={count}"
        )


def validate_committed_profile(profile_id: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    LOCK.assert_exact_python()
    index = LOCK.validate_lock_index(root=ROOT, index_path=INDEX_PATH)
    current = LOCK.current_profile_id()
    expected = profile_id or current
    if expected != current:
        raise LOCK.DependencyLockError(
            f"requested profile {expected!r} does not match current platform {current!r}"
        )
    record = index["profiles"][expected]
    profile_path = _safe_relative_path(record["path"])
    profile = LOCK.validate_profile_manifest(
        root=ROOT,
        manifest_path=profile_path.relative_to(ROOT),
        enforce_current_platform=True,
    )
    for group, lock_record in profile["locks"].items():
        lock_path = _safe_relative_path(lock_record["path"])
        expected_parent = (ROOT / "requirements" / "locks" / expected).resolve()
        try:
            lock_path.relative_to(expected_parent)
        except ValueError as exc:
            raise LOCK.DependencyLockError(
                f"{group} lock is outside profile directory {expected}"
            ) from exc
        _validate_lock_text(lock_path, int(lock_record["package_count"]))
    return index, profile


def _venv_python(directory: Path) -> Path:
    path = directory / "bin" / "python"
    if not path.exists():
        raise RuntimeError(f"virtualenv Python missing: {path}")
    return path


def _run(command: list[str | Path], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    rendered = [str(item) for item in command]
    subprocess.run(rendered, cwd=cwd, env=env, check=True)


def _output(command: list[str | Path], *, cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _install_locked(python: Path, profile: dict[str, Any], groups: tuple[str, ...]) -> None:
    for group in groups:
        lock_path = _safe_relative_path(profile["locks"][group]["path"])
        _run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--require-hashes",
                "--no-deps",
                "-r",
                lock_path,
            ]
        )
    version = _output([python, "-m", "pip", "--version"])
    if not version.startswith("pip 26.2.1 "):
        raise RuntimeError(f"locked pip version mismatch: {version}")


def _offline_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PIP_NO_INDEX": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PYTHONUTF8": "1",
            "SOURCE_DATE_EPOCH": "0",
        }
    )
    return env


def _smoke(python: Path, environment: Path) -> None:
    _run(
        [
            python,
            "-c",
            (
                "import importlib.metadata as m; import twelve_six; "
                "import twelve_six.inference.cli; "
                "assert m.version('twelve-six-ai') == '0.2.0.dev0'; "
                "print(twelve_six.__version__)"
            ),
        ]
    )
    command = environment / "bin" / "twelve-six-generate"
    if not command.exists():
        raise RuntimeError("console script twelve-six-generate was not installed")
    completed = subprocess.run(
        [str(command), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if "usage: twelve-six-generate" not in completed.stdout:
        raise RuntimeError("console-script help smoke did not expose expected CLI")


def _installed_distributions(python: Path) -> list[dict[str, str]]:
    code = (
        "import importlib.metadata as m,json; "
        "rows=sorted((d.metadata.get('Name') or '',d.version) for d in m.distributions()); "
        "print(json.dumps([{'name':n,'version':v} for n,v in rows if n],sort_keys=True))"
    )
    value = json.loads(_output([python, "-c", code]))
    if not isinstance(value, list):
        raise TypeError("installed distribution inventory must be a list")
    return value


def _run_repo_checks(python: Path) -> None:
    _run([python, "tools/check_repo_policy.py"])
    _run([python, "-m", "ruff", "check", "src", "tests", "tools"])
    _run([python, "-m", "pytest", "-q", "tests/test_s0_convergence_integration.py"])
    _run([python, "-m", "pytest", "-q"])
    _run([python, "tools/validate_stage_candidate.py", "configs/releases/s0_candidate.template.json"])


def verify_install(
    *,
    profile_id: str | None,
    source_sha: str | None,
    evidence_out: Path | None,
    run_repo_checks: bool,
) -> dict[str, Any]:
    index, profile = validate_committed_profile(profile_id)
    if evidence_out is not None and _GIT_SHA.fullmatch(source_sha or "") is None:
        raise ValueError("--source-sha must be a full 40-character lowercase Git SHA for evidence")

    with tempfile.TemporaryDirectory(prefix="twelve-six-lock-") as temp_name:
        temp = Path(temp_name)
        editable_env = temp / "editable"
        wheel_env = temp / "wheel"
        wheel_dir = temp / "dist"
        venv.EnvBuilder(with_pip=True, clear=True).create(editable_env)
        editable_python = _venv_python(editable_env)
        _install_locked(editable_python, profile, ("toolchain", "runtime", "dev"))
        offline = _offline_env()
        _run(
            [
                editable_python,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "-e",
                ROOT,
            ],
            env=offline,
        )
        _smoke(editable_python, editable_env)
        wheel_dir.mkdir()
        _run(
            [
                editable_python,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                wheel_dir,
                ROOT,
            ],
            env=offline,
        )
        wheels = sorted(wheel_dir.glob("twelve_six_ai-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected exactly one project wheel, found {len(wheels)}")
        wheel = wheels[0]
        if run_repo_checks:
            _run_repo_checks(editable_python)

        venv.EnvBuilder(with_pip=True, clear=True).create(wheel_env)
        wheel_python = _venv_python(wheel_env)
        _install_locked(wheel_python, profile, ("toolchain", "runtime"))
        _run(
            [
                wheel_python,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                wheel,
            ],
            env=offline,
        )
        _smoke(wheel_python, wheel_env)
        installed = _installed_distributions(wheel_python)

        evidence: dict[str, Any] = {
            "schema_version": "12-6.locked-environment-evidence.v1",
            "source_sha": source_sha or "UNBOUND_LOCAL",
            "profile_id": profile["profile_id"],
            "python": profile["python"],
            "lock_index": {
                "path": INDEX_PATH.as_posix(),
                "file_sha256": _sha256_file(ROOT / INDEX_PATH),
                "index_sha256": index["index_sha256"],
            },
            "lock_profile": {
                "manifest_sha256": profile["manifest_sha256"],
                "file_sha256": index["profiles"][profile["profile_id"]]["sha256"],
            },
            "wheel": {
                "filename": wheel.name,
                "sha256": _sha256_file(wheel),
            },
            "installed_distributions": installed,
            "installed_distributions_sha256": hashlib.sha256(_canonical_bytes(installed)).hexdigest(),
            "verification": {
                "committed_lock_validation": "PASS",
                "editable_install_import_cli": "PASS",
                "wheel_install_import_cli": "PASS",
                "repo_checks": "PASS" if run_repo_checks else "NOT_RUN",
            },
        }
        evidence["evidence_sha256"] = hashlib.sha256(_canonical_bytes(evidence)).hexdigest()
        if evidence_out is not None:
            destination = evidence_out if evidence_out.is_absolute() else ROOT / evidence_out
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--source-sha")
    parser.add_argument("--evidence-out", type=Path)
    parser.add_argument("--run-repo-checks", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    index, profile = validate_committed_profile(args.profile)
    if args.validate_only:
        print(f"profile={profile['profile_id']}")
        print(f"index_sha256={index['index_sha256']}")
        print(f"manifest_sha256={profile['manifest_sha256']}")
        return 0

    evidence = verify_install(
        profile_id=args.profile,
        source_sha=args.source_sha,
        evidence_out=args.evidence_out,
        run_repo_checks=args.run_repo_checks,
    )
    print(f"profile={evidence['profile_id']}")
    print(f"evidence_sha256={evidence['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
