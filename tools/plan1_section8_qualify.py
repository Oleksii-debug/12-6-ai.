"""Plan-1-only terminal qualification; reports evidence, never self-declares DONE.

Run from a clean repository checkout with the project's dev dependencies installed.
This LOCAL_FREE runner neither installs packages from the network nor trains a model.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = (
    "tests/test_system_architecture_section0.py",
    "tests/test_artifact_identity_section1.py",
    "tests/test_plan1_dependency_inventory.py",
    "tests/test_plan1_section3_reuse.py",
    "tests/test_dependency_lock.py",
    "tests/test_plan1_section4_lock_trust_boundary.py",
    "tests/test_dependency_security.py",
    "tests/test_plan1_section5_notices.py",
    "tests/test_plan1_section5_release_fixture.py",
    "tests/test_plan1_section6_contracts.py",
    "tests/test_plan1_section7_backend_compatibility.py",
    "tests/test_plan1_section8_foundation.py",
)
RUFF = (
    "src/twelve_six/contracts.py",
    "src/twelve_six/backend_compatibility.py",
    "src/twelve_six/integration/dependency_lock.py",
    "src/twelve_six/integration/dependency_security.py",
    "src/twelve_six/integration/release_fixture.py",
    "tests/test_plan1_section8_foundation.py",
    "tools/plan1_section8_qualify.py",
)


def run(*argv: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        argv, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False, timeout=600,
    )
    if result.returncode:
        print(result.stdout[-12000:], file=sys.stderr)
        raise RuntimeError(f"qualification command failed ({result.returncode}): {argv!r}")
    return result.stdout


def git_state() -> str:
    sha = run("git", "rev-parse", "HEAD").strip()
    if len(sha) != 40 or any(ch not in "0123456789abcdef" for ch in sha):
        raise RuntimeError("requires exact full SHA-1 Git checkout")
    run("git", "diff", "--exit-code", "HEAD", "--")
    run("git", "diff", "--cached", "--exit-code", "HEAD", "--")
    return sha


def wheel_digest(directory: Path, env: dict[str, str]) -> tuple[str, str]:
    run(sys.executable, "-m", "pip", "wheel", ".", "--no-deps",
        "--no-index", "--no-build-isolation", "--wheel-dir", str(directory), env=env)
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("expected exactly one locally built wheel")
    return wheels[0].name, hashlib.sha256(wheels[0].read_bytes()).hexdigest()


def main() -> int:
    try:
        source_sha = git_state()
        env = dict(os.environ, PIP_NO_INDEX="1", PYTHONHASHSEED="0",
                   SOURCE_DATE_EPOCH="315532800")
        run(sys.executable, "-m", "ruff", "check", *RUFF, env=env)
        run(sys.executable, "-m", "pytest", "-q", *TESTS, env=env)
        run(sys.executable, "-m", "compileall", "-q", "src/twelve_six",
            "tools/plan1_section8_qualify.py", env=env)
        with tempfile.TemporaryDirectory(prefix="plan1-s8-") as temporary:
            first, second = Path(temporary) / "first", Path(temporary) / "second"
            first.mkdir()
            second.mkdir()
            a, b = wheel_digest(first, env), wheel_digest(second, env)
        if a != b:
            raise RuntimeError("clean wheel rebuild is not byte-reproducible")
        if git_state() != source_sha:
            raise RuntimeError("checkout changed during qualification")
        print(json.dumps({
            "plan": 1, "section": 8, "candidate_sha": source_sha,
            "scope": "repository-local fixture qualification only",
            "tests": list(TESTS), "ruff": "PASS", "pytest": "PASS",
            "compileall": "PASS", "wheel_rebuild": "BYTE_IDENTICAL",
            "wheel_name": a[0], "wheel_sha256": a[1],
            "paid_compute": False, "physical_platform_certified": False,
            "plan8_readiness_granted": False, "terminal_done": False,
        }, sort_keys=True))
        return 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"PLAN1_S8_NOT_QUALIFIED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
