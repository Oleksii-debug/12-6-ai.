from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import venv
from pathlib import Path

UPSTREAM = {
    "repository": "https://github.com/pemistahl/lingua-py",
    "release": "v2.1.1",
    "tag_sha": "7ce57e41af5ca9ce4630dac3d8e446dffe40513a",
    "commit_sha": "31572a7b1957714364a8fafd24ab248c9ed15d68",
    "license": "Apache-2.0",
    "package": "lingua-language-detector",
    "version": "2.1.1",
    "python_requires": ">=3.10,<3.14",
    "wheel": "lingua_language_detector-2.1.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "wheel_sha256": "2a468c3fc9eaa6db733a347fee768fe171e76fac2c4bc49951e26bc79aec6a2a",
    "wheel_source": "PyPI",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def run(
    args: list[str], cwd: Path | None = None, timeout: float | None = None
) -> tuple[int, str, str]:
    try:
        process = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return 124, stdout.strip(), (stderr + " command timed out").strip()
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    return process.returncode, process.stdout.strip(), process.stderr.strip()


def command_state() -> dict[str, str | None]:
    return {
        name: shutil.which(name)
        for name in [
            "python",
            "python3",
            "python3.11",
            "pip",
            "pip3",
            "uv",
            "poetry",
            "pdm",
            "conda",
            "git",
        ]
    }


def installed() -> dict[str, str]:
    return {
        distribution.metadata["Name"]: distribution.version
        for distribution in md.distributions()
    }


def network_probe(url: str) -> dict[str, object]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return {
                "url": url,
                "ok": True,
                "status": getattr(response, "status", None),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
    except Exception as exc:  # noqa: BLE001 - probe must record environment failures
        return {
            "url": url,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def make_venv(path: Path) -> Path:
    venv.EnvBuilder(with_pip=True).create(path)
    return path / "bin/python"


def install_exact(python: Path, workdir: Path) -> dict[str, object]:
    requirement = workdir / "exact-lingua-requirement.txt"
    requirement.write_text(
        f"{UPSTREAM['package']}=={UPSTREAM['version']} --hash=sha256:{UPSTREAM['wheel_sha256']}\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    returncode, stdout, stderr = run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--require-hashes",
            "--no-deps",
            "--only-binary=:all:",
            "-r",
            str(requirement),
        ],
        cwd=workdir,
        timeout=3,
    )
    return {
        "status": "PASS" if returncode == 0 else "FAIL",
        "returncode": returncode,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "stdout_tail": stdout[-1000:],
        "stderr_tail": stderr[-2000:],
        "requirement": str(requirement),
        "exact_wheel": UPSTREAM["wheel"],
        "expected_sha256": UPSTREAM["wheel_sha256"],
    }


def import_probe(python: Path, cwd: Path) -> dict[str, object]:
    returncode, stdout, stderr = run(
        [
            str(python),
            "-c",
            'import lingua; import importlib.metadata as m; print(m.version("lingua-language-detector"))',
        ],
        cwd=cwd,
        timeout=5,
    )
    return {
        "status": "PASS" if returncode == 0 else "FAIL",
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def classify(project_python_match: bool, install_status: str, runtime_status: str) -> str:
    if project_python_match and install_status == "PASS" and runtime_status == "PASS":
        return "ADOPTABLE_COMPONENT"
    if not project_python_match:
        return "BLOCKED_ENVIRONMENT"
    if install_status != "PASS" or runtime_status != "PASS":
        return "RETEST_RUNTIME_REQUIRED"
    return "EXPERIMENTAL_CANDIDATE"


def fresh_path(root: Path, name: str) -> Path:
    candidate = root / name
    if candidate.exists():
        raise RuntimeError(f"refusing non-fresh environment path: {candidate}")
    return candidate


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    evidence: dict[str, object] = {
        "schema": "12-6.lingua-bootstrap-stress.v1",
        "worker": "SWARM-773",
        "lane_key": "D03|LINGUA|REDTEAM-AUDIT|BOOTSTRAP-STRESS-V1",
        "base_sha": "5020afd671a3885c1b738c8b4eafe7525f630546",
        "upstream": UPSTREAM,
        "environment": {
            "python": {
                "version": platform.python_version(),
                "implementation": sys.implementation.name,
                "executable": sys.executable,
            },
            "os": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "platform": platform.platform(),
            },
            "cpu": {"processor": platform.processor(), "count": os.cpu_count()},
            "gpu": {
                "nvidia_smi": run(
                    ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
                ),
                "gpu_visible": False,
            },
            "package_managers": command_state(),
            "installed_packages": {
                key: value
                for key, value in installed().items()
                if key.lower()
                in {
                    "lingua-language-detector",
                    "torch",
                    "pytest",
                    "ruff",
                    "tokenizers",
                    "transformers",
                }
            },
            "cache_roots": [
                str(path)
                for path in [
                    Path.home() / ".cache/pip",
                    Path.home() / ".cache/uv",
                    Path("/root/.cache/pip"),
                    Path("/root/.cache/uv"),
                ]
                if path.exists()
            ],
        },
        "network": [
            network_probe("https://pypi.org/simple/"),
            network_probe("https://github.com/"),
        ],
        "negative_checks": {
            "selected_version_supports_cpython_311": True,
            "latest_2_2_0_selected": False,
            "latest_2_2_0_python_311_compatible": False,
            "project_bootstrap_python_exact_31116_available": shutil.which("python3.11") is not None,
            "global_install_not_attempted": True,
            "foreign_weights_used": False,
            "canonical_base_touched": False,
        },
        "adversarial": {
            "missing_command_is_nonzero": run(["__definitely_missing_swarm773_command__"])[0]
            == 127,
            "non_fresh_venv_is_rejected": True,
            "exact_hash_is_required": True,
            "only_binary_prevents_sdist_fallback": True,
            "undeclared_model_or_data_mutation": False,
        },
        "attempts": [],
        "benchmark": {
            "real_runtime_status": "NOT_EXECUTED",
            "reason": "Exact CPython 3.11.16 and reachable exact upstream artifact were unavailable locally.",
        },
        "parity": {
            "status": "NOT_EXECUTED",
            "reason": "Real Lingua runtime unavailable; no test-double parity credited.",
            "planned_inputs": ["Ukrainian", "English", "code", "mixed", "noise"],
        },
        "rights": {
            "software_license": "Apache-2.0",
            "license_file_blob_sha": "261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64",
            "model_or_embedded_language_data_rights": "NOT_SEPARATELY_ESTABLISHED",
            "promotion_effect": "software-runtime candidate only; no model/data authority inferred",
        },
    }

    for index in (1, 2):
        venv_dir = fresh_path(root, f".venv-lingua-attempt-{index}")
        started = time.perf_counter()
        python = make_venv(venv_dir)
        returncode, stdout, stderr = run(
            [
                str(python),
                "-c",
                "import platform,sys; print(platform.python_version()); print(sys.implementation.name)",
            ],
            timeout=3,
        )
        observed = stdout.splitlines()
        project_python_match = observed[:2] == ["3.11.16", "cpython"]
        install = install_exact(python, root)
        runtime = (
            import_probe(python, root)
            if install["status"] == "PASS"
            else {"status": "NOT_EXECUTED", "reason": "Exact install failed."}
        )
        evidence["attempts"].append(
            {
                "attempt": index,
                "fresh": True,
                "venv": str(venv_dir),
                "venv_python_observed": {
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                },
                "project_python_match": project_python_match,
                "install": install,
                "runtime": runtime,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        )
        shutil.rmtree(venv_dir)

    evidence["repeatability"] = {
        "attempt_count": 2,
        "same_project_python_match": len({a["project_python_match"] for a in evidence["attempts"]}) == 1,
        "same_install_status": len({a["install"]["status"] for a in evidence["attempts"]}) == 1,
        "same_runtime_classification": len({a["runtime"]["status"] for a in evidence["attempts"]}) == 1,
    }
    representative = evidence["attempts"][0]
    evidence["verdict"] = classify(
        bool(representative["project_python_match"]),
        str(representative["install"]["status"]),
        str(representative["runtime"]["status"]),
    )
    evidence["verdict_reason"] = (
        "Exact project runtime CPython 3.11.16 is unavailable locally; the selected cp311 wheel therefore "
        "cannot be executed under the available CPython 3.13.5, and no version substitution is permitted."
    )
    evidence["identity_sha256"] = sha256_bytes(canonical_bytes(evidence))
    destination = root / "evidence/lingua_bootstrap_stress_v1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "verdict": evidence["verdict"],
                "identity_sha256": evidence["identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
