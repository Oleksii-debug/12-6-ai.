#!/usr/bin/env python3
"""Execute the canonical 12-6 checkpoint from a sealed artifact on Windows.

This runner owns environment/install/evidence mechanics only. Checkpoint verification,
model loading, tokenization, and generation stay in the installed first-party package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import venv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "12-6.windows-canonical-checkpoint-execution.v1"
AUTHORITY = "FREE_HOSTED_CPU_WINDOWS_RUNTIME_EVIDENCE_NOT_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
EXACT_PYTHON = "3.11.9"
PROMPT = "Україна — 12-6"

EXPECTED_DISTRIBUTIONS = {
    "filelock": "3.32.4",
    "fsspec": "2026.7.0",
    "jinja2": "3.1.6",
    "markupsafe": "3.0.3",
    "mpmath": "1.3.0",
    "networkx": "3.6.1",
    "numpy": "2.4.6",
    "packaging": "26.3",
    "pip": "26.2.1",
    "safetensors": "0.8.0",
    "setuptools": "84.0.0",
    "sympy": "1.14.0",
    "torch": "2.13.0",
    "twelve-six-ai": "0.2.0.dev0",
    "typing-extensions": "4.16.0",
    "wheel": "0.48.0",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    )


def _run(
    command: Sequence[str | Path],
    *,
    input_text: str | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    rendered = [str(item) for item in command]
    result = subprocess.run(
        rendered,
        input=input_text,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=os.environ.copy(),
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr if capture else ""
        raise RuntimeError(f"command failed ({result.returncode}): {rendered!r}\n{detail}")
    return result


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _require_host() -> None:
    if sys.implementation.name != "cpython" or platform.python_version() != EXACT_PYTHON:
        raise RuntimeError(
            f"expected CPython {EXACT_PYTHON}; got "
            f"{sys.implementation.name} {platform.python_version()}"
        )
    if platform.system() != "Windows" or platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError(f"expected Windows AMD64; got {platform.system()} {platform.machine()}")


def _create_environment(root: Path) -> tuple[Path, Path]:
    environment = root / "Виконання Python With Spaces"
    venv.EnvBuilder(with_pip=True, clear=True).create(environment)
    python = environment / "Scripts" / "python.exe"
    cli = environment / "Scripts" / "twelve-six-generate.exe"
    if not python.is_file():
        raise RuntimeError("venv Python was not created")
    return python, cli


def _offline_install(root: Path, python: Path) -> Path:
    profile_root = root / "requirements" / "locks" / "windows-x86_64"
    toolchain_wheels = root / "wheelhouse" / "toolchain"
    runtime_wheels = root / "wheelhouse" / "runtime"
    project_wheels = sorted((root / "project").glob("twelve_six_ai-*.whl"))
    if len(project_wheels) != 1:
        raise RuntimeError("expected exactly one twelve-six-ai project wheel")

    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--find-links",
            toolchain_wheels,
            "--require-hashes",
            "--no-deps",
            "-r",
            profile_root / "toolchain.lock.txt",
        ]
    )
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--find-links",
            runtime_wheels,
            "--require-hashes",
            "--no-deps",
            "-r",
            profile_root / "runtime.lock.txt",
        ]
    )
    _run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-deps",
            project_wheels[0],
        ]
    )
    _run([python, "-m", "pip", "check"])
    return project_wheels[0]


def _installed_versions(python: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json; "
        f"names={list(EXPECTED_DISTRIBUTIONS)!r}; "
        "print(json.dumps({n:m.version(n) for n in names},sort_keys=True))"
    )
    result = _run([python, "-c", code], capture=True)
    installed = json.loads(result.stdout)
    if installed != EXPECTED_DISTRIBUTIONS:
        raise RuntimeError(f"installed distribution mismatch: {installed!r}")
    return installed


def _torch_runtime(python: Path) -> dict[str, Any]:
    code = (
        "import json,torch; "
        "print(json.dumps({'runtime_version':torch.__version__,"
        "'cuda_available':torch.cuda.is_available(),'cuda_runtime':torch.version.cuda},sort_keys=True))"
    )
    return json.loads(_run([python, "-c", code], capture=True).stdout)


def _validate_checkpoint(root: Path, python: Path) -> dict[str, Any]:
    checkpoint = root / "checkpoint"
    evidence_path = root / "checkpoint-evidence.json"
    _run(
        [
            python,
            "-m",
            "twelve_six.inference.s0_artifact",
            "validate",
            "--checkpoint",
            checkpoint,
            "--evidence",
            evidence_path,
        ],
        capture=True,
    )
    return _load_object(evidence_path)


def _generate(root: Path, cli: Path, checkpoint_id: str, source_sha: str) -> dict[str, Any]:
    if not cli.is_file():
        raise RuntimeError("installed twelve-six-generate console script is missing")
    checkpoint = root / "checkpoint"
    command = [cli, "--checkpoint", checkpoint, "--greedy", "--max-new-tokens", "8"]
    plain = _run(command, input_text=PROMPT, capture=True)
    json_run = _run([*command, "--json"], input_text=PROMPT, capture=True)
    payload = json.loads(json_run.stdout)

    backend = payload.get("backend")
    if not isinstance(backend, dict) or backend.get("backend") != "first_party_torch":
        raise RuntimeError("Windows generation did not use first-party Torch backend")
    if backend.get("git_sha") != source_sha:
        raise RuntimeError("Windows backend source identity mismatch")
    if backend.get("checkpoint_id") != checkpoint_id:
        raise RuntimeError("Windows backend checkpoint identity mismatch")
    if list(payload.get("prompt_token_ids", [])) != list(PROMPT.encode("utf-8")):
        raise RuntimeError("Ukrainian stdin UTF-8 token mapping mismatch")
    if payload.get("mode") != "greedy" or not isinstance(payload.get("text"), str):
        raise RuntimeError("JSON generation diagnostics contract mismatch")

    results = root / "windows-results"
    results.mkdir(exist_ok=True)
    (results / "plain-output.txt").write_text(plain.stdout, encoding="utf-8", newline="")
    (results / "plain-diagnostics.txt").write_text(plain.stderr, encoding="utf-8", newline="")
    (results / "json-output.json").write_text(json_run.stdout, encoding="utf-8", newline="")
    (results / "json-diagnostics.txt").write_text(json_run.stderr, encoding="utf-8", newline="")

    return {
        "prompt": PROMPT,
        "input_transport": "UTF-8_STDIN_PIPE",
        "prompt_utf8_sha256": _sha256_bytes(PROMPT.encode("utf-8")),
        "plain_output_sha256": _sha256_bytes(plain.stdout.encode("utf-8")),
        "plain_diagnostics_sha256": _sha256_bytes(plain.stderr.encode("utf-8")),
        "json_output_sha256": _sha256_bytes(json_run.stdout.encode("utf-8")),
        "json_diagnostics_sha256": _sha256_bytes(json_run.stderr.encode("utf-8")),
        "json_backend": backend,
        "generated_token_ids": payload.get("generated_token_ids"),
        "mode": payload.get("mode"),
        "stop_reason": payload.get("stop_reason"),
    }


def execute(root: Path, source_sha: str) -> dict[str, Any]:
    _require_host()
    root = root.resolve()
    if " " not in str(root) or not any(ord(ch) > 127 for ch in str(root)):
        raise RuntimeError("execution root must contain both spaces and Unicode")

    manifest = _load_object(root / "artifact-manifest.json")
    if manifest.get("source_sha") != source_sha or manifest.get("repository") != REPOSITORY:
        raise RuntimeError("sealed artifact source/repository identity mismatch")
    if manifest.get("physical_repository_trailing_dot_blocker") is not True:
        raise RuntimeError("physical trailing-dot repository blocker was not preserved")
    if manifest.get("windows_repository_checkout_required") is not False:
        raise RuntimeError("artifact incorrectly requires a Windows repository checkout")

    python, cli = _create_environment(root)
    project_wheel = _offline_install(root, python)
    installed = _installed_versions(python)
    torch_runtime = _torch_runtime(python)
    checkpoint_evidence = _validate_checkpoint(root, python)
    checkpoint_id = checkpoint_evidence["checkpoint"]["checkpoint_id"]
    if checkpoint_evidence.get("candidate_sha") != source_sha:
        raise RuntimeError("checkpoint source SHA differs from workflow source SHA")
    if checkpoint_id != manifest["checkpoint"]["checkpoint_id"]:
        raise RuntimeError("checkpoint ID differs from sealed artifact manifest")

    generation = _generate(root, cli, checkpoint_id, source_sha)
    project_sha = _sha256_file(project_wheel)
    if project_sha != manifest["project_wheel"]["sha256"]:
        raise RuntimeError("project wheel SHA differs from sealed artifact manifest")

    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "repository": REPOSITORY,
        "physical_repository_checkout": {
            "status": "BLOCKED_BY_TRAILING_DOT",
            "repository_checkout_used": False,
            "artifact_only_safe_path_used": True,
            "safe_path_has_spaces": True,
            "safe_path_has_unicode": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "installed_distributions": installed,
            "torch_distribution_version": installed["torch"],
            "torch_runtime": torch_runtime,
            "offline_install": True,
            "pip_no_index": True,
            "require_hashes": True,
            "project_wheel_sha256": project_sha,
            "windows_profile_file_sha256": manifest["windows_profile"]["file_sha256"],
        },
        "checkpoint": {
            "validated_on_windows": True,
            "checkpoint_id": checkpoint_id,
            "source_sha": checkpoint_evidence["candidate_sha"],
            "first_party_backend": generation["json_backend"]["backend"],
        },
        "generation": generation,
        "accessibility_boundary": {
            "keyboard_text_interface_suitability": "PASS_PIPE_STDIN_TEXT_STDOUT_NO_TTY_REQUIRED",
            "manual_nvda_accessibility": "NOT_TESTED_REQUIRES_HUMAN",
        },
        "claims": {
            "canonical_torch_checkpoint_executed_on_windows": True,
            "foreign_pretrained_weights_used": False,
            "instruction_or_chat_behavior_added": False,
            "promotion_authority": False,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    results = root / "windows-results"
    (results / "windows-canonical-evidence.json").write_text(
        json.dumps(evidence, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = execute(args.root, args.source_sha)
    print(
        json.dumps(
            {
                "checkpoint_id": evidence["checkpoint"]["checkpoint_id"],
                "evidence_sha256": evidence["evidence_sha256"],
                "first_party_backend": evidence["checkpoint"]["first_party_backend"],
                "manual_nvda_accessibility": evidence["accessibility_boundary"][
                    "manual_nvda_accessibility"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
