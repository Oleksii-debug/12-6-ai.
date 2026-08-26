"""Deterministic capability-to-lock execution bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any
import venv

SCHEMA = "12-6.execution-capabilities.v1"
MANIFEST_SCHEMA = "12-6.execution-environment-manifest.v1"
CPU_EXTRA_INDEX = "https://download.pytorch.org/whl/cpu"
LOCK_LINE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*==([^\s;@/\\]+)(?: --hash=sha256:[0-9a-f]{64})+$"
)
NAME_NORMALIZER = re.compile(r"[-_.]+")


class ExecutionBootstrapError(RuntimeError):
    """Raised when an execution capability cannot be materialized exactly."""


def _canonical_name(value: str) -> str:
    return NAME_NORMALIZER.sub("-", value.strip()).lower()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _safe(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ExecutionBootstrapError(f"unsafe lock path: {relative}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ExecutionBootstrapError(f"lock path escapes repository: {relative}") from exc
    return resolved


def _load_registry(root: Path) -> dict[str, Any]:
    value = json.loads(
        (root / "requirements/execution/capabilities.json").read_text(encoding="utf-8")
    )
    if value.get("schema") != SCHEMA:
        raise ExecutionBootstrapError("execution capability registry schema mismatch")
    python = value["python"]
    if platform.python_version() != python["version"]:
        raise ExecutionBootstrapError(
            f"bootstrap requires CPython {python['version']}, got {platform.python_version()}"
        )
    if sys.implementation.name != python["implementation"]:
        raise ExecutionBootstrapError("Python implementation mismatch")
    return value


def _validate_lock(root: Path, role: str, record: dict[str, Any]) -> dict[str, str]:
    path = _safe(root, record["path"])
    if not path.is_file():
        raise ExecutionBootstrapError(f"{role}: lock is missing: {path}")
    if _sha256_file(path) != record["sha256"]:
        raise ExecutionBootstrapError(f"{role}: lock SHA-256 drift")
    packages: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if LOCK_LINE.fullmatch(line) is None:
            raise ExecutionBootstrapError(f"{role}: non-exact/unhashed line {number}")
        name, tail = line.split("==", 1)
        canonical = _canonical_name(name)
        version = tail.split()[0]
        if canonical in packages:
            raise ExecutionBootstrapError(f"{role}: duplicate package {canonical}")
        packages[canonical] = version
    if len(packages) != record["package_count"]:
        raise ExecutionBootstrapError(f"{role}: package-count drift")
    return packages


def _command_capability(command: str, registry: dict[str, Any]) -> str | None:
    words = command.strip().split()
    if not words:
        return None
    if words[:3] == ["python", "-m", "pytest"] or words[0] == "pytest":
        return "tests"
    if words[:3] == ["python", "-m", "ruff"] or words[0] == "ruff":
        return "lint"
    return registry.get("dependency_executables", {}).get(Path(words[0]).name)


def resolve_plan(root: Path, capabilities: list[str], commands: list[str]) -> dict[str, Any]:
    registry = _load_registry(root)
    declared = list(dict.fromkeys(cap.strip() for cap in capabilities if cap.strip()))
    if not declared:
        raise ExecutionBootstrapError("at least one capability must be declared")

    capability_records = registry["capabilities"]
    for capability in declared:
        record = capability_records.get(capability)
        if not isinstance(record, dict):
            raise ExecutionBootstrapError(f"unknown capability: {capability}")
        if record.get("status") != "available":
            raise ExecutionBootstrapError(
                f"{capability}: {record.get('status', 'unavailable')} "
                f"({record.get('reason', 'no exact lock')})"
            )
        for requirement in record.get("requires", []):
            if requirement not in declared:
                raise ExecutionBootstrapError(
                    f"{capability} requires declared capability {requirement}"
                )

    for command in commands:
        needed = _command_capability(command, registry)
        if needed is not None and needed not in declared:
            raise ExecutionBootstrapError(
                f"command {command!r} requires undeclared capability {needed}"
            )

    roles = ["toolchain"]
    if "cuda" in declared:
        if "runtime" not in declared:
            raise ExecutionBootstrapError("cuda requires runtime")
        roles.append("cuda_runtime")
    elif "runtime" in declared or "distributed" in declared:
        roles.append("cpu_runtime")
    if "tokenizer" in declared:
        if "runtime" not in declared:
            roles.append("tokenizer_support")
        roles.append("tokenizer_overlay")
    if "transformers" in declared:
        if "runtime" not in declared:
            roles.append("transformers_support")
        roles.append("transformers_overlay")
    if "tests" in declared or "lint" in declared:
        roles.append("dev")
    roles = list(dict.fromkeys(roles))

    merged: dict[str, str] = {}
    locks = []
    for role in roles:
        record = registry["locks"][role]
        packages = _validate_lock(root, role, record)
        for name, version in packages.items():
            if name in merged and merged[name] != version:
                raise ExecutionBootstrapError(
                    f"lock conflict for {name}: {merged[name]} versus {version}"
                )
            merged[name] = version
        locks.append(
            {
                "role": role,
                "path": record["path"],
                "sha256": record["sha256"],
                "package_count": record["package_count"],
            }
        )

    forbidden = [
        name
        for name in merged
        if name.startswith("nvidia-") or name.startswith("cuda-") or name == "triton"
    ]
    if "cuda" not in declared and forbidden:
        raise ExecutionBootstrapError(
            "non-CUDA plan inherited CUDA packages: " + ", ".join(sorted(forbidden))
        )

    imports: list[str] = []
    executables: list[str] = []
    for capability in declared:
        imports.extend(capability_records[capability].get("imports", []))
        executables.extend(capability_records[capability].get("executables", []))

    plan: dict[str, Any] = {
        "schema": "12-6.execution-plan.v1",
        "capabilities": declared,
        "commands": commands,
        "locks": locks,
        "imports": sorted(set(imports)),
        "executables": sorted(set(executables)),
        "package_count": len(merged),
        "cuda_packages_present": bool(forbidden),
        "d08_authority": registry["d08_authority"],
    }
    plan["identity_sha256"] = hashlib.sha256(_canonical_bytes(plan)).hexdigest()
    return plan


def _venv_python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def _run(command: list[str | Path], cwd: Path, *, env: dict[str, str] | None = None) -> None:
    rendered = [str(item) for item in command]
    print("+", " ".join(rendered), flush=True)
    subprocess.run(rendered, cwd=cwd, check=True, env=env)


def _install_lock(python: Path, root: Path, lock: dict[str, Any]) -> None:
    command: list[str | Path] = [
        python,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--require-hashes",
        "--no-deps",
    ]
    if lock["role"] == "cpu_runtime":
        command += ["--extra-index-url", CPU_EXTRA_INDEX]
    command += ["-r", _safe(root, lock["path"])]
    _run(command, root)


def _probe_imports(python: Path, modules: list[str], cwd: Path) -> None:
    if not modules:
        return
    code = (
        "import importlib\n"
        f"mods={modules!r}\n"
        "missing=[]\n"
        "for name in mods:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        missing.append((name,type(exc).__name__,str(exc)))\n"
        "assert not missing, 'missing imports: '+repr(missing)\n"
    )
    _run([python, "-c", code], cwd)


def _probe_executables(python: Path, executables: list[str]) -> None:
    missing = []
    for name in executables:
        candidates = [python.parent / name]
        if platform.system() == "Windows":
            candidates.append(python.parent / f"{name}.exe")
        if not any(path.is_file() for path in candidates):
            missing.append(name)
    if missing:
        raise ExecutionBootstrapError(
            "missing declared executables in bootstrap venv: " + ", ".join(sorted(missing))
        )


def _installed(python: Path, cwd: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m,json\n"
        "print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()},sort_keys=True))"
    )
    output = subprocess.run(
        [str(python), "-c", code],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(output)


def preflight(
    root: Path,
    python: Path,
    plan: dict[str, Any],
    allow_no_gpu: bool,
) -> dict[str, Any]:
    _probe_imports(python, plan["imports"], root)
    _probe_executables(python, plan["executables"])
    cuda = {
        "software_capability_declared": "cuda" in plan["capabilities"],
        "hardware_visible": False,
        "hardware_claim": False,
        "no_gpu_preflight": False,
    }
    if "cuda" in plan["capabilities"]:
        output = subprocess.run(
            [str(python), "-c", "import torch; print(int(torch.cuda.is_available()))"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        visible = output == "1"
        cuda["hardware_visible"] = visible
        cuda["hardware_claim"] = visible
        if not visible:
            if not allow_no_gpu:
                raise ExecutionBootstrapError(
                    "CUDA software is installed but no CUDA hardware is visible"
                )
            cuda["no_gpu_preflight"] = True
    return {"status": "PASS", "cuda": cuda}


def bootstrap(
    root: Path,
    venv_dir: Path,
    capabilities: list[str],
    commands: list[str],
    manifest_path: Path,
    allow_no_gpu: bool,
) -> dict[str, Any]:
    plan = resolve_plan(root, capabilities, commands)
    if venv_dir.exists():
        raise ExecutionBootstrapError(f"refusing non-fresh venv: {venv_dir}")
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = _venv_python(venv_dir)
    for lock in plan["locks"]:
        _install_lock(python, root, lock)
    proof = preflight(root, python, plan, allow_no_gpu)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "plan": plan,
        "python": {
            "implementation": sys.implementation.name,
            "bootstrap_version": platform.python_version(),
            "venv_executable": str(python),
        },
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "packages": _installed(python, root),
        "preflight": proof,
    }
    manifest["identity_sha256"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("plan", "bootstrap", "preflight"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--capabilities", required=True)
    parser.add_argument("--command", action="append", default=[])
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-no-gpu", action="store_true")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    capabilities = _csv(args.capabilities)

    if args.action == "plan":
        print(json.dumps(resolve_plan(root, capabilities, args.command), indent=2, sort_keys=True))
        return 0
    if args.action == "bootstrap":
        if args.venv is None or args.manifest is None:
            parser.error("bootstrap requires --venv and --manifest")
        manifest = bootstrap(
            root,
            args.venv.resolve(),
            capabilities,
            args.command,
            args.manifest.resolve(),
            args.allow_no_gpu,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "identity_sha256": manifest["identity_sha256"],
                    "capabilities": manifest["plan"]["capabilities"],
                    "cuda": manifest["preflight"]["cuda"],
                },
                sort_keys=True,
            )
        )
        return 0

    if args.venv is None:
        parser.error("preflight requires --venv")
    plan = resolve_plan(root, capabilities, args.command)
    print(
        json.dumps(
            preflight(root, _venv_python(args.venv.resolve()), plan, args.allow_no_gpu),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
