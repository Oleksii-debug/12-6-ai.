"""ENV-151 deterministic capability-to-lock execution bootstrap.

Stdlib-only by design: this file can validate an experiment contract before any
project/runtime dependency is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any

SCHEMA = "12-6.execution-capabilities.v1"
MANIFEST_SCHEMA = "12-6.execution-environment-manifest.v1"
_LOCK_LINE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*==([^\s;@/\\]+)"
    r"(?: --hash=sha256:[0-9a-f]{64})+$"
)
_NAME = re.compile(r"[-_.]+")


class ExecutionBootstrapError(RuntimeError):
    pass


def _canonical_name(value: str) -> str:
    return _NAME.sub("-", value.strip()).lower()


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
    path = root / "requirements/execution/capabilities.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ExecutionBootstrapError("execution capability registry schema mismatch")
    if platform.python_version() != value["python"]["version"]:
        raise ExecutionBootstrapError(
            f"bootstrap requires CPython {value['python']['version']}, got {platform.python_version()}"
        )
    if sys.implementation.name != value["python"]["implementation"]:
        raise ExecutionBootstrapError("Python implementation mismatch")
    return value


def _validate_lock(root: Path, role: str, record: dict[str, Any]) -> dict[str, str]:
    path = _safe(root, record["path"])
    if not path.is_file():
        raise ExecutionBootstrapError(f"{role}: lock is missing: {path}")
    digest = _sha256_file(path)
    if digest != record["sha256"]:
        raise ExecutionBootstrapError(f"{role}: lock SHA-256 drift")
    packages: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _LOCK_LINE.fullmatch(line) is None:
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
    executable_map = registry.get("dependency_executables", {})
    name = Path(words[0]).name
    return executable_map.get(name)


def resolve_plan(root: Path, capabilities: list[str], commands: list[str]) -> dict[str, Any]:
    registry = _load_registry(root)
    declared: list[str] = []
    for cap in capabilities:
        cap = cap.strip()
        if cap and cap not in declared:
            declared.append(cap)
    if not declared:
        raise ExecutionBootstrapError("at least one capability must be declared")
    cap_records = registry["capabilities"]
    for cap in declared:
        record = cap_records.get(cap)
        if not isinstance(record, dict):
            raise ExecutionBootstrapError(f"unknown capability: {cap}")
        if record.get("status") != "available":
            raise ExecutionBootstrapError(
                f"{cap}: {record.get('status', 'unavailable')} ({record.get('reason', 'no exact lock')})"
            )
        for requirement in record.get("requires", []):
            if requirement not in declared:
                raise ExecutionBootstrapError(f"{cap} requires declared capability {requirement}")
    for command in commands:
        needed = _command_capability(command, registry)
        if needed is not None and needed not in declared:
            raise ExecutionBootstrapError(f"command {command!r} requires undeclared capability {needed}")
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
    ordered_roles: list[str] = []
    for role in roles:
        if role not in ordered_roles:
            ordered_roles.append(role)
    merged: dict[str, str] = {}
    lock_records: list[dict[str, Any]] = []
    for role in ordered_roles:
        record = registry["locks"][role]
        packages = _validate_lock(root, role, record)
        for name, version in packages.items():
            old = merged.get(name)
            if old is not None and old != version:
                raise ExecutionBootstrapError(f"lock conflict for {name}: {old} versus {version}")
            merged[name] = version
        lock_record: dict[str, Any] = {
            "role": role,
            "path": record["path"],
            "sha256": record["sha256"],
            "package_count": record["package_count"],
        }
        if "index_url" in record:
            lock_record["index_url"] = record["index_url"]
        if "extra_index_url" in record:
            lock_record["extra_index_url"] = record["extra_index_url"]
        lock_records.append(lock_record)
    forbidden = [
        name for name in merged
        if name.startswith("nvidia-") or name.startswith("cuda-") or name == "triton"
    ]
    if "cuda" not in declared and forbidden:
        raise ExecutionBootstrapError(
            "non-CUDA plan inherited CUDA packages: " + ", ".join(sorted(forbidden))
        )
    imports: list[str] = []
    executables: list[str] = []
    for cap in declared:
        imports.extend(cap_records[cap].get("imports", []))
        executables.extend(cap_records[cap].get("executables", []))
    payload = {
        "schema": "12-6.execution-plan.v1",
        "capabilities": declared,
        "commands": commands,
        "locks": lock_records,
        "imports": sorted(set(imports)),
        "executables": sorted(set(executables)),
        "package_count": len(merged),
        "cuda_packages_present": bool(forbidden),
        "d08_authority": registry["d08_authority"],
    }
    payload["identity_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def _venv_python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")


def _run(command: list[str | Path], cwd: Path) -> None:
    subprocess.run([str(item) for item in command], cwd=cwd, check=True)


def _probe_imports(python: Path, modules: list[str], cwd: Path) -> None:
    if not modules:
        return
    code = (
        "import importlib\n"
        f"mods={modules!r}\n"
        "missing=[]\n"
        "for name in mods:\n"
        "  try: importlib.import_module(name)\n"
        "  except Exception as e: missing.append((name, type(e).__name__, str(e)))\n"
        "assert not missing, 'missing imports: '+repr(missing)\n"
    )
    _run([python, "-c", code], cwd)


def _probe_executables(python: Path, executables: list[str]) -> None:
    bindir = python.parent
    missing = []
    for name in executables:
        candidates = [bindir / name]
        if platform.system() == "Windows":
            candidates.append(bindir / f"{name}.exe")
        if not any(path.is_file() for path in candidates):
            missing.append(name)
    if missing:
        raise ExecutionBootstrapError(
            "missing declared executables in bootstrap venv: " + ", ".join(sorted(missing))
        )


def _installed(python: Path, cwd: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m, json\n"
        "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions()}, sort_keys=True))"
    )
    completed = subprocess.run(
        [str(python), "-c", code], cwd=cwd, check=True, capture_output=True, text=True
    )
    return json.loads(completed.stdout)


def preflight(root: Path, python: Path, plan: dict[str, Any], allow_no_gpu: bool) -> dict[str, Any]:
    _probe_imports(python, plan["imports"], root)
    _probe_executables(python, plan["executables"])
    cuda = {
        "software_capability_declared": "cuda" in plan["capabilities"],
        "hardware_visible": False,
        "hardware_claim": False,
        "no_gpu_preflight": False,
    }
    if "cuda" in plan["capabilities"]:
        completed = subprocess.run(
            [str(python), "-c", "import torch; print(int(torch.cuda.is_available()))"],
            cwd=root, check=True, capture_output=True, text=True
        )
        visible = completed.stdout.strip() == "1"
        cuda["hardware_visible"] = visible
        cuda["hardware_claim"] = visible
        if not visible:
            if not allow_no_gpu:
                raise ExecutionBootstrapError("CUDA software is installed but no CUDA hardware is visible")
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
        command: list[str | Path] = [
            python, "-m", "pip", "install", "--disable-pip-version-check", "--require-hashes", "--no-deps"
        ]
        if "index_url" in lock:
            command.extend(["--index-url", lock["index_url"]])
        if "extra_index_url" in lock:
            command.extend(["--extra-index-url", lock["extra_index_url"]])
        command.extend(["-r", _safe(root, lock["path"])])
        _run(command, cwd=root)
    proof = preflight(root, python, plan, allow_no_gpu)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "plan": plan,
        "python": {"implementation": sys.implementation.name, "bootstrap_version": platform.python_version(), "venv_executable": str(python)},
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "packages": _installed(python, root),
        "preflight": proof,
    }
    manifest["identity_sha256"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        manifest = bootstrap(root, args.venv.resolve(), capabilities, args.command, args.manifest.resolve(), args.allow_no_gpu)
        print(json.dumps({"status": "PASS", "identity_sha256": manifest["identity_sha256"], "capabilities": manifest["plan"]["capabilities"], "cuda": manifest["preflight"]["cuda"]}, sort_keys=True))
        return 0
    if args.venv is None:
        parser.error("preflight requires --venv")
    print(json.dumps(preflight(root, _venv_python(args.venv.resolve()), resolve_plan(root, capabilities, args.command), args.allow_no_gpu), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
