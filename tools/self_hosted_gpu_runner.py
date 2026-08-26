"""GPU-200 provider-neutral self-hosted CUDA runner contract.

The host preflight is stdlib-only so it can reject an unsuitable runner before the
exact project runtime is installed. Runtime identity is collected only after the
ENV-151 universal bootstrap has installed the declared CUDA purpose environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

HOST_SCHEMA = "12-6.self-hosted-gpu-host-preflight.v1"
ENV_SCHEMA = "12-6.self-hosted-gpu-environment-preflight.v1"
SMOKE_SCHEMA = "12-6.self-hosted-gpu-smoke.v1"
REQUIRED_LABELS = ("self-hosted", "linux", "x64", "gpu", "cuda", "twelve-six-ai")
_SHA40 = re.compile(r"^[0-9a-fA-F]{40}$")
_CUDA_VERSION = re.compile(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)")


class RunnerContractError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _with_identity(payload: dict[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value.pop("identity_sha256", None)
    value["identity_sha256"] = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    return value


def _write_evidence(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    value = _with_identity(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def _read_evidence(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise RunnerContractError("EVIDENCE_SCHEMA_MISMATCH", f"unexpected evidence schema: {path}")
    expected = value.get("identity_sha256")
    if not isinstance(expected, str):
        raise RunnerContractError("EVIDENCE_IDENTITY_MISSING", f"missing evidence identity: {path}")
    unhashed = dict(value)
    unhashed.pop("identity_sha256", None)
    actual = hashlib.sha256(_canonical_bytes(unhashed)).hexdigest()
    if actual != expected:
        raise RunnerContractError(
            "EVIDENCE_IDENTITY_MISMATCH", f"evidence identity mismatch: {path}"
        )
    return value


def validate_exact_sha(value: str) -> str:
    if _SHA40.fullmatch(value) is None:
        raise RunnerContractError(
            "INVALID_TARGET_SHA",
            "target SHA must be exactly 40 hexadecimal characters",
        )
    return value.lower()


def validate_scheduler_labels(labels: Iterable[str]) -> list[str]:
    normalized = sorted({item.strip() for item in labels if item.strip()})
    missing = [label for label in REQUIRED_LABELS if label not in normalized]
    if missing:
        raise RunnerContractError(
            "RUNNER_LABEL_CONTRACT",
            "scheduler selector is missing required labels: " + ", ".join(missing),
        )
    return normalized


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RunnerContractError(
            "GIT_IDENTITY_UNAVAILABLE", "unable to resolve checked-out Git HEAD"
        )
    return completed.stdout.strip().lower()


def require_exact_checkout(root: Path, expected_sha: str) -> str:
    expected = validate_exact_sha(expected_sha)
    actual = _git_head(root)
    if actual != expected:
        raise RunnerContractError(
            "CHECKOUT_SHA_MISMATCH",
            f"checked-out SHA {actual} does not match requested SHA {expected}",
        )
    return actual


def _parse_mib(value: str) -> int:
    return int(round(float(value.strip())))


def parse_nvidia_rows(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise RunnerContractError(
                "NVIDIA_SMI_PARSE_FAILED", "unexpected nvidia-smi query shape"
            )
        index, name, uuid, driver, total_mib, free_mib = fields
        rows.append(
            {
                "physical_index": int(index),
                "name": name,
                "uuid": uuid,
                "driver_version": driver,
                "memory_total_mib": _parse_mib(total_mib),
                "memory_free_mib": _parse_mib(free_mib),
            }
        )
    if not rows:
        raise RunnerContractError("NO_CUDA_DEVICE", "nvidia-smi reported no GPU devices")
    return rows


def query_nvidia_gpus() -> tuple[list[dict[str, Any]], str | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RunnerContractError("NO_CUDA_DEVICE", "nvidia-smi is unavailable on this runner")
    query = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,uuid,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if query.returncode != 0:
        raise RunnerContractError("NO_CUDA_DEVICE", "nvidia-smi could not enumerate a usable GPU")
    rows = parse_nvidia_rows(query.stdout)
    banner = subprocess.run(
        [executable],
        check=False,
        capture_output=True,
        text=True,
    )
    cuda_max = None
    if banner.returncode == 0:
        match = _CUDA_VERSION.search(banner.stdout)
        if match is not None:
            cuda_max = match.group(1)
    return rows, cuda_max


def select_gpu(rows: list[dict[str, Any]], physical_index: int) -> dict[str, Any]:
    if physical_index < 0:
        raise RunnerContractError("INVALID_GPU_INDEX", "GPU index must be non-negative")
    for row in rows:
        if row["physical_index"] == physical_index:
            return row
    raise RunnerContractError(
        "GPU_INDEX_NOT_VISIBLE",
        f"requested GPU index {physical_index} is not visible",
    )


def _gib_to_mib(value: float) -> int:
    if value < 0:
        raise RunnerContractError(
            "INVALID_RESOURCE_REQUIREMENT", "resource requirements cannot be negative"
        )
    return int(round(value * 1024.0))


def require_vram_headroom(
    gpu: dict[str, Any], required_gib: float, reserve_gib: float
) -> dict[str, int]:
    required_mib = _gib_to_mib(required_gib)
    reserve_mib = _gib_to_mib(reserve_gib)
    free_mib = int(gpu["memory_free_mib"])
    if required_mib + reserve_mib > free_mib:
        raise RunnerContractError(
            "INSUFFICIENT_VRAM_HEADROOM",
            "requested VRAM plus reserve exceeds currently free device memory",
        )
    return {
        "required_mib": required_mib,
        "reserve_mib": reserve_mib,
        "free_mib": free_mib,
        "headroom_after_request_mib": free_mib - required_mib,
    }


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def require_durable_root(root: Path, workspace: Path | None, runner_temp: Path | None) -> Path:
    if not root.is_absolute():
        raise RunnerContractError(
            "DURABLE_PATH_NOT_ABSOLUTE", "durable checkpoint root must be absolute"
        )
    resolved = root.resolve()
    if not resolved.is_dir():
        raise RunnerContractError(
            "DURABLE_PATH_MISSING", "durable checkpoint root must already exist"
        )
    for ephemeral in (workspace, runner_temp):
        if ephemeral is None:
            continue
        e = ephemeral.resolve()
        if resolved == e or _is_within(resolved, e):
            raise RunnerContractError(
                "DURABLE_PATH_EPHEMERAL",
                "durable checkpoint root must not be inside the workspace or runner temp directory",
            )
    probe = resolved / f".gpu200-write-probe-{os.getpid()}"
    try:
        with probe.open("x", encoding="utf-8") as handle:
            handle.write("gpu200\n")
    except OSError as exc:
        raise RunnerContractError(
            "DURABLE_PATH_NOT_WRITABLE", "durable checkpoint root is not writable"
        ) from exc
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass
    return resolved


def require_disk_headroom(root: Path, required_gib: float, reserve_gib: float) -> dict[str, int]:
    required_bytes = _gib_to_mib(required_gib) * 1024 * 1024
    reserve_bytes = _gib_to_mib(reserve_gib) * 1024 * 1024
    usage = shutil.disk_usage(root)
    if required_bytes + reserve_bytes > usage.free:
        raise RunnerContractError(
            "INSUFFICIENT_DISK_HEADROOM",
            "requested durable disk plus reserve exceeds currently free space",
        )
    return {
        "required_bytes": required_bytes,
        "reserve_bytes": reserve_bytes,
        "free_bytes": usage.free,
        "headroom_after_request_bytes": usage.free - required_bytes,
    }


def _failure_payload(schema: str, phase: str, error: RunnerContractError) -> dict[str, Any]:
    return {
        "schema": schema,
        "phase": phase,
        "status": "REFUSED",
        "reason_code": error.code,
        "message": str(error),
    }


def host_preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    checkout_sha = require_exact_checkout(root, args.expected_sha)
    labels = validate_scheduler_labels(args.scheduler_label)
    rows, cuda_max = query_nvidia_gpus()
    gpu = select_gpu(rows, args.gpu_index)
    vram = require_vram_headroom(gpu, args.required_vram_gib, args.vram_reserve_gib)
    workspace = Path(os.environ["GITHUB_WORKSPACE"]) if os.environ.get("GITHUB_WORKSPACE") else None
    runner_temp = Path(os.environ["RUNNER_TEMP"]) if os.environ.get("RUNNER_TEMP") else None
    durable = require_durable_root(args.durable_root, workspace, runner_temp)
    disk = require_disk_headroom(durable, args.required_disk_gib, args.disk_reserve_gib)
    payload = {
        "schema": HOST_SCHEMA,
        "phase": "host_preflight",
        "status": "PASS",
        "source_sha": checkout_sha,
        "scheduler_selector_labels": labels,
        "required_labels": list(REQUIRED_LABELS),
        "gpu": {
            **gpu,
            "nvidia_reported_max_cuda_version": cuda_max,
        },
        "resource_gate": {"vram": vram, "durable_disk": disk},
        "durable_checkpoint_root": str(durable),
    }
    return _write_evidence(args.evidence.resolve(), payload)


def _probe_runtime(python: Path, root: Path) -> dict[str, Any]:
    code = r'''
import json, platform, torch
if not torch.cuda.is_available():
    raise SystemExit(41)
if torch.cuda.device_count() < 1:
    raise SystemExit(42)
p = torch.cuda.get_device_properties(0)
print(json.dumps({
    "python_version": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "torch_version": torch.__version__,
    "torch_cuda_build_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "cuda_available": bool(torch.cuda.is_available()),
    "visible_device_count": int(torch.cuda.device_count()),
    "visible_device_0": {
        "name": p.name,
        "compute_capability": [int(p.major), int(p.minor)],
        "total_memory_bytes": int(p.total_memory),
    },
}, sort_keys=True))
'''
    completed = subprocess.run(
        [str(python), "-c", code],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RunnerContractError(
            "CUDA_RUNTIME_NOT_USABLE",
            "exact purpose environment cannot initialize a visible CUDA device",
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunnerContractError(
            "CUDA_RUNTIME_PROBE_INVALID", "CUDA runtime probe returned invalid JSON"
        ) from exc


def environment_preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    checkout_sha = require_exact_checkout(root, args.expected_sha)
    host = _read_evidence(args.host_evidence.resolve(), HOST_SCHEMA)
    if host.get("status") != "PASS" or host.get("source_sha") != checkout_sha:
        raise RunnerContractError(
            "HOST_PREFLIGHT_BINDING_MISMATCH",
            "host preflight is not bound to this source SHA",
        )
    rows, cuda_max = query_nvidia_gpus()
    current = select_gpu(rows, int(host["gpu"]["physical_index"]))
    for key in ("uuid", "name", "driver_version", "memory_total_mib"):
        if current[key] != host["gpu"][key]:
            raise RunnerContractError(
                "GPU_IDENTITY_CHANGED",
                f"GPU identity field changed after bootstrap: {key}",
            )
    runtime = _probe_runtime(args.venv_python.resolve(), root)
    if not runtime.get("cuda_available"):
        raise RunnerContractError("CUDA_RUNTIME_NOT_USABLE", "PyTorch reports CUDA unavailable")
    torch_name = str(runtime["visible_device_0"]["name"])
    if torch_name != str(host["gpu"]["name"]):
        raise RunnerContractError(
            "GPU_RUNTIME_IDENTITY_MISMATCH",
            "PyTorch and nvidia-smi GPU names differ",
        )
    total_bytes = int(runtime["visible_device_0"]["total_memory_bytes"])
    nvidia_total_bytes = int(host["gpu"]["memory_total_mib"]) * 1024 * 1024
    if abs(total_bytes - nvidia_total_bytes) > 256 * 1024 * 1024:
        raise RunnerContractError(
            "GPU_RUNTIME_IDENTITY_MISMATCH",
            "PyTorch and nvidia-smi total-memory identities differ materially",
        )
    payload = {
        "schema": ENV_SCHEMA,
        "phase": "environment_preflight",
        "status": "PASS",
        "source_sha": checkout_sha,
        "host_preflight_identity_sha256": host["identity_sha256"],
        "gpu_uuid": host["gpu"]["uuid"],
        "driver_version": host["gpu"]["driver_version"],
        "nvidia_reported_max_cuda_version": cuda_max,
        "runtime": runtime,
    }
    return _write_evidence(args.evidence.resolve(), payload)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    checkout_sha = require_exact_checkout(root, args.expected_sha)
    host = _read_evidence(args.host_evidence.resolve(), HOST_SCHEMA)
    environment = _read_evidence(args.environment_evidence.resolve(), ENV_SCHEMA)
    if host.get("source_sha") != checkout_sha or environment.get("source_sha") != checkout_sha:
        raise RunnerContractError(
            "SMOKE_EVIDENCE_BINDING_MISMATCH",
            "preflight evidence is not bound to this source SHA",
        )
    if environment.get("host_preflight_identity_sha256") != host.get("identity_sha256"):
        raise RunnerContractError(
            "SMOKE_EVIDENCE_BINDING_MISMATCH",
            "environment preflight does not bind the host preflight",
        )
    durable_root = Path(host["durable_checkpoint_root"])
    checkpoint_dir = durable_root / "12-6-ai" / "gpu200" / checkout_sha
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "cuda-smoke-checkpoint.pt"
    script = r'''
import hashlib, json, math, sys, torch
from pathlib import Path
path = Path(sys.argv[1])
torch.manual_seed(200)
torch.cuda.reset_peak_memory_stats(0)
x = torch.arange(4096, device="cuda:0", dtype=torch.float32).reshape(64, 64) / 4096.0
y = torch.mm(x, x.transpose(0, 1))
if not bool(torch.isfinite(y).all().item()):
    raise SystemExit(51)
torch.save(y.detach().cpu(), path)
loaded = torch.load(path, map_location="cuda:0", weights_only=True)
if not bool(torch.equal(y, loaded)):
    raise SystemExit(52)
z = torch.mm(loaded, x)
if not bool(torch.isfinite(z).all().item()):
    raise SystemExit(53)
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps({
    "checkpoint_bytes": path.stat().st_size,
    "checkpoint_sha256": digest,
    "finite_before_checkpoint": True,
    "checkpoint_reload_equal": True,
    "finite_post_reload_compute": True,
    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    "result_checksum": float(z.sum().item()),
}, sort_keys=True))
'''
    completed = subprocess.run(
        [str(args.venv_python.resolve()), "-c", script, str(checkpoint)],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RunnerContractError(
            "CUDA_SMOKE_FAILED", "CUDA allocation/checkpoint/reload smoke failed"
        )
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RunnerContractError(
            "CUDA_SMOKE_OUTPUT_INVALID", "CUDA smoke returned invalid JSON"
        ) from exc
    payload = {
        "schema": SMOKE_SCHEMA,
        "phase": "cuda_smoke",
        "status": "PASS",
        "source_sha": checkout_sha,
        "host_preflight_identity_sha256": host["identity_sha256"],
        "environment_preflight_identity_sha256": environment["identity_sha256"],
        "gpu_uuid": host["gpu"]["uuid"],
        "durable_checkpoint": {
            "path": str(checkpoint),
            "bytes": int(result["checkpoint_bytes"]),
            "sha256": result["checkpoint_sha256"],
        },
        "cuda": {
            "finite_before_checkpoint": result["finite_before_checkpoint"],
            "checkpoint_reload_equal": result["checkpoint_reload_equal"],
            "finite_post_reload_compute": result["finite_post_reload_compute"],
            "peak_allocated_bytes": int(result["peak_allocated_bytes"]),
            "peak_reserved_bytes": int(result["peak_reserved_bytes"]),
            "result_checksum": result["result_checksum"],
        },
    }
    return _write_evidence(args.evidence.resolve(), payload)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--evidence", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)

    host = sub.add_parser("host-preflight")
    _add_common(host)
    host.add_argument("--scheduler-label", action="append", default=[])
    host.add_argument("--gpu-index", type=int, default=0)
    host.add_argument("--durable-root", type=Path, required=True)
    host.add_argument("--required-vram-gib", type=float, required=True)
    host.add_argument("--vram-reserve-gib", type=float, default=1.0)
    host.add_argument("--required-disk-gib", type=float, required=True)
    host.add_argument("--disk-reserve-gib", type=float, default=2.0)

    env = sub.add_parser("environment-preflight")
    _add_common(env)
    env.add_argument("--venv-python", type=Path, required=True)
    env.add_argument("--host-evidence", type=Path, required=True)

    smoke = sub.add_parser("smoke")
    _add_common(smoke)
    smoke.add_argument("--venv-python", type=Path, required=True)
    smoke.add_argument("--host-evidence", type=Path, required=True)
    smoke.add_argument("--environment-evidence", type=Path, required=True)
    return parser


def _safe_summary(payload: dict[str, Any]) -> str:
    summary = {
        "status": payload.get("status"),
        "phase": payload.get("phase"),
        "identity_sha256": payload.get("identity_sha256"),
    }
    if payload.get("status") == "REFUSED":
        summary["reason_code"] = payload.get("reason_code")
    return json.dumps(summary, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    schema = {
        "host-preflight": HOST_SCHEMA,
        "environment-preflight": ENV_SCHEMA,
        "smoke": SMOKE_SCHEMA,
    }[args.action]
    try:
        if args.action == "host-preflight":
            payload = host_preflight(args)
        elif args.action == "environment-preflight":
            payload = environment_preflight(args)
        else:
            payload = run_smoke(args)
    except RunnerContractError as exc:
        payload = _write_evidence(
            args.evidence.resolve(),
            _failure_payload(schema, args.action.replace("-", "_"), exc),
        )
        print(_safe_summary(payload))
        return 2
    print(_safe_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
