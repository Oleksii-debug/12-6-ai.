"""Dependency-free hardware inventory for C01 run manifests."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _total_memory_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        try:
            success = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        except (AttributeError, OSError):
            return None
        return int(status.ullTotalPhys) if success else None

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        physical_pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if page_size <= 0 or physical_pages <= 0:
        return None
    return page_size * physical_pages


def _torch_profile() -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_devices": [],
            "mps_available": False,
        }

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "major": int(properties.major),
                    "minor": int(properties.minor),
                }
            )

    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    return {
        "installed": True,
        "version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_device_count": len(devices),
        "cuda_devices": devices,
        "mps_available": mps_available,
    }


def collect_hardware_profile(*, working_directory: str | Path | None = None) -> dict[str, Any]:
    """Return factual local hardware/runtime inventory without inferring cost authorization."""

    working_path = Path(working_directory) if working_directory is not None else Path.cwd()
    disk = shutil.disk_usage(working_path)
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "cpu": {
            "logical_cores": os.cpu_count(),
        },
        "memory": {
            "total_bytes": _total_memory_bytes(),
        },
        "disk": {
            "path": str(working_path.resolve()),
            "total_bytes": int(disk.total),
            "used_bytes": int(disk.used),
            "free_bytes": int(disk.free),
        },
        "torch": _torch_profile(),
        "execution_context": {
            "github_actions": os.environ.get("GITHUB_ACTIONS") == "true",
            "runner_os": os.environ.get("RUNNER_OS"),
            "runner_arch": os.environ.get("RUNNER_ARCH"),
        },
        "authorization": {
            "cost_inferred": False,
            "note": "Hardware discovery never implies permission to incur metered cost.",
        },
    }


def main() -> int:
    print(json.dumps(collect_hardware_profile(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
