#!/usr/bin/env python3
"""Execute one D03 clean-successor script with terminal V7 as a data-only package.

Terminal V7's package __init__ imports the model stack (and therefore torch), but the
clean-successor path needs only `twelve_six.data.*`. This bootstrap reproduces the
incumbent `verify_next100_065f_v8_authority._install_v7_namespace` seam: it installs
an empty package namespace rooted at the exact V7 checkout and then executes the
requested local script in-process. It never stubs torch, imports model code, changes
matcher semantics, or supplies model/data artifacts.

The immutable historical fetcher performs one complete exact-source transaction per
call, including the response read and its historical byte/hash/blob acceptance checks.
The physical successor may retry that whole call only for bounded transient transport
failures. Retries cannot substitute content or turn an authority mismatch into a pass.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import runpy
import socket
import sys
import time
import types
import urllib.error
from pathlib import Path
from typing import Any, Callable

MAX_TRANSIENT_FETCH_ATTEMPTS = 3
RETRY_DELAYS_SECONDS = (1.0, 2.0)


class DataOnlyBootstrapError(RuntimeError):
    """Fail-closed terminal-V7 data-only bootstrap error."""


def install_v7_namespace(v7_root: Path) -> None:
    package_root = (v7_root / "src" / "twelve_six").resolve()
    if not package_root.is_dir():
        raise DataOnlyBootstrapError(f"missing terminal V7 package root: {package_root}")

    existing = sys.modules.get("twelve_six")
    if existing is not None:
        paths = [str(Path(value).resolve()) for value in getattr(existing, "__path__", [])]
        if str(package_root) not in paths:
            raise DataOnlyBootstrapError(
                "twelve_six already loaded from a different authority"
            )
        return

    package = types.ModuleType("twelve_six")
    package.__package__ = "twelve_six"
    package.__path__ = [str(package_root)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "twelve_six", loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = [str(package_root)]
    sys.modules["twelve_six"] = package


def _is_transient_url_error(exc: urllib.error.URLError) -> bool:
    reason = exc.reason
    return isinstance(reason, (TimeoutError, socket.timeout, ConnectionResetError))


def build_bounded_exact_fetch_retry(
    original: Callable[..., Any],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[..., Any]:
    """Retry the immutable whole fetch call only for transient transport failures.

    The wrapped historical function owns request construction, response reading, size
    limits and exact byte/hash/blob verification. HTTP errors, certificate/DNS failures
    and every authority/content exception remain single-attempt/fail-closed.
    """

    def bounded_fetch(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(MAX_TRANSIENT_FETCH_ATTEMPTS):
            try:
                return original(*args, **kwargs)
            except urllib.error.URLError as exc:
                if not _is_transient_url_error(exc):
                    raise
                transient: BaseException = exc
            except (TimeoutError, socket.timeout, ConnectionResetError) as exc:
                transient = exc

            if attempt + 1 >= MAX_TRANSIENT_FETCH_ATTEMPTS:
                raise transient
            sleep(RETRY_DELAYS_SECONDS[attempt])
        raise AssertionError("unreachable transient-retry state")

    return bounded_fetch


def install_bounded_exact_fetch_retry() -> None:
    """Patch only terminal V7's immutable exact-source transaction entrypoint."""

    audit = importlib.import_module("twelve_six.data.cross_source_capacity_audit")
    original = getattr(audit, "fetch_exact_source", None)
    if not callable(original):
        raise DataOnlyBootstrapError(
            "terminal V7 exact-source fetch entrypoint is missing or non-callable"
        )
    if getattr(original, "_d03_bounded_exact_fetch_retry", False):
        return

    wrapped = build_bounded_exact_fetch_retry(original)
    setattr(wrapped, "_d03_bounded_exact_fetch_retry", True)
    audit.fetch_exact_source = wrapped


def main() -> int:
    if len(sys.argv) < 3:
        raise DataOnlyBootstrapError(
            "usage: run_d03_nomis_free_v7_data_only_v1.py V7_ROOT SCRIPT [ARGS...]"
        )
    v7_root = Path(sys.argv[1])
    script = Path(sys.argv[2])
    if not script.is_file():
        raise DataOnlyBootstrapError(f"missing local execution script: {script}")

    install_v7_namespace(v7_root)
    install_bounded_exact_fetch_retry()
    sys.argv = [str(script), *sys.argv[3:]]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
