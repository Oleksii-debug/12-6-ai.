#!/usr/bin/env python3
"""Execute one D03 clean-successor script with terminal V7 as a data-only package.

Terminal V7's package __init__ imports the model stack (and therefore torch), but the
clean-successor path needs only `twelve_six.data.*`. This bootstrap reproduces the
incumbent `verify_next100_065f_v8_authority._install_v7_namespace` seam: it installs
an empty package namespace rooted at the exact V7 checkout and then executes the
requested local script in-process. It never stubs torch, imports model code, changes
matcher semantics, or supplies model/data artifacts.

Before any candidate script executes, this bootstrap authenticates the Product
checkout, complete base-main bulk closure, and exact historical V7 Git tree through
the independent verifier on the same Product head.  After producer execution it
independently verifies source/Data526 artifacts before returning success.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
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


def _load_authority_verifier(current_root: Path) -> Any:
    path = current_root / "tools/verify_d03_nomis_free_execution_authority_v1.py"
    spec = importlib.util.spec_from_file_location("_swarm2065_execution_authority", path)
    if spec is None or spec.loader is None:
        raise DataOnlyBootstrapError(f"cannot load execution authority verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arg_value(argv: list[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        raise DataOnlyBootstrapError(f"{name} requires a value")
    return argv[index + 1]


def _require_arg(argv: list[str], name: str) -> str:
    value = _arg_value(argv, name)
    if value is None:
        raise DataOnlyBootstrapError(f"missing required authority argument: {name}")
    return value


def _authority_output_paths(script: Path, argv: list[str]) -> list[Path]:
    if script.name == "run_d03_nomis_free_clean_successor_v1.py" and argv and argv[0] == "run":
        return [Path(_require_arg(argv, "--report")), Path(_require_arg(argv, "--survivor"))]
    if script.name == "materialize_d03_nomis_free_data526_successor_v1.py":
        return [
            Path(_require_arg(argv, "--historical-records-jsonl")),
            Path(_require_arg(argv, "--records-jsonl")),
            Path(_require_arg(argv, "--inventory-json")),
            Path(_require_arg(argv, "--evidence-json")),
        ]
    return []


def _require_create_only_outputs(script: Path, argv: list[str]) -> None:
    outputs = _authority_output_paths(script, argv)
    _seen: set[Path] = set()
    for path in outputs:
        resolved = path.resolve()
        if resolved in _seen:
            raise DataOnlyBootstrapError(f"authority outputs must be distinct: {path}")
        _seen.add(resolved)
        if path.exists():
            raise DataOnlyBootstrapError(f"authority output already exists (no overwrite): {path}")


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
    """Retry only the immutable whole exact-source transaction on transient transport."""

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


def _post_verify(
    *,
    verifier: Any,
    current_root: Path,
    expected_head: str,
    script: Path,
    script_args: list[str],
) -> None:
    name = script.name
    if name == "run_d03_nomis_free_clean_successor_v1.py":
        command = script_args[0] if script_args else ""
        if command not in {"run", "verify"}:
            return
        report_path = Path(_require_arg(script_args, "--report"))
        survivor_path = Path(_require_arg(script_args, "--survivor"))
        verifier.verify_source_authority(
            current_root=current_root,
            expected_product_head=expected_head,
            report=verifier._read_json(report_path),
            survivor=verifier._read_json(survivor_path),
        )
        print("PASS_SWARM2065_BOOTSTRAP_SOURCE_POSTVERIFY")
        return

    if name == "materialize_d03_nomis_free_data526_successor_v1.py":
        verifier.verify_data526_authority(
            current_root=current_root,
            expected_product_head=expected_head,
            source_report=verifier._read_json(Path(_require_arg(script_args, "--source-report"))),
            survivor=verifier._read_json(Path(_require_arg(script_args, "--survivor"))),
            historical_records_path=Path(
                _require_arg(script_args, "--historical-records-jsonl")
            ),
            records_path=Path(_require_arg(script_args, "--records-jsonl")),
            inventory=verifier._read_json(Path(_require_arg(script_args, "--inventory-json"))),
            evidence=verifier._read_json(Path(_require_arg(script_args, "--evidence-json"))),
            pr623_root=Path(_require_arg(script_args, "--historical-materializer-root")),
        )
        print("PASS_SWARM2065_BOOTSTRAP_DATA526_POSTVERIFY")


def main() -> int:
    if len(sys.argv) < 3:
        raise DataOnlyBootstrapError(
            "usage: run_d03_nomis_free_v7_data_only_v1.py V7_ROOT SCRIPT [ARGS...]"
        )
    v7_root = Path(sys.argv[1])
    script = Path(sys.argv[2])
    script_args = list(sys.argv[3:])
    if not script.is_file():
        raise DataOnlyBootstrapError(f"missing local execution script: {script}")

    # Authority-producing runs must be tied to the workflow/event supplied Product
    # head.  A caller-supplied CLI value alone is intentionally insufficient here.
    expected_head = os.environ.get("SOURCE_SHA")
    if expected_head is None:
        raise DataOnlyBootstrapError("SOURCE_SHA authority is required")

    current_root = Path(".").resolve()
    verifier = _load_authority_verifier(current_root)
    verifier.verify_product_checkout(current_root, expected_head)
    verifier.verify_v7_checkout(v7_root)

    historical_root_value = _arg_value(script_args, "--historical-materializer-root")
    if historical_root_value is not None:
        verifier.verify_pr623_checkout(Path(historical_root_value))

    # The candidate producers historically used replace-in-place writes.  The
    # canonical authority carrier tightens this to create-only outputs, so a
    # second execution cannot overwrite retained evidence from the first.
    _require_create_only_outputs(script, script_args)

    # Avoid manufacturing untracked __pycache__ files in the authenticated V7 tree.
    sys.dont_write_bytecode = True
    install_v7_namespace(v7_root)
    install_bounded_exact_fetch_retry()
    sys.argv = [str(script), *script_args]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        if code not in (None, 0):
            if isinstance(code, int):
                return code
            raise

    _post_verify(
        verifier=verifier,
        current_root=current_root,
        expected_head=expected_head,
        script=script,
        script_args=script_args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
