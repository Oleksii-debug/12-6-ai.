#!/usr/bin/env python3
"""Run one workflow phase and always emit a CI-161 failure report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twelve_six.experiment_failure import (  # noqa: E402
    FailurePhase,
    FailureSignal,
    build_report,
    make_signal_for_process,
    read_start_marker,
    safe_name,
    write_report,
)

TAIL_BYTES = 64 * 1024


def _required_module_missing(names: list[str]) -> str | None:
    for name in names:
        if importlib.util.find_spec(name) is None:
            return safe_name(name)
    return None


def _digest_and_tail(handle) -> tuple[dict[str, object], bytes]:
    handle.flush()
    handle.seek(0)
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        digest.update(chunk)
    handle.seek(max(total - TAIL_BYTES, 0))
    tail = handle.read(TAIL_BYTES)
    return {"bytes": total, "sha256": digest.hexdigest()}, tail


def _command_hash(command: list[str]) -> str:
    raw = b"\0".join(part.encode("utf-8", errors="surrogatepass") for part in command)
    return hashlib.sha256(raw).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=[p.value for p in FailurePhase])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--workflow")
    parser.add_argument("--run-id")
    parser.add_argument("--required-module", action="append", default=[])
    parser.add_argument("--start-marker", type=Path)
    parser.add_argument("--require-experiment-start", action="store_true")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    phase = FailurePhase(args.phase)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("command is required after --")

    missing = _required_module_missing(list(args.required_module))
    if missing:
        signal = FailureSignal(
            phase=phase,
            return_code=78,
            optimizer_steps_completed=read_start_marker(args.start_marker),
            missing_dependency=missing,
        )
        report = build_report(
            signal,
            diagnostic_codes=(f"PYTHON_MODULE_MISSING:{missing}",),
            diagnostic_summary=f"required Python module unavailable before phase execution: {missing}",
            source_sha=args.source_sha,
            workflow=args.workflow,
            run_id=args.run_id,
            command_sha256=_command_hash(command),
            executable=Path(command[0]).name,
        )
        write_report(args.report, report)
        return 78

    timed_out = False
    return_code = 0
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=args.timeout_seconds,
                check=False,
                env=os.environ.copy(),
            )
            return_code = int(completed.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = 124
        stdout_meta, stdout_tail = _digest_and_tail(stdout_file)
        stderr_meta, stderr_tail = _digest_and_tail(stderr_file)

    optimizer_steps = read_start_marker(args.start_marker)
    signal, codes = make_signal_for_process(
        phase=phase,
        return_code=return_code,
        optimizer_steps_completed=optimizer_steps,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        timed_out=timed_out,
    )
    if return_code == 0 and args.require_experiment_start and optimizer_steps == 0:
        signal = FailureSignal(
            phase=phase,
            return_code=70,
            optimizer_steps_completed=0,
        )
        return_code = 70
        codes = (*codes, "OPTIMIZER_STEP_MARKER_MISSING")

    report = build_report(
        signal,
        diagnostic_codes=codes,
        diagnostic_summary=(
            "phase command completed"
            if return_code == 0
            else "phase command failed; raw output not retained"
        ),
        source_sha=args.source_sha,
        workflow=args.workflow,
        run_id=args.run_id,
        command_sha256=_command_hash(command),
        executable=Path(command[0]).name,
        stdout_digest=stdout_meta,
        stderr_digest=stderr_meta,
    )
    write_report(args.report, report)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
