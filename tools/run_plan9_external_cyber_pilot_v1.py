"""Run the bounded external pilot through separate fresh Python processes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tools.plan9_external_cyber_pilot_v1 import canonical, sha


def _write_once(path: Path, payload: dict) -> None:
    raw = canonical(payload)
    try:
        with path.open("xb") as output:
            output.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError(f"previous pilot result differs: {path.name}") from None


def _invoke(action: str, *arguments: str) -> dict:
    raw = subprocess.check_output([
        sys.executable, "-m", "tools.plan9_external_cyber_pilot_v1", action, *arguments,
    ])
    return json.loads(raw)


def run(args: argparse.Namespace) -> dict:
    if sys.platform != "linux":
        raise ValueError("pilot execution requires Linux on an external CUDA host")
    if not args.execute:
        raise ValueError("pass --execute to run the four-update pilot")
    for path in (args.output, args.scratch, args.bundle):
        if not path.is_dir():
            raise ValueError(f"expected an existing directory: {path}")
    common = ("--bundle", str(args.bundle), "--scratch", str(args.scratch), "--lock-profile", args.lock_profile)
    packet = _invoke("plan", *common)
    if packet["packet_sha256"] != args.packet_sha256:
        raise ValueError("pilot packet differs from the locally reviewed digest")
    execution = (
        "--execute", "--expected-packet-sha256", args.packet_sha256,
        "--install-report", str(args.install_report),
        "--install-report-sha256", sha(args.install_report.read_bytes()),
        "--benchmark-receipt", str(args.benchmark_receipt),
        "--benchmark-receipt-sha256", args.benchmark_receipt_sha256,
    )
    _write_once(args.output / "packet.json", packet)
    namespace = args.scratch / packet["run_manifest_sha256"]
    status = _invoke("status", *common) if (namespace / "control.sqlite3").is_file() else None
    if status is None or status["phase"] == "UNPREPARED":
        status = _invoke("prepare", *common, *execution)
    elif status["phase"] in {"FAILED", "RUNNING"}:
        status = _invoke("run", *common, *execution, "--recover", "--expected-receipt-sha256",
                         status["receipt_sha256"])
    if status["phase"] == "PREPARED":
        if status["committed_step"] != 0 or status["tokens_seen"] != 0:
            raise ValueError("prepared pilot receipt has unexpected progress")
        _write_once(args.output / "initial.json", status)
        status = _invoke("run", *common, *execution, "--expected-receipt-sha256", status["receipt_sha256"])
    if status["phase"] == "PAUSED":
        if status["committed_step"] != 2 or status["tokens_seen"] != 1303:
            raise ValueError("pilot did not pause at the expected data boundary")
        _write_once(args.output / "paused.json", status)
        status = _invoke("run", *common, *execution, "--expected-receipt-sha256", status["receipt_sha256"])
    if status["phase"] != "COMPLETED" or status["committed_step"] != 4 or status["tokens_seen"] != 2696:
        raise ValueError("pilot did not complete at the expected data boundary")
    _write_once(args.output / "completed.json", status)
    proof = _invoke("verify-checkpoint", *common, "--expected-receipt-sha256", status["receipt_sha256"])
    if proof["checkpoint_verified"] is not True:
        raise ValueError("final checkpoint readback did not pass")
    _write_once(args.output / "checkpoint-proof.json", proof)
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--packet-sha256", required=True)
    parser.add_argument("--lock-profile", default="linux-x86_64", choices=("linux-x86_64", "linux-aarch64"))
    parser.add_argument("--install-report", required=True, type=Path)
    parser.add_argument("--benchmark-receipt", required=True, type=Path)
    parser.add_argument("--benchmark-receipt-sha256", required=True)
    args = parser.parse_args()
    try:
        sys.stdout.buffer.write(canonical(run(args)))
        return 0
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"pilot orchestration stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
