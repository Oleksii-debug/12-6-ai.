"""Keyboard/NVDA-friendly exact-source Plan-8 S5 bridge.

Only approved signed Section-4 test packets can launch; no arbitrary commands.
Raw host logs stay local and are never posted to public GitHub by default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from plan8_physical_operator import _public_key_verifier
from plan8_physical_operator import main as physical_operator
from twelve_six.evidence_control_bridge import (
    bridge_status,
    build_host_dispatch,
    claim_bridge_dispatch,
    publish_bridge_return_github,
    record_bridge_return,
    stage_bridge_dispatch,
    stop_bridge_dispatch,
)
from twelve_six.physical_qualification import load_verified_signed_packet


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _load_dispatch(args):
    verified = load_verified_signed_packet(
        args.packet, signature_verifier=_public_key_verifier(args.authority_keys)
    )
    dispatch = build_host_dispatch(
        verified, repo_root=args.repo_root,
        dispatch_id=args.dispatch_id,
        scenario_id=args.scenario_id,
        physical_gate_id=args.physical_gate_id,
    )
    return dispatch, verified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Signed PC/GitHub/AI test bridge, text-only; no automatic retry."
    )
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("status", "stop"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--spool", type=Path, required=True)
        cmd.add_argument("--dispatch-sha256", required=True)
    for name in ("stage", "execute", "verify", "publish"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--spool", type=Path, required=True)
        cmd.add_argument("--packet", type=Path, required=True)
        cmd.add_argument("--authority-keys", type=Path, required=True)
        cmd.add_argument("--repo-root", type=Path, required=True)
        cmd.add_argument("--dispatch-id", required=True)
        cmd.add_argument("--scenario-id", required=True)
        cmd.add_argument("--physical-gate-id", required=True)
        if name != "stage":
            cmd.add_argument("--host-keys", type=Path, required=True)
            cmd.add_argument("--receipt", type=Path, required=True)
            cmd.add_argument("--log", type=Path, required=True)
        if name == "publish":
            cmd.add_argument("--github-token-env", default="PLAN8_GITHUB_EVIDENCE_TOKEN")
        if name == "execute":
            cmd.add_argument("--host-private-key", type=Path, required=True)
            cmd.add_argument("--host-key-id", required=True)
    args = parser.parse_args(argv)
    if args.action == "status":
        _emit(bridge_status(args.spool, args.dispatch_sha256))
        return 0
    if args.action == "stop":
        stop_bridge_dispatch(args.spool, args.dispatch_sha256)
        _emit({"state": "STOP_REQUESTED", "stop_running_action": "Ctrl+C"})
        return 0

    dispatch, verified = _load_dispatch(args)
    identity = dispatch.identity_sha256()
    if args.action == "stage":
        manifest = stage_bridge_dispatch(
            args.spool, dispatch=dispatch, verified_packet=verified,
            signed_packet_bytes=args.packet.read_bytes(),
        )
        _emit({
            "state": "STAGED", "dispatch_sha256": identity,
            "candidate_sha": manifest["dispatch"]["target_git_sha"],
            "next": "execute only on trusted host; never auto-retry unknown effects",
        })
        return 0
    if args.action == "execute":
        claim_bridge_dispatch(args.spool, identity)
        rc = physical_operator([
            "run",
            "--packet", str(args.packet),
            "--authority-keys", str(args.authority_keys),
            "--host-keys", str(args.host_keys),
            "--host-private-key", str(args.host_private_key),
            "--host-key-id", args.host_key_id,
            "--repo-root", str(args.repo_root),
            "--receipt", str(args.receipt), "--log", str(args.log),
        ])
        if rc:
            _emit({"state": "CLAIMED_OUTCOME_UNKNOWN", "return_code": rc})
            return rc

    if args.action == "publish":
        if args.github_token_env != "PLAN8_GITHUB_EVIDENCE_TOKEN":
            raise ValueError("only the dedicated GitHub token environment variable is allowed")
        token = os.environ.get("PLAN8_GITHUB_EVIDENCE_TOKEN")
        if not token:
            raise ValueError("GitHub evidence token is unavailable")
        published = publish_bridge_return_github(
            args.spool, dispatch=dispatch, verified_packet=verified,
            evidence_path=args.receipt, log_path=args.log,
            agent_source_bytes=(
                Path(__import__("twelve_six.physical_qualification", fromlist=["__file__"]).__file__)
                .read_bytes()
            ),
            artifact_root=args.repo_root,
            evidence_signature_verifier=_public_key_verifier(args.host_keys),
            github_token=token,
        )
        _emit({
            "state": "GITHUB_HASH_ONLY_RECEIPTS_VERIFIED",
            "dispatch_sha256": identity,
            "published": published["publication"],
            "raw_private_logs_uploaded": False,
        })
        return 0

    report = record_bridge_return(
        args.spool, dispatch=dispatch, verified_packet=verified,
        evidence_path=args.receipt, log_path=args.log,
        agent_source_bytes=(
            Path(__import__("twelve_six.physical_qualification", fromlist=["__file__"]).__file__)
            .read_bytes()
        ),
        artifact_root=args.repo_root,
        evidence_signature_verifier=_public_key_verifier(args.host_keys),
    )
    _emit({
        "state": "VERIFIED_RETURN",
        "dispatch_sha256": identity,
        "verdict": report["verdict"],
        "outbound": str(args.spool / "outbound" / identity),
        "physical_verified": True,
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("STOPPED: Ctrl+C; existing claim prevents replay", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, ValueError, ImportError, RuntimeError, KeyError) as exc:
        print("DENIED: " + str(exc), file=sys.stderr)
        raise SystemExit(2) from None
