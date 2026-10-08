"""Keyboard-first, fail-closed Plan 8 physical qualification operator.

No key generation, arbitrary commands, remote execution, or implicit physical PASS.
Run from an exact clean Git checkout with PYTHONPATH=src.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import stat
import sys
from pathlib import Path

from twelve_six.physical_qualification import (
    ExecutionMode,
    execute_qualification,
    load_verified_signed_packet,
    verify_qualification_evidence,
    write_evidence_bundle,
)

_KEY_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_MAX_STORE_BYTES = 65536
_MAX_STATUS_BYTES = 4 * 1024 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    obj: dict[str, object] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def _load_json(path: Path, limit: int) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("input must be a regular non-symlink file")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("input exceeds byte limit")
    result = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    if type(result) is not dict:
        raise ValueError("JSON root must be an object")
    return result


def _public_key_verifier(path: Path):
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    doc = _load_json(path, _MAX_STORE_BYTES)
    if set(doc) != {"schema_version", "keys"} or doc["schema_version"] != (
        "12-6.physical-trust-store.v1"
    ):
        raise ValueError("unsupported trust store")
    keys = doc["keys"]
    if type(keys) is not dict or not 1 <= len(keys) <= 64:
        raise ValueError("trust store must have 1..64 keys")
    verified_keys = {}
    for key_id, encoded in keys.items():
        if type(key_id) is not str or not _KEY_ID.fullmatch(key_id):
            raise ValueError("invalid trust-store key id")
        if type(encoded) is not str:
            raise ValueError("public key must be base64")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) != 32:
            raise ValueError("Ed25519 public key must be 32 bytes")
        verified_keys[key_id] = Ed25519PublicKey.from_public_bytes(raw)

    def verify(key_id: str, message: bytes, signature: bytes) -> bool:
        key = verified_keys.get(key_id)
        if key is None:
            return False
        try:
            key.verify(signature, message)
            return True
        except InvalidSignature:
            return False

    return verify


def _host_signer(path: Path, repo_root: Path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if path.is_symlink() or not path.is_file():
        raise ValueError("host signing key must be a regular non-symlink file")
    if path.resolve().is_relative_to(repo_root.resolve()):
        raise ValueError("host signing key must live outside the Git repository")
    metadata = path.stat()
    if os.name != "nt" and stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("host signing key must have owner-only permissions")
    with path.open("rb") as stream:
        raw = stream.read(33)
    if len(raw) != 32:
        raise ValueError("host signing key must contain 32 raw Ed25519 bytes")
    key = Ed25519PrivateKey.from_private_bytes(raw)
    return lambda _key_id, message: key.sign(message)


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan 8 physical qualification: keyboard and screen-reader friendly."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="Read plain-text status; does not verify trust")
    status.add_argument("--receipt", type=Path, required=True)
    for command in ("run", "verify"):
        item = commands.add_parser(command, help="Signed packet execution or verification")
        item.add_argument("--packet", type=Path, required=True)
        item.add_argument("--authority-keys", type=Path, required=True)
        item.add_argument("--host-keys", type=Path, required=True)
        item.add_argument("--repo-root", type=Path, required=True)
        item.add_argument("--receipt", type=Path, required=True)
        item.add_argument("--log", type=Path, required=True)
        if command == "run":
            item.add_argument("--host-private-key", type=Path, required=True)
            item.add_argument("--host-key-id", required=True)
        else:
            item.add_argument("--allow-simulation", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "status":
        if not args.receipt.exists():
            _emit({"state": "NOT_RUN", "receipt": str(args.receipt)})
            return 0
        payload = _load_json(args.receipt, _MAX_STATUS_BYTES)
        _emit({
            "state": "UNVERIFIED_RECEIPT",
            "claimed_verdict": payload.get("verdict", "UNKNOWN"),
            "verification": "NOT_CHECKED",
            "receipt": str(args.receipt),
        })
        return 0

    authority_verify = _public_key_verifier(args.authority_keys)
    host_verify = _public_key_verifier(args.host_keys)
    packet = load_verified_signed_packet(
        args.packet, signature_verifier=authority_verify
    )
    repo_root = args.repo_root.resolve()
    if args.command == "run":
        if type(args.host_key_id) is not str or not _KEY_ID.fullmatch(args.host_key_id):
            raise ValueError("invalid host key id")
        if args.receipt.exists() or args.log.exists():
            raise ValueError("receipt/log already exists: use new paths to avoid overwrite")
        signer = _host_signer(args.host_private_key, repo_root)
        _emit({
            "state": "RUNNING",
            "candidate_sha": packet.packet.target_git_sha,
            "mode": packet.packet.execution_mode.value,
            "stop": "Press Ctrl+C to terminate the test process tree",
        })
        receipt, log = execute_qualification(
            packet,
            repo_root=repo_root,
            evidence_signing_key_id=args.host_key_id,
            evidence_signer=signer,
        )
        write_evidence_bundle(args.receipt, args.log, evidence=receipt, log_bytes=log)
        require_real = packet.packet.execution_mode is ExecutionMode.REAL_HOST
    else:
        require_real = not args.allow_simulation

    validated = verify_qualification_evidence(
        args.receipt,
        args.log,
        verified_packet=packet,
        agent_source_bytes=(
            Path(__import__("twelve_six.physical_qualification", fromlist=["__file__"]).__file__)
            .read_bytes()
        ),
        artifact_root=repo_root,
        require_real_pass=require_real,
        evidence_signature_verifier=host_verify,
    )
    _emit({
        "state": "VERIFIED",
        "candidate_sha": packet.packet.target_git_sha,
        "mode": packet.packet.execution_mode.value,
        "verdict": validated.get("verdict"),
        "physical_required": require_real,
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("STOPPED: Ctrl+C; child process cleanup requested", file=sys.stderr)
        raise SystemExit(130) from None
    except (OSError, ValueError, ImportError, RuntimeError, KeyError) as error:
        print("DENIED: " + str(error), file=sys.stderr)
        raise SystemExit(2) from None
