#!/usr/bin/env python3
"""Compose two clean G05/G06 replay receipts into one terminal scoped authority."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.data.clean_g05_g06_authority_v1 import (
    CleanG05G06AuthorityError,
    build_clean_g05_g06_authority,
    verify_clean_g05_g06_authority,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CleanG05G06AuthorityError(f"{path} must contain a JSON object")
    return value


def _cjson(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise CleanG05G06AuthorityError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-a", required=True, type=Path)
    parser.add_argument("--expected-replay-a-identity", required=True)
    parser.add_argument("--replay-b", required=True, type=Path)
    parser.add_argument("--expected-replay-b-identity", required=True)
    parser.add_argument("--composition-preflight", required=True, type=Path)
    parser.add_argument("--expected-composition-preflight-identity", required=True)
    parser.add_argument("--expected-terminal-g06-qualification-identity", required=True)
    parser.add_argument("--expected-upstream-clean-successor-authority", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        authority = build_clean_g05_g06_authority(
            replay_a=_load(args.replay_a),
            expected_replay_a_identity_sha256=args.expected_replay_a_identity,
            replay_b=_load(args.replay_b),
            expected_replay_b_identity_sha256=args.expected_replay_b_identity,
            composition_preflight=_load(args.composition_preflight),
            expected_composition_preflight_identity_sha256=(
                args.expected_composition_preflight_identity
            ),
            expected_terminal_g06_qualification_identity_sha256=(
                args.expected_terminal_g06_qualification_identity
            ),
            expected_upstream_clean_successor_authority_identity_sha256=(
                args.expected_upstream_clean_successor_authority
            ),
        )
        identity = authority["authority_identity_sha256"]
        verify_clean_g05_g06_authority(
            authority,
            expected_authority_identity_sha256=identity,
            expected_upstream_clean_successor_authority_identity_sha256=(
                args.expected_upstream_clean_successor_authority
            ),
            expected_replay_a_identity_sha256=args.expected_replay_a_identity,
            expected_replay_b_identity_sha256=args.expected_replay_b_identity,
            expected_composition_preflight_identity_sha256=(
                args.expected_composition_preflight_identity
            ),
            expected_terminal_g06_qualification_identity_sha256=(
                args.expected_terminal_g06_qualification_identity
            ),
        )
        raw = _cjson(authority)
        if b"normalized_payload" in raw:
            raise CleanG05G06AuthorityError("source text marker leaked into authority")
        _write_new(args.output, raw)
    except (OSError, json.JSONDecodeError, CleanG05G06AuthorityError) as exc:
        raise SystemExit(f"clean G05/G06 authority composition failed closed: {exc}") from exc

    print(identity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
