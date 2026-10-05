from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.swarm_claim_snapshot import build_claim_snapshot, strict_json_loads

_INPUT_KEYS = {
    "repository",
    "main_sha",
    "generated_at_utc",
    "coverage",
    "claims",
    "scheduler_reservations",
    "queued_actions",
    "in_progress_actions",
}


def build_from_input(document: dict[str, Any]) -> dict[str, Any]:
    if type(document) is not dict:
        raise ValueError("snapshot input must be an object")
    if set(document) != _INPUT_KEYS:
        missing = sorted(_INPUT_KEYS - set(document))
        extra = sorted(set(document) - _INPUT_KEYS)
        raise ValueError(f"snapshot input keys mismatch: missing={missing}, extra={extra}")
    return build_claim_snapshot(
        repository=document["repository"],
        main_sha=document["main_sha"],
        generated_at_utc=document["generated_at_utc"],
        coverage=document["coverage"],
        claims=document["claims"],
        scheduler_reservations=document["scheduler_reservations"],
        queued_actions=document["queued_actions"],
        in_progress_actions=document["in_progress_actions"],
    )


def write_create_only(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fail-closed SWARM-300 immutable claim snapshot "
            "from normalized live inputs."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    raw = args.input.read_text(encoding="utf-8")
    document = strict_json_loads(raw)
    snapshot = build_from_input(document)
    write_create_only(args.output, snapshot)
    print(f"SWARM_CLAIM_SNAPSHOT_SHA256={snapshot['snapshot_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
