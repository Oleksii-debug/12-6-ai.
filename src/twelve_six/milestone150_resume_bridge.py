"""Fresh-process MILESTONE-150 resume with JSON-semantic run-manifest comparison.

The original ladder implementation self-hashes its run manifest through canonical JSON,
but its in-memory TrainerConfig contains a tuple for AdamW betas. A persisted JSON
manifest necessarily reloads that value as a list. The semantic identity is unchanged,
yet direct Python container equality rejects the round trip. This bridge canonicalizes
the freshly rebuilt run manifest through JSON before the original resume contract runs.
No training configuration, checkpoint identity, corpus identity, or hash is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from twelve_six import milestone150_learned_base_ladder as ladder


def json_semantic(value: dict[str, Any]) -> dict[str, Any]:
    """Return the exact JSON round-trip representation used by persisted manifests."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("run manifest canonicalization must preserve a JSON object")
    return decoded


def resume(repo: Path, source_sha: str, out: Path, scale: str) -> dict[str, Any]:
    """Execute the original resume path with only its rebuilt manifest JSON-normalized."""
    original: Callable[..., dict[str, Any]] = ladder._run_manifest

    def canonical_run_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return json_semantic(original(*args, **kwargs))

    ladder._run_manifest = canonical_run_manifest
    try:
        return ladder.resume(repo, source_sha, out, scale)
    finally:
        ladder._run_manifest = original


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", choices=ladder.SCALE_ORDER, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    resume(args.repo_root, args.source_sha, args.output_dir, args.scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
