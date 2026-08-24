from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from twelve_six.integration.live_authority import validate_live_promotion_authority
from twelve_six.integration.release_attestation import load_release_attestation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate D10 release evidence locally and resolve claimed promotion authority "
            "against live GitHub evidence."
        )
    )
    parser.add_argument("attestation", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        attestation = load_release_attestation(args.attestation)
        manifest = validate_live_promotion_authority(
            attestation,
            repo_root=args.repo_root,
            artifact_root=args.artifact_root,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"live promotion authority: BLOCKED: {exc}", file=sys.stderr)
        return 2

    result = {
        "repository": attestation.repository,
        "stage": attestation.stage,
        "status": attestation.status.value,
        "candidate_sha": attestation.candidate_sha,
        "live_authority_verified": True,
        "audits_pass": bool(manifest is not None and manifest.audits_pass()),
        "promotion_created": False,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
