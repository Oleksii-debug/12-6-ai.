from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.inference.artifact_transport import (
    build_s0_artifact_transport_manifest,
    validate_s0_artifact_transport_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or validate an S0 artifact transport manifest."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--payload-root", type=Path, required=True)
    build.add_argument("--source-sha", required=True)
    build.add_argument("--manifest-out", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--payload-root", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--expected-source-sha", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        payload = build_s0_artifact_transport_manifest(
            args.payload_root,
            source_sha=args.source_sha,
        )
        args.manifest_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(args.manifest_out)
        return 0

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_s0_artifact_transport_manifest(
        args.payload_root,
        payload,
        expected_source_sha=args.expected_source_sha,
    )
    print("D05 S0 artifact transport: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
