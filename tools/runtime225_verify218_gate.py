"""Validate VERIFY-218 and emit exact learned-source coordinates for RUNTIME-225."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.inference.verify218_authority import (
    load_json_object,
    validate_verify218_authority,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-manifest", type=Path, required=True)
    parser.add_argument("--verify-artifact-metadata", type=Path, required=True)
    parser.add_argument("--verify-run-metadata", type=Path, required=True)
    parser.add_argument("--verify-artifact-id", type=int, required=True)
    parser.add_argument("--verify-artifact-digest", required=True)
    parser.add_argument("--verify-run-id", type=int, required=True)
    parser.add_argument("--verify-source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = validate_verify218_authority(
        load_json_object(args.verify_manifest, label="VERIFY-218 authority manifest"),
        load_json_object(args.verify_artifact_metadata, label="VERIFY-218 artifact metadata"),
        load_json_object(args.verify_run_metadata, label="VERIFY-218 run metadata"),
        verifier_artifact_id=args.verify_artifact_id,
        verifier_artifact_digest=args.verify_artifact_digest,
        verifier_run_id=args.verify_run_id,
        verifier_source_sha=args.verify_source_sha,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evidence, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
