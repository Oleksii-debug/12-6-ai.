"""Validate a retained S0 inference bundle before downstream platform execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from twelve_six.inference.portable_artifact import (
    PortableArtifactError,
    validate_portable_runtime_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed validation for a retained S0 inference artifact bundle."
    )
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        report = validate_portable_runtime_artifact(
            args.artifact_root,
            expected_source_sha=args.expected_source_sha,
        )
    except PortableArtifactError as exc:
        print(f"portable-artifact: FAIL {exc}", file=sys.stderr)
        return 2

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        "portable-artifact: PASS "
        f"source={report['source_sha']} "
        f"checkpoint_id={report['checkpoint_id']} "
        f"files={report['artifact_file_count']} "
        f"windows={report['windows_handoff']['runtime_status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
