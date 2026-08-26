#!/usr/bin/env python3
"""Assess the fail-closed learned-20M launch packet."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from twelve_six.learned20m_readiness import (
    R01_CAMPAIGN_BLOB_SHA1,
    assess_learned20m_readiness,
)

DEFAULT_PATH = Path("configs/research/r01_learned20m_launch_readiness_v1.json")
DEFAULT_R01_CAMPAIGN_PATH = Path(
    "configs/research/r01_20m_to_100m_scaling_campaign_v1.json"
)


def git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    prefix = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(prefix + payload).hexdigest()


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_PATH
    campaign_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_R01_CAMPAIGN_PATH

    observed_campaign_blob = git_blob_sha1(campaign_path)
    if observed_campaign_blob != R01_CAMPAIGN_BLOB_SHA1:
        print(
            json.dumps(
                {
                    "error": "r01_campaign_file_identity_mismatch",
                    "expected_git_blob_sha1": R01_CAMPAIGN_BLOB_SHA1,
                    "observed_git_blob_sha1": observed_campaign_blob,
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 2

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        print(json.dumps({"error": "launch packet root must be an object"}, sort_keys=True))
        return 2
    result = assess_learned20m_readiness(payload).as_dict()
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result["material_training_authorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
