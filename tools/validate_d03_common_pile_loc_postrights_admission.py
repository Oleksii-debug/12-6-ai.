#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.d03_loc_postrights_admission import CONFIG_PATH, load_and_validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.repo_root.resolve()
    payload = load_and_validate(root / CONFIG_PATH, repo_root=root)
    summary = {
        "authority_id": payload["authority_id"],
        "status": payload["status"],
        "source_admitted_candidate_records": payload["truth_boundary"][
            "source_admitted_candidate_records"
        ],
        "source_admitted_candidate_bytes": payload["truth_boundary"][
            "source_admitted_candidate_bytes"
        ],
        "authorized_optimized_target_exposure": payload["truth_boundary"][
            "authorized_optimized_target_exposure"
        ],
        "training_executed": payload["truth_boundary"]["model_training_executed"],
        "learned_weights_created": payload["truth_boundary"]["learned_weights_created"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
