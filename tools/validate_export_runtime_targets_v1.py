from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.export_runtime_parity import load_target_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the D07 export runtime target manifest")
    parser.add_argument(
        "manifest",
        nargs="?",
        default="configs/research/export_runtime_targets_v1.json",
    )
    args = parser.parse_args()
    manifest = load_target_manifest(Path(args.manifest))
    result = {
        "result": "VALID_CANDIDATE_MANIFEST",
        "manifest_identity": manifest["manifest_identity"],
        "targets": [
            {"id": target["id"], "candidate_state": target["candidate_state"]}
            for target in manifest["targets"]
        ],
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
