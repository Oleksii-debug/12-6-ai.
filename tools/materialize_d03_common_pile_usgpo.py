from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twelve_six.data.common_pile_usgpo_intake import materialize_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a bounded, zero-credit USGPO candidate window."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/data/d03_common_pile_usgpo_intake_v1.json",
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = materialize_path(config, args.source, args.output_dir)
    summary = {
        "candidate_bytes": manifest["artifacts"]["candidate_jsonl"]["bytes"],
        "candidate_sha256": manifest["artifacts"]["candidate_jsonl"]["sha256"],
        "manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "source_transport_verified": manifest["source_transport"]["verified"],
        "status": manifest["status"],
        "training_authorized_bytes": manifest["training_authorized_bytes"],
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
