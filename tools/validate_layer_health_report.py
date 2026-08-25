from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.layer_health import validate_layer_health_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TRAIN-54 layer-health report")
    parser.add_argument("report", type=Path)
    parser.add_argument("--expected-source-sha")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("layer-health report must be a JSON object")
    validate_layer_health_report(report, expected_source_sha=args.expected_source_sha)
    print(f"{report['schema_version']}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
