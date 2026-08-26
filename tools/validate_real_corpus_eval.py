from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.real_corpus_evaluation import verify_ladder_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate an EVAL-131 ladder report")
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    value = json.loads(args.report.read_text(encoding="utf-8"))
    digest = verify_ladder_report(value)
    print(json.dumps({"status": "PASS", "report_sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
