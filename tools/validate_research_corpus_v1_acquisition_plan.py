from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.corpus_acquisition_plan import load_and_validate_acquisition_plan

DEFAULT_PLAN = Path("configs/data/research_corpus_v1_scalable_acquisition_plan_v1.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Research Corpus V1 acquisition planning without granting capacity."
    )
    parser.add_argument("plan", nargs="?", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    summary = load_and_validate_acquisition_plan(args.plan)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
