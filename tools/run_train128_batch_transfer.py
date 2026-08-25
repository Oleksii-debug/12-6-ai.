"""Run TRAIN-128 matched-token effective-batch transfer on the TRAIN-53 real corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.batch_transfer import run_batch_transfer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--locked-environment-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    locked = json.loads(args.locked_environment_evidence.read_text(encoding="utf-8"))
    report = run_batch_transfer(
        Path.cwd(),
        source_sha=args.source_sha,
        locked_environment_evidence=locked,
        seed=args.seed,
        torch_threads=args.torch_threads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "authority": report["authority"],
        "source_sha": report["identity"]["source_sha"],
        "selected_500k": report["models"]["500k"]["selected_by_0_5pct_quality_rule"],
        "selected_1m": report["models"]["1m"]["selected_by_0_5pct_quality_rule"],
        "heuristic": report["heuristic"],
        "report_sha256": report["report_sha256"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
