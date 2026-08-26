from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.real_corpus_holdout import build_immutable_holdout


def _rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build immutable EVAL-131 UA/EN/code holdouts")
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--evaluation-corpus-sha256", required=True)
    parser.add_argument("--benchmark-registry-sha256", required=True)
    parser.add_argument("--reference-bundle-sha256", required=True)
    parser.add_argument("--decontamination-report-sha256", required=True)
    args = parser.parse_args(argv)
    result = build_immutable_holdout(
        _rows(args.input_jsonl),
        args.output_dir,
        suite_name=args.suite_name,
        evaluation_corpus_identity_sha256=args.evaluation_corpus_sha256,
        benchmark_registry_sha256=args.benchmark_registry_sha256,
        decontamination_reference_bundle_sha256=args.reference_bundle_sha256,
        decontamination_report_sha256=args.decontamination_report_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
