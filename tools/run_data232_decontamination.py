"""CLI for DATA-232 decontamination authority V2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.decontamination_authority_v2 import (
    build_blocker_report,
    build_report,
    verify_report,
    write_immutable_report,
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: JSONL row must be an object")
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("scan", "blocker", "verify"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--authorities", type=Path)
    parser.add_argument("--training-jsonl", type=Path)
    parser.add_argument("--evaluation-jsonl", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.action == "verify":
        verify_report(_json(args.report))
        print(json.dumps({"status": "PASS", "report": str(args.report)}, sort_keys=True))
        return 0
    if args.authorities is None:
        parser.error("blocker/scan requires --authorities")
    authorities = _json(args.authorities)
    if args.action == "blocker":
        report = build_blocker_report(
            authorities,
            reason="No terminal DATA-230 corpus identity/inventory is published at this worker cutoff.",
        )
    else:
        if args.config is None or args.training_jsonl is None or args.evaluation_jsonl is None:
            parser.error("scan requires --config, --training-jsonl, and --evaluation-jsonl")
        config = _json(args.config)
        report = build_report(
            _jsonl(args.training_jsonl),
            _jsonl(args.evaluation_jsonl),
            training_corpus_identity=config["training_corpus_identity"],
            selection_validation_identity=config["selection_validation_identity"],
            final_test_identity=config["final_test_identity"],
            authorities=authorities,
            thresholds=config.get("thresholds"),
            quarantine_cross_source_families=bool(config.get("quarantine_cross_source_families", True)),
        )
    write_immutable_report(args.report, report)
    print(json.dumps({"status": report["status"], "report_sha256": report["report_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
