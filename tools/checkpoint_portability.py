from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.checkpoint.portability import (
    consume_checkpoint_portability_bundle,
    produce_checkpoint_portability_bundle,
    validate_consumer_report,
    validate_producer_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce or consume D05 S0 checkpoint portability evidence."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    produce = subparsers.add_parser(
        "produce",
        help="create a verified x86_64 checkpoint bundle",
    )
    produce.add_argument("--repo-root", type=Path, default=Path("."))
    produce.add_argument("--source-sha", required=True)
    produce.add_argument("--output-dir", type=Path, required=True)
    produce.add_argument("--require-architecture", default="x86_64")

    consume = subparsers.add_parser(
        "consume",
        help="verify and restore a producer bundle on a second architecture",
    )
    consume.add_argument("--repo-root", type=Path, default=Path("."))
    consume.add_argument("--source-sha", required=True)
    consume.add_argument("--bundle-dir", type=Path, required=True)
    consume.add_argument("--output", type=Path, required=True)
    consume.add_argument("--require-architecture", default="aarch64")

    validate_producer = subparsers.add_parser(
        "validate-producer",
        help="validate one producer.json report",
    )
    validate_producer.add_argument("report", type=Path)

    validate_consumer = subparsers.add_parser(
        "validate-consumer",
        help="validate one consumer report against its producer report",
    )
    validate_consumer.add_argument("--producer", type=Path, required=True)
    validate_consumer.add_argument("--consumer", type=Path, required=True)
    return parser


def _load_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "produce":
        report = produce_checkpoint_portability_bundle(
            args.repo_root,
            source_sha=args.source_sha,
            output_dir=args.output_dir,
            require_architecture=args.require_architecture,
        )
        result = {
            "status": "PASS",
            "schema_version": report["schema_version"],
            "source_sha": report["source"]["source_sha"],
            "architecture": report["source"]["architecture"],
            "checkpoint_id": report["checkpoint"]["checkpoint_id"],
            "report_sha256": report["report_sha256"],
            "output_dir": str(args.output_dir),
        }
    elif args.command == "consume":
        report = consume_checkpoint_portability_bundle(
            args.repo_root,
            source_sha=args.source_sha,
            bundle_dir=args.bundle_dir,
            output=args.output,
            require_architecture=args.require_architecture,
        )
        result = {
            "status": "PASS",
            "schema_version": report["schema_version"],
            "source_sha": report["source"]["source_sha"],
            "architecture": report["source"]["architecture"],
            "producer_architecture": report["producer"]["architecture"],
            "checkpoint_id": report["checkpoint"]["checkpoint_id"],
            "report_sha256": report["report_sha256"],
            "output": str(args.output),
        }
    elif args.command == "validate-producer":
        report = _load_object(args.report)
        result = validate_producer_report(report)
    else:
        producer = _load_object(args.producer)
        consumer = _load_object(args.consumer)
        result = validate_consumer_report(consumer, producer=producer)

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
