#!/usr/bin/env python3
"""Differentially verify indexed execution against an exact incumbent V3 runtime."""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - non-POSIX local verifier fallback
    resource = None

from twelve_six.data.incumbent_dedup_indexed_execution import (
    DEFAULT_MAX_INDEX_POSTINGS,
    DEFAULT_MAX_PAIR_EXPANSIONS,
    attest_incumbent_runtime,
    audit_payloads_indexed,
    candidate_pair_indices_with_stats,
    execution_stats,
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _max_rss_kib() -> int | None:
    if resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value // 1024 if value > 10_000_000 else value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-module", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--payload-map", type=Path, required=True)
    parser.add_argument("--max-candidate-pairs", type=int, default=5_000_000)
    parser.add_argument("--max-index-postings", type=int, default=DEFAULT_MAX_INDEX_POSTINGS)
    parser.add_argument("--max-pair-expansions", type=int, default=DEFAULT_MAX_PAIR_EXPANSIONS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    v3 = importlib.import_module(args.v3_module)
    attest_incumbent_runtime(v3)
    inventory = _json(args.inventory)
    mapping = _json(args.payload_map)
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in mapping.items()
    ):
        raise SystemExit("payload map must be a JSON object of source_id -> file path")
    payloads = {key: Path(value).read_bytes() for key, value in mapping.items()}

    reference = v3.audit_payloads(inventory, payloads)
    indexed = audit_payloads_indexed(
        v3,
        inventory,
        payloads,
        max_candidate_pairs=args.max_candidate_pairs,
        max_index_postings=args.max_index_postings,
        max_pair_expansions=args.max_pair_expansions,
    )
    if v3.v1._canonical_bytes(reference) != v3.v1._canonical_bytes(indexed):
        raise SystemExit("indexed report differs from incumbent all-pairs report")

    rows, _ = v3._validate_inventory(inventory)
    validated = v3.v1._validate_inventory(v3._as_v1_inventory(rows))
    fingerprints = [v3._fingerprint(row, payloads[row["source_id"]]) for row in validated]
    started = time.perf_counter()
    pairs, work = candidate_pair_indices_with_stats(
        v3.v1,
        fingerprints,
        max_candidate_pairs=args.max_candidate_pairs,
        max_index_postings=args.max_index_postings,
        max_pair_expansions=args.max_pair_expansions,
    )
    index_wall_clock_seconds = time.perf_counter() - started
    result = {
        "schema_version": "12-6.d03-incumbent-dedup-indexed-equivalence.v2",
        "local_free_only": True,
        "model_training_executed": False,
        "canonical_capacity_credit": 0,
        "reports_byte_identical": True,
        "reference_report_sha256": reference["report_sha256"],
        "indexed_report_sha256": indexed["report_sha256"],
        "index_wall_clock_seconds": round(index_wall_clock_seconds, 6),
        "process_max_rss_kib": _max_rss_kib(),
        "work_limits": {
            "max_candidate_pairs": args.max_candidate_pairs,
            "max_index_postings": args.max_index_postings,
            "max_pair_expansions": args.max_pair_expansions,
        },
        **execution_stats(
            len(fingerprints),
            len(pairs),
            index_postings=work["index_postings"],
            pair_expansion_attempts=work["pair_expansion_attempts"],
            unique_bucket_signatures=work["unique_bucket_signatures"],
        ),
    }
    rendered = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
