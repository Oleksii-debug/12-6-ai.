from __future__ import annotations

import argparse
import json
import platform
import time
import tracemalloc
from collections import Counter
from importlib.util import find_spec
from pathlib import Path

from twelve_six.data import ExactDedupPlan, StreamingShardPlan


def benchmark(records: int, shard_count: int, partitions: int) -> dict[str, object]:
    if records <= 0:
        raise ValueError("records must be positive")
    shard_plan = StreamingShardPlan(corpus_identity_sha256="1" * 64, shard_count=shard_count)
    dedup_plan = ExactDedupPlan(
        corpus_identity_sha256="1" * 64,
        input_uri="file:///synthetic/input",
        survivor_uri="file:///synthetic/survivors",
        duplicate_uri="file:///synthetic/duplicates",
        partitions=partitions,
    )

    tracemalloc.start()
    started = time.perf_counter()
    shard_counts: Counter[int] = Counter()
    partition_counts: Counter[int] = Counter()
    for index in range(records):
        record_id = f"synthetic-{index:09d}"
        shard_counts[shard_plan.shard_for_record_id(record_id)] += 1
        content_sha256 = __import__("hashlib").sha256(record_id.encode()).hexdigest()
        partition_counts[dedup_plan.partition_for(content_sha256)] += 1
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "schema_version": "12-6.corpus-foundation-benchmark.v1",
        "evidence_scope": "LOCAL_FREE_SYNTHETIC_MECHANICS_ONLY",
        "records": records,
        "shard_count": shard_count,
        "exact_dedup_partitions": partitions,
        "elapsed_seconds": elapsed,
        "records_per_second": records / elapsed,
        "peak_tracemalloc_bytes": peak_bytes,
        "peak_tracemalloc_bytes_per_record": peak_bytes / records,
        "shard_min_records": min(shard_counts.values()),
        "shard_max_records": max(shard_counts.values()),
        "partition_min_records": min(partition_counts.values()),
        "partition_max_records": max(partition_counts.values()),
        "shard_plan_sha256": shard_plan.manifest()["plan_sha256"],
        "exact_dedup_plan_sha256": dedup_plan.manifest()["plan_sha256"],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "paid_cost_usd": 0,
        "datatrove_installed": find_spec("datatrove") is not None,
        "datatrove_minhash_execution": "NOT_EXECUTED_BY_THIS_MECHANICS_BENCHMARK",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=100_000)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--partitions", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.records, args.shards, args.partitions)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
