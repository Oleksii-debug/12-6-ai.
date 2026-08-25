"""LOCAL_FREE synthetic mechanics benchmark for D03/D09 corpus foundations."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import time
import tracemalloc
from pathlib import Path

from twelve_six.data.corpus_foundation import (
    DataTroveDedupPlan,
    SQLiteExactDedupIndex,
    ShardArtifact,
    StreamingShardPlan,
    build_resume_manifest,
)

REGISTRY = "a" * 64
RESERVED = "b" * 64


def run(record_count: int, duplicate_every: int) -> dict[str, object]:
    if record_count <= 0 or duplicate_every <= 1:
        raise ValueError("record_count must be >0 and duplicate_every must be >1")
    plan = StreamingShardPlan(
        REGISTRY,
        RESERVED,
        "file:///synthetic/shards",
        "d09-local-free-v1",
        32,
        256,
        8 * 1024 * 1024,
    )
    dedup_plan = DataTroveDedupPlan(
        REGISTRY,
        RESERVED,
        "file:///synthetic/input",
        "file:///synthetic/dedup",
        "file:///synthetic/logs",
        tasks=32,
        workers=4,
    )
    unique = 0
    duplicates = 0
    shard_counts = [0] * plan.shard_count
    tracemalloc.start()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        with SQLiteExactDedupIndex(Path(tmp) / "exact.sqlite3") as index:
            for position in range(record_count):
                logical = position - 1 if position and position % duplicate_every == 0 else position
                record_id = f"synthetic-{position:09d}"
                fingerprint = hashlib.sha256(f"payload-{logical}".encode()).hexdigest()
                if index.seen_or_add(fingerprint):
                    duplicates += 1
                    continue
                unique += 1
                shard_counts[plan.assign(record_id)] += 1
            index.commit()
        db_bytes = (Path(tmp) / "exact.sqlite3").stat().st_size
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    artifacts = [
        ShardArtifact(
            shard_index=index,
            part_index=0,
            uri=f"file:///synthetic/shards/{index:05d}-00000.parquet",
            sha256=hashlib.sha256(f"shard-{index}".encode()).hexdigest(),
            size_bytes=count * 100,
            records=count,
        )
        for index, count in enumerate(shard_counts)
    ]
    resume = build_resume_manifest(
        plan,
        artifacts,
        dedup_plan_sha256=dedup_plan.manifest()["plan_sha256"],
    )
    core = {
        "schema_version": "12-6.corpus-foundation-local-benchmark.v1",
        "execution_class": "LOCAL_FREE_SYNTHETIC",
        "record_count": record_count,
        "duplicate_every": duplicate_every,
        "unique_records": unique,
        "duplicates_removed": duplicates,
        "elapsed_seconds": round(elapsed, 6),
        "records_per_second": round(record_count / elapsed, 2),
        "peak_tracemalloc_bytes": peak_bytes,
        "sqlite_bytes": db_bytes,
        "shard_count": plan.shard_count,
        "min_unique_records_per_shard": min(shard_counts),
        "max_unique_records_per_shard": max(shard_counts),
        "streaming_plan_sha256": plan.manifest()["plan_sha256"],
        "dedup_plan_sha256": dedup_plan.manifest()["plan_sha256"],
        "resume_manifest_sha256": resume["resume_manifest_sha256"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scope_note": (
            "Mechanics-only synthetic benchmark. Not corpus quality, DataTrove MinHash throughput, "
            "remote fsspec throughput, or paid/GPU evidence."
        ),
    }
    digest = hashlib.sha256(
        (json.dumps(core, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    return {**core, "report_sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=25_000)
    parser.add_argument("--duplicate-every", type=int, default=11)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.records, args.duplicate_every)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
