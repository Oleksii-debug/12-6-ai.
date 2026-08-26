from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import time
import tracemalloc
from pathlib import Path

from twelve_six.data.external_sources import (
    RIGHTS_REVIEW_REQUIRED,
    ExternalSourceSpec,
    RightsDecision,
    SnapshotSpec,
    build_external_source_registry,
)
from twelve_six.data.source_acquisition import (
    plan_from_registered_source,
    verify_and_stage_local_mirror,
)


def benchmark(size_mib: int, chunk_mib: int) -> dict[str, object]:
    if size_mib <= 0 or chunk_mib <= 0:
        raise ValueError("size_mib and chunk_mib must be positive")
    size_bytes = size_mib * 1024 * 1024
    chunk_size_bytes = chunk_mib * 1024 * 1024
    if size_bytes % chunk_size_bytes:
        raise ValueError("size_mib must be divisible by chunk_mib")
    pattern = bytes(range(256)) * (chunk_size_bytes // 256)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_path = root / "source.bin"
        digest = hashlib.sha256()
        with source_path.open("wb") as handle:
            for _ in range(size_bytes // chunk_size_bytes):
                handle.write(pattern)
                digest.update(pattern)

        snapshot = SnapshotSpec(
            uri="https://example.invalid/synthetic/snapshots/v1/source.bin",
            sha256=digest.hexdigest(),
            size_bytes=size_bytes,
            retrieved_at="2026-08-24T00:00:00Z",
            upstream_version="v1",
            retrieval_method="synthetic_local_fixture",
        )
        rights = RightsDecision(
            status=RIGHTS_REVIEW_REQUIRED,
            license_id="NOASSERTION",
            terms_url="https://example.invalid/terms",
            allows_model_training=False,
            allows_derivatives=False,
            allows_redistribution=False,
            policy_ref="policy://benchmark/unreviewed",
            reviewed_at="2026-08-24T00:00:00Z",
            reviewer_ref="role://synthetic-benchmark",
        )
        source = ExternalSourceSpec(
            source_id="synthetic-source",
            source_version="v1",
            provider="local-synthetic",
            source_url="https://example.invalid/synthetic",
            source_kind="binary_fixture",
            purpose="pretraining",
            synthetic=True,
            benchmark_material=False,
            held_out=False,
            snapshot=snapshot,
            rights=rights,
        )
        registry = build_external_source_registry([source])
        plan = plan_from_registered_source(
            registry,
            "synthetic-source",
            "v1",
            (root / "staged.bin").resolve().as_uri(),
            chunk_size_bytes=chunk_size_bytes,
        )

        tracemalloc.start()
        started = time.perf_counter()
        receipt = verify_and_stage_local_mirror(plan, source_path)
        elapsed = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "schema_version": "12-6.source-acquisition-benchmark.v1",
            "evidence_scope": "LOCAL_FREE_SYNTHETIC_SOURCE_ACQUISITION_ONLY",
            "bytes": size_bytes,
            "mib": size_bytes / (1024 * 1024),
            "chunk_size_bytes": chunk_size_bytes,
            "chunk_count": receipt.chunk_count,
            "elapsed_seconds": elapsed,
            "mib_per_second": (size_bytes / (1024 * 1024)) / elapsed,
            "peak_tracemalloc_bytes": peak_bytes,
            "peak_to_chunk_ratio": peak_bytes / chunk_size_bytes,
            "verified_sha256_match": receipt.verified_sha256 == digest.hexdigest(),
            "plan_sha256": plan.manifest()["plan_sha256"],
            "receipt_sha256": receipt.manifest()["receipt_sha256"],
            "rights_status_observed": receipt.rights_status_observed,
            "training_eligibility_evaluated": False,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "compute_class": "LOCAL_FREE",
            "paid_cost_usd": 0,
            "remote_fsspec_execution": "NOT_TESTED",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size-mib", type=int, default=32)
    parser.add_argument("--chunk-mib", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(args.size_mib, args.chunk_mib)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
