from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twelve_six.data.external_sources import (  # noqa: E402
    ExternalSourceSpec,
    ReservedSetSpec,
    RightsDecision,
    SnapshotSpec,
    build_external_source_registry,
    build_reserved_fingerprint_registry,
)
from twelve_six.data.pretraining_factory import (  # noqa: E402
    FactoryPlan,
    finalize_jsonl_and_tokenizer_input,
    prepare_exact_stage,
    run_local_fixture_near_dedup,
    validate_datatrove_runtime,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _build_rows(count: int) -> tuple[list[dict[str, str]], str]:
    rows: list[dict[str, str]] = []
    prior: list[str] = []
    for index in range(count):
        text = (
            f"Document {index} contains deliberately varied English synthetic training text "
            f"with stable provenance. Topic {index % 97} group {index % 31} sequence {index}. "
            "This controlled record exists only for LOCAL_FREE data-factory mechanics evidence."
        )
        if index % 29 == 0 and prior:
            text = prior[-1]
        elif index % 41 == 0 and prior:
            text = prior[-1].replace("mechanics evidence", "mechanics evidence revised")
        prior.append(text)
        rows.append({"document_id": f"doc-{index:06d}", "text": text, "language_hint": "en"})
    contamination = (
        "A synthetic reserved benchmark sentence is injected to prove exact normalized "
        "decontamination before deduplication in the LOCAL_FREE factory benchmark."
    )
    rows.append({"document_id": "reserved", "text": contamination, "language_hint": "en"})
    rows.append(
        {
            "document_id": "pii",
            "text": (
                "A synthetic PII test record contains benchmark@example.com and must be rejected "
                "before it can enter exact deduplication or any tokenizer handoff."
            ),
            "language_hint": "en",
        }
    )
    return rows, contamination


def _build_inputs(work: Path, count: int):
    rows, contamination = _build_rows(count)
    raw = work / "raw.jsonl"
    raw_bytes = b"".join(_canonical(row) for row in rows)
    raw.write_bytes(raw_bytes)
    registry = build_external_source_registry(
        [
            ExternalSourceSpec(
                source_id="local-synthetic",
                source_version="v1",
                provider="12-6-local-benchmark",
                source_url="https://example.invalid/local-synthetic",
                source_kind="jsonl_text_v1",
                purpose="pretraining",
                synthetic=True,
                benchmark_material=False,
                held_out=False,
                snapshot=SnapshotSpec(
                    uri=raw.as_uri(),
                    sha256=_sha(raw_bytes),
                    size_bytes=len(raw_bytes),
                    retrieved_at="2026-08-25T00:00:00Z",
                    upstream_version="v1",
                    retrieval_method="local-generated-fixture",
                ),
                rights=RightsDecision(
                    status="APPROVED_FOR_TRAINING",
                    license_id="CC0-1.0",
                    terms_url="https://example.invalid/local-synthetic-terms",
                    allows_model_training=True,
                    allows_derivatives=True,
                    allows_redistribution=True,
                    policy_ref="LOCAL_FREE_TEST_FIXTURE_ONLY",
                    reviewed_at="2026-08-25T00:00:00Z",
                    reviewer_ref="benchmark-generator",
                ),
            )
        ]
    )
    registry_id = registry["registry_identity_sha256"]
    receipt = {
        "schema_version": "12-6.source-retrieval-receipt.v1",
        "source_registry_identity_sha256": registry_id,
        "source_id": "local-synthetic",
        "source_version": "v1",
        "destination_uri": raw.as_uri(),
        "expected_sha256": _sha(raw_bytes),
        "verified_sha256": _sha(raw_bytes),
        "verified_size_bytes": len(raw_bytes),
        "verification": "PASS",
        "training_eligibility_evaluated": False,
    }
    inventory_core = {
        "schema_version": "12-6.source-retrieval-inventory.v1",
        "source_registry_identity_sha256": registry_id,
        "receipts": [receipt],
        "rights_semantics": "INVENTORY_IS_NOT_TRAINING_APPROVAL",
    }
    inventory = {**inventory_core, "inventory_sha256": _sha(_canonical(inventory_core))}
    reserved = build_reserved_fingerprint_registry(
        [
            ReservedSetSpec(
                set_id="local-reserved",
                version="v1",
                source_id="benchmark-fixture",
                purpose="benchmark",
                normalized_sha256=(_sha(contamination.encode("utf-8")),),
            )
        ]
    )
    return rows, registry, inventory, reserved


def main(record_count: int = 1_200) -> None:
    work = ROOT / "_local_benchmark"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    rows, registry, inventory, reserved = _build_inputs(work, record_count)
    output = work / "out"
    output.mkdir()
    plan = FactoryPlan(
        source_registry_sha256=registry["registry_identity_sha256"],
        retrieval_inventory_sha256=inventory["inventory_sha256"],
        reserved_registry_sha256=reserved["registry_identity_sha256"],
        output_uri=output.as_uri(),
        shard_count=8,
        max_records_in_memory=128,
    )
    tracemalloc.start()
    started = time.perf_counter()
    exact = prepare_exact_stage(plan, registry, inventory, reserved)
    exact_seconds = time.perf_counter() - started
    started = time.perf_counter()
    near = run_local_fixture_near_dedup(plan, exact["records_uri"])
    near_seconds = time.perf_counter() - started
    started = time.perf_counter()
    final = finalize_jsonl_and_tokenizer_input(
        plan, near["output_uri"], near_dedup_evidence=near
    )
    final_seconds = time.perf_counter() - started
    elapsed = exact_seconds + near_seconds + final_seconds
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    resumed_exact = prepare_exact_stage(plan, registry, inventory, reserved)
    resumed_final = finalize_jsonl_and_tokenizer_input(
        plan, near["output_uri"], near_dedup_evidence=near
    )
    try:
        validate_datatrove_runtime()
        datatrove_runtime = "AVAILABLE"
    except RuntimeError:
        datatrove_runtime = "NOT_INSTALLED"
    report = {
        "schema_version": "12-6.pretraining-data-factory-local-benchmark.v1",
        "execution_class": "LOCAL_FREE_SYNTHETIC",
        "input_documents": len(rows),
        "exact_unique_records": exact["record_count"],
        "exact_duplicates_removed": exact["counters"].get("exact_duplicates_removed", 0),
        "benchmark_contamination_rejected": exact["counters"].get(
            "benchmark_contamination_rejected", 0
        ),
        "pii_rejected": exact["counters"].get("rejected_pii_email", 0),
        "fixture_near_duplicates_removed": near["removed"],
        "final_documents": final["documents"],
        "train_documents": final["split_documents"]["train"],
        "validation_documents": final["split_documents"]["validation"],
        "elapsed_seconds": round(elapsed, 6),
        "exact_seconds": round(exact_seconds, 6),
        "near_seconds": round(near_seconds, 6),
        "final_seconds": round(final_seconds, 6),
        "records_per_second": round(len(rows) / elapsed, 2),
        "peak_tracemalloc_bytes": peak,
        "exact_stage_resumed": resumed_exact["resumed"],
        "final_stage_resumed": resumed_final["resumed"],
        "fsspec_version": __import__("fsspec").__version__,
        "datatrove_runtime": datatrove_runtime,
        "production_near_dedup_executed": False,
        "parquet_executed": False,
        "truth_boundary": (
            "Fixture near-dedup reuses bounded S0 Jaccard only. DataTrove 0.10.0 and Parquet "
            "remain optional production backends and were not installed for retained evidence."
        ),
    }
    report["report_sha256"] = _sha(_canonical(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
