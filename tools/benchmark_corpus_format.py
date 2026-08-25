"""LOCAL_FREE PERF-146 corpus physical-format benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from twelve_six.data.corpus_format import (
    PYARROW_EXPERIMENT_VERSION,
    canonical_bytes,
    iter_layout_rows,
    iter_training_records,
    layout_data_paths,
    materialize_layout,
    packed_training_trace,
    seek_layout_row,
    sha256_file,
    verify_layout,
)
from twelve_six.data.corpus_v01 import build_corpus
from twelve_six.data.shard_scale import write_scale_fixture
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing.core import batch_examples, collate_rows, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

REPORT_SCHEMA = "12-6.perf146-corpus-format-benchmark.v1"
MIB = 1024 * 1024


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * 1024 if value < 10**10 else value)


def _drop_page_cache_hint(paths: tuple[Path, ...]) -> dict[str, Any]:
    supported = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")
    attempted = 0
    succeeded = 0
    if hasattr(os, "sync"):
        os.sync()
    if supported:
        for path in paths:
            attempted += 1
            with path.open("rb") as handle:
                try:
                    os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    continue
                succeeded += 1
    return {
        "method": "posix_fadvise_dontneed_best_effort" if supported else "unavailable",
        "attempted_files": attempted,
        "successful_hints": succeeded,
        "kernel_drop_caches_privileged": False,
    }


def _consume_layout(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    documents = 0
    text_bytes = 0
    for row in iter_layout_rows(root, manifest):
        documents += 1
        text = row["text"]
        if not isinstance(text, str):
            raise TypeError("text must remain a string")
        text_bytes += len(text.encode("utf-8"))
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    return {
        "documents": documents,
        "logical_text_bytes": text_bytes,
        "wall_seconds": wall,
        "logical_mib_per_s": (text_bytes / MIB) / wall if wall else math.inf,
        "process_cpu_seconds": cpu,
        "process_cpu_core_percent": (cpu / wall * 100.0) if wall else 0.0,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _seek_samples(manifest: dict[str, Any], *, limit: int = 12) -> list[tuple[int, int]]:
    samples: list[tuple[int, int]] = []
    rng = random.Random(146)
    shards = manifest["shards"]
    for shard_index, shard in enumerate(shards):
        documents = int(shard["documents"])
        if documents <= 0:
            continue
        anchors = {0, documents // 2, documents - 1, rng.randrange(documents)}
        for ordinal in sorted(anchors):
            samples.append((shard_index, ordinal))
    rng.shuffle(samples)
    return samples[:limit]


def _measure_seek(
    root: Path,
    manifest: dict[str, Any],
    *,
    coldish: bool,
) -> dict[str, Any]:
    values: list[float] = []
    entries = manifest["shards"]
    for shard_index, ordinal in _seek_samples(manifest):
        if coldish:
            path = root / entries[shard_index]["relative_path"]
            _drop_page_cache_hint((path,))
        start = time.perf_counter()
        row = seek_layout_row(
            root,
            manifest,
            shard_index=shard_index,
            record_ordinal=ordinal,
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        if not isinstance(row.get("text"), str):
            raise AssertionError("seek returned an invalid row")
        values.append(elapsed)
    return {
        "samples": len(values),
        "median_ms": statistics.median(values) if values else 0.0,
        "p95_ms": _percentile(values, 0.95),
        "max_ms": max(values, default=0.0),
        "cache_state": "best_effort_cold" if coldish else "warm",
    }


def _tensor_batches(
    root: Path,
    manifest: dict[str, Any],
    *,
    sequence_length: int,
    batch_size: int,
):
    examples = iter_packed_examples(
        iter_training_records(root, manifest),
        ByteTokenizer(),
        expected_split="train",
        sequence_length=sequence_length,
    )
    for group in batch_examples(examples, batch_size=batch_size):
        rows = collate_rows(group, target_mode="target_ids")
        yield {
            name: torch.tensor(values, dtype=torch.long)
            for name, values in rows.items()
            if name in {"input_ids", "target_ids", "loss_mask"}
        }


def _measure_trainer(
    root: Path,
    manifest: dict[str, Any],
    *,
    stage_config: Path,
    steps: int,
    sequence_length: int,
    batch_size: int,
) -> dict[str, Any] | None:
    if steps <= 0:
        return None
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    stage = load_stage_config(stage_config)
    torch.manual_seed(146)
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(
        model,
        TrainerConfig(
            max_steps=steps,
            learning_rate=1e-4,
            weight_decay=0.0,
            gradient_clip_norm=None,
            seed=146,
        ),
    )
    batches = iter(
        _tensor_batches(
            root,
            manifest,
            sequence_length=sequence_length,
            batch_size=batch_size,
        )
    )
    data_wait = 0.0
    train_compute = 0.0
    tokens = 0
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    for _ in range(steps):
        next_start = time.perf_counter()
        batch = next(batches)
        data_wait += time.perf_counter() - next_start
        train_start = time.perf_counter()
        metrics = trainer.train_microbatch(batch)
        train_compute += time.perf_counter() - train_start
        tokens += metrics.tokens
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    return {
        "model_stage": stage.stage,
        "model_parameters": stage.model.parameter_count(),
        "tokenizer": "s0-byte-v1",
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "steps": steps,
        "loss_tokens": tokens,
        "wall_seconds": wall,
        "tokens_per_s": tokens / wall if wall else math.inf,
        "data_wait_seconds": data_wait,
        "data_wait_percent": (100.0 * data_wait / wall) if wall else 0.0,
        "train_compute_seconds": train_compute,
        "process_cpu_seconds": cpu,
        "process_cpu_core_percent": (cpu / wall * 100.0) if wall else 0.0,
    }


def _worker_measure(args: argparse.Namespace) -> None:
    root = Path(args.layout_root)
    manifest = verify_layout(root)
    paths = layout_data_paths(root, manifest)
    cache = _drop_page_cache_hint(paths)
    cold = _consume_layout(root, manifest)
    warm = _consume_layout(root, manifest)
    seek_warm = _measure_seek(root, manifest, coldish=False)
    seek_cold = _measure_seek(root, manifest, coldish=True)
    trainer = _measure_trainer(
        root,
        manifest,
        stage_config=Path(args.stage_config),
        steps=args.trainer_steps,
        sequence_length=args.trainer_sequence_length,
        batch_size=args.trainer_batch_size,
    )
    payload = {
        "layout_identity_sha256": manifest["layout_identity_sha256"],
        "physical_format": manifest["physical_format"],
        "compression": manifest["compression"],
        "data_bytes": manifest["data_bytes"],
        "cold_cache_control": cache,
        "cold_read": cold,
        "warm_read": warm,
        "restart_seek_warm": seek_warm,
        "restart_seek_best_effort_cold": seek_cold,
        "trainer": trainer,
        "peak_rss_bytes": _rss_bytes(),
    }
    print(json.dumps(payload, sort_keys=True))


def _run_worker(
    script: Path,
    root: Path,
    *,
    stage_config: Path,
    trainer_steps: int,
    trainer_sequence_length: int,
    trainer_batch_size: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--layout-root",
        str(root),
        "--stage-config",
        str(stage_config),
        "--trainer-steps",
        str(trainer_steps),
        "--trainer-sequence-length",
        str(trainer_sequence_length),
        "--trainer-batch-size",
        str(trainer_batch_size),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def _materialize_candidates(
    source_root: Path,
    work_root: Path,
    *,
    dataset_name: str,
) -> dict[str, dict[str, Any]]:
    specs = (
        ("jsonl", "jsonl", "none"),
        ("parquet_none", "parquet", "none"),
        ("parquet_zstd", "parquet", "zstd"),
    )
    results: dict[str, dict[str, Any]] = {}
    for name, physical_format, compression in specs:
        target = work_root / dataset_name / name
        target.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        manifest = materialize_layout(
            source_root,
            target,
            physical_format=physical_format,
            compression=compression,
            row_group_rows=4096,
            expected_pyarrow_version=(
                PYARROW_EXPERIMENT_VERSION if physical_format == "parquet" else None
            ),
        )
        build_seconds = time.perf_counter() - start
        results[name] = {
            "root": str(target),
            "build_seconds": build_seconds,
            "manifest": manifest,
        }
    return results


def _assert_equivalent_layouts(candidates: dict[str, dict[str, Any]]) -> dict[str, Any]:
    logical = {
        item["manifest"]["source_logical_identity_sha256"] for item in candidates.values()
    }
    row_traces = {
        item["manifest"]["logical_trace"]["ordered_logical_rows_sha256"]
        for item in candidates.values()
    }
    identity_traces = {
        item["manifest"]["logical_trace"]["ordered_record_identity_sha256"]
        for item in candidates.values()
    }
    if len(logical) != 1 or len(row_traces) != 1 or len(identity_traces) != 1:
        raise AssertionError("physical layouts changed logical corpus identity/order")
    token_traces: dict[str, dict[str, Any]] = {}
    for name, item in candidates.items():
        token_traces[name] = packed_training_trace(Path(item["root"]), item["manifest"])
    trace_hashes = {
        item["packed_training_trace_sha256"] for item in token_traces.values()
    }
    examples = {item["examples"] for item in token_traces.values()}
    loss_tokens = {item["loss_tokens"] for item in token_traces.values()}
    if len(trace_hashes) != 1 or len(examples) != 1 or len(loss_tokens) != 1:
        raise AssertionError("physical layouts changed tokenized training sequence trace")
    return {
        "source_logical_identity_sha256": next(iter(logical)),
        "ordered_logical_rows_sha256": next(iter(row_traces)),
        "ordered_record_identity_sha256": next(iter(identity_traces)),
        "packed_training_trace_sha256": next(iter(trace_hashes)),
        "training_examples": next(iter(examples)),
        "training_loss_tokens": next(iter(loss_tokens)),
        "all_layouts_equivalent": True,
    }


def _dataset_report(
    source_root: Path,
    work_root: Path,
    *,
    name: str,
    script: Path,
    stage_config: Path,
    trainer_steps: int,
) -> dict[str, Any]:
    candidates = _materialize_candidates(source_root, work_root, dataset_name=name)
    equivalence = _assert_equivalent_layouts(candidates)
    measured: dict[str, Any] = {}
    for candidate_name, item in candidates.items():
        observation = _run_worker(
            script,
            Path(item["root"]),
            stage_config=stage_config,
            trainer_steps=trainer_steps,
            trainer_sequence_length=64,
            trainer_batch_size=2,
        )
        observation["build_seconds"] = item["build_seconds"]
        observation["layout_manifest_sha256"] = sha256_file(
            Path(item["root"]) / "manifest.json"
        )
        measured[candidate_name] = observation
    return {"equivalence": equivalence, "candidates": measured}


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _decision(real: dict[str, Any]) -> dict[str, Any]:
    candidates = real["candidates"]
    jsonl = candidates["jsonl"]
    zstd = candidates["parquet_zstd"]
    none = candidates["parquet_none"]
    trainer_jsonl = jsonl.get("trainer")
    trainer_zstd = zstd.get("trainer")
    trainer_ratio = None
    if trainer_jsonl and trainer_zstd:
        trainer_ratio = _ratio(
            trainer_zstd["tokens_per_s"], trainer_jsonl["tokens_per_s"]
        )
    zstd_disk_ratio = _ratio(zstd["data_bytes"], jsonl["data_bytes"])
    zstd_warm_ratio = _ratio(
        zstd["warm_read"]["logical_mib_per_s"],
        jsonl["warm_read"]["logical_mib_per_s"],
    )
    none_warm_ratio = _ratio(
        none["warm_read"]["logical_mib_per_s"],
        jsonl["warm_read"]["logical_mib_per_s"],
    )
    if trainer_ratio is not None and trainer_ratio < 0.95:
        recommendation = "KEEP_JSONL"
        reason = "Parquet zstd reduced end-to-end Trainer throughput by more than 5%."
    elif zstd_disk_ratio is not None and zstd_disk_ratio < 0.75 and (
        trainer_ratio is None or trainer_ratio >= 0.95
    ):
        recommendation = "PARQUET_ZSTD_FOR_10M"
        reason = (
            "Parquet zstd materially reduces storage without a material "
            "Trainer-throughput regression."
        )
    elif none_warm_ratio is not None and none_warm_ratio > 1.10:
        recommendation = "PARQUET_UNCOMPRESSED_FOR_10M"
        reason = "Uncompressed Parquet materially improves warm feed throughput."
    else:
        recommendation = "KEEP_JSONL"
        reason = "No measured Parquet advantage clears the conservative migration threshold."
    return {
        "recommendation": recommendation,
        "reason": reason,
        "thresholds": {
            "max_trainer_regression_fraction": 0.05,
            "material_disk_saving_fraction": 0.25,
            "material_read_gain_fraction": 0.10,
        },
        "real_corpus_ratios": {
            "parquet_zstd_to_jsonl_data_bytes": zstd_disk_ratio,
            "parquet_zstd_to_jsonl_warm_read": zstd_warm_ratio,
            "parquet_none_to_jsonl_warm_read": none_warm_ratio,
            "parquet_zstd_to_jsonl_trainer_tokens_per_s": trainer_ratio,
        },
        "dependency_gate": (
            "Recommendation is experimental until D08 admits the exact PyArrow runtime; "
            "this benchmark does not mutate canonical dependency authority."
        ),
    }


def _main_benchmark(args: argparse.Namespace) -> None:
    repository_root = Path.cwd()
    if args.work_dir is None:
        temp = tempfile.TemporaryDirectory(prefix="perf146-")
        work_root = Path(temp.name)
    else:
        temp = None
        work_root = Path(args.work_dir)
        shutil.rmtree(work_root, ignore_errors=True)
        work_root.mkdir(parents=True)
    try:
        sources = work_root / "sources"
        real_source = sources / "data25-real"
        scale_source = sources / "scale-fixture"
        real_source.mkdir(parents=True)
        scale_source.parent.mkdir(parents=True, exist_ok=True)

        real_build_start = time.perf_counter()
        real_manifest = build_corpus(
            repository_root / "configs/data/corpus_v01.json",
            real_source,
        )
        real_source_build = time.perf_counter() - real_build_start

        scale_build_start = time.perf_counter()
        scale_manifest = write_scale_fixture(
            scale_source,
            records=args.scale_records,
            text_bytes=args.scale_text_bytes,
            input_parts=args.scale_input_parts,
        )
        scale_source_build = time.perf_counter() - scale_build_start

        layouts = work_root / "layouts"
        script = Path(__file__).resolve()
        stage_config = repository_root / "configs/stages/s3_10m.json"
        real_report = _dataset_report(
            real_source,
            layouts,
            name="real",
            script=script,
            stage_config=stage_config,
            trainer_steps=args.trainer_steps,
        )
        scale_report = _dataset_report(
            scale_source,
            layouts,
            name="scale",
            script=script,
            stage_config=stage_config,
            trainer_steps=0,
        )

        payload = {
            "schema": REPORT_SCHEMA,
            "authority": "LOCAL_FREE_EXPERIMENTAL_NON_PROMOTING",
            "source_head": os.environ.get("PERF146_SOURCE_SHA"),
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "pyarrow_expected": PYARROW_EXPERIMENT_VERSION,
                "platform": sys.platform,
                "cpu_count": os.cpu_count(),
                "cuda_available": torch.cuda.is_available(),
            },
            "audit": {
                "d04_reader": "PyArrow ParquetFile.iter_batches seam already exists",
                "datatrove_role": (
                    "D03 preprocessing/dedup/materialization; "
                    "not Trainer reader authority"
                ),
                "canonical_d08_pyarrow": False,
                "experimental_pyarrow_overlay": PYARROW_EXPERIMENT_VERSION,
            },
            "real_current_corpus": {
                "source_manifest_identity_sha256": real_manifest[
                    "corpus_identity_sha256"
                ],
                "source_documents": real_manifest["counters"]["accepted_documents"],
                "source_build_seconds": real_source_build,
                **real_report,
            },
            "generated_scale_fixture": {
                "fixture_identity_sha256": scale_manifest["fixture_identity_sha256"],
                "records": scale_manifest["records"],
                "text_bytes_per_record": scale_manifest["text_bytes_per_record"],
                "logical_text_bytes": scale_manifest["byte_tokens"],
                "source_build_seconds": scale_source_build,
                "truth_boundary": scale_manifest["authority"],
                **scale_report,
            },
            "decision": _decision(real_report),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_bytes(payload))
        print(json.dumps(payload, sort_keys=True))
    finally:
        if temp is not None:
            temp.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("perf146-corpus-format-benchmark.json")
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--scale-records", type=int, default=12000)
    parser.add_argument("--scale-text-bytes", type=int, default=4096)
    parser.add_argument("--scale-input-parts", type=int, default=12)
    parser.add_argument("--trainer-steps", type=int, default=3)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--layout-root", type=Path)
    parser.add_argument(
        "--stage-config", type=Path, default=Path("configs/stages/s3_10m.json")
    )
    parser.add_argument("--trainer-sequence-length", type=int, default=64)
    parser.add_argument("--trainer-batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.worker:
        if args.layout_root is None:
            parser.error("--worker requires --layout-root")
        _worker_measure(args)
        return
    _main_benchmark(args)


if __name__ == "__main__":
    main()
