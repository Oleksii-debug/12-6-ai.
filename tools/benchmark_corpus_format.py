"""LOCAL_FREE PERF-146 corpus physical-format benchmark."""

from __future__ import annotations

import argparse
import gc
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
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from twelve_six.data.corpus_format import (
    PYARROW_EXPERIMENT_VERSION,
    canonical_bytes,
    iter_layout_rows,
    layout_data_paths,
    materialize_layout,
    packed_training_trace,
    seek_layout_row,
    sha256_file,
    verify_layout,
)
from twelve_six.data.corpus_v01 import build_corpus
from twelve_six.data.shard_scale import (
    build_sharded_corpus,
    data25_input_files,
    verify_sharded_corpus,
    write_scale_fixture,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing.core import PACKING_CONFIG_HASH
from twelve_six.packing.scale_contracts import MixturePlan, MixtureSource
from twelve_six.packing.sharded_storage import (
    LogicalShardFile,
    build_physical_shard_dataset,
)
from twelve_six.packing.streaming import build_dataloader
from twelve_six.tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from twelve_six.training import Trainer, TrainerConfig

REPORT_SCHEMA = "12-6.perf146-corpus-format-benchmark.v2"
MIB = 1024 * 1024


def _rss_bytes(kind: int = resource.RUSAGE_SELF) -> int:
    value = resource.getrusage(kind).ru_maxrss
    return int(value * 1024 if value < 10**10 else value)


def _cpu_seconds(kind: int = resource.RUSAGE_SELF) -> float:
    usage = resource.getrusage(kind)
    return float(usage.ru_utime + usage.ru_stime)


def _plan(
    *,
    plan_id: str,
    source_name: str,
    source_identity: str,
    shards: int,
) -> MixturePlan:
    return MixturePlan(
        plan_id=plan_id,
        tokenizer_config_sha256=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
        packing_config_sha256=PACKING_CONFIG_HASH,
        sources=(MixtureSource(source_name, source_identity, 1),),
        seed=107,
        num_shards=shards,
        shard_seed=107_2026,
    )


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
    cpu_start = _cpu_seconds()
    documents = 0
    text_bytes = 0
    for row in iter_layout_rows(root, manifest):
        documents += 1
        text = row["text"]
        if not isinstance(text, str):
            raise TypeError("text must remain a string")
        text_bytes += len(text.encode("utf-8"))
    wall = time.perf_counter() - wall_start
    cpu = _cpu_seconds() - cpu_start
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


def _seek_samples(manifest: dict[str, Any], *, limit: int = 16) -> list[tuple[int, int]]:
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
        "scope": "physical_logical_shard_ordinal_seek",
    }


def _logical_shard_from_source_path(relative_path: str) -> tuple[str, int]:
    path = Path(relative_path)
    split = path.parent.name
    stem = path.stem
    if not stem.startswith("shard-"):
        raise ValueError(f"DATA-107 logical shard path expected, got {relative_path!r}")
    return split, int(stem.removeprefix("shard-"))


def _layout_files(
    root: Path,
    manifest: dict[str, Any],
    *,
    split: str,
) -> tuple[LogicalShardFile, ...]:
    file_format = str(manifest["physical_format"])
    files: list[LogicalShardFile] = []
    for entry in manifest["shards"]:
        source_relative = str(entry["source_relative_path"])
        entry_split, logical_shard = _logical_shard_from_source_path(source_relative)
        if entry_split != split:
            continue
        files.append(
            LogicalShardFile(
                logical_shard,
                str(root / str(entry["relative_path"])),
                file_format,
            )
        )
    if not files:
        raise RuntimeError(f"layout contains no {split!r} DATA-107 logical shards")
    return tuple(files)


def _build_dataset(
    root: Path,
    manifest: dict[str, Any],
    *,
    plan: MixturePlan,
    source_name: str,
    sequence_length: int,
    merged_cursor=None,
):
    return build_physical_shard_dataset(
        plan,
        ByteTokenizer(),
        _layout_files(root, manifest, split="train"),
        source_name=source_name,
        split="train",
        merged_cursor=merged_cursor,
        sequence_length=sequence_length,
        parquet_batch_rows=4096,
    )


def _same_envelope(expected, observed) -> bool:
    if expected.loss_tokens != observed.loss_tokens:
        return False
    if expected.examples != observed.examples:
        return False
    if expected.cursor_after.to_dict() != observed.cursor_after.to_dict():
        return False
    if set(expected.batch) != set(observed.batch):
        return False
    return all(torch.equal(expected.batch[key], observed.batch[key]) for key in expected.batch)


def _measure_d04_restart(
    root: Path,
    manifest: dict[str, Any],
    *,
    plan: MixturePlan,
    source_name: str,
    coldish: bool,
    cut_batches: int = 128,
    sequence_length: int = 128,
    batch_size: int = 2,
) -> dict[str, Any]:
    dataset = _build_dataset(
        root,
        manifest,
        plan=plan,
        source_name=source_name,
        sequence_length=sequence_length,
    )
    loader = build_dataloader(dataset, batch_size=batch_size, num_workers=0)
    iterator = iter(loader)
    checkpoint = None
    for _ in range(cut_batches):
        checkpoint = next(iterator)
    if checkpoint is None:
        raise AssertionError("restart checkpoint was not produced")
    expected = next(iterator)
    cursor = checkpoint.cursor_after
    del iterator
    del loader
    del dataset
    gc.collect()

    if coldish:
        _drop_page_cache_hint(layout_data_paths(root, manifest))
    resumed_dataset = _build_dataset(
        root,
        manifest,
        plan=plan,
        source_name=source_name,
        sequence_length=sequence_length,
        merged_cursor=cursor,
    )
    start = time.perf_counter()
    resumed_loader = build_dataloader(
        resumed_dataset,
        batch_size=batch_size,
        num_workers=0,
    )
    resumed_iterator = iter(resumed_loader)
    observed = next(resumed_iterator)
    elapsed = time.perf_counter() - start
    exact = _same_envelope(expected, observed)
    del resumed_iterator
    del resumed_loader
    del resumed_dataset
    gc.collect()
    if not exact:
        raise RuntimeError("D04 fresh-loader restart changed the next Trainer envelope")
    return {
        "cut_batches": cut_batches,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "seconds_to_first_resumed_batch": elapsed,
        "resume_exact": True,
        "cache_state": "best_effort_cold" if coldish else "warm",
        "scope": "actual_D04_cursor_fresh_loader_restart",
    }


def _measure_trainer(
    root: Path,
    manifest: dict[str, Any],
    *,
    stage_config: Path,
    plan: MixturePlan,
    source_name: str,
    steps: int,
    sequence_length: int,
    batch_size: int,
    loader_workers: int,
) -> dict[str, Any] | None:
    if steps <= 0:
        return None
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
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
            precision="fp32",
        ),
    )
    dataset = _build_dataset(
        root,
        manifest,
        plan=plan,
        source_name=source_name,
        sequence_length=sequence_length,
    )
    child_cpu_before = _cpu_seconds(resource.RUSAGE_CHILDREN)
    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        num_workers=loader_workers,
        prefetch_factor=2,
    )
    iterator = iter(loader)
    data_wait = 0.0
    train_compute = 0.0
    steady_data_wait = 0.0
    steady_train_compute = 0.0
    tokens = 0
    steady_tokens = 0
    self_cpu_start = _cpu_seconds()
    wall_start = time.perf_counter()
    for step in range(steps):
        next_start = time.perf_counter()
        envelope = next(iterator)
        waited = time.perf_counter() - next_start
        data_wait += waited
        train_start = time.perf_counter()
        metrics = trainer.train_microbatch(envelope.batch)
        trained = time.perf_counter() - train_start
        train_compute += trained
        if metrics.tokens != envelope.loss_tokens:
            raise RuntimeError("D04 envelope token count changed at Trainer boundary")
        tokens += envelope.loss_tokens
        if step:
            steady_data_wait += waited
            steady_train_compute += trained
            steady_tokens += envelope.loss_tokens
    wall = time.perf_counter() - wall_start
    self_cpu = _cpu_seconds() - self_cpu_start
    if trainer.tokens_seen != tokens:
        raise RuntimeError("Trainer token accounting disagrees with D04 envelopes")
    del iterator
    del loader
    del dataset
    gc.collect()
    child_cpu = _cpu_seconds(resource.RUSAGE_CHILDREN) - child_cpu_before
    total = data_wait + train_compute
    steady_total = steady_data_wait + steady_train_compute
    return {
        "model_stage": stage.stage,
        "model_parameters": stage.model.parameter_count(),
        "tokenizer": "s0-byte-v1",
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "loader_workers": loader_workers,
        "prefetch_factor": 2,
        "steps": steps,
        "loss_tokens": tokens,
        "wall_seconds": wall,
        "tokens_per_s": tokens / total if total else math.inf,
        "steady_tokens_per_s_excluding_first_batch": (
            steady_tokens / steady_total if steady_total else math.inf
        ),
        "data_wait_seconds": data_wait,
        "data_wait_percent": (100.0 * data_wait / total) if total else 0.0,
        "steady_data_wait_percent_excluding_first_batch": (
            100.0 * steady_data_wait / steady_total if steady_total else 0.0
        ),
        "train_compute_seconds": train_compute,
        "self_cpu_seconds": self_cpu,
        "dataloader_children_cpu_seconds": child_cpu,
        "aggregate_cpu_core_percent": (
            100.0 * (self_cpu + child_cpu) / wall if wall else 0.0
        ),
        "self_peak_rss_bytes": _rss_bytes(resource.RUSAGE_SELF),
        "dataloader_children_peak_rss_bytes": _rss_bytes(resource.RUSAGE_CHILDREN),
    }


def _worker_plan(args: argparse.Namespace) -> MixturePlan:
    return _plan(
        plan_id=args.plan_id,
        source_name=args.source_name,
        source_identity=args.source_identity,
        shards=args.logical_shards,
    )


def _worker_measure(args: argparse.Namespace) -> None:
    root = Path(args.layout_root)
    manifest = verify_layout(root)
    plan = _worker_plan(args)
    if manifest.get("source_logical_identity_sha256") != args.sharded_identity:
        raise RuntimeError("worker layout is not bound to the expected DATA-107 corpus")
    paths = layout_data_paths(root, manifest)
    cache = _drop_page_cache_hint(paths)
    cold = _consume_layout(root, manifest)
    warm = _consume_layout(root, manifest)
    seek_warm = _measure_seek(root, manifest, coldish=False)
    seek_cold = _measure_seek(root, manifest, coldish=True)
    d04_restart_warm = _measure_d04_restart(
        root,
        manifest,
        plan=plan,
        source_name=args.source_name,
        coldish=False,
    )
    d04_restart_cold = _measure_d04_restart(
        root,
        manifest,
        plan=plan,
        source_name=args.source_name,
        coldish=True,
    )
    trainer = _measure_trainer(
        root,
        manifest,
        stage_config=Path(args.stage_config),
        plan=plan,
        source_name=args.source_name,
        steps=args.trainer_steps,
        sequence_length=args.trainer_sequence_length,
        batch_size=args.trainer_batch_size,
        loader_workers=args.loader_workers,
    )
    payload = {
        "layout_identity_sha256": manifest["layout_identity_sha256"],
        "source_logical_identity_sha256": manifest["source_logical_identity_sha256"],
        "physical_format": manifest["physical_format"],
        "compression": manifest["compression"],
        "data_bytes": manifest["data_bytes"],
        "cold_cache_control": cache,
        "cold_read": cold,
        "warm_read": warm,
        "physical_seek_warm": seek_warm,
        "physical_seek_best_effort_cold": seek_cold,
        "d04_restart_warm": d04_restart_warm,
        "d04_restart_best_effort_cold": d04_restart_cold,
        "trainer": trainer,
        "worker_self_peak_rss_bytes": _rss_bytes(resource.RUSAGE_SELF),
        "worker_children_peak_rss_bytes": _rss_bytes(resource.RUSAGE_CHILDREN),
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
    loader_workers: int,
    plan: MixturePlan,
    plan_id: str,
    source_name: str,
    source_identity: str,
    sharded_identity: str,
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
        "--loader-workers",
        str(loader_workers),
        "--plan-id",
        plan_id,
        "--source-name",
        source_name,
        "--source-identity",
        source_identity,
        "--logical-shards",
        str(plan.num_shards),
        "--sharded-identity",
        sharded_identity,
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
    loader_workers: int,
    plan: MixturePlan,
    plan_id: str,
    source_name: str,
    source_identity: str,
) -> dict[str, Any]:
    source_manifest = json.loads(
        (source_root / "manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest["plan_sha256"] != plan.sha256:
        raise RuntimeError("DATA-107 source manifest does not match benchmark MixturePlan")
    if source_manifest["source_corpus_identity_sha256"] != source_identity:
        raise RuntimeError("DATA-107 source manifest changed source dataset identity")
    sharded_identity = str(source_manifest["corpus_identity_sha256"])
    candidates = _materialize_candidates(source_root, work_root, dataset_name=name)
    equivalence = _assert_equivalent_layouts(candidates)
    if equivalence["source_logical_identity_sha256"] != sharded_identity:
        raise RuntimeError("physical layouts are not bound to current DATA-107 identity")
    measured: dict[str, Any] = {}
    for candidate_name, item in candidates.items():
        observation = _run_worker(
            script,
            Path(item["root"]),
            stage_config=stage_config,
            trainer_steps=trainer_steps,
            trainer_sequence_length=64,
            trainer_batch_size=2,
            loader_workers=loader_workers,
            plan=plan,
            plan_id=plan_id,
            source_name=source_name,
            source_identity=source_identity,
            sharded_identity=sharded_identity,
        )
        observation["build_seconds"] = item["build_seconds"]
        observation["layout_manifest_sha256"] = sha256_file(
            Path(item["root"]) / "manifest.json"
        )
        measured[candidate_name] = observation
    return {
        "source_dataset_identity_sha256": source_identity,
        "source_data107_identity_sha256": sharded_identity,
        "plan_sha256": plan.sha256,
        "equivalence": equivalence,
        "candidates": measured,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _trainer_value(candidate: dict[str, Any], key: str) -> float | None:
    trainer = candidate.get("trainer")
    if trainer is None:
        return None
    return float(trainer[key])


def _decision(real: dict[str, Any]) -> dict[str, Any]:
    candidates = real["candidates"]
    jsonl = candidates["jsonl"]
    zstd = candidates["parquet_zstd"]
    none = candidates["parquet_none"]
    jsonl_steady = _trainer_value(
        jsonl, "steady_tokens_per_s_excluding_first_batch"
    )
    zstd_steady = _trainer_value(
        zstd, "steady_tokens_per_s_excluding_first_batch"
    )
    none_steady = _trainer_value(
        none, "steady_tokens_per_s_excluding_first_batch"
    )
    zstd_trainer_ratio = (
        _ratio(zstd_steady, jsonl_steady)
        if zstd_steady is not None and jsonl_steady is not None
        else None
    )
    none_trainer_ratio = (
        _ratio(none_steady, jsonl_steady)
        if none_steady is not None and jsonl_steady is not None
        else None
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
    jsonl_wait = _trainer_value(
        jsonl, "steady_data_wait_percent_excluding_first_batch"
    )
    zstd_wait = _trainer_value(
        zstd, "steady_data_wait_percent_excluding_first_batch"
    )
    zstd_wait_increase = (
        zstd_wait - jsonl_wait
        if zstd_wait is not None and jsonl_wait is not None
        else None
    )
    if zstd_trainer_ratio is not None and zstd_trainer_ratio < 0.95:
        recommendation = "KEEP_JSONL"
        reason = "Parquet zstd reduced steady S3 Trainer throughput by more than 5%."
    elif zstd_wait_increase is not None and zstd_wait_increase > 1.0:
        recommendation = "KEEP_JSONL"
        reason = "Parquet zstd increased steady foreground data wait by over 1 percentage point."
    elif zstd_disk_ratio is not None and zstd_disk_ratio < 0.75 and (
        zstd_trainer_ratio is None or zstd_trainer_ratio >= 0.95
    ):
        recommendation = "PARQUET_ZSTD_FOR_10M"
        reason = (
            "Parquet zstd materially reduces storage without a material steady "
            "Trainer-throughput or data-wait regression."
        )
    elif (
        none_warm_ratio is not None
        and none_warm_ratio > 1.10
        and (none_trainer_ratio is None or none_trainer_ratio >= 0.95)
    ):
        recommendation = "PARQUET_UNCOMPRESSED_FOR_10M"
        reason = "Uncompressed Parquet materially improves warm feed throughput."
    else:
        recommendation = "KEEP_JSONL"
        reason = "No measured Parquet advantage clears the conservative migration threshold."
    return {
        "recommendation": recommendation,
        "reason": reason,
        "thresholds": {
            "max_steady_trainer_regression_fraction": 0.05,
            "max_steady_data_wait_increase_percentage_points": 1.0,
            "material_disk_saving_fraction": 0.25,
            "material_read_gain_fraction": 0.10,
        },
        "real_corpus_ratios": {
            "parquet_zstd_to_jsonl_data_bytes": zstd_disk_ratio,
            "parquet_zstd_to_jsonl_warm_read": zstd_warm_ratio,
            "parquet_none_to_jsonl_warm_read": none_warm_ratio,
            "parquet_zstd_to_jsonl_steady_trainer_tokens_per_s": zstd_trainer_ratio,
            "parquet_none_to_jsonl_steady_trainer_tokens_per_s": none_trainer_ratio,
            "parquet_zstd_steady_data_wait_increase_percentage_points": zstd_wait_increase,
        },
        "dependency_gate": (
            "Recommendation is experimental until D08 admits the exact PyArrow runtime; "
            "this benchmark does not mutate canonical dependency authority."
        ),
    }


def _prepare_real_source(
    repository_root: Path,
    work_root: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], MixturePlan, dict[str, Any]]:
    data25_root = work_root / "data25-d03-source"
    start = time.perf_counter()
    data25 = build_corpus(
        repository_root / "configs/data/corpus_v01.json",
        data25_root,
    )
    data25_build_seconds = time.perf_counter() - start
    identity = str(data25["corpus_identity_sha256"])
    plan_id = "data107-data25-v01"
    source_name = "data25-v01"
    plan = _plan(
        plan_id=plan_id,
        source_name=source_name,
        source_identity=identity,
        shards=64,
    )
    sharded_root = work_root / "data25-data107-incumbent"
    start = time.perf_counter()
    sharded, observation = build_sharded_corpus(
        data25_input_files(data25_root),
        sharded_root,
        source_corpus_identity_sha256=identity,
        plan=plan,
        workers=1,
        target_shard_byte_tokens=384 * 1024,
        target_shard_size_bytes=768 * 1024,
        training_eligible=True,
        truth_boundary=(
            "DATA25_V0.1_TRAINING_TRUTH_PROJECT_AUTHORED_ONLY_ZERO_EXTERNAL_ELIGIBLE_SOURCES"
        ),
    )
    sharded_build_seconds = time.perf_counter() - start
    verify_sharded_corpus(sharded_root)
    prep = {
        "d03_build_seconds": data25_build_seconds,
        "data107_build_seconds": sharded_build_seconds,
        "data107_build_observation": asdict(observation),
        "source_documents": data25["counters"]["accepted_documents"],
        "source_byte_tokens": sum(
            int(item["byte_tokens"]) for item in data25["by_split"].values()
        ),
        "logical_shards": plan.num_shards,
        "plan_id": plan_id,
        "source_name": source_name,
    }
    return sharded_root, data25, sharded, plan, prep


def _prepare_scale_source(
    work_root: Path,
    *,
    records: int,
    text_bytes: int,
    input_parts: int,
) -> tuple[Path, dict[str, Any], dict[str, Any], MixturePlan, dict[str, Any]]:
    fixture_root = work_root / "scale-fixture-raw"
    start = time.perf_counter()
    fixture = write_scale_fixture(
        fixture_root,
        records=records,
        text_bytes=text_bytes,
        input_parts=input_parts,
    )
    fixture_build_seconds = time.perf_counter() - start
    identity = str(fixture["fixture_identity_sha256"])
    plan_id = "perf146-data107-scale-v1"
    source_name = "perf146-scale"
    plan = _plan(
        plan_id=plan_id,
        source_name=source_name,
        source_identity=identity,
        shards=64,
    )
    input_files = tuple(
        fixture_root / str(item["path"])
        for item in fixture["files"]
    )
    target_tokens = (int(fixture["byte_tokens"]) + plan.num_shards - 1) // plan.num_shards
    sharded_root = work_root / "scale-fixture-data107-incumbent"
    start = time.perf_counter()
    sharded, observation = build_sharded_corpus(
        input_files,
        sharded_root,
        source_corpus_identity_sha256=identity,
        plan=plan,
        workers=1,
        target_shard_byte_tokens=target_tokens,
        target_shard_size_bytes=target_tokens + MIB,
        training_eligible=False,
        truth_boundary="PROJECT_GENERATED_SCALE_FIXTURE_NOT_TRAINING_CORPUS_TRUTH",
    )
    sharded_build_seconds = time.perf_counter() - start
    verify_sharded_corpus(sharded_root)
    prep = {
        "raw_fixture_build_seconds": fixture_build_seconds,
        "data107_build_seconds": sharded_build_seconds,
        "data107_build_observation": asdict(observation),
        "logical_shards": plan.num_shards,
        "target_shard_byte_tokens": target_tokens,
        "plan_id": plan_id,
        "source_name": source_name,
    }
    return sharded_root, fixture, sharded, plan, prep


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
        sources.mkdir(parents=True)
        real_root, data25, real_sharded, real_plan, real_prep = _prepare_real_source(
            repository_root,
            sources,
        )
        scale_root, fixture, scale_sharded, scale_plan, scale_prep = _prepare_scale_source(
            sources,
            records=args.scale_records,
            text_bytes=args.scale_text_bytes,
            input_parts=args.scale_input_parts,
        )

        layouts = work_root / "layouts"
        script = Path(__file__).resolve()
        stage_config = repository_root / "configs/stages/s3_10m.json"
        real_report = _dataset_report(
            real_root,
            layouts,
            name="real",
            script=script,
            stage_config=stage_config,
            trainer_steps=args.trainer_steps,
            loader_workers=args.loader_workers,
            plan=real_plan,
            plan_id=str(real_prep["plan_id"]),
            source_name=str(real_prep["source_name"]),
            source_identity=str(data25["corpus_identity_sha256"]),
        )
        scale_report = _dataset_report(
            scale_root,
            layouts,
            name="scale",
            script=script,
            stage_config=stage_config,
            trainer_steps=0,
            loader_workers=0,
            plan=scale_plan,
            plan_id=str(scale_prep["plan_id"]),
            source_name=str(scale_prep["source_name"]),
            source_identity=str(fixture["fixture_identity_sha256"]),
        )

        payload = {
            "schema": REPORT_SCHEMA,
            "authority": "LOCAL_FREE_EXPERIMENTAL_NON_PROMOTING",
            "paid_compute": False,
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
                "d04_reader": "maintained JSONL and PyArrow Parquet logical-shard adapters",
                "datatrove_role": (
                    "D03 preprocessing/dedup/materialization; "
                    "not Trainer reader authority"
                ),
                "canonical_d08_pyarrow": False,
                "experimental_pyarrow_overlay": PYARROW_EXPERIMENT_VERSION,
                "incumbent_physical_boundary": (
                    "DATA-107 stable logical-shard JSONL; formats re-encode those exact rows"
                ),
            },
            "real_current_corpus": {
                "source_dataset_identity_sha256": data25["corpus_identity_sha256"],
                "source_data107_identity_sha256": real_sharded["corpus_identity_sha256"],
                "source_plan_sha256": real_plan.sha256,
                "source_preparation": real_prep,
                **real_report,
            },
            "generated_scale_fixture": {
                "fixture_identity_sha256": fixture["fixture_identity_sha256"],
                "source_data107_identity_sha256": scale_sharded["corpus_identity_sha256"],
                "source_plan_sha256": scale_plan.sha256,
                "records": fixture["records"],
                "text_bytes_per_record": fixture["text_bytes_per_record"],
                "logical_text_bytes": fixture["byte_tokens"],
                "truth_boundary": fixture["authority"],
                "source_preparation": scale_prep,
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
    parser.add_argument("--trainer-steps", type=int, default=5)
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--layout-root", type=Path)
    parser.add_argument(
        "--stage-config", type=Path, default=Path("configs/stages/s3_10m.json")
    )
    parser.add_argument("--trainer-sequence-length", type=int, default=64)
    parser.add_argument("--trainer-batch-size", type=int, default=2)
    parser.add_argument("--plan-id", default="")
    parser.add_argument("--source-name", default="")
    parser.add_argument("--source-identity", default="")
    parser.add_argument("--logical-shards", type=int, default=0)
    parser.add_argument("--sharded-identity", default="")
    args = parser.parse_args()
    if args.worker:
        if args.layout_root is None:
            parser.error("--worker requires --layout-root")
        if not all(
            (
                args.plan_id,
                args.source_name,
                args.source_identity,
                args.sharded_identity,
                args.logical_shards > 0,
            )
        ):
            parser.error("--worker requires complete DATA-107 plan identity arguments")
        _worker_measure(args)
        return
    _main_benchmark(args)


if __name__ == "__main__":
    main()
