"""Execute DATA-107 LOCAL_FREE scale, determinism, restart, and Trainer evidence."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

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
from twelve_six.packing.sharded_storage import LogicalShardFile, build_physical_shard_dataset
from twelve_six.packing.streaming import build_dataloader
from twelve_six.tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

REPORT_SCHEMA = "12-6.data107-shard-scale-evidence.v1"


def _plan(*, plan_id: str, source_name: str, source_identity: str, shards: int) -> MixturePlan:
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


def _signature(manifest: dict[str, object]) -> list[tuple[object, ...]]:
    return [
        (
            item["split"],
            item["logical_shard"],
            item["content_sha256"],
            item["manifest_sha256"],
            item["documents"],
            item["byte_tokens"],
        )
        for item in manifest["shards"]  # type: ignore[index]
    ]


def _rss_bytes(kind: int) -> int:
    value = resource.getrusage(kind).ru_maxrss
    return int(value * 1024 if value < 10**10 else value)


def _measure_d04_trainer(
    *,
    repo_root: Path,
    shard_root: Path,
    manifest: dict[str, object],
    plan: MixturePlan,
    source_name: str,
    steps: int,
    loader_workers: int,
) -> dict[str, object]:
    files = tuple(
        LogicalShardFile(
            int(item["logical_shard"]),
            str(shard_root / str(item["relative_path"])),
            "jsonl",
        )
        for item in manifest["shards"]  # type: ignore[index]
        if item["split"] == "train"
    )
    tokenizer = ByteTokenizer()
    dataset = build_physical_shard_dataset(
        plan,
        tokenizer,
        files,
        source_name=source_name,
        split="train",
        sequence_length=128,
    )
    loader = build_dataloader(
        dataset,
        batch_size=4,
        num_workers=loader_workers,
        prefetch_factor=2,
    )
    stage = load_stage_config(repo_root / "configs/stages/s0_10k.json")
    torch.manual_seed(1337)
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    model = TwelveSixDecoder(stage.model, stage.init)
    trainer = Trainer(
        model,
        TrainerConfig(
            max_steps=steps,
            learning_rate=3e-4,
            seed=1337,
            precision="fp32",
        ),
    )

    iterator = iter(loader)
    data_wait = 0.0
    train_time = 0.0
    steady_data_wait = 0.0
    steady_train_time = 0.0
    delivered_tokens = 0
    losses: list[float] = []
    first_cursor = None
    last_cursor = None
    self_rss_before = _rss_bytes(resource.RUSAGE_SELF)
    for step in range(steps):
        start = time.perf_counter()
        envelope = next(iterator)
        waited = time.perf_counter() - start
        data_wait += waited
        if step:
            steady_data_wait += waited
        start = time.perf_counter()
        metrics = trainer.train_microbatch(envelope.batch)
        trained = time.perf_counter() - start
        train_time += trained
        if step:
            steady_train_time += trained
        delivered_tokens += envelope.loss_tokens
        losses.append(float(metrics.loss))
        if first_cursor is None:
            first_cursor = envelope.cursor_after.to_dict()
        last_cursor = envelope.cursor_after.to_dict()

    total = data_wait + train_time
    steady_total = steady_data_wait + steady_train_time
    self_peak = _rss_bytes(resource.RUSAGE_SELF)
    del iterator
    del loader
    gc.collect()
    child_peak = _rss_bytes(resource.RUSAGE_CHILDREN)
    if trainer.tokens_seen != delivered_tokens:
        raise RuntimeError("D04 envelope token accounting disagrees with Trainer")
    return {
        "model_stage": stage.stage,
        "model_parameters": stage.model.parameter_count(),
        "model_identity_sha256": stage.model.identity_sha256(),
        "trainer_steps": steps,
        "loader_workers": loader_workers,
        "batch_size": 4,
        "sequence_length": 128,
        "delivered_loss_tokens": delivered_tokens,
        "trainer_tokens_seen": trainer.tokens_seen,
        "data_wait_seconds": data_wait,
        "trainer_seconds": train_time,
        "data_wait_percent": 100.0 * data_wait / total,
        "steady_state_data_wait_percent_excluding_first_batch": (
            100.0 * steady_data_wait / steady_total if steady_total else 0.0
        ),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "self_peak_rss_bytes": self_peak,
        "self_rss_before_bytes": self_rss_before,
        "children_peak_rss_bytes_after_loader_shutdown": child_peak,
        "memory_scope": (
            "linux_getrusage_self_plus_terminated_dataloader_children; "
            "torch allocator included in process RSS"
        ),
        "first_delivered_cursor": first_cursor,
        "last_delivered_cursor": last_cursor,
    }


def run(
    *,
    repo_root: Path,
    work_dir: Path,
    output: Path,
    source_sha: str,
    stress_records: int,
    stress_text_bytes: int,
    stress_input_parts: int,
    stress_shards: int,
    trainer_steps: int,
) -> dict[str, object]:
    if len(source_sha) != 40 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ValueError("--source-sha must be an exact lowercase 40-hex commit")
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    evidence_dir = work_dir / "retained-manifests"
    evidence_dir.mkdir()

    data25_root = work_dir / "data25-source"
    data25 = build_corpus(repo_root / "configs/data/corpus_v01.json", data25_root)
    data25_identity = str(data25["corpus_identity_sha256"])
    data25_plan = _plan(
        plan_id="data107-data25-v01",
        source_name="data25-v01",
        source_identity=data25_identity,
        shards=64,
    )
    data25_inputs = data25_input_files(data25_root)
    data25_one, data25_one_obs = build_sharded_corpus(
        data25_inputs,
        work_dir / "data25-build-w1",
        source_corpus_identity_sha256=data25_identity,
        plan=data25_plan,
        workers=1,
        target_shard_byte_tokens=384 * 1024,
        target_shard_size_bytes=768 * 1024,
        training_eligible=True,
        truth_boundary=(
            "DATA25_V0.1_TRAINING_TRUTH_PROJECT_AUTHORED_ONLY_ZERO_EXTERNAL_ELIGIBLE_SOURCES"
        ),
    )
    data25_four, data25_four_obs = build_sharded_corpus(
        data25_inputs,
        work_dir / "data25-build-w4",
        source_corpus_identity_sha256=data25_identity,
        plan=data25_plan,
        workers=4,
        target_shard_byte_tokens=384 * 1024,
        target_shard_size_bytes=768 * 1024,
        training_eligible=True,
        truth_boundary=(
            "DATA25_V0.1_TRAINING_TRUTH_PROJECT_AUTHORED_ONLY_ZERO_EXTERNAL_ELIGIBLE_SOURCES"
        ),
    )
    if (
        data25_one["corpus_identity_sha256"] != data25_four["corpus_identity_sha256"]
        or _signature(data25_one) != _signature(data25_four)
    ):
        raise RuntimeError("DATA-25 shard identity changed across process counts")
    verify_sharded_corpus(work_dir / "data25-build-w1")
    verify_sharded_corpus(work_dir / "data25-build-w4")

    fixture_root = work_dir / "stress-source"
    fixture = write_scale_fixture(
        fixture_root,
        records=stress_records,
        text_bytes=stress_text_bytes,
        input_parts=stress_input_parts,
    )
    if int(fixture["byte_tokens"]) < 200_000_000:
        raise RuntimeError("scale fixture must exercise at least 200M byte tokens")
    stress_identity = str(fixture["fixture_identity_sha256"])
    stress_plan = _plan(
        plan_id="data107-stress-256m",
        source_name="data107-stress",
        source_identity=stress_identity,
        shards=stress_shards,
    )
    stress_inputs = tuple(
        fixture_root / str(item["path"])
        for item in fixture["files"]  # type: ignore[index]
    )
    target_tokens = (int(fixture["byte_tokens"]) + stress_shards - 1) // stress_shards
    stress_one, stress_one_obs = build_sharded_corpus(
        stress_inputs,
        work_dir / "stress-build-w1",
        source_corpus_identity_sha256=stress_identity,
        plan=stress_plan,
        workers=1,
        target_shard_byte_tokens=target_tokens,
        target_shard_size_bytes=target_tokens + 1024 * 1024,
        training_eligible=False,
        truth_boundary="PROJECT_GENERATED_SCALE_FIXTURE_NOT_TRAINING_CORPUS_TRUTH",
    )

    interrupted = False
    stress_w4_root = work_dir / "stress-build-w4-resumed"
    try:
        build_sharded_corpus(
            stress_inputs,
            stress_w4_root,
            source_corpus_identity_sha256=stress_identity,
            plan=stress_plan,
            workers=4,
            target_shard_byte_tokens=target_tokens,
            target_shard_size_bytes=target_tokens + 1024 * 1024,
            stop_after_shards=7,
            training_eligible=False,
            truth_boundary="PROJECT_GENERATED_SCALE_FIXTURE_NOT_TRAINING_CORPUS_TRUTH",
        )
    except InterruptedError as exc:
        if str(exc) != "DATA107_INTENTIONAL_INTERRUPTION":
            raise
        interrupted = True
    if not interrupted or (stress_w4_root / "manifest.json").exists():
        raise RuntimeError(
            "intentional interrupted build did not fail closed before global publish"
        )
    stress_four, stress_four_obs = build_sharded_corpus(
        stress_inputs,
        stress_w4_root,
        source_corpus_identity_sha256=stress_identity,
        plan=stress_plan,
        workers=4,
        target_shard_byte_tokens=target_tokens,
        target_shard_size_bytes=target_tokens + 1024 * 1024,
        training_eligible=False,
        truth_boundary="PROJECT_GENERATED_SCALE_FIXTURE_NOT_TRAINING_CORPUS_TRUTH",
    )
    if stress_four_obs.resumed_complete_shards < 7:
        raise RuntimeError("resume did not reuse already complete transactional shards")
    if (
        stress_one["corpus_identity_sha256"] != stress_four["corpus_identity_sha256"]
        or _signature(stress_one) != _signature(stress_four)
    ):
        raise RuntimeError("stress shard identity changed across process counts/restart")
    verify_sharded_corpus(work_dir / "stress-build-w1")
    verify_sharded_corpus(stress_w4_root)

    trainer_measurement = _measure_d04_trainer(
        repo_root=repo_root,
        shard_root=stress_w4_root,
        manifest=stress_four,
        plan=stress_plan,
        source_name="data107-stress",
        steps=trainer_steps,
        loader_workers=2,
    )

    retained = {
        "data25_w1": data25_one,
        "data25_w4": data25_four,
        "stress_w1": stress_one,
        "stress_w4_resumed": stress_four,
        "stress_fixture": fixture,
    }
    for name, payload in retained.items():
        (evidence_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    report = {
        "schema": REPORT_SCHEMA,
        "source_sha": source_sha,
        "authority": "LOCAL_FREE_GITHUB_HOSTED_CPU_SCALE_EVIDENCE",
        "paid_compute": False,
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_count": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
        },
        "incumbents": {
            "data25_corpus_identity_sha256": data25_identity,
            "d04_sharding_version": "record-id-sha256-v1",
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "packing_config_sha256": PACKING_CONFIG_HASH,
        },
        "data25_repackage": {
            "truth_boundary": data25_one["truth_boundary"],
            "source_documents": data25["counters"]["accepted_documents"],
            "source_byte_tokens": sum(
                int(item["byte_tokens"]) for item in data25["by_split"].values()
            ),
            "logical_shards": data25_plan.num_shards,
            "corpus_identity_sha256": data25_one["corpus_identity_sha256"],
            "process_count_1": asdict(data25_one_obs),
            "process_count_4": asdict(data25_four_obs),
            "identical_global_identity": True,
            "identical_shard_content_and_manifest_hashes": True,
        },
        "stress_fixture": {
            "training_eligible": False,
            "fixture_identity_sha256": stress_identity,
            "records": fixture["records"],
            "byte_tokens": fixture["byte_tokens"],
            "input_parts": fixture["input_parts"],
            "logical_shards": stress_shards,
            "target_shard_byte_tokens": target_tokens,
            "build_process_count_1": asdict(stress_one_obs),
            "build_process_count_4_after_restart": asdict(stress_four_obs),
            "intentional_interruption_before_global_manifest": True,
            "resumed_complete_shards": stress_four_obs.resumed_complete_shards,
            "identical_global_identity": True,
            "identical_shard_content_and_manifest_hashes": True,
            "corpus_identity_sha256": stress_one["corpus_identity_sha256"],
        },
        "d04_trainer_stream": trainer_measurement,
        "invariants": {
            "immutable_record_identity_drives_sharding": True,
            "filesystem_enumeration_not_in_corpus_identity": True,
            "bounded_sort_chunk_bytes": 8 * 1024 * 1024,
            "max_open_files_per_worker": 16,
            "per_shard_content_hash": True,
            "per_shard_manifest_hash": True,
            "per_shard_source_modality_counts": True,
            "global_manifest_published_last": True,
            "partial_shard_complete_marker_forbidden": True,
            "restart_reuses_only_verified_complete_shards": True,
            "stress_fixture_separate_from_training_truth": True,
        },
        "reproduction_command": (
            "PYTHONPATH=src python tools/run_data107_shard_scale.py "
            "--repo-root . --source-sha <EXACT_40_HEX_HEAD> "
            "--work-dir data107-scale-work --output data107-scale-evidence.json"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--work-dir", type=Path, default=Path("data107-scale-work"))
    parser.add_argument("--output", type=Path, default=Path("data107-scale-evidence.json"))
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--stress-records", type=int, default=4096)
    parser.add_argument("--stress-text-bytes", type=int, default=65536)
    parser.add_argument("--stress-input-parts", type=int, default=16)
    parser.add_argument("--stress-shards", type=int, default=64)
    parser.add_argument("--trainer-steps", type=int, default=16)
    args = parser.parse_args()
    report = run(
        repo_root=args.repo_root.resolve(),
        work_dir=args.work_dir.resolve(),
        output=args.output.resolve(),
        source_sha=args.source_sha,
        stress_records=args.stress_records,
        stress_text_bytes=args.stress_text_bytes,
        stress_input_parts=args.stress_input_parts,
        stress_shards=args.stress_shards,
        trainer_steps=args.trainer_steps,
    )
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "source_sha": report["source_sha"],
                "stress_byte_tokens": report["stress_fixture"]["byte_tokens"],
                "stress_identity": report["stress_fixture"]["corpus_identity_sha256"],
                "data_wait_percent": report["d04_trainer_stream"]["data_wait_percent"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
