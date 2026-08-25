"""LOCAL_FREE synthetic throughput evidence for next-stage streaming/packing."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from twelve_six.tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer

from .core import PACKING_CONFIG_HASH
from .scale_contracts import MixturePlan, MixtureSource
from .sharded_storage import LogicalShardFile, build_physical_shard_dataset
from .streaming import (
    build_dataloader,
    iter_jsonl_stream,
    iter_packed_stream,
    iter_trainer_batches,
    prefetch_bounded,
)

BENCHMARK_SCHEMA = "12-6.streaming-packing-benchmark.v2"


@dataclass(frozen=True, slots=True)
class SyntheticCorpus:
    documents: int
    utf8_bytes: int
    expected_loss_tokens: int
    unsharded_file_bytes: int
    physical_shard_file_bytes: int
    physical_shard_files: int


@dataclass(frozen=True, slots=True)
class Measurement:
    mode: str
    wall_seconds: float
    examples: int
    loss_tokens: int
    loss_tokens_per_second: float
    examples_per_second: float
    source_mib_per_second: float
    peak_tracemalloc_bytes: int
    memory_scope: str


def _make_plan() -> MixturePlan:
    return MixturePlan(
        plan_id="local-free-streaming-benchmark",
        tokenizer_config_sha256=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
        packing_config_sha256=PACKING_CONFIG_HASH,
        sources=(MixtureSource("synthetic", "8" * 64, 1),),
        seed=1337,
        num_shards=256,
        shard_seed=7331,
    )


def _synthetic_text(index: int) -> str:
    units = 4 + (index % 29)
    base = (
        f"record={index:08d} English general pretraining. "
        "Український текст для перевірки UTF-8 морфології. "
        "def f(x): return x * x  # code sample\n"
    )
    return base * units


def write_synthetic_sources(
    unsharded_path: Path,
    physical_dir: Path,
    *,
    documents: int,
    plan: MixturePlan,
) -> tuple[SyntheticCorpus, tuple[LogicalShardFile, ...]]:
    """Write identical logical content as one JSONL file and stable physical shards."""
    if documents <= 0:
        raise ValueError("documents must be positive")
    physical_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[int, object] = {}
    paths: dict[int, Path] = {}
    utf8_bytes = 0
    expected_loss_tokens = 0
    try:
        with unsharded_path.open("w", encoding="utf-8", newline="\n") as unsharded:
            for index in range(documents):
                record_id = f"synthetic-{index:08d}"
                text = _synthetic_text(index)
                text_bytes = len(text.encode("utf-8"))
                utf8_bytes += text_bytes
                expected_loss_tokens += max(text_bytes - 1, 0)
                line = (
                    json.dumps(
                        {"id": record_id, "text": text},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                unsharded.write(line)
                logical_shard = plan.shard_for_record(record_id)
                if logical_shard not in handles:
                    path = physical_dir / f"part-{logical_shard:05d}.jsonl"
                    paths[logical_shard] = path
                    handles[logical_shard] = path.open("w", encoding="utf-8", newline="\n")
                handles[logical_shard].write(line)
    finally:
        for handle in handles.values():
            handle.close()
    files = tuple(
        LogicalShardFile(shard, str(paths[shard]), "jsonl") for shard in sorted(paths)
    )
    physical_bytes = sum(Path(file.path).stat().st_size for file in files)
    corpus = SyntheticCorpus(
        documents=documents,
        utf8_bytes=utf8_bytes,
        expected_loss_tokens=expected_loss_tokens,
        unsharded_file_bytes=unsharded_path.stat().st_size,
        physical_shard_file_bytes=physical_bytes,
        physical_shard_files=len(files),
    )
    if corpus.unsharded_file_bytes != corpus.physical_shard_file_bytes:
        raise RuntimeError("physical-shard packaging changed corpus JSONL byte membership")
    return corpus, files


def _measurement(
    *,
    mode: str,
    start: float,
    examples: int,
    loss_tokens: int,
    source_bytes: int,
    peak_bytes: int,
    memory_scope: str,
) -> Measurement:
    wall = time.perf_counter() - start
    return Measurement(
        mode=mode,
        wall_seconds=wall,
        examples=examples,
        loss_tokens=loss_tokens,
        loss_tokens_per_second=loss_tokens / wall,
        examples_per_second=examples / wall,
        source_mib_per_second=(source_bytes / (1024 * 1024)) / wall,
        peak_tracemalloc_bytes=peak_bytes,
        memory_scope=memory_scope,
    )


def measure_direct(
    path: Path,
    corpus: SyntheticCorpus,
    plan: MixturePlan,
    *,
    prefetch_items: int,
) -> Measurement:
    tokenizer = ByteTokenizer()
    records = prefetch_bounded(
        iter_jsonl_stream(path, split="train"),
        max_items=prefetch_items,
    )
    items = iter_packed_stream(
        records,
        tokenizer,
        plan,
        source_name="synthetic",
        split="train",
        sequence_length=128,
    )
    batches = iter_trainer_batches(items, batch_size=32, target_mode="labels")
    tracemalloc.start()
    start = time.perf_counter()
    examples = 0
    loss_tokens = 0
    for batch in batches:
        examples += batch.examples
        loss_tokens += batch.loss_tokens
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if loss_tokens != corpus.expected_loss_tokens:
        raise RuntimeError(
            f"direct exact loss-token accounting drifted: {loss_tokens} != "
            f"{corpus.expected_loss_tokens}"
        )
    return _measurement(
        mode=f"direct-unsharded-prefetch-{prefetch_items}",
        start=start,
        examples=examples,
        loss_tokens=loss_tokens,
        source_bytes=corpus.unsharded_file_bytes,
        peak_bytes=peak,
        memory_scope="main_process_python_allocations_only",
    )


def measure_physical_dataloader(
    files: tuple[LogicalShardFile, ...],
    corpus: SyntheticCorpus,
    plan: MixturePlan,
    *,
    workers: int,
) -> Measurement:
    tokenizer = ByteTokenizer()
    dataset = build_physical_shard_dataset(
        plan,
        tokenizer,
        files,
        source_name="synthetic",
        split="train",
        sequence_length=128,
    )
    loader = build_dataloader(
        dataset,
        batch_size=32,
        num_workers=workers,
        prefetch_factor=2,
    )
    tracemalloc.start()
    start = time.perf_counter()
    examples = 0
    loss_tokens = 0
    for batch in loader:
        examples += batch.examples
        loss_tokens += batch.loss_tokens
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if loss_tokens != corpus.expected_loss_tokens:
        raise RuntimeError(
            f"physical DataLoader exact loss-token accounting drifted: {loss_tokens} != "
            f"{corpus.expected_loss_tokens}"
        )
    return _measurement(
        mode=f"physical-logical-shards-torch-workers-{workers}",
        start=start,
        examples=examples,
        loss_tokens=loss_tokens,
        source_bytes=corpus.physical_shard_file_bytes,
        peak_bytes=peak,
        memory_scope="main_process_python_allocations_only_workers_excluded",
    )


def run_benchmark(*, documents: int, output: Path) -> dict[str, object]:
    plan = _make_plan()
    with tempfile.TemporaryDirectory(prefix="twelve-six-streaming-") as directory:
        root = Path(directory)
        source = root / "synthetic.jsonl"
        corpus, physical_files = write_synthetic_sources(
            source,
            root / "physical-shards",
            documents=documents,
            plan=plan,
        )
        direct = measure_direct(source, corpus, plan, prefetch_items=32)
        dataloader = measure_physical_dataloader(
            physical_files,
            corpus,
            plan,
            workers=2,
        )
    report = {
        "schema": BENCHMARK_SCHEMA,
        "authority": "LOCAL_FREE_SYNTHETIC_DATA_DELIVERY_EVIDENCE_NOT_GPU_CAPACITY",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "corpus": asdict(corpus),
        "measurements": [asdict(direct), asdict(dataloader)],
        "invariants": {
            "exact_loss_token_accounting": True,
            "document_isolated_training_path": True,
            "logical_shards": plan.num_shards,
            "physical_shards_opened_by_owner_only": True,
            "world_size_change_requires_merged_logical_shard_cursor": True,
            "bounded_thread_prefetch_items": 32,
            "dataloader_prefetch_factor": 2,
            "gpu_capacity_claim": False,
            "tracemalloc_worker_memory_included": False,
            "parquet_backend": "IMPLEMENTED_OPTIONAL_PYARROW_RUNTIME_NOT_CLAIMED_BY_THIS_BENCHMARK",
        },
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("streaming-packing-benchmark.json"))
    args = parser.parse_args()
    report = run_benchmark(documents=args.documents, output=args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
