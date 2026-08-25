"""DATA-107 deterministic transactional corpus sharding at hundreds-of-millions scale."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import resource
import shutil
import time
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from twelve_six.packing.scale_contracts import MixturePlan

SHARDED_CORPUS_SCHEMA = "12-6.sharded-corpus.v1"
SHARD_MANIFEST_SCHEMA = "12-6.logical-shard-manifest.v1"
STRESS_FIXTURE_SCHEMA = "12-6.data107-scale-fixture.v1"
COMPLETE_MARKER = b"DATA107_COMPLETE\n"


class ShardScaleError(ValueError):
    """Raised when DATA-107 storage/restart invariants fail closed."""


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB. macOS reports bytes. CI authority is Linux, keep fallback explicit.
    return int(value * 1024 if value < 10**10 else value)


def _row_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    record_id = payload.get("id", payload.get("record_id"))
    text = payload.get("text")
    split = payload.get("split", "train")
    source_id = payload.get("source_id", "unknown")
    modality = payload.get("modality", "unknown")
    if not isinstance(record_id, str) or not record_id:
        raise ShardScaleError("input record requires non-empty id/record_id")
    if not isinstance(text, str):
        raise ShardScaleError(f"record {record_id!r} requires text")
    if not isinstance(split, str) or not split:
        raise ShardScaleError(f"record {record_id!r} requires split")
    if not isinstance(source_id, str) or not source_id:
        raise ShardScaleError(f"record {record_id!r} requires source_id")
    if not isinstance(modality, str) or not modality:
        raise ShardScaleError(f"record {record_id!r} requires modality")
    text_bytes = len(text.encode("utf-8"))
    claimed = payload.get("byte_tokens")
    if claimed is not None and int(claimed) != text_bytes:
        raise ShardScaleError(f"record {record_id!r} byte-token accounting drift")
    output = {
        "id": record_id,
        "text": text,
        "split": split,
        "source_id": source_id,
        "modality": modality,
        "byte_tokens": text_bytes,
    }
    for key in (
        "source_version",
        "stratum",
        "external",
        "project_authored",
        "content_sha256",
    ):
        if key in payload:
            output[key] = payload[key]
    return output


class _HandlePool:
    def __init__(self, max_open_files: int) -> None:
        if max_open_files <= 0:
            raise ShardScaleError("max_open_files must be positive")
        self.max_open_files = max_open_files
        self._handles: OrderedDict[Path, Any] = OrderedDict()

    def append(self, path: Path, payload: bytes) -> None:
        handle = self._handles.pop(path, None)
        if handle is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("ab")
        self._handles[path] = handle
        handle.write(payload)
        while len(self._handles) > self.max_open_files:
            _, old = self._handles.popitem(last=False)
            old.close()

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()


@dataclass(frozen=True, slots=True)
class BuildObservation:
    workers: int
    wall_seconds: float
    peak_rss_bytes: int
    resumed_complete_shards: int
    input_files: int


def _worker_ingest(
    worker_id: int,
    paths: tuple[str, ...],
    work_root: str,
    plan: MixturePlan,
    max_open_files: int,
) -> dict[str, int]:
    pool = _HandlePool(max_open_files)
    documents = 0
    byte_tokens = 0
    try:
        for raw_path in paths:
            with Path(raw_path).open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ShardScaleError(
                            f"{raw_path}:{line_number}: invalid JSON"
                        ) from exc
                    if not isinstance(payload, dict):
                        raise ShardScaleError(f"{raw_path}:{line_number}: object required")
                    row = _row_from_payload(payload)
                    logical_shard = plan.shard_for_record(row["id"])
                    path = (
                        Path(work_root)
                        / f"worker-{worker_id:04d}"
                        / row["split"]
                        / f"shard-{logical_shard:05d}.unsorted"
                    )
                    pool.append(path, canonical_bytes(row))
                    documents += 1
                    byte_tokens += int(row["byte_tokens"])
    finally:
        pool.close()
    return {"documents": documents, "byte_tokens": byte_tokens}


def _iter_fragment_rows(paths: Sequence[Path]) -> Iterator[tuple[str, bytes]]:
    for path in paths:
        if not path.exists():
            continue
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ShardScaleError(f"{path}:{line_number}: invalid staged JSON") from exc
                record_id = payload.get("id")
                if not isinstance(record_id, str) or not record_id:
                    raise ShardScaleError(f"{path}:{line_number}: staged id invalid")
                yield record_id, line


def _write_sorted_chunks(
    rows: Iterable[tuple[str, bytes]],
    chunk_dir: Path,
    *,
    sort_chunk_bytes: int,
) -> list[Path]:
    if sort_chunk_bytes <= 0:
        raise ShardScaleError("sort_chunk_bytes must be positive")
    shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[Path] = []
    pending: list[tuple[str, bytes]] = []
    pending_bytes = 0

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        pending.sort(key=lambda item: item[0])
        path = chunk_dir / f"chunk-{len(chunks):05d}.jsonl"
        with path.open("wb") as handle:
            for _, line in pending:
                handle.write(line)
        chunks.append(path)
        pending = []
        pending_bytes = 0

    for record_id, line in rows:
        pending.append((record_id, line))
        pending_bytes += len(line)
        if pending_bytes >= sort_chunk_bytes:
            flush()
    flush()
    return chunks


def _read_chunk_item(handle: Any) -> tuple[str, bytes] | None:
    line = handle.readline()
    if not line:
        return None
    payload = json.loads(line)
    record_id = payload.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ShardScaleError("sorted chunk contains invalid id")
    return record_id, line


def _merge_chunks(
    chunks: Sequence[Path],
    output_partial: Path,
) -> dict[str, Any]:
    output_partial.parent.mkdir(parents=True, exist_ok=True)
    handles = [path.open("rb") for path in chunks]
    heap: list[tuple[str, int, bytes]] = []
    for index, handle in enumerate(handles):
        item = _read_chunk_item(handle)
        if item is not None:
            heapq.heappush(heap, (item[0], index, item[1]))

    digest = hashlib.sha256()
    documents = 0
    byte_tokens = 0
    by_source: dict[str, Counter[str]] = {}
    by_modality: dict[str, Counter[str]] = {}
    previous_id: str | None = None
    try:
        with output_partial.open("wb") as output:
            while heap:
                record_id, index, line = heapq.heappop(heap)
                if previous_id == record_id:
                    raise ShardScaleError(f"duplicate immutable record id {record_id!r}")
                previous_id = record_id
                payload = json.loads(line)
                tokens = int(payload["byte_tokens"])
                source = str(payload["source_id"])
                modality = str(payload["modality"])
                output.write(line)
                digest.update(line)
                documents += 1
                byte_tokens += tokens
                source_counter = by_source.setdefault(source, Counter())
                source_counter["documents"] += 1
                source_counter["byte_tokens"] += tokens
                modality_counter = by_modality.setdefault(modality, Counter())
                modality_counter["documents"] += 1
                modality_counter["byte_tokens"] += tokens
                item = _read_chunk_item(handles[index])
                if item is not None:
                    heapq.heappush(heap, (item[0], index, item[1]))
            output.flush()
            os.fsync(output.fileno())
    finally:
        for handle in handles:
            handle.close()
    return {
        "sha256": digest.hexdigest(),
        "documents": documents,
        "byte_tokens": byte_tokens,
        "size_bytes": output_partial.stat().st_size,
        "by_source": {key: dict(value) for key, value in sorted(by_source.items())},
        "by_modality": {key: dict(value) for key, value in sorted(by_modality.items())},
    }


def _marker_path(data_path: Path) -> Path:
    return data_path.with_suffix(data_path.suffix + ".complete")


def _manifest_path(data_path: Path) -> Path:
    return data_path.with_suffix(".manifest.json")


def _verify_complete_shard(data_path: Path) -> dict[str, Any] | None:
    marker = _marker_path(data_path)
    manifest_path = _manifest_path(data_path)
    if not (data_path.exists() and manifest_path.exists() and marker.exists()):
        return None
    if marker.read_bytes() != COMPLETE_MARKER:
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SHARD_MANIFEST_SCHEMA:
        return None
    expected_manifest_hash = manifest.get("manifest_sha256")
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if expected_manifest_hash != sha256_bytes(canonical_bytes(core)):
        return None
    if manifest.get("content_sha256") != sha256_file(data_path):
        return None
    return manifest


def _publish_shard(
    output_root: Path,
    work_root: Path,
    *,
    split: str,
    logical_shard: int,
    workers: int,
    plan: MixturePlan,
    target_shard_byte_tokens: int,
    target_shard_size_bytes: int,
    sort_chunk_bytes: int,
) -> tuple[dict[str, Any], bool]:
    data_path = output_root / split / f"shard-{logical_shard:05d}.jsonl"
    existing = _verify_complete_shard(data_path)
    if existing is not None:
        if (
            existing.get("plan_sha256") == plan.sha256
            and existing.get("split") == split
            and existing.get("logical_shard") == logical_shard
            and existing.get("target_shard_byte_tokens") == target_shard_byte_tokens
            and existing.get("target_shard_size_bytes") == target_shard_size_bytes
        ):
            return existing, True
        raise ShardScaleError(
            f"existing complete shard {data_path} belongs to another build contract"
        )

    fragments = tuple(
        work_root
        / f"worker-{worker_id:04d}"
        / split
        / f"shard-{logical_shard:05d}.unsorted"
        for worker_id in range(workers)
    )
    chunk_dir = work_root / "sort" / split / f"shard-{logical_shard:05d}"
    chunks = _write_sorted_chunks(
        _iter_fragment_rows(fragments),
        chunk_dir,
        sort_chunk_bytes=sort_chunk_bytes,
    )
    if not chunks:
        return {}, False

    partial = data_path.with_suffix(data_path.suffix + ".partial")
    stats = _merge_chunks(chunks, partial)
    core = {
        "schema": SHARD_MANIFEST_SCHEMA,
        "logical_shard": logical_shard,
        "split": split,
        "relative_path": str(data_path.relative_to(output_root)),
        "plan_sha256": plan.sha256,
        "target_shard_byte_tokens": target_shard_byte_tokens,
        "target_shard_size_bytes": target_shard_size_bytes,
        "content_sha256": stats["sha256"],
        "documents": stats["documents"],
        "byte_tokens": stats["byte_tokens"],
        "size_bytes": stats["size_bytes"],
        "by_source": stats["by_source"],
        "by_modality": stats["by_modality"],
    }
    manifest = {**core, "manifest_sha256": sha256_bytes(canonical_bytes(core))}
    manifest_partial = _manifest_path(data_path).with_suffix(".json.partial")
    manifest_partial.parent.mkdir(parents=True, exist_ok=True)
    manifest_partial.write_bytes(canonical_bytes(manifest))
    os.replace(partial, data_path)
    os.replace(manifest_partial, _manifest_path(data_path))
    marker_partial = _marker_path(data_path).with_suffix(".complete.partial")
    marker_partial.write_bytes(COMPLETE_MARKER)
    os.replace(marker_partial, _marker_path(data_path))
    shutil.rmtree(chunk_dir, ignore_errors=True)
    verified = _verify_complete_shard(data_path)
    if verified is None:
        raise ShardScaleError(f"published shard {data_path} failed verification")
    return verified, False


def build_sharded_corpus(
    input_files: Sequence[str | Path],
    output_root: str | Path,
    *,
    source_corpus_identity_sha256: str,
    plan: MixturePlan,
    workers: int,
    target_shard_byte_tokens: int,
    target_shard_size_bytes: int,
    sort_chunk_bytes: int = 8 * 1024 * 1024,
    max_open_files_per_worker: int = 16,
    stop_after_shards: int | None = None,
    training_eligible: bool,
    truth_boundary: str,
) -> tuple[dict[str, Any], BuildObservation]:
    """Build byte-stable logical shards transactionally with bounded working memory."""
    if workers <= 0:
        raise ShardScaleError("workers must be positive")
    if target_shard_byte_tokens <= 0 or target_shard_size_bytes <= 0:
        raise ShardScaleError("shard targets must be positive")
    inputs = tuple(Path(path) for path in input_files)
    if not inputs or any(not path.is_file() for path in inputs):
        raise ShardScaleError("all input files must exist")
    # Caller supplies manifest order. We sort by content-addressed path string only for process
    # partitioning; record->shard and final record order are independent of this enumeration.
    inputs = tuple(sorted(inputs, key=lambda path: str(path)))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    work_root = output_root / ".data107-work"
    shutil.rmtree(work_root / "workers", ignore_errors=True)
    (work_root / "workers").mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    assignments = [
        tuple(str(path) for index, path in enumerate(inputs) if index % workers == worker_id)
        for worker_id in range(workers)
    ]
    if workers == 1:
        ingest = [
            _worker_ingest(
                0,
                assignments[0],
                str(work_root / "workers"),
                plan,
                max_open_files_per_worker,
            )
        ]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _worker_ingest,
                    worker_id,
                    assignments[worker_id],
                    str(work_root / "workers"),
                    plan,
                    max_open_files_per_worker,
                )
                for worker_id in range(workers)
            ]
            ingest = [future.result() for future in futures]

    seen_splits: set[str] = set()
    for worker_dir in sorted((work_root / "workers").glob("worker-*")):
        for split_dir in worker_dir.iterdir():
            if split_dir.is_dir():
                seen_splits.add(split_dir.name)
    if not seen_splits:
        raise ShardScaleError("ingestion produced no records")

    shard_manifests: list[dict[str, Any]] = []
    resumed = 0
    published_this_run = 0
    for split in sorted(seen_splits):
        for logical_shard in range(plan.num_shards):
            manifest, was_resumed = _publish_shard(
                output_root,
                work_root / "workers",
                split=split,
                logical_shard=logical_shard,
                workers=workers,
                plan=plan,
                target_shard_byte_tokens=target_shard_byte_tokens,
                target_shard_size_bytes=target_shard_size_bytes,
                sort_chunk_bytes=sort_chunk_bytes,
            )
            if not manifest:
                continue
            shard_manifests.append(manifest)
            if was_resumed:
                resumed += 1
            else:
                published_this_run += 1
            if stop_after_shards is not None and published_this_run >= stop_after_shards:
                raise InterruptedError("DATA107_INTENTIONAL_INTERRUPTION")

    stable_shards = [
        {
            key: value
            for key, value in manifest.items()
            if key
            in {
                "logical_shard",
                "split",
                "relative_path",
                "plan_sha256",
                "target_shard_byte_tokens",
                "target_shard_size_bytes",
                "content_sha256",
                "manifest_sha256",
                "documents",
                "byte_tokens",
                "size_bytes",
                "by_source",
                "by_modality",
            }
        }
        for manifest in sorted(
            shard_manifests, key=lambda item: (item["split"], item["logical_shard"])
        )
    ]
    totals = Counter()
    for shard in stable_shards:
        totals["documents"] += int(shard["documents"])
        totals["byte_tokens"] += int(shard["byte_tokens"])
        totals["size_bytes"] += int(shard["size_bytes"])
    expected_documents = sum(int(item["documents"]) for item in ingest)
    expected_tokens = sum(int(item["byte_tokens"]) for item in ingest)
    if totals["documents"] != expected_documents or totals["byte_tokens"] != expected_tokens:
        raise ShardScaleError("published shard membership/counts do not match ingested corpus")

    core = {
        "schema": SHARDED_CORPUS_SCHEMA,
        "source_corpus_identity_sha256": source_corpus_identity_sha256,
        "plan_sha256": plan.sha256,
        "num_logical_shards": plan.num_shards,
        "sharding_version": "record-id-sha256-v1",
        "final_record_order": "record_id_lexicographic_external_merge_v1",
        "target_shard_byte_tokens": target_shard_byte_tokens,
        "target_shard_size_bytes": target_shard_size_bytes,
        "training_eligible": training_eligible,
        "truth_boundary": truth_boundary,
        "totals": dict(totals),
        "shards": stable_shards,
    }
    corpus_identity = sha256_bytes(canonical_bytes(core))
    manifest = {**core, "corpus_identity_sha256": corpus_identity}
    manifest_path = output_root / "manifest.json"
    manifest_partial = output_root / "manifest.json.partial"
    manifest_partial.write_bytes(canonical_bytes(manifest))
    os.replace(manifest_partial, manifest_path)
    shutil.rmtree(work_root, ignore_errors=True)
    observation = BuildObservation(
        workers=workers,
        wall_seconds=time.perf_counter() - start,
        peak_rss_bytes=_rss_bytes(),
        resumed_complete_shards=resumed,
        input_files=len(inputs),
    )
    return manifest, observation


def verify_sharded_corpus(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SHARDED_CORPUS_SCHEMA:
        raise ShardScaleError("unsupported sharded corpus manifest")
    core = {key: value for key, value in manifest.items() if key != "corpus_identity_sha256"}
    expected = sha256_bytes(canonical_bytes(core))
    if manifest.get("corpus_identity_sha256") != expected:
        raise ShardScaleError("global corpus identity mismatch")
    for shard in manifest["shards"]:
        data_path = root / shard["relative_path"]
        verified = _verify_complete_shard(data_path)
        if verified is None or verified.get("manifest_sha256") != shard["manifest_sha256"]:
            raise ShardScaleError(f"incomplete or corrupt shard {data_path}")
    return manifest


def write_scale_fixture(
    output_root: str | Path,
    *,
    records: int,
    text_bytes: int,
    input_parts: int,
) -> dict[str, Any]:
    """Write a large deterministic project-generated stress fixture, never corpus truth."""
    if records <= 0 or text_bytes < 128 or input_parts <= 0:
        raise ShardScaleError("invalid stress fixture dimensions")
    root = Path(output_root)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    handles = [(root / f"part-{index:03d}.jsonl").open("wb") for index in range(input_parts)]
    total = 0
    try:
        for index in range(records):
            modality = "code" if index % 5 == 0 else "natural"
            source_id = f"data107-stress:{'code' if modality == 'code' else 'text'}"
            prefix = (
                f"DATA107 stress record {index:08d} source={source_id} modality={modality}. "
            )
            filler = "deterministic storage streaming restart provenance scale "
            text = (prefix + filler * ((text_bytes // len(filler)) + 2))[:text_bytes]
            if len(text.encode("utf-8")) != text_bytes:
                raise AssertionError("stress fixture text must be exact ASCII byte length")
            row = {
                "id": f"data107-stress-{index:08d}",
                "text": text,
                "split": "train",
                "source_id": source_id,
                "source_version": "1",
                "modality": modality,
                "stratum": "code" if modality == "code" else "en",
                "external": False,
                "project_authored": True,
                "byte_tokens": text_bytes,
            }
            handles[index % input_parts].write(canonical_bytes(row))
            total += text_bytes
    finally:
        for handle in handles:
            handle.close()
    files = [
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.glob("part-*.jsonl"))
    ]
    core = {
        "schema": STRESS_FIXTURE_SCHEMA,
        "authority": "PROJECT_GENERATED_SCALE_FIXTURE_NOT_TRAINING_CORPUS_TRUTH",
        "training_eligible": False,
        "records": records,
        "text_bytes_per_record": text_bytes,
        "byte_tokens": total,
        "input_parts": input_parts,
        "files": files,
    }
    manifest = {**core, "fixture_identity_sha256": sha256_bytes(canonical_bytes(core))}
    (root / "manifest.json").write_bytes(canonical_bytes(manifest))
    return manifest


def data25_input_files(data25_root: str | Path) -> tuple[Path, ...]:
    root = Path(data25_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return tuple(root / item["path"] for item in manifest["shards"])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--records", type=int, default=4096)
    fixture.add_argument("--text-bytes", type=int, default=65536)
    fixture.add_argument("--input-parts", type=int, default=16)
    args = parser.parse_args()
    if args.command == "fixture":
        print(
            json.dumps(
                write_scale_fixture(
                    args.output,
                    records=args.records,
                    text_bytes=args.text_bytes,
                    input_parts=args.input_parts,
                ),
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
