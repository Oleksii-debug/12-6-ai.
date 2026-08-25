"""Physical corpus layouts that preserve an existing logical corpus identity."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from array import array
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from twelve_six.packing.core import TextRecord, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer

PHYSICAL_LAYOUT_SCHEMA = "12-6.corpus-physical-layout.v1"
PYARROW_EXPERIMENT_VERSION = "25.0.1"


class CorpusFormatError(ValueError):
    """Raised when physical layout conversion would alter corpus truth."""


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CorpusFormatError(f"{path}: JSON object required")
    return payload


def _source_identity(manifest: Mapping[str, Any]) -> str:
    for key in ("corpus_identity_sha256", "fixture_identity_sha256"):
        value = manifest.get(key)
        if isinstance(value, str) and len(value) == 64:
            return value
    raise CorpusFormatError("source manifest lacks a logical corpus/fixture identity")


def _source_entries(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = manifest.get("shards")
    if raw is None:
        raw = manifest.get("files")
    if not isinstance(raw, list) or not raw:
        raise CorpusFormatError("source manifest must contain non-empty shards/files")
    entries: list[Mapping[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise CorpusFormatError("source shard entry must be an object")
        entries.append(item)
    return tuple(entries)


def _entry_relative_path(entry: Mapping[str, Any]) -> str:
    value = entry.get("relative_path", entry.get("path"))
    if not isinstance(value, str) or not value:
        raise CorpusFormatError("source shard entry lacks path/relative_path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise CorpusFormatError("source shard path must stay below the corpus root")
    return value


def _entry_expected_hash(entry: Mapping[str, Any]) -> str | None:
    value = entry.get("content_sha256", entry.get("sha256"))
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise CorpusFormatError("source shard hash must be a SHA-256 digest")
    return value


def _record_id(row: Mapping[str, Any]) -> str:
    value = row.get("id", row.get("record_id"))
    if not isinstance(value, str) or not value:
        raise CorpusFormatError("logical record requires non-empty id/record_id")
    return value


def _record_text(row: Mapping[str, Any]) -> str:
    value = row.get("text")
    if not isinstance(value, str):
        raise CorpusFormatError(f"record {_record_id(row)!r} requires text")
    return value


def _record_split(row: Mapping[str, Any]) -> str:
    value = row.get("split", "train")
    if not isinstance(value, str) or not value:
        raise CorpusFormatError(f"record {_record_id(row)!r} requires split")
    return value


def _content_hash(row: Mapping[str, Any]) -> str:
    claimed = row.get("content_sha256")
    actual = hashlib.sha256(_record_text(row).encode("utf-8")).hexdigest()
    if claimed is not None:
        if not isinstance(claimed, str) or claimed != actual:
            raise CorpusFormatError(f"record {_record_id(row)!r} content hash drift")
    return actual


def iter_jsonl_rows(
    path: str | Path, *, verify_content: bool = False
) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusFormatError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(payload, dict):
                raise CorpusFormatError(f"{path}:{line_number}: object required")
            _record_id(payload)
            _record_text(payload)
            _record_split(payload)
            if verify_content:
                _content_hash(payload)
            yield payload


def require_pyarrow_version(expected: str = PYARROW_EXPERIMENT_VERSION) -> str:
    try:
        observed = metadata.version("pyarrow")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("Parquet corpus layouts require the maintained pyarrow runtime") from exc
    if observed != expected:
        raise RuntimeError(f"unsupported pyarrow version {observed}; expected exactly {expected}")
    return observed


def _import_parquet() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet corpus layouts require the maintained pyarrow runtime") from exc
    return pa, pq


@dataclass(slots=True)
class _Trace:
    rows: int = 0
    text_bytes: int = 0
    row_digest: Any = field(default_factory=hashlib.sha256, init=False, repr=False)
    identity_digest: Any = field(default_factory=hashlib.sha256, init=False, repr=False)

    def update(self, row: Mapping[str, Any]) -> None:
        record_id = _record_id(row)
        text = _record_text(row)
        content_hash = _content_hash(row)
        self.row_digest.update(canonical_bytes(dict(row)))
        self.identity_digest.update(
            canonical_bytes({"content_sha256": content_hash, "record_id": record_id})
        )
        self.rows += 1
        self.text_bytes += len(text.encode("utf-8"))

    def result(self) -> dict[str, Any]:
        return {
            "documents": self.rows,
            "logical_text_bytes": self.text_bytes,
            "ordered_logical_rows_sha256": self.row_digest.hexdigest(),
            "ordered_record_identity_sha256": self.identity_digest.hexdigest(),
        }


def _trace_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    trace = _Trace()
    for row in rows:
        trace.update(row)
    return trace.result()


def _layout_entries(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = manifest.get("shards")
    if not isinstance(raw, list) or not raw:
        raise CorpusFormatError("physical layout manifest has no shards")
    entries: list[Mapping[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise CorpusFormatError("physical shard entry must be an object")
        entries.append(item)
    return tuple(entries)


def iter_layout_rows(
    layout_root: str | Path,
    manifest: Mapping[str, Any],
    *,
    batch_rows: int = 4096,
) -> Iterator[dict[str, Any]]:
    root = Path(layout_root)
    physical_format = manifest.get("physical_format")
    if physical_format not in {"jsonl", "parquet"}:
        raise CorpusFormatError("unsupported physical layout format")
    if batch_rows <= 0:
        raise CorpusFormatError("batch_rows must be positive")
    if physical_format == "parquet":
        _, pq = _import_parquet()
    for entry in _layout_entries(manifest):
        relative_path = _entry_relative_path(entry)
        path = root / relative_path
        expected = _entry_expected_hash(entry)
        if expected is not None and sha256_file(path) != expected:
            raise CorpusFormatError(f"physical shard hash mismatch: {relative_path}")
        if physical_format == "jsonl":
            yield from iter_jsonl_rows(path)
            continue
        parquet = pq.ParquetFile(str(path))
        for batch in parquet.iter_batches(batch_size=batch_rows):
            for row in batch.to_pylist():
                if not isinstance(row, dict):
                    raise CorpusFormatError("Parquet batch row must be an object")
                _record_id(row)
                _record_text(row)
                _record_split(row)
                yield row


def layout_trace(layout_root: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return _trace_rows(iter_layout_rows(layout_root, manifest))


def iter_training_records(
    layout_root: str | Path,
    manifest: Mapping[str, Any],
    *,
    split: str = "train",
) -> Iterator[TextRecord]:
    for row in iter_layout_rows(layout_root, manifest):
        if _record_split(row) == split:
            yield TextRecord(record_id=_record_id(row), text=_record_text(row), split=split)


def _hash_short_array(digest: Any, values: Sequence[int]) -> None:
    packed = array("h", values)
    if sys.byteorder == "little":
        packed.byteswap()
    digest.update(len(values).to_bytes(4, "big"))
    digest.update(packed.tobytes())


def packed_training_trace(
    layout_root: str | Path,
    manifest: Mapping[str, Any],
    *,
    sequence_length: int = 128,
    split: str = "train",
) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    digest = hashlib.sha256()
    examples = 0
    loss_tokens = 0
    records = iter_training_records(layout_root, manifest, split=split)
    for example in iter_packed_examples(
        records,
        tokenizer,
        expected_split=split,
        sequence_length=sequence_length,
    ):
        _hash_short_array(digest, example.input_ids)
        _hash_short_array(digest, example.labels)
        digest.update(bytes(example.attention_mask))
        digest.update(bytes(example.loss_mask))
        for record_id in example.record_ids:
            encoded = record_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
        examples += 1
        loss_tokens += example.num_loss_tokens
    return {
        "sequence_length": sequence_length,
        "split": split,
        "examples": examples,
        "loss_tokens": loss_tokens,
        "packed_training_trace_sha256": digest.hexdigest(),
    }


def _prepare_output_root(output_root: Path) -> None:
    if output_root.exists():
        if any(output_root.iterdir()):
            raise CorpusFormatError(f"output root is not empty: {output_root}")
    else:
        output_root.mkdir(parents=True)


def _source_manifest_path(source_root: Path) -> Path:
    path = source_root / "manifest.json"
    if not path.is_file():
        raise CorpusFormatError(f"source manifest not found: {path}")
    return path


def _write_jsonl_shard(source: Path, target: Path) -> dict[str, Any]:
    trace = _Trace()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as output:
        for row in iter_jsonl_rows(source):
            trace.update(row)
            output.write(canonical_bytes(row))
    result = trace.result()
    return {
        **result,
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def _write_parquet_shard(
    source: Path,
    target: Path,
    *,
    compression: str,
    row_group_rows: int,
) -> dict[str, Any]:
    pa, pq = _import_parquet()
    trace = _Trace()
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    schema = None
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal writer, schema, pending
        if not pending:
            return
        table = pa.Table.from_pylist(pending)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(
                str(target),
                schema,
                compression=None if compression == "none" else compression,
                use_dictionary=True,
                write_statistics=True,
            )
        elif table.schema != schema:
            raise CorpusFormatError(f"row schema changed within shard {source}")
        writer.write_table(table, row_group_size=row_group_rows)
        pending = []

    try:
        for row in iter_jsonl_rows(source):
            trace.update(row)
            pending.append(row)
            if len(pending) >= row_group_rows:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise CorpusFormatError(f"source shard contains no records: {source}")
    parquet = pq.ParquetFile(str(target))
    result = trace.result()
    return {
        **result,
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "row_groups": parquet.metadata.num_row_groups,
    }


def materialize_layout(
    source_root: str | Path,
    output_root: str | Path,
    *,
    physical_format: str,
    compression: str = "none",
    row_group_rows: int = 4096,
    expected_pyarrow_version: str | None = None,
) -> dict[str, Any]:
    """Re-encode an existing logical corpus without changing rows, order, or identity."""
    if physical_format not in {"jsonl", "parquet"}:
        raise CorpusFormatError("physical_format must be jsonl or parquet")
    if row_group_rows <= 0:
        raise CorpusFormatError("row_group_rows must be positive")
    if physical_format == "jsonl" and compression != "none":
        raise CorpusFormatError("incumbent JSONL layout is intentionally uncompressed")
    if physical_format == "parquet" and compression not in {"none", "zstd"}:
        raise CorpusFormatError("PERF-146 compares only Parquet none/zstd")
    pyarrow_version = None
    if physical_format == "parquet":
        pyarrow_version = (
            require_pyarrow_version(expected_pyarrow_version)
            if expected_pyarrow_version is not None
            else metadata.version("pyarrow")
        )
        _import_parquet()

    source_root = Path(source_root)
    output_root = Path(output_root)
    _prepare_output_root(output_root)
    source_manifest_path = _source_manifest_path(source_root)
    source_manifest = _load_json(source_manifest_path)
    source_identity = _source_identity(source_manifest)
    entries = _source_entries(source_manifest)

    output_entries: list[dict[str, Any]] = []
    aggregate_source = _Trace()
    for entry in entries:
        relative = _entry_relative_path(entry)
        source_path = source_root / relative
        expected_hash = _entry_expected_hash(entry)
        if not source_path.is_file():
            raise CorpusFormatError(f"source shard missing: {relative}")
        if expected_hash is not None and sha256_file(source_path) != expected_hash:
            raise CorpusFormatError(f"source shard hash mismatch: {relative}")
        target_relative = str(
            Path(relative).with_suffix(".parquet")
            if physical_format == "parquet"
            else Path(relative).with_suffix(".jsonl")
        )
        target_path = output_root / target_relative
        if physical_format == "jsonl":
            stats = _write_jsonl_shard(source_path, target_path)
        else:
            stats = _write_parquet_shard(
                source_path,
                target_path,
                compression=compression,
                row_group_rows=row_group_rows,
            )
        # Fold per-shard trace into a corpus trace by replaying source rows exactly once.
        for row in iter_jsonl_rows(source_path):
            aggregate_source.update(row)
        output_entries.append(
            {
                "source_relative_path": relative,
                "relative_path": target_relative,
                **stats,
            }
        )

    source_trace = aggregate_source.result()
    core = {
        "schema": PHYSICAL_LAYOUT_SCHEMA,
        "source_logical_identity_sha256": source_identity,
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "physical_format": physical_format,
        "compression": compression,
        "row_group_rows": row_group_rows if physical_format == "parquet" else None,
        "pyarrow_version": pyarrow_version,
        "logical_trace": source_trace,
        "shards": output_entries,
        "data_bytes": sum(int(item["size_bytes"]) for item in output_entries),
    }
    layout_identity = hashlib.sha256(canonical_bytes(core)).hexdigest()
    manifest = {**core, "layout_identity_sha256": layout_identity}
    manifest_path = output_root / "manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest))

    observed = layout_trace(output_root, manifest)
    if observed != source_trace:
        shutil.rmtree(output_root, ignore_errors=True)
        raise CorpusFormatError("physical conversion changed logical rows or order")
    return manifest


def verify_layout(layout_root: str | Path) -> dict[str, Any]:
    root = Path(layout_root)
    manifest = _load_json(root / "manifest.json")
    if manifest.get("schema") != PHYSICAL_LAYOUT_SCHEMA:
        raise CorpusFormatError("unsupported physical layout manifest")
    core = {key: value for key, value in manifest.items() if key != "layout_identity_sha256"}
    expected_identity = hashlib.sha256(canonical_bytes(core)).hexdigest()
    if manifest.get("layout_identity_sha256") != expected_identity:
        raise CorpusFormatError("physical layout identity mismatch")
    for entry in _layout_entries(manifest):
        path = root / _entry_relative_path(entry)
        expected_hash = _entry_expected_hash(entry)
        if expected_hash is None or sha256_file(path) != expected_hash:
            raise CorpusFormatError(f"physical shard hash mismatch: {path}")
    observed = layout_trace(root, manifest)
    if observed != manifest.get("logical_trace"):
        raise CorpusFormatError("physical layout logical trace mismatch")
    return manifest


def seek_layout_row(
    layout_root: str | Path,
    manifest: Mapping[str, Any],
    *,
    shard_index: int,
    record_ordinal: int,
) -> dict[str, Any]:
    entries = _layout_entries(manifest)
    if not 0 <= shard_index < len(entries):
        raise CorpusFormatError("shard_index outside layout")
    if record_ordinal < 0:
        raise CorpusFormatError("record_ordinal must be non-negative")
    path = Path(layout_root) / _entry_relative_path(entries[shard_index])
    if manifest.get("physical_format") == "jsonl":
        for index, row in enumerate(iter_jsonl_rows(path)):
            if index == record_ordinal:
                return row
        raise CorpusFormatError("record ordinal outside JSONL shard")
    if manifest.get("physical_format") != "parquet":
        raise CorpusFormatError("unsupported physical layout format")
    _, pq = _import_parquet()
    parquet = pq.ParquetFile(str(path))
    remaining = record_ordinal
    for row_group in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(row_group).num_rows
        if remaining < count:
            rows = parquet.read_row_group(row_group).slice(remaining, 1).to_pylist()
            if not rows:
                break
            row = rows[0]
            if not isinstance(row, dict):
                raise CorpusFormatError("Parquet row must be an object")
            return row
        remaining -= count
    raise CorpusFormatError("record ordinal outside Parquet shard")


def layout_data_paths(layout_root: str | Path, manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    root = Path(layout_root)
    return tuple(root / _entry_relative_path(item) for item in _layout_entries(manifest))
