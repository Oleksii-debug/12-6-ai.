from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data.corpus_format import (
    CorpusFormatError,
    PYARROW_EXPERIMENT_VERSION,
    canonical_bytes,
    materialize_layout,
    packed_training_trace,
    seek_layout_row,
    sha256_file,
    verify_layout,
)


def _source(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    root = tmp_path / "source"
    shards = root / "shards"
    shards.mkdir(parents=True)
    rows = [
        {
            "record_id": f"record-{index:03d}",
            "source_id": "fixture",
            "source_version": "1",
            "stratum": "en" if index % 2 else "uk",
            "modality": "natural",
            "split": "validation" if index in {3, 8} else "train",
            "external": False,
            "project_authored": True,
            "content_sha256": hashlib.sha256(
                (
                    f"document {index} українська English text " * (index + 2)
                ).encode("utf-8")
            ).hexdigest(),
            "byte_tokens": len(
                (
                    f"document {index} українська English text " * (index + 2)
                ).encode("utf-8")
            ),
            "text": f"document {index} українська English text " * (index + 2),
        }
        for index in range(12)
    ]
    groups = (rows[:5], rows[5:])
    entries = []
    for index, group in enumerate(groups):
        path = shards / f"part-{index:02d}.jsonl"
        path.write_bytes(b"".join(canonical_bytes(row) for row in group))
        entries.append(
            {
                "path": f"shards/{path.name}",
                "sha256": sha256_file(path),
                "documents": len(group),
            }
        )
    core = {"schema": "test-corpus", "shards": entries}
    identity = hashlib.sha256(canonical_bytes(core)).hexdigest()
    manifest = {**core, "corpus_identity_sha256": identity}
    (root / "manifest.json").write_bytes(canonical_bytes(manifest))
    return root, rows


def test_jsonl_reencoding_preserves_logical_identity_and_seek(tmp_path: Path) -> None:
    source, rows = _source(tmp_path)
    output = tmp_path / "jsonl"
    manifest = materialize_layout(source, output, physical_format="jsonl")
    verified = verify_layout(output)

    assert verified["source_logical_identity_sha256"] == json.loads(
        (source / "manifest.json").read_text(encoding="utf-8")
    )["corpus_identity_sha256"]
    assert manifest["logical_trace"]["documents"] == len(rows)
    assert seek_layout_row(output, manifest, shard_index=0, record_ordinal=4) == rows[4]
    assert seek_layout_row(output, manifest, shard_index=1, record_ordinal=2) == rows[7]


def test_parquet_none_and_zstd_preserve_rows_and_full_packed_trace(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source, rows = _source(tmp_path)
    jsonl_root = tmp_path / "jsonl"
    none_root = tmp_path / "parquet-none"
    zstd_root = tmp_path / "parquet-zstd"
    layouts = {
        "jsonl": materialize_layout(source, jsonl_root, physical_format="jsonl"),
        "none": materialize_layout(
            source,
            none_root,
            physical_format="parquet",
            compression="none",
            row_group_rows=3,
            expected_pyarrow_version=PYARROW_EXPERIMENT_VERSION,
        ),
        "zstd": materialize_layout(
            source,
            zstd_root,
            physical_format="parquet",
            compression="zstd",
            row_group_rows=3,
            expected_pyarrow_version=PYARROW_EXPERIMENT_VERSION,
        ),
    }

    logical = {item["source_logical_identity_sha256"] for item in layouts.values()}
    row_traces = {
        item["logical_trace"]["ordered_logical_rows_sha256"] for item in layouts.values()
    }
    id_traces = {
        item["logical_trace"]["ordered_record_identity_sha256"] for item in layouts.values()
    }
    packed = {
        name: packed_training_trace(root, layouts[name], sequence_length=32)
        for name, root in {
            "jsonl": jsonl_root,
            "none": none_root,
            "zstd": zstd_root,
        }.items()
    }

    assert len(logical) == len(row_traces) == len(id_traces) == 1
    assert len({item["packed_training_trace_sha256"] for item in packed.values()}) == 1
    assert len({item["loss_tokens"] for item in packed.values()}) == 1
    assert seek_layout_row(none_root, layouts["none"], shard_index=1, record_ordinal=6) == rows[11]
    assert seek_layout_row(zstd_root, layouts["zstd"], shard_index=0, record_ordinal=1) == rows[1]


def test_source_hash_drift_fails_before_materialization(tmp_path: Path) -> None:
    source, _ = _source(tmp_path)
    shard = source / "shards/part-00.jsonl"
    shard.write_text(shard.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(CorpusFormatError, match="source shard hash mismatch"):
        materialize_layout(source, tmp_path / "out", physical_format="jsonl")


def test_nonempty_output_root_fails_closed(tmp_path: Path) -> None:
    source, _ = _source(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "sentinel").write_text("keep", encoding="utf-8")

    with pytest.raises(CorpusFormatError, match="not empty"):
        materialize_layout(source, output, physical_format="jsonl")
    assert (output / "sentinel").read_text(encoding="utf-8") == "keep"
