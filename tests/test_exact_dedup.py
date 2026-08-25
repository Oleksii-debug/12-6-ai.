import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.data.corpus_foundation import CorpusFoundationError
from twelve_six.data.exact_dedup import ExactDedupPolicy, run_exact_dedup


def cjson(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def row(record_id, source_id, text, *, split="train", stratum="en", modality="natural"):
    raw = text.encode()
    return {
        "record_id": record_id,
        "source_id": source_id,
        "source_version": "v1",
        "stratum": stratum,
        "modality": modality,
        "split": split,
        "content_sha256": sha(raw),
        "byte_tokens": len(raw),
        "text": text,
    }


def corpus(tmp_path: Path, groups):
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    shards = []
    for index, rows in enumerate(groups):
        path = shard_dir / f"part-{index:05d}.jsonl"
        payload = b"".join(cjson(item) for item in rows)
        path.write_bytes(payload)
        shards.append(
            {
                "path": f"shards/{path.name}",
                "sha256": sha(payload),
                "size_bytes": len(payload),
                "documents": len(rows),
            }
        )
    core = {
        "schema_version": "12-6.corpus-manifest.v1",
        "corpus_version": "test",
        "shards": shards,
    }
    manifest = {**core, "corpus_identity_sha256": sha(cjson(core))}
    path = tmp_path / "manifest.json"
    path.write_bytes(cjson(manifest))
    return path


def test_real_fields_exact_key_alias_provenance_and_distributions(tmp_path):
    shared = "normalized identical content"
    manifest = corpus(
        tmp_path,
        [
            [
                row("a1", "source-a", shared),
                row("a2", "source-a", shared),
                row("v1", "source-a", "validation", split="validation"),
            ],
            [
                row("b1", "source-b", shared),
                row("b2", "source-b", "unique code", stratum="code", modality="code"),
            ],
        ],
    )
    result = run_exact_dedup(corpus_manifest=manifest, output_dir=tmp_path / "out")
    m = result["metrics"]
    assert (m["input_documents"], m["output_documents"], m["documents_removed"]) == (4, 2, 2)
    assert m["exact_duplicate_groups"] == 1
    assert m["cross_source_duplicate_aliases"] == 1
    assert m["within_source_duplicate_aliases"] == 1
    assert m["skipped_non_train_documents"] == 1
    aliases = [
        json.loads(line)
        for line in (tmp_path / "out/evidence/discarded_aliases.jsonl")
        .read_text()
        .splitlines()
    ]
    assert aliases[0]["winner"]["record_id"] == "a1"
    assert aliases[0]["alias"]["record_id"] == "a2"
    assert aliases[1]["alias"]["record_id"] == "b1"


def test_policy_change_changes_corpus_identity(tmp_path):
    manifest = corpus(tmp_path, [[row("x", "s", "one normalized record")]])
    a = run_exact_dedup(corpus_manifest=manifest, output_dir=tmp_path / "a")
    b = run_exact_dedup(
        corpus_manifest=manifest,
        output_dir=tmp_path / "b",
        policy=ExactDedupPolicy(output_serialization="canonical-jsonl-utf8-v1-revision"),
    )
    assert a["corpus_identity_sha256"] != b["corpus_identity_sha256"]


def test_shard_boundary_restart_matches_clean_run(tmp_path):
    manifest = corpus(
        tmp_path,
        [
            [row("a", "s", "alpha")],
            [row("b", "s", "beta")],
            [row("c", "s", "gamma")],
        ],
    )
    clean = run_exact_dedup(corpus_manifest=manifest, output_dir=tmp_path / "clean")
    partial = run_exact_dedup(
        corpus_manifest=manifest,
        output_dir=tmp_path / "resume",
        stop_after_input_shards=1,
    )
    assert partial["status"] == "PARTIAL"
    resumed = run_exact_dedup(
        corpus_manifest=manifest,
        output_dir=tmp_path / "resume",
        resume=True,
    )
    assert resumed["corpus_identity_sha256"] == clean["corpus_identity_sha256"]
    assert resumed["run_sha256"] == clean["run_sha256"]


def test_hash_mismatch_fails_closed(tmp_path):
    bad = row("x", "s", "normalized")
    bad["content_sha256"] = "0" * 64
    manifest = corpus(tmp_path, [[bad]])
    with pytest.raises(CorpusFoundationError, match="hash mismatch"):
        run_exact_dedup(corpus_manifest=manifest, output_dir=tmp_path / "out")


def test_manifest_identity_mismatch_fails_closed(tmp_path):
    manifest = corpus(tmp_path, [[row("x", "s", "normalized")]])
    value = json.loads(manifest.read_text())
    value["corpus_identity_sha256"] = "0" * 64
    manifest.write_bytes(cjson(value))
    with pytest.raises(CorpusFoundationError, match="identity mismatch"):
        run_exact_dedup(corpus_manifest=manifest, output_dir=tmp_path / "out")
