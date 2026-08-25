from __future__ import annotations

import json

import pytest

from twelve_six.data.shard_scale import (
    ShardScaleError,
    build_sharded_corpus,
    verify_sharded_corpus,
    write_scale_fixture,
)
from twelve_six.packing.core import PACKING_CONFIG_HASH
from twelve_six.packing.scale_contracts import MixturePlan, MixtureSource
from twelve_six.tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH


def _plan(identity: str, *, shards: int = 8) -> MixturePlan:
    return MixturePlan(
        plan_id="data107-test",
        tokenizer_config_sha256=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_sha256=BYTE_VOCAB_HASH,
        packing_config_sha256=PACKING_CONFIG_HASH,
        sources=(MixtureSource("fixture", identity, 1),),
        seed=107,
        num_shards=shards,
        shard_seed=107_2026,
    )


def _inputs(root, fixture):
    return tuple(root / item["path"] for item in fixture["files"])


def _signature(manifest):
    return [
        (
            item["split"],
            item["logical_shard"],
            item["content_sha256"],
            item["manifest_sha256"],
            item["documents"],
            item["byte_tokens"],
        )
        for item in manifest["shards"]
    ]


def test_worker_count_does_not_change_logical_or_physical_identity(tmp_path) -> None:
    source = tmp_path / "source"
    fixture = write_scale_fixture(source, records=96, text_bytes=512, input_parts=7)
    identity = fixture["fixture_identity_sha256"]
    plan = _plan(identity)
    inputs = _inputs(source, fixture)
    one, _ = build_sharded_corpus(
        inputs,
        tmp_path / "one",
        source_corpus_identity_sha256=identity,
        plan=plan,
        workers=1,
        target_shard_byte_tokens=6144,
        target_shard_size_bytes=8192,
        sort_chunk_bytes=2048,
        training_eligible=False,
        truth_boundary="TEST_FIXTURE",
    )
    three, _ = build_sharded_corpus(
        tuple(reversed(inputs)),
        tmp_path / "three",
        source_corpus_identity_sha256=identity,
        plan=plan,
        workers=3,
        target_shard_byte_tokens=6144,
        target_shard_size_bytes=8192,
        sort_chunk_bytes=2048,
        training_eligible=False,
        truth_boundary="TEST_FIXTURE",
    )
    assert one["corpus_identity_sha256"] == three["corpus_identity_sha256"]
    assert _signature(one) == _signature(three)
    assert (tmp_path / "one/manifest.json").read_bytes() == (
        tmp_path / "three/manifest.json"
    ).read_bytes()


def test_interrupted_build_publishes_no_global_manifest_and_resumes(tmp_path) -> None:
    source = tmp_path / "source"
    fixture = write_scale_fixture(source, records=80, text_bytes=384, input_parts=5)
    identity = fixture["fixture_identity_sha256"]
    plan = _plan(identity)
    inputs = _inputs(source, fixture)
    root = tmp_path / "resumed"
    with pytest.raises(InterruptedError, match="DATA107_INTENTIONAL_INTERRUPTION"):
        build_sharded_corpus(
            inputs,
            root,
            source_corpus_identity_sha256=identity,
            plan=plan,
            workers=2,
            target_shard_byte_tokens=4096,
            target_shard_size_bytes=8192,
            sort_chunk_bytes=1024,
            stop_after_shards=2,
            training_eligible=False,
            truth_boundary="TEST_FIXTURE",
        )
    assert not (root / "manifest.json").exists()
    assert len(list(root.glob("train/*.complete"))) == 2
    assert not list(root.glob("train/*.partial"))

    resumed, observation = build_sharded_corpus(
        inputs,
        root,
        source_corpus_identity_sha256=identity,
        plan=plan,
        workers=3,
        target_shard_byte_tokens=4096,
        target_shard_size_bytes=8192,
        sort_chunk_bytes=1024,
        training_eligible=False,
        truth_boundary="TEST_FIXTURE",
    )
    assert observation.resumed_complete_shards == 2
    assert verify_sharded_corpus(root)["corpus_identity_sha256"] == resumed[
        "corpus_identity_sha256"
    ]


def test_verifier_fails_closed_on_shard_tamper(tmp_path) -> None:
    source = tmp_path / "source"
    fixture = write_scale_fixture(source, records=24, text_bytes=256, input_parts=3)
    identity = fixture["fixture_identity_sha256"]
    plan = _plan(identity, shards=4)
    root = tmp_path / "build"
    manifest, _ = build_sharded_corpus(
        _inputs(source, fixture),
        root,
        source_corpus_identity_sha256=identity,
        plan=plan,
        workers=1,
        target_shard_byte_tokens=2048,
        target_shard_size_bytes=4096,
        sort_chunk_bytes=512,
        training_eligible=False,
        truth_boundary="TEST_FIXTURE",
    )
    first = root / manifest["shards"][0]["relative_path"]
    with first.open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(ShardScaleError, match="incomplete or corrupt shard"):
        verify_sharded_corpus(root)


def test_shard_manifests_include_source_and_modality_counts(tmp_path) -> None:
    source = tmp_path / "source"
    fixture = write_scale_fixture(source, records=30, text_bytes=256, input_parts=3)
    identity = fixture["fixture_identity_sha256"]
    plan = _plan(identity, shards=4)
    root = tmp_path / "build"
    manifest, _ = build_sharded_corpus(
        _inputs(source, fixture),
        root,
        source_corpus_identity_sha256=identity,
        plan=plan,
        workers=1,
        target_shard_byte_tokens=2048,
        target_shard_size_bytes=4096,
        training_eligible=False,
        truth_boundary="TEST_FIXTURE",
    )
    shard = manifest["shards"][0]
    physical_manifest = json.loads(
        (root / shard["relative_path"]).with_suffix(".manifest.json").read_text()
    )
    assert physical_manifest["by_source"]
    assert physical_manifest["by_modality"]
    assert physical_manifest["content_sha256"] == shard["content_sha256"]
    assert physical_manifest["manifest_sha256"] == shard["manifest_sha256"]
