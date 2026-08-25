from __future__ import annotations

import json
from pathlib import Path

from twelve_six.data.corpus_v02 import (
    ORIGIN_PROJECT,
    _exact_dedup,
    _policy,
    _project_rows,
    _require_config,
)
from twelve_six.model import ModelSpec
from twelve_six.tokenization.base import TokenizerIdentity


def _config() -> dict[str, object]:
    return {
        "schema_version": "12-6.corpus-build-config.v2",
        "corpus_version": "0.2.0",
        "external_candidate_registry": "configs/data/external_source_candidates_ua_en_v1.json",
        "target_pre_filter_byte_tokens": {"uk": 500, "en": 500, "code": 500},
        "validation_basis_points": 500,
        "split_salt": "test",
        "shard_target_bytes": 1024,
        "project_authored": {
            "enabled": True,
            "source_version": "0.1.0",
            "max_candidates_per_stratum": 100,
        },
        "near_dedup": {
            "natural": {
                "name": "natural_calibrated_5g_20x5",
                "n_grams": 5,
                "num_buckets": 20,
                "hashes_per_bucket": 5,
            },
            "code": {
                "name": "code_calibrated_4g_16x6",
                "n_grams": 4,
                "num_buckets": 16,
                "hashes_per_bucket": 6,
            },
            "seed": 1,
            "hash_precision": 64,
        },
        "decontamination": {
            "benchmark_registry": {
                "schema_version": "12-6.benchmark-registry.v1",
                "benchmarks": [],
                "manifest_sha256": "10f7454f77eb2dc3871eeafa5055b1969eab42954eb8e19e61565f217c67df31",
            },
            "n_grams": 5,
            "num_buckets": 14,
            "hashes_per_bucket": 8,
            "seed": 1,
            "hash_precision": 64,
        },
        "tokenizer": {
            "algorithm": "bpe",
            "library": "tokenizers",
            "version": "0.23.1",
            "vocab_size": 512,
            "min_frequency": 2,
        },
    }


def test_config_and_calibrated_policies() -> None:
    config = _require_config(_config())
    natural = _policy(config, "natural")
    code = _policy(config, "code")
    assert (natural.n_grams, natural.num_buckets, natural.hashes_per_bucket) == (5, 20, 5)
    assert (code.n_grams, code.num_buckets, code.hashes_per_bucket) == (4, 16, 6)


def test_project_material_never_relabels_as_external() -> None:
    rows = _project_rows(_config(), [])
    assert rows
    assert {row["origin_class"] for row in rows} == {ORIGIN_PROJECT}
    assert all(row["project_authored"] is True for row in rows)
    assert all(row["external"] is False for row in rows)


def test_exact_dedup_reuses_incumbent_sqlite_index(tmp_path: Path) -> None:
    row = {
        "record_id": "a",
        "source_id": "project-authored:test",
        "source_version": "1",
        "stratum": "en",
        "modality": "natural",
        "origin_class": ORIGIN_PROJECT,
        "external": False,
        "project_authored": True,
        "raw_identity": "0" * 64,
        "rights_status": "PROJECT_CONTROLLED",
        "license_id": "PROJECT_AUTHORED",
        "text": "A deterministic sufficiently long training record for exact duplicate checking. " * 2,
    }
    duplicate = {**row, "record_id": "b"}
    survivors, report = _exact_dedup([row, duplicate], tmp_path / "dedup.sqlite3")
    assert [item["record_id"] for item in survivors] == ["a"]
    assert report["removed_documents"] == 1
    assert report["engine"] == "incumbent_SQLiteExactDedupIndex"


def test_bpe_512_model_stays_in_100k_vertical() -> None:
    spec = ModelSpec(
        schema_version=1,
        vocab_size=512,
        max_seq_len=256,
        d_model=48,
        n_layers=3,
        n_heads=4,
        n_kv_heads=4,
        head_dim=12,
        d_ff=128,
        rope_rotary_dim=12,
    )
    assert spec.parameter_count() == 107856
    assert 100000 <= spec.parameter_count() < 1000000


def test_tokenizer_identity_dict_alias_is_semantic_alias() -> None:
    identity = TokenizerIdentity(
        version="x",
        config_sha256="0" * 64,
        vocab_sha256="1" * 64,
        vocab_size=512,
        normalization="none",
        encoding="utf-8",
        special_tokens={"<unk>": 0},
    )
    assert identity.as_dict() == identity.to_dict()


def test_checked_in_config_matches_schema() -> None:
    config_path = Path("configs/data/corpus_v02.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert _require_config(config)["corpus_version"] == "0.2.0"
