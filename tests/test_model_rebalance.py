from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six.model import load_stage_config
from twelve_six.model_rebalance import (
    GeometryConstraints,
    TokenizerArtifactIdentity,
    bound_modelspec_identity_sha256,
    build_stage_candidate_table,
    infer_tokenizer_vocab_size,
    one_training_step_smoke,
    search_model_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATH = ROOT / "configs/vocabulary/measured_bpe_472_tokenizer_identity.v1.json"
PROFILES_PATH = ROOT / "configs/vocabulary/model_rebalance_profiles.v1.json"


def _measured_bpe() -> TokenizerArtifactIdentity:
    return TokenizerArtifactIdentity.from_descriptor(IDENTITY_PATH)


def test_tokenizer_artifact_hash_and_actual_id_cardinality_are_exact(tmp_path: Path) -> None:
    payload = {
        "model": {"vocab": {"a": 0, "b": 1}},
        "added_tokens": [{"id": 2, "content": "<experimental>"}],
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    path = tmp_path / "tokenizer.json"
    path.write_bytes(raw)
    identity = TokenizerArtifactIdentity.from_artifact(path)
    assert identity.vocab_size == 3
    assert identity.tokenizer_json_sha256 == hashlib.sha256(raw).hexdigest()
    assert identity.source_kind == "artifact_bytes"


def test_sparse_token_ids_size_embedding_by_max_id_plus_one() -> None:
    assert infer_tokenizer_vocab_size({"model": {"vocab": {"a": 0, "b": 7}}}) == 8


def test_measured_bpe_identity_matches_retained_real_experiment() -> None:
    identity = _measured_bpe()
    assert identity.vocab_size == 472
    assert (
        identity.tokenizer_json_sha256
        == "006c84fc0d05d3bedb5b0bceb587aab1631dd0295cc2063e97823c2121e08be0"
    )
    assert (
        identity.source_evidence_sha256
        == "3307daec835e96a63fd5a7d14543de3d9f0781ec3f7d4d0cc98d992bf0f8bae6"
    )


def test_100k_search_reuses_incumbent_and_finds_known_rebalance() -> None:
    s1 = load_stage_config(ROOT / "configs/stages/s1_100k.json")
    result = search_model_geometry(
        s1.model,
        target_parameters=100_000,
        tokenizer=_measured_bpe(),
        constraints=GeometryConstraints(
            n_layers=(2, 3, 4),
            n_heads=(2, 4),
            head_dims=(8, 12, 16),
            max_candidates=5,
        ),
    )
    winner = result.candidates[0]
    assert winner.parameter_count == 99_024
    assert winner.target_delta == -976
    assert winner.model.d_model == 48
    assert winner.model.n_layers == 3
    assert winner.model.n_heads == 4
    assert winner.model.n_kv_heads == 4
    assert winner.model.head_dim == 12
    assert winner.model.d_ff == 112
    assert winner.embedding_parameters == 22_656
    assert winner.embedding_fraction == pytest.approx(22_656 / 99_024)
    assert winner.block_fraction == pytest.approx(winner.block_parameters / 99_024)
    assert winner.head_valid is True


def test_bound_identity_changes_with_exact_tokenizer_artifact_hash() -> None:
    s1 = load_stage_config(ROOT / "configs/stages/s1_100k.json")
    tokenizer_a = _measured_bpe()
    tokenizer_b = replace(tokenizer_a, tokenizer_json_sha256="f" * 64)
    assert tokenizer_a.identity_sha256() != tokenizer_b.identity_sha256()
    assert (
        bound_modelspec_identity_sha256(s1.model, tokenizer_a)
        != bound_modelspec_identity_sha256(s1.model, tokenizer_b)
    )


def test_unreasonable_embedding_share_fails_closed() -> None:
    s1 = load_stage_config(ROOT / "configs/stages/s1_100k.json")
    oversized = TokenizerArtifactIdentity(
        vocab_size=4096,
        tokenizer_json_sha256="a" * 64,
        source_kind="synthetic_test",
    )
    result = search_model_geometry(
        s1.model,
        target_parameters=100_000,
        tokenizer=oversized,
        constraints=GeometryConstraints(
            n_layers=(2, 3, 4),
            n_heads=(2, 4),
            head_dims=(8, 12, 16),
            max_embedding_fraction=0.30,
            max_target_delta_fraction=0.10,
        ),
    )
    assert result.candidates == ()
    assert sum(result.rejected.values()) > 0


def test_stage_table_covers_required_parameter_targets() -> None:
    table = build_stage_candidate_table(
        profiles_path=PROFILES_PATH,
        tokenizer=_measured_bpe(),
        repo_root=ROOT,
    )
    assert [stage["target_parameters"] for stage in table["stages"]] == [
        100_000,
        250_000,
        500_000,
        1_000_000,
        10_000_000,
    ]
    winners = {stage["stage"]: stage["candidates"][0] for stage in table["stages"]}
    assert winners["100K"]["parameter_count"] == 99_024
    assert winners["250K"]["parameter_count"] == 249_920
    assert winners["500K"]["parameter_count"] == 497_760
    assert winners["1M"]["parameter_count"] == 999_552
    assert winners["10M"]["parameter_count"] == 9_997_568
    assert all(winner["head_valid"] for winner in winners.values())
    assert all(winner["embedding_fraction"] <= 0.30 for winner in winners.values())


def test_representative_real_models_construct_and_train_one_step() -> None:
    table = build_stage_candidate_table(
        profiles_path=PROFILES_PATH,
        tokenizer=_measured_bpe(),
        repo_root=ROOT,
    )
    by_stage = {stage["stage"]: stage for stage in table["stages"]}
    for stage_name in ("100K", "1M"):
        profile = next(
            profile
            for profile in json.loads(PROFILES_PATH.read_text())["profiles"]
            if profile["stage"] == stage_name
        )
        base = load_stage_config(ROOT / profile["base_stage_config"]).model
        result = search_model_geometry(
            base,
            target_parameters=profile["target_parameters"],
            tokenizer=_measured_bpe(),
            constraints=GeometryConstraints.from_dict(profile["constraints"]),
        )
        winner = result.candidates[0]
        assert winner.parameter_count == by_stage[stage_name]["candidates"][0]["parameter_count"]
        smoke = one_training_step_smoke(winner, sequence_length=4, seed=20260825)
        assert smoke["status"] == "PASS"
        assert smoke["optimizer_steps"] == 1
        assert smoke["parameter_changed"] is True
        assert smoke["parameter_count"] == winner.parameter_count
