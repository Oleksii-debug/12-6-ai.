from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from twelve_six.tokenization.migration_compatibility import (
    TokenizerMigrationError,
    assess_tokenizer_checkpoint_migration,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _tokenizer(
    *,
    version: str = "learned-v1",
    config: str = "config-v1",
    vocab: str = "vocab-v1",
    vocab_size: int = 512,
    normalization: str = "none",
    encoding: str = "utf-8",
    special_tokens: dict[str, int] | None = None,
) -> dict:
    return {
        "version": version,
        "config_sha256": _sha(config),
        "vocab_sha256": _sha(vocab),
        "vocab_size": vocab_size,
        "normalization": normalization,
        "encoding": encoding,
        "special_tokens": special_tokens if special_tokens is not None else {"<unk>": 0},
    }


def _model(
    vocab_size: int = 512,
    *,
    tied: bool = True,
    lm_head_bias: bool = False,
) -> dict:
    return {
        "vocab_size": vocab_size,
        "tie_word_embeddings": tied,
        "lm_head_bias": lm_head_bias,
    }


def _checkpoint(tokenizer: dict, model: dict) -> dict:
    return {
        "tokenizer_hash": tokenizer["config_sha256"],
        "tokenizer_vocab_hash": tokenizer["vocab_sha256"],
        "model_spec": deepcopy(model),
    }


def _decision(
    source: dict,
    target: dict,
    *,
    source_model: dict | None = None,
    target_model: dict | None = None,
    reuse: bool = True,
) -> dict:
    source_model = source_model or _model(source["vocab_size"])
    target_model = target_model or _model(target["vocab_size"])
    return assess_tokenizer_checkpoint_migration(
        target_tokenizer_identity=target,
        target_model_spec=target_model,
        reuse_checkpoint_weights=reuse,
        source_tokenizer_identity=source,
        source_model_spec=source_model,
        checkpoint_identity=_checkpoint(source, source_model),
    )


def test_exact_identity_allows_only_tokenizer_compatibility_claim() -> None:
    source = _tokenizer()
    result = _decision(source, deepcopy(source))
    assert result["status"] == "EXACT_TOKENIZER_IDENTITY_COMPATIBLE"
    assert result["checkpoint_weight_reuse_allowed_by_tokenizer_contract"] is True
    assert result["full_checkpoint_compatibility_proven"] is False
    assert result["token_id_mapping_assumed"] is False
    assert result["training_authorized_by_this_contract"] is False


def test_same_vocab_size_but_changed_vocab_hash_blocks_reuse() -> None:
    source = _tokenizer(vocab="ordered-vocab-a")
    target = _tokenizer(vocab="ordered-vocab-permuted")
    result = _decision(source, target)
    assert result["status"] == "BLOCK_CHECKPOINT_REUSE_TOKENIZER_IDENTITY_CHANGED"
    assert result["changed_tokenizer_fields"] == ["vocab_sha256"]
    assert result["vocabulary_identity_changed"] is True
    assert result["vocabulary_parameter_surfaces_requiring_fresh_init_or_explicit_migration"] == [
        "token_embedding.weight"
    ]


def test_special_token_id_drift_blocks_reuse_even_when_vocab_size_is_unchanged() -> None:
    source = _tokenizer(special_tokens={"<unk>": 0, "<pad>": 1})
    target = _tokenizer(special_tokens={"<unk>": 1, "<pad>": 0})
    result = _decision(source, target)
    assert "special_tokens" in result["changed_tokenizer_fields"]
    assert result["checkpoint_weight_reuse_allowed_by_tokenizer_contract"] is False
    assert result["token_strings_used_to_infer_id_equivalence"] is False


def test_config_or_normalization_drift_blocks_silent_reuse_without_id_guessing() -> None:
    source = _tokenizer(config="config-a", normalization="none")
    target = _tokenizer(config="config-b", normalization="NFKC")
    result = _decision(source, target)
    assert result["changed_tokenizer_fields"] == ["config_sha256", "normalization"]
    assert result["vocabulary_identity_changed"] is False
    assert result["checkpoint_weight_reuse_allowed_by_tokenizer_contract"] is False
    assert result["vocabulary_parameter_surfaces_requiring_fresh_init_or_explicit_migration"] == []


def test_vocab_size_change_reports_all_untied_vocabulary_surfaces() -> None:
    source = _tokenizer(vocab_size=512)
    target = _tokenizer(vocab="vocab-768", vocab_size=768)
    result = _decision(
        source,
        target,
        source_model=_model(512, tied=False, lm_head_bias=True),
        target_model=_model(768, tied=False, lm_head_bias=True),
    )
    assert result["vocabulary_identity_changed"] is True
    assert result["vocabulary_parameter_surfaces_requiring_fresh_init_or_explicit_migration"] == [
        "token_embedding.weight",
        "lm_head.weight",
        "lm_head.bias",
    ]


def test_future_fresh_stage_does_not_assume_learned20m_tokenizer() -> None:
    target = _tokenizer(version="future-100m-tokenizer", vocab="future-vocab", vocab_size=1024)
    result = assess_tokenizer_checkpoint_migration(
        target_tokenizer_identity=target,
        target_model_spec=_model(1024),
        reuse_checkpoint_weights=False,
    )
    assert result["status"] == "FRESH_INITIALIZATION_TARGET_TOKENIZER_BOUND"
    assert result["source_identity_present"] is False
    assert result["source_tokenizer_identity"] is None
    assert result["token_id_mapping_assumed"] is False


def test_changed_tokenizer_with_source_but_no_reuse_requires_fresh_initialization() -> None:
    source = _tokenizer()
    target = _tokenizer(vocab="new-vocab")
    result = _decision(source, target, reuse=False)
    assert result["status"] == "FRESH_INITIALIZATION_REQUIRED_TOKENIZER_CHANGED"
    assert result["checkpoint_weight_reuse_allowed_by_tokenizer_contract"] is False


def test_target_model_vocab_must_match_target_tokenizer() -> None:
    with pytest.raises(TokenizerMigrationError, match="target ModelSpec vocab_size"):
        assess_tokenizer_checkpoint_migration(
            target_tokenizer_identity=_tokenizer(vocab_size=768),
            target_model_spec=_model(512),
            reuse_checkpoint_weights=False,
        )


def test_checkpoint_source_binding_must_match_exact_source_vocab_identity() -> None:
    source = _tokenizer()
    target = deepcopy(source)
    source_model = _model()
    checkpoint = _checkpoint(source, source_model)
    checkpoint["tokenizer_vocab_hash"] = _sha("wrong-vocab")
    with pytest.raises(TokenizerMigrationError, match="checkpoint vocabulary hash"):
        assess_tokenizer_checkpoint_migration(
            target_tokenizer_identity=target,
            target_model_spec=_model(),
            reuse_checkpoint_weights=True,
            source_tokenizer_identity=source,
            source_model_spec=source_model,
            checkpoint_identity=checkpoint,
        )


def test_partial_source_identity_is_rejected_instead_of_inferred() -> None:
    with pytest.raises(TokenizerMigrationError, match="must be supplied together"):
        assess_tokenizer_checkpoint_migration(
            target_tokenizer_identity=_tokenizer(),
            target_model_spec=_model(),
            reuse_checkpoint_weights=False,
            source_tokenizer_identity=_tokenizer(),
        )
