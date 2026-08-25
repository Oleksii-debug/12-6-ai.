from __future__ import annotations

import json
from pathlib import Path

from twelve_six.model import ModelSpec, TwelveSixDecoder, count_trainable_parameters


CONFIG = Path("configs/experiments/model16_weight_tying_250k.v1.json")


def _specs() -> tuple[ModelSpec, ModelSpec]:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    tied = ModelSpec.from_dict(payload["candidates"]["A_tied"]["model"])
    untied = ModelSpec.from_dict(payload["candidates"]["B_untied"]["model"])
    return tied, untied


def test_model16_candidates_are_matched_and_identity_distinct() -> None:
    tied, untied = _specs()
    assert tied.tie_word_embeddings is True
    assert untied.tie_word_embeddings is False
    assert tied.parameter_count() == 267_912
    assert untied.parameter_count() == 267_336
    assert abs(tied.parameter_count() - untied.parameter_count()) / tied.parameter_count() < 0.01
    assert untied.parameter_count() < tied.parameter_count()
    assert tied.identity_sha256() != untied.identity_sha256()
    assert tied.d_ff == 192
    assert untied.d_ff == 170


def test_model16_runtime_alias_semantics_match_modelspec() -> None:
    tied, untied = _specs()
    tied_model = TwelveSixDecoder(tied)
    untied_model = TwelveSixDecoder(untied)
    assert tied_model.lm_head.weight is tied_model.token_embedding.weight
    assert untied_model.lm_head.weight is not untied_model.token_embedding.weight
    assert count_trainable_parameters(tied_model) == tied.parameter_count()
    assert count_trainable_parameters(untied_model) == untied.parameter_count()
    # PyTorch state_dict exposes both names even for a shared Parameter. The
    # ModelSpec/checkpoint identity, not key presence alone, carries tie semantics.
    assert "token_embedding.weight" in tied_model.state_dict()
    assert "lm_head.weight" in tied_model.state_dict()
