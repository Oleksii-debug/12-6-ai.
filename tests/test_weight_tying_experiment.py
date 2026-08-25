from __future__ import annotations

import json
from pathlib import Path

from twelve_six.model import InitSpec, TwelveSixDecoder
from twelve_six.weight_tying_experiment import matched_specs


def test_model16_candidates_are_near_iso_parameter_and_identity_distinct() -> None:
    (tied_name, tied), (untied_name, untied) = matched_specs()
    assert tied_name == "tied"
    assert untied_name == "untied"
    assert tied.parameter_count() == 267_912
    assert untied.parameter_count() == 268_200
    assert abs(untied.parameter_count() - tied.parameter_count()) / tied.parameter_count() < 0.002
    assert tied.d_model == untied.d_model == 72
    assert tied.n_layers == untied.n_layers == 4
    assert tied.n_heads == untied.n_heads == 6
    assert tied.n_kv_heads == untied.n_kv_heads == 6
    assert tied.head_dim == untied.head_dim == 12
    assert tied.max_seq_len == untied.max_seq_len == 256
    assert tied.vocab_size == untied.vocab_size == 256
    assert tied.d_ff == 192
    assert untied.d_ff == 171
    assert tied.tie_word_embeddings is True
    assert untied.tie_word_embeddings is False
    assert tied.identity_sha256() != untied.identity_sha256()


def test_model16_runtime_alias_and_state_dict_semantics() -> None:
    (_, tied), (_, untied) = matched_specs()
    tied_model = TwelveSixDecoder(tied, InitSpec())
    untied_model = TwelveSixDecoder(untied, InitSpec())

    assert tied_model.lm_head.weight is tied_model.token_embedding.weight
    assert untied_model.lm_head.weight is not untied_model.token_embedding.weight

    tied_state = tied_model.state_dict()
    untied_state = untied_model.state_dict()
    for state in (tied_state, untied_state):
        assert "token_embedding.weight" in state
        assert "lm_head.weight" in state

    assert tied_state["token_embedding.weight"].data_ptr() == tied_state["lm_head.weight"].data_ptr()
    assert untied_state["token_embedding.weight"].data_ptr() != untied_state["lm_head.weight"].data_ptr()


def test_model16_config_matches_executable_specs() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "configs/research/model16_weight_tying_250k.json").read_text(encoding="utf-8")
    )
    (_, tied), (_, untied) = matched_specs()
    assert payload["authority"] == "RESEARCH_ONLY_NOT_CANONICAL"
    assert payload["controls"]["optimized_tokens"] == 32_760
    assert payload["candidates"]["tied"]["expected_parameters"] == tied.parameter_count()
    assert payload["candidates"]["untied"]["expected_parameters"] == untied.parameter_count()
    assert payload["matching"]["absolute_parameter_delta"] == 288
