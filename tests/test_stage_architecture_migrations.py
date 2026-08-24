from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from twelve_six import load_stage_config

ROOT = Path(__file__).resolve().parents[1]
STAGES = ROOT / "configs" / "stages"
ALTERNATIVES = STAGES / "alternatives"

S0_MODEL_ID = "86c75b31dff05b7b5db9f6ed068c571a6ead01ba663412fe630f5e52b09d9b6b"
S0_INIT_ID = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"


def _s0_snapshot() -> tuple[dict[str, object], dict[str, object], int, str, str]:
    stage = load_stage_config(STAGES / "s0_10k.json")
    return (
        deepcopy(stage.model.to_dict()),
        deepcopy(stage.init.to_dict()),
        stage.model.parameter_count(),
        stage.model.identity_sha256(),
        stage.init.identity_sha256(),
    )


def test_s0_semantic_authority_is_pinned() -> None:
    model, init, count, model_id, init_id = _s0_snapshot()
    assert count == 10_140
    assert model_id == S0_MODEL_ID
    assert init_id == S0_INIT_ID
    assert model == {
        "schema_version": 1,
        "vocab_size": 256,
        "max_seq_len": 128,
        "d_model": 20,
        "n_layers": 1,
        "n_heads": 2,
        "n_kv_heads": 2,
        "head_dim": 10,
        "d_ff": 56,
        "activation": "swiglu",
        "norm_kind": "rmsnorm",
        "norm_placement": "pre",
        "norm_eps": 1e-5,
        "position_embedding": "rope",
        "rope_theta": 10_000.0,
        "rope_rotary_dim": 10,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "final_norm": True,
        "tie_word_embeddings": True,
        "lm_head_bias": False,
    }
    assert init == {
        "schema_version": 1,
        "family": "normal",
        "std": 0.02,
        "residual_branch_scale": "sqrt_2_layers",
    }


@pytest.mark.parametrize(
    ("filename", "expected", "identity", "attention_variant"),
    [
        (
            "s1_100k_gqa_untied.candidate.json",
            101_328,
            "27f05952739e2c8e95155a187e193138c174302d164fce8e68da5c3c8501848c",
            "gqa",
        ),
        (
            "s2_1m_mqa_bias.candidate.json",
            995_552,
            "c029915fb7b0c120c16ca1f30a3b335cdfdb8b252f82aa2fb34efda667fd6f3e",
            "mqa",
        ),
        (
            "s3_10m_explicit_q_gqa.candidate.json",
            9_999_680,
            "ebf3a73851c273211ff9f5f242d28afe22b109e22aacb998e5c0e86d5ff09a55",
            "gqa",
        ),
        (
            "s4_100m_gqa_longer_context.candidate.json",
            99_797_760,
            "d6ce8b0f44d5601c56fa0b39bfe77cc8863203d3c6ee32701cf897b5a80ab979",
            "gqa",
        ),
    ],
)
def test_non_frozen_s1_s4_alternatives_have_exact_identity(
    filename: str,
    expected: int,
    identity: str,
    attention_variant: str,
) -> None:
    path = ALTERNATIVES / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    stage = load_stage_config(path)

    assert stage.expected_parameters == expected
    assert stage.model.parameter_count() == expected
    assert stage.model.identity_sha256() == identity
    assert payload["status"] == "engineering_candidate_not_frozen"
    assert payload["promotion_allowed"] is False
    assert payload["requires_preceding_stage_pass"] is True

    if stage.model.n_kv_heads == stage.model.n_heads:
        actual_variant = "mha"
    elif stage.model.n_kv_heads == 1:
        actual_variant = "mqa"
    else:
        actual_variant = "gqa"
    assert actual_variant == attention_variant


def test_loading_future_stage_configs_does_not_mutate_s0() -> None:
    before = _s0_snapshot()

    later_paths = [
        STAGES / "s1_100k.json",
        STAGES / "s2_1m.json",
        STAGES / "s3_10m.json",
        STAGES / "s4_100m.candidate.json",
        STAGES / "s5_400m.candidate.json",
        STAGES / "s6_1b.candidate.json",
        STAGES / "s7_3b.candidate.json",
        *sorted(ALTERNATIVES.glob("*.candidate.json")),
    ]
    for path in later_paths:
        load_stage_config(path)

    assert _s0_snapshot() == before
