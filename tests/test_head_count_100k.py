import pytest

from twelve_six.model import load_stage_config
from twelve_six.training.head_count_100k import HeadCountExperimentError, mha_geometry


STAGE = "configs/stages/s1_100k.json"


def test_mha_head_candidates_are_exact_parameter_comparable() -> None:
    stage = load_stage_config(STAGE)
    expected = {
        2: (24, "8336a9d35df4d53e633eeac1cd66e8c77a4874157ae29204bbbf53e141e0a754"),
        3: (16, "4c128d03df1f6d8037e1fb4d4ee211a1b4278e40154ad48167475b27a45a25e2"),
        4: (12, "2f0aa97a5d19e98c4e292fd5f1b454ada45ec4d2c7324e14ab7e48af19908ce6"),
        6: (8, "2c8e14492ef1086f090edd1314814648d6d1e335a474f99bf9ef85dd8aea15de"),
        8: (6, "7be887123e6bd54c67cf559a99d407891933d6cbe1383cf4e42ead6aeb1ec685"),
    }
    for heads, (head_dim, identity) in expected.items():
        candidate = mha_geometry(stage.model, n_heads=heads)
        assert candidate.parameter_count() == 107_856
        assert candidate.head_dim == head_dim
        assert candidate.rope_rotary_dim == head_dim
        assert candidate.n_heads == candidate.n_kv_heads == heads
        assert candidate.q_dim == candidate.kv_dim == candidate.d_model == 48
        assert candidate.identity_sha256() == identity


def test_incumbent_is_exact_control() -> None:
    stage = load_stage_config(STAGE)
    assert mha_geometry(stage.model, n_heads=4) == stage.model


def test_invalid_head_and_rope_geometries_fail_closed() -> None:
    stage = load_stage_config(STAGE)
    with pytest.raises(HeadCountExperimentError, match="divide d_model"):
        mha_geometry(stage.model, n_heads=5)
    with pytest.raises(HeadCountExperimentError, match="even head_dim"):
        mha_geometry(stage.model, n_heads=16)
    with pytest.raises(HeadCountExperimentError, match="positive integer"):
        mha_geometry(stage.model, n_heads=0)
