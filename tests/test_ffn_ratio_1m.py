from dataclasses import replace

import pytest

from twelve_six.model import load_stage_config
from twelve_six.training.ffn_ratio_1m import (
    FFNRatioExperimentError,
    solve_d_ff_for_attention_width,
)


CONTROL = "configs/stages/alternatives/s2_1m_byte_gqa.candidate.json"


def test_exact_iso_parameter_solver_and_control_identity() -> None:
    stage = load_stage_config(CONTROL)
    expected = {
        24: (320, "2ae3c59828f10a7b55b5a393db2679183b14baef203bd24780ac67533ab8f79f"),
        28: (304, "4cb12aec01a02fb9b987182db84f2c7cd982d07341f43cc309fb3d09d112c1dd"),
        32: (288, "18284b303eb31cef5191ddb3ed4ddba5ce51789aadf4b14cc90d4226c5c527b5"),
        36: (272, "c4ed4d844ae9764670cc7bf6b60449869bb28bf7d5d895991d8a48210ee4ea03"),
        40: (256, "eb03aeb9598f77eee2f78d4e9bd91c5c230c1fae4d9d429bb21f54edc7aeb129"),
    }
    for head_dim, (d_ff, identity) in expected.items():
        spec = solve_d_ff_for_attention_width(
            stage.model,
            head_dim=head_dim,
            target_parameters=stage.expected_parameters,
        )
        assert spec.parameter_count() == 992_896
        assert spec.d_ff == d_ff
        assert spec.identity_sha256() == identity
    assert solve_d_ff_for_attention_width(
        stage.model, head_dim=32, target_parameters=stage.expected_parameters
    ) == stage.model


def test_negative_rope_and_unsolved_geometry_fail_closed() -> None:
    stage = load_stage_config(CONTROL)
    with pytest.raises(FFNRatioExperimentError, match="even"):
        solve_d_ff_for_attention_width(
            stage.model, head_dim=31, target_parameters=stage.expected_parameters
        )
    with pytest.raises(FFNRatioExperimentError, match="exactly compensated"):
        solve_d_ff_for_attention_width(
            stage.model, head_dim=30, target_parameters=stage.expected_parameters
        )
    with pytest.raises(ValueError, match="rotary_dim"):
        replace(stage.model, head_dim=24, rope_rotary_dim=32)


def test_solver_preserves_non_ffn_controls() -> None:
    stage = load_stage_config(CONTROL)
    candidate = solve_d_ff_for_attention_width(
        stage.model, head_dim=24, target_parameters=stage.expected_parameters
    )
    assert candidate.vocab_size == stage.model.vocab_size == 256
    assert candidate.max_seq_len == stage.model.max_seq_len == 512
    assert candidate.d_model == stage.model.d_model == 128
    assert candidate.n_layers == stage.model.n_layers == 6
    assert candidate.n_heads == stage.model.n_heads == 4
    assert candidate.n_kv_heads == stage.model.n_kv_heads == 2
    assert candidate.position_embedding == "rope"
    assert candidate.rope_rotary_dim == candidate.head_dim
