import hashlib
import json
from dataclasses import replace

import pytest

from twelve_six.model import load_stage_config
from twelve_six.scaling import DenseScalingTemplate, solve_dense_scaling_candidates
from twelve_six.training.ffn_ratio_1m import (
    EXPERIMENT_PATH,
    FFNRatioExperimentError,
    solve_d_ff_for_attention_width,
)


CONTROL = "configs/stages/alternatives/s2_1m_byte_gqa.candidate.json"


def _experiment() -> dict[str, object]:
    with open(EXPERIMENT_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def test_exact_experiment_config_identity() -> None:
    payload = _experiment()
    observed = payload.pop("config_identity_sha256")
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == observed


def test_exact_iso_parameter_solver_and_control_identity() -> None:
    stage = load_stage_config(CONTROL)
    expected = {
        16: (352, "e35166a7b54f1fbd48665827d76f786be470671f296480fe77c2bfe1b3b85bbf"),
        24: (320, "2ae3c59828f10a7b55b5a393db2679183b14baef203bd24780ac67533ab8f79f"),
        32: (288, "18284b303eb31cef5191ddb3ed4ddba5ce51789aadf4b14cc90d4226c5c527b5"),
        40: (256, "eb03aeb9598f77eee2f78d4e9bd91c5c230c1fae4d9d429bb21f54edc7aeb129"),
        48: (224, "4b81bbc2665fe395354210564fa1028e904fe6221bb1a7aedc44f5a68fa20b0a"),
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


def test_live_d11_solver_reproduces_every_configured_candidate() -> None:
    stage = load_stage_config(CONTROL)
    for entry in _experiment()["candidates"]:
        template = DenseScalingTemplate(
            vocab_size=stage.model.vocab_size,
            max_seq_len=stage.model.max_seq_len,
            d_model=stage.model.d_model,
            n_layers=stage.model.n_layers,
            n_heads=stage.model.n_heads,
            n_kv_heads=stage.model.n_kv_heads,
            head_dim=int(entry["head_dim"]),
            d_ff_multiple=32,
            rope_theta=stage.model.rope_theta,
            rope_rotary_dim=int(entry["head_dim"]),
            attention_bias=stage.model.attention_bias,
            mlp_bias=stage.model.mlp_bias,
            final_norm=stage.model.final_norm,
            tie_word_embeddings=stage.model.tie_word_embeddings,
            lm_head_bias=stage.model.lm_head_bias,
        )
        solved = solve_dense_scaling_candidates(
            stage.expected_parameters,
            (template,),
            max_results=4,
            max_relative_error=0.001,
        )
        exact = [candidate for candidate in solved if candidate.exact_parameters == 992_896]
        assert len(exact) == 1
        assert exact[0].spec.d_ff == int(entry["d_ff"])
        assert exact[0].model_identity_sha256 == entry["model_identity_sha256"]


def test_negative_geometry_fails_closed() -> None:
    stage = load_stage_config(CONTROL)
    with pytest.raises(FFNRatioExperimentError, match="even"):
        solve_d_ff_for_attention_width(
            stage.model, head_dim=31, target_parameters=stage.expected_parameters
        )
    with pytest.raises(FFNRatioExperimentError, match="exactly compensated"):
        solve_d_ff_for_attention_width(
            stage.model, head_dim=30, target_parameters=stage.expected_parameters + 1
        )
    with pytest.raises(ValueError, match="rotary_dim"):
        replace(stage.model, head_dim=24, rope_rotary_dim=32)
    with pytest.raises(ValueError, match="divisible"):
        replace(stage.model, n_kv_heads=3)


def test_solver_preserves_non_allocation_controls() -> None:
    stage = load_stage_config(CONTROL)
    baseline = stage.model.to_dict()
    allocation_fields = {"head_dim", "rope_rotary_dim", "d_ff"}
    for entry in _experiment()["candidates"]:
        candidate = solve_d_ff_for_attention_width(
            stage.model,
            head_dim=int(entry["head_dim"]),
            target_parameters=stage.expected_parameters,
        ).to_dict()
        assert {
            key: value for key, value in candidate.items() if key not in allocation_fields
        } == {
            key: value for key, value in baseline.items() if key not in allocation_fields
        }


def test_truth_boundary_excludes_s0_selection_and_freeze() -> None:
    truth = _experiment()["truth_boundary"]
    assert truth["s0_loss_selection_allowed"] is False
    assert truth["architecture_frozen"] is False
    assert truth["paid_compute"] is False
