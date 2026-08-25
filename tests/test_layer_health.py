from __future__ import annotations

import copy

import pytest
import torch

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.training.config import TrainerConfig
from twelve_six.training.layer_health import (
    capture_layer_health_window,
    controlled_layer_health_specs,
    detect_depth_health,
    hook_count,
)
from twelve_six.training.trainer import Trainer


def _tiny_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=32,
        max_seq_len=32,
        d_model=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        head_dim=8,
        d_ff=32,
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10_000.0,
        rope_rotary_dim=8,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )


def _batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
        dtype=torch.long,
    )
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def _model_snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _assert_snapshot_equal(
    model: TwelveSixDecoder,
    expected: dict[str, torch.Tensor],
) -> None:
    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        assert torch.equal(actual[name], tensor), name


def _assert_nested_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def test_controlled_layer_health_scales_match_research41_subset() -> None:
    specs = controlled_layer_health_specs()
    assert [spec.parameter_count() for spec in specs] == [95_568, 467_808, 1_037_696]
    assert [spec.n_layers for spec in specs] == [3, 4, 5]
    assert {spec.vocab_size for spec in specs} == {256}
    assert {spec.max_seq_len for spec in specs} == {256}


def test_diagnostic_window_collects_required_layer_metrics_and_removes_hooks() -> None:
    torch.manual_seed(17)
    model = TwelveSixDecoder(_tiny_spec(), InitSpec())
    before = _model_snapshot(model)
    rng_before = torch.random.get_rng_state().clone()
    hooks_before = hook_count(model)

    report = capture_layer_health_window(
        model,
        _batch(),
        label="initialization",
        optimizer_step=0,
        tokens_seen=0,
        gradient_clip_norm=1.0,
    )

    assert hook_count(model) == hooks_before
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    _assert_snapshot_equal(model, before)
    assert report["clipping"]["clipping_applied_by_diagnostic"] is False
    assert report["init_spec"]["residual_branch_scale"] == "sqrt_2_layers"
    assert len(report["layers"]) == 2
    for layer in report["layers"]:
        for name in (
            "residual_in",
            "attention_norm_out",
            "attention_out",
            "residual_after_attention",
            "mlp_norm_out",
            "mlp_out",
            "residual_out",
        ):
            assert layer[name]["finite"] is True
            assert layer[name]["rms"] >= 0.0
            assert layer[name]["variance"] >= 0.0
        assert set(layer["gradient_norms"]) == {
            "attention",
            "mlp",
            "norm",
            "attention_norm",
            "mlp_norm",
        }


def test_diagnostic_window_restores_preexisting_gradients() -> None:
    torch.manual_seed(19)
    model = TwelveSixDecoder(_tiny_spec(), InitSpec())
    first = next(model.parameters())
    first.grad = torch.full_like(first, 0.125)
    expected = first.grad.detach().clone()

    capture_layer_health_window(
        model,
        _batch(),
        label="with_existing_grad",
        optimizer_step=0,
        tokens_seen=0,
        gradient_clip_norm=1.0,
    )

    assert first.grad is not None
    assert torch.equal(first.grad, expected)
    for parameter in list(model.parameters())[1:]:
        assert parameter.grad is None


def test_hooks_are_removed_even_when_diagnostic_forward_fails() -> None:
    torch.manual_seed(23)
    model = TwelveSixDecoder(_tiny_spec(), InitSpec())
    hooks_before = hook_count(model)
    bad = _batch()
    bad["input_ids"] = torch.full_like(bad["input_ids"], 999)
    bad["labels"] = bad["input_ids"].clone()

    with pytest.raises((IndexError, RuntimeError)):
        capture_layer_health_window(
            model,
            bad,
            label="expected_failure",
            optimizer_step=0,
            tokens_seen=0,
            gradient_clip_norm=1.0,
        )

    assert hook_count(model) == hooks_before


def test_diagnostic_windows_do_not_change_deterministic_training_trajectory() -> None:
    config = TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=3,
        scheduler="constant",
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=31,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )

    torch.manual_seed(31)
    control_model = TwelveSixDecoder(_tiny_spec(), InitSpec())
    control = Trainer(control_model, config, device="cpu")
    for _ in range(3):
        control.train_microbatch(_batch())

    torch.manual_seed(31)
    probed_model = TwelveSixDecoder(_tiny_spec(), InitSpec())
    probed = Trainer(probed_model, config, device="cpu")
    capture_layer_health_window(
        probed_model,
        _batch(),
        label="initialization",
        optimizer_step=0,
        tokens_seen=0,
        gradient_clip_norm=1.0,
    )
    for step in range(1, 4):
        probed.train_microbatch(_batch())
        capture_layer_health_window(
            probed_model,
            _batch(),
            label=f"step_{step}",
            optimizer_step=probed.optimizer_step,
            tokens_seen=probed.tokens_seen,
            gradient_clip_norm=1.0,
        )

    _assert_snapshot_equal(probed_model, _model_snapshot(control_model))
    assert probed.micro_step == control.micro_step
    assert probed.optimizer_step == control.optimizer_step
    assert probed.tokens_seen == control.tokens_seen
    _assert_nested_equal(copy.deepcopy(probed.optimizer.state_dict()), control.optimizer.state_dict())


def _synthetic_layers(residual: list[float], gradients: list[float]) -> list[dict[str, object]]:
    return [
        {
            "residual_out": {"rms": residual[index]},
            "gradient_norms": {
                "attention": gradients[index],
                "mlp": gradients[index],
                "norm": gradients[index],
            },
        }
        for index in range(len(residual))
    ]


def test_depth_detector_flags_explosion_and_vanishing_without_theory_claim() -> None:
    exploding = detect_depth_health(_synthetic_layers([1.0, 1.3, 3.0], [1.0, 3.0, 12.0]))
    assert exploding["status"] == "DEPTH_TREND_WARNING"
    assert "RESIDUAL_DEPTH_EXPLOSION_ENDPOINT" in exploding["signals"]
    assert exploding["heuristics"]["theoretical_thresholds_claimed"] is False

    vanishing = detect_depth_health(_synthetic_layers([1.0, 0.7, 0.3], [1.0, 0.5, 0.05]))
    assert vanishing["status"] == "DEPTH_TREND_WARNING"
    assert "RESIDUAL_DEPTH_VANISHING_ENDPOINT" in vanishing["signals"]
