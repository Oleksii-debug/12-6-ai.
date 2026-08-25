from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from twelve_six import fixed_token_research as ft
from twelve_six.depth_width_500k import LayerTelemetry, candidate_specs, config_payload
from twelve_six.model import InitSpec, TwelveSixDecoder
from twelve_six.training import Trainer


def test_model09_candidate_family_is_predeclared_tied_and_iso_parameter() -> None:
    specs = candidate_specs()
    assert list(specs) == [
        "shallow_wide", "mid_shallow", "balanced", "deep_narrow", "very_deep_narrow"
    ]
    assert [spec.parameter_count() for spec in specs.values()] == [
        496_808, 502_544, 495_456, 497_680, 503_496
    ]
    assert [spec.n_layers for spec in specs.values()] == [2, 3, 4, 6, 8]
    assert [spec.d_model for spec in specs.values()] == [136, 112, 96, 80, 72]
    counts = [spec.parameter_count() for spec in specs.values()]
    assert min(counts) >= 450_000
    assert max(counts) <= 550_000
    assert max(counts) - min(counts) == 8_040
    for spec in specs.values():
        allocation = spec.parameter_breakdown()
        assert spec.tie_word_embeddings is True
        assert spec.d_model == spec.n_heads * spec.head_dim
        assert allocation["token_embedding"] == 256 * spec.d_model
        assert allocation["lm_head_extra"] == 0
        assert allocation["total"] == spec.parameter_count()


def test_model09_committed_config_matches_generated_payload() -> None:
    root = Path(__file__).resolve().parents[1]
    committed = json.loads(
        (root / "configs/experiments/model09_depth_width_500k.v1.json").read_text(encoding="utf-8")
    )
    assert committed == config_payload()


def test_layer_telemetry_records_real_normalized_block_gradients_and_updates() -> None:
    spec = candidate_specs()["shallow_wide"]
    torch.manual_seed(ft.DEFAULT_SEED)
    model = TwelveSixDecoder(spec, InitSpec())
    trainer = Trainer(model, ft._trainer_config(max_steps=1, seed=ft.DEFAULT_SEED), device="cpu")
    raw = torch.arange(4 * 64, dtype=torch.long).reshape(4, 64) % 256
    batch = ft._aligned_batch(raw, 17)
    history: list[dict[str, object]] = []
    telemetry = LayerTelemetry(model, history)
    before = telemetry.begin()
    metrics = trainer.train_microbatch(batch)
    rows = telemetry.finish(
        optimizer_step=trainer.optimizer_step,
        optimized_tokens=trainer.tokens_seen,
        valid_tokens=17,
        before=before,
    )
    telemetry.close()
    assert metrics.optimizer_stepped is True
    assert metrics.tokens == 17
    assert trainer.tokens_seen == 17
    assert len(rows) == spec.n_layers
    assert len(history) == spec.n_layers
    for row in rows:
        assert int(row["optimized_tokens"]) == 17
        for key in (
            "activation_rms", "activation_max_abs", "pre_clip_grad_norm",
            "update_to_weight_ratio",
        ):
            assert math.isfinite(float(row[key]))
            assert float(row[key]) >= 0.0


def test_parent_exact_mask_allows_non_multiple_fixed_token_budget() -> None:
    raw = torch.arange(4 * 64, dtype=torch.long).reshape(4, 64) % 256
    batch = ft._aligned_batch(raw, 13)
    assert int(batch["loss_mask"].sum().item()) == 13
    assert int(torch.count_nonzero(batch["target_ids"] != -100).item()) == 4 * 63
