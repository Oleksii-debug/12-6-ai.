from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.training.config import TrainerConfig
from twelve_six.training.loss import causal_lm_loss
from twelve_six.training.scale_runtime import (
    ActivationCheckpointedDecoder,
    ExternallyPlacedTrainer,
    build_meta_decoder,
    estimate_scale_resources,
    reset_materialized_decoder_parameters_,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "configs" / "stages" / "s5_400m.scale05_candidate.json"


def _small_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=16,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        d_ff=64,
        rope_rotary_dim=8,
    )


def test_400m_candidate_is_exact_and_meta_constructible() -> None:
    candidate = load_stage_config(CANDIDATE)
    assert candidate.model.parameter_count() == 401_273_856
    assert candidate.expected_parameters == 401_273_856
    assert candidate.model.identity_sha256() == (
        "ef44d5eac5bdf90a39e644076d43decd4e20d5d9eeb11f93af9985776f124310"
    )

    model = build_meta_decoder(candidate.model, candidate.init)
    assert sum(parameter.numel() for parameter in model.parameters()) == 401_273_856
    assert all(parameter.device.type == "meta" for parameter in model.parameters())


def test_400m_resource_budget_accounts_for_optimizer_attention_and_kv_cache() -> None:
    spec = load_stage_config(CANDIDATE).model
    estimate = estimate_scale_resources(
        spec,
        sequence_length=4096,
        microbatch_size=1,
        activation_checkpointing=True,
    )

    assert estimate.parameters == 401_273_856
    assert estimate.embedding_parameters == 33_554_432
    assert 0.083 < estimate.embedding_fraction < 0.084
    assert estimate.persistent_parameter_bytes_per_rank == 1_605_095_424
    assert estimate.persistent_gradient_bytes_per_rank == 1_605_095_424
    assert estimate.persistent_optimizer_bytes_per_rank == 3_210_190_848
    assert estimate.persistent_total_bytes_per_rank == 6_420_381_696
    assert estimate.full_training_checkpoint_bytes == 4_815_286_272
    assert estimate.weight_only_checkpoint_bytes == 1_605_095_424
    assert estimate.kv_cache_bytes_per_token_per_sequence == 30_720
    assert estimate.estimated_activation_bytes_per_microbatch == 617_611_264
    assert estimate.estimated_training_flops_per_token == 3_917_592_576

    sharded = estimate_scale_resources(
        spec,
        sequence_length=4096,
        microbatch_size=1,
        activation_checkpointing=True,
        world_size=4,
        fsdp2_sharded=True,
    )
    assert sharded.persistent_total_bytes_per_rank == 1_605_095_424


def test_activation_checkpointing_preserves_logits_gradients_and_state_dict_keys() -> None:
    spec = _small_spec()
    init = InitSpec()
    torch.manual_seed(11)
    plain = TwelveSixDecoder(spec, init)
    checkpointed = ActivationCheckpointedDecoder(spec, init)
    checkpointed.load_state_dict(plain.state_dict())

    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
        dtype=torch.long,
    )
    plain.train()
    checkpointed.train()

    plain_output = plain(input_ids).logits
    checkpointed_output = checkpointed(input_ids).logits
    torch.testing.assert_close(checkpointed_output, plain_output, rtol=0.0, atol=0.0)

    causal_lm_loss(plain_output, input_ids).backward()
    causal_lm_loss(checkpointed_output, input_ids).backward()

    plain_grads = dict(plain.named_parameters())
    checkpointed_grads = dict(checkpointed.named_parameters())
    assert plain.state_dict().keys() == checkpointed.state_dict().keys()
    assert plain_grads.keys() == checkpointed_grads.keys()
    for name in plain_grads:
        assert plain_grads[name].grad is not None
        assert checkpointed_grads[name].grad is not None
        torch.testing.assert_close(
            checkpointed_grads[name].grad,
            plain_grads[name].grad,
            rtol=1e-6,
            atol=1e-7,
        )


def test_meta_materialization_reinitializes_and_reties_small_decoder() -> None:
    spec = _small_spec()
    model = build_meta_decoder(spec, InitSpec())
    model.to_empty(device="cpu")
    torch.manual_seed(17)
    reset_materialized_decoder_parameters_(model)

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert model.lm_head.weight is model.token_embedding.weight
    assert sum(parameter.numel() for parameter in model.parameters()) == spec.parameter_count()

    logits = model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long)).logits
    assert logits.shape == (1, 4, 256)
    assert torch.isfinite(logits).all()


def test_externally_placed_trainer_does_not_move_model_and_can_update() -> None:
    class NoMoveDecoder(ActivationCheckpointedDecoder):
        def to(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("ExternallyPlacedTrainer must not call model.to()")

    model = NoMoveDecoder(_small_spec(), InitSpec())
    config = TrainerConfig(
        learning_rate=1e-3,
        max_steps=1,
        gradient_accumulation_steps=2,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=23,
    )
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, betas=config.betas)
    trainer = ExternallyPlacedTrainer(
        model,
        config,
        device="cpu",
        optimizer=optimizer,
    )

    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)}
    first = trainer.train_microbatch(batch)
    second = trainer.train_microbatch(batch)

    assert not first.optimizer_stepped
    assert second.optimizer_stepped
    assert trainer.optimizer_step == 1
    trainer.assert_checkpoint_safe()
