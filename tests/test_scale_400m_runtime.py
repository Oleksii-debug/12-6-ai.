from pathlib import Path

import pytest
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


def small_spec() -> ModelSpec:
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


def test_candidate_meta_and_resource_budget() -> None:
    candidate = load_stage_config(CANDIDATE)
    spec = candidate.model
    assert spec.parameter_count() == candidate.expected_parameters == 400_421_888
    assert spec.vocab_size == 256
    assert spec.identity_sha256() == (
        "9e6e59bbd7bece16a367fe2b4649079b5a2b6c92b44a99d7db892cc8db3684d2"
    )
    model = build_meta_decoder(spec, candidate.init)
    assert sum(p.numel() for p in model.parameters()) == 400_421_888
    assert all(p.device.type == "meta" for p in model.parameters())

    estimate = estimate_scale_resources(spec, sequence_length=4096)
    assert estimate.embedding_parameters == 262_144
    assert estimate.persistent_total_bytes_per_rank == 6_406_750_208
    assert estimate.full_training_checkpoint_bytes == 4_805_062_656
    assert estimate.weight_only_checkpoint_bytes == 1_601_687_552
    assert estimate.kv_cache_bytes_per_token_per_sequence == 30_720
    assert estimate.estimated_activation_bytes_per_microbatch == 369_623_040
    assert estimate.estimated_training_flops_per_token == 5_216_641_024

    sharded = estimate_scale_resources(
        spec,
        sequence_length=4096,
        world_size=4,
        fsdp2_sharded=True,
    )
    assert sharded.persistent_total_bytes_per_rank == 1_601_687_552


def test_checkpointed_decoder_matches_plain_forward_and_backward() -> None:
    spec = small_spec()
    torch.manual_seed(11)
    plain = TwelveSixDecoder(spec, InitSpec())
    checked = ActivationCheckpointedDecoder(spec, InitSpec())
    checked.load_state_dict(plain.state_dict())
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)

    plain_logits = plain(ids).logits
    checked_logits = checked(ids).logits
    torch.testing.assert_close(checked_logits, plain_logits, rtol=0.0, atol=0.0)
    causal_lm_loss(plain_logits, ids).backward()
    causal_lm_loss(checked_logits, ids).backward()
    for (name_a, a), (name_b, b) in zip(plain.named_parameters(), checked.named_parameters()):
        assert name_a == name_b
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(b.grad, a.grad, rtol=1e-6, atol=1e-7)


def test_meta_materialization_and_external_trainer_update() -> None:
    spec = small_spec()
    model = build_meta_decoder(spec, InitSpec())
    model.to_empty(device="cpu")
    torch.manual_seed(17)
    reset_materialized_decoder_parameters_(model)
    assert model.lm_head.weight is model.token_embedding.weight
    assert torch.isfinite(model(torch.tensor([[1, 2, 3]])).logits).all()

    class NoMoveDecoder(ActivationCheckpointedDecoder):
        def to(self, *args, **kwargs):
            raise AssertionError("ExternallyPlacedTrainer must not call model.to()")

    model2 = NoMoveDecoder(spec, InitSpec())
    config = TrainerConfig(
        learning_rate=1e-3,
        max_steps=1,
        gradient_accumulation_steps=2,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=23,
    )
    trainer = ExternallyPlacedTrainer(
        model2,
        config,
        device="cpu",
        optimizer=AdamW(model2.parameters(), lr=1e-3),
    )
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])}
    assert not trainer.train_microbatch(batch).optimizer_stepped
    assert trainer.train_microbatch(batch).optimizer_stepped
    trainer.assert_checkpoint_safe()


def test_fsdp2_dtensor_gradient_norm(tmp_path: Path) -> None:
    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        pytest.skip("requires ownership of a torch.distributed process group")
    try:
        from torch.distributed.fsdp import fully_shard
    except (ImportError, AttributeError):
        pytest.skip("FSDP2 unavailable")

    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{tmp_path / 'store'}",
        rank=0,
        world_size=1,
    )
    try:
        model = ActivationCheckpointedDecoder(small_spec(), InitSpec())
        for block in model.blocks:
            fully_shard(block)
        fully_shard(model)
        config = TrainerConfig(
            learning_rate=1e-3,
            max_steps=1,
            gradient_accumulation_steps=1,
            gradient_clip_norm=1.0,
            precision="fp32",
            seed=29,
        )
        metrics = ExternallyPlacedTrainer(model, config, device="cpu").train_microbatch(
            {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])}
        )
        assert metrics.optimizer_stepped
        assert metrics.grad_norm is not None and metrics.grad_norm > 0.0
    finally:
        torch.distributed.destroy_process_group()
