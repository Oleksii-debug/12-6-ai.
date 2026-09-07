from __future__ import annotations

import torch

from twelve_six.checkpoint import (
    CheckpointIdentity,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.s3_engineering import (
    S3_CURRENT_EXPECTED_PARAMETERS,
    S3_CURRENT_MODEL_SHA256,
    s3_current_model_spec,
)
from twelve_six.training.config import TrainerConfig
from twelve_six.training.single_gpu import (
    SingleDeviceOOMError,
    SingleDeviceStepRunner,
    build_synthetic_lm_batch,
    greedy_inference_after_training,
    model_storage_dtypes,
    move_batch_to_device,
    optimizer_state_dtypes,
    resolve_single_device,
    seed_before_model_init,
)
from twelve_six.training.trainer import Trainer


def _hex(char: str, length: int = 64) -> str:
    return char * length


def test_scale03_single_gpu_execution_binding_matches_live_product_s3() -> None:
    stage = load_stage_config(
        "configs/stages/alternatives/s3_10m_scale03_byte_gqa.execution.json"
    )
    live = s3_current_model_spec()

    assert stage.stage == "S3"
    assert stage.expected_parameters == S3_CURRENT_EXPECTED_PARAMETERS == 10_000_640
    assert stage.model.parameter_count() == S3_CURRENT_EXPECTED_PARAMETERS
    assert stage.model.identity_sha256() == S3_CURRENT_MODEL_SHA256
    assert stage.model.to_dict() == live.to_dict()


def test_explicit_cuda_request_fails_closed_when_cuda_is_unavailable() -> None:
    if torch.cuda.is_available():
        return
    try:
        resolve_single_device("cuda:0")
    except RuntimeError as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("unavailable CUDA request must fail closed")


def test_auto_device_cpu_fallback_and_transfer_truth() -> None:
    if torch.cuda.is_available():
        return
    device, runtime = resolve_single_device("auto", allow_cpu_fallback=True)
    batch = build_synthetic_lm_batch(
        vocab_size=32,
        batch_size=2,
        sequence_length=8,
        seed=7,
    )
    moved, transfer = move_batch_to_device(batch, device, non_blocking=True)

    assert runtime.resolved == "cpu"
    assert all(value.device.type == "cpu" for value in moved.values())
    assert transfer.requested_non_blocking is True
    assert transfer.effective_non_blocking is False
    assert transfer.all_cpu_sources_pinned is False


def test_s2_1m_cpu_checkpoint_resume_and_inference_seam(tmp_path) -> None:
    stage = load_stage_config("configs/stages/s2_1m.json")
    device = torch.device("cpu")
    config = TrainerConfig(
        learning_rate=3e-4,
        max_steps=2,
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=1515,
        deterministic_algorithms=True,
    )

    seed_before_model_init(config.seed, device)
    model = TwelveSixDecoder(stage.model, stage.init)
    assert model_storage_dtypes(model) == ["torch.float32"]
    trainer = Trainer(model, config, device=device)
    runner = SingleDeviceStepRunner(trainer)
    first = runner.train_microbatch(
        build_synthetic_lm_batch(
            vocab_size=stage.model.vocab_size,
            batch_size=1,
            sequence_length=8,
            seed=100,
        )
    )
    assert first.trainer.optimizer_step == 1
    assert first.trainer.grad_norm is not None
    assert first.tokens_per_second > 0
    assert optimizer_state_dtypes(trainer) == ["torch.float32"]

    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    before_reload = greedy_inference_after_training(
        model,
        prompt,
        device=device,
        max_new_tokens=2,
    )

    training_identity = {
        "stage": stage.stage,
        "init_spec_sha256": stage.init.identity_sha256(),
        "fixture": "train15_synthetic_mechanics_only",
        "trainer": {
            "precision": config.precision,
            "seed": config.seed,
            "max_steps": config.max_steps,
        },
    }
    identity = CheckpointIdentity(
        git_sha=_hex("a", 40),
        model_spec=stage.model.to_dict(),
        parameter_count=stage.model.parameter_count(),
        tokenizer_hash=_hex("b"),
        tokenizer_vocab_hash=_hex("c"),
        dataset_manifest_hash=_hex("d"),
        run_manifest_hash=_hex("e"),
        training_config=training_identity,
        seed=config.seed,
        precision=config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={"name": "AdamW"},
        scheduler=None,
    )
    checkpoint_dir = tmp_path / "step-000001"
    save_trainer_checkpoint(
        checkpoint_dir,
        model=model,
        trainer=trainer,
        identity=identity,
    )

    seed_before_model_init(config.seed + 1, device)
    resumed_model = TwelveSixDecoder(stage.model, stage.init)
    resumed_trainer = Trainer(resumed_model, config, device=device)
    load_trainer_checkpoint(
        checkpoint_dir,
        model=resumed_model,
        trainer=resumed_trainer,
        expected_model_spec_hash=stage.model.identity_sha256(),
        expected_tokenizer_hash=_hex("b"),
        expected_tokenizer_vocab_hash=_hex("c"),
        expected_dataset_manifest_hash=_hex("d"),
        expected_run_manifest_hash=_hex("e"),
        expected_seed=config.seed,
    )

    assert resumed_trainer.optimizer_step == 1
    assert resumed_trainer.tokens_seen == trainer.tokens_seen
    after_reload = greedy_inference_after_training(
        resumed_model,
        prompt,
        device=device,
        max_new_tokens=2,
    )
    torch.testing.assert_close(after_reload, before_reload, rtol=0, atol=0)

    resumed = SingleDeviceStepRunner(resumed_trainer).train_microbatch(
        build_synthetic_lm_batch(
            vocab_size=stage.model.vocab_size,
            batch_size=1,
            sequence_length=8,
            seed=101,
        )
    )
    assert resumed.trainer.optimizer_step == 2


def test_runner_is_poisoned_after_oom() -> None:
    class _Optimizer:
        def __init__(self) -> None:
            self.zeroed = False

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True
            self.zeroed = True

    class _OOMTrainer:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.optimizer = _Optimizer()

        def train_microbatch(self, batch):
            del batch
            raise torch.cuda.OutOfMemoryError("synthetic OOM")

    trainer = _OOMTrainer()
    runner = SingleDeviceStepRunner(trainer)  # type: ignore[arg-type]
    batch = {"input_ids": torch.ones((1, 2), dtype=torch.long)}

    try:
        runner.train_microbatch(batch)
    except SingleDeviceOOMError:
        pass
    else:
        raise AssertionError("OOM must be converted into fail-closed runner state")
    assert trainer.optimizer.zeroed is True

    try:
        runner.train_microbatch(batch)
    except RuntimeError as exc:
        assert "restore a verified checkpoint" in str(exc)
    else:
        raise AssertionError("poisoned runner must not retry in memory")
