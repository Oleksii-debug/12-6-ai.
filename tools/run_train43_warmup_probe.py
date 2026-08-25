#!/usr/bin/env python3
"""Execute the TRAIN-43 controlled warmup sweep on S1 and S2 geometries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import build_optimizer
from twelve_six.training.warmup_schedule import WarmupScheduleConfig, apply_learning_rate


def _loss(model: TwelveSixDecoder, input_ids: torch.Tensor) -> torch.Tensor:
    logits = model(input_ids).logits
    return F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
        input_ids[:, 1:].contiguous().view(-1),
    )


def _state_hash(model: TwelveSixDecoder) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _make_trace(
    *,
    vocab_size: int,
    steps: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    trace: list[torch.Tensor] = []
    active_vocab = min(64, vocab_size)
    for step in range(steps):
        starts = torch.randint(0, active_vocab, (batch_size, 1), generator=generator)
        rows: list[list[int]] = []
        for row in range(batch_size):
            values = [int(starts[row, 0])]
            phase = (step * 7 + row * 3) % active_vocab
            for token_index in range(1, sequence_length):
                values.append((values[-1] * 5 + 1 + phase + token_index % 3) % active_vocab)
            rows.append(values)
        trace.append(torch.tensor(rows, dtype=torch.long))
    return trace


@torch.no_grad()
def _validation_loss(model: TwelveSixDecoder, batches: list[torch.Tensor]) -> float:
    model.eval()
    values = [float(_loss(model, batch).detach()) for batch in batches]
    model.train()
    return sum(values) / len(values)


def _global_grad_norm(model: TwelveSixDecoder) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach().float().pow(2)))
    return math.sqrt(total)


def _global_weight_norm(model: TwelveSixDecoder) -> float:
    total = sum(float(torch.sum(parameter.detach().float().pow(2))) for parameter in model.parameters())
    return math.sqrt(total)


def _run_candidate(
    *,
    stage_path: Path,
    scale_name: str,
    warmup_steps: int,
    base_lr: float,
    experiment_steps: int,
    schedule_horizon_steps: int,
    batch_size: int,
    sequence_length: int,
    seed: int,
) -> dict[str, object]:
    stage = load_stage_config(stage_path)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = TwelveSixDecoder(stage.model, stage.init)
    initial_state_sha256 = _state_hash(model)

    optimizer_config = TrainerConfig(
        learning_rate=base_lr,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=experiment_steps,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
    )
    optimizer = build_optimizer(model, optimizer_config)
    schedule = WarmupScheduleConfig(
        base_learning_rate=base_lr,
        warmup_steps=warmup_steps,
        experiment_steps=experiment_steps,
        schedule_horizon_steps=schedule_horizon_steps,
    )

    train_trace = _make_trace(
        vocab_size=stage.model.vocab_size,
        steps=experiment_steps,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed + 1000,
    )
    validation_trace = _make_trace(
        vocab_size=stage.model.vocab_size,
        steps=4,
        batch_size=batch_size,
        sequence_length=sequence_length,
        seed=seed + 2000,
    )

    initial_validation_loss = _validation_loss(model, validation_trace)
    frozen_early_losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in train_trace[:20]:
            frozen_early_losses.append(float(_loss(model, batch).detach()))
    model.train()

    records: list[dict[str, object]] = []
    validation_curve = [{"tokens": 0, "loss": initial_validation_loss}]
    tokens_seen = 0
    clip_count = 0
    finite_state_failure = False

    for step_index, batch in enumerate(train_trace):
        learning_rate = apply_learning_rate(optimizer, step_index, schedule)
        optimizer.zero_grad(set_to_none=True)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        loss = _loss(model, batch)
        if not torch.isfinite(loss).item():
            finite_state_failure = True
            break
        loss.backward()
        grad_norm = _global_grad_norm(model)
        if not math.isfinite(grad_norm):
            finite_state_failure = True
            break
        clipped = grad_norm > 1.0
        clip_count += int(clipped)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        weight_norm = _global_weight_norm(model)
        optimizer.step()

        delta_sq = 0.0
        for parameter, previous in zip(model.parameters(), before, strict=True):
            delta_sq += float(torch.sum((parameter.detach().float() - previous.float()).pow(2)))
        update_ratio = math.sqrt(delta_sq) / max(weight_norm, 1e-30)
        tokens_seen += batch_size * (sequence_length - 1)
        records.append(
            {
                "step": step_index + 1,
                "tokens": tokens_seen,
                "loss": float(loss.detach()),
                "learning_rate": learning_rate,
                "grad_norm": grad_norm,
                "clipped": clipped,
                "update_ratio": update_ratio,
            }
        )
        if step_index < 20 or (step_index + 1) % 10 == 0 or step_index + 1 == experiment_steps:
            validation_curve.append(
                {"tokens": tokens_seen, "loss": _validation_loss(model, validation_trace)}
            )

    early = records[:20]
    early_loss_spike = max(
        (float(record["loss"]) - frozen_early_losses[index] for index, record in enumerate(early)),
        default=float("nan"),
    )
    early_grad_norm_max = max((float(record["grad_norm"]) for record in early), default=float("nan"))
    early_update_ratio_max = max(
        (float(record["update_ratio"]) for record in early), default=float("nan")
    )
    update_ratios = sorted(float(record["update_ratio"]) for record in records)
    median_update_ratio = update_ratios[len(update_ratios) // 2] if update_ratios else None
    early_clip_frequency = sum(bool(record["clipped"]) for record in early) / max(len(early), 1)
    recovery_tokens = next(
        (
            int(point["tokens"])
            for point in validation_curve[1:]
            if float(point["loss"]) <= initial_validation_loss
        ),
        None,
    )

    return {
        "scale": scale_name,
        "parameters": stage.expected_parameters,
        "warmup_steps": warmup_steps,
        "base_lr": base_lr,
        "experiment_steps": experiment_steps,
        "schedule_horizon_steps": schedule_horizon_steps,
        "initial_state_sha256": initial_state_sha256,
        "batch_trace_seed": seed + 1000,
        "validation_trace_seed": seed + 2000,
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": float(validation_curve[-1]["loss"]),
        "early_loss_spike_vs_frozen_same_batch": early_loss_spike,
        "early_grad_norm_max": early_grad_norm_max,
        "early_clip_frequency": early_clip_frequency,
        "clip_frequency": clip_count / max(len(records), 1),
        "early_update_ratio_max": early_update_ratio_max,
        "median_update_ratio": median_update_ratio,
        "tokens_to_validation_below_initial": recovery_tokens,
        "finite_state_failure": finite_state_failure,
        "steps_completed": len(records),
        "tokens_seen": tokens_seen,
        "validation_curve": validation_curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--schedule-horizon-steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--warmup-steps", default="0,5,10,20")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    candidates = [int(value) for value in args.warmup_steps.split(",")]
    stages = [
        ("S1_100K", Path("configs/stages/s1_100k.json")),
        ("S2_1M", Path("configs/stages/s2_1m.json")),
    ]
    summaries: list[dict[str, object]] = []
    for scale_name, stage_path in stages:
        for warmup_steps in candidates:
            summaries.append(
                _run_candidate(
                    stage_path=stage_path,
                    scale_name=scale_name,
                    warmup_steps=warmup_steps,
                    base_lr=args.learning_rate,
                    experiment_steps=args.steps,
                    schedule_horizon_steps=args.schedule_horizon_steps,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    seed=args.seed,
                )
            )

    for scale_name, _ in stages:
        hashes = {
            str(row["initial_state_sha256"])
            for row in summaries
            if row["scale"] == scale_name
        }
        if len(hashes) != 1:
            raise RuntimeError(f"initialization identity mismatch within {scale_name}")

    payload = {
        "schema_version": "12-6.train43-warmup-evidence.v1",
        "worker_id": "TRAIN-43-WARMUP",
        "protocol": {
            "optimizer": "AdamW",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "base_learning_rate": args.learning_rate,
            "warmup_steps": candidates,
            "experiment_steps": args.steps,
            "schedule_horizon_steps": args.schedule_horizon_steps,
            "scheduler": "linear_warmup_then_cosine_fixed_horizon",
            "device": "cpu",
            "precision": "fp32",
        },
        "summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
