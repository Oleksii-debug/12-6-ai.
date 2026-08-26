#!/usr/bin/env python3
"""Run only the preregistered TRAIN-344 mechanics stability probes on an exact ~20M ModelSpec."""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch

from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training import Trainer, TrainerConfig

CONTRACT_PATH = Path("configs/experiments/train344_20m_optimizer_transfer_contract.json")


class ProbeError(RuntimeError):
    pass


def _all_parameters_finite(model: torch.nn.Module) -> bool:
    return all(torch.isfinite(p.detach()).all().item() for p in model.parameters())


def _sample_parameter(model: torch.nn.Module) -> torch.Tensor:
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.numel():
            return parameter.detach().reshape(-1)[: min(parameter.numel(), 4096)].float().clone()
    raise ProbeError("model has no trainable parameter sample")


def _sample_update_ratio(before: torch.Tensor, model: torch.nn.Module) -> float:
    after = _sample_parameter(model)
    denom = float(torch.linalg.vector_norm(before).item())
    delta = float(torch.linalg.vector_norm(after - before).item())
    return delta / denom if denom > 0.0 else 0.0


def _trace(vocab_size: int, *, seed: int, steps: int, sequence_length: int) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    result = []
    for _ in range(steps):
        ids = torch.randint(0, vocab_size, (1, sequence_length), generator=generator, dtype=torch.long)
        result.append({"input_ids": ids, "labels": ids.clone()})
    return result


def _run_arm(stage, contract: dict, lr: float, trace: list[dict[str, torch.Tensor]]) -> dict:
    probe = contract["bounded_stability_probe"]
    opt = contract["preregistered_optimizer"]
    seed = int(probe["seed"])
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    config = TrainerConfig(
        learning_rate=lr,
        weight_decay=opt["weight_decay"],
        betas=tuple(opt["betas"]),
        eps=opt["eps"],
        max_steps=probe["optimizer_steps_per_lr"],
        warmup_steps=opt["warmup_steps"],
        scheduler=opt["scheduler"],
        gradient_accumulation_steps=opt["gradient_accumulation_steps"],
        gradient_clip_norm=opt["gradient_clip_norm"],
        precision=opt["precision"],
        seed=seed,
        deterministic_algorithms=opt["deterministic_algorithms"],
        deterministic_warn_only=False,
    )
    trainer = Trainer(model, config, device="cpu")
    rows = []
    started = time.perf_counter()
    for batch in trace:
        before = _sample_parameter(model)
        metrics = trainer.train_microbatch(batch)
        ratio = _sample_update_ratio(before, model)
        finite = _all_parameters_finite(model)
        if not finite:
            raise ProbeError(f"non-finite parameter after optimizer_step={trainer.optimizer_step}")
        row = {
            "optimizer_step": metrics.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
            "loss": metrics.update_loss if metrics.update_loss is not None else metrics.loss,
            "grad_norm": metrics.grad_norm,
            "clipped": metrics.grad_norm is not None and metrics.grad_norm > opt["gradient_clip_norm"],
            "sample_update_to_weight_ratio": ratio,
            "parameters_finite": finite,
        }
        if not math.isfinite(float(row["loss"])) or not math.isfinite(float(row["grad_norm"])):
            raise ProbeError(f"non-finite metric at optimizer_step={trainer.optimizer_step}")
        rows.append(row)
    elapsed = time.perf_counter() - started
    trainer.assert_checkpoint_safe()
    expected_steps = int(probe["optimizer_steps_per_lr"])
    expected_tokens = int(probe["optimized_tokens_per_lr"])
    stable = (
        trainer.optimizer_step == expected_steps
        and trainer.tokens_seen == expected_tokens
        and len(rows) == expected_steps
        and all(row["parameters_finite"] for row in rows)
    )
    result = {
        "learning_rate": lr,
        "status": "STABLE_MECHANICS_ONLY" if stable else "UNSTABLE_MECHANICS_ONLY",
        "optimizer_steps": trainer.optimizer_step,
        "optimized_tokens": trainer.tokens_seen,
        "loss_first": rows[0]["loss"],
        "loss_final": rows[-1]["loss"],
        "grad_norm_max": max(float(row["grad_norm"]) for row in rows),
        "clip_frequency": sum(bool(row["clipped"]) for row in rows) / len(rows),
        "sample_update_to_weight_ratio_max": max(float(row["sample_update_to_weight_ratio"]) for row in rows),
        "wall_seconds": elapsed,
        "rows": rows,
    }
    del trainer, model
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", type=Path, default=Path("evidence/train344/stability-probe.json"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    contract = json.loads((repo / CONTRACT_PATH).read_text(encoding="utf-8"))
    if contract["learned_transfer_budget"]["authorized_unique_nonignored_causal_loss_positions_now"] != 0:
        raise ProbeError("TRAIN-344 contract unexpectedly authorizes learned-corpus exposure")
    stage = load_stage_config(repo / args.model_config)
    count = stage.model.parameter_count()
    low, high = contract["target_20m_gate"]["allowed_parameter_window"]
    if not low <= count <= high:
        raise ProbeError(f"parameter count {count} outside preregistered ~20M window")
    if stage.model.max_seq_len < contract["preregistered_optimizer"]["sequence_length"]:
        raise ProbeError("candidate model context is below preregistered sequence length")
    torch.set_num_threads(2)
    trace = _trace(
        stage.model.vocab_size,
        seed=contract["bounded_stability_probe"]["seed"],
        steps=contract["bounded_stability_probe"]["optimizer_steps_per_lr"],
        sequence_length=contract["preregistered_optimizer"]["sequence_length"],
    )
    arms = []
    for lr in contract["preregistered_optimizer"]["lr_transfer"]["candidates"]:
        arms.append(_run_arm(stage, contract, float(lr), trace))
    if sum(int(arm["optimized_tokens"]) for arm in arms) != contract["bounded_stability_probe"]["total_optimized_tokens_all_lr_arms"]:
        raise ProbeError("total optimized-token probe budget drift")
    report = {
        "schema": "12-6.train344-20m-stability-probe.v1",
        "worker_id": contract["worker_id"],
        "contract_identity_sha256": contract["identity_sha256"],
        "model_config": args.model_config,
        "parameter_count": count,
        "model_spec_sha256": stage.model.identity_sha256(),
        "init_spec_sha256": stage.init.identity_sha256(),
        "data_role": "DETERMINISTIC_SYNTHETIC_MECHANICS_ONLY",
        "selection_authority": "NONE",
        "arms": arms,
        "overall_status": "STABLE_MECHANICS_ONLY" if all(a["status"] == "STABLE_MECHANICS_ONLY" for a in arms) else "UNSTABLE_MECHANICS_ONLY",
        "learned_corpus_updates": 0,
        "paid_compute": False,
        "quality_winner": None,
    }
    output = repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "arms"}, sort_keys=True))
    return 0 if report["overall_status"] == "STABLE_MECHANICS_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
