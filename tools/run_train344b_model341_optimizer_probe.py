#!/usr/bin/env python3
"""Run the frozen 3x32 TRAIN-344 mechanics probe on exact MODEL-341 only."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.training import Trainer, TrainerConfig

from validate_train344b_model341_optimizer_mechanics import build_report, load_contract


class ProbeError(RuntimeError):
    pass


def tensor_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        value = parameter.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def all_parameters_finite(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter.detach()).all().item()) for parameter in model.parameters())


def sample_parameter(model: torch.nn.Module) -> torch.Tensor:
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.numel():
            return parameter.detach().reshape(-1)[: min(parameter.numel(), 4096)].float().clone()
    raise ProbeError("model exposes no trainable parameter sample")


def sample_update_ratio(before: torch.Tensor, model: torch.nn.Module) -> float:
    after = sample_parameter(model)
    denominator = float(torch.linalg.vector_norm(before).item())
    delta = float(torch.linalg.vector_norm(after - before).item())
    return delta / denominator if denominator > 0.0 else 0.0


def make_trace(vocab_size: int, *, seed: int, steps: int, sequence_length: int) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    trace: list[dict[str, torch.Tensor]] = []
    for _ in range(steps):
        ids = torch.randint(0, vocab_size, (1, sequence_length), generator=generator, dtype=torch.long)
        trace.append({"input_ids": ids, "labels": ids.clone()})
    return trace


def run_arm(stage, contract: dict, learning_rate: float, trace: list[dict[str, torch.Tensor]]) -> dict:
    probe = contract["bounded_probe"]
    opt = contract["optimizer"]
    seed = int(probe["seed"])
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    initial_model_digest = tensor_digest(model)
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=learning_rate,
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
        ),
        device="cpu",
    )
    rows: list[dict] = []
    started = time.perf_counter()
    for batch in trace:
        before = sample_parameter(model)
        metrics = trainer.train_microbatch(batch)
        finite = all_parameters_finite(model)
        loss = metrics.update_loss if metrics.update_loss is not None else metrics.loss
        grad_norm = metrics.grad_norm
        if grad_norm is None:
            raise ProbeError("missing pre-clip gradient norm")
        if not math.isfinite(float(loss)) or not math.isfinite(float(grad_norm)):
            raise ProbeError(f"non-finite metric at optimizer step {trainer.optimizer_step}")
        if not finite:
            raise ProbeError(f"non-finite parameter after optimizer step {trainer.optimizer_step}")
        rows.append(
            {
                "optimizer_step": trainer.optimizer_step,
                "tokens_seen": trainer.tokens_seen,
                "loss": float(loss),
                "global_preclip_grad_norm": float(grad_norm),
                "clip_activated": float(grad_norm) > float(opt["gradient_clip_norm"]),
                "sample_update_to_weight_ratio": sample_update_ratio(before, model),
                "parameters_finite": finite,
            }
        )
    trainer.assert_checkpoint_safe()
    elapsed = time.perf_counter() - started
    expected_steps = int(probe["optimizer_steps_per_lr"])
    expected_targets = int(probe["optimized_causal_targets_per_lr"])
    stable = trainer.optimizer_step == expected_steps and trainer.tokens_seen == expected_targets and len(rows) == expected_steps
    result = {
        "learning_rate": learning_rate,
        "status": "STABLE_MECHANICS_ONLY" if stable else "UNSTABLE_MECHANICS_ONLY",
        "initial_model_digest_sha256": initial_model_digest,
        "final_model_digest_sha256": tensor_digest(model),
        "optimizer_steps": trainer.optimizer_step,
        "optimized_causal_targets": trainer.tokens_seen,
        "loss_first": rows[0]["loss"],
        "loss_final": rows[-1]["loss"],
        "global_preclip_grad_norm_max": max(row["global_preclip_grad_norm"] for row in rows),
        "clip_frequency": sum(bool(row["clip_activated"]) for row in rows) / len(rows),
        "sample_update_to_weight_ratio_max": max(row["sample_update_to_weight_ratio"] for row in rows),
        "wall_seconds": elapsed,
        "rows": rows,
    }
    del trainer, model
    gc.collect()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("evidence/train344b/model341-optimizer-probe.json"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    readiness = build_report(repo)
    contract = load_contract(repo)
    target = contract["target_model"]
    stage = load_stage_config(repo / target["config_path"])

    torch.set_num_threads(2)
    trace = make_trace(
        stage.model.vocab_size,
        seed=contract["bounded_probe"]["seed"],
        steps=contract["bounded_probe"]["optimizer_steps_per_lr"],
        sequence_length=contract["optimizer"]["sequence_length"],
    )
    arms = [run_arm(stage, contract, float(lr), trace) for lr in contract["optimizer"]["learning_rate_candidates"]]
    initial_digests = {arm["initial_model_digest_sha256"] for arm in arms}
    if len(initial_digests) != 1:
        raise ProbeError("LR arms did not start from byte-identical parameter state")
    total_targets = sum(int(arm["optimized_causal_targets"]) for arm in arms)
    if total_targets != contract["bounded_probe"]["total_optimized_causal_targets"]:
        raise ProbeError("total synthetic mechanics target budget drift")
    overall = "STABLE_MECHANICS_ONLY" if all(arm["status"] == "STABLE_MECHANICS_ONLY" for arm in arms) else "UNSTABLE_MECHANICS_ONLY"
    report = {
        "schema": "12-6.train344b-model341-optimizer-probe.v2",
        "worker_id": contract["worker_id"],
        "contract_identity_sha256": contract["identity_sha256"],
        "source_model_authority_sha": target["authority_sha"],
        "model_spec_sha256": target["model_spec_sha256"],
        "init_spec_sha256": target["init_spec_sha256"],
        "parameter_count": target["parameter_count"],
        "data_role": contract["bounded_probe"]["data_role"],
        "same_initial_model_digest_sha256": next(iter(initial_digests)),
        "arms": arms,
        "overall_status": overall,
        "learned_corpus_optimizer_updates": 0,
        "long_training": False,
        "paid_compute": False,
        "selection_authority": "NONE",
        "quality_winner": None,
        "dependency_firewall": readiness["dependency_firewall"],
    }
    output = (repo / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "arms"}, sort_keys=True))
    return 0 if overall == "STABLE_MECHANICS_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
