"""TRAIN-130 LOCAL_FREE parameter-group LR experiment.

Thin experiment surface only: model, tokenizer and Trainer are first-party incumbents.
No canonical optimizer/trainer/config is modified.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.model import InitSpec, ModelSpec, RMSNorm, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

BASE_LR = 3e-4
REDUCED_MULTIPLIER = 0.5
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.0
CLIP_NORM = 1.0
BATCH_SIZE = 4
SEQUENCE_LENGTH = 64
TOKENS_PER_STEP = BATCH_SIZE * (SEQUENCE_LENGTH - 1)
DEFAULT_STEPS = 96
DEFAULT_SEEDS = (1337, 1338)
VARIANTS = ("A", "B", "C")


def controlled_specs() -> dict[str, ModelSpec]:
    def make(d_model: int, n_layers: int, n_heads: int, head_dim: int, d_ff: int) -> ModelSpec:
        return ModelSpec(
            schema_version=1,
            vocab_size=256,
            max_seq_len=256,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_heads,
            head_dim=head_dim,
            d_ff=d_ff,
            activation="swiglu",
            norm_kind="rmsnorm",
            norm_placement="pre",
            norm_eps=1e-5,
            position_embedding="rope",
            rope_theta=10_000.0,
            rope_rotary_dim=head_dim,
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
            final_norm=True,
            tie_word_embeddings=True,
            lm_head_bias=False,
        )

    specs = {
        "500k": make(96, 4, 6, 16, 256),
        "1m": make(128, 5, 8, 16, 352),
    }
    expected = {"500k": 467_808, "1m": 1_037_696}
    actual = {name: spec.parameter_count() for name, spec in specs.items()}
    if actual != expected:
        raise RuntimeError(f"controlled parameter-family drift: {actual!r} != {expected!r}")
    return specs


def trainer_config(*, seed: int, max_steps: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=BASE_LR,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        eps=EPS,
        max_steps=max_steps,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=CLIP_NORM,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or any(not isinstance(r.get("text"), str) or not r["text"] for r in records):
        raise ValueError(f"invalid corpus JSONL: {path}")
    ids = [str(r.get("id")) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate record id in {path}")
    return records


def byte_stream(records: list[dict[str, Any]], tokenizer: ByteTokenizer) -> bytes:
    return b"\n".join(bytes(tokenizer.encode(str(r["text"]))) for r in records) + b"\n"


def make_batch(stream: bytes, step: int) -> torch.Tensor:
    width = BATCH_SIZE * SEQUENCE_LENGTH
    base = (step * width) % len(stream)
    rows = []
    for batch_index in range(BATCH_SIZE):
        start = (base + batch_index * SEQUENCE_LENGTH) % len(stream)
        rows.append([stream[(start + offset) % len(stream)] for offset in range(SEQUENCE_LENGTH)])
    return torch.tensor(rows, dtype=torch.long)


def logical_parameter_groups(model: TwelveSixDecoder) -> tuple[dict[str, list[torch.nn.Parameter]], dict[str, list[str]]]:
    if id(model.token_embedding.weight) != id(model.lm_head.weight):
        raise RuntimeError("TRAIN-130 requires tied embedding/output weights")
    norm_ids = {id(module.weight) for module in model.modules() if isinstance(module, RMSNorm)}
    embedding_id = id(model.token_embedding.weight)
    groups = {"base": [], "tied_embedding_output": [], "norm": []}
    names = {key: [] for key in groups}
    seen: set[int] = set()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        pid = id(parameter)
        if pid in seen:
            raise RuntimeError(f"duplicate trainable parameter object: {name}")
        seen.add(pid)
        if pid == embedding_id:
            group = "tied_embedding_output"
        elif pid in norm_ids:
            group = "norm"
        else:
            group = "base"
        groups[group].append(parameter)
        names[group].append(name)
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if seen != expected:
        raise RuntimeError("parameter grouping is not an exact trainable-parameter partition")
    if len(groups["tied_embedding_output"]) != 1:
        raise RuntimeError("tied embedding/output group must contain exactly one shared tensor")
    return groups, names


def build_experiment_optimizer(model: TwelveSixDecoder, variant: str) -> tuple[torch.optim.AdamW, dict[str, float], dict[str, list[str]]]:
    groups, names = logical_parameter_groups(model)
    if variant == "A":
        ordered = [parameter for _, parameter in sorted(model.named_parameters(), key=lambda item: item[0])]
        optimizer = torch.optim.AdamW(
            ordered, lr=BASE_LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY
        )
        return optimizer, {key: BASE_LR for key in groups}, names

    lrs = {"base": BASE_LR, "tied_embedding_output": BASE_LR, "norm": BASE_LR}
    if variant == "B":
        lrs["tied_embedding_output"] *= REDUCED_MULTIPLIER
    elif variant == "C":
        lrs["norm"] *= REDUCED_MULTIPLIER
    elif variant == "D":
        lrs["tied_embedding_output"] *= REDUCED_MULTIPLIER
        lrs["norm"] *= REDUCED_MULTIPLIER
    else:
        raise ValueError(f"unknown variant: {variant}")
    order = ("base", "tied_embedding_output", "norm")
    optimizer = torch.optim.AdamW(
        [
            {"params": groups[group], "lr": lrs[group], "group_name": group}
            for group in order
        ],
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )
    return optimizer, lrs, names


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    state = optimizer.state_dict()
    digest.update(json.dumps(state["param_groups"], sort_keys=True, default=str).encode("utf-8"))
    for index, values in sorted(state["state"].items()):
        digest.update(str(index).encode("ascii"))
        for key, value in sorted(values.items()):
            digest.update(str(key).encode("utf-8"))
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.numpy().tobytes(order="C"))
            else:
                digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


class GroupTelemetry:
    """Read-only optimizer-step hook; does not alter gradients or parameters."""

    def __init__(self, model: TwelveSixDecoder, names: dict[str, list[str]], sample_every: int = 8):
        self.model = model
        self.names = names
        self.sample_every = sample_every
        self.calls = 0
        self.before: dict[str, dict[str, torch.Tensor]] | None = None
        self.pre: dict[str, dict[str, float]] | None = None
        self.samples: list[dict[str, dict[str, float]]] = []

    def _parameter_map(self) -> dict[str, torch.nn.Parameter]:
        return dict(self.model.named_parameters())

    def pre_hook(self, _optimizer, _args, _kwargs):
        self.calls += 1
        if (self.calls - 1) % self.sample_every:
            self.before = None
            self.pre = None
            return
        pmap = self._parameter_map()
        before: dict[str, dict[str, torch.Tensor]] = {}
        pre: dict[str, dict[str, float]] = {}
        for group, names in self.names.items():
            before[group] = {name: pmap[name].detach().clone() for name in names}
            weight_sq = sum(float((pmap[name].detach().float() ** 2).sum()) for name in names)
            grad_sq = sum(
                float((pmap[name].grad.detach().float() ** 2).sum())
                for name in names
                if pmap[name].grad is not None
            )
            pre[group] = {"weight_l2": math.sqrt(weight_sq), "grad_l2_post_clip": math.sqrt(grad_sq)}
        self.before = before
        self.pre = pre

    def collect_after_step(self) -> None:
        if self.before is None or self.pre is None:
            return
        pmap = self._parameter_map()
        sample: dict[str, dict[str, float]] = {}
        for group, names in self.names.items():
            update_sq = 0.0
            for name in names:
                delta = pmap[name].detach().float() - self.before[group][name].float()
                update_sq += float((delta * delta).sum())
            update_l2 = math.sqrt(update_sq)
            weight_l2 = self.pre[group]["weight_l2"]
            sample[group] = {
                **self.pre[group],
                "update_l2": update_l2,
                "update_weight_ratio": update_l2 / weight_l2 if weight_l2 else 0.0,
            }
        self.samples.append(sample)
        self.before = None
        self.pre = None

    def summary(self) -> dict[str, dict[str, float]]:
        if not self.samples:
            raise RuntimeError("no telemetry samples captured")
        result = {}
        for group in self.names:
            values = [sample[group] for sample in self.samples]
            result[group] = {
                "samples": len(values),
                "mean_weight_l2": sum(v["weight_l2"] for v in values) / len(values),
                "mean_grad_l2_post_clip": sum(v["grad_l2_post_clip"] for v in values) / len(values),
                "mean_update_weight_ratio": sum(v["update_weight_ratio"] for v in values) / len(values),
                "max_update_weight_ratio": max(v["update_weight_ratio"] for v in values),
            }
        return result


@torch.no_grad()
def evaluate(model: TwelveSixDecoder, tokenizer: ByteTokenizer, records: list[dict[str, Any]]) -> dict[str, Any]:
    before = model_state_sha256(model)
    was_training = model.training
    model.eval()
    total = lexical = 0.0
    count = lexical_count = 0
    for record in records:
        ids = tokenizer.encode(str(record["text"]))
        start = 0
        while start < len(ids) - 1:
            chunk = ids[start : start + model.spec.max_seq_len]
            if len(chunk) < 2:
                break
            input_ids = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
            logits = model(input_ids).logits[:, :-1, :]
            targets = input_ids[:, 1:]
            losses = F.cross_entropy(
                logits.reshape(-1, model.spec.vocab_size), targets.reshape(-1), reduction="none"
            )
            total += float(losses.sum())
            count += losses.numel()
            flat_targets = targets.reshape(-1)
            mask = (
                ((flat_targets >= 65) & (flat_targets <= 90))
                | ((flat_targets >= 97) & (flat_targets <= 122))
                | (flat_targets >= 128)
            )
            lexical += float(losses[mask].sum())
            lexical_count += int(mask.sum())
            start += model.spec.max_seq_len - 1
    model.train(was_training)
    after = model_state_sha256(model)
    if before != after:
        raise RuntimeError("evaluation mutated model state")
    loss = total / count
    lexical_loss = lexical / lexical_count
    return {
        "loss": loss,
        "bpb": loss / math.log(2.0),
        "lexical_bpb": lexical_loss / math.log(2.0),
        "tokens": count,
        "lexical_tokens": lexical_count,
        "model_state_non_mutation": True,
    }


def fresh_components(spec: ModelSpec, seed: int, max_steps: int, variant: str):
    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, InitSpec())
    optimizer, lr_map, names = build_experiment_optimizer(model, variant)
    trainer = Trainer(model, trainer_config(seed=seed, max_steps=max_steps), optimizer=optimizer, device="cpu")
    return model, trainer, lr_map, names


def resume_exactness_probe(spec: ModelSpec, seed: int, variant: str, stream: bytes) -> dict[str, Any]:
    split, replay = 24, 4
    model, trainer, _, names = fresh_components(spec, seed, split + replay + 4, variant)
    observer = GroupTelemetry(model, names, sample_every=10_000)
    handle = trainer.optimizer.register_step_pre_hook(observer.pre_hook)
    for step in range(split):
        trainer.train_microbatch({"input_ids": make_batch(stream, step)})
        observer.collect_after_step()
    model_state = copy.deepcopy(model.state_dict())
    trainer_state = copy.deepcopy(trainer.state_dict())
    losses_a = []
    for step in range(split, split + replay):
        metrics = trainer.train_microbatch({"input_ids": make_batch(stream, step)})
        losses_a.append(metrics.update_loss)
    model_hash_a = model_state_sha256(model)
    optimizer_hash_a = optimizer_state_sha256(trainer.optimizer)
    handle.remove()

    restored_model, restored_trainer, _, restored_names = fresh_components(spec, seed, split + replay + 4, variant)
    restored_model.load_state_dict(model_state)
    restored_trainer.load_state_dict(trainer_state)
    restored_observer = GroupTelemetry(restored_model, restored_names, sample_every=10_000)
    restored_handle = restored_trainer.optimizer.register_step_pre_hook(restored_observer.pre_hook)
    losses_b = []
    for step in range(split, split + replay):
        metrics = restored_trainer.train_microbatch({"input_ids": make_batch(stream, step)})
        losses_b.append(metrics.update_loss)
    restored_handle.remove()
    passed = (
        losses_a == losses_b
        and model_hash_a == model_state_sha256(restored_model)
        and optimizer_hash_a == optimizer_state_sha256(restored_trainer.optimizer)
    )
    return {
        "split_step": split,
        "replay_steps": replay,
        "losses_exact": losses_a == losses_b,
        "model_state_exact": model_hash_a == model_state_sha256(restored_model),
        "optimizer_state_exact": optimizer_hash_a == optimizer_state_sha256(restored_trainer.optimizer),
        "passed": passed,
    }


def run_one(spec: ModelSpec, scale: str, variant: str, seed: int, steps: int, stream: bytes, tokenizer: ByteTokenizer, validation: list[dict[str, Any]]) -> dict[str, Any]:
    model, trainer, lr_map, names = fresh_components(spec, seed, steps, variant)
    initial = evaluate(model, tokenizer, validation)
    observer = GroupTelemetry(model, names, sample_every=8)
    handle = trainer.optimizer.register_step_pre_hook(observer.pre_hook)
    losses = []
    grad_norms = []
    eval_curve = [{"step": 0, "optimized_tokens": 0, **initial}]
    started = time.perf_counter()
    for step in range(steps):
        metrics = trainer.train_microbatch({"input_ids": make_batch(stream, step)})
        observer.collect_after_step()
        losses.append(float(metrics.update_loss))
        grad_norms.append(float(metrics.grad_norm))
        if (step + 1) % 32 == 0 or step + 1 == steps:
            eval_curve.append(
                {"step": step + 1, "optimized_tokens": trainer.tokens_seen, **evaluate(model, tokenizer, validation)}
            )
    wall = time.perf_counter() - started
    handle.remove()
    return {
        "scale": scale,
        "variant": variant,
        "seed": seed,
        "parameters": spec.parameter_count(),
        "spec": spec.to_dict(),
        "group_learning_rates": lr_map,
        "group_parameter_names": names,
        "initial_eval": initial,
        "final_eval": eval_curve[-1],
        "eval_curve": eval_curve,
        "train": {
            "first32_mean_loss": sum(losses[:32]) / min(32, len(losses)),
            "last32_mean_loss": sum(losses[-32:]) / min(32, len(losses)),
            "max_preclip_grad_norm": max(grad_norms),
            "clip_fraction": sum(value > CLIP_NORM for value in grad_norms) / len(grad_norms),
        },
        "group_metrics": observer.summary(),
        "wall_seconds": wall,
        "tokens_per_second": trainer.tokens_seen / wall,
    }


def mean(values):
    return sum(values) / len(values)


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for scale in ("500k", "1m"):
        summary[scale] = {}
        for variant in VARIANTS:
            subset = [r for r in runs if r["scale"] == scale and r["variant"] == variant]
            summary[scale][variant] = {
                "final_bpb_mean": mean([r["final_eval"]["bpb"] for r in subset]),
                "final_bpb_by_seed": [r["final_eval"]["bpb"] for r in subset],
                "final_lexical_bpb_mean": mean([r["final_eval"]["lexical_bpb"] for r in subset]),
            }
        control = summary[scale]["A"]
        for variant in ("B", "C"):
            candidate = summary[scale][variant]
            candidate["relative_bpb_improvement_vs_A"] = (
                control["final_bpb_mean"] - candidate["final_bpb_mean"]
            ) / control["final_bpb_mean"]
            candidate["relative_lexical_bpb_improvement_vs_A"] = (
                control["final_lexical_bpb_mean"] - candidate["final_lexical_bpb_mean"]
            ) / control["final_lexical_bpb_mean"]
    return summary


def git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/experiments/train130/bounded-real-train.jsonl"))
    parser.add_argument("--validation-jsonl", type=Path, default=Path("data/experiments/train130/bounded-real-validation.jsonl"))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seeds", default="1337,1338")
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--output", type=Path, default=Path("reports/train130/param_group_lr_reproduction.json"))
    args = parser.parse_args()

    root = args.repo_root.resolve()
    source_sha = git_head(root)
    if args.expected_source_sha and source_sha != args.expected_source_sha:
        raise RuntimeError(f"exact-checkout mismatch: {source_sha} != {args.expected_source_sha}")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    tokenizer = ByteTokenizer()
    train = read_jsonl(root / args.train_jsonl)
    validation = read_jsonl(root / args.validation_jsonl)
    if {r["id"] for r in train} & {r["id"] for r in validation}:
        raise RuntimeError("train/validation id overlap")
    if any(bool(r.get("synthetic", True)) for r in train + validation):
        raise RuntimeError("TRAIN-130 bounded-real experiment rejects synthetic records")
    stream = byte_stream(train, tokenizer)
    specs = controlled_specs()
    seeds = tuple(int(value) for value in args.seeds.split(","))

    runs = []
    resume = []
    for scale, spec in specs.items():
        for variant in VARIANTS:
            for seed in seeds:
                runs.append(run_one(spec, scale, variant, seed, args.steps, stream, tokenizer, validation))
            probe = resume_exactness_probe(spec, seeds[0], variant, stream)
            probe.update({"scale": scale, "variant": variant})
            if not probe["passed"]:
                raise RuntimeError(f"resume exactness failed: {scale}/{variant}")
            resume.append(probe)

    summary = summarize(runs)
    d_eligible = all(
        summary[scale][variant]["relative_bpb_improvement_vs_A"] > 0
        for scale in specs
        for variant in ("B", "C")
    )
    d_runs: list[dict[str, Any]] = []
    if d_eligible:
        for scale, spec in specs.items():
            for seed in seeds:
                d_runs.append(run_one(spec, scale, "D", seed, args.steps, stream, tokenizer, validation))

    report = {
        "schema": "12-6.train130-param-group-lr.reproduction.v1",
        "authority": "LOCAL_FREE_EXPERIMENTAL_OPTIMIZER_EVIDENCE_NOT_CANONICAL_PROMOTION",
        "source_sha": source_sha,
        "protocol": {
            "base_lr": BASE_LR,
            "reduced_multiplier": REDUCED_MULTIPLIER,
            "betas": list(BETAS),
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "schedule": "constant",
            "gradient_clip_norm": CLIP_NORM,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "steps": args.steps,
            "seeds": list(seeds),
        },
        "models": {name: {"parameters": spec.parameter_count(), "spec": spec.to_dict()} for name, spec in specs.items()},
        "runs": runs,
        "resume_exactness": resume,
        "summary": summary,
        "D_status": "RUN" if d_eligible else "NOT_RUN_PREREQUISITE_FAILED",
        "D_runs": d_runs,
        "canonical_optimizer_change": "NONE",
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "source_sha": source_sha, "D_status": report["D_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
