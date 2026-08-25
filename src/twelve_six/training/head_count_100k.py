"""Exact-parameter MHA query-head granularity experiment at ~100K."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from twelve_six.model import ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training.config import TrainerConfig
from twelve_six.training.loss import causal_lm_loss
from twelve_six.training.trainer import Trainer

SCHEMA = "12-6.model13-head-count-100k.v1"
STAGE_PATH = "configs/stages/s1_100k.json"
PLAN_PATH = "configs/experiments/model13_head_count_100k.v1.json"


class HeadCountExperimentError(ValueError):
    pass


def mha_geometry(control: ModelSpec, *, n_heads: int) -> ModelSpec:
    if not isinstance(n_heads, int) or isinstance(n_heads, bool) or n_heads <= 0:
        raise HeadCountExperimentError("n_heads must be a positive integer")
    if control.d_model % n_heads:
        raise HeadCountExperimentError("n_heads must divide d_model exactly")
    head_dim = control.d_model // n_heads
    if head_dim % 2:
        raise HeadCountExperimentError("RoPE requires even head_dim")
    candidate = replace(
        control,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        head_dim=head_dim,
        rope_rotary_dim=head_dim,
    )
    if candidate.q_dim != control.d_model or candidate.kv_dim != control.d_model:
        raise HeadCountExperimentError("MHA candidate must preserve q/kv projection width")
    return candidate


def _load_texts(path: Path) -> list[str]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        text = item.get("text") if isinstance(item, dict) else None
        if not isinstance(text, str) or not text:
            raise HeadCountExperimentError(f"invalid text row in {path}")
        rows.append(text)
    if not rows:
        raise HeadCountExperimentError(f"no text rows in {path}")
    return rows


def _trace(texts: Sequence[str], tokenizer: ByteTokenizer, *, count: int, length: int):
    result = []
    for index in range(count):
        ids = tokenizer.encode(texts[index % len(texts)])[:length]
        if len(ids) < 2:
            raise HeadCountExperimentError("trace row too short")
        tensor = torch.tensor([ids], dtype=torch.long)
        result.append({"input_ids": tensor, "labels": tensor.clone()})
    return result


def _hash_trace(trace: Sequence[Mapping[str, torch.Tensor]]) -> str:
    raw = json.dumps(
        [item["input_ids"].tolist() for item in trace],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _state_hash(model: TwelveSixDecoder) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _eval(model: TwelveSixDecoder, trace: Sequence[Mapping[str, torch.Tensor]]) -> float:
    model.eval()
    weighted = 0.0
    tokens = 0
    for batch in trace:
        logits = model(batch["input_ids"]).logits
        labels = batch["labels"]
        n = int(labels[:, 1:].ne(-100).sum().item())
        loss = causal_lm_loss(logits, labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError("non-finite validation loss")
        weighted += float(loss.item()) * n
        tokens += n
    return weighted / tokens


def _run_one(
    spec: ModelSpec,
    *,
    candidate_id: str,
    init_spec: Any,
    train_trace: Sequence[Mapping[str, torch.Tensor]],
    validation_trace: Sequence[Mapping[str, torch.Tensor]],
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    initial_state_sha = _state_hash(model)
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=1e-2,
            weight_decay=0.0,
            max_steps=len(train_trace),
            scheduler="constant",
            gradient_accumulation_steps=1,
            gradient_clip_norm=1.0,
            precision="fp32",
            seed=seed,
            deterministic_algorithms=True,
        ),
        device="cpu",
    )
    train_curve = []
    val_curve = [{"optimizer_step": 0, "loss": _eval(model, validation_trace)}]
    latencies = []
    for step, batch in enumerate(train_trace, 1):
        start = time.perf_counter()
        metric = trainer.train_microbatch(batch)
        latencies.append(time.perf_counter() - start)
        if metric.grad_norm is None:
            raise RuntimeError("optimizer-step gradient norm missing")
        train_curve.append(
            {
                "optimizer_step": metric.optimizer_step,
                "loss": metric.loss,
                "grad_norm": metric.grad_norm,
                "tokens": metric.tokens,
            }
        )
        if step % 2 == 0 or step == len(train_trace):
            val_curve.append({"optimizer_step": step, "loss": _eval(model, validation_trace)})
    breakdown = spec.parameter_breakdown()
    attention_total = spec.n_layers * breakdown["attention_per_layer"]
    finite = all(
        math.isfinite(point["loss"]) and math.isfinite(point["grad_norm"])
        for point in train_curve
    ) and all(math.isfinite(point["loss"]) for point in val_curve)
    return {
        "candidate_id": candidate_id,
        "model": spec.to_dict(),
        "model_identity_sha256": spec.identity_sha256(),
        "initial_parameter_state_sha256": initial_state_sha,
        "parameter_count": spec.parameter_count(),
        "head_geometry": {
            "n_heads": spec.n_heads,
            "n_kv_heads": spec.n_kv_heads,
            "head_dim": spec.head_dim,
            "rope_rotary_dim": spec.rope_rotary_dim,
            "q_projection_dim": spec.q_dim,
            "kv_projection_dim": spec.kv_dim,
        },
        "attention_projection_parameters": attention_total,
        "attention_projection_parameter_share": attention_total / spec.parameter_count(),
        "training_curve": train_curve,
        "validation_curve": val_curve,
        "step_latency_seconds": latencies,
        "step_latency_median_seconds": sorted(latencies)[len(latencies) // 2],
        "gradient_norm_min": min(point["grad_norm"] for point in train_curve),
        "gradient_norm_max": max(point["grad_norm"] for point in train_curve),
        "numerically_stable": finite,
    }


def run_matrix(
    repo_root: str | Path,
    *,
    source_sha: str,
    steps: int = 10,
    sequence_length: int = 128,
    seed: int = 20260825,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    stage = load_stage_config(root / STAGE_PATH)
    if stage.model.identity_sha256() != "2f0aa97a5d19e98c4e292fd5f1b454ada45ec4d2c7324e14ab7e48af19908ce6":
        raise HeadCountExperimentError("S1 incumbent identity drift")
    plan = json.loads((root / PLAN_PATH).read_text(encoding="utf-8"))
    tokenizer = ByteTokenizer()
    train_trace = _trace(
        _load_texts(root / "data/s0/packaged/train.jsonl"),
        tokenizer,
        count=steps,
        length=min(sequence_length, stage.model.max_seq_len),
    )
    validation_texts = _load_texts(root / "data/s0/packaged/validation.jsonl")
    validation_trace = _trace(
        validation_texts,
        tokenizer,
        count=len(validation_texts),
        length=min(sequence_length, stage.model.max_seq_len),
    )
    rows = []
    for entry in plan["candidates"]:
        spec = mha_geometry(stage.model, n_heads=int(entry["n_heads"]))
        if spec.head_dim != int(entry["head_dim"]):
            raise HeadCountExperimentError(f"{entry['id']} head_dim drift")
        if spec.identity_sha256() != entry["model_identity_sha256"]:
            raise HeadCountExperimentError(f"{entry['id']} identity drift")
        rows.append(
            _run_one(
                spec,
                candidate_id=entry["id"],
                init_spec=stage.init,
                train_trace=train_trace,
                validation_trace=validation_trace,
                seed=seed,
            )
        )
    if {row["parameter_count"] for row in rows} != {stage.expected_parameters}:
        raise RuntimeError("head-count matrix is not exact parameter comparable")
    initial_states = {row["initial_parameter_state_sha256"] for row in rows}
    if len(initial_states) != 1:
        raise RuntimeError("head-count candidates did not start from bitwise-identical weights")
    stable = [row for row in rows if row["numerically_stable"]]
    if not stable:
        raise RuntimeError("no numerically stable head candidate")
    ranked = sorted(
        stable,
        key=lambda row: (
            row["validation_curve"][-1]["loss"],
            row["step_latency_median_seconds"],
            abs(row["head_geometry"]["n_heads"] - 4),
        ),
    )
    provisional = ranked[0]
    report = {
        "schema": SCHEMA,
        "authority": "LOCAL_FREE_HEAD_GEOMETRY_EXPERIMENT_NOT_ARCHITECTURE_FREEZE",
        "source_sha": source_sha,
        "stage_incumbent_model_identity_sha256": stage.model.identity_sha256(),
        "fixed_controls": {
            "d_model": stage.model.d_model,
            "d_ff": stage.model.d_ff,
            "n_layers": stage.model.n_layers,
            "vocab_size": stage.model.vocab_size,
            "tokenizer": "s0-byte-v1 compatibility fixture",
            "context_limit": stage.model.max_seq_len,
            "executed_sequence_length": sequence_length,
            "optimizer": "AdamW",
            "learning_rate": 1e-2,
            "seed": seed,
            "init_spec_sha256": stage.init.identity_sha256(),
            "position_encoding_family": "rope",
            "attention_family": "MHA",
            "train_trace_sha256": _hash_trace(train_trace),
            "validation_trace_sha256": _hash_trace(validation_trace),
            "bitwise_identical_initial_weights": True,
        },
        "candidates": rows,
        "provisional_head_geometry": {
            "candidate_id": provisional["candidate_id"],
            **provisional["head_geometry"],
            "selection_basis": "finite real training, final held-out fixture loss, then median step latency",
            "status": "PROVISIONAL_NOT_FROZEN",
        },
        "claims": {
            "gqa_research_duplicated": False,
            "positional_encoding_changed": False,
            "paid_compute_used": False,
            "architecture_frozen": False,
        },
    }
    raw = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report["evidence_sha256"] = hashlib.sha256(raw).hexdigest()
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args(argv)
    report = run_matrix(
        args.repo_root,
        source_sha=args.source_sha,
        steps=args.steps,
        sequence_length=args.sequence_length,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
