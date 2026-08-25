"""Controlled ~1M SwiGLU/attention parameter-allocation experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training.config import TrainerConfig
from twelve_six.training.loss import causal_lm_loss
from twelve_six.training.trainer import Trainer

SCHEMA = "12-6.model12-ffn-ratio-1m.v1"
AUTHORITY = "LOCAL_FREE_ARCHITECTURE_EXPERIMENT_NOT_STAGE_OR_ARCHITECTURE_FREEZE"
CONTROL_PATH = "configs/stages/alternatives/s2_1m_byte_gqa.candidate.json"
EXPERIMENT_PATH = "configs/experiments/model12_ffn_ratio_1m.v1.json"


class FFNRatioExperimentError(ValueError):
    pass


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def solve_d_ff_for_attention_width(
    control: ModelSpec, *, head_dim: int, target_parameters: int
) -> ModelSpec:
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0:
        raise FFNRatioExperimentError("head_dim must be a positive integer")
    if head_dim % 2:
        raise FFNRatioExperimentError("RoPE head_dim must be even")
    if target_parameters <= 0:
        raise FFNRatioExperimentError("target_parameters must be positive")
    if control.n_heads % control.n_kv_heads:
        raise FFNRatioExperimentError("invalid control query/KV geometry")
    probe = replace(control, head_dim=head_dim, rope_rotary_dim=head_dim, d_ff=1)
    slope = 3 * probe.d_model * probe.n_layers
    constant = probe.parameter_count() - slope
    numerator = target_parameters - constant
    if numerator <= 0 or numerator % slope:
        raise FFNRatioExperimentError(
            "requested attention width cannot be exactly compensated by integer d_ff"
        )
    d_ff = numerator // slope
    if d_ff <= 0:
        raise FFNRatioExperimentError("solved d_ff must be positive")
    candidate = replace(probe, d_ff=d_ff)
    if candidate.parameter_count() != target_parameters:
        raise AssertionError("exact parameter solver drift")
    return candidate


def _load_texts(path: Path) -> list[str]:
    texts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"]:
            raise FFNRatioExperimentError(f"{path}:{line_number} invalid text record")
        texts.append(row["text"])
    if not texts:
        raise FFNRatioExperimentError(f"{path} contains no texts")
    return texts


def _trace(
    texts: Sequence[str], tokenizer: ByteTokenizer, *, steps: int, sequence_length: int
) -> list[dict[str, torch.Tensor]]:
    batches: list[dict[str, torch.Tensor]] = []
    for step in range(steps):
        ids = tokenizer.encode(texts[step % len(texts)])[:sequence_length]
        if len(ids) < 2:
            raise FFNRatioExperimentError("trace record encodes to fewer than two tokens")
        tensor = torch.tensor([ids], dtype=torch.long)
        batches.append({"input_ids": tensor, "labels": tensor.clone()})
    return batches


def _trace_identity(batches: Sequence[Mapping[str, torch.Tensor]]) -> str:
    return _canonical_hash([batch["input_ids"].tolist() for batch in batches])


@torch.no_grad()
def _evaluate(model: TwelveSixDecoder, batches: Sequence[Mapping[str, torch.Tensor]]) -> float:
    model.eval()
    weighted = 0.0
    count = 0
    for batch in batches:
        logits = model(batch["input_ids"]).logits
        labels = batch["labels"]
        tokens = int(labels[:, 1:].ne(-100).sum().item())
        loss = causal_lm_loss(logits, labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError("non-finite validation loss")
        weighted += float(loss.item()) * tokens
        count += tokens
    if count <= 0:
        raise RuntimeError("validation trace has no scoreable tokens")
    return weighted / count


def _snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _rss_hwm_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class _Telemetry:
    def __init__(self, model: TwelveSixDecoder) -> None:
        self.activation: list[dict[str, float]] = [
            {"attn_rms_sum": 0.0, "mlp_rms_sum": 0.0, "observations": 0.0}
            for _ in model.blocks
        ]
        self.grad_sq = [0.0 for _ in model.blocks]
        self.grad_nonzero = [0 for _ in model.blocks]
        self.grad_elements = [0 for _ in model.blocks]
        self.current_tokens = 1
        self._handles: list[Any] = []
        for index, block in enumerate(model.blocks):
            self._handles.append(block.attn.register_forward_hook(self._activation_hook(index, "attn")))
            self._handles.append(block.mlp.register_forward_hook(self._activation_hook(index, "mlp")))
            for parameter in block.parameters():
                self._handles.append(parameter.register_hook(self._grad_hook(index)))

    def _activation_hook(self, index: int, kind: str):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            value = output.detach().float()
            rms = float(torch.sqrt(torch.mean(value * value)).item())
            self.activation[index][f"{kind}_rms_sum"] += rms
            if kind == "mlp":
                self.activation[index]["observations"] += 1.0
        return hook

    def _grad_hook(self, index: int):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            normalized = grad.detach().float() / max(self.current_tokens, 1)
            self.grad_sq[index] += float(torch.sum(normalized * normalized).item())
            self.grad_nonzero[index] += int(normalized.ne(0).sum().item())
            self.grad_elements[index] += normalized.numel()
            return grad
        return hook

    def set_tokens(self, tokens: int) -> None:
        self.current_tokens = tokens

    def report(self) -> dict[str, Any]:
        activation = []
        gradients = []
        for index, stats in enumerate(self.activation):
            observations = max(int(stats["observations"]), 1)
            activation.append(
                {
                    "layer": index,
                    "attn_output_rms_mean": stats["attn_rms_sum"] / observations,
                    "mlp_output_rms_mean": stats["mlp_rms_sum"] / observations,
                }
            )
            gradients.append(
                {
                    "layer": index,
                    "gradient_l2_accumulated": math.sqrt(self.grad_sq[index]),
                    "nonzero_gradient_fraction": (
                        self.grad_nonzero[index] / self.grad_elements[index]
                        if self.grad_elements[index]
                        else 0.0
                    ),
                }
            )
        return {"activation_scales": activation, "per_layer_gradient_norms": gradients}

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def _candidate_from_entry(control: ModelSpec, entry: Mapping[str, Any], target: int) -> ModelSpec:
    candidate = solve_d_ff_for_attention_width(
        control, head_dim=int(entry["head_dim"]), target_parameters=target
    )
    if candidate.d_ff != int(entry["d_ff"]):
        raise FFNRatioExperimentError(f"{entry['id']} d_ff identity drift")
    if candidate.identity_sha256() != str(entry["model_identity_sha256"]):
        raise FFNRatioExperimentError(f"{entry['id']} ModelSpec identity drift")
    return candidate


def run_candidate(
    *,
    candidate_id: str,
    spec: ModelSpec,
    init_spec: InitSpec,
    train_trace: Sequence[Mapping[str, torch.Tensor]],
    validation_trace: Sequence[Mapping[str, torch.Tensor]],
    seed: int,
    eval_every: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    before = _snapshot(model)
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            betas=(0.9, 0.95),
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
    telemetry = _Telemetry(model)
    validation_curve = [{"optimizer_step": 0, "loss": _evaluate(model, validation_trace)}]
    train_curve: list[dict[str, Any]] = []
    step_times: list[float] = []
    for index, batch in enumerate(train_trace, 1):
        tokens = int(batch["labels"][:, 1:].ne(-100).sum().item())
        telemetry.set_tokens(tokens)
        start = time.perf_counter()
        metrics = trainer.train_microbatch(batch)
        step_times.append(time.perf_counter() - start)
        if metrics.grad_norm is None or not math.isfinite(metrics.grad_norm):
            raise RuntimeError(f"{candidate_id} missing finite gradient norm")
        train_curve.append(
            {
                "optimizer_step": metrics.optimizer_step,
                "loss": metrics.loss,
                "update_loss": metrics.update_loss,
                "grad_norm": metrics.grad_norm,
                "tokens": metrics.tokens,
            }
        )
        if index % eval_every == 0 or index == len(train_trace):
            validation_curve.append(
                {"optimizer_step": trainer.optimizer_step, "loss": _evaluate(model, validation_trace)}
            )
    trainer.assert_checkpoint_safe()
    telemetry_report = telemetry.report()
    telemetry.close()

    changed = 0
    total = 0
    for name, parameter in model.named_parameters():
        delta = parameter.detach() - before[name]
        changed += int(delta.ne(0).sum().item())
        total += delta.numel()
    breakdown = spec.parameter_breakdown()
    total_params = breakdown["total"]
    return {
        "candidate_id": candidate_id,
        "model": spec.to_dict(),
        "model_identity_sha256": spec.identity_sha256(),
        "parameter_count": total_params,
        "ffn_expansion_ratio": spec.d_ff / spec.d_model,
        "allocation": {
            "attention_parameters_total": spec.n_layers * breakdown["attention_per_layer"],
            "mlp_parameters_total": spec.n_layers * breakdown["mlp_per_layer"],
            "embedding_parameters": breakdown["token_embedding"],
            "attention_share": spec.n_layers * breakdown["attention_per_layer"] / total_params,
            "mlp_share": spec.n_layers * breakdown["mlp_per_layer"] / total_params,
            "block_share": breakdown["blocks_total"] / total_params,
        },
        "training_curve": train_curve,
        "validation_curve": validation_curve,
        "telemetry": telemetry_report,
        "parameter_utilization": {
            "changed_parameter_elements": changed,
            "trainable_parameter_elements": total,
            "changed_fraction": changed / total,
        },
        "runtime": {
            "step_seconds": step_times,
            "step_seconds_mean": sum(step_times) / len(step_times),
            "step_seconds_median": sorted(step_times)[len(step_times) // 2],
            "rss_hwm_bytes": _rss_hwm_bytes(),
            "model_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "optimizer_tensor_bytes": _tensor_bytes(trainer.optimizer.state_dict()),
        },
        "numerically_stable": all(
            math.isfinite(point["loss"]) and math.isfinite(point["grad_norm"])
            for point in train_curve
        )
        and all(math.isfinite(point["loss"]) for point in validation_curve),
    }


def run_matrix(
    repo_root: str | Path,
    *,
    source_sha: str,
    steps: int = 8,
    sequence_length: int = 128,
    seed: int = 20260825,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if steps < 2 or sequence_length < 2:
        raise FFNRatioExperimentError("steps and sequence_length must be >= 2")
    control_stage = load_stage_config(root / CONTROL_PATH)
    control = control_stage.model
    if control.identity_sha256() != "18284b303eb31cef5191ddb3ed4ddba5ce51789aadf4b14cc90d4226c5c527b5":
        raise FFNRatioExperimentError("executable ~1M incumbent drifted")
    experiment = json.loads((root / EXPERIMENT_PATH).read_text(encoding="utf-8"))
    tokenizer = ByteTokenizer()
    if tokenizer.vocab_size != control.vocab_size:
        raise FFNRatioExperimentError("control tokenizer vocabulary mismatch")
    train_texts = _load_texts(root / "data/s0/packaged/train.jsonl")
    validation_texts = _load_texts(root / "data/s0/packaged/validation.jsonl")
    train_trace = _trace(
        train_texts, tokenizer, steps=steps, sequence_length=min(sequence_length, control.max_seq_len)
    )
    validation_trace = _trace(
        validation_texts,
        tokenizer,
        steps=len(validation_texts),
        sequence_length=min(sequence_length, control.max_seq_len),
    )
    results = []
    for entry in experiment["candidates"]:
        spec = _candidate_from_entry(control, entry, control_stage.expected_parameters)
        results.append(
            run_candidate(
                candidate_id=str(entry["id"]),
                spec=spec,
                init_spec=control_stage.init,
                train_trace=train_trace,
                validation_trace=validation_trace,
                seed=seed,
                eval_every=int(experiment["evaluation_every_steps"]),
            )
        )
    control_rows = [row for row in results if row["candidate_id"] == experiment["control_id"]]
    if len(control_rows) != 1 or control_rows[0]["model_identity_sha256"] != control.identity_sha256():
        raise RuntimeError("matrix does not contain exact incumbent control")
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "repository": "Oleksii-debug/12-6-ai.",
        "source_sha": source_sha,
        "control_id": experiment["control_id"],
        "control_model_identity_sha256": control.identity_sha256(),
        "parameter_target": control_stage.expected_parameters,
        "fixed_controls": {
            "tokenizer": "s0-byte-v1",
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "data": "controlled S0 compatibility fixture",
            "max_model_context": control.max_seq_len,
            "executed_sequence_length": sequence_length,
            "optimizer": "AdamW",
            "learning_rate": 1e-3,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "init_spec_sha256": control_stage.init.identity_sha256(),
            "seed": seed,
            "train_trace_sha256": _trace_identity(train_trace),
            "validation_trace_sha256": _trace_identity(validation_trace),
        },
        "candidates": results,
        "selection": {
            "winner_selected_from_s0_loss": False,
            "architecture_frozen": False,
            "recommendation_status": "EXPERIMENTAL_MECHANICS_ONLY",
            "note": (
                "The tiny S0 compatibility-fixture loss is retained as a diagnostic curve "
                "and is explicitly excluded from architecture winner selection. Transfer "
                "requires representative next-stage held-out data."
            ),
        },
        "claims": {
            "paid_compute_used": False,
            "tokenizer_changed_across_candidates": False,
            "data_changed_across_candidates": False,
            "context_changed_across_candidates": False,
            "gqa_ratio_changed_across_candidates": False,
            "architecture_freeze_granted": False,
        },
    }
    report["evidence_sha256"] = _canonical_hash(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema") != SCHEMA:
        raise FFNRatioExperimentError("wrong evidence schema")
    candidate_rows = report.get("candidates")
    if not isinstance(candidate_rows, list) or len(candidate_rows) < 3:
        raise FFNRatioExperimentError("insufficient candidate matrix")
    if {int(row["parameter_count"]) for row in candidate_rows} != {992_896}:
        raise FFNRatioExperimentError("candidate matrix is not exact iso-parameter")
    if not all(bool(row["numerically_stable"]) for row in candidate_rows):
        raise FFNRatioExperimentError("non-finite candidate evidence")
    payload = dict(report)
    observed = payload.pop("evidence_sha256", None)
    if observed != _canonical_hash(payload):
        raise FFNRatioExperimentError("evidence self-hash mismatch")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=8)
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
    validate_report(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
