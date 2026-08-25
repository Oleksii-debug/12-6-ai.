"""LOCAL_FREE controlled architecture sweeps for MODEL-12 and MODEL-13.

This module is experimental evidence infrastructure. It never promotes or freezes a
stage and it refuses to use the S0 compatibility fixture as ~1M quality authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training.loss import causal_lm_loss

FFN_SCHEMA = "12-6.model12-ffn-ratio-1m.v1"
HEAD_SCHEMA = "12-6.model13-head-count-100k.v1"
EVIDENCE_SCHEMA = "12-6.model12-13-architecture-sweeps.v1"
AUTHORITY = "LOCAL_FREE_ENGINEERING_EXPERIMENT_NOT_STAGE_OR_QUALITY_EVIDENCE"


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _load_texts(path: Path) -> list[str]:
    rows: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        _require(isinstance(row, dict), f"{path}:{line_number} must be an object")
        text = row.get("text")
        _require(isinstance(text, str) and bool(text), f"{path}:{line_number} missing text")
        rows.append(text)
    _require(bool(rows), f"{path} has no texts")
    return rows


def _rss_bytes() -> int | None:
    """Current RSS on Linux without adding a dependency to the locked environment."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        return None


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _compose_spec(config: Mapping[str, Any], candidate: Mapping[str, Any]) -> ModelSpec:
    fixed = config["fixed_controls"]
    schema = config["schema_version"]
    if schema == FFN_SCHEMA:
        payload = {
            "schema_version": 1,
            "vocab_size": 256,
            "max_seq_len": int(fixed["max_seq_len"]),
            "d_model": int(fixed["d_model"]),
            "n_layers": int(fixed["n_layers"]),
            "n_heads": int(fixed["n_heads"]),
            "n_kv_heads": int(fixed["n_kv_heads"]),
            "head_dim": int(candidate["head_dim"]),
            "d_ff": int(candidate["d_ff"]),
            "rope_rotary_dim": int(candidate["rope_rotary_dim"]),
        }
    elif schema == HEAD_SCHEMA:
        payload = {
            "schema_version": 1,
            "vocab_size": 512,
            "max_seq_len": int(fixed["max_seq_len"]),
            "d_model": int(fixed["d_model"]),
            "n_layers": int(fixed["n_layers"]),
            "n_heads": int(candidate["n_heads"]),
            "n_kv_heads": int(candidate["n_kv_heads"]),
            "head_dim": int(candidate["head_dim"]),
            "d_ff": int(fixed["d_ff"]),
            "rope_rotary_dim": int(candidate["rope_rotary_dim"]),
        }
    else:
        raise ValueError(f"unsupported experiment schema: {schema}")
    payload.update(
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=float(fixed["rope_theta"]),
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )
    return ModelSpec.from_dict(payload)


def validate_experiment_config(config: Mapping[str, Any]) -> None:
    schema = config.get("schema_version")
    _require(schema in {FFN_SCHEMA, HEAD_SCHEMA}, "unsupported experiment schema")
    _require(config.get("authority") == AUTHORITY, "experiment authority drift")
    candidates = config.get("candidates")
    _require(isinstance(candidates, list) and len(candidates) >= 3, "candidate set missing")
    controls = 0
    counts: set[int] = set()
    for candidate in candidates:
        _require(isinstance(candidate, Mapping), "candidate must be an object")
        spec = _compose_spec(config, candidate)
        actual = spec.parameter_count()
        _require(actual == candidate.get("expected_parameters"), "candidate parameter drift")
        _require(spec.identity_sha256() == candidate.get("model_identity_sha256"), "identity drift")
        _require(spec.rope_rotary_dim == spec.head_dim, "full-head RoPE geometry drift")
        counts.add(actual)
        controls += int(candidate.get("control") is True)
        if schema == HEAD_SCHEMA:
            _require(spec.n_heads == spec.n_kv_heads, "MODEL-13 must remain MHA")
            _require(spec.q_dim == spec.d_model and spec.kv_dim == spec.d_model, "MHA width drift")
    _require(len(counts) == 1, "candidates are not exact iso-parameter")
    _require(controls == 1, "exactly one incumbent control is required")
    if schema == FFN_SCHEMA:
        incumbent = config["source"]["incumbent_model_identity_sha256"]
        control = next(item for item in candidates if item.get("control") is True)
        _require(control["model_identity_sha256"] == incumbent, "~1M control no longer matches incumbent")
        _require(config["selection_policy"]["quality_selection_from_fixture_loss_allowed"] is False, "fixture quality selection must be forbidden")
    else:
        incumbent = config["source"]["incumbent_model_identity_sha256"]
        control = next(item for item in candidates if item.get("control") is True)
        _require(control["model_identity_sha256"] == incumbent, "100K control no longer matches incumbent")


def _init_spec(config: Mapping[str, Any]) -> InitSpec:
    raw = config["fixed_controls"]["init"]
    return InitSpec(
        schema_version=1,
        family=str(raw["family"]),
        std=float(raw["std"]),
        residual_branch_scale=str(raw["residual_branch_scale"]),
    )


def _token_ids(tokenizer: ByteTokenizer, text: str, sequence_length: int) -> list[int]:
    ids = tokenizer.encode(text)[:sequence_length]
    _require(len(ids) >= 2, "trace row must encode to at least two tokens")
    return ids


def _trace_sha256(
    tokenizer: ByteTokenizer,
    texts: Sequence[str],
    *,
    steps: int,
    sequence_length: int,
) -> str:
    trace = [
        _token_ids(tokenizer, texts[step % len(texts)], sequence_length)
        for step in range(steps)
    ]
    return _hash_json(trace)


@torch.no_grad()
def _evaluate(
    model: TwelveSixDecoder,
    tokenizer: ByteTokenizer,
    texts: Sequence[str],
    sequence_length: int,
) -> tuple[float, int]:
    model.eval()
    weighted = 0.0
    tokens = 0
    for text in texts:
        ids = torch.tensor([_token_ids(tokenizer, text, sequence_length)], dtype=torch.long)
        loss = causal_lm_loss(model(ids).logits, ids)
        count = ids.shape[1] - 1
        _require(torch.isfinite(loss).item(), "evaluation produced non-finite loss")
        weighted += float(loss.item()) * count
        tokens += count
    _require(tokens > 0, "evaluation has no target tokens")
    return weighted / tokens, tokens


def _grad_l2(parameters: Sequence[torch.nn.Parameter]) -> tuple[float, int, int]:
    squared = 0.0
    nonzero = 0
    total = 0
    for parameter in parameters:
        total += parameter.numel()
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        _require(torch.isfinite(grad).all().item(), "non-finite gradient")
        squared += float(torch.sum(grad * grad).item())
        nonzero += int(grad.ne(0).sum().item())
    return math.sqrt(squared), nonzero, total


def _layer_gradients(model: TwelveSixDecoder) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for index, block in enumerate(model.blocks):
        block_norm, block_nonzero, block_total = _grad_l2(list(block.parameters()))
        attn_norm, _, _ = _grad_l2(list(block.attn.parameters()))
        mlp_norm, _, _ = _grad_l2(list(block.mlp.parameters()))
        rows.append(
            {
                "layer": index,
                "block_grad_l2": block_norm,
                "attn_grad_l2": attn_norm,
                "mlp_grad_l2": mlp_norm,
                "block_nonzero_grad_fraction": block_nonzero / block_total,
            }
        )
    return rows


class _ActivationRecorder:
    def __init__(self, model: TwelveSixDecoder) -> None:
        self.enabled = False
        self.current: dict[str, dict[str, float]] = {}
        self.handles = []
        for index, block in enumerate(model.blocks):
            self.handles.append(block.attn.register_forward_hook(self._hook(index, "attn")))
            self.handles.append(block.mlp.register_forward_hook(self._hook(index, "mlp")))

    def _hook(self, index: int, kind: str):
        def capture(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            if not self.enabled:
                return
            tensor = output.detach().float()
            self.current[f"layer_{index}.{kind}"] = {
                "rms": float(torch.sqrt(torch.mean(tensor * tensor)).item()),
                "max_abs": float(tensor.abs().max().item()),
            }
        return capture

    def begin(self) -> None:
        self.current = {}
        self.enabled = True

    def end(self) -> dict[str, dict[str, float]]:
        self.enabled = False
        return dict(self.current)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _weight_delta(model: TwelveSixDecoder, before: Mapping[str, torch.Tensor]) -> dict[str, float | int]:
    squared = 0.0
    changed = 0
    total = 0
    max_abs = 0.0
    for name, parameter in model.named_parameters():
        delta = parameter.detach().float() - before[name].float()
        squared += float(torch.sum(delta * delta).item())
        changed += int(delta.ne(0).sum().item())
        total += delta.numel()
        max_abs = max(max_abs, float(delta.abs().max().item()))
    return {
        "l2": math.sqrt(squared),
        "max_abs": max_abs,
        "changed_parameter_elements": changed,
        "trainable_parameter_elements": total,
        "changed_fraction": changed / total,
    }


def _run_candidate(
    root: Path,
    config: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    fixed = config["fixed_controls"]
    sequence_length = int(fixed["executed_sequence_length"])
    tokenizer = ByteTokenizer()
    train_texts = _load_texts(root / str(fixed["train_data"]))
    validation_texts = _load_texts(root / str(fixed["validation_data"]))
    trace_sha256 = _trace_sha256(tokenizer, train_texts, steps=steps, sequence_length=sequence_length)

    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    spec = _compose_spec(config, candidate)
    model = TwelveSixDecoder(spec, _init_spec(config))
    before = _snapshot(model)
    model_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    optimizer_cfg = fixed["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["learning_rate"]),
        betas=tuple(float(x) for x in optimizer_cfg["betas"]),
        eps=float(optimizer_cfg["eps"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )
    recorder = _ActivationRecorder(model)
    train_curve: list[dict[str, float | int]] = []
    validation_curve: list[dict[str, float | int]] = []
    gradient_curve: list[dict[str, Any]] = []
    activation_curve: list[dict[str, Any]] = []
    utilization_curve: list[dict[str, float | int]] = []
    step_times: list[float] = []
    clip_events = 0
    rss_start = _rss_bytes()
    rss_peak = rss_start

    initial_train, train_eval_tokens = _evaluate(model, tokenizer, train_texts, sequence_length)
    initial_val, validation_eval_tokens = _evaluate(model, tokenizer, validation_texts, sequence_length)
    train_curve.append({"step": 0, "eval_loss": initial_train, "tokens": train_eval_tokens})
    validation_curve.append({"step": 0, "loss": initial_val, "tokens": validation_eval_tokens})

    for step in range(1, steps + 1):
        ids = torch.tensor(
            [_token_ids(tokenizer, train_texts[(step - 1) % len(train_texts)], sequence_length)],
            dtype=torch.long,
        )
        valid_tokens = ids.shape[1] - 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        recorder.begin()
        start = time.perf_counter()
        loss = causal_lm_loss(model(ids).logits, ids)
        (loss * valid_tokens).backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(valid_tokens)
        global_norm, nonzero, total = _grad_l2(list(model.parameters()))
        layers = _layer_gradients(model)
        activations = recorder.end()
        _require(math.isfinite(float(loss.detach().item())) and math.isfinite(global_norm), "non-finite training state")
        if global_norm > float(optimizer_cfg["gradient_clip_norm"]):
            clip_events += 1
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(optimizer_cfg["gradient_clip_norm"]),
            error_if_nonfinite=True,
        )
        optimizer.step()
        step_times.append(time.perf_counter() - start)
        current_rss = _rss_bytes()
        if current_rss is not None:
            rss_peak = current_rss if rss_peak is None else max(rss_peak, current_rss)

        train_curve.append({"step": step, "batch_loss": float(loss.detach().item()), "optimized_tokens": valid_tokens})
        gradient_curve.append({"step": step, "global_preclip_grad_l2": global_norm, "layers": layers})
        activation_curve.append({"step": step, "activations": activations})
        utilization_curve.append(
            {
                "step": step,
                "nonzero_grad_elements": nonzero,
                "trainable_elements": total,
                "nonzero_grad_fraction": nonzero / total,
            }
        )
        train_eval, train_tokens = _evaluate(model, tokenizer, train_texts, sequence_length)
        val_eval, val_tokens = _evaluate(model, tokenizer, validation_texts, sequence_length)
        train_curve.append({"step": step, "eval_loss": train_eval, "tokens": train_tokens})
        validation_curve.append({"step": step, "loss": val_eval, "tokens": val_tokens})

    recorder.close()
    delta = _weight_delta(model, before)
    breakdown = spec.parameter_breakdown()
    attention_total = breakdown["attention_per_layer"] * spec.n_layers
    mlp_total = breakdown["mlp_per_layer"] * spec.n_layers
    steady = step_times[2:] if len(step_times) > 2 else step_times
    result = {
        "candidate_id": candidate["candidate_id"],
        "control": candidate.get("control") is True,
        "seed": seed,
        "model_identity_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "model_spec": spec.to_dict(),
        "parameter_breakdown": breakdown,
        "attention_projection_parameter_share": attention_total / spec.parameter_count(),
        "mlp_parameter_share": mlp_total / spec.parameter_count(),
        "trace_sha256": trace_sha256,
        "training_curve": train_curve,
        "validation_curve": validation_curve,
        "per_layer_gradient_norms": gradient_curve,
        "activation_scales": activation_curve,
        "parameter_utilization": utilization_curve,
        "weight_delta": delta,
        "clip_frequency": clip_events / steps,
        "step_time_seconds": {
            "all": step_times,
            "mean": statistics.mean(step_times),
            "steady_state_median": statistics.median(steady),
        },
        "memory": {
            "model_parameter_bytes": model_bytes,
            "optimizer_tensor_bytes_after_training": _tensor_bytes(optimizer.state_dict()),
            "rss_start_bytes": rss_start,
            "rss_peak_sampled_bytes": rss_peak,
            "rss_peak_sampled_delta_bytes": (
                None if rss_start is None or rss_peak is None else max(0, rss_peak - rss_start)
            ),
        },
        "numerical_stability": {"all_finite": True},
    }
    return result


def _aggregate_head_runs(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["candidate_id"]), []).append(run)
    rows = []
    for candidate_id, items in grouped.items():
        rows.append(
            {
                "candidate_id": candidate_id,
                "seeds": [item["seed"] for item in items],
                "parameter_count": items[0]["parameter_count"],
                "model_identity_sha256": items[0]["model_identity_sha256"],
                "mean_steady_state_median_step_seconds": statistics.mean(
                    float(item["step_time_seconds"]["steady_state_median"]) for item in items
                ),
                "mean_final_validation_loss": statistics.mean(
                    float(item["validation_curve"][-1]["loss"]) for item in items
                ),
                "validation_loss_population_sd": statistics.pstdev(
                    float(item["validation_curve"][-1]["loss"]) for item in items
                ),
                "mean_clip_frequency": statistics.mean(float(item["clip_frequency"]) for item in items),
                "all_finite": all(bool(item["numerical_stability"]["all_finite"]) for item in items),
            }
        )
    return sorted(rows, key=lambda row: row["candidate_id"])


def run_experiments(root: Path, ffn_config_path: Path, head_config_path: Path) -> dict[str, Any]:
    ffn_config = _load_json(ffn_config_path)
    head_config = _load_json(head_config_path)
    validate_experiment_config(ffn_config)
    validate_experiment_config(head_config)

    ffn_fixed = ffn_config["fixed_controls"]
    ffn_runs = [
        _run_candidate(
            root,
            ffn_config,
            candidate,
            seed=int(ffn_fixed["seed"]),
            steps=int(ffn_fixed["optimizer_steps"]),
        )
        for candidate in ffn_config["candidates"]
    ]
    _require(len({run["trace_sha256"] for run in ffn_runs}) == 1, "MODEL-12 token trace drift")

    head_fixed = head_config["fixed_controls"]
    head_runs = [
        _run_candidate(
            root,
            head_config,
            candidate,
            seed=int(seed),
            steps=int(head_fixed["optimizer_steps_per_seed"]),
        )
        for seed in head_fixed["seeds"]
        for candidate in head_config["candidates"]
    ]
    for seed in head_fixed["seeds"]:
        seed_runs = [run for run in head_runs if run["seed"] == seed]
        _require(len({run["trace_sha256"] for run in seed_runs}) == 1, "MODEL-13 token trace drift")

    head_aggregate = _aggregate_head_runs(head_runs)
    eligible = [row for row in head_aggregate if row["all_finite"]]
    provisional = min(eligible, key=lambda row: row["mean_steady_state_median_step_seconds"])
    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA,
        "authority": AUTHORITY,
        "paid_compute_used": False,
        "model12_ffn_ratio_1m": {
            "status": "EXECUTED",
            "runs": ffn_runs,
            "quality_winner": None,
            "quality_selection_forbidden": True,
            "architecture_freeze": False,
            "conclusion": (
                "All candidates are mechanics/stability probes only because the fixed S0 fixture "
                "is explicitly non-authoritative for ~1M quality selection."
            ),
        },
        "model13_head_count_100k": {
            "status": "EXECUTED_MULTI_SEED",
            "runs": head_runs,
            "aggregate": head_aggregate,
            "provisional_head_geometry": provisional["candidate_id"],
            "selection_basis": "finite multi-seed runs and lowest mean steady-state median LOCAL_FREE CPU step time",
            "architecture_freeze": False,
        },
    }
    evidence["evidence_sha256"] = _hash_json(evidence)
    return evidence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MODEL-12 and MODEL-13 LOCAL_FREE architecture sweeps.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--ffn-config", type=Path, default=Path("configs/experiments/model12_ffn_ratio_1m.json"))
    parser.add_argument("--head-config", type=Path, default=Path("configs/experiments/model13_head_count_100k.json"))
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = run_experiments(args.repo_root.resolve(), args.ffn_config, args.head_config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "evidence_sha256": evidence["evidence_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
