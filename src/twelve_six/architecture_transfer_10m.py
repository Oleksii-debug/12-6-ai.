"""Bounded matched-parameter architecture-transfer experiment for MODEL-142.

This module intentionally tests one axis only: MHA versus 8Q/4KV versus the
current 8Q/2KV S3 incumbent. It is LOCAL_FREE controlled evidence, not an
architecture freeze or a representative-corpus quality claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import resource
import statistics
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .model import InitSpec, ModelSpec, TwelveSixDecoder
from .tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)

SCHEMA = "12-6.model142-10m-transfer-matrix.v1"
RUN_SCHEMA = "12-6.model142-10m-transfer-run.v1"
AUTHORITY = "LOCAL_FREE_CONTROLLED_TRANSFER_EVIDENCE_NOT_ARCHITECTURE_FREEZE"
INCUMBENT_MODEL_SHA = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
INCUMBENT_INIT_SHA = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("id"), str):
            raise ValueError(f"{path}:{line_number} invalid record")
        if not isinstance(value.get("text"), str) or not value["text"]:
            raise ValueError(f"{path}:{line_number} invalid text")
        records.append(value)
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def _byte_stream(records: list[dict[str, Any]], tokenizer: ByteTokenizer) -> bytes:
    return b"\n".join(bytes(tokenizer.encode(str(record["text"]))) for record in records) + b"\n"


def _make_batch(
    stream: bytes, *, step: int, batch_size: int, sequence_length: int
) -> torch.Tensor:
    width = batch_size * sequence_length
    base = (step * width) % len(stream)
    rows: list[list[int]] = []
    for batch_index in range(batch_size):
        start = (base + batch_index * sequence_length) % len(stream)
        rows.append(
            [stream[(start + offset) % len(stream)] for offset in range(sequence_length)]
        )
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def _validation(
    model: TwelveSixDecoder,
    records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_targets = 0
    for record in records:
        token_ids = tokenizer.encode(str(record["text"]))
        start = 0
        while start < len(token_ids) - 1:
            chunk = token_ids[start : start + model.spec.max_seq_len]
            if len(chunk) < 2:
                break
            ids = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
            logits = model(ids).logits
            total_nll += float(
                F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, model.spec.vocab_size),
                    ids[:, 1:].reshape(-1),
                    reduction="sum",
                ).item()
            )
            total_targets += ids.shape[1] - 1
            start += model.spec.max_seq_len - 1
    model.train(was_training)
    ce = total_nll / total_targets
    return {"ce_nats": ce, "bpb": ce / math.log(2.0), "targets": total_targets}


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _gradient_health(parameters: list[torch.nn.Parameter]) -> dict[str, float]:
    total = 0
    finite = 0
    nonzero = 0
    grad_sq = 0.0
    weight_sq = 0.0
    for parameter in parameters:
        total += parameter.numel()
        weight_sq += float(torch.sum(parameter.detach().float().square()).item())
        if parameter.grad is not None:
            gradient = parameter.grad.detach()
            finite += int(torch.isfinite(gradient).sum().item())
            nonzero += int((gradient != 0).sum().item())
            grad_sq += float(torch.sum(gradient.float().square()).item())
    return {
        "grad_finite_fraction": finite / total,
        "grad_nonzero_fraction": nonzero / total,
        "grad_norm": math.sqrt(grad_sq),
        "weight_norm": math.sqrt(weight_sq),
    }


def _update_health(
    parameters: list[torch.nn.Parameter], initial: list[torch.Tensor]
) -> dict[str, float]:
    delta_sq = 0.0
    initial_sq = 0.0
    changed = 0
    total = 0
    for parameter, initial_value in zip(parameters, initial, strict=True):
        current = parameter.detach().float()
        delta = current - initial_value
        delta_sq += float(torch.sum(delta.square()).item())
        initial_sq += float(torch.sum(initial_value.square()).item())
        changed += int((delta != 0).sum().item())
        total += parameter.numel()
    return {
        "delta_over_initial_weight": math.sqrt(delta_sq) / (math.sqrt(initial_sq) + 1e-30),
        "changed_fraction": changed / total,
    }


def load_experiment_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != "MODEL-142-10M-TRANSFER-MATRIX-v1":
        raise ValueError("unexpected MODEL-142 experiment identity")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValueError("MODEL-142 must contain exactly three candidates")
    incumbent_count = None
    for candidate in candidates:
        spec = ModelSpec.from_dict(candidate["model"])
        count = spec.parameter_count()
        if count != int(candidate["expected_parameters"]):
            raise ValueError(f"parameter count drift for {candidate['id']}")
        if candidate["id"] == "incumbent_gqa_8q2kv":
            incumbent_count = count
            if spec.identity_sha256() != INCUMBENT_MODEL_SHA:
                raise ValueError("incumbent ModelSpec identity drift")
    if incumbent_count != 10_000_640:
        raise ValueError("current 10M incumbent parameter count drift")
    for candidate in candidates:
        delta = int(candidate["expected_parameters"]) - incumbent_count
        if abs(delta) / incumbent_count > 0.001:
            raise ValueError("candidate exceeds 0.1% parameter-match tolerance")
    controls = config["controls"]
    if controls["tokenizer"] != BYTE_TOKENIZER_VERSION:
        raise ValueError("tokenizer drift")
    if controls["init"]["expected_identity_sha256"] != INCUMBENT_INIT_SHA:
        raise ValueError("InitSpec binding drift")
    if int(controls["optimized_causal_tokens"]) != (
        int(controls["optimizer_steps"])
        * int(controls["batch_size"])
        * (int(controls["training_sequence_length"]) - 1)
    ):
        raise ValueError("optimized-token budget drift")
    return config


def _candidate(config: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [item for item in config["candidates"] if item["id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"unknown candidate: {candidate_id}")
    return matches[0]


def run_candidate(
    *,
    repo_root: Path,
    source_sha: str,
    config_path: Path,
    candidate_id: str,
    seed: int,
    torch_threads: int = 2,
) -> dict[str, Any]:
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("exact source checkout mismatch")
    config = load_experiment_config(config_path)
    controls = config["controls"]
    if seed not in [int(value) for value in controls["seeds"]]:
        raise ValueError("seed is not predeclared")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = ByteTokenizer()
    train_path = repo_root / controls["train_path"]
    validation_path = repo_root / controls["validation_path"]
    train_records = _read_records(train_path)
    validation_records = _read_records(validation_path)
    train_ids = {str(item["id"]) for item in train_records}
    validation_ids = {str(item["id"]) for item in validation_records}
    if train_ids & validation_ids:
        raise RuntimeError("train/validation overlap")
    stream = _byte_stream(train_records, tokenizer)

    entry = _candidate(config, candidate_id)
    spec = ModelSpec.from_dict(entry["model"])
    init = InitSpec()
    if init.identity_sha256() != INCUMBENT_INIT_SHA:
        raise RuntimeError("InitSpec identity drift")
    model = TwelveSixDecoder(spec, init)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial = [parameter.detach().float().clone() for parameter in parameters]
    block_initial = [
        [parameter.detach().float().clone() for parameter in block.parameters()]
        for block in model.blocks
    ]
    optimizer_config = controls["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )

    steps = int(controls["optimizer_steps"])
    batch_size = int(controls["batch_size"])
    sequence_length = int(controls["training_sequence_length"])
    checkpoints = {0: _validation(model, validation_records, tokenizer)}
    checkpoint_steps = {steps // 4, steps // 2, steps}
    train_losses: list[float] = []
    grad_norms: list[float] = []
    step_times: list[float] = []
    clip_count = 0
    final_activations: dict[int, dict[str, tuple[float, float]]] = {}
    final_gradient_health: dict[str, float] | None = None
    final_layer_health: list[dict[str, float]] | None = None

    started = time.perf_counter()
    for step_index in range(1, steps + 1):
        hooks: list[Any] = []
        if step_index == steps:
            for layer_index, block in enumerate(model.blocks):
                def attn_hook(_module: Any, _inputs: Any, output: torch.Tensor, *, layer: int = layer_index) -> None:
                    final_activations.setdefault(layer, {})["attn"] = (
                        float(output.detach().float().pow(2).mean().sqrt().item()),
                        float(output.detach().abs().max().item()),
                    )

                def mlp_hook(_module: Any, _inputs: Any, output: torch.Tensor, *, layer: int = layer_index) -> None:
                    final_activations.setdefault(layer, {})["mlp"] = (
                        float(output.detach().float().pow(2).mean().sqrt().item()),
                        float(output.detach().abs().max().item()),
                    )

                hooks.append(block.attn.register_forward_hook(attn_hook))
                hooks.append(block.mlp.register_forward_hook(mlp_hook))

        batch = _make_batch(
            stream,
            step=step_index - 1,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        step_started = time.perf_counter()
        logits = model(batch).logits
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, spec.vocab_size),
            batch[:, 1:].reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(optimizer_config["gradient_clip_norm"])
            ).item()
        )
        clip_count += int(grad_norm > float(optimizer_config["gradient_clip_norm"]))
        grad_norms.append(grad_norm)
        if step_index == steps:
            final_gradient_health = _gradient_health(parameters)
            final_layer_health = [
                _gradient_health(list(block.parameters())) for block in model.blocks
            ]
        optimizer.step()
        step_times.append(time.perf_counter() - step_started)
        train_losses.append(float(loss.detach().item()))
        for hook in hooks:
            hook.remove()

        if step_index in checkpoint_steps:
            observed = _validation(model, validation_records, tokenizer)
            observed["last_train_ce_nats"] = train_losses[-1]
            observed["last_train_bpb"] = train_losses[-1] / math.log(2.0)
            observed["recent4_train_bpb"] = statistics.mean(train_losses[-4:]) / math.log(2.0)
            checkpoints[step_index] = observed

    optimization_wall = time.perf_counter() - started
    if final_gradient_health is None or final_layer_health is None:
        raise RuntimeError("final health capture missing")
    global_update = _update_health(parameters, initial)
    layer_updates = [
        _update_health(list(block.parameters()), block_initial[index])
        for index, block in enumerate(model.blocks)
    ]
    for index, health in enumerate(final_layer_health):
        health.update(
            {
                "layer": index,
                "update_over_initial_weight": layer_updates[index]["delta_over_initial_weight"],
                "changed_fraction": layer_updates[index]["changed_fraction"],
                "attn_activation_rms": final_activations[index]["attn"][0],
                "attn_activation_max": final_activations[index]["attn"][1],
                "mlp_activation_rms": final_activations[index]["mlp"][0],
                "mlp_activation_max": final_activations[index]["mlp"][1],
            }
        )

    breakdown = spec.parameter_breakdown()
    model_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    optimizer_bytes = _tensor_bytes(optimizer.state_dict()["state"])
    kv_elements = 2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * spec.max_seq_len
    optimized_tokens = steps * batch_size * (sequence_length - 1)
    report: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "candidate": candidate_id,
        "seed": seed,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "torch_threads": torch_threads,
            "paid_compute": False,
        },
        "controls": {
            "dataset_id": controls["dataset_id"],
            "dataset_identity_sha256": controls["dataset_identity_sha256"],
            "train_sha256": _file_sha256(train_path),
            "validation_sha256": _file_sha256(validation_path),
            "tokenizer_id": BYTE_TOKENIZER_VERSION,
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "optimizer": optimizer_config,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "optimizer_steps": steps,
            "optimized_causal_tokens": optimized_tokens,
            "init_spec": asdict(init),
            "init_identity_sha256": init.identity_sha256(),
        },
        "model": {
            "spec": spec.to_dict(),
            "model_identity_sha256": spec.identity_sha256(),
            "parameters": breakdown["total"],
            "parameter_delta_vs_incumbent": breakdown["total"] - 10_000_640,
            "parameter_allocation": breakdown,
        },
        "kv_cache": {
            "batch": 1,
            "context": spec.max_seq_len,
            "elements": kv_elements,
            "bf16_bytes": kv_elements * 2,
            "fp32_bytes": kv_elements * 4,
        },
        "metrics": {
            "checkpoints": checkpoints,
            "initial_heldout_bpb": checkpoints[0]["bpb"],
            "final_heldout_bpb": checkpoints[steps]["bpb"],
            "final_train_bpb": train_losses[-1] / math.log(2.0),
            "mean_last8_train_bpb": statistics.mean(train_losses[-8:]) / math.log(2.0),
            "preclip_grad_norm_mean": statistics.mean(grad_norms),
            "preclip_grad_norm_max": max(grad_norms),
            "clip_rate": clip_count / steps,
            "steady_median_step_s": statistics.median(step_times[4:-1] or step_times),
            "mean_step_s": statistics.mean(step_times),
            "optimized_tokens_per_s": optimized_tokens / sum(step_times),
            "optimization_wall_s": optimization_wall,
            "global_gradient_health": final_gradient_health,
            "global_update": global_update,
            "per_layer_health": final_layer_health,
            "model_tensor_bytes": model_bytes,
            "optimizer_tensor_bytes": optimizer_bytes,
            "rss_hwm_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
        "truth_boundary": config["truth_boundary"],
    }
    report["report_sha256"] = _canonical_hash(report)
    return report


def summarize_matrix(config: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    expected_ids = [str(item["id"]) for item in config["candidates"]]
    expected_seeds = [int(value) for value in config["controls"]["seeds"]]
    if len(runs) != len(expected_ids) * len(expected_seeds):
        raise ValueError("incomplete MODEL-142 matrix")
    by_candidate: dict[str, list[dict[str, Any]]] = {candidate_id: [] for candidate_id in expected_ids}
    for run in runs:
        if run["candidate"] not in by_candidate or int(run["seed"]) not in expected_seeds:
            raise ValueError("unexpected run identity")
        by_candidate[run["candidate"]].append(run)
    for candidate_runs in by_candidate.values():
        if sorted(int(run["seed"]) for run in candidate_runs) != sorted(expected_seeds):
            raise ValueError("paired-seed matrix drift")

    aggregates: dict[str, Any] = {}
    for candidate_id, candidate_runs in by_candidate.items():
        candidate_runs = sorted(candidate_runs, key=lambda run: int(run["seed"]))
        final_bpb = [float(run["metrics"]["final_heldout_bpb"]) for run in candidate_runs]
        train_bpb = [float(run["metrics"]["final_train_bpb"]) for run in candidate_runs]
        throughput = [float(run["metrics"]["optimized_tokens_per_s"]) for run in candidate_runs]
        aggregates[candidate_id] = {
            "seeds": expected_seeds,
            "final_heldout_bpb": final_bpb,
            "mean_final_heldout_bpb": statistics.mean(final_bpb),
            "sd_final_heldout_bpb": statistics.stdev(final_bpb),
            "mean_final_train_bpb": statistics.mean(train_bpb),
            "mean_optimized_tokens_per_s": statistics.mean(throughput),
            "bf16_full_context_kv_bytes": int(candidate_runs[0]["kv_cache"]["bf16_bytes"]),
            "parameters": int(candidate_runs[0]["model"]["parameters"]),
            "mean_clip_rate": statistics.mean(float(run["metrics"]["clip_rate"]) for run in candidate_runs),
            "min_final_grad_nonzero_fraction": min(
                float(run["metrics"]["global_gradient_health"]["grad_nonzero_fraction"])
                for run in candidate_runs
            ),
            "min_changed_fraction": min(
                float(run["metrics"]["global_update"]["changed_fraction"])
                for run in candidate_runs
            ),
        }

    incumbent = aggregates["incumbent_gqa_8q2kv"]
    transfer = aggregates["transfer_gqa_8q4kv"]
    mha = aggregates["mha_8q8kv"]
    transfer_vs_incumbent = [
        transfer_value - incumbent_value
        for transfer_value, incumbent_value in zip(
            transfer["final_heldout_bpb"], incumbent["final_heldout_bpb"], strict=True
        )
    ]
    transfer_vs_mha = [
        transfer_value - mha_value
        for transfer_value, mha_value in zip(
            transfer["final_heldout_bpb"], mha["final_heldout_bpb"], strict=True
        )
    ]
    health_ok = (
        transfer["min_final_grad_nonzero_fraction"] == 1.0
        and transfer["min_changed_fraction"] == 1.0
    )
    paired_win = all(delta < 0 for delta in transfer_vs_incumbent)
    decision = "TRANSFER_SUPPORTED_BOUNDED" if paired_win and health_ok else "KEEP_10M_INCUMBENT"
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "aggregates": aggregates,
        "paired_deltas_bpb": {
            "transfer_gqa_8q4kv_minus_incumbent_gqa_8q2kv": transfer_vs_incumbent,
            "transfer_gqa_8q4kv_minus_mha_8q8kv": transfer_vs_mha,
        },
        "decision": decision,
        "decision_interpretation": (
            "8Q/4KV transfers as the preferred bounded S3 research candidate; this does not freeze "
            "the 10M architecture and must be rechecked on a representative corpus before promotion."
            if decision == "TRANSFER_SUPPORTED_BOUNDED"
            else "The bounded evidence does not justify replacing the current 10M incumbent."
        ),
        "truth_boundary": config["truth_boundary"],
    }
    summary["report_sha256"] = _canonical_hash(summary)
    return summary


def validate_summary(summary: dict[str, Any]) -> None:
    if summary.get("schema") != SCHEMA or summary.get("authority") != AUTHORITY:
        raise ValueError("MODEL-142 summary identity drift")
    if summary.get("decision") not in {"TRANSFER_SUPPORTED_BOUNDED", "KEEP_10M_INCUMBENT"}:
        raise ValueError("invalid MODEL-142 decision")
    expected = summary["report_sha256"]
    unsigned = dict(summary)
    unsigned.pop("report_sha256")
    if expected != _canonical_hash(unsigned):
        raise ValueError("MODEL-142 summary hash mismatch")
