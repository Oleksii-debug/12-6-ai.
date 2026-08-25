"""Decisive small-model initialization experiment built on MODEL-34 candidates."""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training.init_stability import _activation_hooks, _global_grad_norm

SCHEMA = "12-6.init-seeds-experiment.v1"
AUTHORITY = "LOCAL_FREE_RESEARCH_EVIDENCE_NOT_CANONICAL_INITSPEC_CHANGE"
CANDIDATES = (
    "stage_default",
    "s1_width_reference_control",
    "unscaled_residual_control",
)


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item.get("id"), str) or not isinstance(item.get("text"), str):
            raise ValueError(f"{path}:{line_number} invalid record")
        records.append(item)
    if not records:
        raise ValueError(f"{path} empty")
    return records


def _byte_stream(records: list[dict[str, Any]], tokenizer: ByteTokenizer) -> bytes:
    return b"\n".join(bytes(tokenizer.encode(str(item["text"]))) for item in records) + b"\n"


def _make_batch(
    stream: bytes,
    *,
    step: int,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    width = batch_size * sequence_length
    base = (step * width) % len(stream)
    rows: list[list[int]] = []
    for batch_index in range(batch_size):
        start = (base + batch_index * sequence_length) % len(stream)
        rows.append([stream[(start + offset) % len(stream)] for offset in range(sequence_length)])
    return torch.tensor(rows, dtype=torch.long)


@torch.no_grad()
def _heldout_loss(
    model: TwelveSixDecoder,
    records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    *,
    sequence_length: int,
) -> tuple[float, int]:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for record in records:
        ids = tokenizer.encode(str(record["text"]))
        start = 0
        while start < len(ids) - 1:
            chunk = ids[start : start + sequence_length]
            if len(chunk) < 2:
                break
            batch = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
            logits = model(batch).logits
            nll = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, model.spec.vocab_size),
                batch[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_tokens += len(chunk) - 1
            start += sequence_length - 1
    model.train(was_training)
    if total_tokens <= 0:
        raise RuntimeError("held-out split produced no causal targets")
    return total_nll / total_tokens, total_tokens


def _logits_entropy(logits: torch.Tensor) -> float:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return float((-(probs * log_probs).sum(dim=-1).mean()).item())


def _candidate_init(
    base: InitSpec,
    *,
    candidate: str,
    d_model: int,
    width_reference: int,
) -> InitSpec:
    if candidate == "stage_default":
        return base
    if candidate == "unscaled_residual_control":
        return InitSpec(
            schema_version=base.schema_version,
            family=base.family,
            std=base.std,
            residual_branch_scale="none",
        )
    if candidate == "s1_width_reference_control":
        return InitSpec(
            schema_version=base.schema_version,
            family=base.family,
            std=base.std * math.sqrt(width_reference / d_model),
            residual_branch_scale=base.residual_branch_scale,
        )
    raise ValueError(f"unsupported candidate {candidate!r}")


def _model_spec(payload: dict[str, Any]) -> ModelSpec:
    spec = ModelSpec.from_dict(payload["model"])
    if spec.parameter_count() != int(payload["expected_parameters"]):
        raise ValueError(f"{payload['label']} parameter count drift")
    if spec.identity_sha256() != str(payload["expected_model_identity_sha256"]):
        raise ValueError(f"{payload['label']} ModelSpec identity drift")
    return spec


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _run_one(
    *,
    spec: ModelSpec,
    base_init: InitSpec,
    candidate: str,
    width_reference: int,
    seed: int,
    stream: bytes,
    validation_records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    training: dict[str, Any],
) -> dict[str, Any]:
    init_spec = _candidate_init(
        base_init,
        candidate=candidate,
        d_model=spec.d_model,
        width_reference=width_reference,
    )
    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["eps"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    sequence_length = int(training["sequence_length"])
    token_budget = int(training["optimized_token_budget"])
    tokens_per_step = batch_size * (sequence_length - 1)
    if token_budget % tokens_per_step:
        raise ValueError("optimized token budget must divide exactly by tokens per step")
    steps = token_budget // tokens_per_step
    clip_norm = float(training["gradient_clip_norm"])

    initial_heldout, heldout_tokens = _heldout_loss(
        model,
        validation_records,
        tokenizer,
        sequence_length=sequence_length,
    )
    activation_stats, handles = _activation_hooks(model)
    losses: list[float] = []
    grad_norms: list[float] = []
    clip_factors: list[float] = []
    step_seconds: list[float] = []
    initial_entropy: float | None = None
    initial_layer_grad_norms: dict[str, float] = {}

    model.train()
    for step in range(steps):
        batch = _make_batch(
            stream,
            step=step,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        logits = model(batch).logits
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, spec.vocab_size),
            batch[:, 1:].reshape(-1),
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("non-finite init experiment loss")
        loss.backward()
        grad_norm = _global_grad_norm(model)
        if not math.isfinite(grad_norm):
            raise RuntimeError("non-finite init experiment gradient")
        if step == 0:
            initial_entropy = _logits_entropy(logits)
            for index, block in enumerate(model.blocks):
                total = torch.zeros((), dtype=torch.float64)
                for parameter in block.parameters():
                    if parameter.grad is not None:
                        total += parameter.grad.detach().double().square().sum().cpu()
                initial_layer_grad_norms[f"blocks.{index}"] = float(total.sqrt().item())
            for handle in handles:
                handle.remove()
        factor = min(1.0, clip_norm / max(grad_norm, 1e-30))
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        step_seconds.append(time.perf_counter() - started)
        losses.append(float(loss.detach().item()))
        grad_norms.append(grad_norm)
        clip_factors.append(factor)

    if initial_entropy is None:
        raise RuntimeError("initial entropy not captured")
    final_heldout, final_heldout_tokens = _heldout_loss(
        model,
        validation_records,
        tokenizer,
        sequence_length=sequence_length,
    )
    if final_heldout_tokens != heldout_tokens:
        raise RuntimeError("heldout token count drift")
    residual_keys = sorted(
        (key for key in activation_stats if key.endswith(".residual_stream")),
        key=lambda key: int(key.split(".")[1]),
    )
    residual_rms = {
        key: float(activation_stats[key]["rms"])
        for key in residual_keys
    }
    all_activation_rms = {
        key: float(value["rms"])
        for key, value in activation_stats.items()
    }
    finite = (
        math.isfinite(initial_entropy)
        and math.isfinite(initial_heldout)
        and math.isfinite(final_heldout)
        and all(math.isfinite(value) for value in losses)
        and all(math.isfinite(value) for value in grad_norms)
        and all(math.isfinite(value) for value in all_activation_rms.values())
    )
    tail = losses[-min(8, len(losses)) :]
    return {
        "candidate": candidate,
        "seed": seed,
        "parameters": spec.parameter_count(),
        "model_identity_sha256": spec.identity_sha256(),
        "init_spec": init_spec.to_dict(),
        "init_identity_sha256": init_spec.identity_sha256(),
        "initial_logits_entropy": initial_entropy,
        "initial_heldout_loss": initial_heldout,
        "final_heldout_loss": final_heldout,
        "heldout_tokens": heldout_tokens,
        "initial_activation_rms": all_activation_rms,
        "initial_residual_rms": residual_rms,
        "initial_layer_grad_norms": initial_layer_grad_norms,
        "initial_global_grad_norm": grad_norms[0],
        "pre_clip_grad_norm_mean": _mean(grad_norms),
        "pre_clip_grad_norm_max": max(grad_norms),
        "clip_factor_min": min(clip_factors),
        "clip_fraction": sum(value < 1.0 for value in clip_factors) / len(clip_factors),
        "early_training_losses": losses[: min(8, len(losses))],
        "training_loss_tail_mean": _mean(tail),
        "mean_step_seconds": _mean(step_seconds),
        "optimized_tokens": token_budget,
        "steps": steps,
        "all_finite": finite,
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    final_losses = [float(item["final_heldout_loss"]) for item in runs]
    initial_entropy = [float(item["initial_logits_entropy"]) for item in runs]
    grad = [float(item["initial_global_grad_norm"]) for item in runs]
    clip = [float(item["clip_fraction"]) for item in runs]
    final_residual = [
        list(item["initial_residual_rms"].values())[-1]
        for item in runs
    ]
    return {
        "final_heldout_loss_mean": _mean(final_losses),
        "initial_logits_entropy_mean": _mean(initial_entropy),
        "initial_global_grad_norm_mean": _mean(grad),
        "clip_fraction_mean": _mean(clip),
        "final_block_residual_rms_mean": _mean(final_residual),
        "all_finite": all(bool(item["all_finite"]) for item in runs),
    }


def run_init_seed_experiment(
    *,
    repo_root: Path,
    source_sha: str,
    config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("exact-checkout mismatch")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unexpected MODEL-19 schema")
    if tuple(payload["candidates"]) != CANDIDATES:
        raise ValueError("MODEL-19 candidate set drift")
    training = payload["training"]
    torch.set_num_threads(int(training["torch_threads"]))
    torch.use_deterministic_algorithms(True)

    base_init = InitSpec.from_dict(payload["base_init"])
    if base_init.identity_sha256() != payload["expected_base_init_identity_sha256"]:
        raise ValueError("canonical InitSpec v1 identity drift")
    width_reference = int(payload["width_reference"])
    specs = {
        item["label"]: _model_spec(item)
        for item in payload["scales"]
    }
    tokenizer = ByteTokenizer()
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_records = _read_jsonl(train_path)
    validation_records = _read_jsonl(validation_path)
    if {item["id"] for item in train_records} & {item["id"] for item in validation_records}:
        raise RuntimeError("train/validation overlap")
    stream = _byte_stream(train_records, tokenizer)

    runs: list[dict[str, Any]] = []
    for scale_label, spec in specs.items():
        for candidate in CANDIDATES:
            for seed in (int(value) for value in training["seeds"]):
                result = _run_one(
                    spec=spec,
                    base_init=base_init,
                    candidate=candidate,
                    width_reference=width_reference,
                    seed=seed,
                    stream=stream,
                    validation_records=validation_records,
                    tokenizer=tokenizer,
                    training=training,
                )
                result["scale"] = scale_label
                runs.append(result)

    aggregates: dict[str, dict[str, Any]] = {}
    for scale_label in specs:
        aggregates[scale_label] = {}
        for candidate in CANDIDATES:
            subset = [
                item for item in runs
                if item["scale"] == scale_label and item["candidate"] == candidate
            ]
            aggregates[scale_label][candidate] = _aggregate(subset)

    gates = payload["rejection_gates"]
    classifications: dict[str, dict[str, Any]] = {}
    for scale_label in specs:
        default = aggregates[scale_label]["stage_default"]
        classifications[scale_label] = {}
        for candidate in CANDIDATES:
            agg = aggregates[scale_label][candidate]
            residual_ratio = (
                float(agg["final_block_residual_rms_mean"])
                / float(default["final_block_residual_rms_mean"])
            )
            grad_ratio = (
                float(agg["initial_global_grad_norm_mean"])
                / max(float(default["initial_global_grad_norm_mean"]), 1e-30)
            )
            reasons: list[str] = []
            if not bool(agg["all_finite"]):
                reasons.append("non_finite")
            if residual_ratio > float(gates["max_final_residual_rms_ratio_to_default"]):
                reasons.append("residual_rms_growth")
            if (
                grad_ratio > float(gates["max_initial_grad_ratio_to_default"])
                and float(agg["clip_fraction_mean"]) > float(gates["clip_fraction_threshold"])
            ):
                reasons.append("gradient_clip_instability")
            classifications[scale_label][candidate] = {
                "status": "REJECT_UNSTABLE" if reasons else "STABLE",
                "reasons": reasons,
                "final_residual_rms_ratio_to_default": residual_ratio,
                "initial_grad_ratio_to_default": grad_ratio,
            }

    width_500 = aggregates["research_500k"]["s1_width_reference_control"]
    default_500 = aggregates["research_500k"]["stage_default"]
    width_improvement = (
        float(default_500["final_heldout_loss_mean"])
        - float(width_500["final_heldout_loss_mean"])
    ) / float(default_500["final_heldout_loss_mean"])
    width_stable = all(
        classifications[scale]["s1_width_reference_control"]["status"] == "STABLE"
        for scale in specs
    )
    decisive_threshold = float(payload["decision"]["min_relative_heldout_improvement_for_v2"])
    decision = (
        "PROPOSE_INITSPEC_V2_EXPERIMENT"
        if width_stable and width_improvement >= decisive_threshold
        else "KEEP_INITSPEC_V1"
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "config_sha256": _sha256_file(config_path),
        "dataset_manifest_sha256": _sha256_file(manifest_path),
        "tokenizer_identity": asdict(tokenizer.identity),
        "base_init_identity_sha256": base_init.identity_sha256(),
        "runs": runs,
        "aggregates": aggregates,
        "classifications": classifications,
        "recommendation": {
            "decision": decision,
            "width_aware_relative_heldout_improvement_at_500k": width_improvement,
            "minimum_decisive_improvement": decisive_threshold,
            "canonical_initspec_changed": False,
            "note": (
                "A v2 is proposed only when the single width-aware alternative is numerically "
                "stable and clears the predeclared held-out improvement threshold. Otherwise "
                "the incumbent Normal(0,0.02)+sqrt(2L) residual scaling remains primary."
            ),
        },
        "truth_boundary": {
            "paid_compute_used": False,
            "canonical_initspec_changed": False,
            "canonical_stage_configs_changed": False,
            "tiny_controlled_fixture": True,
        },
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
