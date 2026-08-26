"""TRAIN-196: controlled effective loss-token batch scaling at the accepted ~10M recipe.

The experiment changes only gradient accumulation for the statistical batch comparison.
A separate same-effective-batch microbatch-shape control measures CPU implementation
throughput without using that timing result as evidence for the statistical batch choice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

from twelve_six import milestone100_first_learned as m100
from twelve_six.checkpoint import hash_json, sha256_file
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.packing import PACKING_VERSION, TextRecord, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

SCHEMA = "12-6.train196-10m-batch-scaling.v1"
WORKER_SCHEMA = "12-6.train196-10m-batch-worker.v1"
PREREG_SCHEMA = "12-6.train196-10m-batch-preregistration.v1"
AUTHORITY = "LOCAL_FREE_CPU_SPECIFIC_BATCH_SCALING_NOT_GPU_THROUGHPUT_AUTHORITY"
REPOSITORY = "Oleksii-debug/12-6-ai."
STAGE_CONFIG = Path("configs/stages/alternatives/s3_10m_scale03_byte_gqa.execution.json")
EXPECTED_PARAMETERS = 10_000_640
EXPECTED_MODEL_SHA256 = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
EXPECTED_INIT_SHA256 = "86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
EXPECTED_CORPUS_SHA256 = m100.EXPECTED_CORPUS_ID

SEQUENCE_LENGTH = 64
BASE_MICROBATCH_EXAMPLES = 4
BASE_MICROBATCH_LOSS_TOKENS = BASE_MICROBATCH_EXAMPLES * (SEQUENCE_LENGTH - 1)  # 252
TOTAL_BASE_MICROBATCHES = 256
TOTAL_OPTIMIZED_LOSS_TOKENS = BASE_MICROBATCH_LOSS_TOKENS * TOTAL_BASE_MICROBATCHES  # 64,512
VALIDATION_FULL_EXAMPLES_PER_STRATUM = 32
NOISE_MICROBATCHES = 8
DEFAULT_SEEDS = (1515, 1616, 1717)
TORCH_THREADS = 2

LEARNING_RATE = 3e-4
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.0
GRADIENT_CLIP_NORM = 1.0
WARMUP_STEPS = 0
SCHEDULER = "constant"
PRECISION = "fp32"

# The first three entries are the statistical candidates.  The fourth is a
# same-effective-batch implementation control and never competes for selection.
CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "label": "batch-504",
        "microbatch_examples": 4,
        "accumulation": 2,
        "effective_loss_tokens": 504,
        "role": "statistical",
    },
    {
        "label": "batch-1008",
        "microbatch_examples": 4,
        "accumulation": 4,
        "effective_loss_tokens": 1008,
        "role": "statistical",
    },
    {
        "label": "batch-2016",
        "microbatch_examples": 4,
        "accumulation": 8,
        "effective_loss_tokens": 2016,
        "role": "statistical",
    },
    {
        "label": "hardware-1008-micro8",
        "microbatch_examples": 8,
        "accumulation": 2,
        "effective_loss_tokens": 1008,
        "role": "hardware_control",
    },
)


class Train196Error(RuntimeError):
    """Raised when the controlled TRAIN-196 contract cannot be preserved."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Train196Error(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Train196Error(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _require_head(repo: Path, source_sha: str) -> None:
    _require(len(source_sha) == 40 and all(c in "0123456789abcdef" for c in source_sha), "source SHA must be full lowercase 40-hex")
    _require(_git_head(repo) == source_sha, "exact source checkout mismatch")


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tensor_digest_update(digest: "hashlib._Hash", tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes())


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if platform.system() == "Linux" else value


def _tensor_tree_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_tree_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_bytes(item) for item in value)
    return 0


def _load_model_truth(repo: Path) -> tuple[ModelSpec, InitSpec, dict[str, Any]]:
    raw = _read_json(repo / STAGE_CONFIG)
    spec = ModelSpec.from_dict(dict(raw["model"]))
    init = InitSpec(**raw["init"])
    _require(spec.parameter_count() == EXPECTED_PARAMETERS, "accepted 10M parameter count drift")
    _require(spec.identity_sha256() == EXPECTED_MODEL_SHA256, "accepted 10M ModelSpec identity drift")
    _require(init.identity_sha256() == EXPECTED_INIT_SHA256, "accepted InitSpec identity drift")
    _require(raw.get("canonical_base") == "random_init", "accepted 10M base must remain random_init")
    return spec, init, {
        "path": STAGE_CONFIG.as_posix(),
        "sha256": sha256_file(repo / STAGE_CONFIG),
        "model_sha256": spec.identity_sha256(),
        "init_sha256": init.identity_sha256(),
        "parameters": spec.parameter_count(),
    }


def _full_examples(
    corpus: Path,
    manifest: dict[str, Any],
    tokenizer: ByteTokenizer,
    split: str,
    stratum: str,
) -> Iterator[Any]:
    records = (
        TextRecord(str(row["record_id"]), str(row["text"]), str(row["split"]))
        for row in m100._rows(corpus, manifest, split, stratum)
    )
    for example in iter_packed_examples(
        records,
        tokenizer,
        expected_split=split,
        sequence_length=SEQUENCE_LENGTH,
        cross_document=False,
    ):
        valid = sum(1 for item in example.labels[1:] if item != -100)
        if valid == SEQUENCE_LENGTH - 1:
            yield example


def _batch_from_examples(examples: Sequence[Any]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([item.input_ids for item in examples], dtype=torch.long),
        "labels": torch.tensor([item.labels for item in examples], dtype=torch.long),
    }


def _batch_loss_tokens(batch: Mapping[str, torch.Tensor]) -> int:
    return int(batch["labels"][:, 1:].ne(-100).sum().item())


def _materialize_base_trace(
    corpus: Path,
    manifest: dict[str, Any],
    tokenizer: ByteTokenizer,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, Any]]:
    iterators = {
        stratum: _full_examples(corpus, manifest, tokenizer, "train", stratum)
        for stratum in ("uk", "en", "code")
    }
    trace: list[dict[str, torch.Tensor]] = []
    strata: list[str] = []
    digest = hashlib.sha256()
    for index in range(TOTAL_BASE_MICROBATCHES):
        stratum = m100.MIXTURE[index % len(m100.MIXTURE)]
        examples = []
        for _ in range(BASE_MICROBATCH_EXAMPLES):
            try:
                examples.append(next(iterators[stratum]))
            except StopIteration as exc:
                raise Train196Error(f"{stratum} full-window training trace exhausted") from exc
        batch = _batch_from_examples(examples)
        _require(_batch_loss_tokens(batch) == BASE_MICROBATCH_LOSS_TOKENS, "base microbatch loss-token count drift")
        trace.append(batch)
        strata.append(stratum)
        digest.update(index.to_bytes(4, "little"))
        digest.update(stratum.encode("ascii"))
        _tensor_digest_update(digest, batch["input_ids"])
        _tensor_digest_update(digest, batch["labels"])
    total = sum(_batch_loss_tokens(batch) for batch in trace)
    _require(total == TOTAL_OPTIMIZED_LOSS_TOKENS, "optimized loss-token budget drift")
    return trace, {
        "sha256": digest.hexdigest(),
        "microbatches": len(trace),
        "microbatch_examples": BASE_MICROBATCH_EXAMPLES,
        "sequence_length": SEQUENCE_LENGTH,
        "loss_tokens_per_microbatch": BASE_MICROBATCH_LOSS_TOKENS,
        "optimized_loss_tokens": total,
        "strata_sequence_sha256": hashlib.sha256("\n".join(strata).encode("ascii")).hexdigest(),
        "mixture_pattern": list(m100.MIXTURE),
    }


def _materialize_validation(
    corpus: Path,
    manifest: dict[str, Any],
    tokenizer: ByteTokenizer,
) -> tuple[dict[str, list[dict[str, torch.Tensor]]], dict[str, Any]]:
    result: dict[str, list[dict[str, torch.Tensor]]] = {}
    digest = hashlib.sha256()
    tokens = 0
    for stratum in ("uk", "en", "code"):
        iterator = _full_examples(corpus, manifest, tokenizer, "validation", stratum)
        examples = []
        for _ in range(VALIDATION_FULL_EXAMPLES_PER_STRATUM):
            try:
                examples.append(next(iterator))
            except StopIteration as exc:
                raise Train196Error(f"{stratum} validation full-window trace exhausted") from exc
        batches = []
        for offset in range(0, len(examples), 16):
            batch = _batch_from_examples(examples[offset : offset + 16])
            batches.append(batch)
            tokens += _batch_loss_tokens(batch)
            digest.update(stratum.encode("ascii"))
            _tensor_digest_update(digest, batch["input_ids"])
            _tensor_digest_update(digest, batch["labels"])
        result[stratum] = batches
    return result, {
        "sha256": digest.hexdigest(),
        "split": "validation",
        "full_examples_per_stratum": VALIDATION_FULL_EXAMPLES_PER_STRATUM,
        "scoreable_loss_tokens": tokens,
        "sequence_length": SEQUENCE_LENGTH,
    }


def _group_base_trace(
    base_trace: Sequence[Mapping[str, torch.Tensor]],
    microbatch_examples: int,
) -> list[dict[str, torch.Tensor]]:
    _require(microbatch_examples in (4, 8), "unsupported microbatch implementation")
    if microbatch_examples == 4:
        return [
            {"input_ids": batch["input_ids"], "labels": batch["labels"]}
            for batch in base_trace
        ]
    grouped: list[dict[str, torch.Tensor]] = []
    for offset in range(0, len(base_trace), 2):
        pair = base_trace[offset : offset + 2]
        _require(len(pair) == 2, "hardware-control trace ended on partial grouping")
        grouped.append(
            {
                "input_ids": torch.cat([item["input_ids"] for item in pair], dim=0),
                "labels": torch.cat([item["labels"] for item in pair], dim=0),
            }
        )
    return grouped


def _evaluation(
    model: TwelveSixDecoder,
    validation: Mapping[str, Sequence[Mapping[str, torch.Tensor]]],
) -> dict[str, Any]:
    before = _state_sha256(model)
    training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    by_stratum: dict[str, Any] = {}
    try:
        with torch.no_grad():
            for stratum in ("uk", "en", "code"):
                stratum_nll = 0.0
                stratum_tokens = 0
                for batch in validation[stratum]:
                    logits = model(batch["input_ids"]).logits[:, :-1, :].contiguous()
                    targets = batch["labels"][:, 1:].contiguous()
                    token_count = int(targets.ne(-100).sum().item())
                    nll = F.cross_entropy(
                        logits.reshape(-1, model.spec.vocab_size),
                        targets.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    stratum_nll += float(nll.item())
                    stratum_tokens += token_count
                _require(stratum_tokens > 0, f"no held-out tokens for {stratum}")
                by_stratum[stratum] = {
                    "bpb": stratum_nll / math.log(2.0) / stratum_tokens,
                    "loss_nats": stratum_nll / stratum_tokens,
                    "tokens": stratum_tokens,
                }
                total_nll += stratum_nll
                total_tokens += stratum_tokens
    finally:
        model.train(training)
    after = _state_sha256(model)
    _require(before == after, "held-out evaluation mutated model state")
    return {
        "bpb": total_nll / math.log(2.0) / total_tokens,
        "loss_nats": total_nll / total_tokens,
        "tokens": total_tokens,
        "by_stratum": by_stratum,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "non_mutating": True,
    }


def _gradient_noise_proxy(
    model: TwelveSixDecoder,
    base_trace: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    samples = list(base_trace[:NOISE_MICROBATCHES])
    _require(len(samples) == NOISE_MICROBATCHES, "insufficient gradient-noise samples")
    before = _state_sha256(model)
    training = model.training
    rng = torch.random.get_rng_state()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    sums = [torch.zeros_like(parameter, dtype=torch.float32) for parameter in params]
    second_moment = 0.0
    token_counts: list[int] = []
    losses: list[float] = []
    try:
        model.eval()
        for batch in samples:
            model.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"]).logits[:, :-1, :].contiguous()
            targets = batch["labels"][:, 1:].contiguous()
            tokens = int(targets.ne(-100).sum().item())
            loss = F.cross_entropy(
                logits.reshape(-1, model.spec.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )
            _require(bool(torch.isfinite(loss).item()), "non-finite gradient-probe loss")
            loss.backward()
            norm_sq = 0.0
            for index, parameter in enumerate(params):
                if parameter.grad is None:
                    continue
                grad = parameter.grad.detach().float()
                _require(bool(torch.isfinite(grad).all().item()), "non-finite gradient-probe gradient")
                sums[index].add_(grad)
                norm_sq += float(grad.double().square().sum().item())
            second_moment += norm_sq
            token_counts.append(tokens)
            losses.append(float(loss.item()))
        n = len(samples)
        second_moment /= n
        signal = 0.0
        for value in sums:
            mean = value.double().div(float(n))
            signal += float(mean.square().sum().item())
        covariance = max(0.0, (second_moment - signal) * n / max(n - 1, 1))
        ratio = covariance / max(signal, 1e-30)
        mean_tokens = statistics.fmean(token_counts)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(training)
        torch.random.set_rng_state(rng)
    after = _state_sha256(model)
    _require(before == after, "gradient-noise probe mutated model weights")
    return {
        "samples": len(samples),
        "mean_microbatch_loss_tokens": mean_tokens,
        "mean_loss": statistics.fmean(losses),
        "mean_gradient_signal_squared": signal,
        "gradient_second_moment": second_moment,
        "unbiased_covariance_trace": covariance,
        "trace_cov_over_mean_gradient_squared": ratio,
        "loss_token_noise_proxy": ratio * mean_tokens,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "non_mutating": True,
    }


def _snapshot_parameters(model: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]


def _relative_update_l2(model: torch.nn.Module, before: Sequence[torch.Tensor]) -> float:
    denominator = 0.0
    numerator = 0.0
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    _require(len(parameters) == len(before), "parameter snapshot cardinality drift")
    for old, current in zip(before, parameters, strict=True):
        old64 = old.double()
        delta64 = current.detach().double() - old64
        denominator += float(old64.square().sum().item())
        numerator += float(delta64.square().sum().item())
    return math.sqrt(numerator) / max(math.sqrt(denominator), 1e-30)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    _require(bool(values), "distribution requires values")
    ordered = sorted(float(value) for value in values)
    def q(fraction: float) -> float:
        index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
        return ordered[index]
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _trainer_config(*, seed: int, accumulation: int, optimizer_steps: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        eps=EPS,
        max_steps=optimizer_steps,
        warmup_steps=WARMUP_STEPS,
        scheduler=SCHEDULER,
        gradient_accumulation_steps=accumulation,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
        precision=PRECISION,
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


def _worker(
    *,
    repo: Path,
    source_sha: str,
    output_dir: Path,
    seed: int,
    candidate: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    _require_head(repo, source_sha)
    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)
    manifest = _read_json(output_dir / "corpus-manifest.json")
    _require(manifest["corpus_identity_sha256"] == EXPECTED_CORPUS_SHA256, "corpus identity drift")
    _require(manifest["train_validation_content_overlap"] == 0, "train/validation leakage")
    corpus = output_dir / "corpus-a"
    tokenizer = ByteTokenizer()
    _require(tokenizer.vocab_size == 256 and not tokenizer.identity.special_tokens, "byte-tokenizer identity drift")
    spec, init, model_truth = _load_model_truth(repo)
    base_trace, trace_identity = _materialize_base_trace(corpus, manifest, tokenizer)
    validation, validation_identity = _materialize_validation(corpus, manifest, tokenizer)

    microbatch_examples = int(candidate["microbatch_examples"])
    accumulation = int(candidate["accumulation"])
    trace = _group_base_trace(base_trace, microbatch_examples)
    _require(len(trace) % accumulation == 0, "trace does not end on accumulation boundary")
    optimizer_steps = len(trace) // accumulation
    expected_effective = microbatch_examples * (SEQUENCE_LENGTH - 1) * accumulation
    _require(expected_effective == int(candidate["effective_loss_tokens"]), "candidate effective loss-token identity drift")

    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init)
    _require(sum(parameter.numel() for parameter in model.parameters()) == EXPECTED_PARAMETERS, "runtime parameter count drift")
    init_state_sha256 = _state_sha256(model)
    initial_eval = _evaluation(model, validation)
    initial_noise = _gradient_noise_proxy(model, base_trace)

    config = _trainer_config(seed=seed, accumulation=accumulation, optimizer_steps=optimizer_steps)
    trainer = Trainer(model, config, device="cpu")
    update_ratios: list[float] = []
    grad_norms: list[float] = []
    effective_update_tokens: list[int] = []
    clip_count = 0
    train_call_seconds = 0.0
    pending_update_tokens = 0
    before_update: list[torch.Tensor] | None = None

    for micro_index, batch in enumerate(trace):
        if micro_index % accumulation == 0:
            before_update = _snapshot_parameters(model)
            pending_update_tokens = 0
        pending_update_tokens += _batch_loss_tokens(batch)
        started = time.perf_counter()
        metrics = trainer.train_microbatch(batch)
        train_call_seconds += time.perf_counter() - started
        if not metrics.optimizer_stepped:
            continue
        _require(before_update is not None, "missing pre-update parameter snapshot")
        _require(metrics.grad_norm is not None, "optimizer step missing pre-clip grad norm")
        ratio = _relative_update_l2(model, before_update)
        update_ratios.append(ratio)
        grad_norm = float(metrics.grad_norm)
        grad_norms.append(grad_norm)
        effective_update_tokens.append(pending_update_tokens)
        if grad_norm > GRADIENT_CLIP_NORM:
            clip_count += 1
        before_update = None

    trainer.assert_checkpoint_safe()
    _require(trainer.optimizer_step == optimizer_steps, "optimizer update count drift")
    _require(trainer.tokens_seen == TOTAL_OPTIMIZED_LOSS_TOKENS, "optimized loss-token budget drift")
    _require(all(value == expected_effective for value in effective_update_tokens), "effective update-token count drift")
    final_eval = _evaluation(model, validation)
    final_noise = _gradient_noise_proxy(model, base_trace)
    final_state_sha256 = _state_sha256(model)
    optimizer_state_bytes = _tensor_tree_bytes(trainer.state_dict().optimizer)
    model_parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())

    report = {
        "schema": WORKER_SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "seed": seed,
        "candidate": dict(candidate),
        "model": model_truth,
        "data": {
            "corpus_identity_sha256": manifest["corpus_identity_sha256"],
            "packing_version": PACKING_VERSION,
            "cross_document": False,
            "base_trace": trace_identity,
            "validation_trace": validation_identity,
            "full_windows_only": True,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": LEARNING_RATE,
            "betas": list(BETAS),
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRADIENT_CLIP_NORM,
            "warmup_steps": WARMUP_STEPS,
            "scheduler": SCHEDULER,
            "precision": PRECISION,
            "gradient_accumulation_steps": accumulation,
        },
        "initial_state_sha256": init_state_sha256,
        "final_state_sha256": final_state_sha256,
        "initial_heldout": initial_eval,
        "final_heldout": final_eval,
        "gradient_noise": {"initial": initial_noise, "final": final_noise},
        "training": {
            "optimized_loss_tokens": trainer.tokens_seen,
            "optimizer_updates": trainer.optimizer_step,
            "microbatches": len(trace),
            "loss_tokens_per_update": expected_effective,
            "train_call_wall_seconds": train_call_seconds,
            "throughput_loss_tokens_per_second": trainer.tokens_seen / train_call_seconds,
            "clip_count": clip_count,
            "clip_rate": clip_count / trainer.optimizer_step,
            "preclip_gradient_norm": _distribution(grad_norms),
            "global_update_to_weight_l2": _distribution(update_ratios),
            "model_parameter_bytes": model_parameter_bytes,
            "optimizer_state_bytes": optimizer_state_bytes,
            "peak_rss_bytes": _peak_rss_bytes(),
        },
        "machine": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "device": "cpu",
            "github_actions": os.environ.get("GITHUB_ACTIONS") == "true",
            "paid_compute": False,
        },
    }
    report["report_sha256"] = hash_json(report)
    _write_json(output, report)
    return report


def _bootstrap_ci(values: Sequence[float], *, seed: int = 196, replicates: int = 10_000) -> dict[str, float]:
    _require(bool(values), "bootstrap requires paired values")
    rng = random.Random(seed)
    observed = statistics.fmean(values)
    draws: list[float] = []
    n = len(values)
    for _ in range(replicates):
        draws.append(statistics.fmean(values[rng.randrange(n)] for _ in range(n)))
    draws.sort()
    lo = draws[max(0, math.floor(0.025 * (replicates - 1)))]
    hi = draws[min(replicates - 1, math.ceil(0.975 * (replicates - 1)))]
    return {"mean": observed, "ci95_low": lo, "ci95_high": hi, "replicates": replicates}


def _aggregate_candidate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bpbs = [float(item["final_heldout"]["bpb"]) for item in records]
    throughput = [float(item["training"]["throughput_loss_tokens_per_second"]) for item in records]
    wall = [float(item["training"]["train_call_wall_seconds"]) for item in records]
    clip = [float(item["training"]["clip_rate"]) for item in records]
    update = [float(item["training"]["global_update_to_weight_l2"]["median"]) for item in records]
    memory = [int(item["training"]["peak_rss_bytes"]) for item in records]
    noise = [float(item["gradient_noise"]["final"]["loss_token_noise_proxy"]) for item in records]
    return {
        "seeds": [int(item["seed"]) for item in records],
        "final_bpb": {"values": bpbs, "median": statistics.median(bpbs), "mean": statistics.fmean(bpbs)},
        "clip_rate": {"values": clip, "median": statistics.median(clip)},
        "update_ratio_median": {"values": update, "median": statistics.median(update)},
        "final_gradient_noise_loss_token_proxy": {"values": noise, "median": statistics.median(noise)},
        "train_wall_seconds": {"values": wall, "median": statistics.median(wall)},
        "throughput_loss_tokens_per_second": {"values": throughput, "median": statistics.median(throughput)},
        "peak_rss_bytes": {"values": memory, "median": statistics.median(memory)},
        "optimizer_updates": int(records[0]["training"]["optimizer_updates"]),
        "optimized_loss_tokens": int(records[0]["training"]["optimized_loss_tokens"]),
    }


def _selection(
    statistical: Mapping[str, Sequence[Mapping[str, Any]]],
    aggregates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    labels = list(statistical)
    best = min(labels, key=lambda label: float(aggregates[label]["final_bpb"]["median"]))
    best_median = float(aggregates[best]["final_bpb"]["median"])
    paired: dict[str, Any] = {}
    eligible: list[str] = []
    for label in labels:
        by_seed = {int(item["seed"]): float(item["final_heldout"]["bpb"]) for item in statistical[label]}
        best_by_seed = {int(item["seed"]): float(item["final_heldout"]["bpb"]) for item in statistical[best]}
        _require(set(by_seed) == set(best_by_seed), "paired seed set drift")
        differences = [by_seed[seed] - best_by_seed[seed] for seed in sorted(by_seed)]
        relative = [difference / best_by_seed[seed] for difference, seed in zip(differences, sorted(by_seed), strict=True)]
        paired[label] = {
            "bpb_minus_best_by_seed": differences,
            "relative_minus_best_by_seed": relative,
            "paired_bootstrap_mean_bpb_difference": _bootstrap_ci(differences, seed=196 + sum(ord(c) for c in label)),
        }
        median_relative = (float(aggregates[label]["final_bpb"]["median"]) - best_median) / best_median
        if median_relative <= 0.005 and max(relative) <= 0.01:
            eligible.append(label)
    effective = {item["label"]: int(item["effective_loss_tokens"]) for item in CANDIDATES if item["role"] == "statistical"}
    selected = min(eligible, key=lambda label: effective[label]) if eligible else best
    largest = max(labels, key=lambda label: effective[label])
    grid_edge = selected == largest and best == largest
    return {
        "primary_metric": "paired final held-out BPB",
        "practical_tie_rule": "within 0.5% of best median BPB and no paired seed worse than best by >1%; choose smaller effective batch among ties",
        "best_raw_median_bpb": best,
        "quality_equivalent_candidates": sorted(eligible, key=lambda label: effective[label]),
        "selected": selected,
        "selected_effective_loss_tokens": effective[selected],
        "grid_edge": grid_edge,
        "status": "PROVISIONAL_GRID_EDGE_REQUIRES_LARGER_BATCH_CONTROL" if grid_edge else "SELECTED_WITHIN_TESTED_GRID",
        "paired_comparisons": paired,
    }


def _hardware_comparison(
    statistical_1008: Sequence[Mapping[str, Any]],
    hardware: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left = {int(item["seed"]): item for item in statistical_1008}
    right = {int(item["seed"]): item for item in hardware}
    _require(set(left) == set(right), "hardware-control paired seed set drift")
    bpb_diffs = []
    speedups = []
    for seed in sorted(left):
        bpb_diffs.append(float(right[seed]["final_heldout"]["bpb"]) - float(left[seed]["final_heldout"]["bpb"]))
        speedups.append(
            float(right[seed]["training"]["throughput_loss_tokens_per_second"])
            / float(left[seed]["training"]["throughput_loss_tokens_per_second"])
        )
    return {
        "effective_loss_tokens": 1008,
        "reference": "4x64 examples with accumulation 4",
        "challenger": "8x64 examples with accumulation 2",
        "paired_final_bpb_difference_challenger_minus_reference": bpb_diffs,
        "max_abs_final_bpb_difference": max(abs(value) for value in bpb_diffs),
        "paired_throughput_ratio_challenger_over_reference": speedups,
        "median_throughput_ratio": statistics.median(speedups),
        "quality_equivalence_tolerance_bpb": 1e-4,
        "quality_equivalent": max(abs(value) for value in bpb_diffs) <= 1e-4,
        "authority": "CPU_SPECIFIC_MICROBATCH_IMPLEMENTATION_ONLY",
    }


def _candidate_by_label(label: str) -> dict[str, Any]:
    for item in CANDIDATES:
        if item["label"] == label:
            return dict(item)
    raise Train196Error(f"unknown candidate: {label}")


def _preregistration(source_sha: str, model_truth: Mapping[str, Any], corpus_manifest: Mapping[str, Any], seeds: Sequence[int]) -> dict[str, Any]:
    value = {
        "schema": PREREG_SCHEMA,
        "source_sha": source_sha,
        "authority": AUTHORITY,
        "fixed_controls": {
            "model": dict(model_truth),
            "corpus_identity_sha256": corpus_manifest["corpus_identity_sha256"],
            "tokenizer": "s0-byte-v1 / vocab 256 / no special tokens",
            "sequence_length": SEQUENCE_LENGTH,
            "optimizer": {
                "name": "AdamW",
                "learning_rate": LEARNING_RATE,
                "betas": list(BETAS),
                "eps": EPS,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "warmup_steps": WARMUP_STEPS,
                "scheduler": SCHEDULER,
                "precision": PRECISION,
            },
            "total_optimized_loss_tokens": TOTAL_OPTIMIZED_LOSS_TOKENS,
            "base_microbatch": f"{BASE_MICROBATCH_EXAMPLES}x{SEQUENCE_LENGTH} -> {BASE_MICROBATCH_LOSS_TOKENS} causal loss tokens",
        },
        "paired_seeds": list(seeds),
        "statistical_candidates": [dict(item) for item in CANDIDATES if item["role"] == "statistical"],
        "microbatch_hardware_control": [dict(item) for item in CANDIDATES if item["role"] == "hardware_control"],
        "measures": [
            "held-out BPB overall and uk/en/code",
            "gradient noise proxy",
            "pre-clip gradient norm and clip rate",
            "global update/weight L2 ratio",
            "optimizer updates",
            "training-call wall time and valid loss-token throughput",
            "peak RSS and optimizer/model memory",
        ],
        "decision_rule": "paired final held-out BPB first; candidates within 0.5% of best median with no paired seed >1% worse are practical ties; choose the smaller batch among ties; largest-grid winner remains provisional",
        "hardware_boundary": "microbatch shape is evaluated only at fixed 1008 effective loss tokens; CPU throughput is never used to infer GPU throughput",
        "gpu_retest_preregistration": {
            "required": True,
            "trigger": "before using TRAIN-196 timing/memory to choose a GPU microbatch implementation",
            "preserve": "selected effective loss-token batch, model, corpus trace, LR/betas/eps/WD/clipping/schedule, total optimized tokens and paired seeds",
            "compare": "at least two GPU-feasible microbatch/accumulation decompositions with identical grouped examples",
            "precision": "use the accepted GPU precision only after universal-bootstrap CUDA capability and hardware preflight; do not silently downgrade",
        },
        "paid_compute": False,
    }
    value["identity_sha256"] = hash_json(value)
    return value


def run(
    *,
    repo: Path,
    source_sha: str,
    output_dir: Path,
    seeds: Sequence[int],
) -> dict[str, Any]:
    _require_head(repo, source_sha)
    _require(len(seeds) >= 3 and len(set(seeds)) == len(seeds), "TRAIN-196 requires at least three unique paired seeds")
    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_spec, init_spec, model_truth = _load_model_truth(repo)
    _require(model_spec.parameter_count() == EXPECTED_PARAMETERS and init_spec.identity_sha256() == EXPECTED_INIT_SHA256, "model truth drift")
    corpus_manifest = m100._build_corpus(repo, output_dir)
    _require(corpus_manifest["corpus_identity_sha256"] == EXPECTED_CORPUS_SHA256, "DATA-25 corpus identity drift")
    _require(corpus_manifest["train_validation_content_overlap"] == 0, "DATA-25 leakage")
    tokenizer = ByteTokenizer()
    base_trace, base_trace_identity = _materialize_base_trace(output_dir / "corpus-a", corpus_manifest, tokenizer)
    _, validation_identity = _materialize_validation(output_dir / "corpus-a", corpus_manifest, tokenizer)
    prereg = _preregistration(source_sha, model_truth, corpus_manifest, seeds)
    prereg["base_trace_identity"] = base_trace_identity
    prereg["validation_trace_identity"] = validation_identity
    prereg["identity_sha256"] = hash_json({key: value for key, value in prereg.items() if key != "identity_sha256"})
    _write_json(output_dir / "train196-preregistration.json", prereg)

    workers = output_dir / "workers"
    workers.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for seed in seeds:
            worker_path = workers / f"{candidate['label']}-seed{seed}.json"
            command = [
                sys.executable,
                "-m",
                "twelve_six.train196_batch_scaling",
                "worker",
                "--repo-root",
                str(repo),
                "--source-sha",
                source_sha,
                "--output-dir",
                str(output_dir),
                "--candidate",
                str(candidate["label"]),
                "--seed",
                str(seed),
                "--output",
                str(worker_path),
            ]
            subprocess.run(command, cwd=repo, check=True)
            records.append(_read_json(worker_path))

    # Pairing invariants: all candidates for a seed must begin from byte-identical
    # weights and consume the same underlying 4x64 example trace.
    for seed in seeds:
        paired = [item for item in records if int(item["seed"]) == seed]
        _require(len({item["initial_state_sha256"] for item in paired}) == 1, f"seed {seed}: initial weights drift across candidates")
        _require(len({item["data"]["base_trace"]["sha256"] for item in paired}) == 1, f"seed {seed}: data trace drift across candidates")
        _require(len({item["data"]["validation_trace"]["sha256"] for item in paired}) == 1, f"seed {seed}: validation trace drift across candidates")
        _require(all(int(item["training"]["optimized_loss_tokens"]) == TOTAL_OPTIMIZED_LOSS_TOKENS for item in paired), f"seed {seed}: optimized-token budget drift")

    statistical: dict[str, list[dict[str, Any]]] = {}
    hardware: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        subset = [item for item in records if item["candidate"]["label"] == candidate["label"]]
        _require(len(subset) == len(seeds), f"missing worker records for {candidate['label']}")
        if candidate["role"] == "statistical":
            statistical[candidate["label"]] = subset
        else:
            hardware = subset
    aggregates = {label: _aggregate_candidate(items) for label, items in statistical.items()}
    selection = _selection(statistical, aggregates)
    hardware_result = _hardware_comparison(statistical["batch-1008"], hardware)

    report = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source_sha": source_sha,
        "repository": REPOSITORY,
        "preregistration_identity_sha256": prereg["identity_sha256"],
        "model": model_truth,
        "data": {
            "corpus_identity_sha256": corpus_manifest["corpus_identity_sha256"],
            "base_trace": base_trace_identity,
            "validation_trace": validation_identity,
            "total_optimized_loss_tokens": TOTAL_OPTIMIZED_LOSS_TOKENS,
        },
        "fixed_optimizer": prereg["fixed_controls"]["optimizer"],
        "paired_seeds": list(seeds),
        "statistical_candidates": aggregates,
        "selection": selection,
        "microbatch_hardware_efficiency": hardware_result,
        "machine_scope": {
            "device": "cpu",
            "cuda_available": False,
            "torch_threads": TORCH_THREADS,
            "cpu_specific_timing_and_memory": True,
            "paid_compute": False,
        },
        "gpu_retest_preregistration": prereg["gpu_retest_preregistration"],
        "truth_boundary": {
            "batch_choice_authority": "held-out quality under the frozen CPU fp32 recipe and exact diagnostic data trace",
            "cpu_throughput_authority": "this CPU implementation only",
            "gpu_throughput_authority": False,
            "broad_corpus_stage_promotion": False,
            "optimizer_retuning": False,
            "model_retuning": False,
            "largest_grid_winner_requires_followup": selection["grid_edge"],
        },
        "worker_reports": [
            {
                "candidate": item["candidate"]["label"],
                "seed": item["seed"],
                "report_sha256": item["report_sha256"],
            }
            for item in records
        ],
    }
    report["report_sha256"] = hash_json(report)
    _write_json(output_dir / "train196-report.json", report)
    return report


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repo-root", type=Path, default=Path("."))
    run_parser.add_argument("--source-sha", required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    worker = sub.add_parser("worker")
    worker.add_argument("--repo-root", type=Path, default=Path("."))
    worker.add_argument("--source-sha", required=True)
    worker.add_argument("--output-dir", type=Path, required=True)
    worker.add_argument("--candidate", required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    repo = args.repo_root.resolve()
    if args.command == "worker":
        report = _worker(
            repo=repo,
            source_sha=args.source_sha,
            output_dir=args.output_dir.resolve(),
            seed=args.seed,
            candidate=_candidate_by_label(args.candidate),
            output=args.output.resolve(),
        )
        print(json.dumps({
            "candidate": report["candidate"]["label"],
            "seed": report["seed"],
            "final_bpb": report["final_heldout"]["bpb"],
            "clip_rate": report["training"]["clip_rate"],
            "optimizer_updates": report["training"]["optimizer_updates"],
            "throughput": report["training"]["throughput_loss_tokens_per_second"],
        }, sort_keys=True))
        return 0
    report = run(repo=repo, source_sha=args.source_sha, output_dir=args.output_dir.resolve(), seeds=args.seeds)
    print(json.dumps({
        "report_sha256": report["report_sha256"],
        "selection": report["selection"],
        "microbatch_hardware_efficiency": report["microbatch_hardware_efficiency"],
        "machine_scope": report["machine_scope"],
        "gpu_retest_preregistration": report["gpu_retest_preregistration"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
