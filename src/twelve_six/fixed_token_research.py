"""Strict fixed-token scaling and iso-parameter depth/width experiments.

This is an additive research layer over RESEARCH41.  It deliberately reuses the
incumbent decoder, Trainer, byte-tokenizer/data trace, initialization family and
held-out evaluator instead of creating a second training/scaling framework.

The key contract is stronger than RESEARCH41 v1: every requested token budget
must be exactly reachable by the frozen optimizer batch trace.  The runner never
rounds a budget up or down, and evaluation targets never enter Trainer.tokens_seen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

try:  # resource is POSIX-only; exact tensor bytes remain portable.
    import resource
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts.
    resource = None

from .checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
    verify_checkpoint,
)
from .model import InitSpec, ModelSpec, TwelveSixDecoder
from .scaling_experiment import (
    PACKING_ID,
    _byte_stream,
    _canonical_hash,
    _file_sha256,
    _git_head,
    _make_batch,
    _model_spec,
    _read_jsonl,
    _trainer_config,
    _validation_loss,
    controlled_specs,
)
from .tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from .training import Trainer

FIXED_SCHEMA = "12-6.fixed-token-scaling.v1"
DEPTH_WIDTH_SCHEMA = "12-6.depth-width-500k.v1"
AUTHORITY = "LOCAL_FREE_CONTROLLED_RESEARCH_EVIDENCE_NOT_STAGE_PROMOTION_OR_PAID_AUTHORIZATION"
COMPUTE_PROXY = "6 * trainable_parameters * optimized_valid_causal_loss_tokens"
EXACT_TOKEN_BUDGETS = (4_284, 16_632, 65_772)
BATCH_SIZE = 4
SEQUENCE_LENGTH = 64
TOKENS_PER_UPDATE = BATCH_SIZE * (SEQUENCE_LENGTH - 1)
SEED = 1337
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

# Predeclared before execution.  The d_ff values below are not validation-tuned;
# they are the nearest exact-algebra matches to 500K for five fixed depth/width
# anchors.  Tied embedding/output cost is included by ModelSpec.parameter_count().
_DEPTH_WIDTH_GEOMETRIES: tuple[tuple[str, dict[str, int]], ...] = (
    (
        "shallow_wide",
        {"d_model": 136, "n_layers": 2, "n_heads": 4, "n_kv_heads": 4, "head_dim": 34, "d_ff": 384},
    ),
    (
        "mid_shallow",
        {"d_model": 112, "n_layers": 3, "n_heads": 4, "n_kv_heads": 4, "head_dim": 28, "d_ff": 320},
    ),
    (
        "balanced",
        {"d_model": 96, "n_layers": 4, "n_heads": 4, "n_kv_heads": 4, "head_dim": 24, "d_ff": 280},
    ),
    (
        "deep_narrow",
        {"d_model": 80, "n_layers": 6, "n_heads": 4, "n_kv_heads": 4, "head_dim": 20, "d_ff": 224},
    ),
    (
        "very_deep_narrow",
        {"d_model": 72, "n_layers": 8, "n_heads": 4, "n_kv_heads": 4, "head_dim": 18, "d_ff": 184},
    ),
)
_EXPECTED_DEPTH_WIDTH_COUNTS = (496_808, 502_544, 495_456, 497_680, 503_496)
_EXPECTED_SCALING_COUNTS = (95_568, 267_912, 467_808, 1_037_696)


def validate_exact_token_budgets(
    budgets: tuple[int, ...] = EXACT_TOKEN_BUDGETS,
    *,
    tokens_per_update: int = TOKENS_PER_UPDATE,
) -> tuple[int, ...]:
    """Fail closed unless every budget lands exactly on a frozen update boundary."""
    if tokens_per_update <= 0:
        raise ValueError("tokens_per_update must be positive")
    if not budgets or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("token budgets must be strictly increasing and unique")
    for budget in budgets:
        if budget <= 0:
            raise ValueError("token budgets must be positive")
        if budget % tokens_per_update:
            raise ValueError(
                "strict fixed-token budget is not reachable without changing the frozen "
                f"batch/update trace: budget={budget}, tokens_per_update={tokens_per_update}"
            )
    return budgets


def depth_width_specs() -> tuple[tuple[str, ModelSpec], ...]:
    """Return the predeclared tied-embedding ~500K depth/width family."""
    result = tuple((label, _model_spec(geometry)) for label, geometry in _DEPTH_WIDTH_GEOMETRIES)
    counts = tuple(spec.parameter_count() for _, spec in result)
    if counts != _EXPECTED_DEPTH_WIDTH_COUNTS:
        raise RuntimeError(f"depth/width parameter drift: {counts!r} != {_EXPECTED_DEPTH_WIDTH_COUNTS!r}")
    for _, spec in result:
        breakdown = spec.parameter_breakdown()
        if not spec.tie_word_embeddings or breakdown["lm_head_extra"] != 0:
            raise RuntimeError("depth/width experiment requires exactly tied embedding/output weights")
        if breakdown["token_embedding"] != spec.vocab_size * spec.d_model:
            raise RuntimeError("tied embedding parameter accounting drift")
        if spec.vocab_size != 256 or spec.max_seq_len != 256:
            raise RuntimeError("depth/width controls require vocab/context 256/256")
    return result


def _hash_state(value: Any) -> str:
    """Deterministically fingerprint nested data/tensors without pickle."""
    digest = hashlib.sha256()

    def feed(item: Any) -> None:
        if is_dataclass(item) and not isinstance(item, type):
            feed(asdict(item))
            return
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"T")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(item, Mapping):
            digest.update(b"M")
            for key in sorted(item, key=lambda entry: repr(entry)):
                feed(key)
                feed(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"L")
            for entry in item:
                feed(entry)
            return
        digest.update(type(item).__name__.encode("utf-8"))
        digest.update(repr(item).encode("utf-8"))

    feed(value)
    return digest.hexdigest()


def _model_fingerprint(model: TwelveSixDecoder) -> str:
    return _hash_state(model.state_dict())


def _trainer_fingerprint(trainer: Trainer) -> str:
    return _hash_state(trainer.state_dict())


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(key) + _tensor_bytes(entry) for key, entry in value.items())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(entry) for entry in value)
    return 0


def _parameter_snapshot(model: TwelveSixDecoder) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _update_statistics(
    before: Mapping[str, torch.Tensor],
    model: TwelveSixDecoder,
) -> tuple[float, float, dict[int, float]]:
    delta_sq = 0.0
    weight_sq = 0.0
    layer_delta_sq: dict[int, float] = {}
    layer_weight_sq: dict[int, float] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        prior = before[name]
        current = parameter.detach().cpu()
        delta = current.float() - prior.float()
        d2 = float(torch.sum(delta * delta).item())
        w2 = float(torch.sum(prior.float() * prior.float()).item())
        delta_sq += d2
        weight_sq += w2
        if name.startswith("blocks."):
            layer = int(name.split(".", 2)[1])
            layer_delta_sq[layer] = layer_delta_sq.get(layer, 0.0) + d2
            layer_weight_sq[layer] = layer_weight_sq.get(layer, 0.0) + w2
    global_ratio = math.sqrt(delta_sq) / max(math.sqrt(weight_sq), 1e-30)
    layer_ratios = {
        layer: math.sqrt(layer_delta_sq[layer]) / max(math.sqrt(layer_weight_sq[layer]), 1e-30)
        for layer in sorted(layer_delta_sq)
    }
    return math.sqrt(delta_sq), global_ratio, layer_ratios


def _process_hwm_bytes() -> int | None:
    if resource is None:
        return None
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB, macOS reports bytes.
    return raw if sys.platform == "darwin" else raw * 1024


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class _LayerRecorder:
    """Training-only per-block activation and raw-gradient instrumentation."""

    def __init__(self, model: TwelveSixDecoder) -> None:
        self.n_layers = len(model.blocks)
        self.capture = False
        self._grad_sq: dict[int, float] = {}
        self._activation_rms: dict[int, float] = {}
        self._activation_max_abs: dict[int, float] = {}
        self.history: dict[int, dict[str, list[float]]] = {
            index: {"activation_rms": [], "activation_max_abs": [], "grad_norm": [], "update_ratio": []}
            for index in range(self.n_layers)
        }
        self._handles: list[Any] = []
        for index, block in enumerate(model.blocks):
            self._handles.append(block.register_forward_hook(self._activation_hook(index)))
            for parameter in block.parameters():
                if parameter.requires_grad:
                    self._handles.append(parameter.register_hook(self._gradient_hook(index)))

    def _activation_hook(self, layer: int):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            if not self.capture:
                return
            detached = output.detach().float()
            self._activation_rms[layer] = float(torch.sqrt(torch.mean(detached * detached)).item())
            self._activation_max_abs[layer] = float(detached.abs().max().item())

        return hook

    def _gradient_hook(self, layer: int):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if self.capture:
                detached = grad.detach().float()
                self._grad_sq[layer] = self._grad_sq.get(layer, 0.0) + float(
                    torch.sum(detached * detached).item()
                )
            return grad

        return hook

    def begin_step(self) -> None:
        self._grad_sq.clear()
        self._activation_rms.clear()
        self._activation_max_abs.clear()
        self.capture = True

    def finish_step(self, *, valid_tokens: int, update_ratios: Mapping[int, float]) -> None:
        self.capture = False
        if valid_tokens <= 0:
            raise RuntimeError("layer recorder requires positive valid-token count")
        for layer in range(self.n_layers):
            if layer not in self._activation_rms or layer not in self._grad_sq:
                raise RuntimeError(f"missing layer telemetry for block {layer}")
            self.history[layer]["activation_rms"].append(self._activation_rms[layer])
            self.history[layer]["activation_max_abs"].append(self._activation_max_abs[layer])
            self.history[layer]["grad_norm"].append(math.sqrt(self._grad_sq[layer]) / valid_tokens)
            self.history[layer]["update_ratio"].append(float(update_ratios[layer]))

    def close(self) -> None:
        self.capture = False
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summary(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer, fields in self.history.items():
            rows.append(
                {
                    "layer": layer,
                    "mean_activation_rms": statistics.fmean(fields["activation_rms"]),
                    "max_activation_abs": max(fields["activation_max_abs"]),
                    "mean_grad_norm": statistics.fmean(fields["grad_norm"]),
                    "max_grad_norm": max(fields["grad_norm"]),
                    "mean_update_to_weight_ratio": statistics.fmean(fields["update_ratio"]),
                }
            )
        return rows


def _evaluation_guard(
    model: TwelveSixDecoder,
    trainer: Trainer,
    validation_records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    *,
    expected_validation_tokens: int | None,
) -> tuple[float, int, dict[str, Any]]:
    before_tokens = trainer.tokens_seen
    before_step = trainer.optimizer_step
    before_model = _model_fingerprint(model)
    before_trainer = _trainer_fingerprint(trainer)
    validation_loss, validation_tokens = _validation_loss(model, validation_records, tokenizer)
    after = {
        "trainer_tokens_unchanged": trainer.tokens_seen == before_tokens,
        "optimizer_step_unchanged": trainer.optimizer_step == before_step,
        "model_state_unchanged": _model_fingerprint(model) == before_model,
        "trainer_state_unchanged": _trainer_fingerprint(trainer) == before_trainer,
        "optimized_validation_tokens": 0,
    }
    if expected_validation_tokens is not None and validation_tokens != expected_validation_tokens:
        raise RuntimeError("validation target count drift")
    if not all(
        bool(after[key])
        for key in (
            "trainer_tokens_unchanged",
            "optimizer_step_unchanged",
            "model_state_unchanged",
            "trainer_state_unchanged",
        )
    ):
        raise RuntimeError(f"evaluation mutated optimized state: {after!r}")
    return validation_loss, validation_tokens, after


def _checkpoint_identity(
    *,
    source_sha: str,
    spec: ModelSpec,
    init_spec: InitSpec,
    trainer: Trainer,
    trainer_config: Any,
    dataset_manifest_hash: str,
    run_manifest_hash: str,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        dataset_manifest_hash=dataset_manifest_hash,
        run_manifest_hash=run_manifest_hash,
        training_config={
            "trainer": asdict(trainer_config),
            "init_spec_sha256": init_spec.identity_sha256(),
            "data": {
                "packing_version": PACKING_ID,
                "batch_size": BATCH_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "valid_causal_targets_per_update": TOKENS_PER_UPDATE,
            },
        },
        seed=SEED,
        precision="fp32",
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": trainer_config.learning_rate,
            "betas": list(trainer_config.betas),
            "eps": trainer_config.eps,
            "weight_decay": trainer_config.weight_decay,
        },
        scheduler=None,
    )


def _fresh_verified_resume(
    *,
    model: TwelveSixDecoder,
    trainer: Trainer,
    spec: ModelSpec,
    init_spec: InitSpec,
    trainer_config: Any,
    source_sha: str,
    dataset_manifest_hash: str,
    run_manifest_hash: str,
    checkpoint_root: Path,
) -> tuple[TwelveSixDecoder, Trainer, dict[str, Any]]:
    trainer.assert_checkpoint_safe()
    before_model = _model_fingerprint(model)
    before_trainer = _trainer_fingerprint(trainer)
    checkpoint_dir = checkpoint_root / f"step-{trainer.optimizer_step:06d}"
    identity = _checkpoint_identity(
        source_sha=source_sha,
        spec=spec,
        init_spec=init_spec,
        trainer=trainer,
        trainer_config=trainer_config,
        dataset_manifest_hash=dataset_manifest_hash,
        run_manifest_hash=run_manifest_hash,
    )
    save_started = time.perf_counter()
    manifest = save_trainer_checkpoint(
        checkpoint_dir,
        model=model,
        trainer=trainer,
        identity=identity,
    )
    save_seconds = time.perf_counter() - save_started
    verified = verify_checkpoint(checkpoint_dir)

    torch.manual_seed(SEED + 99_999)
    fresh_model = TwelveSixDecoder(spec, init_spec)
    fresh_trainer = Trainer(fresh_model, trainer_config, device="cpu")
    load_started = time.perf_counter()
    load_trainer_checkpoint(
        checkpoint_dir,
        model=fresh_model,
        trainer=fresh_trainer,
        restore_rng=True,
        expected_git_sha=source_sha,
        expected_model_spec_hash=hash_json(spec.to_dict()),
        expected_tokenizer_hash=BYTE_TOKENIZER_HASH,
        expected_tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        expected_dataset_manifest_hash=dataset_manifest_hash,
        expected_run_manifest_hash=run_manifest_hash,
        expected_seed=SEED,
    )
    load_seconds = time.perf_counter() - load_started
    model_match = _model_fingerprint(fresh_model) == before_model
    trainer_match = _trainer_fingerprint(fresh_trainer) == before_trainer
    counters_match = (
        fresh_trainer.tokens_seen == trainer.tokens_seen
        and fresh_trainer.optimizer_step == trainer.optimizer_step
        and fresh_trainer.micro_step == trainer.micro_step
    )
    if not (model_match and trainer_match and counters_match):
        raise RuntimeError("fresh-process-equivalent checkpoint resume state mismatch")
    proof = {
        "passed": True,
        "checkpoint_id": manifest["checkpoint_id"],
        "verified_checkpoint_id": verified["checkpoint_id"],
        "optimizer_step": fresh_trainer.optimizer_step,
        "optimized_tokens": fresh_trainer.tokens_seen,
        "model_fingerprint_match": model_match,
        "trainer_fingerprint_match": trainer_match,
        "counters_match": counters_match,
        "checkpoint_bytes": _directory_bytes(checkpoint_dir),
        "save_seconds": save_seconds,
        "load_seconds": load_seconds,
    }
    return fresh_model, fresh_trainer, proof


def _run_candidate(
    *,
    label: str,
    spec: ModelSpec,
    source_sha: str,
    train_stream: bytes,
    validation_records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    dataset_manifest_hash: str,
    checkpoint_root: Path,
    collect_layer_stats: bool,
) -> dict[str, Any]:
    budgets = validate_exact_token_budgets()
    max_steps = budgets[-1] // TOKENS_PER_UPDATE
    init_spec = InitSpec()
    trainer_config = _trainer_config(max_steps=max_steps, seed=SEED)
    run_manifest = {
        "schema": "12-6.fixed-token-run-manifest.v1",
        "source_sha": source_sha,
        "model_spec_sha256": spec.identity_sha256(),
        "init_spec_sha256": init_spec.identity_sha256(),
        "tokenizer": BYTE_TOKENIZER_VERSION,
        "tokenizer_hash": BYTE_TOKENIZER_HASH,
        "vocab_hash": BYTE_VOCAB_HASH,
        "dataset_manifest_hash": dataset_manifest_hash,
        "packing": PACKING_ID,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "valid_targets_per_update": TOKENS_PER_UPDATE,
        "budgets": list(budgets),
        "trainer_config": asdict(trainer_config),
        "seed": SEED,
    }
    run_manifest_hash = _canonical_hash(run_manifest)

    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init_spec)
    trainer = Trainer(model, trainer_config, device="cpu")
    recorder = _LayerRecorder(model) if collect_layer_stats else None
    initial_loss, validation_tokens, initial_guard = _evaluation_guard(
        model,
        trainer,
        validation_records,
        tokenizer,
        expected_validation_tokens=None,
    )
    checkpoints: list[dict[str, Any]] = []
    train_curve: list[dict[str, Any]] = []
    resume_proof: dict[str, Any] | None = None
    clipped_steps = 0
    step_wall = 0.0
    run_started = time.perf_counter()
    next_budget = 0

    for step in range(max_steps):
        expected_before = step * TOKENS_PER_UPDATE
        if trainer.tokens_seen != expected_before or trainer.optimizer_step != step:
            raise RuntimeError(
                "pre-update token accounting drift: "
                f"step={step}, tokens={trainer.tokens_seen}, optimizer_step={trainer.optimizer_step}"
            )
        batch = _make_batch(
            train_stream,
            step=step,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
        )
        before = _parameter_snapshot(model)
        if recorder is not None:
            recorder.begin_step()
        started = time.perf_counter()
        metrics = trainer.train_microbatch({"input_ids": batch})
        elapsed = time.perf_counter() - started
        step_wall += elapsed
        delta_l2, update_ratio, layer_update_ratios = _update_statistics(before, model)
        if recorder is not None:
            recorder.finish_step(valid_tokens=metrics.tokens, update_ratios=layer_update_ratios)

        expected_after = (step + 1) * TOKENS_PER_UPDATE
        if metrics.tokens != TOKENS_PER_UPDATE:
            raise RuntimeError(
                f"valid causal token drift: observed {metrics.tokens}, expected {TOKENS_PER_UPDATE}"
            )
        if not metrics.optimizer_stepped or metrics.optimizer_step != step + 1:
            raise RuntimeError("optimizer update cadence drift")
        if trainer.tokens_seen != expected_after:
            raise RuntimeError(
                f"post-update token accounting drift: {trainer.tokens_seen} != {expected_after}"
            )
        if metrics.grad_norm is None or not math.isfinite(metrics.grad_norm):
            raise RuntimeError("missing/non-finite pre-clip gradient norm")
        if metrics.grad_norm > 1.0:
            clipped_steps += 1
        train_curve.append(
            {
                "optimizer_step": metrics.optimizer_step,
                "optimized_tokens": trainer.tokens_seen,
                "train_loss": metrics.loss,
                "pre_clip_grad_norm": metrics.grad_norm,
                "update_l2": delta_l2,
                "update_to_weight_ratio": update_ratio,
                "step_seconds": elapsed,
            }
        )

        if next_budget < len(budgets) and trainer.tokens_seen == budgets[next_budget]:
            validation_loss, checked_validation_tokens, eval_guard = _evaluation_guard(
                model,
                trainer,
                validation_records,
                tokenizer,
                expected_validation_tokens=validation_tokens,
            )
            checkpoints.append(
                {
                    "requested_token_budget": budgets[next_budget],
                    "optimized_tokens": trainer.tokens_seen,
                    "optimizer_steps": trainer.optimizer_step,
                    "validation_loss": validation_loss,
                    "bits_per_byte": validation_loss / math.log(2.0),
                    "validation_loss_improvement_from_init": initial_loss - validation_loss,
                    "compute_proxy": 6 * spec.parameter_count() * trainer.tokens_seen,
                    "last_train_loss": metrics.loss,
                    "last_pre_clip_grad_norm": metrics.grad_norm,
                    "last_update_to_weight_ratio": update_ratio,
                    "validation_tokens_evaluated": checked_validation_tokens,
                    "evaluation_guard": eval_guard,
                }
            )
            if budgets[next_budget] == budgets[-2]:
                if recorder is not None:
                    recorder.close()
                model, trainer, resume_proof = _fresh_verified_resume(
                    model=model,
                    trainer=trainer,
                    spec=spec,
                    init_spec=init_spec,
                    trainer_config=trainer_config,
                    source_sha=source_sha,
                    dataset_manifest_hash=dataset_manifest_hash,
                    run_manifest_hash=run_manifest_hash,
                    checkpoint_root=checkpoint_root / label,
                )
                if recorder is not None:
                    old_history = recorder.history
                    recorder = _LayerRecorder(model)
                    recorder.history = old_history
            next_budget += 1
        elif next_budget < len(budgets) and trainer.tokens_seen > budgets[next_budget]:
            raise RuntimeError("strict token checkpoint was skipped; refusing rounded evidence")

    if next_budget != len(budgets):
        raise RuntimeError("not all exact token checkpoints were reached")
    if trainer.tokens_seen != budgets[-1]:
        raise RuntimeError("final optimized token count is not the exact budget")
    if resume_proof is None or not resume_proof["passed"]:
        raise RuntimeError("required fresh-object checkpoint resume was not proved")
    end_to_end = time.perf_counter() - run_started
    layer_summary = None
    if recorder is not None:
        recorder.close()
        layer_summary = recorder.summary()

    model_tensor_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    optimizer_tensor_bytes = _tensor_bytes(trainer.optimizer.state_dict())
    grad_norms = [float(point["pre_clip_grad_norm"]) for point in train_curve]
    update_ratios = [float(point["update_to_weight_ratio"]) for point in train_curve]
    step_seconds = [float(point["step_seconds"]) for point in train_curve]
    return {
        "label": label,
        "model_spec": spec.to_dict(),
        "model_identity_sha256": spec.identity_sha256(),
        "parameter_breakdown": spec.parameter_breakdown(),
        "parameters": spec.parameter_count(),
        "run_manifest_sha256": run_manifest_hash,
        "initial_validation_loss": initial_loss,
        "validation_tokens": validation_tokens,
        "initial_evaluation_guard": initial_guard,
        "checkpoints": checkpoints,
        "train_curve": train_curve,
        "resume_proof": resume_proof,
        "gradient_update_summary": {
            "mean_pre_clip_grad_norm": statistics.fmean(grad_norms),
            "max_pre_clip_grad_norm": max(grad_norms),
            "clip_threshold": 1.0,
            "clip_steps": clipped_steps,
            "clip_frequency": clipped_steps / len(train_curve),
            "mean_update_to_weight_ratio": statistics.fmean(update_ratios),
            "max_update_to_weight_ratio": max(update_ratios),
        },
        "timing": {
            "training_step_wall_seconds": step_wall,
            "end_to_end_wall_seconds": end_to_end,
            "mean_step_seconds": statistics.fmean(step_seconds),
            "median_step_seconds": statistics.median(step_seconds),
            "optimized_tokens_per_end_to_end_wall_second": budgets[-1] / end_to_end,
        },
        "memory": {
            "model_tensor_bytes": model_tensor_bytes,
            "optimizer_tensor_bytes": optimizer_tensor_bytes,
            "model_plus_optimizer_tensor_bytes": model_tensor_bytes + optimizer_tensor_bytes,
            "process_max_rss_bytes": _process_hwm_bytes(),
            "process_max_rss_scope": "process_high_water_mark_cumulative_not_candidate_isolated",
        },
        "layer_summary": layer_summary,
    }


def _final_checkpoint(run: Mapping[str, Any]) -> Mapping[str, Any]:
    return run["checkpoints"][-1]


def _rank_fixed_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for run in runs:
        final = _final_checkpoint(run)
        gain = float(run["initial_validation_loss"]) - float(final["validation_loss"])
        parameters = int(run["parameters"])
        compute = int(final["compute_proxy"])
        wall = float(run["timing"]["end_to_end_wall_seconds"])
        entries.append(
            {
                "label": run["label"],
                "parameters": parameters,
                "final_validation_loss": float(final["validation_loss"]),
                "validation_improvement": gain,
                "validation_improvement_per_parameter": gain / parameters,
                "validation_improvement_per_compute_proxy_unit": gain / compute,
                "validation_improvement_per_wall_second": gain / wall,
            }
        )
    def order(metric: str, *, reverse: bool) -> list[dict[str, Any]]:
        return sorted(entries, key=lambda row: float(row[metric]), reverse=reverse)

    best_loss = min(float(row["final_validation_loss"]) for row in entries)
    # Predeclared decision rule: use the smallest candidate within 5% relative
    # held-out loss of the best model at the largest exact token budget.
    eligible = sorted(
        (
            row for row in entries
            if float(row["final_validation_loss"]) <= best_loss * 1.05
        ),
        key=lambda row: int(row["parameters"]),
    )
    recommendation = eligible[0]
    return {
        "at_optimized_tokens": EXACT_TOKEN_BUDGETS[-1],
        "best_validation": order("final_validation_loss", reverse=False),
        "validation_improvement_per_parameter": order(
            "validation_improvement_per_parameter", reverse=True
        ),
        "validation_improvement_per_compute": order(
            "validation_improvement_per_compute_proxy_unit", reverse=True
        ),
        "validation_improvement_per_wall_second": order(
            "validation_improvement_per_wall_second", reverse=True
        ),
        "research_vehicle_rule": "smallest_parameter_count_with_final_validation_loss_within_5_percent_of_best",
        "recommended_primary_small_model": {
            "label": recommendation["label"],
            "parameters": recommendation["parameters"],
            "final_validation_loss": recommendation["final_validation_loss"],
            "best_validation_loss": best_loss,
        },
    }


def _rank_depth_width(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        runs,
        key=lambda run: (
            float(_final_checkpoint(run)["validation_loss"]),
            float(run["timing"]["median_step_seconds"]),
        ),
    )
    winner = ordered[0]
    return {
        "selection_rule": "lowest_predeclared_final_held_out_loss_then_lower_median_step_time_exact_tiebreak",
        "at_optimized_tokens": EXACT_TOKEN_BUDGETS[-1],
        "ordered_candidates": [
            {
                "label": run["label"],
                "parameters": run["parameters"],
                "validation_loss": _final_checkpoint(run)["validation_loss"],
                "bits_per_byte": _final_checkpoint(run)["bits_per_byte"],
                "median_step_seconds": run["timing"]["median_step_seconds"],
                "model_plus_optimizer_tensor_bytes": run["memory"]["model_plus_optimizer_tensor_bytes"],
            }
            for run in ordered
        ],
        "recommended_500k_geometry": {
            "label": winner["label"],
            "parameters": winner["parameters"],
            "model_spec": winner["model_spec"],
            "validation_loss": _final_checkpoint(winner)["validation_loss"],
            "bits_per_byte": _final_checkpoint(winner)["bits_per_byte"],
        },
    }


def _common_inputs(repo_root: Path, source_sha: str) -> tuple[ByteTokenizer, bytes, list[dict[str, Any]], str, dict[str, Any]]:
    if not _HEX40.fullmatch(source_sha):
        raise ValueError("source_sha must be lowercase exact 40-hex")
    observed = _git_head(repo_root)
    if observed != source_sha:
        raise RuntimeError(f"exact-checkout mismatch: expected {source_sha}, observed {observed}")
    validate_exact_token_budgets()
    tokenizer = ByteTokenizer()
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_records = _read_jsonl(train_path)
    validation_records = _read_jsonl(validation_path)
    train_ids = {str(record["id"]) for record in train_records}
    validation_ids = {str(record["id"]) for record in validation_records}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        raise RuntimeError(f"train/validation record overlap: {overlap!r}")
    stream = _byte_stream(train_records, tokenizer)
    data = {
        "train_path": str(train_path.relative_to(repo_root)),
        "validation_path": str(validation_path.relative_to(repo_root)),
        "manifest_path": str(manifest_path.relative_to(repo_root)),
        "train_sha256": _file_sha256(train_path),
        "validation_sha256": _file_sha256(validation_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "train_validation_record_overlap": overlap,
    }
    return tokenizer, stream, validation_records, data["manifest_sha256"], data


def run_fixed_scaling(
    *, repo_root: Path, source_sha: str, output_path: Path, torch_threads: int = 2
) -> dict[str, Any]:
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    tokenizer, train_stream, validation_records, dataset_manifest_hash, data = _common_inputs(
        repo_root, source_sha
    )
    specs = controlled_specs()
    if tuple(spec.parameter_count() for spec in specs) != _EXPECTED_SCALING_COUNTS:
        raise RuntimeError("incumbent controlled scaling family drift")
    with tempfile.TemporaryDirectory(prefix="research06-fixed-") as temporary:
        root = Path(temporary)
        runs = [
            _run_candidate(
                label=f"fixed_{spec.parameter_count()}",
                spec=spec,
                source_sha=source_sha,
                train_stream=train_stream,
                validation_records=validation_records,
                tokenizer=tokenizer,
                dataset_manifest_hash=dataset_manifest_hash,
                checkpoint_root=root,
                collect_layer_stats=False,
            )
            for spec in specs
        ]
    report: dict[str, Any] = {
        "schema": FIXED_SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": "Oleksii-debug/12-6-ai.", "git_sha": source_sha, "incumbent": "RESEARCH41_PR_162"},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_threads": torch_threads,
            "paid_compute": False,
        },
        "controls": {
            "tokenizer_id": BYTE_TOKENIZER_VERSION,
            "tokenizer_hash": BYTE_TOKENIZER_HASH,
            "vocab_hash": BYTE_VOCAB_HASH,
            "vocab_size": 256,
            "model_max_seq_len": 256,
            "packing_id": PACKING_ID,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "valid_causal_loss_tokens_per_update": TOKENS_PER_UPDATE,
            "exact_token_budgets": list(EXACT_TOKEN_BUDGETS),
            "seed": SEED,
            "init_spec": asdict(InitSpec()),
            "optimizer_recipe": asdict(
                _trainer_config(max_steps=EXACT_TOKEN_BUDGETS[-1] // TOKENS_PER_UPDATE, seed=SEED)
            ),
            "compute_proxy": COMPUTE_PROXY,
        },
        "data": data,
        "runs": runs,
        "ranking": _rank_fixed_runs(runs),
        "historical_research41_budget_repair": {
            "historical_requested_budgets": [4_096, 16_384, 65_536],
            "historical_actual_optimized_tokens": list(EXACT_TOKEN_BUDGETS),
            "repair": "new strict contract names actual reachable budgets and rejects non-divisible budgets instead of rounding",
        },
        "truth_boundary": {
            "held_out_generalization_measured": True,
            "train_loss_used_as_generalization": False,
            "evaluation_tokens_optimized": False,
            "representative_scale_corpus_claimed": False,
            "stage_freeze": False,
            "promotion_authority": False,
            "paid_compute_authority": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    validate_report(report, expected_source_sha=source_sha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_depth_width(
    *, repo_root: Path, source_sha: str, output_path: Path, torch_threads: int = 2
) -> dict[str, Any]:
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    tokenizer, train_stream, validation_records, dataset_manifest_hash, data = _common_inputs(
        repo_root, source_sha
    )
    with tempfile.TemporaryDirectory(prefix="model09-depth-width-") as temporary:
        root = Path(temporary)
        runs = [
            _run_candidate(
                label=label,
                spec=spec,
                source_sha=source_sha,
                train_stream=train_stream,
                validation_records=validation_records,
                tokenizer=tokenizer,
                dataset_manifest_hash=dataset_manifest_hash,
                checkpoint_root=root,
                collect_layer_stats=True,
            )
            for label, spec in depth_width_specs()
        ]
    report: dict[str, Any] = {
        "schema": DEPTH_WIDTH_SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": "Oleksii-debug/12-6-ai.", "git_sha": source_sha, "parent_experiment": "RESEARCH41_PR_162"},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_threads": torch_threads,
            "paid_compute": False,
        },
        "controls": {
            "tokenizer_id": BYTE_TOKENIZER_VERSION,
            "tokenizer_hash": BYTE_TOKENIZER_HASH,
            "vocab_hash": BYTE_VOCAB_HASH,
            "vocab_size": 256,
            "model_max_seq_len": 256,
            "packing_id": PACKING_ID,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "valid_causal_loss_tokens_per_update": TOKENS_PER_UPDATE,
            "exact_token_budgets": list(EXACT_TOKEN_BUDGETS),
            "seed": SEED,
            "init_spec": asdict(InitSpec()),
            "optimizer_recipe": asdict(
                _trainer_config(max_steps=EXACT_TOKEN_BUDGETS[-1] // TOKENS_PER_UPDATE, seed=SEED)
            ),
            "compute_proxy": COMPUTE_PROXY,
            "candidate_set_predeclared_not_validation_tuned": True,
        },
        "data": data,
        "candidate_parameter_counts": [spec.parameter_count() for _, spec in depth_width_specs()],
        "runs": runs,
        "ranking": _rank_depth_width(runs),
        "truth_boundary": {
            "held_out_generalization_measured": True,
            "train_loss_used_as_generalization": False,
            "evaluation_tokens_optimized": False,
            "architecture_search_extended_after_validation": False,
            "representative_scale_corpus_claimed": False,
            "stage_freeze": False,
            "promotion_authority": False,
            "paid_compute_authority": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    validate_report(report, expected_source_sha=source_sha)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_report(report: Mapping[str, Any], *, expected_source_sha: str | None = None) -> None:
    schema = report.get("schema")
    if schema not in {FIXED_SCHEMA, DEPTH_WIDTH_SCHEMA}:
        raise ValueError("unexpected research report schema")
    unsigned = dict(report)
    observed_hash = unsigned.pop("report_sha256", None)
    if observed_hash != _canonical_hash(unsigned):
        raise ValueError("research report self-hash mismatch")
    source_sha = report["source"]["git_sha"]
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ValueError("research report source SHA mismatch")
    if report["runtime"].get("paid_compute") is not False:
        raise ValueError("LOCAL_FREE report cannot claim paid compute")
    controls = report["controls"]
    if tuple(controls["exact_token_budgets"]) != EXACT_TOKEN_BUDGETS:
        raise ValueError("exact token budget drift")
    if int(controls["valid_causal_loss_tokens_per_update"]) != TOKENS_PER_UPDATE:
        raise ValueError("valid causal token/update drift")
    expected_counts = _EXPECTED_SCALING_COUNTS if schema == FIXED_SCHEMA else _EXPECTED_DEPTH_WIDTH_COUNTS
    runs = report["runs"]
    if tuple(int(run["parameters"]) for run in runs) != expected_counts:
        raise ValueError("candidate parameter-family drift")
    for run in runs:
        if not run["resume_proof"].get("passed"):
            raise ValueError("fresh-object resume proof missing")
        if int(run["resume_proof"]["optimized_tokens"]) != EXACT_TOKEN_BUDGETS[-2]:
            raise ValueError("resume token boundary drift")
        checkpoints = run["checkpoints"]
        if len(checkpoints) != len(EXACT_TOKEN_BUDGETS):
            raise ValueError("exact checkpoint count drift")
        for expected_budget, checkpoint in zip(EXACT_TOKEN_BUDGETS, checkpoints, strict=True):
            if int(checkpoint["requested_token_budget"]) != expected_budget:
                raise ValueError("requested exact budget drift")
            if int(checkpoint["optimized_tokens"]) != expected_budget:
                raise ValueError("optimized tokens do not exactly equal fixed budget")
            expected_proxy = 6 * int(run["parameters"]) * expected_budget
            if int(checkpoint["compute_proxy"]) != expected_proxy:
                raise ValueError("compute proxy drift")
            loss = float(checkpoint["validation_loss"])
            bpb = float(checkpoint["bits_per_byte"])
            if not (math.isfinite(loss) and loss > 0 and math.isfinite(bpb)):
                raise ValueError("invalid held-out loss/BPB")
            if not math.isclose(bpb, loss / math.log(2.0), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("byte-token BPB drift")
            guard = checkpoint["evaluation_guard"]
            if int(guard.get("optimized_validation_tokens", -1)) != 0:
                raise ValueError("evaluation tokens entered optimized-token accounting")
            for key in (
                "trainer_tokens_unchanged",
                "optimizer_step_unchanged",
                "model_state_unchanged",
                "trainer_state_unchanged",
            ):
                if guard.get(key) is not True:
                    raise ValueError("evaluation mutation guard failed")
        if schema == DEPTH_WIDTH_SCHEMA:
            if run["layer_summary"] is None or len(run["layer_summary"]) != int(run["model_spec"]["n_layers"]):
                raise ValueError("depth/width layer telemetry incomplete")
            breakdown = run["parameter_breakdown"]
            if int(breakdown["lm_head_extra"]) != 0:
                raise ValueError("depth/width tied embedding accounting drift")
    truth = report["truth_boundary"]
    if truth.get("held_out_generalization_measured") is not True:
        raise ValueError("held-out measurement truth boundary missing")
    for key in ("train_loss_used_as_generalization", "evaluation_tokens_optimized", "stage_freeze", "promotion_authority", "paid_compute_authority"):
        if truth.get(key) is not False:
            raise ValueError(f"truth boundary overclaim: {key}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("scaling", "depth-width"):
        run = sub.add_parser(name)
        run.add_argument("--repo-root", type=Path, default=Path("."))
        run.add_argument("--source-sha", required=True)
        run.add_argument("--output", type=Path, required=True)
        run.add_argument("--torch-threads", type=int, default=2)
    validate = sub.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "scaling":
        run_fixed_scaling(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            output_path=args.output,
            torch_threads=args.torch_threads,
        )
        return 0
    if args.command == "depth-width":
        run_depth_width(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            output_path=args.output,
            torch_threads=args.torch_threads,
        )
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_report(report, expected_source_sha=args.expected_source_sha)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
