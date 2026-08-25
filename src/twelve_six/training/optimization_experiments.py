"""Controlled next-scale optimization experiments for 12-6 AI.

This is an engineering experiment harness, not stage evidence. It reuses exact
stage/model/data identities and measures optimizer behavior without modifying
Trainer semantics or declaring the controlled S0 fixture to be later-stage data.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from twelve_six.model import TwelveSixDecoder, load_stage_config
from twelve_six.packing import (
    batch_examples,
    collate_rows,
    iter_packed_examples,
    load_jsonl_records,
)
from twelve_six.tokenization import ByteTokenizer

from .config import TrainerConfig
from .loss import causal_lm_loss
from .s0_evidence_contract import (
    DATASET_IDENTITY_SHA256,
    DATASET_MANIFEST_SHA256,
    INIT_SPEC_SHA256,
    PACKING_CONFIG_SHA256,
    TOKENIZER_CONFIG_SHA256,
    TOKENIZER_VOCAB_SHA256,
    validate_locked_environment_evidence,
)
from .trainer import Trainer

PLAN_SCHEMA_VERSION = "12-6.optimization-experiment-plan.v1"
EVIDENCE_SCHEMA_VERSION = "12-6.optimization-experiment-evidence.v1"
AUTHORITY = "LOCAL_FREE_ENGINEERING_OPTIMIZATION_EVIDENCE_PROVISIONAL"
REPOSITORY = "Oleksii-debug/12-6-ai."
FIXTURE_PURPOSE = "CONTROLLED_S0_FIXTURE_ONLY_NOT_LATER_STAGE_CORPUS_OR_TOKENIZER"
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")

_STAGE_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "S1": {
        "path": "configs/stages/s1_100k.json",
        "parameter_count": 107_856,
        "modelspec_sha256": "2f0aa97a5d19e98c4e292fd5f1b454ada45ec4d2c7324e14ab7e48af19908ce6",
    },
    "S2": {
        "path": "configs/stages/s2_1m.json",
        "parameter_count": 1_066_112,
        "modelspec_sha256": "2889fdea4d17b5f592686c1a1a2fcd7dd16a9a029219351e95973ccfdef60566",
    },
    "S3": {
        "path": "configs/stages/s3_10m.json",
        "parameter_count": 10_059_840,
        "modelspec_sha256": "3b6fc1b397e6fea69c2f249ce8ab8eedaad8ca1b13b88b8d2328a6abcf34791a",
    },
}

OptimizerKind = Literal["adamw"]
SchedulerKind = Literal["constant", "linear_warmup", "cosine"]


class OptimizationExperimentError(ValueError):
    """Raised when an optimization experiment contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OptimizationExperimentError(message)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(name: str, value: Any, *, positive: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    if positive:
        _require(result > 0.0, f"{name} must be > 0")
    return result


@dataclass(frozen=True, slots=True)
class OptimizationRecipe:
    """Experiment-facing optimizer configuration.

    Trainer still owns numerical execution. This object makes the tested recipe
    explicit and fail-closed without widening the shared Trainer API.
    """

    name: str
    optimizer: OptimizerKind
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    scheduler: SchedulerKind
    warmup_fraction: float
    gradient_clip_norm: float | None
    decay_embeddings: bool
    precision: Literal["fp32"] = "fp32"

    def __post_init__(self) -> None:
        _require(bool(self.name.strip()), "recipe name must not be empty")
        _require(self.optimizer == "adamw", "only AdamW is authorized in this experiment")
        _finite_number("learning_rate", self.learning_rate, positive=True)
        _require(self.weight_decay >= 0.0, "weight_decay must be >= 0")
        _require(len(self.betas) == 2, "betas must contain two values")
        for beta in self.betas:
            _require(
                isinstance(beta, (int, float))
                and not isinstance(beta, bool)
                and math.isfinite(float(beta))
                and 0.0 <= float(beta) < 1.0,
                "betas must be finite values in [0, 1)",
            )
        _finite_number("eps", self.eps, positive=True)
        _require(
            self.scheduler in {"constant", "linear_warmup", "cosine"},
            "unsupported scheduler",
        )
        _require(
            math.isfinite(self.warmup_fraction)
            and 0.0 <= self.warmup_fraction < 1.0,
            "warmup_fraction must be in [0, 1)",
        )
        if self.gradient_clip_norm is not None:
            _finite_number("gradient_clip_norm", self.gradient_clip_norm, positive=True)
        _require(isinstance(self.decay_embeddings, bool), "decay_embeddings must be boolean")
        _require(self.precision == "fp32", "this controlled experiment is fp32 only")

    def materialize(
        self,
        *,
        schedule_horizon_steps: int,
        seed: int,
    ) -> tuple[TrainerConfig, dict[str, Any]]:
        _require(schedule_horizon_steps > 0, "schedule_horizon_steps must be positive")
        warmup_steps = int(round(schedule_horizon_steps * self.warmup_fraction))
        if self.warmup_fraction > 0.0:
            warmup_steps = max(1, warmup_steps)
        warmup_steps = min(warmup_steps, schedule_horizon_steps - 1)
        config = TrainerConfig(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
            eps=self.eps,
            max_steps=schedule_horizon_steps,
            warmup_steps=warmup_steps,
            scheduler=self.scheduler,
            gradient_accumulation_steps=1,
            gradient_clip_norm=self.gradient_clip_norm,
            precision=self.precision,
            seed=seed,
            deterministic_algorithms=True,
            deterministic_warn_only=False,
        )
        materialized = asdict(self)
        materialized["warmup_steps"] = warmup_steps
        materialized["schedule_horizon_steps"] = schedule_horizon_steps
        materialized["trainer_config_sha256"] = _canonical_hash(asdict(config))
        return config, materialized


def _parse_recipe(name: str, raw: Mapping[str, Any]) -> OptimizationRecipe:
    betas = raw.get("betas")
    _require(
        isinstance(betas, list)
        and len(betas) == 2
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in betas),
        f"recipe {name}: betas must be a two-number list",
    )
    clip = raw.get("gradient_clip_norm")
    _require(
        clip is None or (isinstance(clip, (int, float)) and not isinstance(clip, bool)),
        f"recipe {name}: invalid gradient_clip_norm",
    )
    return OptimizationRecipe(
        name=name,
        optimizer=str(raw.get("optimizer")),  # type: ignore[arg-type]
        learning_rate=float(raw.get("learning_rate")),
        weight_decay=float(raw.get("weight_decay")),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(raw.get("eps")),
        scheduler=str(raw.get("scheduler")),  # type: ignore[arg-type]
        warmup_fraction=float(raw.get("warmup_fraction")),
        gradient_clip_norm=None if clip is None else float(clip),
        decay_embeddings=raw.get("decay_embeddings"),
        precision=str(raw.get("precision", "fp32")),  # type: ignore[arg-type]
    )


def load_experiment_plan(path: str | Path) -> dict[str, Any]:
    """Load and validate the committed experiment plan."""
    plan_path = Path(path)
    raw = json.loads(plan_path.read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), "experiment plan root must be an object")
    _require(raw.get("schema_version") == PLAN_SCHEMA_VERSION, "wrong plan schema")
    recipes = raw.get("recipes")
    _require(isinstance(recipes, dict) and recipes, "plan recipes missing")
    parsed_recipes = {
        name: _parse_recipe(name, value)
        for name, value in recipes.items()
        if isinstance(name, str) and isinstance(value, Mapping)
    }
    _require(len(parsed_recipes) == len(recipes), "invalid recipe entry")
    stages = raw.get("stages")
    _require(isinstance(stages, list) and stages, "plan stages missing")
    seen_stages: set[str] = set()
    for stage_plan in stages:
        _require(isinstance(stage_plan, dict), "stage plan must be an object")
        stage = stage_plan.get("stage")
        _require(stage in _STAGE_EXPECTATIONS, f"unsupported experiment stage: {stage!r}")
        _require(stage not in seen_stages, f"duplicate stage plan: {stage}")
        seen_stages.add(stage)
        for field in ("execution_steps", "schedule_horizon_steps", "batch_size", "sequence_length"):
            value = stage_plan.get(field)
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value > 0,
                f"{stage}.{field} must be a positive integer",
            )
        _require(
            stage_plan["execution_steps"] <= stage_plan["schedule_horizon_steps"],
            f"{stage}: execution_steps exceed schedule horizon",
        )
        names = stage_plan.get("recipes")
        _require(isinstance(names, list) and names, f"{stage}: recipes missing")
        _require(len(set(names)) == len(names), f"{stage}: duplicate recipes")
        _require(all(name in parsed_recipes for name in names), f"{stage}: unknown recipe")
    _require(seen_stages == {"S1", "S2", "S3"}, "plan must cover S1, S2, and S3")
    transfer = raw.get("unexecuted_transfer_targets")
    _require(isinstance(transfer, list) and transfer, "unexecuted transfer targets missing")
    _require(
        all(isinstance(item, dict) and item.get("executed") is False for item in transfer),
        "transfer targets must stay explicitly unexecuted",
    )
    return {
        "raw": raw,
        "recipes": parsed_recipes,
        "path": str(plan_path),
        "file_sha256": _sha256_file(plan_path),
    }


def _tensor_batches(
    root: Path,
    *,
    split: str,
    tokenizer: ByteTokenizer,
    batch_size: int,
    sequence_length: int,
) -> tuple[list[dict[str, Tensor]], tuple[str, ...], int, int]:
    records = tuple(load_jsonl_records(root / f"data/s0/packaged/{split}.jsonl", split=split))
    examples = tuple(
        iter_packed_examples(
            records,
            tokenizer,
            expected_split=split,
            sequence_length=sequence_length,
        )
    )
    _require(bool(examples), f"{split} fixture produced no packed examples")
    batches: list[dict[str, Tensor]] = []
    max_token_id = -1
    for group in batch_examples(examples, batch_size=batch_size, drop_last=False):
        rows = collate_rows(group, target_mode="labels")
        input_ids = torch.tensor(rows["input_ids"], dtype=torch.long)
        labels = torch.tensor(rows["labels"], dtype=torch.long)
        max_token_id = max(max_token_id, int(input_ids.max().item()))
        batches.append({"input_ids": input_ids, "labels": labels})
    return (
        batches,
        tuple(record.record_id for record in records),
        sum(example.num_loss_tokens for example in examples),
        max_token_id,
    )


@torch.no_grad()
def _evaluate(model: TwelveSixDecoder, batches: Sequence[Mapping[str, Tensor]]) -> float:
    model.eval()
    weighted_loss = 0.0
    tokens = 0
    for batch in batches:
        logits = model(batch["input_ids"]).logits
        labels = batch["labels"]
        scoreable = int(labels[:, 1:].ne(-100).sum().item())
        loss = causal_lm_loss(logits, labels)
        _require(torch.isfinite(loss).item(), "evaluation produced non-finite loss")
        weighted_loss += float(loss.item()) * scoreable
        tokens += scoreable
    _require(tokens > 0, "evaluation produced zero scoreable tokens")
    return weighted_loss / tokens


def _snapshot(model: TwelveSixDecoder) -> dict[str, Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _parameter_l2(model: TwelveSixDecoder) -> float:
    total = 0.0
    for parameter in model.parameters():
        value = parameter.detach().float()
        total += float(torch.sum(value * value).item())
    return math.sqrt(total)


def _update_metrics(
    model: TwelveSixDecoder,
    before: Mapping[str, Tensor],
    *,
    parameter_l2_before: float,
) -> dict[str, float | int]:
    squared = 0.0
    max_abs = 0.0
    changed = 0
    total = 0
    for name, parameter in model.named_parameters():
        delta = parameter.detach().float() - before[name].float()
        squared += float(torch.sum(delta * delta).item())
        max_abs = max(max_abs, float(delta.abs().max().item()))
        changed += int(delta.ne(0).sum().item())
        total += delta.numel()
    update_l2 = math.sqrt(squared)
    relative = update_l2 / parameter_l2_before if parameter_l2_before > 0.0 else math.inf
    return {
        "update_l2": update_l2,
        "relative_update_l2": relative,
        "update_max_abs": max_abs,
        "changed_parameter_elements": changed,
        "trainable_parameter_elements": total,
    }


def _value_tensor_bytes(value: Any) -> int:
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_value_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_value_tensor_bytes(item) for item in value)
    return 0


def _optimizer_state_tensor_bytes(trainer: Trainer) -> int:
    return sum(_value_tensor_bytes(state) for state in trainer.optimizer.state.values())


def _model_parameter_bytes(model: TwelveSixDecoder) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def _tensor_digest(hasher: Any, tensor: Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(str(tuple(value.shape)).encode("ascii"))
    hasher.update(value.view(torch.uint8).numpy().tobytes())


def _model_fingerprint(model: TwelveSixDecoder) -> str:
    hasher = hashlib.sha256()
    for name, parameter in model.named_parameters():
        hasher.update(name.encode("utf-8"))
        _tensor_digest(hasher, parameter)
    return hasher.hexdigest()


def _batch_trace_sha256(batches: Sequence[Mapping[str, Tensor]], steps: int) -> str:
    hasher = hashlib.sha256()
    for index, batch in enumerate(islice(cycle(batches), steps), start=1):
        hasher.update(str(index).encode("ascii"))
        _tensor_digest(hasher, batch["input_ids"])
        _tensor_digest(hasher, batch["labels"])
    return hasher.hexdigest()


def _percentile(values: Sequence[float], fraction: float) -> float:
    _require(bool(values), "percentile requires values")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return float(ordered[index])


def _stable_model(model: TwelveSixDecoder) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def _build_experiment_optimizer(
    model: TwelveSixDecoder,
    config: TrainerConfig,
    recipe: OptimizationRecipe,
):
    if recipe.decay_embeddings or recipe.weight_decay == 0.0:
        return None
    embedding = model.token_embedding.weight
    decay_parameters = [
        parameter for parameter in model.parameters() if parameter is not embedding
    ]
    _require(bool(decay_parameters), "no non-embedding parameters available for AdamW")
    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": config.weight_decay},
            {"params": [embedding], "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def _run_recipe(
    stage_path: Path,
    *,
    recipe: OptimizationRecipe,
    execution_steps: int,
    schedule_horizon_steps: int,
    train_batches: list[dict[str, Tensor]],
    validation_batches: list[dict[str, Tensor]],
    seed: int,
) -> dict[str, Any]:
    stage = load_stage_config(stage_path)
    config, materialized = recipe.materialize(
        schedule_horizon_steps=schedule_horizon_steps,
        seed=seed,
    )
    torch.manual_seed(seed)
    model = TwelveSixDecoder(stage.model, stage.init)
    initial_model_sha256 = _model_fingerprint(model)
    initial_validation_loss = _evaluate(model, validation_batches)
    optimizer = _build_experiment_optimizer(model, config, recipe)
    trainer = Trainer(model, config, device="cpu", optimizer=optimizer)
    model_bytes = _model_parameter_bytes(model)

    progression: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    for step_index, batch in enumerate(
        islice(cycle(train_batches), execution_steps),
        start=1,
    ):
        before = _snapshot(model)
        parameter_l2_before = _parameter_l2(model)
        step_start = time.perf_counter()
        try:
            metrics = trainer.train_microbatch(batch)
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            failure = {
                "step": step_index,
                "exception_type": type(exc).__name__,
            }
            break
        step_wall_seconds = time.perf_counter() - step_start
        update = _update_metrics(
            model,
            before,
            parameter_l2_before=parameter_l2_before,
        )
        finite_model = _stable_model(model)
        grad_norm = metrics.grad_norm
        clip = recipe.gradient_clip_norm
        progression.append(
            {
                "step": step_index,
                "loss": metrics.loss,
                "update_loss": metrics.update_loss,
                "learning_rate": metrics.learning_rate,
                "grad_norm": grad_norm,
                "clip_would_activate": (
                    clip is not None and grad_norm is not None and grad_norm > clip
                ),
                "step_wall_seconds": step_wall_seconds,
                "optimizer_state_tensor_bytes": _optimizer_state_tensor_bytes(trainer),
                "model_parameters_finite": finite_model,
                **update,
            }
        )
        if not finite_model:
            failure = {
                "step": step_index,
                "exception_type": "NonFiniteModelParameters",
            }
            break

    wall_seconds = time.perf_counter() - wall_start
    process_cpu_seconds = time.process_time() - cpu_start
    final_validation_loss = _evaluate(model, validation_batches) if progression else None
    grad_norms = [
        float(item["grad_norm"])
        for item in progression
        if item["grad_norm"] is not None
    ]
    relative_updates = [float(item["relative_update_l2"]) for item in progression]
    step_times = [float(item["step_wall_seconds"]) for item in progression]
    optimizer_bytes = [
        int(item["optimizer_state_tensor_bytes"]) for item in progression
    ]
    losses = [float(item["loss"]) for item in progression]
    status = "PASS" if failure is None and len(progression) == execution_steps else "FAIL"

    summary: dict[str, Any] = {
        "status": status,
        "steps_requested": execution_steps,
        "steps_completed": len(progression),
        "initial_validation_loss": initial_validation_loss,
        "final_validation_loss": final_validation_loss,
        "training_loss_first": losses[0] if losses else None,
        "training_loss_last": losses[-1] if losses else None,
        "gradient_norm_min": min(grad_norms) if grad_norms else None,
        "gradient_norm_max": max(grad_norms) if grad_norms else None,
        "gradient_norm_median": statistics.median(grad_norms) if grad_norms else None,
        "clip_activation_count": sum(
            bool(item["clip_would_activate"]) for item in progression
        ),
        "relative_update_l2_median": (
            statistics.median(relative_updates) if relative_updates else None
        ),
        "relative_update_l2_max": max(relative_updates) if relative_updates else None,
        "step_wall_seconds_median": statistics.median(step_times) if step_times else None,
        "step_wall_seconds_p95": _percentile(step_times, 0.95) if step_times else None,
        "optimizer_state_tensor_bytes_final": optimizer_bytes[-1] if optimizer_bytes else 0,
        "optimizer_state_bytes_per_parameter": (
            optimizer_bytes[-1] / stage.expected_parameters if optimizer_bytes else 0.0
        ),
        "model_parameter_bytes": model_bytes,
        "optimizer_group_weight_decays": [
            float(group["weight_decay"]) for group in trainer.optimizer.param_groups
        ],
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": process_cpu_seconds,
    }
    return {
        "recipe": materialized,
        "recipe_sha256": _canonical_hash(materialized),
        "initial_model_sha256": initial_model_sha256,
        "batch_trace_sha256": _batch_trace_sha256(train_batches, execution_steps),
        "summary": summary,
        "failure": failure,
        "progression": progression,
    }


def _stage_identity(root: Path, stage_name: str) -> tuple[Path, dict[str, Any]]:
    expectation = _STAGE_EXPECTATIONS[stage_name]
    path = root / expectation["path"]
    stage = load_stage_config(path)
    _require(stage.stage == stage_name, f"{stage_name}: wrong stage config")
    _require(
        stage.expected_parameters == expectation["parameter_count"],
        f"{stage_name}: parameter-count drift",
    )
    _require(
        stage.model.identity_sha256() == expectation["modelspec_sha256"],
        f"{stage_name}: ModelSpec drift",
    )
    _require(stage.init.identity_sha256() == INIT_SPEC_SHA256, f"{stage_name}: InitSpec drift")
    return path, {
        "stage": stage_name,
        "stage_config_path": expectation["path"],
        "stage_config_file_sha256": _sha256_file(path),
        "modelspec_sha256": stage.model.identity_sha256(),
        "initspec_sha256": stage.init.identity_sha256(),
        "parameter_count": stage.expected_parameters,
        "model_vocab_size": stage.model.vocab_size,
        "max_seq_len": stage.model.max_seq_len,
    }


def _summary_view(evidence: Mapping[str, Any]) -> dict[str, Any]:
    stages = evidence["stages"]
    return {
        "schema_version": evidence["schema_version"],
        "authority": evidence["authority"],
        "source_sha": evidence["identity"]["source_sha"],
        "evidence_sha256": evidence["evidence_sha256"],
        "stages": {
            stage_name: {
                recipe_name: result["summary"]
                for recipe_name, result in stage["results"].items()
            }
            for stage_name, stage in stages.items()
        },
    }


def run_optimization_experiments(
    root: str | Path,
    *,
    source_sha: str,
    locked_environment_evidence: Mapping[str, Any],
    plan_path: str | Path = "configs/runs/optimizer_experiments.experimental.json",
    seed: int = 1337,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the committed small AdamW experiment matrix on S1/S2/S3."""
    _require(
        _GIT_SHA.fullmatch(source_sha) is not None,
        "source SHA must be full lowercase Git SHA",
    )
    _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, "seed invalid")
    root = Path(root).resolve()
    plan_file = Path(plan_path)
    if not plan_file.is_absolute():
        plan_file = root / plan_file
    plan = load_experiment_plan(plan_file)
    environment = validate_locked_environment_evidence(
        locked_environment_evidence,
        source_sha=source_sha,
    )
    tokenizer = ByteTokenizer()
    _require(tokenizer.vocab_size == 256, "controlled byte-tokenizer drift")

    identity = {
        "repository": REPOSITORY,
        "source_sha": source_sha,
        "plan_path": str(plan_file.relative_to(root)),
        "plan_file_sha256": plan["file_sha256"],
        "environment": environment,
        "fixture": {
            "purpose": FIXTURE_PURPOSE,
            "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
            "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "tokenizer_vocab_sha256": TOKENIZER_VOCAB_SHA256,
            "packing_config_sha256": PACKING_CONFIG_SHA256,
            "tokenizer_vocab_size": tokenizer.vocab_size,
        },
    }
    stage_outputs: dict[str, Any] = {}
    for stage_plan in plan["raw"]["stages"]:
        stage_name = stage_plan["stage"]
        stage_path, stage_identity = _stage_identity(root, stage_name)
        _require(
            stage_plan["sequence_length"] <= stage_identity["max_seq_len"],
            f"{stage_name}: sequence length exceeds model context",
        )
        train_batches, train_ids, train_tokens, train_max_id = _tensor_batches(
            root,
            split="train",
            tokenizer=tokenizer,
            batch_size=stage_plan["batch_size"],
            sequence_length=stage_plan["sequence_length"],
        )
        validation_batches, validation_ids, validation_tokens, validation_max_id = (
            _tensor_batches(
                root,
                split="validation",
                tokenizer=tokenizer,
                batch_size=stage_plan["batch_size"],
                sequence_length=stage_plan["sequence_length"],
            )
        )
        _require(not (set(train_ids) & set(validation_ids)), f"{stage_name}: split overlap")
        _require(
            max(train_max_id, validation_max_id) < stage_identity["model_vocab_size"],
            f"{stage_name}: fixture token exceeds model vocabulary",
        )
        results: dict[str, Any] = {}
        for recipe_name in stage_plan["recipes"]:
            recipe = plan["recipes"][recipe_name]
            results[recipe_name] = _run_recipe(
                stage_path,
                recipe=recipe,
                execution_steps=stage_plan["execution_steps"],
                schedule_horizon_steps=stage_plan["schedule_horizon_steps"],
                train_batches=train_batches,
                validation_batches=validation_batches,
                seed=seed,
            )
        initial_hashes = {result["initial_model_sha256"] for result in results.values()}
        batch_hashes = {result["batch_trace_sha256"] for result in results.values()}
        _require(len(initial_hashes) == 1, f"{stage_name}: recipes did not share initialization")
        _require(len(batch_hashes) == 1, f"{stage_name}: recipes did not share batch order")
        stage_outputs[stage_name] = {
            "identity": stage_identity,
            "experiment": {
                "execution_steps": stage_plan["execution_steps"],
                "schedule_horizon_steps": stage_plan["schedule_horizon_steps"],
                "batch_size": stage_plan["batch_size"],
                "sequence_length": stage_plan["sequence_length"],
                "train_scoreable_tokens_per_epoch": train_tokens,
                "validation_scoreable_tokens": validation_tokens,
                "train_record_ids": list(train_ids),
                "validation_record_ids": list(validation_ids),
                "shared_initial_model_sha256": next(iter(initial_hashes)),
                "shared_batch_trace_sha256": next(iter(batch_hashes)),
            },
            "results": results,
        }

    evidence: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "authority": AUTHORITY,
        "identity": identity,
        "identity_sha256": _canonical_hash(identity),
        "seed": seed,
        "stages": stage_outputs,
        "unexecuted_transfer_targets": plan["raw"]["unexecuted_transfer_targets"],
        "claims": {
            "later_stage_architecture_frozen": False,
            "later_stage_corpus_or_tokenizer_frozen": False,
            "quality_or_capability_evidence": False,
            "hyperparameters_finalized": False,
            "alternative_optimizer_tested": False,
            "muon_tested": False,
            "paid_compute_authorized_or_used": False,
            "candidate_or_stable_promotion": False,
            "s4_100m_executed": False,
        },
        "runtime": {
            "device": "cpu",
            "precision": "fp32",
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    }
    evidence["evidence_sha256"] = _canonical_hash(evidence)
    validate_optimization_evidence(evidence)
    return evidence, _summary_view(evidence)


def validate_optimization_evidence(evidence: Mapping[str, Any]) -> None:
    """Fail closed on identity drift, incomplete measurements, or overclaim."""
    _require(evidence.get("schema_version") == EVIDENCE_SCHEMA_VERSION, "wrong evidence schema")
    _require(evidence.get("authority") == AUTHORITY, "wrong evidence authority")
    identity = evidence.get("identity")
    _require(isinstance(identity, Mapping), "identity missing")
    _require(identity.get("repository") == REPOSITORY, "repository mismatch")
    source_sha = identity.get("source_sha")
    _require(
        isinstance(source_sha, str) and _GIT_SHA.fullmatch(source_sha) is not None,
        "invalid source SHA",
    )
    _require(evidence.get("identity_sha256") == _canonical_hash(identity), "identity hash mismatch")
    stages = evidence.get("stages")
    _require(isinstance(stages, Mapping), "stages missing")
    _require(set(stages) == {"S1", "S2", "S3"}, "evidence must contain S1/S2/S3")
    for stage_name, stage_output in stages.items():
        _require(isinstance(stage_output, Mapping), f"{stage_name}: output missing")
        stage_identity = stage_output.get("identity")
        _require(isinstance(stage_identity, Mapping), f"{stage_name}: identity missing")
        expected = _STAGE_EXPECTATIONS[stage_name]
        _require(
            stage_identity.get("modelspec_sha256") == expected["modelspec_sha256"],
            f"{stage_name}: ModelSpec mismatch",
        )
        _require(
            stage_identity.get("parameter_count") == expected["parameter_count"],
            f"{stage_name}: parameter count mismatch",
        )
        results = stage_output.get("results")
        _require(isinstance(results, Mapping) and results, f"{stage_name}: results missing")
        for recipe_name, result in results.items():
            _require(isinstance(result, Mapping), f"{stage_name}/{recipe_name}: result missing")
            summary = result.get("summary")
            _require(isinstance(summary, Mapping), f"{stage_name}/{recipe_name}: summary missing")
            _require(
                summary.get("status") in {"PASS", "FAIL"},
                f"{stage_name}/{recipe_name}: invalid status",
            )
            _require(
                isinstance(result.get("progression"), list),
                f"{stage_name}/{recipe_name}: progression missing",
            )
            if summary.get("status") == "PASS":
                _require(
                    summary.get("steps_completed") == summary.get("steps_requested"),
                    f"{stage_name}/{recipe_name}: incomplete PASS",
                )
                _require(
                    summary.get("optimizer_state_tensor_bytes_final", 0) > 0,
                    f"{stage_name}/{recipe_name}: optimizer memory not measured",
                )
                _require(
                    summary.get("relative_update_l2_median", 0.0) > 0.0,
                    f"{stage_name}/{recipe_name}: update magnitude not measured",
                )
                _require(
                    math.isfinite(float(summary["gradient_norm_max"])),
                    f"{stage_name}/{recipe_name}: non-finite gradient norm",
                )
    transfer = evidence.get("unexecuted_transfer_targets")
    _require(isinstance(transfer, list) and transfer, "transfer targets missing")
    _require(
        all(isinstance(item, Mapping) and item.get("executed") is False for item in transfer),
        "unexecuted transfer target became executed",
    )
    claims = evidence.get("claims")
    _require(isinstance(claims, Mapping), "claims missing")
    prohibited_true = {
        "later_stage_architecture_frozen",
        "later_stage_corpus_or_tokenizer_frozen",
        "quality_or_capability_evidence",
        "hyperparameters_finalized",
        "alternative_optimizer_tested",
        "muon_tested",
        "paid_compute_authorized_or_used",
        "candidate_or_stable_promotion",
        "s4_100m_executed",
    }
    _require(
        not any(claims.get(name) is True for name in prohibited_true),
        "evidence overclaims experiment authority",
    )
    supplied_hash = evidence.get("evidence_sha256")
    without_hash = dict(evidence)
    without_hash.pop("evidence_sha256", None)
    _require(supplied_hash == _canonical_hash(without_hash), "evidence self-hash mismatch")
