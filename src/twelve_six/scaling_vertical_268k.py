"""Deep fixed-control 95K vs 268K Base-training vertical built on RESEARCH41."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .checkpoint import (
    CheckpointIdentity,
    hash_json,
    load_trainer_checkpoint,
    save_trainer_checkpoint,
)
from .model import InitSpec, TwelveSixDecoder
from .scaling_experiment import (
    PACKING_ID,
    TOKENIZER_ID,
    _byte_stream,
    _canonical_hash,
    _file_sha256,
    _git_head,
    _make_batch,
    _read_jsonl,
    _trainer_config,
    _validation_loss,
    controlled_specs,
)
from .tokenization import BYTE_TOKENIZER_HASH, BYTE_VOCAB_HASH, ByteTokenizer
from .training import Trainer

SCHEMA = "12-6.learn02.fixed-control-268k.v1"
CONFIG_SCHEMA = "12-6.learn02.fixed-control-268k-config.v1"
AUTHORITY = "LOCAL_FREE_FIXED_CONTROL_EVIDENCE_NOT_PROMOTION_OR_CAPABILITY_AUTHORITY"
INCUMBENT_PR = 162
INCUMBENT_HEAD_AT_FORK = "90c134720760fcf207525d5c111fa2583bbfff3f"
TARGET_PARAMETERS = (95_568, 267_912)


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _state_tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_state_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_state_tensor_bytes(item) for item in value)
    return 0


def _clone_parameters(model: TwelveSixDecoder) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().float().clone() for parameter in model.parameters())


def _update_weight_stats(
    model: TwelveSixDecoder,
    before: tuple[torch.Tensor, ...],
) -> dict[str, float]:
    delta_sq = 0.0
    before_sq = 0.0
    after_sq = 0.0
    for parameter, previous in zip(model.parameters(), before, strict=True):
        current = parameter.detach().float()
        delta_sq += float(torch.sum((current - previous) ** 2).item())
        before_sq += float(torch.sum(previous**2).item())
        after_sq += float(torch.sum(current**2).item())
    delta_l2 = math.sqrt(delta_sq)
    weight_l2_before = math.sqrt(before_sq)
    return {
        "update_l2": delta_l2,
        "weight_l2_before": weight_l2_before,
        "weight_l2_after": math.sqrt(after_sq),
        "update_weight_ratio": delta_l2 / max(weight_l2_before, 1e-30),
    }


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"unsupported LEARN02 config schema: {payload.get('schema')!r}")
    if tuple(int(value) for value in payload["model_parameters"]) != TARGET_PARAMETERS:
        raise ValueError(f"model_parameters must remain {TARGET_PARAMETERS!r}")
    budgets = tuple(int(value) for value in payload["token_budgets"])
    if not budgets or budgets != tuple(sorted(set(budgets))) or budgets[0] <= 0:
        raise ValueError("token_budgets must be positive, unique, and increasing")
    if int(payload["batch_size"]) != 4 or int(payload["sequence_length"]) != 64:
        raise ValueError("LEARN02 preserves the RESEARCH41 batch trace: batch=4, sequence=64")
    if int(payload["seed"]) != 1337:
        raise ValueError("LEARN02 preserves the RESEARCH41 seed=1337")
    if int(payload["torch_threads"]) <= 0:
        raise ValueError("torch_threads must be positive")
    return payload


def _split_identity(train_path: Path, validation_path: Path) -> str:
    return hash_json(
        {
            "train_jsonl_sha256": _file_sha256(train_path),
            "validation_jsonl_sha256": _file_sha256(validation_path),
        }
    )


def _checkpoint_identity(
    *,
    source_sha: str,
    spec: Any,
    init_spec: InitSpec,
    trainer: Trainer,
    dataset_manifest_hash: str,
    split_identity: str,
    run_manifest_hash: str,
    seed: int,
    sequence_length: int,
    batch_size: int,
) -> CheckpointIdentity:
    config = asdict(trainer.config)
    training_config = {
        "training": {
            **config,
            "context_length": spec.max_seq_len,
            "training_sequence_length": sequence_length,
            "batch_size": batch_size,
        },
        "data": {
            "tokenizer_version": TOKENIZER_ID,
            "split_identity": split_identity,
            "packing_version": PACKING_ID,
            "packing_sha256": hash_json(
                {
                    "packing_id": PACKING_ID,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                }
            ),
        },
        "init_spec": init_spec.to_dict(),
        "init_spec_sha256": init_spec.identity_sha256(),
    }
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        dataset_manifest_hash=dataset_manifest_hash,
        run_manifest_hash=run_manifest_hash,
        training_config=training_config,
        seed=seed,
        precision=trainer.config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={
            "name": "AdamW",
            "learning_rate": trainer.config.learning_rate,
            "betas": list(trainer.config.betas),
            "eps": trainer.config.eps,
            "weight_decay": trainer.config.weight_decay,
        },
        scheduler=None,
    )


def _reload_checkpoint(
    *,
    checkpoint_dir: Path,
    spec: Any,
    init_spec: InitSpec,
    trainer_config: Any,
    source_sha: str,
    dataset_manifest_hash: str,
    run_manifest_hash: str,
    seed: int,
) -> tuple[TwelveSixDecoder, Trainer]:
    fresh_model = TwelveSixDecoder(spec, init_spec)
    fresh_trainer = Trainer(fresh_model, trainer_config, device="cpu")
    load_trainer_checkpoint(
        checkpoint_dir,
        model=fresh_model,
        trainer=fresh_trainer,
        restore_rng=True,
        expected_git_sha=source_sha,
        expected_model_spec_hash=spec.identity_sha256(),
        expected_tokenizer_hash=BYTE_TOKENIZER_HASH,
        expected_tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        expected_dataset_manifest_hash=dataset_manifest_hash,
        expected_run_manifest_hash=run_manifest_hash,
        expected_seed=seed,
    )
    return fresh_model, fresh_trainer


def _compare_model_runs(model_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if [run["parameters"] for run in model_runs] != list(TARGET_PARAMETERS):
        raise ValueError("comparison requires the exact 95,568 vs 267,912 family")
    small, large = model_runs
    if len(small["checkpoints"]) != len(large["checkpoints"]):
        raise ValueError("model checkpoint counts differ")
    comparisons: list[dict[str, Any]] = []
    for small_point, large_point in zip(
        small["checkpoints"], large["checkpoints"], strict=True
    ):
        if small_point["requested_token_budget"] != large_point["requested_token_budget"]:
            raise ValueError("requested token checkpoints differ")
        if small_point["optimized_tokens"] != large_point["optimized_tokens"]:
            raise ValueError("optimized tokens differ at equal-token comparison")
        small_loss = float(small_point["validation_loss"])
        large_loss = float(large_point["validation_loss"])
        reduction = small_loss - large_loss
        comparisons.append(
            {
                "requested_token_budget": small_point["requested_token_budget"],
                "optimized_tokens": small_point["optimized_tokens"],
                "validation_loss_95568": small_loss,
                "validation_loss_267912": large_loss,
                "absolute_validation_loss_reduction_267912_vs_95568": reduction,
                "relative_validation_loss_reduction_267912_vs_95568": reduction
                / small_loss,
                "bits_per_byte_95568": small_point["bits_per_byte"],
                "bits_per_byte_267912": large_point["bits_per_byte"],
                "winner": 267_912 if reduction > 0 else 95_568,
            }
        )
    return comparisons


def _run_one_model(
    *,
    spec: Any,
    init_spec: InitSpec,
    trainer_config: Any,
    train_stream: bytes,
    validation_records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    budgets: tuple[int, ...],
    batch_size: int,
    sequence_length: int,
    seed: int,
    checkpoint_root: Path,
    repo_root: Path,
    source_sha: str,
    dataset_manifest_hash: str,
    split_identity: str,
    run_manifest_hash: str,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    trainer = Trainer(model, trainer_config, device="cpu")
    initial_loss, validation_tokens = _validation_loss(model, validation_records, tokenizer)
    before = _clone_parameters(model)
    checkpoints: list[dict[str, Any]] = []
    step_trace: list[dict[str, Any]] = []
    next_budget = 0
    started = time.perf_counter()
    training_seconds = 0.0

    while trainer.optimizer_step < trainer_config.max_steps:
        batch = _make_batch(
            train_stream,
            step=trainer.optimizer_step,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        step_started = time.perf_counter()
        metrics = trainer.train_microbatch({"input_ids": batch})
        training_seconds += time.perf_counter() - step_started
        update_stats = _update_weight_stats(model, before)
        before = _clone_parameters(model)
        step_trace.append(
            {
                "optimizer_step": metrics.optimizer_step,
                "optimized_tokens": trainer.tokens_seen,
                "train_loss": metrics.update_loss,
                "grad_norm": metrics.grad_norm,
                **update_stats,
            }
        )

        while next_budget < len(budgets) and trainer.tokens_seen >= budgets[next_budget]:
            requested = budgets[next_budget]
            validation_loss, checked_tokens = _validation_loss(
                model, validation_records, tokenizer
            )
            if checked_tokens != validation_tokens:
                raise RuntimeError("validation token count drifted")
            checkpoint_dir = (
                checkpoint_root
                / f"params-{spec.parameter_count()}"
                / f"tokens-{trainer.tokens_seen}"
            )
            identity = _checkpoint_identity(
                source_sha=source_sha,
                spec=spec,
                init_spec=init_spec,
                trainer=trainer,
                dataset_manifest_hash=dataset_manifest_hash,
                split_identity=split_identity,
                run_manifest_hash=run_manifest_hash,
                seed=seed,
                sequence_length=sequence_length,
                batch_size=batch_size,
            )
            checkpoint_started = time.perf_counter()
            manifest = save_trainer_checkpoint(
                checkpoint_dir,
                model=model,
                trainer=trainer,
                identity=identity,
            )
            checkpoint_seconds = time.perf_counter() - checkpoint_started

            original = _clone_parameters(model)
            model, trainer = _reload_checkpoint(
                checkpoint_dir=checkpoint_dir,
                spec=spec,
                init_spec=init_spec,
                trainer_config=trainer_config,
                source_sha=source_sha,
                dataset_manifest_hash=dataset_manifest_hash,
                run_manifest_hash=run_manifest_hash,
                seed=seed,
            )
            reloaded = _clone_parameters(model)
            reload_diff = max(
                float(torch.max(torch.abs(a - b)).item())
                for a, b in zip(original, reloaded, strict=True)
            )
            if reload_diff != 0.0:
                raise RuntimeError("checkpoint reload changed model weights")
            before = _clone_parameters(model)
            latest = step_trace[-1]
            checkpoints.append(
                {
                    "requested_token_budget": requested,
                    "optimized_tokens": trainer.tokens_seen,
                    "optimizer_steps": trainer.optimizer_step,
                    "train_loss": latest["train_loss"],
                    "validation_loss": validation_loss,
                    "perplexity": math.exp(validation_loss),
                    "bits_per_byte": validation_loss / math.log(2.0),
                    "grad_norm": latest["grad_norm"],
                    "update_l2": latest["update_l2"],
                    "weight_l2_before": latest["weight_l2_before"],
                    "update_weight_ratio": latest["update_weight_ratio"],
                    "elapsed_seconds": time.perf_counter() - started,
                    "training_seconds": training_seconds,
                    "optimized_tokens_per_training_second": trainer.tokens_seen
                    / training_seconds,
                    "peak_rss_kib": _rss_kib(),
                    "parameter_tensor_bytes": sum(
                        parameter.numel() * parameter.element_size()
                        for parameter in model.parameters()
                    ),
                    "optimizer_tensor_bytes": _state_tensor_bytes(
                        trainer.optimizer.state_dict()
                    ),
                    "checkpoint_seconds": checkpoint_seconds,
                    "checkpoint_id": manifest["checkpoint_id"],
                    "checkpoint_path": str(checkpoint_dir.relative_to(repo_root)),
                    "reload_max_abs_diff": reload_diff,
                }
            )
            next_budget += 1

    if next_budget != len(budgets):
        raise RuntimeError("training ended before all LEARN02 checkpoints")
    return {
        "parameters": spec.parameter_count(),
        "model_spec": spec.to_dict(),
        "model_spec_sha256": spec.identity_sha256(),
        "initial_validation_loss": initial_loss,
        "initial_bits_per_byte": initial_loss / math.log(2.0),
        "validation_tokens": validation_tokens,
        "checkpoints": checkpoints,
        "step_trace": step_trace,
    }


def run_vertical(
    *,
    repo_root: Path,
    source_sha: str,
    config_path: Path,
    output_path: Path,
    checkpoint_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("LEARN02 requires an exact source checkout")
    config = _load_config(config_path)
    budgets = tuple(int(value) for value in config["token_budgets"])
    batch_size = int(config["batch_size"])
    sequence_length = int(config["sequence_length"])
    seed = int(config["seed"])
    torch_threads = int(config["torch_threads"])
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)

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
        raise RuntimeError(f"train/validation overlap: {overlap!r}")

    specs_by_count = {spec.parameter_count(): spec for spec in controlled_specs()}
    specs = tuple(specs_by_count[count] for count in TARGET_PARAMETERS)
    init_spec = InitSpec()
    tokens_per_step = batch_size * (sequence_length - 1)
    max_steps = math.ceil(budgets[-1] / tokens_per_step)
    trainer_config = _trainer_config(max_steps=max_steps, seed=seed)
    dataset_manifest_hash = _file_sha256(manifest_path)
    split_identity = _split_identity(train_path, validation_path)
    run_identity = {
        "schema": CONFIG_SCHEMA,
        "incumbent_pr": INCUMBENT_PR,
        "incumbent_head_at_fork": INCUMBENT_HEAD_AT_FORK,
        "source_sha": source_sha,
        "config": config,
        "model_spec_sha256": [spec.identity_sha256() for spec in specs],
        "init_spec_sha256": init_spec.identity_sha256(),
        "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
        "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
        "dataset_manifest_sha256": dataset_manifest_hash,
        "split_identity": split_identity,
        "packing_id": PACKING_ID,
        "trainer_config": asdict(trainer_config),
    }
    run_manifest_hash = hash_json(run_identity)
    train_stream = _byte_stream(train_records, tokenizer)

    model_runs = [
        _run_one_model(
            spec=spec,
            init_spec=init_spec,
            trainer_config=trainer_config,
            train_stream=train_stream,
            validation_records=validation_records,
            tokenizer=tokenizer,
            budgets=budgets,
            batch_size=batch_size,
            sequence_length=sequence_length,
            seed=seed,
            checkpoint_root=checkpoint_root,
            repo_root=repo_root,
            source_sha=source_sha,
            dataset_manifest_hash=dataset_manifest_hash,
            split_identity=split_identity,
            run_manifest_hash=run_manifest_hash,
        )
        for spec in specs
    ]
    comparisons = _compare_model_runs(model_runs)
    earlier_improvements = [
        point["absolute_validation_loss_reduction_267912_vs_95568"] > 0
        for point in comparisons[:-1]
    ]
    final_improves = (
        comparisons[-1]["absolute_validation_loss_reduction_267912_vs_95568"] > 0
    )
    larger_best = min(
        model_runs[1]["checkpoints"], key=lambda point: point["validation_loss"]
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
            "incumbent_pr": INCUMBENT_PR,
            "incumbent_head_at_fork": INCUMBENT_HEAD_AT_FORK,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": torch_threads,
            "paid_compute": False,
            "foreign_weights": False,
        },
        "controls": {
            "config": config,
            "run_manifest_sha256": run_manifest_hash,
            "tokenizer_id": TOKENIZER_ID,
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "init_spec": init_spec.to_dict(),
            "init_spec_sha256": init_spec.identity_sha256(),
            "packing_id": PACKING_ID,
            "trainer_config": asdict(trainer_config),
            "tokens_per_optimizer_step": tokens_per_step,
            "changed_variable": "model_geometry_only",
        },
        "data": {
            "dataset_manifest_sha256": dataset_manifest_hash,
            "split_identity_sha256": split_identity,
            "train_jsonl_sha256": _file_sha256(train_path),
            "validation_jsonl_sha256": _file_sha256(validation_path),
            "train_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "train_validation_record_overlap": overlap,
            "project_authored_fixture": True,
            "repeated_fixture": True,
        },
        "metric_semantics": {
            "loss": "mean next-byte cross entropy in nats",
            "perplexity": "exp(next-byte cross entropy)",
            "bits_per_byte": "next-byte cross entropy / ln(2) for s0-byte-v1",
            "grad_norm": "Trainer pre-clip gradient L2 norm after token normalization",
            "update_weight_ratio": "L2(parameter update) / L2(parameters before update)",
        },
        "model_runs": model_runs,
        "equal_token_comparison": comparisons,
        "decision": {
            "larger_model_improves_at_final_equal_token_budget": final_improves,
            "larger_model_improves_at_all_prior_measured_budgets": all(
                earlier_improvements
            ),
            "overfit_reversal_observed": any(earlier_improvements)
            and not final_improves,
            "best_267912_checkpoint": {
                "requested_token_budget": larger_best["requested_token_budget"],
                "optimized_tokens": larger_best["optimized_tokens"],
                "validation_loss": larger_best["validation_loss"],
                "bits_per_byte": larger_best["bits_per_byte"],
            },
        },
        "truth_boundary": {
            "broad_language_capability_claim": False,
            "large_corpus_scaling_claim": False,
            "promotion_authority": False,
            "paid_compute_authority": False,
            "scientific_scope": (
                "Geometry comparison on the tiny project-authored S0 fixture only. "
                "Repeated cycling can expose overfit but cannot establish broad capability."
            ),
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def validate_report(report: dict[str, Any], expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("invalid LEARN02 report schema/authority")
    if expected_source_sha is not None and report["source"]["git_sha"] != expected_source_sha:
        raise ValueError("LEARN02 report source SHA mismatch")
    observed_hash = report.get("report_sha256")
    unhashed = dict(report)
    unhashed.pop("report_sha256", None)
    if observed_hash != _canonical_hash(unhashed):
        raise ValueError("LEARN02 report self-hash mismatch")
    if [run["parameters"] for run in report["model_runs"]] != list(TARGET_PARAMETERS):
        raise ValueError("LEARN02 model family drifted")
    if not report["equal_token_comparison"]:
        raise ValueError("LEARN02 report has no equal-token comparisons")
    for run in report["model_runs"]:
        for point in run["checkpoints"]:
            if point["reload_max_abs_diff"] != 0.0:
                raise ValueError("checkpoint reload equivalence failed")
            for field in ("validation_loss", "grad_norm", "update_weight_ratio"):
                if not math.isfinite(float(point[field])):
                    raise ValueError(f"non-finite {field}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--source-sha", required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoint-root", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "run":
        report = run_vertical(
            repo_root=args.repo_root,
            source_sha=args.source_sha,
            config_path=args.config,
            output_path=args.output,
            checkpoint_root=args.checkpoint_root,
        )
        print(json.dumps(report["decision"], indent=2, sort_keys=True))
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_report(report, expected_source_sha=args.expected_source_sha)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
