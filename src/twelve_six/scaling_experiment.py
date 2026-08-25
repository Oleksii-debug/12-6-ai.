"""Controlled LOCAL_FREE empirical scaling experiments for 12-6 AI.

This module studies a compact fixed-tokenizer/fixed-data family. The output is
research evidence, not a stage freeze, promotion gate, or paid-compute authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import subprocess
import sys
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
from .training import Trainer, TrainerConfig

SCHEMA = "12-6.scaling-experiment.v1"
AUTHORITY = "LOCAL_FREE_EMPIRICAL_SCALING_EVIDENCE_NOT_PROMOTION_OR_BUDGET_AUTHORIZATION"
TOKENIZER_ID = BYTE_TOKENIZER_VERSION
PACKING_ID = "research41-byte-stream-cyclic-v1"
COMPUTE_PROXY = "6 * trainable_parameters * optimized_tokens"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

_DEFAULT_GEOMETRIES: tuple[dict[str, int], ...] = (
    {
        "d_model": 48,
        "n_layers": 3,
        "n_heads": 4,
        "n_kv_heads": 4,
        "head_dim": 12,
        "d_ff": 128,
    },
    {
        "d_model": 72,
        "n_layers": 4,
        "n_heads": 6,
        "n_kv_heads": 6,
        "head_dim": 12,
        "d_ff": 192,
    },
    {
        "d_model": 96,
        "n_layers": 4,
        "n_heads": 6,
        "n_kv_heads": 6,
        "head_dim": 16,
        "d_ff": 256,
    },
    {
        "d_model": 128,
        "n_layers": 5,
        "n_heads": 8,
        "n_kv_heads": 8,
        "head_dim": 16,
        "d_ff": 352,
    },
)


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        text = value.get("text")
        record_id = value.get("id")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number} has invalid text")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{path}:{line_number} has invalid id")
        records.append(value)
    if not records:
        raise ValueError(f"{path} contains no records")
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate record ids")
    return records


def _byte_stream(records: list[dict[str, Any]], tokenizer: ByteTokenizer) -> bytes:
    encoded = [bytes(tokenizer.encode(str(record["text"]))) for record in records]
    return b"\n".join(encoded) + b"\n"


def _make_batch(
    stream: bytes,
    *,
    step: int,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    if not stream:
        raise ValueError("training stream must be non-empty")
    width = batch_size * sequence_length
    base = (step * width) % len(stream)
    rows: list[list[int]] = []
    for batch_index in range(batch_size):
        start = (base + batch_index * sequence_length) % len(stream)
        rows.append([stream[(start + offset) % len(stream)] for offset in range(sequence_length)])
    return torch.tensor(rows, dtype=torch.long)


def _model_spec(geometry: dict[str, int]) -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=256,
        d_model=geometry["d_model"],
        n_layers=geometry["n_layers"],
        n_heads=geometry["n_heads"],
        n_kv_heads=geometry["n_kv_heads"],
        head_dim=geometry["head_dim"],
        d_ff=geometry["d_ff"],
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10_000.0,
        rope_rotary_dim=geometry["head_dim"],
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )


def controlled_specs() -> tuple[ModelSpec, ...]:
    """Return the fixed-vocabulary/context family used by RESEARCH41."""
    specs = tuple(_model_spec(geometry) for geometry in _DEFAULT_GEOMETRIES)
    counts = tuple(spec.parameter_count() for spec in specs)
    expected = (95_568, 267_912, 467_808, 1_037_696)
    if counts != expected:
        raise RuntimeError(f"controlled parameter-family drift: {counts!r} != {expected!r}")
    if {spec.vocab_size for spec in specs} != {256}:
        raise RuntimeError("controlled family must keep vocab_size fixed at 256")
    if {spec.max_seq_len for spec in specs} != {256}:
        raise RuntimeError("controlled family must keep max_seq_len fixed at 256")
    return specs


def _trainer_config(*, max_steps: int, seed: int) -> TrainerConfig:
    return TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=max_steps,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=seed,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


@torch.no_grad()
def _validation_loss(
    model: TwelveSixDecoder,
    records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
) -> tuple[float, int]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    max_seq_len = model.spec.max_seq_len
    for record in records:
        token_ids = tokenizer.encode(str(record["text"]))
        start = 0
        while start < len(token_ids) - 1:
            chunk = token_ids[start : start + max_seq_len]
            if len(chunk) < 2:
                break
            input_ids = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
            logits = model(input_ids).logits
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, model.spec.vocab_size),
                input_ids[:, 1:].reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += int(input_ids.shape[1] - 1)
            start += max_seq_len - 1
    model.train(was_training)
    if total_tokens <= 0:
        raise RuntimeError("validation split produced no next-token targets")
    result = total_loss / total_tokens
    if not math.isfinite(result):
        raise RuntimeError("validation loss is non-finite")
    return result, total_tokens


def _fit_log_plane(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit log(loss) = b0 + bN*log(N) + bT*log(T) inside the measured box only."""
    if len(points) < 6:
        raise ValueError("at least six observations are required for the empirical fit")
    rows: list[list[float]] = []
    target: list[float] = []
    for point in points:
        parameters = int(point["parameters"])
        tokens = int(point["optimized_tokens"])
        loss = float(point["validation_loss"])
        if parameters <= 0 or tokens <= 0 or not math.isfinite(loss) or loss <= 0:
            raise ValueError("fit observations must be positive and finite")
        rows.append([1.0, math.log(parameters), math.log(tokens)])
        target.append(math.log(loss))
    design = torch.tensor(rows, dtype=torch.float64)
    observed = torch.tensor(target, dtype=torch.float64)
    solution = torch.linalg.lstsq(design, observed).solution
    predicted = design @ solution
    residual = observed - predicted
    ss_residual = float(torch.sum(residual * residual).item())
    centered = observed - torch.mean(observed)
    ss_total = float(torch.sum(centered * centered).item())
    r_squared = 1.0 - ss_residual / ss_total if ss_total > 0 else 1.0
    rmse = math.sqrt(ss_residual / len(points))
    return {
        "relationship": "log(validation_loss)=b0+bN*log(parameters)+bT*log(tokens)",
        "coefficients": {
            "b0": float(solution[0].item()),
            "b_parameters": float(solution[1].item()),
            "b_tokens": float(solution[2].item()),
        },
        "r_squared_log_space": r_squared,
        "rmse_log_space": rmse,
        "observed_parameter_range": [
            min(int(point["parameters"]) for point in points),
            max(int(point["parameters"]) for point in points),
        ],
        "observed_token_range": [
            min(int(point["optimized_tokens"]) for point in points),
            max(int(point["optimized_tokens"]) for point in points),
        ],
        "extrapolation_authorized": False,
    }


def _decision_signals(points: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (int(point["parameters"]), int(point["requested_token_budget"])): point
        for point in points
    }
    largest_budget = max(int(point["requested_token_budget"]) for point in points)
    largest_model = max(int(point["parameters"]) for point in points)
    sorted_models = sorted({int(point["parameters"]) for point in points})
    second_largest_model = sorted_models[-2]
    token_budgets = sorted({int(point["requested_token_budget"]) for point in points})
    middle_budget = token_budgets[-2]
    large_point = by_key[(largest_model, largest_budget)]
    prior_model_point = by_key[(second_largest_model, largest_budget)]
    middle_token_point = by_key[(largest_model, middle_budget)]
    model_gain = float(prior_model_point["validation_loss"]) - float(
        large_point["validation_loss"]
    )
    token_gain = float(middle_token_point["validation_loss"]) - float(
        large_point["validation_loss"]
    )
    return {
        "best_observed_point": {
            "parameters": largest_model,
            "requested_token_budget": largest_budget,
            "optimized_tokens": int(large_point["optimized_tokens"]),
            "validation_loss": float(large_point["validation_loss"]),
        },
        "largest_budget_model_gain": {
            "from_parameters": second_largest_model,
            "to_parameters": largest_model,
            "absolute_validation_loss_reduction": model_gain,
        },
        "largest_model_token_gain": {
            "from_requested_tokens": middle_budget,
            "to_requested_tokens": largest_budget,
            "absolute_validation_loss_reduction": token_gain,
        },
        "budget_interpretation": (
            "Use these local marginal-loss directions only to qualify the next paid pilot; "
            "do not convert this tiny repeated fixture into a euro-optimal large-model law."
        ),
    }


def run_scaling_experiment(
    *,
    repo_root: Path,
    source_sha: str,
    output_path: Path,
    token_budgets: tuple[int, ...] = (4096, 16_384, 65_536),
    batch_size: int = 4,
    sequence_length: int = 64,
    seed: int = 1337,
    torch_threads: int = 2,
) -> dict[str, Any]:
    if not _HEX40.fullmatch(source_sha):
        raise ValueError("source_sha must be a lowercase 40-hex Git SHA")
    observed_head = _git_head(repo_root)
    if observed_head != source_sha:
        raise RuntimeError(f"exact-checkout mismatch: expected {source_sha}, observed {observed_head}")
    if tuple(sorted(set(token_budgets))) != token_budgets or not token_budgets:
        raise ValueError("token_budgets must be strictly increasing and unique")
    if token_budgets[0] <= 0:
        raise ValueError("token budgets must be positive")
    if batch_size <= 0 or sequence_length < 2 or sequence_length > 256:
        raise ValueError("invalid batch_size or sequence_length")
    if torch_threads <= 0:
        raise ValueError("torch_threads must be positive")

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
        raise RuntimeError(f"train/validation record overlap: {overlap!r}")
    train_stream = _byte_stream(train_records, tokenizer)
    tokens_per_step = batch_size * (sequence_length - 1)
    max_steps = math.ceil(token_budgets[-1] / tokens_per_step)
    init_spec = InitSpec()
    trainer_config = _trainer_config(max_steps=max_steps, seed=seed)

    model_runs: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for spec in controlled_specs():
        random.seed(seed)
        torch.manual_seed(seed)
        model = TwelveSixDecoder(spec, init_spec)
        trainer = Trainer(model, trainer_config, device="cpu")
        initial_loss, validation_tokens = _validation_loss(
            model,
            validation_records,
            tokenizer,
        )
        checkpoints: list[dict[str, Any]] = []
        next_budget_index = 0
        started = time.perf_counter()
        for step in range(max_steps):
            batch = _make_batch(
                train_stream,
                step=step,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            metrics = trainer.train_microbatch({"input_ids": batch})
            while (
                next_budget_index < len(token_budgets)
                and trainer.tokens_seen >= token_budgets[next_budget_index]
            ):
                validation_loss, checked_tokens = _validation_loss(
                    model,
                    validation_records,
                    tokenizer,
                )
                if checked_tokens != validation_tokens:
                    raise RuntimeError("validation token count drifted during run")
                requested_budget = token_budgets[next_budget_index]
                point = {
                    "parameters": spec.parameter_count(),
                    "requested_token_budget": requested_budget,
                    "optimized_tokens": trainer.tokens_seen,
                    "optimizer_steps": trainer.optimizer_step,
                    "compute_proxy": 6 * spec.parameter_count() * trainer.tokens_seen,
                    "validation_loss": validation_loss,
                    "last_train_loss": metrics.loss,
                    "last_grad_norm": metrics.grad_norm,
                }
                checkpoints.append(point)
                observations.append(point)
                next_budget_index += 1
        elapsed = time.perf_counter() - started
        if next_budget_index != len(token_budgets):
            raise RuntimeError("training ended before all token checkpoints were observed")
        model_runs.append(
            {
                "model_spec": spec.to_dict(),
                "model_identity_sha256": spec.identity_sha256(),
                "parameters": spec.parameter_count(),
                "initial_validation_loss": initial_loss,
                "validation_tokens": validation_tokens,
                "wall_seconds": elapsed,
                "optimized_tokens_per_wall_second": trainer.tokens_seen / elapsed,
                "checkpoints": checkpoints,
            }
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fit = _fit_log_plane(observations)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "torch_threads": torch_threads,
            "paid_compute": False,
        },
        "controls": {
            "canonical_base": "random_init",
            "tokenizer_id": TOKENIZER_ID,
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "vocab_size": tokenizer.vocab_size,
            "model_max_seq_len": 256,
            "training_sequence_length": sequence_length,
            "batch_size": batch_size,
            "tokens_per_optimizer_step": tokens_per_step,
            "seed": seed,
            "optimizer": asdict(trainer_config),
            "packing_id": PACKING_ID,
            "token_budgets": list(token_budgets),
            "compute_proxy_definition": COMPUTE_PROXY,
        },
        "data": {
            "dataset_id": manifest.get("dataset_id"),
            "dataset_identity_sha256": manifest.get("dataset_identity_sha256"),
            "manifest_sha256": _file_sha256(manifest_path),
            "train_jsonl_sha256": _file_sha256(train_path),
            "validation_jsonl_sha256": _file_sha256(validation_path),
            "train_record_count": len(train_records),
            "validation_record_count": len(validation_records),
            "train_validation_record_overlap": overlap,
            "unique_train_stream_bytes": len(train_stream),
            "repeated_fixture": True,
            "scope_warning": (
                "The project-authored S0 fixture is intentionally tiny and is recycled to reach "
                "token checkpoints; results measure controlled local optimization/generalization "
                "on this fixture, not broad-corpus scaling."
            ),
        },
        "init_spec": init_spec.to_dict(),
        "init_identity_sha256": init_spec.identity_sha256(),
        "model_runs": model_runs,
        "observations": observations,
        "fit": fit,
        "decision_signals": _decision_signals(observations),
        "truth_boundary": {
            "stage_freeze": False,
            "promotion_authority": False,
            "paid_compute_authority": False,
            "one_trillion_parameter_extrapolation": False,
            "euro_optimum_claim": False,
            "fit_valid_only_inside_observed_box": True,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def validate_report(report: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    if set(report) != {
        "schema",
        "authority",
        "source",
        "runtime",
        "controls",
        "data",
        "init_spec",
        "init_identity_sha256",
        "model_runs",
        "observations",
        "fit",
        "decision_signals",
        "truth_boundary",
        "report_sha256",
    }:
        raise ValueError("unexpected scaling report top-level schema")
    if report["schema"] != SCHEMA or report["authority"] != AUTHORITY:
        raise ValueError("scaling report schema/authority mismatch")
    source = report["source"]
    if source.get("repository") != "Oleksii-debug/12-6-ai.":
        raise ValueError("unexpected repository identity")
    source_sha = source.get("git_sha")
    if not isinstance(source_sha, str) or not _HEX40.fullmatch(source_sha):
        raise ValueError("invalid source Git SHA")
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ValueError("scaling report source SHA mismatch")
    if report["runtime"].get("paid_compute") is not False:
        raise ValueError("scaling report may not claim paid compute")
    controls = report["controls"]
    if controls.get("tokenizer_id") != TOKENIZER_ID or controls.get("vocab_size") != 256:
        raise ValueError("controlled tokenizer/vocabulary drift")
    config_hash = controls.get("tokenizer_config_sha256")
    if config_hash is not None and config_hash != BYTE_TOKENIZER_HASH:
        raise ValueError("controlled tokenizer config identity drift")
    vocab_hash = controls.get("tokenizer_vocab_sha256")
    if vocab_hash is not None and vocab_hash != BYTE_VOCAB_HASH:
        raise ValueError("controlled tokenizer vocabulary identity drift")
    if controls.get("model_max_seq_len") != 256:
        raise ValueError("controlled context drift")
    if report["data"].get("train_validation_record_overlap") != []:
        raise ValueError("held-out record isolation failed")
    observations = report["observations"]
    if not isinstance(observations, list) or len(observations) != 12:
        raise ValueError("expected the 4-model x 3-token observation grid")
    parameter_counts = sorted({int(point["parameters"]) for point in observations})
    if parameter_counts != [95_568, 267_912, 467_808, 1_037_696]:
        raise ValueError("controlled parameter family drift")
    for point in observations:
        loss = float(point["validation_loss"])
        if not math.isfinite(loss) or loss <= 0:
            raise ValueError("invalid validation loss")
        expected_proxy = 6 * int(point["parameters"]) * int(point["optimized_tokens"])
        if int(point["compute_proxy"]) != expected_proxy:
            raise ValueError("compute proxy drift")
    truth = report["truth_boundary"]
    required_false = (
        "stage_freeze",
        "promotion_authority",
        "paid_compute_authority",
        "one_trillion_parameter_extrapolation",
        "euro_optimum_claim",
    )
    if any(truth.get(key) is not False for key in required_false):
        raise ValueError("truth boundary was weakened")
    if truth.get("fit_valid_only_inside_observed_box") is not True:
        raise ValueError("fit range boundary was weakened")
    supplied_hash = report["report_sha256"]
    unsigned = dict(report)
    unsigned.pop("report_sha256")
    if supplied_hash != _canonical_hash(unsigned):
        raise ValueError("scaling report self-hash mismatch")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--repo-root", type=Path, default=Path("."))
    run_parser.add_argument("--source-sha", required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--torch-threads", type=int, default=2)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--expected-source-sha")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        report = run_scaling_experiment(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            output_path=args.output,
            torch_threads=args.torch_threads,
        )
        validate_report(report, expected_source_sha=args.source_sha)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("report must be a JSON object")
    validate_report(report, expected_source_sha=args.expected_source_sha)
    print(f"{SCHEMA}: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
