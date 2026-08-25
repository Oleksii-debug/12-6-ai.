#!/usr/bin/env python3
"""Run EVAL-136 exposure-controlled memorization curves on LOCAL_FREE CPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from twelve_six.memorization import (
    aggregate_scores,
    build_canary_suite,
    epoch_schedule,
    hashed_training_probe,
    heldout_bpb,
    memorization_index,
    score_canary,
    stop_diagnostic,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must be a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _midpoint_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=512,
        max_seq_len=256,
        d_model=96,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        head_dim=24,
        d_ff=256,
        rope_rotary_dim=24,
    )


def _model_ladder(repo_root: Path) -> list[tuple[str, ModelSpec, InitSpec]]:
    s1 = load_stage_config(repo_root / "configs/stages/s1_100k.json")
    s2 = load_stage_config(repo_root / "configs/stages/s2_1m.json")
    mid = _midpoint_spec()
    if not 450_000 <= mid.parameter_count() <= 550_000:
        raise RuntimeError("experimental midpoint drifted outside ~500K")
    return [
        ("s1-~100k", s1.model, s1.init),
        ("eval136-~500k", mid, InitSpec()),
        ("s2-~1m", s2.model, s2.init),
    ]


def _batch_for_text(text: str, tokenizer: ByteTokenizer, max_seq_len: int) -> dict[str, Any]:
    token_ids = tokenizer.encode(text)[:max_seq_len]
    if len(token_ids) < 2:
        raise ValueError("training sequence must contain at least two tokens")
    ids = torch.tensor([token_ids], dtype=torch.long)
    return {"input_ids": ids, "labels": ids}


def _score_checkpoint(
    *,
    model: TwelveSixDecoder,
    trainer: Trainer,
    tokenizer: ByteTokenizer,
    suite: Any,
    observed_exposures: dict[str, int],
    validation_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    alternative_count: int,
    passage_sample_count: int,
    seed: int,
    previous_bpb: float | None,
) -> dict[str, Any]:
    before_state = (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen)
    before_digest = _state_digest(model)
    validation_texts = [str(row["text"]) for row in validation_rows]
    bpb = heldout_bpb(model, tokenizer, validation_texts)
    scores = [
        score_canary(
            model,
            tokenizer,
            canary,
            observed_exposures=observed_exposures.get(canary.canary_id, 0),
            alternative_count=alternative_count,
        )
        for canary in suite.canaries
    ]
    curve = aggregate_scores(scores)
    memorization = {"memorization_index": memorization_index(curve)}
    passage_metrics = hashed_training_probe(
        model, tokenizer, train_rows, sample_count=passage_sample_count, seed=seed
    )
    diagnostic = stop_diagnostic(curve, previous_bpb=previous_bpb, current_bpb=bpb)
    after_digest = _state_digest(model)
    after_state = (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen)
    if before_digest != after_digest or before_state != after_state:
        raise RuntimeError("evaluation mutated model weights or Trainer counters")
    return {
        "optimizer_step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
        "t_over_n": trainer.tokens_seen / model.spec.parameter_count(),
        "heldout_bpb": bpb,
        "canary_curve": curve,
        "canary_scores": scores,
        "memorization": memorization,
        "non_canary_training_passages": passage_metrics,
        "diagnostic": diagnostic,
        "evaluation_non_mutating": True,
        "model_state_sha256": before_digest,
    }


def _train_model_curve(
    *,
    model_id: str,
    spec: ModelSpec,
    init_spec: InitSpec,
    tokenizer: ByteTokenizer,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    suite: Any,
    checkpoint_tokens: list[int],
    learning_rate: float,
    alternative_count: int,
    passage_sample_count: int,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    target_max = max(checkpoint_tokens)
    trainer = Trainer(
        model,
        TrainerConfig(
            learning_rate=learning_rate,
            weight_decay=0.0,
            max_steps=max(1024, target_max * 2 + 32),
            scheduler="constant",
            gradient_accumulation_steps=1,
            gradient_clip_norm=1.0,
            precision="fp32",
            seed=seed,
            deterministic_algorithms=True,
            deterministic_warn_only=True,
        ),
        device="cpu",
    )
    observed = {canary.canary_id: 0 for canary in suite.canaries}
    checkpoints: list[dict[str, Any]] = []
    previous_bpb: float | None = None
    epoch = 0
    schedule: list[dict[str, Any]] = []
    cursor = 0

    for target_tokens in checkpoint_tokens:
        while trainer.tokens_seen < target_tokens:
            if cursor >= len(schedule):
                schedule = epoch_schedule(train_rows, suite, epoch=epoch, seed=seed)
                epoch += 1
                cursor = 0
            item = schedule[cursor]
            cursor += 1
            metrics = trainer.train_microbatch(
                _batch_for_text(str(item["text"]), tokenizer, spec.max_seq_len)
            )
            if not metrics.optimizer_stepped or not math.isfinite(metrics.loss):
                raise RuntimeError("training failed to complete a finite optimizer step")
            if item["kind"] == "canary":
                observed[str(item["canary_id"])] += 1

        checkpoint = _score_checkpoint(
            model=model,
            trainer=trainer,
            tokenizer=tokenizer,
            suite=suite,
            observed_exposures=observed,
            validation_rows=validation_rows,
            train_rows=train_rows,
            alternative_count=alternative_count,
            passage_sample_count=passage_sample_count,
            seed=seed,
            previous_bpb=previous_bpb,
        )
        checkpoint["target_tokens"] = target_tokens
        checkpoints.append(checkpoint)
        previous_bpb = float(checkpoint["heldout_bpb"])

    onset = next(
        (
            {
                "checkpoint_index": index,
                "tokens_seen": int(item["tokens_seen"]),
                "t_over_n": float(item["t_over_n"]),
            }
            for index, item in enumerate(checkpoints)
            if item["diagnostic"]["diagnostic_stop"]
        ),
        None,
    )
    return {
        "model_id": model_id,
        "parameter_count": spec.parameter_count(),
        "checkpoints": checkpoints,
        "relationship": {
            "first_validation_plus_disproportionate_memorization": onset,
            "interpretation": "association diagnostic only; not a privacy claim",
        },
    }


def run(repo_root: Path, output_dir: Path, *, profile: str) -> dict[str, Any]:
    config_path = repo_root / "configs/evaluation/memorization_curve_v1.json"
    config = _load_json(config_path)
    if config.get("schema_version") != "12-6.memorization-run-config.v1":
        raise ValueError("unsupported memorization run config")
    profile_config = config["profiles"].get(profile)
    if not isinstance(profile_config, dict):
        raise ValueError(f"unknown profile: {profile}")

    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_rows = _load_jsonl(train_path)
    validation_rows = _load_jsonl(validation_path)
    tokenizer = ByteTokenizer()
    seed = int(config["seed"])
    suite = build_canary_suite(
        seed=seed,
        exposures=tuple(int(value) for value in config["exposures_per_cycle"]),
        replicas=int(config["replicas_per_exposure"]),
        continuation_chars=int(config["continuation_chars"]),
    )
    checkpoint_tokens = sorted(set(int(value) for value in profile_config["checkpoint_tokens"]))
    if not checkpoint_tokens or checkpoint_tokens[0] != 0:
        raise ValueError("checkpoint_tokens must include random-init checkpoint 0")

    model_curves: list[dict[str, Any]] = []
    for model_index, (model_id, spec, init_spec) in enumerate(_model_ladder(repo_root)):
        model_curves.append(
            _train_model_curve(
                model_id=model_id,
                spec=spec,
                init_spec=init_spec,
                tokenizer=tokenizer,
                train_rows=train_rows,
                validation_rows=validation_rows,
                suite=suite,
                checkpoint_tokens=checkpoint_tokens,
                learning_rate=float(profile_config["learning_rate"]),
                alternative_count=int(profile_config["alternative_count"]),
                passage_sample_count=int(profile_config["passage_sample_count"]),
                seed=seed + model_index,
            )
        )

    report = {
        "schema_version": "12-6.memorization-curve.v1",
        "suite": suite.public(),
        "run_identity": {
            "worker_id": "EVAL-136-MEMORIZATION-CURVE",
            "execution_class": "LOCAL_FREE_CPU",
            "profile": profile,
            "config_sha256": _sha256_file(config_path),
            "dataset_manifest_sha256": _sha256_file(manifest_path),
            "train_sha256": _sha256_file(train_path),
            "validation_sha256": _sha256_file(validation_path),
            "tokenizer_config_sha256": tokenizer.identity.config_sha256,
            "checkpoint_tokens": checkpoint_tokens,
        },
        "models": model_curves,
        "safety_boundary": {
            "canary_text_emitted": False,
            "non_canary_training_text_emitted": False,
            "privacy_claim": "NONE",
        },
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "canary_suite_manifest.json").write_text(
        json.dumps(suite.public(), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "memorization_curve_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EVAL-136 LOCAL_FREE memorization curves")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "local_full"), default="smoke")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run(args.repo_root.resolve(), args.output_dir.resolve(), profile=args.profile)
    compact = {
        model["model_id"]: {
            "parameter_count": model["parameter_count"],
            "onset": model["relationship"][
                "first_validation_plus_disproportionate_memorization"
            ],
        }
        for model in report["models"]
    }
    print(json.dumps(compact, sort_keys=True))
    print(f"report_sha256={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
