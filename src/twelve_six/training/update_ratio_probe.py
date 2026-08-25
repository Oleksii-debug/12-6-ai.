"""LOCAL_FREE TRAIN-52 update-magnitude probe across ~100K/~500K/~1M models.

This is optimization-observability evidence, not a model-quality comparison or stage
promotion.  Each scale is trained twice from identical fp32 initialization and batch
trace: once with update telemetry disabled and once enabled.  Exact model and trainer
state hashes must match, while the enabled run records bounded global/per-block update
magnitude and direct probe overhead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.tokenization import ByteTokenizer

from .config import TrainerConfig
from .observability import TrainingObserver
from .trainer import Trainer

SCHEMA_VERSION = "12-6.train52-update-ratio-probe.v1"
AUTHORITY = "LOCAL_FREE_OPTIMIZATION_OBSERVABILITY_NOT_STAGE_PROMOTION"
UPDATE_SAMPLE_RETENTION_LIMIT = 64


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(node: Any) -> None:
        if is_dataclass(node) and not isinstance(node, type):
            visit(asdict(node))
            return
        if isinstance(node, torch.Tensor):
            tensor = node.detach().cpu().contiguous()
            digest.update(b"tensor:")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(b":")
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(b":")
            digest.update(tensor.numpy().tobytes(order="C"))
            digest.update(b";")
            return
        if isinstance(node, dict):
            digest.update(b"{")
            for key in sorted(node, key=lambda item: str(item)):
                visit(str(key))
                visit(node[key])
            digest.update(b"}")
            return
        if isinstance(node, (list, tuple)):
            digest.update(b"[")
            for item in node:
                visit(item)
            digest.update(b"]")
            return
        digest.update(
            json.dumps(node, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
                "utf-8"
            )
        )
        digest.update(b";")

    visit(value)
    return digest.hexdigest()


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _read_text_records(path: Path) -> list[str]:
    texts: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        text = record.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number} missing non-empty text")
        texts.append(text)
    if not texts:
        raise ValueError(f"{path} contains no usable text")
    return texts


def _byte_stream(texts: list[str], tokenizer: ByteTokenizer) -> bytes:
    return b"\n".join(bytes(tokenizer.encode(text)) for text in texts) + b"\n"


def _make_batch(
    stream: bytes,
    *,
    step: int,
    batch_size: int,
    sequence_length: int,
) -> dict[str, torch.Tensor]:
    width = batch_size * sequence_length
    base = (step * width) % len(stream)
    rows: list[list[int]] = []
    for batch_index in range(batch_size):
        start = (base + batch_index * sequence_length) % len(stream)
        rows.append([stream[(start + offset) % len(stream)] for offset in range(sequence_length)])
    return {"input_ids": torch.tensor(rows, dtype=torch.long)}


def _research_500k_spec() -> ModelSpec:
    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=256,
        d_model=96,
        n_layers=4,
        n_heads=6,
        n_kv_heads=6,
        head_dim=16,
        d_ff=256,
        activation="swiglu",
        norm_kind="rmsnorm",
        norm_placement="pre",
        norm_eps=1e-5,
        position_embedding="rope",
        rope_theta=10_000.0,
        rope_rotary_dim=16,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        final_norm=True,
        tie_word_embeddings=True,
        lm_head_bias=False,
    )
    if spec.parameter_count() != 467_808:
        raise RuntimeError("TRAIN-52 500K control geometry drift")
    return spec


def _scale_specs(root: Path) -> tuple[tuple[str, ModelSpec, InitSpec, str], ...]:
    s1_path = root / "configs/stages/s1_100k.json"
    s2_path = root / "configs/stages/s2_1m.json"
    s1 = load_stage_config(s1_path)
    s2 = load_stage_config(s2_path)
    return (
        ("~100K", s1.model, s1.init, f"canonical:{s1_path.relative_to(root)}"),
        ("~500K", _research_500k_spec(), InitSpec(), "RESEARCH41:467808-fixed-control"),
        ("~1M", s2.model, s2.init, f"canonical:{s2_path.relative_to(root)}"),
    )


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


def _run_once(
    *,
    spec: ModelSpec,
    init_spec: InitSpec,
    source_sha: str,
    scale_label: str,
    stream: bytes,
    steps: int,
    seed: int,
    batch_size: int,
    sequence_length: int,
    enable_update_magnitude: bool,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = TwelveSixDecoder(spec, init_spec)
    config = _trainer_config(max_steps=steps, seed=seed)
    trainer = Trainer(model, config, device="cpu")
    run_identity = {
        "repository": "Oleksii-debug/12-6-ai.",
        "source_sha": source_sha,
        "worker": "TRAIN-52-UPDATE-RATIO",
        "scale_label": scale_label,
        "model_spec_sha256": spec.identity_sha256(),
        "init_spec_sha256": init_spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "training_config": asdict(config),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "batch_trace": "CYCLIC_BYTE_STREAM_STEP_INDEX_V1",
    }
    observer = TrainingObserver(
        run_identity,
        device="cpu",
        max_step_samples=UPDATE_SAMPLE_RETENTION_LIMIT,
        gpu_sample_every_steps=max(steps + 1, 2),
        enable_update_magnitude=enable_update_magnitude,
        update_sample_every_steps=1,
        max_update_samples=UPDATE_SAMPLE_RETENTION_LIMIT,
    )
    losses: list[float] = []
    started = time.perf_counter()
    for step in range(steps):
        metrics = observer.train_microbatch(
            trainer,
            _make_batch(
                stream,
                step=step,
                batch_size=batch_size,
                sequence_length=sequence_length,
            ),
            data_wait_seconds=0.0,
        )
        losses.append(float(metrics.loss))
    wall_seconds = time.perf_counter() - started
    trainer.assert_checkpoint_safe()
    summary = observer.summary()
    return {
        "run_identity_sha256": observer.run_identity_sha256,
        "model_state_sha256": _model_state_sha256(model),
        "trainer_state_sha256": _tree_sha256(trainer.state_dict()),
        "optimizer_steps": trainer.optimizer_step,
        "optimized_tokens": trainer.tokens_seen,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "wall_seconds": wall_seconds,
        "telemetry_summary": summary,
        "update_samples": [asdict(sample) for sample in observer.update_samples],
    }


def _worst_updates(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"global": None, "per_block": None}
    global_sample = max(
        samples,
        key=lambda item: (
            -1.0
            if item["global_metrics"]["update_weight_ratio"] is None
            else float(item["global_metrics"]["update_weight_ratio"])
        ),
    )
    block_candidates: list[dict[str, Any]] = []
    for sample in samples:
        for block, metrics in sample["per_block"].items():
            ratio = metrics["update_weight_ratio"]
            if ratio is not None:
                block_candidates.append(
                    {
                        "optimizer_step": sample["optimizer_step"],
                        "block": block,
                        **metrics,
                    }
                )
    worst_block = (
        max(block_candidates, key=lambda item: float(item["update_weight_ratio"]))
        if block_candidates
        else None
    )
    return {
        "global": {
            "optimizer_step": global_sample["optimizer_step"],
            **global_sample["global_metrics"],
        },
        "per_block": worst_block,
    }


def run_probe(
    *,
    repo_root: Path,
    source_sha: str,
    steps: int = 12,
    seed: int = 1337,
    batch_size: int = 2,
    sequence_length: int = 64,
    torch_threads: int = 2,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("exact-checkout mismatch for TRAIN-52 probe")
    if steps <= 0 or batch_size <= 0 or sequence_length < 2 or torch_threads <= 0:
        raise ValueError("steps/batch_size/torch_threads must be positive and sequence_length >= 2")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    tokenizer = ByteTokenizer()
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    stream = _byte_stream(_read_text_records(train_path), tokenizer)

    scale_results: list[dict[str, Any]] = []
    for scale_label, spec, init_spec, geometry_source in _scale_specs(repo_root):
        if sequence_length > spec.max_seq_len:
            raise ValueError(f"sequence_length exceeds {scale_label} context")
        disabled = _run_once(
            spec=spec,
            init_spec=init_spec,
            source_sha=source_sha,
            scale_label=scale_label,
            stream=stream,
            steps=steps,
            seed=seed,
            batch_size=batch_size,
            sequence_length=sequence_length,
            enable_update_magnitude=False,
        )
        enabled = _run_once(
            spec=spec,
            init_spec=init_spec,
            source_sha=source_sha,
            scale_label=scale_label,
            stream=stream,
            steps=steps,
            seed=seed,
            batch_size=batch_size,
            sequence_length=sequence_length,
            enable_update_magnitude=True,
        )
        exact_model = disabled["model_state_sha256"] == enabled["model_state_sha256"]
        exact_trainer = disabled["trainer_state_sha256"] == enabled["trainer_state_sha256"]
        exact_identity = disabled["run_identity_sha256"] == enabled["run_identity_sha256"]
        exact_counters = (
            disabled["optimizer_steps"] == enabled["optimizer_steps"]
            and disabled["optimized_tokens"] == enabled["optimized_tokens"]
        )
        if not all((exact_model, exact_trainer, exact_identity, exact_counters)):
            raise RuntimeError(f"update telemetry changed deterministic state at {scale_label}")
        update_summary = enabled["telemetry_summary"]["update_magnitude"]
        scale_results.append(
            {
                "scale": scale_label,
                "geometry_source": geometry_source,
                "parameters": spec.parameter_count(),
                "model_spec_sha256": spec.identity_sha256(),
                "optimizer_steps": enabled["optimizer_steps"],
                "optimized_tokens": enabled["optimized_tokens"],
                "determinism": {
                    "same_run_identity_hash_enabled_vs_disabled": exact_identity,
                    "same_model_state_hash_enabled_vs_disabled": exact_model,
                    "same_trainer_state_hash_enabled_vs_disabled": exact_trainer,
                    "same_counters_enabled_vs_disabled": exact_counters,
                    "model_state_sha256": enabled["model_state_sha256"],
                    "trainer_state_sha256": enabled["trainer_state_sha256"],
                },
                "loss": {
                    "first": enabled["first_loss"],
                    "final": enabled["final_loss"],
                    "enabled_equals_disabled_final": enabled["final_loss"] == disabled["final_loss"],
                },
                "overhead": {
                    "disabled_wall_seconds": disabled["wall_seconds"],
                    "enabled_wall_seconds": enabled["wall_seconds"],
                    "paired_wall_overhead_fraction": (
                        enabled["wall_seconds"] / disabled["wall_seconds"] - 1.0
                        if disabled["wall_seconds"] > 0.0
                        else None
                    ),
                    **update_summary["overhead"],
                },
                "worst_observed_update": _worst_updates(enabled["update_samples"]),
                "pathology_candidates": update_summary["pathology_candidates"],
                "update_summary": update_summary,
                "update_samples": enabled["update_samples"],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "source": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": source_sha,
            "train_data_path": str(train_path.relative_to(repo_root)),
            "train_data_sha256": _sha256_file(train_path),
        },
        "protocol": {
            "precision": "fp32",
            "deterministic_algorithms": True,
            "seed": seed,
            "steps_per_enabled_or_disabled_run": steps,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "torch_threads": torch_threads,
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "update_sample_retention_limit": UPDATE_SAMPLE_RETENTION_LIMIT,
            "telemetry_enters_run_identity_or_training_state": False,
        },
        "scales": scale_results,
        "claims": {
            "paid_compute_used": False,
            "stage_promotion": False,
            "quality_comparison_across_vocabularies": False,
        },
    }


def validate_probe(report: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema_version") != SCHEMA_VERSION or report.get("authority") != AUTHORITY:
        raise ValueError("TRAIN-52 report schema/authority mismatch")
    source = report.get("source")
    if not isinstance(source, dict):
        raise ValueError("TRAIN-52 source block missing")
    if expected_source_sha is not None and source.get("git_sha") != expected_source_sha:
        raise ValueError("TRAIN-52 source SHA mismatch")
    scales = report.get("scales")
    if not isinstance(scales, list) or len(scales) != 3:
        raise ValueError("TRAIN-52 requires exactly three scale observations")
    expected_counts = {107_856, 467_808, 1_066_112}
    if {int(item["parameters"]) for item in scales} != expected_counts:
        raise ValueError("TRAIN-52 scale parameter counts drifted")
    for item in scales:
        determinism = item.get("determinism", {})
        required = (
            determinism.get("same_run_identity_hash_enabled_vs_disabled"),
            determinism.get("same_model_state_hash_enabled_vs_disabled"),
            determinism.get("same_trainer_state_hash_enabled_vs_disabled"),
            determinism.get("same_counters_enabled_vs_disabled"),
        )
        if not all(value is True for value in required):
            raise ValueError(f"TRAIN-52 determinism failure at {item.get('scale')}")
        update_summary = item.get("update_summary", {})
        if update_summary.get("status") != "MEASURED":
            raise ValueError(f"TRAIN-52 update telemetry missing at {item.get('scale')}")
        if int(update_summary.get("probe_samples_total", 0)) <= 0:
            raise ValueError(f"TRAIN-52 has no update samples at {item.get('scale')}")
        samples = item.get("update_samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"TRAIN-52 retained update samples missing at {item.get('scale')}")
        if len(samples) > UPDATE_SAMPLE_RETENTION_LIMIT:
            raise ValueError(f"TRAIN-52 update retention exceeded bound at {item.get('scale')}")
        overhead = item.get("overhead", {})
        if int(overhead.get("temporary_snapshot_peak_bytes", 0)) <= 0:
            raise ValueError(f"TRAIN-52 snapshot overhead missing at {item.get('scale')}")
        worst = item.get("worst_observed_update", {}).get("global")
        if not isinstance(worst, dict) or not math.isfinite(float(worst["update_weight_ratio"])):
            raise ValueError(f"TRAIN-52 global update ratio invalid at {item.get('scale')}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--source-sha", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--steps", type=int, default=12)
    run.add_argument("--seed", type=int, default=1337)
    run.add_argument("--batch-size", type=int, default=2)
    run.add_argument("--sequence-length", type=int, default=64)
    run.add_argument("--torch-threads", type=int, default=2)
    validate = subparsers.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "run":
        report = run_probe(
            repo_root=args.repo_root,
            source_sha=args.source_sha,
            steps=args.steps,
            seed=args.seed,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            torch_threads=args.torch_threads,
        )
        validate_probe(report, expected_source_sha=args.source_sha)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_probe(report, expected_source_sha=args.expected_source_sha)


if __name__ == "__main__":
    main()
