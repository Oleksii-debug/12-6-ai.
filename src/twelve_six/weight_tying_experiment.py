"""LOCAL_FREE tied-vs-untied embedding experiment for MODEL-16.

This is a research-only surface. It does not change canonical stage configs or
the global decoder contract. It deliberately reuses RESEARCH41's fixed-control
data, tokenizer, batching and optimizer recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from .checkpoint import CheckpointIdentity, hash_json, save_checkpoint, sha256_file
from .model import InitSpec, ModelSpec, TwelveSixDecoder
from .scaling_experiment import (
    _byte_stream,
    _make_batch,
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

SCHEMA = "12-6.model16-weight-tying.v1"
AUTHORITY = "LOCAL_FREE_RESEARCH_EVIDENCE_NOT_CANONICAL_ARCHITECTURE_FREEZE"
PACKING_ID = "research41-byte-stream-cyclic-v1"
TARGET_STEPS = 130
BATCH_SIZE = 4
SEQUENCE_LENGTH = 64
SEED = 1337
TORCH_THREADS = 2


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def matched_specs() -> tuple[tuple[str, ModelSpec], ...]:
    """Return the near-iso-parameter pair around the RESEARCH41 ~250K control."""
    incumbent = controlled_specs()[1]
    if incumbent.parameter_count() != 267_912:
        raise RuntimeError("RESEARCH41 ~250K control drifted")
    tied = incumbent
    untied = replace(
        incumbent,
        d_ff=171,
        tie_word_embeddings=False,
    )
    if tied.tie_word_embeddings is not True or untied.tie_word_embeddings is not False:
        raise RuntimeError("weight-tying semantics drifted")
    if tied.d_model != untied.d_model:
        raise RuntimeError("d_model must remain identical")
    if (
        tied.n_layers,
        tied.n_heads,
        tied.n_kv_heads,
        tied.head_dim,
        tied.max_seq_len,
        tied.vocab_size,
    ) != (
        untied.n_layers,
        untied.n_heads,
        untied.n_kv_heads,
        untied.head_dim,
        untied.max_seq_len,
        untied.vocab_size,
    ):
        raise RuntimeError("only FFN allocation and tying semantics may differ")
    relative_delta = abs(tied.parameter_count() - untied.parameter_count()) / tied.parameter_count()
    if relative_delta > 0.002:
        raise RuntimeError(f"parameter matching too loose: {relative_delta:.6f}")
    if tied.identity_sha256() == untied.identity_sha256():
        raise RuntimeError("tied and untied ModelSpec identities must differ")
    return (("tied", tied), ("untied", untied))


def _allocation(spec: ModelSpec) -> dict[str, Any]:
    breakdown = spec.parameter_breakdown()
    total = breakdown["total"]
    return {
        "counts": breakdown,
        "fractions": {
            key: value / total
            for key, value in breakdown.items()
            if key
            in {
                "token_embedding",
                "blocks_total",
                "final_norm",
                "lm_head_extra",
            }
        },
    }


def _parameter_group(name: str, model: TwelveSixDecoder) -> str:
    if name == "token_embedding.weight":
        if model.lm_head.weight is model.token_embedding.weight:
            return "embedding_output_shared"
        return "input_embedding"
    if name == "lm_head.weight":
        return "output_head"
    if name.startswith("blocks."):
        return "transformer_blocks"
    if name.startswith("final_norm."):
        return "final_norm"
    return "other"


def _install_gradient_hooks(model: TwelveSixDecoder):
    current: dict[str, float] = {}
    history: dict[str, list[float]] = {}
    handles = []

    for name, parameter in model.named_parameters():
        group = _parameter_group(name, model)

        def hook(grad: torch.Tensor, *, _group: str = group) -> torch.Tensor:
            norm = float(torch.linalg.vector_norm(grad.detach().float()).item())
            current[_group] = current.get(_group, 0.0) + norm * norm
            return grad

        handles.append(parameter.register_hook(hook))

    def begin_step() -> None:
        current.clear()

    def finish_step(tokens: int) -> None:
        if tokens <= 0:
            raise ValueError("tokens must be positive")
        for group, squared_norm in current.items():
            history.setdefault(group, []).append(math.sqrt(squared_norm) / tokens)

    def close() -> None:
        for handle in handles:
            handle.remove()

    return begin_step, finish_step, close, history


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "max": max(values),
    }


def _checkpoint_identity(
    *,
    source_sha: str,
    spec: ModelSpec,
    trainer: Trainer,
    manifest_path: Path,
    candidate: str,
) -> CheckpointIdentity:
    training_config = asdict(trainer.config)
    optimizer_identity = {
        "name": "AdamW",
        "learning_rate": trainer.config.learning_rate,
        "betas": list(trainer.config.betas),
        "eps": trainer.config.eps,
        "weight_decay": trainer.config.weight_decay,
        "gradient_clip_norm": trainer.config.gradient_clip_norm,
    }
    run_manifest = {
        "schema": SCHEMA,
        "candidate": candidate,
        "controls": {
            "tokenizer": BYTE_TOKENIZER_VERSION,
            "packing": PACKING_ID,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "optimizer_steps": TARGET_STEPS,
            "seed": SEED,
        },
        "model_spec_sha256": spec.identity_sha256(),
    }
    return CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        dataset_manifest_hash=sha256_file(manifest_path),
        run_manifest_hash=hash_json(run_manifest),
        training_config=training_config,
        seed=SEED,
        precision=trainer.config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer=optimizer_identity,
        scheduler=None,
    )


def _run_candidate(
    *,
    candidate: str,
    spec: ModelSpec,
    source_sha: str,
    train_stream: bytes,
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    manifest_path: Path,
    checkpoint_root: Path,
) -> dict[str, Any]:
    random.seed(SEED)
    torch.manual_seed(SEED)
    init_spec = InitSpec()
    model = TwelveSixDecoder(spec, init_spec)
    trainer_config = _trainer_config(max_steps=TARGET_STEPS, seed=SEED)
    trainer = Trainer(model, trainer_config, device="cpu")

    shared_alias = model.lm_head.weight is model.token_embedding.weight
    state = model.state_dict()
    state_keys = sorted(state.keys())
    if "token_embedding.weight" not in state or "lm_head.weight" not in state:
        raise RuntimeError("state_dict must expose both embedding and lm_head semantic keys")
    if shared_alias != spec.tie_word_embeddings:
        raise RuntimeError("runtime weight alias disagrees with ModelSpec")
    initial_embedding_head_equal = bool(
        torch.equal(state["token_embedding.weight"], state["lm_head.weight"])
    )

    initial_validation_loss, validation_tokens = _validation_loss(
        model, validation_records, tokenizer
    )
    begin_grad, finish_grad, close_grad, grad_history = _install_gradient_hooks(model)
    step_seconds: list[float] = []
    global_grad_norms: list[float] = []
    train_loss_weighted_sum = 0.0
    train_tokens = 0

    started = time.perf_counter()
    for step in range(TARGET_STEPS):
        batch = _make_batch(
            train_stream,
            step=step,
            batch_size=BATCH_SIZE,
            sequence_length=SEQUENCE_LENGTH,
        )
        begin_grad()
        step_started = time.perf_counter()
        metrics = trainer.train_microbatch({"input_ids": batch})
        step_seconds.append(time.perf_counter() - step_started)
        finish_grad(metrics.tokens)
        train_loss_weighted_sum += metrics.loss * metrics.tokens
        train_tokens += metrics.tokens
        if metrics.grad_norm is not None:
            global_grad_norms.append(metrics.grad_norm)
    elapsed = time.perf_counter() - started
    close_grad()

    if trainer.optimizer_step != TARGET_STEPS:
        raise RuntimeError("optimizer step count drift")
    if trainer.tokens_seen != train_tokens:
        raise RuntimeError("trainer token accounting drift")

    final_validation_loss, checked_validation_tokens = _validation_loss(
        model, validation_records, tokenizer
    )
    if checked_validation_tokens != validation_tokens:
        raise RuntimeError("held-out validation token count drift")

    checkpoint_dir = checkpoint_root / candidate
    manifest = save_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        trainer_state={
            "micro_step": trainer.micro_step,
            "optimizer_step": trainer.optimizer_step,
            "tokens_seen": trainer.tokens_seen,
        },
        identity=_checkpoint_identity(
            source_sha=source_sha,
            spec=spec,
            trainer=trainer,
            manifest_path=manifest_path,
            candidate=candidate,
        ),
    )
    checkpoint_total_bytes = sum(
        path.stat().st_size for path in checkpoint_dir.iterdir() if path.is_file()
    )

    final_state = model.state_dict()
    embedding_head_equal = bool(
        torch.equal(final_state["token_embedding.weight"], final_state["lm_head.weight"])
    )

    return {
        "candidate": candidate,
        "model_spec": spec.to_dict(),
        "model_identity_sha256": spec.identity_sha256(),
        "parameters": spec.parameter_count(),
        "parameter_allocation": _allocation(spec),
        "runtime_semantics": {
            "embedding_lm_head_shared_parameter": shared_alias,
            "state_dict_keys": state_keys,
            "initial_embedding_head_tensor_equal": initial_embedding_head_equal,
            "final_embedding_head_tensor_equal": embedding_head_equal,
        },
        "training": {
            "optimizer_steps": trainer.optimizer_step,
            "optimized_tokens": trainer.tokens_seen,
            "mean_training_loss": train_loss_weighted_sum / train_tokens,
            "last_training_loss": metrics.loss,
            "global_grad_norm": _summary(global_grad_norms),
            "gradient_group_l2_per_token": {
                group: _summary(values) for group, values in sorted(grad_history.items())
            },
            "step_seconds": _summary(step_seconds),
            "wall_seconds": elapsed,
            "optimized_tokens_per_wall_second": trainer.tokens_seen / elapsed,
        },
        "held_out": {
            "validation_tokens": validation_tokens,
            "initial_loss": initial_validation_loss,
            "final_loss": final_validation_loss,
            "loss_improvement": initial_validation_loss - final_validation_loss,
        },
        "checkpoint": {
            "checkpoint_id": manifest["checkpoint_id"],
            "model_spec_hash": manifest["identity"]["model_spec_hash"],
            "weights_bytes": manifest["files"]["weights.safetensors"]["bytes"],
            "total_directory_bytes": checkpoint_total_bytes,
            "serialization": manifest["serialization"],
        },
    }


def run_experiment(*, repo_root: Path, source_sha: str, output_path: Path) -> dict[str, Any]:
    if len(source_sha) != 40 or any(ch not in "0123456789abcdef" for ch in source_sha):
        raise ValueError("source_sha must be lowercase 40-hex")
    observed = os.popen(f"git -C {repo_root} rev-parse HEAD").read().strip()
    if observed != source_sha:
        raise RuntimeError(f"exact-checkout mismatch: expected {source_sha}, observed {observed}")

    torch.set_num_threads(TORCH_THREADS)
    torch.use_deterministic_algorithms(True)
    tokenizer = ByteTokenizer()
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_records = _read_jsonl(train_path)
    validation_records = _read_jsonl(validation_path)
    train_ids = {str(record["id"]) for record in train_records}
    validation_ids = {str(record["id"]) for record in validation_records}
    if train_ids & validation_ids:
        raise RuntimeError("train/validation record overlap")
    train_stream = _byte_stream(train_records, tokenizer)

    with tempfile.TemporaryDirectory(prefix="model16-checkpoints-") as temp:
        checkpoint_root = Path(temp)
        runs = [
            _run_candidate(
                candidate=name,
                spec=spec,
                source_sha=source_sha,
                train_stream=train_stream,
                train_records=train_records,
                validation_records=validation_records,
                tokenizer=tokenizer,
                manifest_path=manifest_path,
                checkpoint_root=checkpoint_root,
            )
            for name, spec in matched_specs()
        ]

    tied, untied = runs
    if tied["training"]["optimized_tokens"] != untied["training"]["optimized_tokens"]:
        raise RuntimeError("optimized token count differs between candidates")
    if tied["model_identity_sha256"] == untied["model_identity_sha256"]:
        raise RuntimeError("ModelSpec identity collision")
    if tied["checkpoint"]["checkpoint_id"] == untied["checkpoint"]["checkpoint_id"]:
        raise RuntimeError("checkpoint identity collision")
    for run in runs:
        if run["checkpoint"]["model_spec_hash"] != hash_json(run["model_spec"]):
            raise RuntimeError("checkpoint ModelSpec identity mismatch")

    parameter_delta = untied["parameters"] - tied["parameters"]
    relative_parameter_delta = abs(parameter_delta) / tied["parameters"]
    heldout_delta_untied_minus_tied = (
        untied["held_out"]["final_loss"] - tied["held_out"]["final_loss"]
    )
    preferred = (
        "tied"
        if tied["held_out"]["final_loss"] <= untied["held_out"]["final_loss"]
        else "untied"
    )

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
            "torch_threads": TORCH_THREADS,
            "paid_compute": False,
        },
        "controls": {
            "data": {
                "manifest_sha256": sha256_file(manifest_path),
                "train_jsonl_sha256": sha256_file(train_path),
                "validation_jsonl_sha256": sha256_file(validation_path),
                "train_records": len(train_records),
                "validation_records": len(validation_records),
            },
            "tokenizer_version": BYTE_TOKENIZER_VERSION,
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "packing_id": PACKING_ID,
            "model_max_seq_len": 256,
            "training_sequence_length": SEQUENCE_LENGTH,
            "batch_size": BATCH_SIZE,
            "optimizer_steps": TARGET_STEPS,
            "optimized_tokens_per_candidate": TARGET_STEPS
            * BATCH_SIZE
            * (SEQUENCE_LENGTH - 1),
            "optimizer": asdict(_trainer_config(max_steps=TARGET_STEPS, seed=SEED)),
            "init_spec": InitSpec().to_dict(),
            "seed": SEED,
        },
        "matching": {
            "tied_parameters": tied["parameters"],
            "untied_parameters": untied["parameters"],
            "untied_minus_tied_parameters": parameter_delta,
            "absolute_relative_parameter_delta": relative_parameter_delta,
            "attention_geometry_identical": True,
            "d_model_identical": True,
            "allocation_change": "untied adds an independent V*d_model output matrix and reduces d_ff 192->171",
        },
        "runs": runs,
        "comparison": {
            "heldout_loss_untied_minus_tied": heldout_delta_untied_minus_tied,
            "experimental_preference": preferred,
            "selection_rule": "lower held-out final loss; parameter mismatch is <=0.2% and untied gets no free parameter advantage",
            "checkpoint_size_untied_minus_tied_bytes": (
                untied["checkpoint"]["total_directory_bytes"]
                - tied["checkpoint"]["total_directory_bytes"]
            ),
            "truth_boundary": (
                "One deterministic LOCAL_FREE seed on the tiny recycled S0 fixture. "
                "This is architecture research evidence, not a canonical architecture freeze."
            ),
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
    if report.get("schema") != SCHEMA:
        raise ValueError("wrong report schema")
    if report.get("runtime", {}).get("paid_compute") is not False:
        raise ValueError("report must be LOCAL_FREE")
    if expected_source_sha is not None and report.get("source", {}).get("git_sha") != expected_source_sha:
        raise ValueError("source SHA mismatch")
    runs = report.get("runs")
    if not isinstance(runs, list) or [run.get("candidate") for run in runs] != ["tied", "untied"]:
        raise ValueError("expected tied and untied runs")
    if runs[0]["training"]["optimized_tokens"] != runs[1]["training"]["optimized_tokens"]:
        raise ValueError("optimized token mismatch")
    if report["matching"]["absolute_relative_parameter_delta"] > 0.002:
        raise ValueError("parameter match exceeds 0.2%")
    if runs[0]["model_identity_sha256"] == runs[1]["model_identity_sha256"]:
        raise ValueError("ModelSpec identity collision")
    if runs[0]["checkpoint"]["checkpoint_id"] == runs[1]["checkpoint"]["checkpoint_id"]:
        raise ValueError("checkpoint identity collision")
    claimed = report.get("report_sha256")
    payload = dict(report)
    payload.pop("report_sha256", None)
    if claimed != _canonical_hash(payload):
        raise ValueError("report hash mismatch")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--source-sha", required=True)
    run.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("report", type=Path)
    validate.add_argument("--expected-source-sha")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "run":
        report = run_experiment(
            repo_root=args.repo_root.resolve(),
            source_sha=args.source_sha,
            output_path=args.output,
        )
        print(json.dumps({
            "matching": report["matching"],
            "comparison": report["comparison"],
            "runs": [
                {
                    "candidate": run["candidate"],
                    "parameters": run["parameters"],
                    "held_out": run["held_out"],
                    "training": {
                        "mean_training_loss": run["training"]["mean_training_loss"],
                        "global_grad_norm": run["training"]["global_grad_norm"],
                        "gradient_group_l2_per_token": run["training"]["gradient_group_l2_per_token"],
                        "step_seconds": run["training"]["step_seconds"],
                    },
                    "checkpoint": run["checkpoint"],
                }
                for run in report["runs"]
            ],
            "report_sha256": report["report_sha256"],
        }, indent=2, sort_keys=True))
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_report(report, expected_source_sha=args.expected_source_sha)
    print("MODEL-16 weight-tying evidence validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
