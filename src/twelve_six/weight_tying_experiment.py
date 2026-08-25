"""Matched-parameter tied-vs-untied embedding experiment for MODEL-16.

This is an additive research surface. It does not modify canonical stage configs or
ModelSpec semantics. The existing ModelSpec v1 ``tie_word_embeddings`` field is the
architecture identity authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIdentity,
    assert_identity,
    hash_json,
    save_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from .model import InitSpec, ModelSpec, TwelveSixDecoder
from .tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from .training import Trainer, TrainerConfig

SCHEMA = "12-6.model16-weight-tying-evidence.v1"
AUTHORITY = "LOCAL_FREE_SMALL_SCALE_RESEARCH_NOT_ARCHITECTURE_FREEZE"
CONFIG_PATH = Path("configs/experiments/model16_weight_tying_250k.v1.json")


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} must contain a JSON object")
        if not isinstance(value.get("id"), str) or not isinstance(value.get("text"), str):
            raise ValueError(f"{path}:{line_number} requires string id/text")
        records.append(value)
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def _byte_stream(records: list[dict[str, Any]], tokenizer: ByteTokenizer) -> bytes:
    return b"\n".join(bytes(tokenizer.encode(str(record["text"]))) for record in records) + b"\n"


def _make_batch(
    stream: bytes,
    *,
    step: int,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    width = batch_size * sequence_length
    base = (step * width) % len(stream)
    rows = []
    for batch_index in range(batch_size):
        start = (base + batch_index * sequence_length) % len(stream)
        rows.append([stream[(start + offset) % len(stream)] for offset in range(sequence_length)])
    return torch.tensor(rows, dtype=torch.long)


def _trainer_config(config: dict[str, Any]) -> TrainerConfig:
    controls = config["controls"]
    optimizer = controls["optimizer"]
    tokens_per_step = int(controls["batch_size"]) * (
        int(controls["training_sequence_length"]) - 1
    )
    budget = int(controls["optimized_tokens_per_seed"])
    if budget % tokens_per_step != 0:
        raise ValueError("MODEL-16 token budget must be an exact multiple of tokens per step")
    return TrainerConfig(
        learning_rate=float(optimizer["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
        betas=tuple(float(item) for item in optimizer["betas"]),
        eps=float(optimizer["eps"]),
        max_steps=budget // tokens_per_step,
        warmup_steps=0,
        scheduler=str(optimizer["scheduler"]),
        gradient_accumulation_steps=1,
        gradient_clip_norm=float(optimizer["gradient_clip_norm"]),
        precision=str(optimizer["precision"]),
        seed=0,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )


@torch.no_grad()
def _held_out_loss(
    model: TwelveSixDecoder,
    records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
) -> tuple[float, int]:
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for record in records:
        ids = tokenizer.encode(str(record["text"]))
        start = 0
        while start < len(ids) - 1:
            chunk = ids[start : start + model.spec.max_seq_len]
            if len(chunk) < 2:
                break
            input_ids = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
            logits = model(input_ids).logits
            total_nll += float(
                F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, model.spec.vocab_size),
                    input_ids[:, 1:].reshape(-1),
                    reduction="sum",
                ).item()
            )
            total_tokens += len(chunk) - 1
            start += model.spec.max_seq_len - 1
    model.train(was_training)
    if total_tokens <= 0:
        raise RuntimeError("held-out split produced no causal targets")
    return total_nll / total_tokens, total_tokens


def _allocation(spec: ModelSpec) -> dict[str, Any]:
    breakdown = spec.parameter_breakdown()
    total = breakdown["total"]
    vocab_parameters = breakdown["token_embedding"] + breakdown["lm_head_extra"]
    return {
        "breakdown": breakdown,
        "vocabulary_parameters": vocab_parameters,
        "vocabulary_share": vocab_parameters / total,
        "block_parameters": breakdown["blocks_total"],
        "block_share": breakdown["blocks_total"] / total,
        "d_ff": spec.d_ff,
    }


def _checkpoint_size(directory: Path) -> dict[str, int]:
    files = {path.name: path.stat().st_size for path in directory.iterdir() if path.is_file()}
    files["total_directory_bytes"] = sum(files.values())
    return files


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return statistics.fmean(values)


def _run_one(
    *,
    repo_root: Path,
    source_sha: str,
    label: str,
    spec: ModelSpec,
    peer_spec: ModelSpec,
    init_spec: InitSpec,
    trainer_template: TrainerConfig,
    train_stream: bytes,
    validation_records: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    seed: int,
    batch_size: int,
    sequence_length: int,
    budget: int,
    dataset_manifest_hash: str,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    config_dict = asdict(trainer_template)
    config_dict["seed"] = seed
    trainer_config = TrainerConfig(**config_dict)
    model = TwelveSixDecoder(spec, init_spec)
    trainer = Trainer(model, trainer_config, device="cpu")
    initial_loss, held_out_tokens = _held_out_loss(model, validation_records, tokenizer)

    tokens_per_step = batch_size * (sequence_length - 1)
    if trainer_config.max_steps * tokens_per_step != budget:
        raise RuntimeError("MODEL-16 exact optimized-token budget drift")

    global_grad_norms: list[float] = []
    token_embedding_grad_norms: list[float] = []
    lm_head_grad_norms: list[float] = []
    step_seconds: list[float] = []
    train_losses: list[float] = []

    def capture_embedding(grad: torch.Tensor) -> torch.Tensor:
        token_embedding_grad_norms.append(float(grad.detach().float().norm().item()) / tokens_per_step)
        return grad

    handles = [model.token_embedding.weight.register_hook(capture_embedding)]
    if model.lm_head.weight is not model.token_embedding.weight:
        def capture_head(grad: torch.Tensor) -> torch.Tensor:
            lm_head_grad_norms.append(float(grad.detach().float().norm().item()) / tokens_per_step)
            return grad
        handles.append(model.lm_head.weight.register_hook(capture_head))

    try:
        for step in range(trainer_config.max_steps):
            batch = _make_batch(
                train_stream,
                step=step,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )
            started = time.perf_counter()
            metrics = trainer.train_microbatch({"input_ids": batch})
            step_seconds.append(time.perf_counter() - started)
            if not metrics.optimizer_stepped or metrics.grad_norm is None or metrics.update_loss is None:
                raise RuntimeError("MODEL-16 expects one optimizer update per microbatch")
            global_grad_norms.append(float(metrics.grad_norm))
            train_losses.append(float(metrics.update_loss))
    finally:
        for handle in handles:
            handle.remove()

    if trainer.tokens_seen != budget:
        raise RuntimeError(f"optimized-token mismatch: {trainer.tokens_seen} != {budget}")
    final_loss, final_held_out_tokens = _held_out_loss(model, validation_records, tokenizer)
    if final_held_out_tokens != held_out_tokens:
        raise RuntimeError("held-out token count drift")

    shared_weight = model.token_embedding.weight is model.lm_head.weight
    if shared_weight != spec.tie_word_embeddings:
        raise RuntimeError("runtime embedding alias semantics disagree with ModelSpec")

    run_manifest_hash = hash_json(
        {
            "experiment": SCHEMA,
            "candidate": label,
            "model_spec_hash": spec.identity_sha256(),
            "seed": seed,
            "optimized_tokens": budget,
        }
    )
    identity = CheckpointIdentity(
        git_sha=source_sha,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash=BYTE_TOKENIZER_HASH,
        tokenizer_vocab_hash=BYTE_VOCAB_HASH,
        dataset_manifest_hash=dataset_manifest_hash,
        run_manifest_hash=run_manifest_hash,
        training_config=asdict(trainer_config),
        seed=seed,
        precision=trainer_config.precision,
        step=trainer.optimizer_step,
        tokens_seen=trainer.tokens_seen,
        optimizer={"name": "AdamW", "config": asdict(trainer_config)},
        scheduler=None,
    )
    with tempfile.TemporaryDirectory(prefix=f"model16-{label}-") as temp:
        checkpoint_dir = Path(temp) / "checkpoint"
        save_started = time.perf_counter()
        manifest = save_checkpoint(
            checkpoint_dir,
            model=model,
            identity=identity,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            trainer_state={
                "micro_step": trainer.micro_step,
                "optimizer_step": trainer.optimizer_step,
                "tokens_seen": trainer.tokens_seen,
            },
        )
        checkpoint_save_seconds = time.perf_counter() - save_started
        verified = verify_checkpoint(checkpoint_dir)
        if verified["checkpoint_id"] != manifest["checkpoint_id"]:
            raise RuntimeError("checkpoint verification identity drift")
        assert_identity(manifest, model_spec_hash=spec.identity_sha256())
        cross_semantics_rejected = False
        try:
            assert_identity(manifest, model_spec_hash=peer_spec.identity_sha256())
        except CheckpointCompatibilityError:
            cross_semantics_rejected = True
        if not cross_semantics_rejected:
            raise RuntimeError("checkpoint identity accepted opposite tie semantics")
        checkpoint_bytes = _checkpoint_size(checkpoint_dir)

    warm_step_seconds = step_seconds[2:] if len(step_seconds) > 2 else step_seconds
    gradient_behavior: dict[str, Any] = {
        "global_grad_norm_mean": _mean(global_grad_norms),
        "global_grad_norm_max": max(global_grad_norms),
        "token_embedding_grad_norm_mean_pre_normalization_scaled_back": _mean(
            token_embedding_grad_norms
        ),
        "embedding_and_lm_head_share_parameter": shared_weight,
    }
    if shared_weight:
        gradient_behavior["lm_head_grad_semantics"] = "combined_into_shared_embedding_parameter"
    else:
        gradient_behavior["lm_head_grad_norm_mean_pre_normalization_scaled_back"] = _mean(
            lm_head_grad_norms
        )

    return {
        "candidate": label,
        "seed": seed,
        "model_spec": spec.to_dict(),
        "model_identity_sha256": spec.identity_sha256(),
        "parameters": spec.parameter_count(),
        "parameter_allocation": _allocation(spec),
        "initial_held_out_loss": initial_loss,
        "final_held_out_loss": final_loss,
        "initial_held_out_bpb": initial_loss / math.log(2.0),
        "final_held_out_bpb": final_loss / math.log(2.0),
        "held_out_tokens": held_out_tokens,
        "final_training_loss": train_losses[-1],
        "mean_last_8_training_loss": _mean(train_losses[-8:]),
        "optimized_tokens": trainer.tokens_seen,
        "optimizer_steps": trainer.optimizer_step,
        "gradient_behavior": gradient_behavior,
        "step_time_seconds": {
            "median_after_warmup": statistics.median(warm_step_seconds),
            "mean_after_warmup": _mean(warm_step_seconds),
        },
        "checkpoint": {
            "checkpoint_id": manifest["checkpoint_id"],
            "model_spec_hash": manifest["identity"]["model_spec_hash"],
            "cross_tie_semantics_rejected": cross_semantics_rejected,
            "save_seconds": checkpoint_save_seconds,
            "bytes": checkpoint_bytes,
            "state_dict_keys_include_both_embedding_and_lm_head": (
                "token_embedding.weight" in model.state_dict()
                and "lm_head.weight" in model.state_dict()
            ),
        },
    }


def run_experiment(*, repo_root: Path, source_sha: str, output_path: Path) -> dict[str, Any]:
    if _git_head(repo_root) != source_sha:
        raise RuntimeError("MODEL-16 exact-checkout mismatch")
    config = json.loads((repo_root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema") != "12-6.model16-weight-tying-experiment.v1":
        raise ValueError("unexpected MODEL-16 config schema")

    candidate_specs = {
        label: ModelSpec.from_dict(candidate["model"])
        for label, candidate in config["candidates"].items()
    }
    if set(candidate_specs) != {"A_tied", "B_untied"}:
        raise ValueError("MODEL-16 requires exactly A_tied and B_untied")
    tied = candidate_specs["A_tied"]
    untied = candidate_specs["B_untied"]
    if not tied.tie_word_embeddings or untied.tie_word_embeddings:
        raise ValueError("MODEL-16 candidate tie semantics drift")
    for label, spec in candidate_specs.items():
        expected = int(config["candidates"][label]["expected_parameters"])
        if spec.parameter_count() != expected:
            raise ValueError(f"{label} parameter-count drift")
    delta_fraction = abs(tied.parameter_count() - untied.parameter_count()) / max(
        tied.parameter_count(), untied.parameter_count()
    )
    if delta_fraction > float(config["acceptance"]["max_parameter_delta_fraction"]):
        raise ValueError("MODEL-16 candidate parameter mismatch exceeds acceptance band")

    controls = config["controls"]
    tokenizer = ByteTokenizer()
    train_path = repo_root / "data/s0/packaged/train.jsonl"
    validation_path = repo_root / "data/s0/packaged/validation.jsonl"
    manifest_path = repo_root / "data/s0/packaged/manifest.json"
    train_records = _read_jsonl(train_path)
    validation_records = _read_jsonl(validation_path)
    train_ids = {str(record["id"]) for record in train_records}
    validation_ids = {str(record["id"]) for record in validation_records}
    if train_ids & validation_ids:
        raise RuntimeError("MODEL-16 train/held-out record overlap")
    train_stream = _byte_stream(train_records, tokenizer)
    init_spec = InitSpec()
    trainer_template = _trainer_config(config)
    batch_size = int(controls["batch_size"])
    sequence_length = int(controls["training_sequence_length"])
    budget = int(controls["optimized_tokens_per_seed"])
    dataset_manifest_hash = sha256_file(manifest_path)

    runs: list[dict[str, Any]] = []
    for seed in [int(value) for value in controls["seeds"]]:
        for label in ("A_tied", "B_untied"):
            spec = candidate_specs[label]
            peer = candidate_specs["B_untied" if label == "A_tied" else "A_tied"]
            runs.append(
                _run_one(
                    repo_root=repo_root,
                    source_sha=source_sha,
                    label=label,
                    spec=spec,
                    peer_spec=peer,
                    init_spec=init_spec,
                    trainer_template=trainer_template,
                    train_stream=train_stream,
                    validation_records=validation_records,
                    tokenizer=tokenizer,
                    seed=seed,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    budget=budget,
                    dataset_manifest_hash=dataset_manifest_hash,
                )
            )

    aggregates: dict[str, Any] = {}
    for label in ("A_tied", "B_untied"):
        selected = [run for run in runs if run["candidate"] == label]
        aggregates[label] = {
            "parameters": selected[0]["parameters"],
            "parameter_allocation": selected[0]["parameter_allocation"],
            "mean_final_held_out_loss": _mean([run["final_held_out_loss"] for run in selected]),
            "mean_final_held_out_bpb": _mean([run["final_held_out_bpb"] for run in selected]),
            "mean_final_training_loss": _mean([run["final_training_loss"] for run in selected]),
            "mean_global_grad_norm": _mean(
                [run["gradient_behavior"]["global_grad_norm_mean"] for run in selected]
            ),
            "median_step_seconds": statistics.median(
                [run["step_time_seconds"]["median_after_warmup"] for run in selected]
            ),
            "median_checkpoint_bytes": int(
                statistics.median(
                    [run["checkpoint"]["bytes"]["total_directory_bytes"] for run in selected]
                )
            ),
        }
    loss_delta = (
        aggregates["B_untied"]["mean_final_held_out_loss"]
        - aggregates["A_tied"]["mean_final_held_out_loss"]
    )
    if abs(loss_delta) < 0.01:
        observed_preference = "INCONCLUSIVE_WITHIN_0.01_NATS"
    elif loss_delta > 0:
        observed_preference = "A_TIED_LOWER_HELD_OUT_LOSS"
    else:
        observed_preference = "B_UNTIED_LOWER_HELD_OUT_LOSS"

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": "Oleksii-debug/12-6-ai.", "git_sha": source_sha},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "paid_compute": False,
        },
        "controls": {
            **controls,
            "tokenizer_version": BYTE_TOKENIZER_VERSION,
            "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
            "tokenizer_vocab_sha256": BYTE_VOCAB_HASH,
            "train_jsonl_sha256": sha256_file(train_path),
            "validation_jsonl_sha256": sha256_file(validation_path),
            "dataset_manifest_sha256": dataset_manifest_hash,
            "init_spec": init_spec.to_dict(),
            "init_identity_sha256": init_spec.identity_sha256(),
        },
        "matched_candidates": {
            label: {
                "model_spec": spec.to_dict(),
                "model_identity_sha256": spec.identity_sha256(),
                "parameters": spec.parameter_count(),
            }
            for label, spec in candidate_specs.items()
        },
        "parameter_delta_fraction": delta_fraction,
        "runs": runs,
        "aggregates": aggregates,
        "decision": {
            "observed_preference": observed_preference,
            "untied_minus_tied_mean_final_held_out_loss": loss_delta,
            "recommendation_scope": (
                "Small controlled S0-fixture research only; do not promote tied or untied globally "
                "without intended-corpus replication."
            ),
        },
        "truth_boundary": {
            "canonical_architecture_changed": False,
            "model_spec_schema_changed": False,
            "architecture_freeze": False,
            "representative_intended_corpus_claimed": False,
            "paid_compute": False,
        },
    }
    report["report_sha256"] = _canonical_hash(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report


def validate_report(report: dict[str, Any], *, expected_source_sha: str | None = None) -> None:
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise ValueError("MODEL-16 report schema/authority mismatch")
    if expected_source_sha is not None and report["source"]["git_sha"] != expected_source_sha:
        raise ValueError("MODEL-16 source SHA mismatch")
    if report["runtime"].get("paid_compute") is not False:
        raise ValueError("MODEL-16 may not claim paid compute")
    candidates = report["matched_candidates"]
    if candidates["A_tied"]["model_spec"]["tie_word_embeddings"] is not True:
        raise ValueError("A_tied semantic drift")
    if candidates["B_untied"]["model_spec"]["tie_word_embeddings"] is not False:
        raise ValueError("B_untied semantic drift")
    if float(report["parameter_delta_fraction"]) > 0.01:
        raise ValueError("MODEL-16 parameter matching weakened")
    for run in report["runs"]:
        if int(run["optimized_tokens"]) != 16128:
            raise ValueError("MODEL-16 token budget drift")
        if run["checkpoint"]["cross_tie_semantics_rejected"] is not True:
            raise ValueError("checkpoint failed to distinguish tie semantics")
        if not math.isfinite(float(run["final_held_out_loss"])):
            raise ValueError("non-finite held-out loss")
    truth = report["truth_boundary"]
    if truth != {
        "canonical_architecture_changed": False,
        "model_spec_schema_changed": False,
        "architecture_freeze": False,
        "representative_intended_corpus_claimed": False,
        "paid_compute": False,
    }:
        raise ValueError("MODEL-16 truth boundary weakened")
    supplied = report["report_sha256"]
    unsigned = dict(report)
    unsigned.pop("report_sha256")
    if supplied != _canonical_hash(unsigned):
        raise ValueError("MODEL-16 report self-hash mismatch")


def _parser() -> argparse.ArgumentParser:
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        report = run_experiment(
            repo_root=args.repo_root.resolve(), source_sha=args.source_sha, output_path=args.output
        )
        validate_report(report, expected_source_sha=args.source_sha)
        print(json.dumps(report["decision"], sort_keys=True))
        return 0
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validate_report(report, expected_source_sha=args.expected_source_sha)
    print(f"{SCHEMA}: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
