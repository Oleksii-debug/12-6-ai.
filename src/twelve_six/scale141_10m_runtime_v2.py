"""SCALE-141 authoritative LOCAL_FREE learned-10M runtime.

This layer corrects two issues found during pre-execution audit of the initial
SCALE-141 scaffold:

1. DATA-25 uses document-isolated packing, so padded 1024-token windows do not
   imply 1023 optimized targets. Campaign gates therefore use Trainer.tokens_seen
   (non-ignored shifted labels), never nominal padded capacity.
2. Fixed evaluation is streamed one packed window at a time so the exact 10M
   model never materializes a 32 x 1024 attention batch on a CPU runner.

The underlying S3 ModelSpec/tokenizer/corpus/optimizer/checkpoint contracts remain
unchanged. Sequence length 256 is an explicit LOCAL_FREE execution adaptation under
the model's max_seq_len=1024; it reduces wasted attention on short isolated DATA-25
documents while preserving every within-document language-model pair encountered.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

from twelve_six import scale141_10m_continuation as core
from twelve_six.checkpoint import hash_json, load_trainer_checkpoint, verify_checkpoint
from twelve_six.model import TwelveSixDecoder
from twelve_six.training import Trainer
from twelve_six.training.observability import TrainingObserver

SCHEMA = "12-6.scale141-10m-learned-fallback.v2"
SEQ = 256
BATCH = 1
TARGET_OPTIMIZED_TOKENS = 2_000_000
RESUME_TOKEN_TARGET = 1_000_000
MAX_OPTIMIZER_STEPS = 20_000
EVAL_TOKEN_TARGETS = (0, 500_000, 1_000_000, 1_500_000, 2_000_000)
MAX_TOKEN_OVERSHOOT = SEQ - 2

_ORIGINAL_RUN_MANIFEST = core._run_manifest
_ORIGINAL_INTERVAL_SUMMARY = core._interval_summary


class Scale141RuntimeError(core.Scale141Error):
    pass


def _install_runtime_contract() -> None:
    core.SEQ = SEQ
    core.BATCH = BATCH
    core.MAX_STEPS = MAX_OPTIMIZER_STEPS
    core.RESUME_STEP = -1  # v2 resumes by actual optimized-token threshold.
    core.EXPECTED_TOKENS_PER_STEP = 0  # variable under isolated-document packing.
    core.EXPECTED_OPTIMIZED_TOKENS = TARGET_OPTIMIZED_TOKENS
    core.EVAL_STEPS = ()
    core._fixed_eval = _fixed_eval_streaming
    core._interval_summary = _interval_summary_v2
    core._run_manifest = _run_manifest_v2


def _run_manifest_v2(
    source_sha: str,
    spec,
    init,
    tok,
    manifest: dict[str, Any],
    cfg,
    locks: dict[str, Any],
    prepared: dict[str, Any],
) -> dict[str, Any]:
    value = _ORIGINAL_RUN_MANIFEST(
        source_sha, spec, init, tok, manifest, cfg, locks, prepared
    )
    value["schema"] = "12-6.scale141-run-manifest.v2"
    value["expected_tokens_per_step"] = None
    value["token_accounting"] = {
        "authority": "Trainer.tokens_seen",
        "definition": "count(labels[:, 1:] != -100) actually optimized by D02 Trainer",
        "nominal_padded_sequence_capacity_is_not_used_as_token_count": True,
        "max_single_step_loss_tokens": SEQ - 1,
    }
    value["target_optimized_tokens"] = TARGET_OPTIMIZED_TOKENS
    value["target_corpus_fraction"] = TARGET_OPTIMIZED_TOKENS / core.TRAIN_CORPUS_BYTES
    value["scheduled_evaluation_steps"] = []
    value["scheduled_evaluation_token_targets"] = list(EVAL_TOKEN_TARGETS)
    value["fresh_process_resume_step"] = None
    value["fresh_process_resume_token_target"] = RESUME_TOKEN_TARGET
    value["max_optimizer_steps_safety_ceiling"] = MAX_OPTIMIZER_STEPS
    value["local_free_adaptation"].update(
        {
            "sequence_length": SEQ,
            "model_max_sequence_length": spec.max_seq_len,
            "sequence_length_reason": (
                "DATA-25 documents are isolated and often shorter than 1024 bytes; "
                "256 reduces padded attention cost on LOCAL_FREE CPU while token "
                "budget is measured from actual non-ignored labels"
            ),
            "document_boundary_policy": "isolate",
            "cross_document_packing": False,
        }
    )
    value.pop("identity_sha256", None)
    value["identity_sha256"] = hash_json(value)
    return value


def _fixed_eval_streaming(
    model: TwelveSixDecoder,
    corpus: Path,
    manifest: dict[str, Any],
    tok,
    *,
    split: str,
    windows: int,
) -> dict[str, Any]:
    before = core._state_hash(model)
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    by_modality: dict[str, Any] = {}
    try:
        with torch.no_grad():
            for stratum in ("uk", "en", "code"):
                modality_nll = 0.0
                modality_tokens = 0
                observed = 0
                for example in core._packed(corpus, manifest, tok, split, stratum):
                    nll, tokens = core._eval_examples(model, [example])
                    modality_nll += nll
                    modality_tokens += tokens
                    observed += 1
                    if observed == windows:
                        break
                if observed != windows or modality_tokens <= 0:
                    raise Scale141RuntimeError(
                        f"insufficient {split}/{stratum} fixed eval windows"
                    )
                loss = modality_nll / modality_tokens
                by_modality[stratum] = {
                    "loss": loss,
                    "bits_per_byte": loss / math.log(2.0),
                    "predicted_byte_tokens": modality_tokens,
                    "windows": observed,
                    "evaluation_batch_windows": 1,
                }
                total_nll += modality_nll
                total_tokens += modality_tokens
    finally:
        model.train(was_training)
    if core._state_hash(model) != before:
        raise Scale141RuntimeError("evaluation mutated model state")
    loss = total_nll / total_tokens
    return {
        "split": split,
        "loss": loss,
        "bits_per_byte": loss / math.log(2.0),
        "predicted_byte_tokens": total_tokens,
        "by_modality": by_modality,
        "evaluation_batch_windows": 1,
        "non_mutation_passed": True,
    }


def _interval_summary_v2(rows: list[dict[str, Any]]) -> dict[str, Any]:
    value = _ORIGINAL_INTERVAL_SUMMARY(rows)
    if rows:
        value["scheduled_step_update_norm"] = rows[-1].get("update_norm")
        value["scheduled_step_relative_update_norm"] = rows[-1].get(
            "relative_update_norm"
        )
        value["actual_optimized_tokens"] = sum(int(row["tokens"]) for row in rows)
    return value


def _checkpoint_path(out: Path, token_target: int) -> Path:
    return out / f"checkpoint-token-{token_target:07d}"


def _scheduled_eval(
    *,
    token_target: int,
    out: Path,
    source_sha: str,
    spec,
    tok,
    manifest: dict[str, Any],
    run: dict[str, Any],
    cfg,
    trainer: Trainer,
    locks: dict[str, Any],
    corpus: Path,
    interval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path = _checkpoint_path(out, token_target)
    checkpoint = core._save(
        path, source_sha, spec, tok, manifest, run, cfg, trainer, locks
    )
    point = core._eval_point(
        trainer.model,
        path,
        corpus,
        manifest,
        tok,
        trainer,
        interval_rows,
    )
    point["scheduled_token_target"] = token_target
    point["threshold_overshoot_tokens"] = trainer.tokens_seen - token_target
    point["checkpoint"] = checkpoint
    return point


def _data_position(step: int) -> dict[str, Any]:
    return {
        "optimizer_step": step,
        "consumed_packed_examples_by_modality": core._steps_by_stratum(step),
        "batch_size": BATCH,
        "mixture_pattern": list(core.MIXTURE),
    }


def _next_threshold(current_tokens: int, targets: tuple[int, ...]) -> int | None:
    for target in targets:
        if current_tokens < target:
            return target
    return None


def _should_capture_update(current_tokens: int, target: int | None) -> bool:
    if target is None:
        return False
    return current_tokens < target <= current_tokens + (SEQ - 1)


def _append_train_row(path: Path, row: dict[str, Any], stratum: str) -> None:
    row["stratum"] = stratum
    core._append(path, row)


def phase1(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    _install_runtime_contract()
    out.mkdir(parents=True, exist_ok=True)
    manifest, tok, spec, init, cfg, locks, run = core._common(
        repo, source_sha, out, True
    )
    corpus = out / "corpus-a"
    torch.manual_seed(core.SEED)
    model = TwelveSixDecoder(spec, init)
    if sum(p.numel() for p in model.parameters()) != core.EXPECTED_PARAMETERS:
        raise Scale141RuntimeError("runtime parameter count mismatch")
    trainer = Trainer(model, cfg, device="cpu")
    observer = TrainingObserver(
        run, device="cpu", max_step_samples=MAX_OPTIMIZER_STEPS + 8
    )

    checkpoint0 = core._save(
        _checkpoint_path(out, 0),
        source_sha,
        spec,
        tok,
        manifest,
        run,
        cfg,
        trainer,
        locks,
    )
    scheduled: dict[str, Any] = {
        "0": core._eval_point(
            model,
            _checkpoint_path(out, 0),
            corpus,
            manifest,
            tok,
            trainer,
            [],
        )
    }
    scheduled["0"]["scheduled_token_target"] = 0
    scheduled["0"]["threshold_overshoot_tokens"] = 0
    scheduled["0"]["checkpoint"] = checkpoint0

    initial_state = core._state_hash(model)
    iterators = core._train_iters(corpus, manifest, tok, 0)
    curve = out / "train-curve.jsonl"
    if curve.exists():
        curve.unlink()
    interval_rows: list[dict[str, Any]] = []
    calibration_step_seconds: list[float] = []
    calibration_save_seconds: list[float] = []
    cadence: dict[str, Any] | None = None
    targets = (500_000, RESUME_TOKEN_TARGET)

    while trainer.tokens_seen < RESUME_TOKEN_TARGET:
        if trainer.optimizer_step >= MAX_OPTIMIZER_STEPS:
            raise Scale141RuntimeError("phase1 hit optimizer-step safety ceiling")
        target = _next_threshold(trainer.tokens_seen, targets)
        index = trainer.optimizer_step
        stratum = core.MIXTURE[index % len(core.MIXTURE)]
        row = core._train_transition(
            observer,
            trainer,
            core._next_batch(iterators[stratum]),
            scheduled=_should_capture_update(trainer.tokens_seen, target),
        )
        if not 1 <= int(row["tokens"]) <= SEQ - 1:
            raise Scale141RuntimeError(
                f"invalid actual optimized-token count: {row['tokens']}"
            )
        interval_rows.append(row)
        _append_train_row(curve, row, stratum)

        if trainer.optimizer_step <= core.CALIBRATION_STEPS:
            calibration_step_seconds.append(float(row["step_seconds"]))
            import time

            started = time.perf_counter()
            core._save(
                out / "recovery-latest",
                source_sha,
                spec,
                tok,
                manifest,
                run,
                cfg,
                trainer,
                locks,
            )
            calibration_save_seconds.append(time.perf_counter() - started)
            if trainer.optimizer_step == core.CALIBRATION_STEPS:
                cadence = core._select_cadence(
                    calibration_step_seconds, calibration_save_seconds
                )
                core._write_json(out / "cadence-runtime.json", cadence)

        crossed = [
            token_target
            for token_target in targets
            if token_target not in (int(key) for key in scheduled)
            and trainer.tokens_seen >= token_target
        ]
        for token_target in crossed:
            scheduled[str(token_target)] = _scheduled_eval(
                token_target=token_target,
                out=out,
                source_sha=source_sha,
                spec=spec,
                tok=tok,
                manifest=manifest,
                run=run,
                cfg=cfg,
                trainer=trainer,
                locks=locks,
                corpus=corpus,
                interval_rows=interval_rows,
            )
            interval_rows = []

        if (
            cadence is not None
            and trainer.optimizer_step
            % int(cadence["checkpoint_every_optimizer_steps"])
            == 0
            and trainer.tokens_seen < RESUME_TOKEN_TARGET
        ):
            core._save(
                out / "recovery-latest",
                source_sha,
                spec,
                tok,
                manifest,
                run,
                cfg,
                trainer,
                locks,
            )

    if cadence is None:
        raise Scale141RuntimeError("runtime checkpoint cadence was not calibrated")
    if not RESUME_TOKEN_TARGET <= trainer.tokens_seen <= RESUME_TOKEN_TARGET + MAX_TOKEN_OVERSHOOT:
        raise Scale141RuntimeError("phase1 actual optimized-token boundary invalid")

    result = {
        "schema": "12-6.scale141-phase1.v2",
        "source_sha": source_sha,
        "process": {"pid": os.getpid(), "python_executable": sys.executable},
        "model": {
            "parameter_count": core.EXPECTED_PARAMETERS,
            "model_spec_sha256": spec.identity_sha256(),
            "init_spec_sha256": init.identity_sha256(),
            "initial_state_sha256": initial_state,
        },
        "checkpoint0": checkpoint0,
        "cadence": cadence,
        "scheduled": scheduled,
        "observer": observer.summary(),
        "optimizer_step": trainer.optimizer_step,
        "tokens_seen": trainer.tokens_seen,
        "data_position": _data_position(trainer.optimizer_step),
        "target_optimized_tokens": RESUME_TOKEN_TARGET,
        "token_accounting": "ACTUAL_NON_IGNORED_SHIFTED_LABELS",
    }
    result["identity_sha256"] = hash_json(result)
    core._write_json(out / "phase1.json", result)
    return result


def resume(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    _install_runtime_contract()
    manifest, tok, spec, init, cfg, locks, run = core._common(
        repo, source_sha, out, False
    )
    corpus = out / "corpus-a"
    phase1_report = core._read_json(out / "phase1.json")
    cadence = core._read_json(out / "cadence-runtime.json")
    resume_path = _checkpoint_path(out, RESUME_TOKEN_TARGET)
    integrity = verify_checkpoint(resume_path)

    torch.manual_seed(core.SEED)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")
    loaded = load_trainer_checkpoint(
        resume_path,
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=True,
        expected_git_sha=source_sha,
        expected_model_spec_hash=spec.identity_sha256(),
        expected_tokenizer_hash=tok.identity.config_sha256,
        expected_dataset_manifest_hash=manifest["corpus_identity_sha256"],
    )
    if loaded.manifest["identity"]["run_manifest_hash"] != run["identity_sha256"]:
        raise Scale141RuntimeError("resume run-manifest mismatch")
    if trainer.optimizer_step != phase1_report["optimizer_step"]:
        raise Scale141RuntimeError("resume optimizer-step mismatch")
    if trainer.tokens_seen != phase1_report["tokens_seen"]:
        raise Scale141RuntimeError("resume optimized-token mismatch")
    if not RESUME_TOKEN_TARGET <= trainer.tokens_seen <= RESUME_TOKEN_TARGET + MAX_TOKEN_OVERSHOOT:
        raise Scale141RuntimeError("resume optimized-token boundary invalid")

    heldout_recheck = core._fixed_eval(
        model,
        corpus,
        manifest,
        tok,
        split="validation",
        windows=core.HELDOUT_WINDOWS_PER_MODALITY,
    )
    prior_bpb = phase1_report["scheduled"][str(RESUME_TOKEN_TARGET)]["heldout"][
        "bits_per_byte"
    ]
    if not math.isclose(
        heldout_recheck["bits_per_byte"], prior_bpb, rel_tol=0.0, abs_tol=1e-9
    ):
        raise Scale141RuntimeError(
            "held-out metric changed after verified fresh-process reload"
        )
    if os.getpid() == phase1_report["process"]["pid"]:
        raise Scale141RuntimeError("resume did not occur in a fresh process")

    observer = TrainingObserver(
        run, device="cpu", max_step_samples=MAX_OPTIMIZER_STEPS + 8
    )
    iterators = core._train_iters(
        corpus, manifest, tok, trainer.optimizer_step
    )
    interval_rows: list[dict[str, Any]] = []
    scheduled = dict(phase1_report["scheduled"])
    targets = (1_500_000, TARGET_OPTIMIZED_TOKENS)
    first_resumed_step: int | None = None

    while trainer.tokens_seen < TARGET_OPTIMIZED_TOKENS:
        if trainer.optimizer_step >= MAX_OPTIMIZER_STEPS:
            raise Scale141RuntimeError("resume hit optimizer-step safety ceiling")
        target = _next_threshold(trainer.tokens_seen, targets)
        index = trainer.optimizer_step
        stratum = core.MIXTURE[index % len(core.MIXTURE)]
        row = core._train_transition(
            observer,
            trainer,
            core._next_batch(iterators[stratum]),
            scheduled=_should_capture_update(trainer.tokens_seen, target),
        )
        if not 1 <= int(row["tokens"]) <= SEQ - 1:
            raise Scale141RuntimeError(
                f"invalid actual optimized-token count: {row['tokens']}"
            )
        interval_rows.append(row)
        _append_train_row(out / "train-curve.jsonl", row, stratum)
        if first_resumed_step is None:
            first_resumed_step = trainer.optimizer_step

        crossed = [
            token_target
            for token_target in targets
            if str(token_target) not in scheduled and trainer.tokens_seen >= token_target
        ]
        for token_target in crossed:
            scheduled[str(token_target)] = _scheduled_eval(
                token_target=token_target,
                out=out,
                source_sha=source_sha,
                spec=spec,
                tok=tok,
                manifest=manifest,
                run=run,
                cfg=cfg,
                trainer=trainer,
                locks=locks,
                corpus=corpus,
                interval_rows=interval_rows,
            )
            interval_rows = []

        if (
            trainer.optimizer_step
            % int(cadence["checkpoint_every_optimizer_steps"])
            == 0
            and trainer.tokens_seen < TARGET_OPTIMIZED_TOKENS
        ):
            core._save(
                out / "recovery-latest",
                source_sha,
                spec,
                tok,
                manifest,
                run,
                cfg,
                trainer,
                locks,
            )

    if first_resumed_step != phase1_report["optimizer_step"] + 1:
        raise Scale141RuntimeError("fresh-process continuation boundary failed")
    if not TARGET_OPTIMIZED_TOKENS <= trainer.tokens_seen <= TARGET_OPTIMIZED_TOKENS + MAX_TOKEN_OVERSHOOT:
        raise Scale141RuntimeError("final actual optimized-token boundary invalid")
    if trainer.tokens_seen >= core.TRAIN_CORPUS_BYTES:
        raise Scale141RuntimeError("campaign would imply corpus replay")

    final_path = _checkpoint_path(out, TARGET_OPTIMIZED_TOKENS)
    final_integrity = verify_checkpoint(final_path)
    modality_tokens: dict[str, int] = {"uk": 0, "en": 0, "code": 0}
    with (out / "train-curve.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            modality_tokens[str(row["stratum"])] += int(row["tokens"])

    report = {
        "schema": SCHEMA,
        "authority": core.AUTHORITY,
        "source_sha": source_sha,
        "incumbent_search_decision": "FALLBACK_NO_VERIFIED_LEARNED_10M",
        "preexecution_audit": {
            "nominal_padded_token_count_rejected": True,
            "actual_trainer_token_accounting_required": True,
            "streaming_eval_batch_windows": 1,
            "v1_scaffold_is_not_execution_authority": True,
        },
        "model": {
            "parameter_count": core.EXPECTED_PARAMETERS,
            "model_spec_sha256": spec.identity_sha256(),
            "init_spec_sha256": init.identity_sha256(),
            "max_sequence_length": spec.max_seq_len,
            "executed_sequence_length": SEQ,
        },
        "tokenizer": run["tokenizer"],
        "corpus": {
            "identity_sha256": manifest["corpus_identity_sha256"],
            "train_byte_tokens": core.TRAIN_CORPUS_BYTES,
            "target_optimized_tokens": TARGET_OPTIMIZED_TOKENS,
            "optimized_tokens": trainer.tokens_seen,
            "threshold_overshoot_tokens": trainer.tokens_seen - TARGET_OPTIMIZED_TOKENS,
            "optimized_tokens_by_modality": modality_tokens,
            "fraction_of_one_train_corpus": trainer.tokens_seen / core.TRAIN_CORPUS_BYTES,
            "corpus_replay": False,
        },
        "optimizer": run["prepared_s3_lineage"]["optimizer"],
        "run_manifest_identity_sha256": run["identity_sha256"],
        "cadence": cadence,
        "checkpoint_integrity": {
            "pre_resume_checkpoint_id": integrity["checkpoint_id"],
            "final_checkpoint_id": final_integrity["checkpoint_id"],
        },
        "fresh_process_resume": {
            "phase1_pid": phase1_report["process"]["pid"],
            "resume_pid": os.getpid(),
            "loaded_step": phase1_report["optimizer_step"],
            "loaded_optimized_tokens": phase1_report["tokens_seen"],
            "first_resumed_step": first_resumed_step,
            "heldout_bpb_before_stop": prior_bpb,
            "heldout_bpb_after_reload": heldout_recheck["bits_per_byte"],
            "metric_recheck_passed": True,
            "data_position_before_resume": phase1_report["data_position"],
        },
        "scheduled": scheduled,
        "scaling_fit": core.SCALING_FIT,
        "observer_phase1": phase1_report["observer"],
        "observer_resume": observer.summary(),
        "success": {
            "exact_10m_geometry": spec.parameter_count() == core.EXPECTED_PARAMETERS,
            "verified_checkpoint_before_continuation": True,
            "heldout_metric_rechecked_before_continuation": True,
            "fresh_process_resume": True,
            "optimized_token_target": trainer.tokens_seen >= TARGET_OPTIMIZED_TOKENS,
            "optimized_token_overshoot_bounded": trainer.tokens_seen - TARGET_OPTIMIZED_TOKENS <= MAX_TOKEN_OVERSHOOT,
            "actual_token_accounting": True,
            "streaming_eval_memory_bound": True,
            "no_corpus_replay": trainer.tokens_seen < core.TRAIN_CORPUS_BYTES,
            "paid_compute": False,
            "scientific_fallback_executed": True,
        },
    }
    report["report_sha256"] = hash_json(report)
    core._write_json(out / "report.json", report)
    return report


def validate(path: Path, expected_source_sha: str | None = None) -> dict[str, Any]:
    report = core._read_json(path)
    supplied = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if supplied != hash_json(unsigned):
        raise Scale141RuntimeError("report self-hash mismatch")
    if report.get("schema") != SCHEMA:
        raise Scale141RuntimeError("unexpected report schema")
    if expected_source_sha is not None and report.get("source_sha") != expected_source_sha:
        raise Scale141RuntimeError("report source SHA mismatch")
    if report["model"]["parameter_count"] != core.EXPECTED_PARAMETERS:
        raise Scale141RuntimeError("report parameter count mismatch")
    if report["corpus"]["optimized_tokens"] < TARGET_OPTIMIZED_TOKENS:
        raise Scale141RuntimeError("optimized-token target not reached")
    if report["corpus"]["threshold_overshoot_tokens"] > MAX_TOKEN_OVERSHOOT:
        raise Scale141RuntimeError("optimized-token overshoot not bounded")
    if report["corpus"]["corpus_replay"] is not False:
        raise Scale141RuntimeError("report admits corpus replay")
    required = (
        "exact_10m_geometry",
        "verified_checkpoint_before_continuation",
        "heldout_metric_rechecked_before_continuation",
        "fresh_process_resume",
        "optimized_token_target",
        "optimized_token_overshoot_bounded",
        "actual_token_accounting",
        "streaming_eval_memory_bound",
        "no_corpus_replay",
        "scientific_fallback_executed",
    )
    if not all(report["success"].get(key) is True for key in required):
        raise Scale141RuntimeError("one or more required scientific gates failed")
    if report["success"].get("paid_compute") is not False:
        raise Scale141RuntimeError("paid compute boundary violated")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("phase1", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--repo-root", type=Path, default=Path("."))
        p.add_argument("--source-sha", required=True)
        p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("validate")
    p.add_argument("report", type=Path)
    p.add_argument("--expected-source-sha")
    args = parser.parse_args()
    if args.command == "phase1":
        value = phase1(
            args.repo_root.resolve(), args.source_sha, args.output_dir.resolve()
        )
    elif args.command == "resume":
        value = resume(
            args.repo_root.resolve(), args.source_sha, args.output_dir.resolve()
        )
    else:
        value = validate(args.report, args.expected_source_sha)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
