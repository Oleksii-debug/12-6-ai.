#!/usr/bin/env python3
"""RECOVER-178 exact-head memorization evidence over the MILESTONE-150 truth model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from twelve_six import milestone100_first_learned as m100
from twelve_six import milestone150_learned_base_ladder as m150
from twelve_six.memorization import (
    aggregate_scores,
    build_canary_suite,
    score_canary,
    stop_diagnostic,
    training_canary_records,
)
from twelve_six.model import TwelveSixDecoder
from twelve_six.training import Trainer

CONFIG = Path("configs/evaluation/recover178_memorization_v1.json")
REPORT_SCHEMA = "12-6.recover178-memorization-scale-report.v1"
FINAL_SCHEMA = "12-6.recover178-memorization-authority.v1"
SCALES = ("100k", "500k", "1m")


class RecoverError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoverError(f"{path} must contain an object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _self_hash(value: dict[str, Any], key: str = "identity_sha256") -> dict[str, Any]:
    out = dict(value)
    out[key] = _hash_json(out)
    return out


def _check_self_hash(value: dict[str, Any], key: str = "identity_sha256") -> None:
    expected = value.get(key)
    unsigned = dict(value)
    unsigned.pop(key, None)
    if not isinstance(expected, str) or expected != _hash_json(unsigned):
        raise RecoverError(f"{key} mismatch")


def _config(repo: Path) -> dict[str, Any]:
    cfg = _read_json(repo / CONFIG)
    if cfg.get("schema_version") != "12-6.recover178-memorization-config.v1":
        raise RecoverError("unsupported RECOVER-178 config schema")
    if tuple(cfg.get("scales", ())) != SCALES:
        raise RecoverError("RECOVER-178 scale order drift")
    checkpoints = tuple(int(v) for v in cfg["checkpoint_optimizer_steps"])
    if checkpoints != (0, 250, 500, 750, 1000):
        raise RecoverError("checkpoint schedule drift")
    return cfg


def _suite(cfg: dict[str, Any]):
    return build_canary_suite(
        seed=int(cfg["seed"]),
        exposures=tuple(int(v) for v in cfg["exposures_per_cycle"]),
        replicas=int(cfg["replicas_per_exposure"]),
        continuation_chars=int(cfg["continuation_chars"]),
    )


def _assert_public_safe(value: Any, path: str = "$") -> None:
    forbidden = {"text", "prefix", "continuation", "source_text", "canary_text"}
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden:
                raise RecoverError(f"unsafe public field {path}.{key}")
            _assert_public_safe(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_public_safe(child, f"{path}[{index}]")


def _schedule(cfg: dict[str, Any], suite: Any, cycle: int) -> list[dict[str, Any] | None]:
    cycle_steps = int(cfg["exposure_cycle_optimizer_steps"])
    events: list[dict[str, Any] | None] = list(training_canary_records(suite))
    if len(events) > cycle_steps:
        raise RecoverError("canary exposure events exceed cycle capacity")
    events.extend([None] * (cycle_steps - len(events)))
    random.Random(f"recover178:{cfg['seed']}:{cycle}").shuffle(events)
    return events


def _schedule_identity(cfg: dict[str, Any], suite: Any, cycles: int) -> str:
    public = []
    for cycle in range(cycles):
        for offset, event in enumerate(_schedule(cfg, suite, cycle)):
            public.append(
                {
                    "cycle": cycle,
                    "offset": offset,
                    "canary_id": None if event is None else str(event["canary_id"]),
                }
            )
    return _hash_json(public)


def _inject_canary(base_batch: dict[str, torch.Tensor], tokenizer: Any, text: str) -> dict[str, torch.Tensor]:
    ids = tokenizer.encode(text)
    if len(ids) < 2 or len(ids) > m150.SEQ:
        raise RecoverError("canary sequence length outside comparable packing limit")
    input_ids = base_batch["input_ids"].clone()
    labels = base_batch["labels"].clone()
    input_ids[0].zero_()
    labels[0].fill_(-100)
    encoded = torch.tensor(ids, dtype=torch.long)
    input_ids[0, : len(ids)] = encoded
    labels[0, : len(ids)] = encoded
    return {"input_ids": input_ids, "labels": labels}


def _packed_train_counts(corpus: Path, manifest: dict[str, Any], tok: Any) -> dict[str, int]:
    counts = {
        stratum: sum(1 for _ in m100._packed(corpus, manifest, tok, "train", stratum))
        for stratum in m150.STRATA
    }
    if any(value <= 0 for value in counts.values()):
        raise RecoverError("DATA-25 train stratum has no packed examples")
    return counts


def _safe_scores(model: TwelveSixDecoder, tok: Any, suite: Any, observed: dict[str, int], alternatives: int):
    scores = []
    for canary in suite.canaries:
        row = score_canary(
            model,
            tok,
            canary,
            observed_exposures=observed.get(canary.canary_id, 0),
            alternative_count=alternatives,
        )
        nll = float(row["nll_per_token"])
        row["mean_log_likelihood_nats_per_token"] = -nll
        row["geometric_mean_token_likelihood"] = math.exp(-nll)
        scores.append(row)
    return scores


def _bind_stop_policy(curve: list[dict[str, Any]], cfg: dict[str, Any], *, model_state_sha256: str) -> dict[str, Any]:
    current = stop_diagnostic(curve, previous_bpb=None, current_bpb=math.inf)
    declared = cfg["stop_policy"]
    thresholds = current["thresholds"]
    if float(thresholds["nll_advantage_nats_per_token"]) < float(declared["minimum_nll_advantage_nats_per_token"]):
        raise RecoverError("current stop policy weakened below declared NLL floor")
    if float(thresholds["exact_recovery_lift"]) < float(declared["minimum_exact_recovery_lift"]):
        raise RecoverError("current stop policy weakened below declared exact-recovery floor")
    return _self_hash(
        {
            "schema": "12-6.recover178-stop-threshold-binding.v1",
            "methodology_id": current["policy_id"],
            "bound_at": "random_init_before_any_optimizer_update",
            "bound_before_optimizer_update": True,
            "model_state_sha256": model_state_sha256,
            "nll_advantage_nats_per_token": float(thresholds["nll_advantage_nats_per_token"]),
            "top_decile_rank": int(thresholds["top_decile_rank"]),
            "rank_percentile_delta": float(declared["rank_percentile_delta"]),
            "exact_recovery_lift": float(thresholds["exact_recovery_lift"]),
            "required_signal_count": int(declared["required_signal_count"]),
            "validation_improvement_required": bool(declared["validation_improvement_required"]),
        }
    )


def _frozen_stop_diagnostic(
    curve: list[dict[str, Any]], *, previous_bpb: float | None, current_bpb: float, binding: dict[str, Any]
) -> dict[str, Any]:
    points = {int(point["exposure_per_cycle"]): point for point in curve}
    control = points[0]
    repeated = points[max(points)]
    nll_advantage = float(control["nll_per_token_median"]) - float(repeated["nll_per_token_median"])
    rank_delta = float(control["rank_percentile_median"]) - float(repeated["rank_percentile_median"])
    exact_lift = float(repeated["exact_recovery_rate"]) - float(control["exact_recovery_rate"])
    signals = {
        "nll": nll_advantage >= float(binding["nll_advantage_nats_per_token"]),
        "rank": float(repeated["rank_median"]) <= int(binding["top_decile_rank"])
        and rank_delta >= float(binding["rank_percentile_delta"]),
        "exact_recovery": exact_lift >= float(binding["exact_recovery_lift"]),
    }
    signal_count = sum(bool(v) for v in signals.values())
    validation_improved = previous_bpb is not None and current_bpb < previous_bpb
    return {
        "policy_id": str(binding["methodology_id"]),
        "threshold_binding_identity_sha256": binding["identity_sha256"],
        "thresholds_frozen_from_random_init": True,
        "validation_improved_since_previous_checkpoint": validation_improved,
        "disproportionate_memorization": signal_count >= int(binding["required_signal_count"]),
        "diagnostic_stop": validation_improved
        and signal_count >= int(binding["required_signal_count"]),
        "signals": signals,
        "observed": {
            "nll_advantage": nll_advantage,
            "rank_percentile_delta": rank_delta,
            "exact_recovery_lift": exact_lift,
        },
        "privacy_claim": "NONE",
    }


def _repetition_record(
    packed_counts: dict[str, int], optimized_base: dict[str, int], consumed_base: dict[str, int]
) -> dict[str, Any]:
    total_available = sum(packed_counts.values())
    total_optimized = sum(optimized_base.values())
    total_consumed = sum(consumed_base.values())
    return {
        "definition": "optimized DATA-25 packed examples divided by one complete packed train corpus",
        "packed_train_examples_available": packed_counts,
        "packed_train_examples_consumed": consumed_base,
        "packed_train_examples_optimized": optimized_base,
        "equivalent_full_corpus_repetitions": total_optimized / total_available,
        "consumption_equivalent_repetitions": total_consumed / total_available,
        "by_stratum_optimized_repetitions": {
            key: optimized_base[key] / packed_counts[key] for key in packed_counts
        },
    }


def _score_checkpoint(
    *,
    model: TwelveSixDecoder,
    trainer: Trainer,
    tok: Any,
    suite: Any,
    observed: dict[str, int],
    corpus: Path,
    manifest: dict[str, Any],
    alternative_count: int,
    previous_bpb: float | None,
    binding: dict[str, Any] | None,
    packed_counts: dict[str, int],
    optimized_base: dict[str, int],
    consumed_base: dict[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    before_hash = m100._state_hash(model)
    before_counters = (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen)
    before_mode = model.training
    heldout = m100._evaluate(model, corpus, manifest, tok)
    scores = _safe_scores(model, tok, suite, observed, alternative_count)
    curve = aggregate_scores(scores)
    after_hash = m100._state_hash(model)
    after_counters = (trainer.micro_step, trainer.optimizer_step, trainer.tokens_seen)
    after_mode = model.training
    if before_hash != after_hash or before_counters != after_counters or before_mode != after_mode:
        raise RecoverError("evaluation mutated model state, counters, or train/eval mode")
    bpb = float(heldout["bits_per_byte"])
    checkpoint = {
        "optimizer_step": trainer.optimizer_step,
        "optimized_tokens": trainer.tokens_seen,
        "heldout": heldout,
        "heldout_bpb": bpb,
        "canary_curve": curve,
        "canary_scores": scores,
        "corpus_repetition_count": _repetition_record(packed_counts, optimized_base, consumed_base),
        "evaluation_non_mutating": True,
        "model_state_sha256": before_hash,
    }
    if binding is not None:
        checkpoint["stop_diagnostic"] = _frozen_stop_diagnostic(
            curve, previous_bpb=previous_bpb, current_bpb=bpb, binding=binding
        )
    return checkpoint, curve, bpb


def prepare(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    cfg = _config(repo)
    manifest, tok, eval_id = m150._common_truth(repo, source_sha, out, build=True)
    suite = _suite(cfg)
    for scale in SCALES:
        m150.model_spec(scale)
    truth = _self_hash(
        {
            "schema": "12-6.recover178-truth.v1",
            "worker_id": cfg["worker_id"],
            "source_sha": source_sha,
            "execution_class": cfg["execution_class"],
            "corpus_identity_sha256": manifest["corpus_identity_sha256"],
            "evaluation_identity": eval_id,
            "tokenizer": {
                "version": tok.identity.version,
                "config_sha256": tok.identity.config_sha256,
                "vocab_sha256": tok.identity.vocab_sha256,
                "vocab_size": tok.identity.vocab_size,
                "special_tokens": dict(tok.identity.special_tokens),
            },
            "scale_model_spec_sha256": {
                scale: m150.model_spec(scale).identity_sha256() for scale in SCALES
            },
            "canary_suite": suite.public(),
            "safety_boundary": cfg["safety_boundary"],
            "ten_million": {
                "status": "NOT_RUN_PREEXISTING_EXACT_GREEN_100K_500K_1M_MATRIX_ABSENT_AT_RECOVERY_START",
                "learned_claim": False,
            },
        }
    )
    _assert_public_safe(truth)
    _write_json(out / "recover178-truth.json", truth)
    return truth


def run_scale(repo: Path, source_sha: str, out: Path, scale: str) -> dict[str, Any]:
    if scale not in SCALES:
        raise RecoverError(f"unsupported scale: {scale}")
    cfg = _config(repo)
    truth = _read_json(out / "recover178-truth.json")
    _check_self_hash(truth)
    manifest, tok, eval_id = m150._common_truth(repo, source_sha, out, build=False)
    if eval_id["identity_sha256"] != truth["evaluation_identity"]["identity_sha256"]:
        raise RecoverError("evaluation identity changed after prepare")

    spec = m150.model_spec(scale)
    init = m150.init_spec()
    trainer_cfg = m150.trainer_config()
    torch.manual_seed(m150.SEED)
    random.seed(m150.SEED)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, trainer_cfg, device="cpu")
    suite = _suite(cfg)
    canary_by_id = {item.canary_id: item for item in suite.canaries}
    observed = {item.canary_id: 0 for item in suite.canaries}
    corpus = out / "corpus-a"
    packed_counts = _packed_train_counts(corpus, manifest, tok)
    optimized_base = {key: 0 for key in packed_counts}
    consumed_base = {key: 0 for key in packed_counts}
    its = m100._train_iters(corpus, manifest, tok, 0)
    batches = {stratum: m100._batches(its[stratum]) for stratum in m150.STRATA}

    checkpoints: list[dict[str, Any]] = []
    initial, initial_curve, initial_bpb = _score_checkpoint(
        model=model,
        trainer=trainer,
        tok=tok,
        suite=suite,
        observed=observed,
        corpus=corpus,
        manifest=manifest,
        alternative_count=int(cfg["alternative_count"]),
        previous_bpb=None,
        binding=None,
        packed_counts=packed_counts,
        optimized_base=optimized_base,
        consumed_base=consumed_base,
    )
    binding = _bind_stop_policy(initial_curve, cfg, model_state_sha256=initial["model_state_sha256"])
    if trainer.optimizer_step != 0:
        raise RecoverError("stop thresholds were not bound before optimizer update")
    scale_out = out / scale
    _write_json(scale_out / "threshold-binding.json", binding)
    initial["stop_diagnostic"] = {
        "policy_id": binding["methodology_id"],
        "threshold_binding_identity_sha256": binding["identity_sha256"],
        "thresholds_frozen_from_random_init": True,
        "diagnostic_stop": False,
        "reason": "random-init threshold-binding checkpoint; no previous held-out BPB",
        "privacy_claim": "NONE",
    }
    checkpoints.append(initial)
    previous_bpb = initial_bpb

    checkpoint_steps = set(int(v) for v in cfg["checkpoint_optimizer_steps"])
    cycle_steps = int(cfg["exposure_cycle_optimizer_steps"])
    max_step = max(checkpoint_steps)
    schedule_cache: dict[int, list[dict[str, Any] | None]] = {}

    for index in range(max_step):
        stratum = m100.MIXTURE[index % len(m100.MIXTURE)]
        base_batch = next(batches[stratum])
        consumed_base[stratum] += m150.BATCH
        cycle = index // cycle_steps
        offset = index % cycle_steps
        events = schedule_cache.setdefault(cycle, _schedule(cfg, suite, cycle))
        event = events[offset]
        batch = base_batch
        if event is None:
            optimized_base[stratum] += m150.BATCH
        else:
            canary_id = str(event["canary_id"])
            canary = canary_by_id[canary_id]
            batch = _inject_canary(base_batch, tok, canary.text)
            observed[canary_id] += 1
            optimized_base[stratum] += m150.BATCH - int(cfg["canary_replacement_rows_per_event"])
        metrics = trainer.train_microbatch(batch)
        if not metrics.optimizer_stepped or metrics.optimizer_step != index + 1:
            raise RecoverError("optimizer trajectory drift")
        if not math.isfinite(metrics.loss):
            raise RecoverError("non-finite training loss")

        if trainer.optimizer_step in checkpoint_steps:
            checkpoint, _, current_bpb = _score_checkpoint(
                model=model,
                trainer=trainer,
                tok=tok,
                suite=suite,
                observed=observed,
                corpus=corpus,
                manifest=manifest,
                alternative_count=int(cfg["alternative_count"]),
                previous_bpb=previous_bpb,
                binding=binding,
                packed_counts=packed_counts,
                optimized_base=optimized_base,
                consumed_base=consumed_base,
            )
            checkpoints.append(checkpoint)
            previous_bpb = current_bpb

    if [item["optimizer_step"] for item in checkpoints] != sorted(checkpoint_steps):
        raise RecoverError("checkpoint trajectory incomplete")
    expected_cycles = math.ceil(max_step / cycle_steps)
    schedule_sha = _schedule_identity(cfg, suite, expected_cycles)
    report = _self_hash(
        {
            "schema": REPORT_SCHEMA,
            "authority": "EXACT_HEAD_LOCAL_FREE_MEMORIZATION_EVIDENCE",
            "worker_id": cfg["worker_id"],
            "source_sha": source_sha,
            "eval136_incumbent_source_sha": cfg["eval136_source_sha"],
            "scale": scale,
            "model_spec": spec.to_dict(),
            "model_spec_sha256": spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "init_spec": init.to_dict(),
            "init_spec_sha256": init.identity_sha256(),
            "corpus_identity_sha256": manifest["corpus_identity_sha256"],
            "evaluation_identity_sha256": eval_id["identity_sha256"],
            "trainer_config": asdict(trainer_cfg),
            "model_init_seed": m150.SEED,
            "canary_schedule_seed": int(cfg["seed"]),
            "canary_suite": suite.public(),
            "schedule": {
                "exposure_cycle_optimizer_steps": cycle_steps,
                "schedule_identity_sha256": schedule_sha,
                "canary_replacement_rows_per_event": int(cfg["canary_replacement_rows_per_event"]),
            },
            "threshold_binding": binding,
            "checkpoints": checkpoints,
            "all_evaluation_non_mutating": all(bool(item["evaluation_non_mutating"]) for item in checkpoints),
            "final_observed_exposures": observed,
            "safety_boundary": {
                "canary_text_emitted": False,
                "source_text_emitted": False,
                "privacy_claim": "NONE",
                "broad_memorization_threshold_claim": False,
            },
        }
    )
    _assert_public_safe(report)
    _write_json(scale_out / "report.json", report)
    return report


def finalize(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    truth = _read_json(out / "recover178-truth.json")
    _check_self_hash(truth)
    reports = {scale: _read_json(out / scale / "report.json") for scale in SCALES}
    for scale, report in reports.items():
        _check_self_hash(report)
        if report["source_sha"] != source_sha or report["scale"] != scale:
            raise RecoverError(f"{scale}: source/scale identity mismatch")
        if report["corpus_identity_sha256"] != truth["corpus_identity_sha256"]:
            raise RecoverError(f"{scale}: corpus identity mismatch")
        if report["evaluation_identity_sha256"] != truth["evaluation_identity"]["identity_sha256"]:
            raise RecoverError(f"{scale}: evaluation identity mismatch")
        if not report["all_evaluation_non_mutating"]:
            raise RecoverError(f"{scale}: evaluation non-mutation failed")
        _assert_public_safe(report)

    final = _self_hash(
        {
            "schema": FINAL_SCHEMA,
            "authority": "EXACT_HEAD_LOCAL_FREE_MEMORIZATION_EVIDENCE",
            "worker_id": "RECOVER-178-EVAL136-MEMORIZATION",
            "source_sha": source_sha,
            "truth_identity_sha256": truth["identity_sha256"],
            "corpus_identity_sha256": truth["corpus_identity_sha256"],
            "evaluation_identity_sha256": truth["evaluation_identity"]["identity_sha256"],
            "scale_order": list(SCALES),
            "scales": {
                scale: {
                    "parameter_count": reports[scale]["parameter_count"],
                    "model_spec_sha256": reports[scale]["model_spec_sha256"],
                    "report_identity_sha256": reports[scale]["identity_sha256"],
                    "threshold_binding_identity_sha256": reports[scale]["threshold_binding"]["identity_sha256"],
                    "checkpoint_optimizer_steps": [c["optimizer_step"] for c in reports[scale]["checkpoints"]],
                    "heldout_bpb": [c["heldout_bpb"] for c in reports[scale]["checkpoints"]],
                    "final_observed_exposures": reports[scale]["final_observed_exposures"],
                    "all_evaluation_non_mutating": True,
                }
                for scale in SCALES
            },
            "ten_million": truth["ten_million"],
            "safety_boundary": {
                "canary_text_emitted": False,
                "source_text_emitted": False,
                "privacy_claim": "NONE",
                "production_readiness_claim": False,
                "intelligence_claim": False,
                "alignment_claim": False,
                "instruction_following_claim": False,
            },
        }
    )
    _assert_public_safe(final)
    _write_json(out / "memorization-report.json", final)
    return final


def validate(path: Path, expected_source_sha: str) -> dict[str, Any]:
    report = _read_json(path)
    _check_self_hash(report)
    if report.get("schema") != FINAL_SCHEMA:
        raise RecoverError("final report schema mismatch")
    if report.get("source_sha") != expected_source_sha:
        raise RecoverError("final report source mismatch")
    if tuple(report.get("scale_order", ())) != SCALES:
        raise RecoverError("final report scale matrix incomplete")
    if report.get("safety_boundary", {}).get("privacy_claim") != "NONE":
        raise RecoverError("privacy claim must remain NONE")
    _assert_public_safe(report)
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run-scale", "finalize"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--repo-root", type=Path, default=Path("."))
        cmd.add_argument("--source-sha", required=True)
        cmd.add_argument("--output-dir", type=Path, required=True)
        if name == "run-scale":
            cmd.add_argument("--scale", choices=SCALES, required=True)
    check = sub.add_parser("validate")
    check.add_argument("path", type=Path)
    check.add_argument("--expected-source-sha", required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate":
        result = validate(args.path.resolve(), args.expected_source_sha)
    else:
        repo = args.repo_root.resolve()
        out = args.output_dir.resolve()
        if args.command == "prepare":
            result = prepare(repo, args.source_sha, out)
        elif args.command == "run-scale":
            result = run_scale(repo, args.source_sha, out, args.scale)
        elif args.command == "finalize":
            result = finalize(repo, args.source_sha, out)
        else:
            raise AssertionError(args.command)
    print(json.dumps({"schema": result.get("schema"), "identity_sha256": result.get("identity_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
