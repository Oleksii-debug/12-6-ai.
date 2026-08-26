"""RECOVER-168: causal long-run recovery for the accepted learned ~1M Base.

This is orchestration only.  It reuses the accepted DATA-25/tokenizer/trainer/
checkpoint/evaluation implementation from LEARN-122 and freezes one 1M-model
trajectory.  No architecture search, foreign weights, or instruction tuning is
performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from twelve_six import learn122_real_500k_long as base
from twelve_six.checkpoint import hash_json, load_trainer_checkpoint, verify_checkpoint
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.model import InitSpec, TwelveSixDecoder
from twelve_six.training import Trainer

SCHEMA = "12-6.recover168-learn121-long-1m.v1"
PROTOCOL_SCHEMA = "12-6.recover168-protocol.v1"
AUTHORITY = "LOCAL_FREE_DATA25_LEARNED_BASE_EVIDENCE_NOT_STAGE_PROMOTION"
BRANCH = "recover168/learn121-long-1m-20260826"
LABEL = "1m"
SEED = 1337
EXPECTED_PARAMS = 1_037_696
MILESTONES = (250_000, 500_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000, 10_000_000)
MIN_REQUIRED_TOKENS = 1_000_000
STOP_POLICY = {
    "name": "causal_first_strict_heldout_regression",
    "selection_metric": "full DATA-25 validation bits_per_byte",
    "decision_timing": "after each preregistered milestone and before any later milestone",
    "rule": "stop before the next milestone when current BPB is strictly greater than the best BPB at any earlier evaluated milestone",
    "minimum_milestone_for_stop_tokens": 500_000,
    "tie_policy": "ties do not trigger stop; earliest minimum is retained as best",
}
PROMPTS = dict(base.PROMPTS) if hasattr(base, "PROMPTS") else {
    "uk": "Українська мова ",
    "en": "The training corpus ",
    "code": "def stable_",
}


class Recover168Error(RuntimeError):
    pass


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Recover168Error(f"{path} must contain a JSON object")
    return value


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _json_stable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _patch_base() -> None:
    if getattr(base, "_RECOVER168_PATCHED", False):
        return
    base.BRANCH = BRANCH
    original_milestones = base._milestones
    original_planned = base._planned_final_target
    original_run_manifest = base._run_manifest

    def milestones(label: str, seed: int) -> tuple[int, ...]:
        if label == LABEL and seed == SEED:
            return MILESTONES
        return original_milestones(label, seed)

    def planned(label: str, seed: int) -> int:
        if label == LABEL and seed == SEED:
            return MILESTONES[-1]
        return original_planned(label, seed)

    def stable_run_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
        value = original_run_manifest(*args, **kwargs)
        stable = _json_stable(value)
        if not isinstance(stable, dict):
            raise Recover168Error("run manifest did not normalize to a JSON object")
        return stable

    base._milestones = milestones
    base._planned_final_target = planned
    base._run_manifest = stable_run_manifest
    base._RECOVER168_PATCHED = True


def _protocol(repo: Path, source_sha: str, root: Path) -> dict[str, Any]:
    manifest = _read(root / "corpus-manifest.json")
    tok = base.ByteTokenizer()
    spec = base._spec(LABEL)
    init = InitSpec()
    cfg = base._trainer_config(SEED)
    locks = base._locks(repo)
    if spec.parameter_count() != EXPECTED_PARAMS:
        raise Recover168Error("accepted 1M parameter count drift")
    value: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": base.REPOSITORY, "git_sha": source_sha, "branch": BRANCH},
        "semantic_incumbent": "LEARN-121 recovered through accepted LEARN-122 mechanics without architecture search",
        "model": {
            "label": LABEL,
            "model_spec": spec.to_dict(),
            "model_spec_sha256": spec.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "init_spec": init.to_dict(),
            "init_spec_sha256": init.identity_sha256(),
            "random_initialization": True,
        },
        "tokenizer": {
            "version": tok.identity.version,
            "config_sha256": tok.identity.config_sha256,
            "vocab_sha256": tok.identity.vocab_sha256,
            "vocab_size": tok.identity.vocab_size,
        },
        "corpus": {
            "identity_sha256": manifest["corpus_identity_sha256"],
            "train_byte_tokens": manifest["by_split"]["train"]["byte_tokens"],
            "validation_byte_tokens": manifest["by_split"]["validation"]["byte_tokens"],
            "train_validation_content_overlap": manifest["train_validation_content_overlap"],
            "train_by_stratum": {s: manifest["by_split_stratum"][f"train:{s}"]["byte_tokens"] for s in ("uk", "en", "code")},
            "truth_boundary": manifest["truth_boundary"],
        },
        "packing": {"version": base.PACKING_VERSION, "sequence_length": base.SEQ, "cross_document": False, "batch_size": base.BATCH},
        "optimizer": {"name": "AdamW", **_json_stable(asdict(cfg))},
        "mixture_pattern": list(base.MIXTURE),
        "preregistered_token_milestones": list(MILESTONES),
        "stop_policy": STOP_POLICY,
        "evaluation": "full DATA-25 validation NLL/BPB with UA/EN/code breakdown and state non-mutation",
        "generation": "first-party checkpoint backend, greedy, 48 new byte tokens",
        "environment_lock_sha256": locks["combined_sha256"],
        "foreign_pretrained_weights": False,
        "sft": False,
        "rlhf": False,
        "dpo": False,
        "paid_compute": False,
        "unsupported_claims": ["intelligence", "production_readiness", "alignment", "instruction_following", "real_world_corpus_representativeness"],
    }
    value["identity_sha256"] = hash_json(value)
    return value


def preflight(repo: Path, source_sha: str) -> dict[str, Any]:
    _patch_base()
    base._require_head(repo, source_sha)
    spec = base._spec(LABEL)
    if spec.parameter_count() != EXPECTED_PARAMS:
        raise Recover168Error("1M incumbent changed")
    retained = _read(repo / base.RETAINED_CORPUS_MANIFEST)
    if retained["corpus_identity_sha256"] != base.EXPECTED_CORPUS_ID:
        raise Recover168Error("DATA-25 retained corpus identity drift")
    if retained["train_validation_content_overlap"] != 0:
        raise Recover168Error("DATA-25 train/validation overlap is nonzero")
    result = {
        "source_sha": source_sha,
        "model_spec_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "corpus_identity_sha256": retained["corpus_identity_sha256"],
        "train_byte_tokens": retained["by_split"]["train"]["byte_tokens"],
        "max_planned_tokens": MILESTONES[-1],
        "planned_fraction_of_train_bytes": MILESTONES[-1] / retained["by_split"]["train"]["byte_tokens"],
        "no_recycling_required_by_budget": MILESTONES[-1] < retained["by_split"]["train"]["byte_tokens"],
        "stop_policy": STOP_POLICY,
    }
    return result


def prepare(repo: Path, source_sha: str, root: Path) -> dict[str, Any]:
    _patch_base()
    base.prepare(repo, source_sha, root)
    value = _protocol(repo, source_sha, root)
    _write(root / "recover168-protocol.json", value)
    return value


def _eval_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [{
        "milestone_requested_tokens": 0,
        "actual_optimized_tokens": 0,
        "optimizer_step": 0,
        "bits_per_byte": float(state["initial_evaluation"]["bits_per_byte"]),
        "by_stratum": state["initial_evaluation"]["by_stratum"],
        "checkpoint": state["checkpoints"][0]["path"],
        "checkpoint_id": state["checkpoints"][0]["checkpoint_id"],
    }]
    for item in state["evaluations"]:
        rows.append({
            "milestone_requested_tokens": int(item["milestone_requested_tokens"]),
            "actual_optimized_tokens": int(item["actual_optimized_tokens"]),
            "optimizer_step": int(item["optimizer_step"]),
            "bits_per_byte": float(item["evaluation"]["bits_per_byte"]),
            "by_stratum": item["evaluation"]["by_stratum"],
            "checkpoint": item["checkpoint"],
            "checkpoint_id": item["checkpoint_id"],
        })
    rows.sort(key=lambda row: row["milestone_requested_tokens"])
    return rows


def _selector(state: dict[str, Any]) -> dict[str, Any]:
    rows = _eval_rows(state)
    best = rows[0]
    trigger = None
    decisions = []
    for row in rows[1:]:
        prior_best = best
        regression = row["bits_per_byte"] > prior_best["bits_per_byte"]
        eligible = row["milestone_requested_tokens"] >= int(STOP_POLICY["minimum_milestone_for_stop_tokens"])
        decision = {
            "milestone_requested_tokens": row["milestone_requested_tokens"],
            "current_bpb": row["bits_per_byte"],
            "prior_best_bpb": prior_best["bits_per_byte"],
            "strict_regression": regression,
            "eligible_to_stop": eligible,
            "stop": bool(regression and eligible and trigger is None),
        }
        decisions.append(decision)
        if regression and eligible and trigger is None:
            trigger = dict(decision)
            trigger["final_checkpoint"] = row["checkpoint"]
            trigger["best_checkpoint"] = prior_best["checkpoint"]
            break
        if row["bits_per_byte"] < best["bits_per_byte"]:
            best = row
    completed = rows[-1]
    return {
        "policy": STOP_POLICY,
        "decisions": decisions,
        "stop_triggered": trigger is not None,
        "stop_trigger": trigger,
        "best": best,
        "final_completed": completed,
    }


def advance(repo: Path, source_sha: str, root: Path, target_tokens: int) -> dict[str, Any]:
    _patch_base()
    if target_tokens not in MILESTONES:
        raise Recover168Error(f"target {target_tokens} is not preregistered")
    run_dir = base._run_dir(root, LABEL, SEED)
    state_path = run_dir / "state.json"
    if state_path.exists():
        existing = _read(state_path)
        selector = _selector(existing)
        if selector["stop_triggered"]:
            skipped = {"status": "SKIPPED_BY_PREDECLARED_OVERFIT_STOP", "target_tokens": target_tokens, "selector": selector}
            _write(root / f"advance-{target_tokens}-skipped.json", skipped)
            return skipped
        if int(existing["latest_optimized_tokens"]) >= target_tokens:
            raise Recover168Error("target is not ahead of existing trajectory")
        resume = True
    else:
        resume = False
    state = base.train_phase(repo, source_sha, root, LABEL, SEED, target_tokens, resume)
    selector = _selector(state)
    _write(root / "selector-state.json", selector)
    if selector["stop_triggered"]:
        _write(root / "STOP.json", selector["stop_trigger"])
    return {"status": "COMPLETED", "target_tokens": target_tokens, "latest_optimized_tokens": state["latest_optimized_tokens"], "selector": selector}


def _curve(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "train-curve.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def _training_stats(curve: list[dict[str, Any]], step: int) -> dict[str, Any]:
    rows = [row for row in curve if int(row["optimizer_step"]) <= step]
    window = rows[-min(100, len(rows)):]
    mean_loss = sum(float(row["loss"]) for row in window) / len(window)
    grads = [float(row["grad_norm"]) for row in rows if row.get("grad_norm") is not None]
    ratios = [float(row["update_to_parameter_ratio"]) for row in rows if row.get("update_to_parameter_ratio") is not None]
    return {
        "trailing_train_steps": len(window),
        "trailing_train_mean_nll": mean_loss,
        "trailing_train_bits_per_byte": mean_loss / math.log(2.0),
        "gradient_samples": len(grads),
        "gradient_norm_mean": sum(grads) / len(grads) if grads else None,
        "gradient_norm_max": max(grads) if grads else None,
        "clip_threshold": 1.0,
        "clip_frequency": sum(1 for value in grads if value > 1.0) / len(grads) if grads else None,
        "update_ratio_samples": len(ratios),
        "update_to_parameter_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "update_to_parameter_ratio_max": max(ratios) if ratios else None,
    }


def _logits_snapshot(checkpoint: Path) -> dict[str, Any]:
    backend = load_first_party_backend(checkpoint)
    outputs = {}
    for name, prompt in PROMPTS.items():
        ids = list(backend.encode(prompt))
        logits = [float(x) for x in backend.next_token_logits(ids)]
        raw = struct.pack("<" + "f" * len(logits), *logits)
        top = sorted(range(len(logits)), key=lambda i: (-logits[i], i))[:8]
        outputs[name] = {
            "prompt": prompt,
            "input_token_count": len(ids),
            "logits_count": len(logits),
            "float32_le_sha256": hashlib.sha256(raw).hexdigest(),
            "argmax_token_id": top[0],
            "top8_token_ids": top,
        }
    return {"backend_diagnostics": backend.diagnostics(), "outputs": outputs}


def _fresh_verify(repo: Path, source_sha: str, root: Path, checkpoint_rel: str, expected_id: str) -> dict[str, Any]:
    _patch_base()
    manifest, tok, locks, protocol = base._common(repo, source_sha, root)
    spec = base._spec(LABEL)
    init = InitSpec()
    cfg = base._trainer_config(SEED)
    run = base._run_manifest(source_sha, LABEL, SEED, spec, init, tok, manifest, cfg, locks, protocol)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, cfg, device="cpu")
    checkpoint = base._run_dir(root, LABEL, SEED) / checkpoint_rel
    checked = verify_checkpoint(checkpoint)
    if checked["checkpoint_id"] != expected_id:
        raise Recover168Error("checkpoint identity changed before fresh verification")
    loaded = load_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=False,
        expected_git_sha=source_sha,
        expected_model_spec_hash=spec.identity_sha256(),
        expected_tokenizer_hash=tok.identity.config_sha256,
        expected_dataset_manifest_hash=manifest["corpus_identity_sha256"],
    )
    if loaded.manifest["identity"]["run_manifest_hash"] != run["identity_sha256"]:
        raise Recover168Error("fresh verification run-manifest binding mismatch")
    before = base._state_hash(model)
    evaluation = base._evaluate(model, root / "corpus-a", manifest, tok)
    after = base._state_hash(model)
    generation = base._generation_preserving_rng(checkpoint)
    logits = _logits_snapshot(checkpoint)
    return {
        "checkpoint": checkpoint_rel,
        "checkpoint_id": checked["checkpoint_id"],
        "loaded_optimizer_step": trainer.optimizer_step,
        "loaded_optimized_tokens": trainer.tokens_seen,
        "load_passed": True,
        "first_party_logits": logits,
        "evaluation": evaluation,
        "model_state_sha256_before_evaluation": before,
        "model_state_sha256_after_evaluation": after,
        "evaluation_non_mutation_passed": before == after and bool(evaluation["non_mutation_passed"]),
        "generation": generation,
    }


def _memorization_controls(root: Path, manifest: dict[str, Any], curve: list[dict[str, Any]], generations: list[dict[str, Any]]) -> dict[str, Any]:
    consumed = {s: sum(int(row["tokens"]) for row in curve if row["stratum"] == s) for s in ("uk", "en", "code")}
    capacities = {s: int(manifest["by_split_stratum"][f"train:{s}"]["byte_tokens"]) for s in ("uk", "en", "code")}
    corpora = {s: "\n".join(str(row["text"]) for row in base._rows(root / "corpus-a", manifest, "train", s)) for s in ("uk", "en", "code")}
    scans = []
    for item in generations:
        milestone = int(item["milestone_requested_tokens"])
        for stratum, output in item["generation"]["outputs"].items():
            text = str(output["text"])
            scans.append({
                "milestone_requested_tokens": milestone,
                "stratum": stratum,
                "generated_chars": len(text),
                "exact_generated_continuation_found_in_same_stratum_training_text": bool(text and len(text) >= 16 and text in corpora[stratum]),
            })
    return {
        "train_validation_content_overlap": int(manifest["train_validation_content_overlap"]),
        "actual_consumed_tokens_by_stratum": consumed,
        "available_train_byte_tokens_by_stratum": capacities,
        "consumption_fraction_by_stratum": {s: consumed[s] / capacities[s] for s in consumed},
        "recycling_occurred": any(consumed[s] > capacities[s] for s in consumed),
        "exact_generation_continuation_scan": scans,
        "any_exact_generation_continuation_match": any(row["exact_generated_continuation_found_in_same_stratum_training_text"] for row in scans),
        "scope_note": "Exact continuation scan is a narrow memorization control, not a general privacy or memorization guarantee.",
    }


def finalize(repo: Path, source_sha: str, root: Path) -> dict[str, Any]:
    _patch_base()
    protocol = _read(root / "recover168-protocol.json")
    if protocol != _protocol(repo, source_sha, root):
        raise Recover168Error("recovery protocol drift")
    run_dir = base._run_dir(root, LABEL, SEED)
    state = _read(run_dir / "state.json")
    if int(state["latest_optimized_tokens"]) < MIN_REQUIRED_TOKENS:
        raise Recover168Error("long-run minimum token budget was not reached")
    selector = _selector(state)
    rows = _eval_rows(state)
    curve = _curve(run_dir)
    by_milestone = []
    for row in rows:
        item = dict(row)
        item["training"] = None if row["optimizer_step"] == 0 else _training_stats(curve, row["optimizer_step"])
        by_milestone.append(item)
    best = selector["best"]
    final = selector["final_completed"]
    best_verify = _fresh_verify(repo, source_sha, root, best["checkpoint"], best["checkpoint_id"])
    final_verify = _fresh_verify(repo, source_sha, root, final["checkpoint"], final["checkpoint_id"])
    manifest = _read(root / "corpus-manifest.json")
    total_wall = sum(float(phase["wall_seconds"]) for phase in state["phases"])
    phase_throughput = []
    for phase in state["phases"]:
        delta = int(phase["end_optimized_tokens"]) - int(phase["start_optimized_tokens"])
        phase_throughput.append({
            "start_tokens": phase["start_optimized_tokens"],
            "end_tokens": phase["end_optimized_tokens"],
            "wall_seconds": phase["wall_seconds"],
            "optimized_tokens_per_second": delta / float(phase["wall_seconds"]) if float(phase["wall_seconds"]) > 0 else None,
            "peak_rss_bytes": phase["peak_rss_bytes"],
            "observer": phase["observer"],
        })
    reproduction = {
        "source_sha": source_sha,
        "python": "3.11.16",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_runner_parts": ["experiments/learn122/runner.00.part", "experiments/learn122/runner.01.part", "experiments/learn122/runner.02.part"],
        "commands": [
            f"python -m twelve_six.recover168_learn121_long_1m prepare --repo-root . --source-sha {source_sha} --output-dir recover168-evidence",
            *[f"python -m twelve_six.recover168_learn121_long_1m advance --repo-root . --source-sha {source_sha} --output-dir recover168-evidence --target-tokens {m}" for m in MILESTONES],
            f"python -m twelve_six.recover168_learn121_long_1m finalize --repo-root . --source-sha {source_sha} --output-dir recover168-evidence",
        ],
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {"repository": base.REPOSITORY, "git_sha": source_sha, "branch": BRANCH},
        "protocol": protocol,
        "model": protocol["model"],
        "tokenizer": protocol["tokenizer"],
        "corpus": protocol["corpus"],
        "optimizer": protocol["optimizer"],
        "optimized_tokens_final": int(state["latest_optimized_tokens"]),
        "optimizer_step_final": int(state["latest_optimizer_step"]),
        "trajectory": by_milestone,
        "selection": selector,
        "best_checkpoint": best,
        "final_checkpoint": final,
        "best_fresh_verification": best_verify,
        "final_fresh_verification": final_verify,
        "fresh_process_resumes": state["fresh_process_resumes"],
        "real_process_termination_and_resume_passed": len(state["fresh_process_resumes"]) >= 1 and all(int(r["first_resumed_optimizer_step"]) == int(r["loaded_optimizer_step"]) + 1 for r in state["fresh_process_resumes"]),
        "compute": {
            "device": "cpu",
            "wall_seconds_total": total_wall,
            "peak_rss_bytes": max(int(phase["peak_rss_bytes"]) for phase in state["phases"]),
            "phase_throughput": phase_throughput,
            "compute_proxy_final": 6 * EXPECTED_PARAMS * int(state["latest_optimized_tokens"]),
        },
        "memorization_controls": _memorization_controls(root, manifest, curve, state["generations"]),
        "raw_base_generation_progression": state["generations"],
        "reproduction": reproduction,
        "claims": {
            "learned_base_artifact": float(final["bits_per_byte"]) < float(rows[0]["bits_per_byte"]),
            "intelligence": None,
            "production_readiness": None,
            "alignment": None,
            "instruction_following": None,
            "real_world_corpus_representativeness": None,
        },
    }
    report["report_sha256"] = hash_json(report)
    _write(root / "report.json", report)
    return report


def validate(path: Path, expected_source_sha: str) -> None:
    report = _read(path)
    if report.get("schema") != SCHEMA:
        raise Recover168Error("report schema mismatch")
    if report["source"]["git_sha"] != expected_source_sha:
        raise Recover168Error("report source SHA mismatch")
    expected = report.pop("report_sha256", None)
    if expected != hash_json(report):
        raise Recover168Error("report self-hash mismatch")
    if report["model"]["parameter_count"] != EXPECTED_PARAMS:
        raise Recover168Error("report parameter count mismatch")
    if not report["real_process_termination_and_resume_passed"]:
        raise Recover168Error("fresh-process resume proof failed")
    for key in ("best_fresh_verification", "final_fresh_verification"):
        proof = report[key]
        if not proof["load_passed"] or not proof["evaluation_non_mutation_passed"]:
            raise Recover168Error(f"fresh checkpoint verification failed: {key}")
        if not proof["first_party_logits"]["outputs"] or not proof["generation"]["outputs"]:
            raise Recover168Error(f"first-party inference proof missing: {key}")
    if report["memorization_controls"]["recycling_occurred"]:
        raise Recover168Error("training data recycling occurred despite frozen budget")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", type=Path, required=True)
    common.add_argument("--source-sha", required=True)
    common.add_argument("--output-dir", type=Path, default=Path("recover168-evidence"))
    sub.add_parser("preflight", parents=[common])
    sub.add_parser("prepare", parents=[common])
    advance_parser = sub.add_parser("advance", parents=[common])
    advance_parser.add_argument("--target-tokens", type=int, required=True)
    sub.add_parser("finalize", parents=[common])
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("report", type=Path)
    validate_parser.add_argument("--expected-source-sha", required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(preflight(args.repo_root.resolve(), args.source_sha), ensure_ascii=False, sort_keys=True))
    elif args.command == "prepare":
        print(json.dumps(prepare(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve()), ensure_ascii=False, sort_keys=True))
    elif args.command == "advance":
        print(json.dumps(advance(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve(), args.target_tokens), ensure_ascii=False, sort_keys=True))
    elif args.command == "finalize":
        print(json.dumps(finalize(args.repo_root.resolve(), args.source_sha, args.output_dir.resolve()), ensure_ascii=False, sort_keys=True))
    else:
        validate(args.report.resolve(), args.expected_source_sha)
        print("RECOVER168_VALIDATE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
