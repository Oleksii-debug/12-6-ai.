"""RESEARCH-192 clean 1M -> 3M -> 10M fixed-recipe scaling transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from twelve_six import milestone150_learned_base_ladder as m150
from twelve_six.checkpoint import hash_json
from twelve_six.model import ModelSpec

SCHEMA = "12-6.research192-scaling-transfer.v1"
ARM_SCHEMA = "12-6.research192-scaling-transfer-arm.v1"
AUTHORITY = "LOCAL_FREE_FIXED_RECIPE_SCALING_TRANSFER_NOT_STAGE_PROMOTION"
REPOSITORY = "Oleksii-debug/12-6-ai."
BRANCH = "research192/one-three-ten-million-20260826"
EXPECTED_CORPUS_ID = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
EXPECTED_EVALUATION_ID = "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"
EXPECTED_TOKEN_BUDGETS = {500: 474_377, 1000: 948_504}
PAIRED_SEEDS = (1337, 1338)
M150_PRODUCER = {
    "source_sha": "5838cd16869dcfcf762368d8673eddf52d51b7e3",
    "workflow_run_id": 32937411703,
    "artifact_id": 9595677772,
    "artifact_name": "milestone150-learned-base-ladder-v1",
    "artifact_sha256": "c00b7e9006320f8916c739a3311e8cc47ad0d0b16957f8ebd7d19233fd9f1c71",
    "ladder_report_sha256": "1f8350bed574a7b78778f0ebb7854ca5311173006820ec27110122f8965c9a5a",
    "one_m_report_identity_sha256": "1b63e8f5096c43b9a36923ddd9d4b8d8a8d1705559f63080c0a287c5520fc738",
}


def _model_payload(
    *, d_model: int, n_layers: int, n_heads: int, d_ff: int
) -> dict[str, Any]:
    if d_model % n_heads:
        raise ValueError("d_model must divide evenly across heads")
    head_dim = d_model // n_heads
    return {
        "schema_version": 1,
        "vocab_size": 256,
        "max_seq_len": 256,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_heads,
        "head_dim": head_dim,
        "d_ff": d_ff,
        "activation": "swiglu",
        "norm_kind": "rmsnorm",
        "norm_placement": "pre",
        "norm_eps": 1e-5,
        "position_embedding": "rope",
        "rope_theta": 10_000.0,
        "rope_rotary_dim": head_dim,
        "attention_bias": False,
        "mlp_bias": False,
        "attention_dropout": 0.0,
        "final_norm": True,
        "tie_word_embeddings": True,
        "lm_head_bias": False,
    }


SCALE_SPECS: dict[str, dict[str, Any]] = {
    "1m": {
        "expected_parameters": 1_037_696,
        "expected_model_spec_sha256": "ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07",
        "provenance": "MILESTONE-150 accepted fixed-control incumbent",
        "model": _model_payload(d_model=128, n_layers=5, n_heads=8, d_ff=352),
    },
    "3m": {
        "expected_parameters": 3_221_184,
        "expected_model_spec_sha256": "3255ebffea76d17e59a19b4de50be616b27e85593a6eebec0db935d7efebb5ea",
        "provenance": (
            "RESEARCH-138 requested ~3.221M bridge transferred into the MILESTONE-150 "
            "MHA/context-256 fixed-control family"
        ),
        "model": _model_payload(d_model=192, n_layers=7, n_heads=12, d_ff=530),
    },
    "10m": {
        "expected_parameters": 10_000_640,
        "expected_model_spec_sha256": "f01cf22d3a44bd72be74691ca4b4a75b093851f45fc2b252c5116eb72370dc53",
        "provenance": (
            "fixed-control MHA/context-256 transfer geometry; parameter-matched to the "
            "10,000,640 S3 count without importing S3 GQA/context/runtime changes"
        ),
        "model": _model_payload(d_model=256, n_layers=12, n_heads=16, d_ff=736),
    },
}

ARM_MATRIX = (
    ("1m", 1337),
    ("1m", 1338),
    ("3m", 1337),
    ("3m", 1338),
    ("10m", 1337),
)


class Research192Error(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Research192Error(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _self_hash(payload: dict[str, Any], key: str = "report_sha256") -> dict[str, Any]:
    value = dict(payload)
    value[key] = hash_json(value)
    return value


def _configure(scale: str, seed: int) -> None:
    if scale not in SCALE_SPECS:
        raise Research192Error(f"unknown scale {scale}")
    if (scale, seed) not in ARM_MATRIX:
        raise Research192Error(f"unpreregistered arm {scale}/seed{seed}")
    m150.SCALE_SPECS = {scale: SCALE_SPECS[scale]}
    m150.SCALE_ORDER = (scale,)
    m150.SEED = seed
    m150.BRANCH = BRANCH
    m150.AUTHORITY = AUTHORITY
    spec = m150.model_spec(scale)
    expected = SCALE_SPECS[scale]
    if spec.parameter_count() != expected["expected_parameters"]:
        raise Research192Error("parameter-count drift")
    if spec.identity_sha256() != expected["expected_model_spec_sha256"]:
        raise Research192Error("ModelSpec identity drift")


def arm_prepare(repo: Path, source_sha: str, out: Path, scale: str, seed: int) -> None:
    _configure(scale, seed)
    truth = m150.prepare(repo, source_sha, out)
    if truth["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise Research192Error("corpus identity drift")
    if truth["evaluation_identity"]["identity_sha256"] != EXPECTED_EVALUATION_ID:
        raise Research192Error("evaluation identity drift")


def arm_phase1(repo: Path, source_sha: str, out: Path, scale: str, seed: int) -> None:
    _configure(scale, seed)
    result = m150.phase1(repo, source_sha, out, scale)
    if int(result["optimizer_step"]) != 500:
        raise Research192Error("phase1 did not stop at optimizer step 500")
    if int(result["optimized_tokens"]) != EXPECTED_TOKEN_BUDGETS[500]:
        raise Research192Error("phase1 optimized-token budget drift")


def arm_resume(repo: Path, source_sha: str, out: Path, scale: str, seed: int) -> None:
    _configure(scale, seed)
    report = m150.resume(repo, source_sha, out, scale)
    if int(report["training"]["optimized_tokens"]) != EXPECTED_TOKEN_BUDGETS[1000]:
        raise Research192Error("final optimized-token budget drift")
    if not bool(report["resume"]["fresh_process"]):
        raise Research192Error("resume was not a fresh process")
    if not bool(report["resume"]["passed"]):
        raise Research192Error("fresh-process resume proof failed")


def arm_verify(repo: Path, source_sha: str, out: Path, scale: str, seed: int) -> None:
    _configure(scale, seed)
    report = m150.verify_scale(repo, source_sha, out, scale)
    if report["fresh_verification"]["status"] != "PASS":
        raise Research192Error("fresh retained-checkpoint verification failed")


def _curve_by_step(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["optimizer_step"])] = row
    return rows


def _checkpoint_row(
    *, report: dict[str, Any], phase1: dict[str, Any], curve: dict[int, dict[str, Any]], step: int
) -> dict[str, Any]:
    if step not in EXPECTED_TOKEN_BUDGETS:
        raise Research192Error("checkpoint step is outside preregistered common budgets")
    eval_row = report["evaluation"]["checkpoints"][str(step)]
    train = curve[step]
    optimized_tokens = int(train["tokens_seen"])
    if optimized_tokens != EXPECTED_TOKEN_BUDGETS[step]:
        raise Research192Error(f"optimized-token mismatch at step {step}")
    train_bpb = float(train["loss"]) / math.log(2.0)
    heldout_bpb = float(eval_row["bits_per_byte"])
    if step == 500:
        wall_seconds = float(phase1["wall_seconds"])
        peak_rss = int(phase1["peak_rss_bytes"])
    else:
        wall_seconds = float(report["compute"]["total_train_wall_seconds"])
        peak_rss = int(report["compute"]["peak_rss_bytes"])
    n_params = int(report["model"]["parameter_count"])
    return {
        "optimizer_step": step,
        "optimized_tokens": optimized_tokens,
        "heldout_bpb": heldout_bpb,
        "ua_bpb": float(eval_row["by_stratum"]["uk"]["bits_per_byte"]),
        "en_bpb": float(eval_row["by_stratum"]["en"]["bits_per_byte"]),
        "code_bpb": float(eval_row["by_stratum"]["code"]["bits_per_byte"]),
        "online_training_bpb": train_bpb,
        "generalization_gap_bpb": heldout_bpb - train_bpb,
        "compute_proxy_6nt": 6 * n_params * optimized_tokens,
        "wall_seconds_end_to_end": wall_seconds,
        "parameter_bytes_fp32": 4 * n_params,
        "peak_rss_bytes": peak_rss,
        "optimized_tokens_per_end_to_end_second": optimized_tokens / max(wall_seconds, 1e-12),
        "gradient_norm": float(train["grad_norm"]),
    }


def arm_summarize(out: Path, scale: str, seed: int) -> dict[str, Any]:
    _configure(scale, seed)
    report = _read_json(out / scale / "report.json")
    phase1 = _read_json(out / scale / "phase1.json")
    curve = _curve_by_step(out / scale / "train-curve.jsonl")
    if report["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise Research192Error("arm corpus mismatch")
    if report["evaluation_identity_sha256"] != EXPECTED_EVALUATION_ID:
        raise Research192Error("arm evaluation mismatch")
    if int(report["training"]["trainer_config"]["seed"]) != seed:
        raise Research192Error("arm seed mismatch")
    if report["fresh_verification"]["status"] != "PASS":
        raise Research192Error("arm verification is not PASS")
    checkpoints = {
        str(step): _checkpoint_row(report=report, phase1=phase1, curve=curve, step=step)
        for step in EXPECTED_TOKEN_BUDGETS
    }
    arm = _self_hash(
        {
            "schema": ARM_SCHEMA,
            "authority": AUTHORITY,
            "scale": scale,
            "seed": seed,
            "source": report["source"],
            "model_spec_sha256": report["model"]["spec_sha256"],
            "parameter_count": report["model"]["parameter_count"],
            "init_spec_sha256": report["model"]["init_spec_sha256"],
            "corpus_identity_sha256": report["corpus_identity_sha256"],
            "evaluation_identity_sha256": report["evaluation_identity_sha256"],
            "tokenizer": report["tokenizer"],
            "training_contract": {
                "trainer_config": report["training"]["trainer_config"],
                "batch_size": report["training"]["batch_size"],
                "sequence_length": report["training"]["sequence_length"],
                "checkpoint_steps_used_for_comparison": sorted(EXPECTED_TOKEN_BUDGETS),
            },
            "checkpoints": checkpoints,
            "resume": report["resume"],
            "fresh_verification": report["fresh_verification"]["status"],
            "truth_boundary": report["truth_boundary"],
        }
    )
    _write_json(out / "research192-arm.json", arm)
    return arm


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_incumbent(m150_root: Path, one_m_arm: dict[str, Any]) -> dict[str, Any]:
    report_path = m150_root / "1m" / "report.json"
    curve_path = m150_root / "1m" / "train-curve.jsonl"
    ladder_path = m150_root / "ladder-report.json"
    report = _read_json(report_path)
    ladder = _read_json(ladder_path)
    if report["identity_sha256"] != M150_PRODUCER["one_m_report_identity_sha256"]:
        raise Research192Error("M150 incumbent 1M report identity mismatch")
    if ladder["report_sha256"] != M150_PRODUCER["ladder_report_sha256"]:
        raise Research192Error("M150 incumbent ladder identity mismatch")
    if report["source"]["git_sha"] != M150_PRODUCER["source_sha"]:
        raise Research192Error("M150 incumbent source mismatch")
    if report["corpus_identity_sha256"] != EXPECTED_CORPUS_ID:
        raise Research192Error("M150 incumbent corpus mismatch")
    if report["evaluation_identity_sha256"] != EXPECTED_EVALUATION_ID:
        raise Research192Error("M150 incumbent evaluation mismatch")
    curve = _curve_by_step(curve_path)
    comparisons = {}
    for step, expected_tokens in EXPECTED_TOKEN_BUDGETS.items():
        old_tokens = int(curve[step]["tokens_seen"])
        if old_tokens != expected_tokens:
            raise Research192Error("M150 incumbent token budget mismatch")
        old_eval = float(report["evaluation"]["checkpoints"][str(step)]["bits_per_byte"])
        new_eval = float(one_m_arm["checkpoints"][str(step)]["heldout_bpb"])
        comparisons[str(step)] = {
            "optimized_tokens": expected_tokens,
            "incumbent_heldout_bpb": old_eval,
            "research192_rerun_heldout_bpb": new_eval,
            "absolute_delta_bpb": new_eval - old_eval,
        }
    return {
        "producer": M150_PRODUCER,
        "incumbent_report_file_sha256": _sha256_file(report_path),
        "incumbent_curve_file_sha256": _sha256_file(curve_path),
        "comparison": comparisons,
        "role": (
            "accepted incumbent reused as a bound reproducibility control; RESEARCH-192 reruns "
            "1M because checkpoint-level wall/paired-seed measurements are required by this study"
        ),
    }


def _nonseed_contract(arm: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(arm["training_contract"]["trainer_config"])
    cfg.pop("seed", None)
    return {
        "corpus": arm["corpus_identity_sha256"],
        "evaluation": arm["evaluation_identity_sha256"],
        "tokenizer": arm["tokenizer"],
        "trainer_config_except_seed": cfg,
        "batch_size": arm["training_contract"]["batch_size"],
        "sequence_length": arm["training_contract"]["sequence_length"],
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compare(arms_root: Path, m150_root: Path, out: Path) -> dict[str, Any]:
    arm_files = sorted(arms_root.rglob("research192-arm.json"))
    arms = [_read_json(path) for path in arm_files]
    by_key = {(str(a["scale"]), int(a["seed"])): a for a in arms}
    if set(by_key) != set(ARM_MATRIX):
        raise Research192Error(f"arm matrix incomplete: {sorted(by_key)}")

    reference_contract = _nonseed_contract(by_key[("1m", 1337)])
    for arm in arms:
        if _nonseed_contract(arm) != reference_contract:
            raise Research192Error("non-size experimental variable drift")
        for step, tokens in EXPECTED_TOKEN_BUDGETS.items():
            if int(arm["checkpoints"][str(step)]["optimized_tokens"]) != tokens:
                raise Research192Error("hidden token advantage detected")

    incumbent = _validate_incumbent(m150_root, by_key[("1m", 1337)])
    per_checkpoint: dict[str, Any] = {}
    for step, tokens in EXPECTED_TOKEN_BUDGETS.items():
        rows = []
        for scale, seed in ARM_MATRIX:
            arm = by_key[(scale, seed)]
            row = dict(arm["checkpoints"][str(step)])
            row.update(
                {
                    "scale": scale,
                    "seed": seed,
                    "parameter_count": int(arm["parameter_count"]),
                    "model_spec_sha256": arm["model_spec_sha256"],
                }
            )
            rows.append(row)
        per_checkpoint[str(step)] = {"optimized_tokens": tokens, "rows": rows}

    seed_pair_summary = {}
    for step in EXPECTED_TOKEN_BUDGETS:
        one = [
            float(by_key[("1m", s)]["checkpoints"][str(step)]["heldout_bpb"])
            for s in PAIRED_SEEDS
        ]
        three = [
            float(by_key[("3m", s)]["checkpoints"][str(step)]["heldout_bpb"])
            for s in PAIRED_SEEDS
        ]
        deltas = [a - b for a, b in zip(one, three)]
        seed_pair_summary[str(step)] = {
            "paired_seeds": list(PAIRED_SEEDS),
            "one_m_mean_heldout_bpb": _mean(one),
            "three_m_mean_heldout_bpb": _mean(three),
            "paired_bpb_improvements_1m_to_3m": deltas,
            "mean_paired_bpb_improvement_1m_to_3m": _mean(deltas),
            "consistent_direction": all(d > 0 for d in deltas) or all(d < 0 for d in deltas),
            "research140_promotion_status": "DESCRIPTIVE_TWO_SEED_NOT_PROMOTION_AUTHORITY",
        }

    pairwise = []
    for step, tokens in EXPECTED_TOKEN_BUDGETS.items():
        for left, right in (("1m", "3m"), ("3m", "10m")):
            common_seeds = sorted(
                set(s for sc, s in ARM_MATRIX if sc == left)
                & set(s for sc, s in ARM_MATRIX if sc == right)
            )
            for seed in common_seeds:
                a = by_key[(left, seed)]
                b = by_key[(right, seed)]
                ar = a["checkpoints"][str(step)]
                br = b["checkpoints"][str(step)]
                improvement = float(ar["heldout_bpb"]) - float(br["heldout_bpb"])
                added_params = int(b["parameter_count"]) - int(a["parameter_count"])
                incremental_compute = 6 * added_params * tokens
                pairwise.append(
                    {
                        "optimizer_step": step,
                        "optimized_tokens": tokens,
                        "seed": seed,
                        "from": left,
                        "to": right,
                        "heldout_bpb_improvement": improvement,
                        "added_parameters": added_params,
                        "quality_improvement_per_added_parameter": improvement / added_params,
                        "incremental_compute_proxy_6_delta_n_t": incremental_compute,
                        "quality_improvement_per_incremental_compute": improvement
                        / incremental_compute,
                    }
                )

    result = _self_hash(
        {
            "schema": SCHEMA,
            "authority": AUTHORITY,
            "source": {"repository": REPOSITORY, "branch": BRANCH},
            "frozen_recipe": reference_contract,
            "scale_specs": {
                scale: {
                    "parameter_count": cfg["expected_parameters"],
                    "model_spec_sha256": cfg["expected_model_spec_sha256"],
                    "model": cfg["model"],
                    "provenance": cfg["provenance"],
                }
                for scale, cfg in SCALE_SPECS.items()
            },
            "common_optimized_token_checkpoints": EXPECTED_TOKEN_BUDGETS,
            "hidden_token_advantage": False,
            "arm_matrix": [
                {"scale": scale, "seed": seed} for scale, seed in ARM_MATRIX
            ],
            "ambiguous_neighbor_pair": {
                "pair": ["1m", "3m"],
                "paired_seeds": list(PAIRED_SEEDS),
                "reason": "RESEARCH-138 identified the ~3.2M bridge as the highest-value interpolation point",
            },
            "incumbent_reuse": incumbent,
            "checkpoints": per_checkpoint,
            "paired_seed_summary": seed_pair_summary,
            "pairwise_scaling_efficiency": pairwise,
            "definitions": {
                "online_training_bpb": "optimizer-step minibatch causal NLL divided by ln(2)",
                "generalization_gap_bpb": "heldout BPB minus online training BPB",
                "compute_proxy": "6 * trainable_parameters * actual_optimized_tokens",
                "wall_seconds_end_to_end": (
                    "phase1 wall through step500, or phase1+fresh-resume wall through step1000; "
                    "includes scheduled evaluation/checkpoint overhead"
                ),
                "parameter_bytes_fp32": "trainable_parameters * 4; optimizer/gradient state excluded",
            },
            "truth_boundary": {
                "local_free_only": True,
                "paid_compute": False,
                "foreign_pretrained_weights": False,
                "sft": False,
                "rlhf": False,
                "dpo": False,
                "representative_external_corpus_claim": False,
                "stage_promotion": False,
                "universal_scaling_law_claim": False,
            },
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "research192-scaling-comparison.json", result)
    return result


def validate_static_contract() -> None:
    for scale, cfg in SCALE_SPECS.items():
        spec = ModelSpec.from_dict(dict(cfg["model"]))
        if spec.parameter_count() != cfg["expected_parameters"]:
            raise Research192Error(f"{scale} parameter count drift")
        if spec.identity_sha256() != cfg["expected_model_spec_sha256"]:
            raise Research192Error(f"{scale} ModelSpec hash drift")
        if spec.n_heads != spec.n_kv_heads or spec.max_seq_len != 256:
            raise Research192Error(f"{scale} left fixed-control MHA/context family")
    if tuple(scale for scale, seed in ARM_MATRIX if scale == "1m") != ("1m", "1m"):
        raise Research192Error("1M paired-seed preregistration drift")
    if tuple(seed for scale, seed in ARM_MATRIX if scale == "3m") != PAIRED_SEEDS:
        raise Research192Error("3M paired-seed preregistration drift")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("prepare", "phase1", "resume", "verify", "summarize"):
        q = sub.add_parser(name)
        q.add_argument("--repo", type=Path, default=Path("."))
        q.add_argument("--source-sha", required=name not in {"summarize"})
        q.add_argument("--out", type=Path, required=True)
        q.add_argument("--scale", choices=sorted(SCALE_SPECS), required=True)
        q.add_argument("--seed", type=int, required=True)
    q = sub.add_parser("compare")
    q.add_argument("--arms-root", type=Path, required=True)
    q.add_argument("--m150-root", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    sub.add_parser("validate-static")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-static":
        validate_static_contract()
        return 0
    if args.command == "compare":
        compare(args.arms_root, args.m150_root, args.out)
        return 0
    if args.command == "prepare":
        arm_prepare(args.repo, args.source_sha, args.out, args.scale, args.seed)
    elif args.command == "phase1":
        arm_phase1(args.repo, args.source_sha, args.out, args.scale, args.seed)
    elif args.command == "resume":
        arm_resume(args.repo, args.source_sha, args.out, args.scale, args.seed)
    elif args.command == "verify":
        arm_verify(args.repo, args.source_sha, args.out, args.scale, args.seed)
    elif args.command == "summarize":
        arm_summarize(args.out, args.scale, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
