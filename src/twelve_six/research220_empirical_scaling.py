"""RESEARCH-220 empirical reporting over the frozen RESEARCH-192 scale producer."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from twelve_six import research192_scaling_transfer as r192
from twelve_six import research212_contract_recovery as r212
from twelve_six.checkpoint import hash_json

SCHEMA = "12-6.research220-one-three-ten-million-v2.v1"
WORKER_ID = "RESEARCH-220-ONE-THREE-TEN-MILLION-V2"
AUTHORITY = "LOCAL_FREE_FIXED_RECIPE_EMPIRICAL_SCALING_THREE_SIZE_DIAGNOSTIC"
RESEARCH212_CONTRACT_IDENTITY = "458cbb22e43bb405029fc256f4d9f29f3ab6b81bcab0db69c9b8cde5d6d5798a"
M150_ONE_M_RANDOM_INIT = "630671c032f4a000a98bc3bf74422e04ed2d6badba32e31b049349d6be9b99f2"
CLIP_THRESHOLD = 1.0


class Research220Error(RuntimeError):
    pass


def readj(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Research220Error(f"{path} must contain a JSON object")
    return value


def writej(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def selfhash(value: dict[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["identity_sha256"] = hash_json(payload)
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise Research220Error(f"{label}: {actual!r} != {expected!r}")


def _read_curve(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise Research220Error(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    if not rows:
        raise Research220Error(f"empty training curve: {path}")
    expected_steps = list(range(1, r192.FINAL_STEP + 1))
    actual_steps = [int(row["optimizer_step"]) for row in rows]
    _require_equal(actual_steps, expected_steps, f"{path} optimizer-step sequence")
    return rows


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise Research220Error("cannot take percentile of empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _grad_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: list[float] = []
    for row in rows:
        raw = row.get("gradient_norm_pre_clip")
        if raw is None:
            raise Research220Error("gradient_norm_pre_clip missing from frozen train curve")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise Research220Error("non-finite or negative gradient norm")
        values.append(value)
    clipped = sum(value > CLIP_THRESHOLD for value in values)
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "clip_threshold": CLIP_THRESHOLD,
        "clip_count": clipped,
        "clip_rate": clipped / len(values),
    }


def curve_telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute preregistered-boundary gradient/clip telemetry without changing training."""
    result: dict[str, Any] = {}
    previous = 0
    for step in r192.CHECKPOINT_STEPS:
        cumulative = rows[:step]
        interval = rows[previous:step]
        boundary = rows[step - 1]
        _require_equal(int(boundary["optimizer_step"]), step, "curve boundary optimizer step")
        _require_equal(
            int(boundary["optimized_tokens"]),
            r192.EXPECTED_TOKEN_BUDGETS[step],
            f"curve boundary optimized tokens step {step}",
        )
        result[str(step)] = {
            "optimizer_step": step,
            "optimized_tokens": int(boundary["optimized_tokens"]),
            "checkpoint_gradient_norm_pre_clip": float(boundary["gradient_norm_pre_clip"]),
            "cumulative": _grad_stats(cumulative),
            "since_prior_boundary": _grad_stats(interval),
        }
        previous = step
    return result


def _arm_inventory(arms_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    inventory: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(arms_root.rglob("research192-arm.json")):
        arm = readj(path)
        key = (str(arm["scale"]), int(arm["seed"]))
        if key in inventory:
            raise Research220Error(f"duplicate arm {key}")
        scale = key[0]
        proof = readj(path.parent / scale / "fresh-verify.json")
        _require_equal(proof.get("status"), "PASS", f"{key} fresh verification")
        verified = {int(row["optimizer_step"]): row for row in proof["checkpoints"]}
        _require_equal(set(verified), set(r192.CHECKPOINT_STEPS), f"{key} verified checkpoint steps")
        for step in r192.CHECKPOINT_STEPS:
            _require_equal(
                int(verified[step]["optimized_tokens"]),
                r192.EXPECTED_TOKEN_BUDGETS[step],
                f"{key} verified tokens step {step}",
            )
        inventory[key] = {
            "arm": arm,
            "root": path.parent,
            "curve": _read_curve(path.parent / scale / "train-curve.jsonl"),
            "phase1": readj(path.parent / scale / "phase1.json"),
            "fresh_verify": proof,
        }
    _require_equal(set(inventory), set(r192.ARM_MATRIX), "executed arm matrix")
    return inventory


def _validate_contract_preflight(preflight: dict[str, Any]) -> dict[str, Any]:
    _require_equal(preflight.get("status"), "PASS", "RESEARCH-212 preflight status")
    _require_equal(preflight.get("corpus_identity_sha256"), r192.EXPECTED_CORPUS_ID, "DATA-25 identity")
    _require_equal(preflight.get("evaluation_identity_sha256"), r192.EXPECTED_EVALUATION_ID, "evaluation identity")
    _require_equal(
        preflight.get("frozen_contract_identity_sha256"),
        RESEARCH212_CONTRACT_IDENTITY,
        "frozen contract identity",
    )
    _require_equal(
        preflight.get("resolved_contract_identity_sha256"),
        RESEARCH212_CONTRACT_IDENTITY,
        "resolved contract identity",
    )
    expected_ledger = {str(k): v for k, v in r192.EXPECTED_TOKEN_BUDGETS.items()}
    _require_equal(preflight.get("optimized_token_ledger"), expected_ledger, "optimized-token ledger")
    expected_models = {
        scale: (cfg["expected_parameters"], cfg["expected_model_spec_sha256"])
        for scale, cfg in r192.SCALE_SPECS.items()
    }
    actual_models = {
        str(row["scale"]): (int(row["parameter_count"]), str(row["model_spec_sha256"]))
        for row in preflight.get("model_construction", [])
    }
    _require_equal(actual_models, expected_models, "fixed MHA model family")
    truth = preflight.get("truth_boundary", {})
    _require_equal(truth.get("paid_compute"), False, "preflight paid-compute boundary")
    _require_equal(truth.get("optimizer_updates"), 0, "preflight optimizer updates")
    return {
        "status": "PASS",
        "source_sha": preflight["source_sha"],
        "contract_identity_sha256": RESEARCH212_CONTRACT_IDENTITY,
        "preflight_identity_sha256": preflight["identity_sha256"],
        "source_exposure_fraction_final": preflight["source_exposure_fraction_final"],
    }


def _normalized_trainer(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result.pop("max_steps", None)
    return result


def _validate_m150_control(
    inventory: dict[tuple[str, int], dict[str, Any]], m150_root: Path
) -> dict[str, Any]:
    report = readj(m150_root / "1m" / "report.json")
    ladder = readj(m150_root / "ladder-report.json")
    _require_equal(report.get("identity_sha256"), r192.M150_PRODUCER["one_m_report_identity_sha256"], "M150 1M report")
    _require_equal(ladder.get("report_sha256"), r192.M150_PRODUCER["ladder_report_sha256"], "M150 ladder report")

    current = inventory[("1m", 1337)]
    arm = current["arm"]
    phase1 = current["phase1"]
    _require_equal(phase1.get("random_init_state_sha256"), M150_ONE_M_RANDOM_INIT, "M150 random-init reproducibility")
    _require_equal(report["model"]["random_init_state_sha256"], M150_ONE_M_RANDOM_INIT, "M150 recorded random-init")
    _require_equal(arm["model_spec_sha256"], report["model"]["spec_sha256"], "M150 1M ModelSpec")
    _require_equal(arm["init_spec_sha256"], report["model"]["init_spec_sha256"], "M150 InitSpec")
    _require_equal(arm["corpus_identity_sha256"], r192.EXPECTED_CORPUS_ID, "M150/current corpus identity")
    _require_equal(arm["evaluation_identity_sha256"], r192.EXPECTED_EVALUATION_ID, "M150/current evaluation identity")
    for field in ("version", "config_sha256", "vocab_sha256", "vocab_size", "special_tokens"):
        _require_equal(arm["tokenizer"][field], report["tokenizer"][field], f"M150 tokenizer {field}")
    _require_equal(arm["packing"]["sequence_length"], report["training"]["sequence_length"], "M150 sequence length")
    _require_equal(arm["batch_size"], report["training"]["batch_size"], "M150 batch size")
    _require_equal(
        _normalized_trainer(arm["trainer_config"]),
        _normalized_trainer(report["training"]["trainer_config"]),
        "M150 AdamW/non-duration trainer recipe",
    )
    return {
        "status": "PASS",
        "artifact_id": r192.M150_PRODUCER["artifact_id"],
        "artifact_sha256": r192.M150_PRODUCER["artifact_sha256"],
        "one_m_report_identity_sha256": report["identity_sha256"],
        "ladder_report_sha256": ladder["report_sha256"],
        "one_m_seed_1337_random_init_state_sha256": M150_ONE_M_RANDOM_INIT,
        "random_init_exact_match": True,
        "role": (
            "reproducibility control for exact 1M initialization/recipe identities; "
            "M150's 474377/948504-token checkpoints are not substituted for the RESEARCH-220 common boundaries"
        ),
    }


def _row_index(comparison: dict[str, Any]) -> dict[tuple[int, str, int], dict[str, Any]]:
    result: dict[tuple[int, str, int], dict[str, Any]] = {}
    for raw_step, block in comparison["checkpoints"].items():
        step = int(raw_step)
        _require_equal(int(block["optimized_tokens"]), r192.EXPECTED_TOKEN_BUDGETS[step], f"comparison T step {step}")
        for row in block["rows"]:
            key = (step, str(row["scale"]), int(row["seed"]))
            if key in result:
                raise Research220Error(f"duplicate comparison row {key}")
            result[key] = dict(row)
    return result


def _mean(values: list[float]) -> float:
    if not values:
        raise Research220Error("empty mean")
    return statistics.fmean(values)


def summarize(
    *, comparison_path: Path, arms_root: Path, m150_root: Path, contract_preflight: Path, out: Path
) -> dict[str, Any]:
    r212.require_frozen_contract()
    r192.validate_static_contract()
    comparison = readj(comparison_path)
    inventory = _arm_inventory(arms_root)
    preflight = _validate_contract_preflight(readj(contract_preflight))
    _require_equal(
        {str(arm["arm"]["source_sha"]) for arm in inventory.values()},
        {str(preflight["source_sha"])},
        "all arm source SHAs vs preflight",
    )
    m150_control = _validate_m150_control(inventory, m150_root)

    telemetry = {key: curve_telemetry(value["curve"]) for key, value in inventory.items()}
    indexed = _row_index(comparison)
    expected_rows = {
        (step, scale, seed)
        for step in r192.CHECKPOINT_STEPS
        for scale, seed in r192.ARM_MATRIX
    }
    _require_equal(set(indexed), expected_rows, "comparison row matrix")

    checkpoints: dict[str, Any] = {}
    for step in r192.CHECKPOINT_STEPS:
        rows: list[dict[str, Any]] = []
        for scale, seed in r192.ARM_MATRIX:
            row = dict(indexed[(step, scale, seed)])
            grad = telemetry[(scale, seed)][str(step)]
            row["gradient"] = grad
            rows.append(row)
        summaries: list[dict[str, Any]] = []
        for scale in ("1m", "3m", "10m"):
            scale_rows = [row for row in rows if row["scale"] == scale]
            summaries.append({
                "scale": scale,
                "parameter_count": int(scale_rows[0]["parameter_count"]),
                "n_preregistered_seeds": len(scale_rows),
                "heldout_bpb_mean": _mean([float(row["heldout_bpb"]) for row in scale_rows]),
                "ua_bpb_mean": _mean([float(row["ua_bpb"]) for row in scale_rows]),
                "en_bpb_mean": _mean([float(row["en_bpb"]) for row in scale_rows]),
                "code_bpb_mean": _mean([float(row["code_bpb"]) for row in scale_rows]),
                "train_bpb_mean": _mean([float(row["training_bpb"]) for row in scale_rows]),
                "generalization_gap_bpb_mean": _mean([float(row["generalization_gap_bpb"]) for row in scale_rows]),
                "checkpoint_grad_norm_mean": _mean([
                    float(row["gradient"]["checkpoint_gradient_norm_pre_clip"]) for row in scale_rows
                ]),
                "cumulative_clip_rate_mean": _mean([
                    float(row["gradient"]["cumulative"]["clip_rate"]) for row in scale_rows
                ]),
                "wall_seconds_mean": _mean([float(row["wall_seconds"]) for row in scale_rows]),
                "throughput_optimized_tokens_per_wall_second_mean": _mean([
                    float(row["throughput_optimized_tokens_per_wall_second"]) for row in scale_rows
                ]),
                "peak_rss_bytes_mean": _mean([float(row["peak_rss_bytes"]) for row in scale_rows]),
                "compute_proxy_6nt": int(scale_rows[0]["compute_proxy_6nt"]),
                "uncertainty_boundary": (
                    "two preregistered paired seeds; descriptive mean only"
                    if len(scale_rows) == 2
                    else "single preregistered seed; no seed-variance claim"
                ),
            })
        checkpoints[str(step)] = {
            "optimized_tokens": r192.EXPECTED_TOKEN_BUDGETS[step],
            "rows": rows,
            "scale_descriptive_means": summaries,
        }

    result = selfhash({
        "schema": SCHEMA,
        "worker_id": WORKER_ID,
        "authority": AUTHORITY,
        "source_sha": preflight["source_sha"],
        "contract_revalidation": preflight,
        "m150_reproducibility_control": m150_control,
        "fixed_recipe": comparison["frozen_non_size_recipe"],
        "scale_specs": comparison["scale_specs"],
        "arm_matrix": comparison["arm_matrix"],
        "common_optimizer_steps": comparison["common_optimizer_steps"],
        "common_optimized_token_budgets": comparison["common_optimized_token_budgets"],
        "hidden_token_advantage": comparison["hidden_token_advantage"],
        "checkpoints": checkpoints,
        "paired_1m_3m": comparison["paired_1m_3m"],
        "quality_gain_efficiency": comparison["pairwise_efficiency"],
        "fresh_checkpoint_verification": {
            "status": "PASS",
            "all_arms": [
                {
                    "scale": scale,
                    "seed": seed,
                    "fresh_verification": inventory[(scale, seed)]["arm"]["fresh_verification"],
                    "checkpoint_ids": [
                        row["checkpoint_id"]
                        for row in inventory[(scale, seed)]["fresh_verify"]["checkpoints"]
                    ],
                }
                for scale, seed in r192.ARM_MATRIX
            ],
        },
        "definitions": {
            **comparison["definitions"],
            "clip_rate": "fraction of recorded optimizer updates with pre-clip global grad norm > frozen clip threshold 1.0",
            "gradient_checkpoint": "pre-clip global grad norm on the update ending exactly at the optimized-token boundary",
            "ua": "DATA-25 held-out 'uk' stratum, labelled UA in this report",
            "quality_gain": "held-out BPB reduction; positive values mean lower BPB at the larger size",
        },
        "truth_boundary": {
            "local_free_only": True,
            "paid_compute": False,
            "foreign_pretrained_weights": False,
            "sft": False,
            "rlhf": False,
            "dpo": False,
            "larger_models_received_extra_optimized_tokens": False,
            "post_hoc_seed_expansion": False,
            "three_sizes_only": True,
            "universal_scaling_law_claim": False,
            "stage_promotion": False,
            "representative_external_corpus_claim": False,
        },
    })
    writej(out, result)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--arms-root", type=Path, required=True)
    p.add_argument("--m150-root", type=Path, required=True)
    p.add_argument("--contract-preflight", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    summarize(
        comparison_path=a.comparison,
        arms_root=a.arms_root,
        m150_root=a.m150_root,
        contract_preflight=a.contract_preflight,
        out=a.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
