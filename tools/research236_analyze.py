from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from research236_prerequisite_gate import (
    DATA25_ID,
    EVAL_ID,
    MATCHED_OPTIMIZED_TOKENS,
    MODEL_SPECS,
    PAIRED_SEEDS,
    TOKENIZER_ID,
    WORKER_ID,
)

# Every directly paired BPB metric below is scored on the same immutable bytes
# for both corpus arms. Corpus-specific training loss is retained separately.
COMMON_EVAL_METRICS = (
    "data25_selection_bpb",
    "external_selection_bpb",
    "common_real_holdout_bpb",
    "ua_bpb",
    "en_bpb",
    "code_bpb",
    "data25_train_probe_bpb",
    "external_train_probe_bpb",
)


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    pos = q * (len(values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def _exact_paired_bootstrap(deltas: list[float]) -> dict[str, float]:
    n = len(deltas)
    means = [statistics.fmean(deltas[i] for i in draw) for draw in itertools.product(range(n), repeat=n)]
    return {
        "p05": _quantile(means, 0.05),
        "p50": _quantile(means, 0.50),
        "p95": _quantile(means, 0.95),
    }


def _validate_cell(cell: dict[str, Any], *, scale: str, corpus: str, seed: int) -> None:
    if cell.get("scale") != scale or cell.get("corpus") != corpus or int(cell.get("seed", -1)) != seed:
        raise ValueError(f"cell identity mismatch for {scale}/{corpus}/seed{seed}")
    if cell.get("tokenizer") != TOKENIZER_ID:
        raise ValueError("tokenizer mismatch")
    if cell.get("evaluation_identity") != EVAL_ID:
        raise ValueError("evaluation identity mismatch")
    spec = MODEL_SPECS[scale]
    if int(cell.get("parameters", -1)) != spec["parameters"]:
        raise ValueError("parameter-count mismatch")
    if cell.get("model_spec_sha256") != spec["model_spec_sha256"]:
        raise ValueError("ModelSpec mismatch")

    actual = int(cell.get("actual_optimized_loss_tokens", -1))
    if actual != MATCHED_OPTIMIZED_TOKENS:
        raise ValueError(f"optimized-token mismatch: {actual}")
    if int(cell.get("source_loss_tokens_consumed", -1)) != MATCHED_OPTIMIZED_TOKENS:
        raise ValueError("source/loss token exposure mismatch")
    if int(cell.get("unique_source_loss_token_positions", -1)) != MATCHED_OPTIMIZED_TOKENS:
        raise ValueError("unique source/loss token position mismatch")
    if int(cell.get("repeated_source_loss_token_positions", -1)) != 0:
        raise ValueError("source/loss token repetition detected")
    if int(cell.get("padded_tensor_positions_counted", -1)) != 0:
        raise ValueError("padded tensor positions were counted as exposure")
    if cell.get("fresh_random_initialization") is not True:
        raise ValueError("arm did not start from random initialization")
    if corpus == "data25" and cell.get("training_corpus_identity") != DATA25_ID:
        raise ValueError("DATA-25 identity mismatch")
    if corpus == "external_real" and not cell.get("training_corpus_identity"):
        raise ValueError("external-real corpus identity missing")

    metrics = cell.get("metrics", {})
    train_bpb = float(metrics["train_bpb"])
    if not math.isfinite(train_bpb):
        raise ValueError("nonfinite metric: train_bpb")
    for metric in COMMON_EVAL_METRICS:
        value = float(metrics[metric])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite metric: {metric}")

    family = cell.get("source_family_bpb")
    if not isinstance(family, dict) or not family:
        raise ValueError("source-family heldout metrics missing")
    if any(not math.isfinite(float(value)) for value in family.values()):
        raise ValueError("nonfinite source-family BPB")


def _paired_summary(rows: dict[int, dict[str, dict[str, Any]]], metric: str) -> dict[str, Any]:
    deltas: list[float] = []
    for seed in PAIRED_SEEDS:
        d25 = float(rows[seed]["data25"]["metrics"][metric])
        ext = float(rows[seed]["external_real"]["metrics"][metric])
        deltas.append(ext - d25)
    wins_external = sum(delta < 0 for delta in deltas)
    wins_data25 = sum(delta > 0 for delta in deltas)
    if wins_external == len(deltas):
        direction = "EXTERNAL_REAL_LOWER_BPB_ALL_PAIRED_SEEDS"
    elif wins_data25 == len(deltas):
        direction = "DATA25_LOWER_BPB_ALL_PAIRED_SEEDS"
    else:
        direction = "MIXED_DIRECTION"
    return {
        "definition": "external_real_trained_minus_data25_trained on identical evaluation bytes; negative favors external-real training for BPB",
        "paired_deltas": deltas,
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "external_real_wins": wins_external,
        "data25_wins": wins_data25,
        "ties": len(deltas) - wins_external - wins_data25,
        "exact_empirical_bootstrap": _exact_paired_bootstrap(deltas),
        "direction": direction,
    }


def _descriptive_training(rows: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for corpus in ("data25", "external_real"):
        values = [float(rows[seed][corpus]["metrics"]["train_bpb"]) for seed in PAIRED_SEEDS]
        out[corpus] = {
            "per_seed": values,
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "comparison_warning": "training BPB is corpus-conditional and is not a same-bytes head-to-head evaluation",
        }
    return out


def _source_family_summary(rows: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    reference_keys: set[str] | None = None
    for seed in PAIRED_SEEDS:
        for corpus in ("data25", "external_real"):
            keys = set(rows[seed][corpus]["source_family_bpb"])
            if reference_keys is None:
                reference_keys = keys
            elif keys != reference_keys:
                raise ValueError("source-family heldout key mismatch across paired arms")
    families = sorted(reference_keys or [])
    if not families:
        raise ValueError("source-family heldout set empty")

    by_corpus: dict[str, Any] = {}
    for corpus in ("data25", "external_real"):
        means: dict[str, float] = {}
        per_seed: dict[str, list[float]] = {}
        for family in families:
            vals = [float(rows[seed][corpus]["source_family_bpb"][family]) for seed in PAIRED_SEEDS]
            per_seed[family] = vals
            means[family] = statistics.fmean(vals)
        worst = max(means, key=means.get)
        best = min(means, key=means.get)
        by_corpus[corpus] = {
            "per_seed": per_seed,
            "mean_bpb_by_family": means,
            "worst_family": worst,
            "worst_family_mean_bpb": means[worst],
            "best_family": best,
            "best_family_mean_bpb": means[best],
            "sensitivity_spread_bpb": means[worst] - means[best],
        }

    paired_by_family: dict[str, Any] = {}
    for family in families:
        deltas = [
            float(rows[seed]["external_real"]["source_family_bpb"][family])
            - float(rows[seed]["data25"]["source_family_bpb"][family])
            for seed in PAIRED_SEEDS
        ]
        paired_by_family[family] = {
            "definition": "external_real_trained_minus_data25_trained; negative favors external-real training",
            "paired_deltas": deltas,
            "mean_delta": statistics.fmean(deltas),
            "median_delta": statistics.median(deltas),
            "exact_empirical_bootstrap": _exact_paired_bootstrap(deltas),
        }
    return {
        "heldout_families": families,
        "by_training_corpus": by_corpus,
        "paired_delta_by_family": paired_by_family,
    }


def _memorization_proxy(rows: dict[int, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    # Negative own-train-probe minus own-selection BPB means the model fits exposed
    # training material more tightly than held-out material. It is a bounded
    # memorization/generalization diagnostic, not a privacy-leakage measurement.
    per_corpus: dict[str, list[float]] = {"data25": [], "external_real": []}
    for seed in PAIRED_SEEDS:
        d25 = rows[seed]["data25"]["metrics"]
        ext = rows[seed]["external_real"]["metrics"]
        per_corpus["data25"].append(float(d25["data25_train_probe_bpb"]) - float(d25["data25_selection_bpb"]))
        per_corpus["external_real"].append(
            float(ext["external_train_probe_bpb"]) - float(ext["external_selection_bpb"])
        )
    delta = [per_corpus["external_real"][i] - per_corpus["data25"][i] for i in range(len(PAIRED_SEEDS))]
    return {
        "definition": "own training-probe BPB minus own selection BPB; more negative indicates stronger exposure-specific fit",
        "privacy_leakage_claim": False,
        "data25": {
            "per_seed": per_corpus["data25"],
            "mean": statistics.fmean(per_corpus["data25"]),
        },
        "external_real": {
            "per_seed": per_corpus["external_real"],
            "mean": statistics.fmean(per_corpus["external_real"]),
        },
        "paired_external_minus_data25": {
            "per_seed": delta,
            "mean": statistics.fmean(delta),
            "exact_empirical_bootstrap": _exact_paired_bootstrap(delta),
        },
    }


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("worker_id") != WORKER_ID:
        raise ValueError("worker id mismatch")
    if payload.get("data230_terminal_identity") in {None, ""}:
        raise ValueError("missing terminal DATA-230 identity")
    if payload.get("common_real_holdout_identity") in {None, ""}:
        raise ValueError("missing common real holdout identity")

    scales = payload.get("scales", {})
    if "500k" not in scales:
        raise ValueError("500k scale is mandatory")
    allowed = {"500k", "1m"}
    if not set(scales).issubset(allowed):
        raise ValueError("unexpected scale")

    out_scales: dict[str, Any] = {}
    for scale, cells in scales.items():
        rows: dict[int, dict[str, dict[str, Any]]] = {}
        for corpus in ("data25", "external_real"):
            corpus_cells = cells.get(corpus, [])
            by_seed = {int(cell["seed"]): cell for cell in corpus_cells}
            if set(by_seed) != set(PAIRED_SEEDS):
                raise ValueError(f"paired seed set mismatch for {scale}/{corpus}")
            for seed, cell in by_seed.items():
                _validate_cell(cell, scale=scale, corpus=corpus, seed=seed)
                rows.setdefault(seed, {})[corpus] = cell

        summaries = {metric: _paired_summary(rows, metric) for metric in COMMON_EVAL_METRICS}
        gaps: dict[str, list[float]] = {"data25": [], "external_real": []}
        for seed in PAIRED_SEEDS:
            for corpus in gaps:
                metrics = rows[seed][corpus]["metrics"]
                gaps[corpus].append(float(metrics["common_real_holdout_bpb"]) - float(metrics["train_bpb"]))

        out_scales[scale] = {
            "training_bpb": _descriptive_training(rows),
            "paired_same_bytes_metric_summaries": summaries,
            "cross_corpus_transfer": {
                "on_data25_selection": summaries["data25_selection_bpb"],
                "on_external_selection": summaries["external_selection_bpb"],
            },
            "memorization_proxy": _memorization_proxy(rows),
            "generalization_gap_common_real_minus_train": {
                corpus: {
                    "per_seed": vals,
                    "mean": statistics.fmean(vals),
                    "median": statistics.median(vals),
                }
                for corpus, vals in gaps.items()
            },
            "source_family_sensitivity": _source_family_summary(rows),
        }

    conclusion = {
        "external_real_automatically_better": False,
        "directional_500k_common_real_holdout": out_scales["500k"]["paired_same_bytes_metric_summaries"][
            "common_real_holdout_bpb"
        ]["direction"],
        "scale_1m_executed": "1m" in out_scales,
        "claim_scope": "paired corpus-origin ablation only; no universal corpus superiority claim",
    }
    return {
        "schema": "12-6.research236-corpus-origin-ablation-result.v1",
        "worker_id": WORKER_ID,
        "matched_actual_optimized_tokens": MATCHED_OPTIMIZED_TOKENS,
        "paired_seeds": list(PAIRED_SEEDS),
        "scales": out_scales,
        "conclusion": conclusion,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(_load(args.input))
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
