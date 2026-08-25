#!/usr/bin/env python3
"""RESEARCH-138: honest LOCAL_FREE scaling-fit diagnostics for the fixed-control ladder.

This tool deliberately separates a balanced 4-scale x 3-token core grid from longer
same-identity stress observations. Model choice is based on leave-one-parameter-scale-out
backtests, not in-sample R^2. Any ~10M result is an extrapolation stress estimate, not a
universal scaling law or an assertion about the current S3 architecture.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "12-6.research138.scaling-fit.v1"
LN2 = math.log(2.0)
TARGET_PARAMETERS = 10_000_640
TARGET_TOKEN_BUDGETS = (16_632, 65_772, 131_292, 262_332)
MODEL_NAMES = (
    "linear_log",
    "log_power",
    "inverse_quarter",
    "inverse_sqrt",
    "linear_log_t2",
    "log_power_t2",
    "compute_log",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _payload_hash(doc: dict[str, Any]) -> str:
    copy = dict(doc)
    copy.pop("payload_sha256", None)
    return hashlib.sha256(_canonical_json(copy)).hexdigest()


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    aug = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular least-squares normal matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            aug[row] = [a - factor * b for a, b in zip(aug[row], aug[col], strict=True)]
    return [aug[i][-1] for i in range(n)]


def _least_squares(x: list[list[float]], y: list[float]) -> list[float]:
    if not x or len(x) != len(y):
        raise ValueError("invalid least-squares inputs")
    p = len(x[0])
    if any(len(row) != p for row in x):
        raise ValueError("ragged design matrix")
    xtx = [[0.0 for _ in range(p)] for _ in range(p)]
    xty = [0.0 for _ in range(p)]
    for row, target in zip(x, y, strict=True):
        for i in range(p):
            xty[i] += row[i] * target
            for j in range(p):
                xtx[i][j] += row[i] * row[j]
    return _solve(xtx, xty)


def _geomean(values: Iterable[float]) -> float:
    items = list(values)
    return math.exp(sum(math.log(value) for value in items) / len(items))


def _features(name: str, n: float, t: float, n0: float, t0: float) -> list[float]:
    x = math.log(n / n0)
    y = math.log(t / t0)
    if name in {"linear_log", "log_power"}:
        return [1.0, x, y]
    if name == "inverse_quarter":
        return [1.0, (n0 / n) ** 0.25, (t0 / t) ** 0.25]
    if name == "inverse_sqrt":
        return [1.0, math.sqrt(n0 / n), math.sqrt(t0 / t)]
    if name in {"linear_log_t2", "log_power_t2"}:
        return [1.0, x, y, y * y]
    if name == "compute_log":
        return [1.0, math.log((n * t) / (n0 * t0))]
    raise KeyError(name)


def _target_is_log(name: str) -> bool:
    return name in {"log_power", "log_power_t2", "compute_log"}


def _fit(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    n0 = _geomean(float(row["parameters"]) for row in rows)
    t0 = _geomean(float(row["optimized_tokens"]) for row in rows)
    x = [_features(name, float(row["parameters"]), float(row["optimized_tokens"]), n0, t0) for row in rows]
    y0 = [float(row["validation_loss_nats"]) for row in rows]
    y = [math.log(value) for value in y0] if _target_is_log(name) else y0
    coefficients = _least_squares(x, y)
    return {"model": name, "coefficients": coefficients, "n0": n0, "t0": t0}


def _predict(fit: dict[str, Any], n: float, t: float) -> float:
    row = _features(fit["model"], n, t, fit["n0"], fit["t0"])
    value = sum(a * b for a, b in zip(row, fit["coefficients"], strict=True))
    return math.exp(value) if _target_is_log(fit["model"]) else value


def _rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _mae(values: list[float]) -> float:
    return sum(abs(value) for value in values) / len(values)


def _loso(core: list[dict[str, Any]], name: str) -> dict[str, Any]:
    scales = sorted({int(row["parameters"]) for row in core})
    residuals: list[dict[str, Any]] = []
    folds = []
    for scale in scales:
        train = [row for row in core if int(row["parameters"]) != scale]
        test = [row for row in core if int(row["parameters"]) == scale]
        fit = _fit(train, name)
        fold_residuals = []
        for row in test:
            prediction = _predict(fit, float(row["parameters"]), float(row["optimized_tokens"]))
            residual = prediction - float(row["validation_loss_nats"])
            fold_residuals.append(residual)
            residuals.append({
                "model": name,
                "held_out_parameters": scale,
                "optimized_tokens": int(row["optimized_tokens"]),
                "actual_loss_nats": float(row["validation_loss_nats"]),
                "predicted_loss_nats": prediction,
                "residual_nats_pred_minus_actual": residual,
            })
        folds.append({
            "held_out_parameters": scale,
            "rmse_nats": _rmse(fold_residuals),
            "mae_nats": _mae(fold_residuals),
            "bias_nats": sum(fold_residuals) / len(fold_residuals),
            "max_abs_residual_nats": max(abs(value) for value in fold_residuals),
        })
    values = [row["residual_nats_pred_minus_actual"] for row in residuals]
    largest = max(scales)
    largest_fold = next(fold for fold in folds if fold["held_out_parameters"] == largest)
    return {
        "model": name,
        "loso_rmse_nats": _rmse(values),
        "loso_mae_nats": _mae(values),
        "loso_bias_nats": sum(values) / len(values),
        "loso_max_abs_residual_nats": max(abs(value) for value in values),
        "largest_scale_holdout_rmse_nats": largest_fold["rmse_nats"],
        "folds": folds,
        "residuals": residuals,
    }


def _full_diagnostics(core: list[dict[str, Any]], name: str) -> dict[str, Any]:
    fit = _fit(core, name)
    residuals = []
    for row in core:
        prediction = _predict(fit, float(row["parameters"]), float(row["optimized_tokens"]))
        residuals.append(prediction - float(row["validation_loss_nats"]))
    return {
        "fit": fit,
        "in_sample_rmse_nats": _rmse(residuals),
        "in_sample_mae_nats": _mae(residuals),
    }


def _extrapolation_admissible(name: str, fit: dict[str, Any], target_grid: list[tuple[int, int]]) -> tuple[bool, list[str]]:
    reasons = []
    predictions = [_predict(fit, n, t) for n, t in target_grid]
    if not all(math.isfinite(value) and value > 0.0 for value in predictions):
        reasons.append("non_positive_or_non_finite_target_prediction")
    if name in {"linear_log", "linear_log_t2"}:
        reasons.append("unbounded_linear_loss_form_not_used_for_out_of_box_extrapolation")
    if name in {"inverse_quarter", "inverse_sqrt"}:
        floor, a, b = fit["coefficients"]
        if floor < 0 or a < 0 or b < 0:
            reasons.append("non_physical_negative_asymptote_or_inverse_coefficient")
    return not reasons, reasons


def _same_point(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        int(a["parameters"]) == int(b["parameters"])
        and int(a["optimized_tokens"]) == int(b["optimized_tokens"])
        and int(a["seed"]) == int(b["seed"])
    )


def _validate_input(doc: dict[str, Any]) -> None:
    if doc.get("schema") != "12-6.research138.observed-experiments.v1":
        raise ValueError("unexpected observed-experiments schema")
    if doc.get("payload_sha256") != _payload_hash(doc):
        raise ValueError("observed-experiments payload hash mismatch")
    ident = doc["shared_identity"]
    required = (
        "tokenizer_id", "tokenizer_config_sha256", "tokenizer_vocab_sha256",
        "corpus_dataset_identity_sha256", "corpus_manifest_sha256",
        "train_jsonl_sha256", "evaluation_jsonl_sha256", "context", "packing_id",
    )
    missing = [key for key in required if key not in ident]
    if missing:
        raise ValueError(f"missing shared identity fields: {missing}")
    core = doc["core_fit_observations"]
    scales = sorted({int(row["parameters"]) for row in core})
    token_budgets = sorted({int(row["optimized_tokens"]) for row in core})
    if len(scales) != 4 or len(token_budgets) != 3 or len(core) != 12:
        raise ValueError("core must be a balanced 4-scale x 3-token grid")
    for scale in scales:
        if len([row for row in core if int(row["parameters"]) == scale]) != 3:
            raise ValueError("unbalanced parameter-scale core")
    if any(int(row["seed"]) != 1337 for row in core):
        raise ValueError("core fit must use seed 1337 only")
    if ident["context"] != 256 or ident["tokenizer_id"] != "s0-byte-v1":
        raise ValueError("unexpected RESEARCH41 fixed-control identity")


def _stress_diagnostics(doc: dict[str, Any], selected_fit: dict[str, Any], core_tmax: int) -> dict[str, Any]:
    core = doc["core_fit_observations"]
    stress = doc["same_identity_stress_observations"]
    rows = []
    for row in stress:
        duplicate_core = any(_same_point(row, item) for item in core)
        prediction = _predict(selected_fit, float(row["parameters"]), float(row["optimized_tokens"]))
        residual = prediction - float(row["validation_loss_nats"])
        rows.append({
            "source": row["source"],
            "parameters": int(row["parameters"]),
            "optimized_tokens": int(row["optimized_tokens"]),
            "seed": int(row["seed"]),
            "actual_loss_nats": float(row["validation_loss_nats"]),
            "predicted_loss_nats": prediction,
            "residual_nats_pred_minus_actual": residual,
            "duplicate_of_core": duplicate_core,
            "beyond_core_token_range": int(row["optimized_tokens"]) > core_tmax,
        })
    beyond = [row for row in rows if row["beyond_core_token_range"]]
    seed_pairs: dict[int, dict[int, float]] = {}
    for row in stress:
        if int(row["parameters"]) != 467808:
            continue
        key = int(row["optimized_tokens"])
        seed_pairs.setdefault(key, {})[int(row["seed"])] = float(row["validation_loss_nats"])
    seed_deltas = [
        {"optimized_tokens": token, "abs_seed_loss_delta_nats": abs(values[1338] - values[1337])}
        for token, values in sorted(seed_pairs.items()) if 1337 in values and 1338 in values
    ]
    return {
        "rows": rows,
        "beyond_core_max_abs_residual_nats": max(abs(row["residual_nats_pred_minus_actual"]) for row in beyond),
        "seed_1337_1338_deltas_at_467808": seed_deltas,
        "max_seed_delta_nats": max(item["abs_seed_loss_delta_nats"] for item in seed_deltas),
    }


def _prediction_report(
    core: list[dict[str, Any]],
    model_diagnostics: dict[str, Any],
    selected_name: str,
    stress: dict[str, Any],
) -> list[dict[str, Any]]:
    fit = model_diagnostics[selected_name]["full"]["fit"]
    scales = sorted({int(row["parameters"]) for row in core})
    tokens = sorted({int(row["optimized_tokens"]) for row in core})
    nmin, nmax = scales[0], scales[-1]
    tmin, tmax = tokens[0], tokens[-1]
    n_span = math.log(nmax / nmin)
    t_span = math.log(tmax / tmin)
    n_factor = 1.0 + max(0.0, math.log(TARGET_PARAMETERS / nmax)) / n_span
    group_maxes = [fold["max_abs_residual_nats"] for fold in model_diagnostics[selected_name]["loso"]["folds"]]
    base_group_conformal = max(group_maxes)

    best_outer = model_diagnostics[selected_name]["loso"]["largest_scale_holdout_rmse_nats"]
    ensemble_names = []
    for name, diagnostic in model_diagnostics.items():
        if not diagnostic["extrapolation_admissible"]:
            continue
        if diagnostic["loso"]["largest_scale_holdout_rmse_nats"] <= 3.0 * best_outer:
            ensemble_names.append(name)

    result = []
    for token_budget in TARGET_TOKEN_BUDGETS:
        central = _predict(fit, TARGET_PARAMETERS, token_budget)
        t_factor = 1.0 + max(0.0, math.log(token_budget / tmax)) / t_span
        extrapolation_half_width = base_group_conformal * n_factor * t_factor
        ensemble_predictions = [
            _predict(model_diagnostics[name]["full"]["fit"], TARGET_PARAMETERS, token_budget)
            for name in ensemble_names
        ]
        model_form_half_spread = max(abs(value - central) for value in ensemble_predictions)
        relevant_stress = [
            abs(row["residual_nats_pred_minus_actual"])
            for row in stress["rows"]
            if row["beyond_core_token_range"] and int(row["optimized_tokens"]) <= int(token_budget * 1.001)
        ]
        structural_penalty = max(relevant_stress) if relevant_stress else 0.0
        half_width = max(extrapolation_half_width, model_form_half_spread) + structural_penalty
        lower = max(0.0, central - half_width)
        upper = central + half_width
        result.append({
            "parameters": TARGET_PARAMETERS,
            "optimized_tokens": token_budget,
            "compute_proxy": 6 * TARGET_PARAMETERS * token_budget,
            "central_model": selected_name,
            "central_loss_nats": central,
            "central_bpb": central / LN2,
            "empirical_90_interval_loss_nats": [lower, upper],
            "empirical_90_interval_bpb": [lower / LN2, upper / LN2],
            "interval_half_width_nats": half_width,
            "group_conformal_base_half_width_nats": base_group_conformal,
            "parameter_extrapolation_factor": n_factor,
            "token_extrapolation_factor": t_factor,
            "model_form_half_spread_nats": model_form_half_spread,
            "long_horizon_structural_penalty_nats": structural_penalty,
            "ensemble_models": ensemble_names,
            "inside_observed_parameter_range": nmin <= TARGET_PARAMETERS <= nmax,
            "inside_core_token_range": tmin <= token_budget <= tmax,
            "coverage_guarantee": False,
            "label": "EXTRAPOLATION_STRESS_INTERVAL_NOT_CALIBRATED_90_PERCENT_COVERAGE",
        })
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_svg(path: Path, predictions: list[dict[str, Any]]) -> None:
    width, height = 760, 420
    left, right, top, bottom = 75, 25, 35, 65
    xs = [math.log10(row["optimized_tokens"]) for row in predictions]
    ymax = max(row["empirical_90_interval_bpb"][1] for row in predictions) * 1.08
    ymin = 0.0

    def px(x: float) -> float:
        return left + (x - min(xs)) / (max(xs) - min(xs)) * (width - left - right)

    def py(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

    points = " ".join(f"{px(x):.1f},{py(row['central_bpb']):.1f}" for x, row in zip(xs, predictions, strict=True))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        '<text x="380" y="22" text-anchor="middle" font-family="sans-serif" font-size="16">RESEARCH-138 ~10M extrapolation stress bands</text>',
        '<text x="380" y="404" text-anchor="middle" font-family="sans-serif" font-size="13">optimized tokens (log10 axis)</text>',
        '<text x="18" y="210" text-anchor="middle" transform="rotate(-90 18 210)" font-family="sans-serif" font-size="13">held-out BPB</text>',
    ]
    for row, x in zip(predictions, xs, strict=True):
        xpix = px(x)
        low, high = row["empirical_90_interval_bpb"]
        lines.append(f'<line x1="{xpix:.1f}" y1="{py(low):.1f}" x2="{xpix:.1f}" y2="{py(high):.1f}" stroke="#666" stroke-width="2"/>')
        lines.append(f'<circle cx="{xpix:.1f}" cy="{py(row["central_bpb"]):.1f}" r="4" fill="black"/>')
        lines.append(f'<text x="{xpix:.1f}" y="{height-bottom+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{row["optimized_tokens"]}</text>')
    lines.append(f'<polyline points="{points}" fill="none" stroke="black" stroke-width="2"/>')
    lines.append('<text x="755" y="414" text-anchor="end" font-family="sans-serif" font-size="9">intervals are stress bands, not coverage guarantees</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(doc: dict[str, Any]) -> dict[str, Any]:
    _validate_input(doc)
    core = doc["core_fit_observations"]
    target_grid = [(TARGET_PARAMETERS, token) for token in TARGET_TOKEN_BUDGETS]
    diagnostics = {}
    for name in MODEL_NAMES:
        loso = _loso(core, name)
        full = _full_diagnostics(core, name)
        admissible, reasons = _extrapolation_admissible(name, full["fit"], target_grid)
        diagnostics[name] = {
            "loso": loso,
            "full": full,
            "extrapolation_admissible": admissible,
            "extrapolation_rejection_reasons": reasons,
        }
    local_interpolator = min(MODEL_NAMES, key=lambda name: diagnostics[name]["loso"]["loso_rmse_nats"])
    admissible = [name for name in MODEL_NAMES if diagnostics[name]["extrapolation_admissible"]]
    extrapolator = min(admissible, key=lambda name: diagnostics[name]["loso"]["largest_scale_holdout_rmse_nats"])
    stress = _stress_diagnostics(
        doc,
        diagnostics[extrapolator]["full"]["fit"],
        max(int(row["optimized_tokens"]) for row in core),
    )
    predictions = _prediction_report(core, diagnostics, extrapolator, stress)
    nmax = max(int(row["parameters"]) for row in core)
    ideal_bridge = math.sqrt(nmax * TARGET_PARAMETERS)
    next_experiment = {
        "purpose": "halve the log-parameter extrapolation gap while testing token-curvature onset",
        "ideal_parameters": round(ideal_bridge),
        "geometry_requirement": "nearest feasible fixed-control MHA/context-256 continuation; do not substitute current 10M GQA/context-1024 geometry",
        "seed": 1337,
        "final_optimized_tokens": 131_292,
        "heldout_checkpoints": [16_632, 65_772, 131_292],
        "compute_proxy_at_ideal_parameters": 6 * round(ideal_bridge) * 131_292,
        "why_not_10m_first": "A log-midpoint scale directly tests whether the favorable largest-scale holdout trend persists before paying the full one-decade extrapolation cost; the 131K checkpoint also probes the first region where same-identity long-horizon curvature appears.",
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": "LOCAL_FREE_EMPIRICAL_FIT_NOT_PROMOTION_OR_COMPUTE_AUTHORIZATION",
        "input_payload_sha256": doc["payload_sha256"],
        "source_provenance": doc["source_provenance"],
        "shared_identity": doc["shared_identity"],
        "fit_population": {
            "rows": len(core),
            "parameter_scales": sorted({int(row["parameters"]) for row in core}),
            "optimized_token_budgets": sorted({int(row["optimized_tokens"]) for row in core}),
            "selection_rule": "balanced RESEARCH41 seed-1337 core only; exact fixed-token replications deduplicated; fixed-compute and longer two-seed points reserved as stress checks",
        },
        "candidate_models": diagnostics,
        "model_selection": {
            "best_average_loso_interpolator": local_interpolator,
            "selected_extrapolator": extrapolator,
            "selection_reason": "lowest largest-parameter-scale holdout RMSE among extrapolation-admissible forms; average LOSO winner is reported separately",
        },
        "same_identity_stress_diagnostics": stress,
        "predictions_10m_parameter_target": {
            "target_parameters": TARGET_PARAMETERS,
            "target_is_existing_s3_geometry": False,
            "conditional_context": 256,
            "conditional_tokenizer": doc["shared_identity"]["tokenizer_id"],
            "applicability_to_current_s3_gqa_context1024": False,
            "predictions": predictions,
        },
        "next_experiment": next_experiment,
        "truth_boundary": {
            **doc["truth_boundary"],
            "leave_one_scale_out_used": True,
            "prediction_intervals_are_formal_coverage_intervals": False,
            "interval_method": "held-out-scale max-residual calibration widened by log-distance extrapolation and observed same-identity long-horizon structural miss",
            "ten_million_parameter_prediction_is_extrapolation": True,
            "fit_called_chinchilla_or_universal_law": False,
            "no_test_source_tuning": True,
        },
    }
    report["report_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    doc = json.loads(args.input.read_text(encoding="utf-8"))
    report = build_report(doc)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "scaling_fit_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(
        args.out_dir / "experiment_table.csv",
        doc["experiment_table"],
        [
            "run_id", "source", "fit_role", "parameters", "optimized_tokens", "compute_proxy",
            "wall_time_seconds", "wall_time_scope", "best_heldout_bpb", "final_heldout_bpb",
            "seed", "context", "tokenizer", "tokenizer_identity", "corpus_identity", "evaluation_identity",
        ],
    )
    residuals = [
        row
        for diagnostic in report["candidate_models"].values()
        for row in diagnostic["loso"]["residuals"]
    ]
    _write_csv(
        args.out_dir / "loso_residuals.csv",
        residuals,
        [
            "model", "held_out_parameters", "optimized_tokens", "actual_loss_nats",
            "predicted_loss_nats", "residual_nats_pred_minus_actual",
        ],
    )
    pred_rows = report["predictions_10m_parameter_target"]["predictions"]
    flattened = []
    for row in pred_rows:
        flattened.append({
            **{key: value for key, value in row.items() if not isinstance(value, (list, dict))},
            "interval_bpb_low": row["empirical_90_interval_bpb"][0],
            "interval_bpb_high": row["empirical_90_interval_bpb"][1],
            "interval_loss_low": row["empirical_90_interval_loss_nats"][0],
            "interval_loss_high": row["empirical_90_interval_loss_nats"][1],
            "ensemble_models": ";".join(row["ensemble_models"]),
        })
    _write_csv(
        args.out_dir / "predictions_10m.csv",
        flattened,
        [
            "parameters", "optimized_tokens", "compute_proxy", "central_model", "central_loss_nats",
            "central_bpb", "interval_loss_low", "interval_loss_high", "interval_bpb_low",
            "interval_bpb_high", "interval_half_width_nats", "group_conformal_base_half_width_nats",
            "parameter_extrapolation_factor", "token_extrapolation_factor", "model_form_half_spread_nats",
            "long_horizon_structural_penalty_nats", "ensemble_models", "inside_observed_parameter_range",
            "inside_core_token_range", "coverage_guarantee", "label",
        ],
    )
    _write_svg(args.out_dir / "prediction_band.svg", pred_rows)
    print(json.dumps({
        "report": str(report_path),
        "report_sha256": report["report_sha256"],
        "local_interpolator": report["model_selection"]["best_average_loso_interpolator"],
        "selected_extrapolator": report["model_selection"]["selected_extrapolator"],
        "next_experiment": report["next_experiment"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
