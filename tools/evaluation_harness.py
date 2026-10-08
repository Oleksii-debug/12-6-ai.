"""Deterministic, public-fixture evaluation for Plan 4 Section 2.

This is a *component* harness, not final-test authorization, a production
champion promoter, or a replacement for the sealed EvaluationVault. It refuses
reserved/unknown dataset classes and has no model/training side effects.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from copy import deepcopy
from typing import Any

SUITE_SCHEMA = "12-6.plan4-evaluation-suite.v1"
PROTOCOL_SCHEMA = "12-6.plan4-evaluation-protocol.v1"
REPORT_SCHEMA = "12-6.plan4-evaluation-report.v1"
COMPARISON_SCHEMA = "12-6.plan4-evaluation-comparison.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED = {
    "text_exact_match": frozenset({"accuracy"}),
    "numeric_regression": frozenset({"mean_absolute_error", "root_mean_square_error"}),
}


class EvaluationHarnessError(ValueError):
    """No failure or unsupported metric may be silently promoted to PASS."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationHarnessError(message)


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value)).hexdigest()


def _json(value: object) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    except (ValueError, TypeError, OverflowError) as exc:
        raise EvaluationHarnessError("non-canonical evaluation payload") from exc


def _identity(value: object) -> bool:
    return type(value) is str and bool(_SHA256.fullmatch(value))


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _exact(value: object, fields: set[str], context: str) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == fields, f"{context} fields invalid")
    return value


def _suite_spec(spec: object) -> dict[str, Any]:
    s = _exact(spec, {"schema_version", "suite_id", "suite_version",
                      "visibility", "kind", "cases"}, "suite")
    _require(s["schema_version"] == SUITE_SCHEMA, "suite schema unsupported")
    _require(s["visibility"] == "public_fixture", "reserved evaluation forbidden")
    for key in ("suite_id", "suite_version"):
        value = s[key]
        _require(type(value) is str and 0 < len(value.strip()) <= 128,
                 f"{key} missing or oversized")
    kind = s["kind"]
    _require(type(kind) is str and kind in _SUPPORTED, "task kind unsupported")
    cases = s["cases"]
    _require(type(cases) is list and 1 <= len(cases) <= 10000, "suite size invalid")
    ids: set[str] = set()
    for case in cases:
        c = _exact(case, {"id", "expected"}, "suite case")
        case_id = c["id"]
        _require(type(case_id) is str and 0 < len(case_id) <= 128,
                 "case ID invalid")
        _require(case_id not in ids, "duplicate case ID")
        ids.add(case_id)
        expected = c["expected"]
        if kind == "text_exact_match":
            _require(type(expected) is str and len(expected) <= 4096,
                     "text answer invalid")
        else:
            _require(_finite(expected), "numeric answer invalid")
    _require(len(_json(s)) <= 2_000_000, "suite exceeds fixture limit")
    return s


def freeze_suite(spec: dict[str, Any]) -> dict[str, Any]:
    """Pin exact examples, schema, task and version; only PUBLIC fixtures."""
    checked = _suite_spec(spec)
    payload = deepcopy(checked)
    return {"spec": payload, "suite_sha256": _hash(payload)}


def _verified_suite(suite: object) -> dict[str, Any]:
    value = _exact(suite, {"spec", "suite_sha256"}, "frozen suite")
    _require(_identity(value["suite_sha256"]), "suite digest invalid")
    checked = _suite_spec(value["spec"])
    _require(_hash(checked) == value["suite_sha256"], "suite digest drift")
    return checked


def freeze_protocol(
    suite: dict[str, Any], *,
    metrics: tuple[str, ...],
    seed: int,
    evaluator_sha256: str,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bind a fixed metric set, RNG seed, evaluator and suite identity."""
    spec = _verified_suite(suite)
    _require(type(seed) is int and 0 <= seed < 2**32, "seed invalid")
    _require(_identity(evaluator_sha256), "evaluator identity invalid")
    _require(type(confidence_level) is float and confidence_level == 0.95,
             "confidence level unsupported")
    _require(type(metrics) is tuple and bool(metrics)
             and all(type(x) is str for x in metrics)
             and len(metrics) == len(set(metrics))
             and set(metrics) <= _SUPPORTED[spec["kind"]],
             "metric unsupported or duplicated")
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "suite_sha256": suite["suite_sha256"],
        "kind": spec["kind"],
        "metrics": sorted(metrics),
        "seed": seed,
        "evaluator_sha256": evaluator_sha256,
        "confidence_level": confidence_level,
    }
    return {**protocol, "protocol_sha256": _hash(protocol)}


def _verified_protocol(suite: dict[str, Any], protocol: object) -> dict[str, Any]:
    p = _exact(protocol, {"schema_version", "suite_sha256", "kind", "metrics",
                          "seed", "evaluator_sha256", "confidence_level",
                          "protocol_sha256"}, "protocol")
    _require(_identity(p["protocol_sha256"]), "protocol identity invalid")
    expected = freeze_protocol(
        suite, metrics=tuple(p["metrics"]) if type(p["metrics"]) is list else (),
        seed=p["seed"], evaluator_sha256=p["evaluator_sha256"],
        confidence_level=p["confidence_level"],
    )
    _require(p == expected, "frozen evaluation protocol drift")
    return p


def _wilson(successes: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z * z / n
    midpoint = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return max(0.0, midpoint - radius), min(1.0, midpoint + radius)


def _numeric_metric(values: list[float], metric: str) -> float:
    if metric == "mean_absolute_error":
        return math.fsum(values) / len(values)
    if metric == "root_mean_square_error":
        return math.sqrt(math.fsum(x * x for x in values) / len(values))
    raise EvaluationHarnessError("metric unsupported")


def _bootstrap(values: list[float], metric: str, seed: int) -> tuple[float, float]:
    # Metric-specific seed keeps intervals invariant under metric enumeration.
    salt = int(_hash(metric)[:8], 16)
    rng = random.Random(seed ^ salt)
    n = len(values)
    draws = sorted(
        _numeric_metric([values[rng.randrange(n)] for _ in range(n)], metric)
        for _ in range(256)
    )
    estimate = _numeric_metric(values, metric)
    return min(estimate, draws[6]), max(estimate, draws[249])


def evaluate_candidate(
    suite: dict[str, Any], protocol: dict[str, Any], *,
    predictions: dict[str, Any],
    model_sha256: str,
    role: str = "candidate",
) -> dict[str, Any]:
    """Score a public fixture; failed/partial predictions raise, never score zero."""
    spec = _verified_suite(suite)
    p = _verified_protocol(suite, protocol)
    _require(_identity(model_sha256), "model identity invalid")
    _require(type(role) is str and role in ("candidate", "champion"),
             "candidate role invalid")
    cases = spec["cases"]
    _require(type(predictions) is dict
             and set(predictions) == {c["id"] for c in cases},
             "missing/extra/failed predictions")
    errors: list[float] = []
    correct = 0
    for case in cases:
        observed = predictions[case["id"]]
        if spec["kind"] == "text_exact_match":
            _require(type(observed) is str and len(observed) <= 4096,
                     "text prediction invalid")
            correct += observed == case["expected"]
        else:
            _require(_finite(observed), "numeric prediction invalid")
            difference = float(observed) - float(case["expected"])
            _require(math.isfinite(difference), "numeric difference overflow")
            errors.append(abs(difference))
    results = []
    for metric in p["metrics"]:
        if metric == "accuracy":
            value = correct / len(cases)
            lower, upper = _wilson(correct, len(cases))
        else:
            value = _numeric_metric(errors, metric)
            lower, upper = _bootstrap(errors, metric, p["seed"])
        _require(all(math.isfinite(x) for x in (value, lower, upper)),
                 "metric arithmetic invalid")
        results.append({"name": metric, "value": value,
                        "confidence_interval_95": [lower, upper],
                        "higher_is_better": metric == "accuracy"})
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "PASS",
        "suite_id": spec["suite_id"],
        "suite_version": spec["suite_version"],
        "suite_sha256": suite["suite_sha256"],
        "protocol_sha256": p["protocol_sha256"],
        "evaluator_sha256": p["evaluator_sha256"],
        "model_sha256": model_sha256,
        "role": role,
        "sample_count": len(cases),
        "metrics": results,
    }
    return {**report, "report_sha256": _hash(report)}


def _verified_report(protocol: dict[str, Any], report: object) -> dict[str, Any]:
    v = _exact(report, {"schema_version", "status", "suite_id", "suite_version",
                        "suite_sha256", "protocol_sha256", "evaluator_sha256",
                        "model_sha256", "role", "sample_count", "metrics",
                        "report_sha256"}, "report")
    _require(v["schema_version"] == REPORT_SCHEMA and v["status"] == "PASS",
             "candidate evaluation not successful")
    _require(_identity(v["report_sha256"]) and
             v["report_sha256"] == _hash({k: x for k, x in v.items()
                                         if k != "report_sha256"}),
             "report digest invalid")
    _require(v["suite_sha256"] == protocol["suite_sha256"]
             and v["protocol_sha256"] == protocol["protocol_sha256"]
             and v["evaluator_sha256"] == protocol["evaluator_sha256"],
             "comparison protocol mismatch")
    _require(_identity(v["model_sha256"]) and v["role"] in ("candidate", "champion"),
             "report model/role invalid")
    _require(type(v["sample_count"]) is int and v["sample_count"] > 0,
             "sample count invalid")
    observed = v["metrics"]
    _require(type(observed) is list
             and [x.get("name") for x in observed if type(x) is dict]
             == protocol["metrics"] and len(observed) == len(protocol["metrics"]),
             "report metrics mismatch")
    for metric in observed:
        _require(type(metric) is dict
                 and set(metric) == {"name", "value", "confidence_interval_95",
                                    "higher_is_better"}, "metric fields invalid")
        limits = metric["confidence_interval_95"]
        _require(_finite(metric["value"])
                 and type(limits) is list and len(limits) == 2
                 and all(_finite(x) for x in limits)
                 and limits[0] <= metric["value"] <= limits[1],
                 "uncertainty invalid")
        _require(metric["higher_is_better"] is (metric["name"] == "accuracy"),
                 "metric direction mismatch")
    return v


def compare_candidates(
    suite: dict[str, Any],
    protocol: dict[str, Any],
    reports: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Compare only same frozen protocol; never silently admit missing metrics."""
    spec = _verified_suite(suite)
    p = _verified_protocol(suite, protocol)
    _require(type(reports) is tuple and len(reports) >= 2,
             "at least two reports required")
    checked = [_verified_report(p, report) for report in reports]
    _require(all(x["suite_id"] == spec["suite_id"]
                 and x["suite_version"] == spec["suite_version"]
                 and x["sample_count"] == len(spec["cases"]) for x in checked),
             "report suite identity mismatch")
    identities = [x["model_sha256"] for x in checked]
    _require(len(identities) == len(set(identities)), "duplicate model identity")
    rankings = {}
    for name in p["metrics"]:
        higher = name == "accuracy"
        sorted_reports = sorted(
            checked, key=lambda r: (
                -next(m["value"] for m in r["metrics"] if m["name"] == name)
                if higher else
                next(m["value"] for m in r["metrics"] if m["name"] == name),
                r["model_sha256"],
            ),
        )
        rankings[name] = [
            {"model_sha256": r["model_sha256"], "role": r["role"],
             "value": next(m["value"] for m in r["metrics"]
                           if m["name"] == name)}
            for r in sorted_reports
        ]
    result = {"schema_version": COMPARISON_SCHEMA,
              "protocol_sha256": p["protocol_sha256"],
              "suite_sha256": suite["suite_sha256"],
              "models": sorted(identities),
              "rankings": rankings,
              "status": "COMPARABLE"}
    return {**result, "comparison_sha256": _hash(result)}
