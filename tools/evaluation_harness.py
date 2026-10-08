"""Plan 4 Section 2: deterministic, identity-frozen fixture evaluation harness.

Only public/selection fixtures enter this module. Reserved final-test labels
remain solely within the Plan 4 Section 1 evaluator vault. This harness does
not train models, evaluate production champions or authorize final-test access.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

SCHEMA = "12-6.plan4-evaluation-harness.v1"
EVALUATOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEFAULT_METRICS = {
    "exact_match": ("accuracy", "wilson95_lower", "wilson95_upper", "stderr"),
    "numeric": ("mae", "bootstrap95_lower", "bootstrap95_upper", "stderr"),
}


class HarnessError(ValueError):
    """A frozen evaluation protocol or candidate packet was not admissible."""


def _hash(value: object) -> str:
    return hashlib.sha256(_encode(value)).hexdigest()


def _encode(value: object) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def _sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _label(value: object, name: str) -> str:
    if type(value) is not str or not value.strip() or len(value) > 120:
        raise HarnessError(f"invalid {name}")
    return value


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def freeze_suite(
    *,
    suite_id: str,
    dataset_version: str,
    task_type: str,
    tasks: list[dict[str, Any]],
    seed: int,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Freeze a public-fixture suite; its digest includes seed, tasks and metrics."""
    _label(suite_id, "suite_id")
    _label(dataset_version, "dataset_version")
    if task_type not in DEFAULT_METRICS or type(task_type) is not str:
        raise HarnessError("unsupported task type")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise HarnessError("seed must be a bounded integer")
    if type(tasks) is not list or not 1 <= len(tasks) <= 10000:
        raise HarnessError("bounded nonempty tasks required")
    normalized = []
    for task in tasks:
        if type(task) is not dict or set(task) != {"id", "answer"}:
            raise HarnessError("task shape drift")
        key = _label(task["id"], "task id")
        answer = task["answer"]
        if task_type == "exact_match":
            if type(answer) is not str or len(answer) > 10000:
                raise HarnessError("exact-match answer invalid")
        elif not _number(answer):
            raise HarnessError("numeric answer invalid")
        normalized.append({"id": key, "answer": answer})
    ids = [item["id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise HarnessError("duplicate task id")
    if metrics is None:
        metrics = list(DEFAULT_METRICS[task_type])
    if (type(metrics) is not list or not 1 <= len(metrics) <= 16
            or any(type(metric) is not str or not metric or len(metric) > 60 for metric in metrics)
            or len(metrics) != len(set(metrics))):
        raise HarnessError("invalid requested metrics")
    payload = {
        "schema_version": SCHEMA,
        "suite_id": suite_id,
        "dataset_version": dataset_version,
        "task_type": task_type,
        "seed": seed,
        "tasks": normalized,
        "metrics": metrics,
    }
    return {**payload, "protocol_sha256": _hash(payload)}


def validate_suite(protocol: object) -> dict[str, Any]:
    if type(protocol) is not dict or set(protocol) != {
        "schema_version", "suite_id", "dataset_version", "task_type",
        "seed", "tasks", "metrics", "protocol_sha256"
    } or protocol.get("schema_version") != SCHEMA:
        raise HarnessError("protocol schema drift")
    if not _sha(protocol.get("protocol_sha256")):
        raise HarnessError("invalid protocol identity")
    rebound = freeze_suite(
        suite_id=protocol["suite_id"],
        dataset_version=protocol["dataset_version"],
        task_type=protocol["task_type"],
        tasks=protocol["tasks"],
        seed=protocol["seed"],
        metrics=protocol["metrics"],
    )
    if rebound != protocol:
        raise HarnessError("protocol identity or content drift")
    return rebound


def _wilson(success: int, n: int) -> tuple[float, float, float]:
    p = success / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - radius), min(1.0, center + radius), math.sqrt(p * (1 - p) / n)


def _bootstrap(errors: list[float], seed: int) -> tuple[float, float, float]:
    n = len(errors)
    if n == 1:
        return errors[0], errors[0], 0.0
    rng = random.Random(seed)
    means = sorted(sum(errors[rng.randrange(n)] for _ in range(n)) / n for _ in range(300))
    avg = sum(errors) / n
    variance = sum((e - avg) ** 2 for e in errors) / (n - 1)
    return means[7], means[292], math.sqrt(variance / n)


def _candidate(
    protocol: dict[str, Any], candidate: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "candidate_id": candidate["candidate_id"],
        "role": candidate["role"],
        "model_sha256": candidate["model_sha256"],
        "protocol_sha256": protocol["protocol_sha256"],
        "status": "ERROR",
        "error_code": None,
        "metrics": {},
    }
    requested = protocol["metrics"]
    try:
        predictions = candidate["predictions"]
        if type(predictions) is not dict or set(predictions) != {
            task["id"] for task in protocol["tasks"]
        }:
            raise HarnessError("candidate prediction IDs do not match suite")
        if protocol["task_type"] == "exact_match":
            if any(type(x) is not str or len(x) > 10000 for x in predictions.values()):
                raise HarnessError("non-string exact-match prediction")
            correct = sum(
                predictions[task["id"]] == task["answer"] for task in protocol["tasks"]
            )
            n = len(protocol["tasks"])
            low, high, stderr = _wilson(correct, n)
            values = {
                "accuracy": correct / n,
                "wilson95_lower": low,
                "wilson95_upper": high,
                "stderr": stderr,
            }
        else:
            if any(not _number(x) for x in predictions.values()):
                raise HarnessError("non-finite numeric prediction")
            errors = [
                abs(predictions[task["id"]] - task["answer"])
                for task in protocol["tasks"]
            ]
            if any(not math.isfinite(x) for x in errors):
                raise HarnessError("numeric metric overflow")
            low, high, stderr = _bootstrap(errors, protocol["seed"])
            values = {
                "mae": sum(errors) / len(errors),
                "bootstrap95_lower": low,
                "bootstrap95_upper": high,
                "stderr": stderr,
            }
        unsupported = False
        for metric in requested:
            if metric not in values:
                result["metrics"][metric] = {"status": "UNSUPPORTED", "value": None}
                unsupported = True
            else:
                result["metrics"][metric] = {"status": "EVALUATED", "value": values[metric]}
        result["status"] = "UNSUPPORTED" if unsupported else "EVALUATED"
    except HarnessError as exc:
        result["error_code"] = str(exc)
        for metric in requested:
            result["metrics"][metric] = {"status": "NOT_COMPUTED", "value": None}
    return result


def evaluate_suite(
    protocol: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate each named model on the exact same frozen, public-fixture suite."""
    frozen = validate_suite(protocol)
    if type(candidates) is not list or not 2 <= len(candidates) <= 100:
        raise HarnessError("champion and candidates required")
    identifiers: set[str] = set()
    champion_count = 0
    for candidate in candidates:
        if type(candidate) is not dict or set(candidate) != {
            "candidate_id", "role", "model_sha256", "predictions"
        }:
            raise HarnessError("candidate packet shape drift")
        key = _label(candidate["candidate_id"], "candidate id")
        if key in identifiers:
            raise HarnessError("duplicate candidate identity")
        identifiers.add(key)
        if candidate["role"] not in ("candidate", "champion"):
            raise HarnessError("invalid model role")
        if not _sha(candidate["model_sha256"]):
            raise HarnessError("model identity invalid")
        if candidate["role"] == "champion":
            champion_count += 1
    if champion_count != 1:
        raise HarnessError("exactly one champion is required")
    results = [_candidate(frozen, item) for item in candidates]
    champion = next(result for result in results if result["role"] == "champion")
    primary = "accuracy" if frozen["task_type"] == "exact_match" else "mae"
    comparisons = []
    for result in results:
        if result["role"] != "candidate":
            continue
        valid = all(
            item["status"] == "EVALUATED"
            and item["metrics"].get(primary, {}).get("status") == "EVALUATED"
            for item in (result, champion)
        )
        comparisons.append({
            "candidate_id": result["candidate_id"],
            "champion_id": champion["candidate_id"],
            "metric": primary,
            "protocol_sha256": frozen["protocol_sha256"],
            "status": "COMPARABLE" if valid else "UNAVAILABLE",
            "delta": (
                result["metrics"][primary]["value"] - champion["metrics"][primary]["value"]
            ) if valid else None,
            "higher_is_better": frozen["task_type"] == "exact_match",
        })
    report = {
        "schema_version": "12-6.plan4-evaluation-report.v1",
        "suite_id": frozen["suite_id"],
        "dataset_version": frozen["dataset_version"],
        "protocol_sha256": frozen["protocol_sha256"],
        "evaluator_sha256": EVALUATOR_SHA256,
        "seed": frozen["seed"],
        "task_count": len(frozen["tasks"]),
        "status": "COMPLETE" if all(x["status"] == "EVALUATED" for x in results) else "INCOMPLETE",
        "candidates": results,
        "comparisons": comparisons,
        "reserved_final_test_access_authorized": False,
        "training_authorized": False,
        "compute_authorized": False,
    }
    return {**report, "report_sha256": _hash(report)}
