"""Fail-closed data readiness checks for capability-oriented pretraining.

This module is intentionally cheap: it evaluates model/corpus metadata only and
never instantiates model weights or starts training.  It separates *data
readiness* from *compute authorization* so a PASS here can never be interpreted
as permission to spend money or launch a long run.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

POLICY_SCHEMA = "12-6.corpus-readiness-policy.v1"
CORPUS_SCHEMA = "12-6.corpus-manifest.v1"


class CorpusReadinessError(ValueError):
    """Raised when readiness inputs are malformed or internally inconsistent."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CorpusReadinessError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CorpusReadinessError(f"{label} must be a non-negative integer")
    return value


def _positive_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CorpusReadinessError(f"{label} must be a positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise CorpusReadinessError(f"{label} must be a finite positive number")
    return converted


def _required_bool(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key)
    if not isinstance(value, bool):
        raise CorpusReadinessError(f"{key} must be boolean")
    return value


def evaluate_corpus_readiness(
    manifest: Mapping[str, Any],
    *,
    parameter_count: int,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate whether a corpus is suitable for capability-oriented pretraining.

    The policy owns the data/parameter floor.  The default project policy uses a
    research-informed 20 training-token/parameter minimum for the byte tokenizer,
    but the evaluator itself does not silently invent or relax that threshold.
    """

    parameter_count = _positive_int(parameter_count, label="parameter_count")
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise CorpusReadinessError("unsupported corpus readiness policy schema")
    if manifest.get("schema_version") != CORPUS_SCHEMA:
        raise CorpusReadinessError("unsupported corpus manifest schema")

    min_tokens_per_parameter = _positive_number(
        policy.get("min_train_tokens_per_parameter"),
        label="min_train_tokens_per_parameter",
    )
    minimum_external_sources = _nonnegative_int(
        policy.get("min_external_training_eligible_sources", 0),
        label="min_external_training_eligible_sources",
    )
    require_external = _required_bool(policy, "require_external_training_data")
    require_diversity = _required_bool(policy, "require_external_source_diversity")
    require_zero_overlap = _required_bool(policy, "require_zero_train_validation_overlap")

    required_strata_raw = policy.get("required_strata", [])
    if not isinstance(required_strata_raw, list) or not required_strata_raw:
        raise CorpusReadinessError("required_strata must be a non-empty list")
    required_strata: tuple[str, ...] = tuple(required_strata_raw)
    if any(not isinstance(item, str) or not item for item in required_strata):
        raise CorpusReadinessError("required_strata must contain non-empty strings")
    if len(set(required_strata)) != len(required_strata):
        raise CorpusReadinessError("required_strata contains duplicates")

    corpus_identity = manifest.get("corpus_identity_sha256")
    by_split = manifest.get("by_split")
    if not isinstance(by_split, Mapping) or not isinstance(by_split.get("train"), Mapping):
        raise CorpusReadinessError("manifest by_split.train is missing")
    train_tokens = _positive_int(
        by_split["train"].get("byte_tokens"), label="manifest train byte_tokens"
    )
    required_train_tokens = math.ceil(parameter_count * min_tokens_per_parameter)
    deficit_train_tokens = max(0, required_train_tokens - train_tokens)
    tokens_per_parameter = train_tokens / parameter_count

    truth = manifest.get("truth_boundary")
    if not isinstance(truth, Mapping):
        raise CorpusReadinessError("manifest truth_boundary is missing")
    contains_external = truth.get("contains_external_training_data")
    external_diversity = truth.get("external_source_diversity_representative")
    if not isinstance(contains_external, bool):
        raise CorpusReadinessError("truth_boundary.contains_external_training_data must be boolean")
    if not isinstance(external_diversity, bool):
        raise CorpusReadinessError(
            "truth_boundary.external_source_diversity_representative must be boolean"
        )

    external_sources = _nonnegative_int(
        manifest.get("external_training_eligible_sources", 0),
        label="external_training_eligible_sources",
    )
    overlap = _nonnegative_int(
        manifest.get("train_validation_content_overlap", 0),
        label="train_validation_content_overlap",
    )

    split_stratum = manifest.get("by_split_stratum")
    if not isinstance(split_stratum, Mapping):
        raise CorpusReadinessError("manifest by_split_stratum is missing")
    stratum_tokens: dict[str, int] = {}
    for stratum in required_strata:
        bucket = split_stratum.get(f"train:{stratum}")
        if not isinstance(bucket, Mapping):
            stratum_tokens[stratum] = 0
            continue
        value = bucket.get("byte_tokens", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CorpusReadinessError(
                f"manifest train:{stratum}.byte_tokens must be a non-negative integer"
            )
        stratum_tokens[stratum] = value

    checks: dict[str, dict[str, Any]] = {
        "corpus_identity": {
            "pass": _is_sha256(corpus_identity),
            "actual": corpus_identity,
        },
        "data_parameter_floor": {
            "pass": train_tokens >= required_train_tokens,
            "train_byte_tokens": train_tokens,
            "parameter_count": parameter_count,
            "tokens_per_parameter": tokens_per_parameter,
            "minimum_tokens_per_parameter": min_tokens_per_parameter,
            "required_train_byte_tokens": required_train_tokens,
            "deficit_train_byte_tokens": deficit_train_tokens,
        },
        "external_training_data": {
            "pass": (not require_external) or contains_external,
            "required": require_external,
            "contains_external_training_data": contains_external,
        },
        "external_source_count": {
            "pass": external_sources >= minimum_external_sources,
            "actual": external_sources,
            "minimum": minimum_external_sources,
        },
        "external_source_diversity": {
            "pass": (not require_diversity) or external_diversity,
            "required": require_diversity,
            "representative": external_diversity,
        },
        "train_validation_isolation": {
            "pass": (not require_zero_overlap) or overlap == 0,
            "required_zero_overlap": require_zero_overlap,
            "content_overlap": overlap,
        },
        "required_strata_present": {
            "pass": all(value > 0 for value in stratum_tokens.values()),
            "train_byte_tokens": stratum_tokens,
        },
    }

    blockers = [name for name, check in checks.items() if check["pass"] is not True]
    status = "READY" if not blockers else "BLOCKED"
    result = {
        "schema_version": "12-6.corpus-readiness-report.v1",
        "policy_id": policy.get("policy_id"),
        "policy_sha256": _sha256_json(policy),
        "intent": policy.get("intent"),
        "status": status,
        "pass": status == "READY",
        "parameter_count": parameter_count,
        "corpus_identity_sha256": corpus_identity,
        "checks": checks,
        "blockers": blockers,
        "training_authorization_inferred": False,
        "training_performed": False,
    }
    result["report_sha256"] = _sha256_json(result)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusReadinessError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CorpusReadinessError(f"JSON root must be an object: {path}")
    return value


def evaluate_policy_file(policy_path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    """Load a pinned project policy, model candidate and corpus manifest."""

    root = Path(repo_root).resolve()
    policy_path = Path(policy_path)
    if not policy_path.is_absolute():
        policy_path = root / policy_path
    policy = _read_json(policy_path)

    model_path_raw = policy.get("model_candidate_path")
    manifest_path_raw = policy.get("corpus_manifest_path")
    if not isinstance(model_path_raw, str) or not model_path_raw:
        raise CorpusReadinessError("model_candidate_path is required")
    if not isinstance(manifest_path_raw, str) or not manifest_path_raw:
        raise CorpusReadinessError("corpus_manifest_path is required")

    model_path = (root / model_path_raw).resolve()
    manifest_path = (root / manifest_path_raw).resolve()
    for label, path in (("model", model_path), ("corpus", manifest_path)):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise CorpusReadinessError(f"{label} path escapes repository") from exc

    model_candidate = _read_json(model_path)
    manifest = _read_json(manifest_path)
    parameter_count = _positive_int(
        model_candidate.get("expected_parameters"), label="model expected_parameters"
    )
    report = evaluate_corpus_readiness(
        manifest, parameter_count=parameter_count, policy=policy
    )
    report["model_candidate_path"] = str(model_path.relative_to(root))
    report["corpus_manifest_path"] = str(manifest_path.relative_to(root))
    return report
