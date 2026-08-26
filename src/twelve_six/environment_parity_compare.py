"""ENV-160 cross-environment comparison over captured first-party traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .environment_parity import DEFAULT_ATOL, DEFAULT_RTOL, decision_policy, hash_json

SCHEMA_COMPARE = "12-6.environment-parity-comparison.v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _checkpoint_lineage(record: Mapping[str, Any]) -> dict[str, Any]:
    identity = dict(record["identity"])
    identity.pop("environment", None)
    identity.pop("environment_hash", None)
    return identity


def _numeric_pairs(left: Any, right: Any, prefix: str = "") -> list[tuple[str, float, float]]:
    pairs: list[tuple[str, float, float]] = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) & set(right)):
            child = f"{prefix}.{key}" if prefix else key
            pairs.extend(_numeric_pairs(left[key], right[key], child))
    elif isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        scalar_lists = all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in left + right
        )
        if left and scalar_lists:
            for index, (a, b) in enumerate(zip(left, right)):
                pairs.append((f"{prefix}[{index}]", float(a), float(b)))
        else:
            for index, (a, b) in enumerate(zip(left, right)):
                pairs.extend(_numeric_pairs(a, b, f"{prefix}[{index}]"))
    elif (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        pairs.append((prefix, float(left), float(right)))
    return pairs


def _tensor_values(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value["values"] for name, value in state["tensors"].items()}


def _numeric_view(trace: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "initial_weights": _tensor_values(trace["initial"]["weights"]),
        "initial_logits": trace["initial"]["logits"]["values"],
        "initial_loss": trace["initial"]["loss"],
        "initial_gradients": {
            name: None if value is None else value["values"]
            for name, value in trace["initial"]["gradients"]["tensors"].items()
        },
        "state_after_step_1": _tensor_values(trace["updates"]["state_after_step_1"]),
        "state_after_step_3": _tensor_values(trace["updates"]["state_after_step_3"]),
        "heldout_loss": trace["heldout_evaluation"]["loss"],
        "step_metrics": trace["updates"]["step_metrics"],
    }


def compare_traces(
    canonical: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
    semantic_checks = {
        "source_sha": canonical["source_sha"] == candidate["source_sha"],
        "model_spec": canonical["model"]["model_spec_sha256"]
        == candidate["model"]["model_spec_sha256"],
        "parameter_count": canonical["model"]["parameter_count"]
        == candidate["model"]["parameter_count"],
        "init_spec": canonical["model"]["init_spec_sha256"]
        == candidate["model"]["init_spec_sha256"],
        "optimizer_config": canonical["optimizer"] == candidate["optimizer"],
        "inputs": canonical["inputs"] == candidate["inputs"],
        "token_counters": canonical["token_counters"] == candidate["token_counters"],
        "checkpoint_step_1_lineage": _checkpoint_lineage(canonical["checkpoints"]["step_1"])
        == _checkpoint_lineage(candidate["checkpoints"]["step_1"]),
        "checkpoint_step_3_lineage": _checkpoint_lineage(canonical["checkpoints"]["step_3"])
        == _checkpoint_lineage(candidate["checkpoints"]["step_3"]),
        "heldout_non_mutation": canonical["heldout_evaluation"]["non_mutation_passed"] is True
        and candidate["heldout_evaluation"]["non_mutation_passed"] is True,
    }
    semantic_pass = all(semantic_checks.values())

    pairs = _numeric_pairs(_numeric_view(canonical), _numeric_view(candidate))
    numeric_pass = True
    max_abs = 0.0
    max_rel = 0.0
    worst: dict[str, Any] | None = None
    for path, left, right in pairs:
        absolute = abs(left - right)
        relative = absolute / max(abs(left), abs(right), 1e-30)
        if absolute > max_abs:
            max_abs = absolute
            worst = {
                "path": path,
                "canonical": left,
                "candidate": right,
                "absolute": absolute,
                "relative": relative,
            }
        max_rel = max(max_rel, relative)
        if not math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
            numeric_pass = False

    bitwise_checks = {
        "initial_weights": canonical["initial"]["weights"]["state_sha256"]
        == candidate["initial"]["weights"]["state_sha256"],
        "initial_logits": canonical["initial"]["logits"]["sha256"]
        == candidate["initial"]["logits"]["sha256"],
        "initial_gradients": canonical["initial"]["gradients"]["gradient_sha256"]
        == candidate["initial"]["gradients"]["gradient_sha256"],
        "state_after_step_1": canonical["updates"]["state_after_step_1"]["state_sha256"]
        == candidate["updates"]["state_after_step_1"]["state_sha256"],
        "state_after_step_3": canonical["updates"]["state_after_step_3"]["state_sha256"]
        == candidate["updates"]["state_after_step_3"]["state_sha256"],
        "checkpoint_id_step_1": canonical["checkpoints"]["step_1"]["checkpoint_id"]
        == candidate["checkpoints"]["step_1"]["checkpoint_id"],
        "checkpoint_id_step_3": canonical["checkpoints"]["step_3"]["checkpoint_id"]
        == candidate["checkpoints"]["step_3"]["checkpoint_id"],
    }

    canonical_fp = canonical["environment_fingerprint"]
    candidate_fp = candidate["environment_fingerprint"]
    same_environment = canonical_fp.get("fingerprint_sha256") == candidate_fp.get(
        "fingerprint_sha256"
    )
    if not semantic_pass:
        classification = "SEMANTIC_DRIFT"
    elif not numeric_pass:
        classification = "NUMERIC_DRIFT_REQUIRES_EXACT_HEAD"
    elif same_environment and all(bitwise_checks.values()):
        classification = "PASS_BITWISE"
    else:
        classification = "PASS_NUMERIC_TOLERANCE"

    canonical_locked = canonical_fp["exact_locked_runtime"] is True
    candidate_locked = candidate_fp["exact_locked_runtime"] is True
    scientific_authority = semantic_pass and numeric_pass and canonical_locked and candidate_locked

    report = {
        "schema": SCHEMA_COMPARE,
        "classification": classification,
        "tolerances": {"atol": atol, "rtol": rtol},
        "semantic_checks": semantic_checks,
        "semantic_pass": semantic_pass,
        "numeric": {
            "compared_scalar_count": len(pairs),
            "pass": numeric_pass,
            "max_absolute_difference": max_abs,
            "max_relative_difference": max_rel,
            "worst_absolute_difference": worst,
        },
        "bitwise_checks": bitwise_checks,
        "canonical_environment_fingerprint": canonical_fp,
        "candidate_environment_fingerprint": candidate_fp,
        "expected_environment_difference_fields": [
            "python.version",
            "torch.version",
            "platform.release",
            "checkpoint.identity.environment",
            "checkpoint.identity.environment_hash",
            "checkpoint_id when environment snapshot differs",
        ],
        "scientific_authority": scientific_authority,
        "truth_boundary": (
            "Cross-version bitwise equality is not required. Source/config/token/checkpoint-lineage "
            "semantics must match exactly after excluding the checkpoint runtime snapshot; fp32 "
            "weights, logits, loss, gradients, update states and held-out values must satisfy the "
            "declared tolerances. A source-equivalent candidate remains debugging evidence."
        ),
        "decision_policy": decision_policy(),
    }
    report["report_sha256"] = hash_json(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    value = compare_traces(
        _read_json(args.canonical),
        _read_json(args.candidate),
        atol=args.atol,
        rtol=args.rtol,
    )
    _write_json(args.output, value)
    print(
        json.dumps(
            {
                "classification": value["classification"],
                "semantic_pass": value["semantic_pass"],
                "numeric": value["numeric"],
                "scientific_authority": value["scientific_authority"],
                "report": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
