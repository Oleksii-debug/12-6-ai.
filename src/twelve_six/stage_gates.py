"""Integrated S0 stage gates layered on the core D06 evaluation engine."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twelve_six.evaluation import (
    GateResult,
    GateStatus,
    S0GatePolicy,
    dump_stage_gate_result,
    evaluate_s0,
    load_json_object,
)

_MISSING = object()


def _get(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _vocab_compatibility_gate(evidence: Mapping[str, Any]) -> GateResult:
    paths = (
        "candidate.model_vocab_size",
        "tokenizer.identity",
        "tokenizer.vocab_size",
        "tokenizer.max_token_id",
    )
    values = {path: _get(evidence, path) for path in paths}
    missing = [path for path, value in values.items() if value is _MISSING]
    if missing:
        return GateResult(
            gate_id="s0.tokenizer_model_vocab",
            title="Tokenizer/model vocabulary compatibility",
            status=GateStatus.NOT_TESTED,
            reason=f"missing evidence: {', '.join(missing)}",
            evidence={"missing": missing},
        )

    identity = values["tokenizer.identity"]
    model_vocab = values["candidate.model_vocab_size"]
    tokenizer_vocab = values["tokenizer.vocab_size"]
    max_token_id = values["tokenizer.max_token_id"]

    if not isinstance(identity, str) or not identity.strip():
        return GateResult(
            gate_id="s0.tokenizer_model_vocab",
            title="Tokenizer/model vocabulary compatibility",
            status=GateStatus.FAIL,
            reason="tokenizer.identity must be a non-empty string",
            evidence={"tokenizer.identity": identity},
        )

    numeric = {
        "candidate.model_vocab_size": model_vocab,
        "tokenizer.vocab_size": tokenizer_vocab,
        "tokenizer.max_token_id": max_token_id,
    }
    invalid = [
        name
        for name, value in numeric.items()
        if isinstance(value, bool) or not isinstance(value, int)
    ]
    if invalid:
        return GateResult(
            gate_id="s0.tokenizer_model_vocab",
            title="Tokenizer/model vocabulary compatibility",
            status=GateStatus.FAIL,
            reason=f"vocabulary fields must be integers: {', '.join(invalid)}",
            evidence=numeric,
        )

    positive_sizes = model_vocab > 0 and tokenizer_vocab > 0
    valid_max_id = max_token_id >= 0
    equal_vocab = model_vocab == tokenizer_vocab
    max_id_fits = max_token_id < model_vocab
    passed = positive_sizes and valid_max_id and equal_vocab and max_id_fits

    if passed:
        reason = (
            f"tokenizer {identity} vocab={tokenizer_vocab} and max_token_id={max_token_id} "
            f"fit model_vocab_size={model_vocab}"
        )
    else:
        reason = (
            f"tokenizer/model vocab incompatible: model={model_vocab}, "
            f"tokenizer={tokenizer_vocab}, max_token_id={max_token_id}"
        )

    return GateResult(
        gate_id="s0.tokenizer_model_vocab",
        title="Tokenizer/model vocabulary compatibility",
        status=GateStatus.PASS if passed else GateStatus.FAIL,
        reason=reason,
        evidence={
            "tokenizer_identity": identity,
            "model_vocab_size": model_vocab,
            "tokenizer_vocab_size": tokenizer_vocab,
            "max_token_id": max_token_id,
        },
    )


def evaluate_s0_integrated(
    evidence: Mapping[str, Any],
    policy: S0GatePolicy | None = None,
) -> dict[str, Any]:
    """Run core S0 gates plus cross-lane integration compatibility gates."""

    result = evaluate_s0(evidence, policy)
    compatibility = _vocab_compatibility_gate(evidence)
    result["tokenizer"] = (
        dict(evidence.get("tokenizer", {}))
        if isinstance(evidence.get("tokenizer", {}), Mapping)
        else evidence.get("tokenizer")
    )
    result["gates"].append(compatibility.to_dict())

    counts = {status.value: 0 for status in GateStatus}
    for gate in result["gates"]:
        counts[gate["status"]] += 1

    required = [gate for gate in result["gates"] if gate.get("required", True)]
    promotion_eligible = all(gate["status"] == GateStatus.PASS.value for gate in required)
    if promotion_eligible:
        overall = GateStatus.PASS
    elif any(gate["status"] == GateStatus.FAIL.value for gate in required):
        overall = GateStatus.FAIL
    else:
        overall = GateStatus.NOT_TESTED

    result["schema_version"] = "12-6.integrated-stage-gate-result.v1"
    result["summary"] = {
        "overall_status": overall.value,
        "promotion_eligible": promotion_eligible,
        "counts": counts,
        "required_gate_count": len(required),
    }
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate integrated 12-6 AI S0 stage gates")
    parser.add_argument("evidence", type=Path, help="machine-readable evidence JSON")
    parser.add_argument("--policy", type=Path, help="optional S0 policy JSON")
    parser.add_argument("--output", type=Path, required=True, help="output result JSON")
    parser.add_argument(
        "--fail-on-ineligible",
        action="store_true",
        help="exit non-zero unless every required gate passes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    evidence = load_json_object(args.evidence)
    policy = (
        S0GatePolicy.from_mapping(load_json_object(args.policy))
        if args.policy is not None
        else S0GatePolicy()
    )
    result = evaluate_s0_integrated(evidence, policy)
    dump_stage_gate_result(result, args.output)
    if args.fail_on_ineligible and not result["summary"]["promotion_eligible"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
