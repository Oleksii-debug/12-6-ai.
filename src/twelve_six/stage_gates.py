"""Integrated S0 stage gates layered on the core D06 evaluation engine."""

from __future__ import annotations

import argparse
import re
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
_EXACT_GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PASSING_AUDIT_VERDICTS = frozenset({"PASS", "PASS_WITH_NOTES"})


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


def _strengthen_exact_candidate_identity(
    gates: list[dict[str, Any]], evidence: Mapping[str, Any]
) -> None:
    """Require an exact lowercase Git object id for the candidate identity gate."""

    candidate_sha = _get(evidence, "candidate.sha")
    if candidate_sha is _MISSING:
        return
    identity_gate = next((gate for gate in gates if gate["gate_id"] == "s0.identity"), None)
    if identity_gate is None or identity_gate["status"] != GateStatus.PASS.value:
        return
    if not isinstance(candidate_sha, str) or _EXACT_GIT_SHA.fullmatch(candidate_sha) is None:
        identity_gate["status"] = GateStatus.FAIL.value
        identity_gate["reason"] = "candidate.sha must be an exact lowercase 40- or 64-hex Git id"
        identity_gate["evidence"]["candidate.sha"] = candidate_sha


def _audit_binding(
    evidence: Mapping[str, Any], audit_key: str, candidate_sha: str
) -> tuple[bool, list[str], dict[str, Any]]:
    base = f"promotion.{audit_key}"
    verdict = _get(evidence, f"{base}.verdict")
    audit_sha = _get(evidence, f"{base}.candidate_sha")
    evidence_ref = _get(evidence, f"{base}.evidence_ref")
    captured = {
        "verdict": verdict if verdict is not _MISSING else None,
        "candidate_sha": audit_sha if audit_sha is not _MISSING else None,
        "evidence_ref": evidence_ref if evidence_ref is not _MISSING else None,
    }
    missing = [
        path
        for path, value in (
            (f"{base}.verdict", verdict),
            (f"{base}.candidate_sha", audit_sha),
            (f"{base}.evidence_ref", evidence_ref),
        )
        if value is _MISSING
    ]
    if missing:
        return False, missing, captured

    blockers: list[str] = []
    if verdict not in _PASSING_AUDIT_VERDICTS:
        blockers.append(f"{base}.verdict must be PASS or PASS_WITH_NOTES")
    if not isinstance(audit_sha, str) or _EXACT_GIT_SHA.fullmatch(audit_sha) is None:
        blockers.append(f"{base}.candidate_sha must be an exact lowercase Git object id")
    elif audit_sha != candidate_sha:
        blockers.append(f"{base}.candidate_sha does not match candidate.sha")
    if not isinstance(evidence_ref, str) or not evidence_ref.strip():
        blockers.append(f"{base}.evidence_ref must be a non-empty durable reference")
    return not blockers, blockers, captured


def _promotion_authority(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate non-capability authority required before D06 may report promotion eligibility.

    This does not replace D10 or either independent auditor. It only prevents an
    all-green D06 evaluation fixture/component bundle from being mislabeled as promotable.
    """

    required_paths = (
        "candidate.sha",
        "candidate.integrated",
        "promotion.candidate_manifest_validated",
        "promotion.candidate_manifest_sha256",
        "promotion.candidate_ci.success",
        "promotion.candidate_ci.run_id",
        "promotion.audit_a.verdict",
        "promotion.audit_a.candidate_sha",
        "promotion.audit_a.evidence_ref",
        "promotion.audit_b.verdict",
        "promotion.audit_b.candidate_sha",
        "promotion.audit_b.evidence_ref",
    )
    missing = [path for path in required_paths if _get(evidence, path) is _MISSING]
    if missing:
        return {
            "status": GateStatus.NOT_TESTED.value,
            "reason": "promotion authority evidence is incomplete",
            "blockers": [f"missing evidence: {path}" for path in missing],
        }

    candidate_sha = _get(evidence, "candidate.sha")
    integrated = _get(evidence, "candidate.integrated")
    manifest_validated = _get(evidence, "promotion.candidate_manifest_validated")
    manifest_sha = _get(evidence, "promotion.candidate_manifest_sha256")
    ci_success = _get(evidence, "promotion.candidate_ci.success")
    ci_run_id = _get(evidence, "promotion.candidate_ci.run_id")

    blockers: list[str] = []
    if not isinstance(candidate_sha, str) or _EXACT_GIT_SHA.fullmatch(candidate_sha) is None:
        blockers.append("candidate.sha must be an exact lowercase 40- or 64-hex Git id")
        candidate_sha_for_binding = ""
    else:
        candidate_sha_for_binding = candidate_sha
    if type(integrated) is not bool or not integrated:
        blockers.append("candidate.integrated must be exact boolean true")
    if type(manifest_validated) is not bool or not manifest_validated:
        blockers.append("promotion.candidate_manifest_validated must be exact boolean true")
    if not isinstance(manifest_sha, str) or _SHA256.fullmatch(manifest_sha) is None:
        blockers.append("promotion.candidate_manifest_sha256 must be a lowercase 64-hex SHA-256")
    if type(ci_success) is not bool or not ci_success:
        blockers.append("promotion.candidate_ci.success must be exact boolean true")
    if isinstance(ci_run_id, bool) or not isinstance(ci_run_id, int) or ci_run_id <= 0:
        blockers.append("promotion.candidate_ci.run_id must be a positive integer")

    audit_evidence: dict[str, Any] = {}
    for audit_key in ("audit_a", "audit_b"):
        ok, audit_blockers, captured = _audit_binding(
            evidence, audit_key, candidate_sha_for_binding
        )
        audit_evidence[audit_key] = captured
        if not ok:
            blockers.extend(audit_blockers)

    audit_a_ref = audit_evidence["audit_a"]["evidence_ref"]
    audit_b_ref = audit_evidence["audit_b"]["evidence_ref"]
    if (
        isinstance(audit_a_ref, str)
        and audit_a_ref.strip()
        and audit_a_ref == audit_b_ref
    ):
        blockers.append("AUDIT-A and AUDIT-B must have distinct durable evidence references")

    return {
        "status": GateStatus.PASS.value if not blockers else GateStatus.FAIL.value,
        "reason": (
            "candidate integration, CI, manifest, and both independent audits are bound"
            if not blockers
            else "promotion authority evidence failed closed"
        ),
        "blockers": blockers,
        "evidence": {
            "candidate_sha": candidate_sha,
            "candidate_integrated": integrated,
            "candidate_manifest_validated": manifest_validated,
            "candidate_manifest_sha256": manifest_sha,
            "candidate_ci": {"success": ci_success, "run_id": ci_run_id},
            **audit_evidence,
        },
    }


def evaluate_s0_integrated(
    evidence: Mapping[str, Any],
    policy: S0GatePolicy | None = None,
) -> dict[str, Any]:
    """Run core S0 gates plus cross-lane compatibility and promotion authority checks."""

    result = evaluate_s0(evidence, policy)
    compatibility = _vocab_compatibility_gate(evidence)
    result["tokenizer"] = (
        dict(evidence.get("tokenizer", {}))
        if isinstance(evidence.get("tokenizer", {}), Mapping)
        else evidence.get("tokenizer")
    )
    result["gates"].append(compatibility.to_dict())
    _strengthen_exact_candidate_identity(result["gates"], evidence)

    counts = {status.value: 0 for status in GateStatus}
    for gate in result["gates"]:
        counts[gate["status"]] += 1

    required = [gate for gate in result["gates"] if gate.get("required", True)]
    evaluation_complete = all(gate["status"] == GateStatus.PASS.value for gate in required)
    if evaluation_complete:
        overall = GateStatus.PASS
    elif any(gate["status"] == GateStatus.FAIL.value for gate in required):
        overall = GateStatus.FAIL
    else:
        overall = GateStatus.NOT_TESTED

    promotion_authority = _promotion_authority(evidence)
    promotion_eligible = (
        evaluation_complete and promotion_authority["status"] == GateStatus.PASS.value
    )

    result["schema_version"] = "12-6.integrated-stage-gate-result.v2"
    result["promotion_authority"] = promotion_authority
    result["summary"] = {
        "overall_status": overall.value,
        "evaluation_complete": evaluation_complete,
        "promotion_eligible": promotion_eligible,
        "promotion_authority_status": promotion_authority["status"],
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
        "--fail-on-incomplete",
        action="store_true",
        help="exit non-zero unless every required D06 evaluation gate passes",
    )
    parser.add_argument(
        "--fail-on-ineligible",
        action="store_true",
        help="exit non-zero unless evaluation and external promotion authority both pass",
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
    if args.fail_on_incomplete and not result["summary"]["evaluation_complete"]:
        return 2
    if args.fail_on-ineligible and not result["summary"]["promotion_eligible"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
