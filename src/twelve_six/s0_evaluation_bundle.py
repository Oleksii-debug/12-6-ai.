"""Bind D04 exact-candidate quality evidence to D02 repeatability evidence.

This adapter is deliberately evaluation-only. It validates that the two independent
machine-evidence streams describe the same exact S0 checkout and frozen identities,
then emits a deterministic audit-ready bundle. It never grants promotion authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twelve_six.stage_gates import evaluate_s0_integrated
from twelve_six.training.s0_repeatability import validate_s0_repeatability_evidence

SCHEMA_VERSION = "12-6.s0-evaluation-repeatability-bundle.v1"
_REPOSITORY = "Oleksii-debug/12-6-ai."
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class S0EvaluationBundleError(ValueError):
    """Raised when quality and repeatability evidence cannot be safely bound."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise S0EvaluationBundleError(message)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} block missing")
    return value


def _canonical_hash(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _bind_identity(
    candidate_evidence: Mapping[str, Any],
    repeatability_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = _mapping(candidate_evidence.get("candidate"), "candidate")
    tokenizer = _mapping(candidate_evidence.get("tokenizer"), "tokenizer")
    dataset = _mapping(candidate_evidence.get("dataset"), "dataset")
    checkpoint = _mapping(candidate_evidence.get("checkpoint"), "checkpoint")
    provenance = _mapping(candidate_evidence.get("provenance"), "provenance")
    repeat_identity = _mapping(
        repeatability_evidence.get("identity"), "repeatability identity"
    )
    environment = _mapping(
        repeat_identity.get("environment"), "repeatability environment"
    )

    candidate_sha = candidate.get("sha")
    _require(
        isinstance(candidate_sha, str) and _GIT_SHA.fullmatch(candidate_sha) is not None,
        "candidate SHA must be a full lowercase 40-hex Git SHA",
    )
    _require(
        provenance.get("repository") == _REPOSITORY,
        "candidate repository identity mismatch",
    )
    _require(
        provenance.get("checkout_head_sha") == candidate_sha,
        "candidate provenance checkout SHA mismatch",
    )
    _require(
        repeat_identity.get("repository") == _REPOSITORY,
        "repeatability repository identity mismatch",
    )
    _require(
        repeat_identity.get("source_sha") == candidate_sha,
        "repeatability evidence is stale for this candidate SHA",
    )

    bindings = {
        "repository": _REPOSITORY,
        "candidate_sha": candidate_sha,
        "modelspec_sha256": candidate.get("modelspec_sha256"),
        "initspec_sha256": candidate.get("initspec_sha256"),
        "parameter_count": candidate.get("parameter_count"),
        "dataset_manifest_sha256": dataset.get("manifest_sha256"),
        "tokenizer_config_sha256": tokenizer.get("config_sha256"),
        "tokenizer_vocab_sha256": tokenizer.get("vocab_sha256"),
        "packing_config_sha256": checkpoint.get("packing_sha256"),
        "environment_lock_file_sha256": checkpoint.get(
            "environment_lock_sha256"
        ),
    }
    comparisons = {
        "modelspec_sha256": repeat_identity.get("modelspec_sha256"),
        "initspec_sha256": repeat_identity.get("initspec_sha256"),
        "parameter_count": repeat_identity.get("parameter_count"),
        "dataset_manifest_sha256": repeat_identity.get("dataset_manifest_sha256"),
        "tokenizer_config_sha256": repeat_identity.get("tokenizer_config_sha256"),
        "tokenizer_vocab_sha256": repeat_identity.get("tokenizer_vocab_sha256"),
        "packing_config_sha256": repeat_identity.get("packing_config_sha256"),
        "environment_lock_file_sha256": environment.get(
            "lock_index_file_sha256"
        ),
    }
    for field, expected in bindings.items():
        if field in {"repository", "candidate_sha"}:
            continue
        if field.endswith("sha256"):
            _require(
                isinstance(expected, str) and _SHA256.fullmatch(expected) is not None,
                f"candidate {field} is not a lowercase SHA-256",
            )
        _require(
            comparisons[field] == expected,
            f"candidate/repeatability identity mismatch: {field}",
        )
    return bindings


def build_s0_evaluation_bundle(
    candidate_evidence: Mapping[str, Any],
    repeatability_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and bind quality plus repeatability evidence for one exact S0 SHA."""
    _require(
        candidate_evidence.get("schema_version")
        == "12-6.s0-real-candidate-evidence.v2",
        "wrong D04 candidate evidence schema",
    )
    validate_s0_repeatability_evidence(repeatability_evidence)
    identity = _bind_identity(candidate_evidence, repeatability_evidence)

    gate_report = evaluate_s0_integrated(candidate_evidence)
    summary = _mapping(gate_report.get("summary"), "stage-gate summary")
    counts = _mapping(summary.get("counts"), "stage-gate counts")
    _require(
        summary.get("evaluation_complete") is True,
        "D04/D06 quality evaluation is incomplete",
    )
    _require(
        summary.get("overall_status") == "PASS",
        "D04/D06 quality evaluation is not PASS",
    )
    _require(counts.get("FAIL") == 0, "quality gate report contains FAIL")
    _require(
        counts.get("NOT_TESTED") == 0,
        "quality gate report contains NOT_TESTED",
    )

    proof = _mapping(repeatability_evidence.get("proof"), "repeatability proof")
    _require(
        proof.get("same_seed_exact_equivalence") is True,
        "same-seed exact repeatability is not proven",
    )
    _require(
        proof.get("different_seed_initialization_diverges") is True,
        "seed causality is not proven at initialization",
    )
    _require(
        proof.get("different_seed_training_diverges") is True,
        "seed causality is not proven for the training trace",
    )
    _require(
        proof.get("validation_optimized_tokens") == 0,
        "repeatability evidence optimized held-out validation tokens",
    )

    repeatability_hash = repeatability_evidence.get("evidence_sha256")
    _require(
        isinstance(repeatability_hash, str)
        and _SHA256.fullmatch(repeatability_hash) is not None,
        "repeatability evidence SHA-256 missing",
    )
    environment = _mapping(
        _mapping(repeatability_evidence.get("identity"), "repeatability identity").get(
            "environment"
        ),
        "repeatability environment",
    )

    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authority": "EVALUATION_AND_REPRODUCIBILITY_EVIDENCE_NOT_PROMOTION",
        "identity": identity,
        "quality": {
            "overall_status": summary.get("overall_status"),
            "evaluation_complete": summary.get("evaluation_complete"),
            "counts": dict(counts),
            "required_gate_count": summary.get("required_gate_count"),
            "candidate_evidence_sha256": _canonical_hash(candidate_evidence),
        },
        "repeatability": {
            "evidence_sha256": repeatability_hash,
            "same_seed_exact_equivalence": proof.get(
                "same_seed_exact_equivalence"
            ),
            "different_seed_initialization_diverges": proof.get(
                "different_seed_initialization_diverges"
            ),
            "different_seed_training_diverges": proof.get(
                "different_seed_training_diverges"
            ),
            "validation_optimized_tokens": proof.get(
                "validation_optimized_tokens"
            ),
            "environment_evidence_sha256": environment.get(
                "environment_evidence_sha256"
            ),
        },
        "promotion_boundary": {
            "bundle_grants_promotion": False,
            "source_promotion_eligible": summary.get("promotion_eligible"),
            "source_promotion_authority_status": summary.get(
                "promotion_authority_status"
            ),
            "required_external_authority": [
                "exact-head CI and integration manifest",
                "independent AUDIT-A verdict on the same candidate SHA",
                "independent AUDIT-B verdict on the same candidate SHA",
                "D10 live governance/release authority",
            ],
        },
    }
    bundle["bundle_sha256"] = _canonical_hash(bundle)
    return bundle


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind strict S0 candidate evaluation to exact-source repeatability evidence"
        )
    )
    parser.add_argument("--candidate-evidence", type=Path, required=True)
    parser.add_argument("--repeatability-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    bundle = build_s0_evaluation_bundle(
        _load_json(args.candidate_evidence),
        _load_json(args.repeatability_evidence),
    )
    _write_json(args.output, bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
