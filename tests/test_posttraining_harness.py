from decimal import Decimal

import pytest

from twelve_six.posttraining.contracts import Candidate, CheckpointRef, LineageKind, VerifierTask
from twelve_six.posttraining.harness import VerificationCase, run_verifier_harness
from twelve_six.posttraining.verifiers import (
    ExactTextVerifier,
    NumericToleranceVerifier,
    VerifierRegistry,
)

HEX_A = "a" * 64
HEX_B = "b" * 64


def _candidate(text: str) -> Candidate:
    return Candidate(
        candidate_id="candidate-1",
        prompt_id="prompt-1",
        text=text,
        checkpoint=CheckpointRef(
            checkpoint_id="base-s0",
            sha256=HEX_A,
            git_sha="f2e94c7",
            stage="S0",
            lineage=LineageKind.BASE,
        ),
        generation_config_sha256=HEX_B,
    )


def _registry() -> VerifierRegistry:
    registry = VerifierRegistry()
    registry.register(ExactTextVerifier())
    registry.register(NumericToleranceVerifier(tolerance=Decimal("0.01")))
    return registry


def test_harness_runs_multiple_verifiers_without_training_side_effects() -> None:
    report = run_verifier_harness(
        _registry(),
        (
            VerificationCase(
                task=VerifierTask(task_id="math-1", prompt="2+2?", reference="4"),
                candidate=_candidate("4"),
                verifier_names=("exact_text", "numeric_tolerance"),
            ),
        ),
    )
    assert report.total_verifications == 2
    assert report.passed_verifications == 2
    assert report.mean_score == 1.0
    assert report.all_cases_passed


def test_harness_preserves_failed_verifier_as_evidence() -> None:
    report = run_verifier_harness(
        _registry(),
        (
            VerificationCase(
                task=VerifierTask(task_id="math-2", prompt="2+2?", reference="4"),
                candidate=_candidate("5"),
                verifier_names=("exact_text",),
            ),
        ),
    )
    assert report.passed_verifications == 0
    assert report.cases[0].results[0].reason == "text mismatch"
    assert not report.all_cases_passed


def test_harness_fails_closed_for_unknown_verifier() -> None:
    with pytest.raises(KeyError, match="unknown verifier"):
        run_verifier_harness(
            _registry(),
            (
                VerificationCase(
                    task=VerifierTask(task_id="math-3", prompt="2+2?", reference="4"),
                    candidate=_candidate("4"),
                    verifier_names=("missing-verifier",),
                ),
            ),
        )


def test_verification_case_requires_a_verifier() -> None:
    with pytest.raises(ValueError, match="at least one verifier"):
        VerificationCase(
            task=VerifierTask(task_id="math-4", prompt="2+2?", reference="4"),
            candidate=_candidate("4"),
            verifier_names=(),
        )
