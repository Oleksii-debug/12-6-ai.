from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.integration.s0_repeatability_intake import (
    RepeatabilityIntakeError,
    validate_repeatability_intake,
    verify_repeatability_ancestry,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "configs"
    / "releases"
    / "s0_candidate_repeatability_convergence_20260825.experimental.json"
)
RUN_CONFIG = ROOT / "configs" / "runs" / "s0_10k.d02_repeatability.json"


def _payload() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_repeatability_successor_is_accepted_but_never_self_promoted() -> None:
    payload = _payload()
    facts = validate_repeatability_intake(payload)

    assert facts["parent_candidate_sha"] == "59193ada9586d0542b027f46d32ac923841fae7f"
    assert facts["repeatability_source_sha"] == "c631c024e641dac102036fafee6d78ba31c067cd"
    assert facts["artifact_id"] == 9539007219
    assert facts["evidence_sha256"] == (
        "263e372d413ca8be98f2ee20210b6ce5a6bed0e25a068519362fa181e519e1f1"
    )
    assert facts["promotion_eligible"] is False
    assert set(facts["workflow_run_ids"].values()) == {
        32778688850,
        32778688832,
        32778688844,
        32778688829,
    }


def test_repeatability_and_parent_sources_are_real_git_ancestors() -> None:
    facts = verify_repeatability_ancestry(_payload(), ROOT)

    assert facts["head_sha"] not in {
        facts["parent_candidate_sha"],
        facts["repeatability_source_sha"],
    }


def test_repeatability_run_config_matches_integrated_evidence_contract() -> None:
    payload = _payload()
    run = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    evidence = payload["accepted_successor"]["evidence"]

    assert run["authority"] == "LOCAL_FREE_OR_FREE_HOSTED_CPU_EVIDENCE_NOT_PROMOTION"
    assert run["parent_d02_evidence_sha"] == "e5a3b551fa509fd6d36f51915cd887f5cc352f69"
    assert run["same_seed"] == evidence["same_seed"] == 1337
    assert run["different_seed"] == evidence["different_seed"] == 1338
    assert run["same_seed_fresh_runs"] == 2
    assert run["different_seed_runs"] == 1
    assert run["validation_optimized_tokens"] == evidence["validation_optimized_tokens"] == 0
    assert run["promotion_authority"] is False


def test_intake_rejects_stale_or_failed_exact_head_workflow() -> None:
    payload = copy.deepcopy(_payload())
    payload["accepted_successor"]["workflows"][0]["conclusion"] = "failure"

    with pytest.raises(RepeatabilityIntakeError, match="not exact-head success"):
        validate_repeatability_intake(payload)


def test_intake_rejects_repeatability_source_sha_drift() -> None:
    payload = copy.deepcopy(_payload())
    payload["accepted_successor"]["evidence"]["source_sha"] = "a" * 40

    with pytest.raises(RepeatabilityIntakeError, match="evidence source SHA drift"):
        validate_repeatability_intake(payload)


def test_intake_rejects_false_same_seed_equivalence() -> None:
    payload = copy.deepcopy(_payload())
    payload["accepted_successor"]["evidence"]["same_seed_exact_equivalence"] = False

    with pytest.raises(RepeatabilityIntakeError, match="same-seed exact equivalence"):
        validate_repeatability_intake(payload)


def test_intake_rejects_missing_seed_causality() -> None:
    payload = copy.deepcopy(_payload())
    payload["accepted_successor"]["evidence"]["different_seed_initialization_diverges"] = False

    with pytest.raises(RepeatabilityIntakeError, match="initialization causality"):
        validate_repeatability_intake(payload)


def test_intake_rejects_validation_optimization() -> None:
    payload = copy.deepcopy(_payload())
    payload["accepted_successor"]["evidence"]["validation_optimized_tokens"] = 1

    with pytest.raises(RepeatabilityIntakeError, match="validation data"):
        validate_repeatability_intake(payload)


def test_intake_rejects_tampered_artifact_digest() -> None:
    payload = copy.deepcopy(_payload())
    payload["accepted_successor"]["artifact"]["digest"] = "sha256:not-a-digest"

    with pytest.raises(RepeatabilityIntakeError, match="SHA-256"):
        validate_repeatability_intake(payload)


@pytest.mark.parametrize(
    "claim",
    [
        "candidate_or_stable_promotion",
        "cross_hardware_bitwise_reproducibility",
        "gpu_reproducibility",
        "distributed_reproducibility",
        "foreign_pretrained_weights_used",
        "instruction_or_alignment_training",
        "paid_compute_authorized_or_used",
    ],
)
def test_intake_rejects_unsafe_or_unproven_claims(claim: str) -> None:
    payload = copy.deepcopy(_payload())
    payload["claims"][claim] = True

    with pytest.raises(RepeatabilityIntakeError, match="unsafe or unsupported claim"):
        validate_repeatability_intake(payload)


def test_intake_preserves_historical_auditor_authority() -> None:
    payload = copy.deepcopy(_payload())
    payload["audits"]["AUDIT-A"]["verdict"] = "PASS"

    with pytest.raises(RepeatabilityIntakeError, match="historical audit verdict"):
        validate_repeatability_intake(payload)
