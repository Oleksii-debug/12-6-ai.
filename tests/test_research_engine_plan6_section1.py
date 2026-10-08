from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from twelve_six.research_engine import (
    STAGES,
    ResearchTrial,
    issue_verified_stage,
    verify_research_bundle,
)

KEY = b"fixture-independent-verifier-key-0123456789"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def trial(seed: int = 17) -> ResearchTrial:
    return ResearchTrial.create(
        protocol_sha256=sha("protocol-v1"), model_sha256=sha("tiny-model"),
        data_sha256=sha("dataset"), artifact_sha256=sha("candidate"),
        seed=seed, config={"batch": 2, "lr": 0.01}, producer_id="model-actor",
    )


def evaluator(seed: int, sample: object) -> object:
    assert isinstance(sample, dict)
    return {"score": seed + sample["x"]}


def proofs(candidate: ResearchTrial):
    return tuple(
        issue_verified_stage(
            candidate, stage=stage, verifier_id="independent-verifier",
            verifier_key=KEY, sample={"x": index},
            expected_output={"score": candidate.seed + index}, evaluator=evaluator,
        )
        for index, stage in enumerate(STAGES)
    )


def verify(candidate: ResearchTrial, evidence):
    return verify_research_bundle(
        candidate, evidence, verifier_id="independent-verifier", verifier_key=KEY,
    )


def test_trial_replay_pyramid_and_recovery() -> None:
    candidate = trial()
    receipt = verify(candidate, proofs(candidate))
    assert len(receipt) == 64
    assert receipt == verify(trial(), proofs(trial()))
    assert trial(18).identity_sha256 != candidate.identity_sha256


@pytest.mark.parametrize("mode", ["missing", "reversed", "duplicated"])
def test_fail_closed_evidence_hierarchy(mode: str) -> None:
    candidate = trial()
    evidence = proofs(candidate)
    bad = {
        "missing": evidence[:-1],
        "reversed": evidence[::-1],
        "duplicated": evidence[:2] + evidence[1:4],
    }[mode]
    with pytest.raises(ValueError):
        verify(candidate, bad)


def test_reject_self_claim_forgery_wrong_verifier_and_wrong_trial() -> None:
    candidate = trial()
    with pytest.raises(ValueError, match="self-issued"):
        issue_verified_stage(
            candidate, stage="independent", verifier_id="model-actor",
            verifier_key=KEY, sample={"x": 1}, expected_output={"score": 18},
            evaluator=evaluator,
        )
    evidence = proofs(candidate)
    with pytest.raises(ValueError):
        verify(candidate, (replace(evidence[0], observed_sha256=sha("lie")),) + evidence[1:])
    with pytest.raises(ValueError):
        verify(trial(19), evidence)
    with pytest.raises(ValueError, match="untrusted"):
        verify_research_bundle(
            candidate, evidence, verifier_id="independent-verifier",
            verifier_key=b"X" * 32,
        )
    with pytest.raises(ValueError):
        verify_research_bundle(
            candidate, evidence, verifier_id="model-actor", verifier_key=KEY,
        )


def test_holdout_mismatch_and_replay_instability_fail_closed() -> None:
    candidate = trial()
    with pytest.raises(ValueError, match="mismatch"):
        issue_verified_stage(
            candidate, stage="holdout", verifier_id="independent-verifier",
            verifier_key=KEY, sample={"x": 2}, expected_output={"score": -1},
            evaluator=evaluator,
        )
    outputs = iter([{"score": 1}, {"score": 2}])
    with pytest.raises(ValueError, match="instability"):
        issue_verified_stage(
            candidate, stage="replay", verifier_id="independent-verifier",
            verifier_key=KEY, sample={"x": 2}, expected_output={"score": 1},
            evaluator=lambda *_: next(outputs),
        )


@pytest.mark.parametrize("config", [
    {"nan": float("nan")}, {"bad": object()}, {"nested": {"bad": object()}},
])
def test_unsafe_or_unserializable_config_is_rejected(config: dict) -> None:
    with pytest.raises(ValueError):
        ResearchTrial.create(
            protocol_sha256=sha("p"), model_sha256=sha("m"),
            data_sha256=sha("d"), artifact_sha256=sha("a"),
            seed=1, config=config, producer_id="model-actor",
        )


@pytest.mark.parametrize("changes", [
    {"config_json": b'{"z":1,"a":2}'},
    {"config_json": b'{"x":1,"x":2}'},
    {"config_json": b'{"x":NaN}'},
    {"seed": True},
    {"protocol_sha256": "A" * 64},
    {"producer_id": ""},
])
def test_invalid_frozen_trial_fails_closed(changes: dict) -> None:
    with pytest.raises(ValueError):
        replace(trial(), **changes)


def test_artifact_or_config_drift_requires_new_evidence() -> None:
    candidate = trial()
    altered = replace(candidate, artifact_sha256=sha("new-candidate"))
    with pytest.raises(ValueError, match="foreign"):
        verify(altered, proofs(candidate))
    assert altered.identity_sha256 != candidate.identity_sha256
