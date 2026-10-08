"""Plan 6 terminal LOCAL_FREE: signed research -> candidate -> promotion -> rollback.

This is fixture-only. It never executes training, a provider, or production writes.
All fourteen incumbent engines are separately exercised by the workflow.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from dataclasses import asdict, replace

import pytest

from twelve_six.champion_lifecycle import (
    Authorization, Descendant, Evaluation, LifecycleEvent, LifecycleState,
    decide_candidate, rollback_promotion,
)
from twelve_six.experience_replay import _mac
from twelve_six.research_engine import (
    STAGES, ResearchTrial, issue_verified_stage, verify_research_bundle,
)

BASE, CHAMPION, CHILD = "a" * 64, "b" * 64, "c" * 64
PROTOCOL, DATA, COMPAT = "d" * 64, "e" * 64, "f" * 64
VERIFIER_KEY, AUTHOR_KEY, STAGE_KEY = b"v" * 32, b"g" * 32, b"s" * 32

# The only Plan-6 engines. Reuse their canonical executable entrypoints.
ENGINES = {
    "research_engine": "verify_research_bundle",
    "post_base_instruction": "train_instruction_descendant",
    "post_base_preference": "train_preference_descendant",
    "post_base_reasoning": "run_verified_rl",
    "post_base_agentic": "train_agentic_descendant",
    "teacher_council": "decide_teacher_council",
    "verified_data_factory": "generate_verified_pool",
    "self_play_engine": "run_self_play",
    "continual_learning": "run_continual_update",
    "experience_replay": "build_replay_candidates",
    "autonomous_curriculum": "plan_curriculum",
    "evolutionary_code": "evaluate_evolution",
    "autonomous_research_loop": "record_experiment",
    "champion_lifecycle": "rollback_promotion",
}


def trial():
    return ResearchTrial.create(
        protocol_sha256=PROTOCOL, model_sha256=CHILD, data_sha256=DATA,
        artifact_sha256=CHILD, seed=17,
        config={"scope": "plan6_fixture", "real_training": False, "paid_compute": False},
        producer_id="candidate-producer",
    )


def stages(research_trial):
    return tuple(
        issue_verified_stage(
            research_trial, stage=stage, verifier_id="independent-verifier",
            verifier_key=STAGE_KEY, sample={"x": index},
            expected_output={"score": 17 + index},
            evaluator=lambda seed, sample: {"score": seed + sample["x"]},
        )
        for index, stage in enumerate(STAGES)
    )


def verified_receipt(research_trial, evidence):
    return verify_research_bundle(
        research_trial, evidence, verifier_id="independent-verifier",
        verifier_key=STAGE_KEY,
    )


def candidate():
    return Descendant("fixture-descendant-1", "candidate-producer", CHAMPION,
                      CHILD, BASE, CHILD, DATA, COMPAT)


def evaluated(subject, receipt, qualified=True):
    raw = Evaluation(subject.identity(), CHAMPION, BASE, COMPAT, PROTOCOL,
                     receipt, "independent-verifier", PROTOCOL,
                     qualified, True, "0" * 64)
    return replace(raw, signature=_mac(VERIFIER_KEY, raw.payload()))


def trusted(state, receipt):
    return dict(
        trusted_state_sha256=state.head_sha256,
        verifier_id="independent-verifier", verifier_version_sha256=PROTOCOL,
        trusted_evidence_roots=frozenset({PROTOCOL, receipt}),
        verifier_key=VERIFIER_KEY, authorizer_id="release-authorizer",
        authorizer_version_sha256=DATA, authorizer_key=AUTHOR_KEY,
    )


def authorized(state, subject, evaluation, action="PROMOTE"):
    raw = Authorization(action, subject.identity(), evaluation.identity(),
                        state.head_sha256, "release-authorizer", DATA, "0" * 64)
    return replace(raw, signature=_mac(AUTHOR_KEY, raw.payload()))


def restore(payload):
    fields = json.loads(payload)
    events = tuple(LifecycleEvent(**event) for event in fields.pop("events"))
    state = LifecycleState(**fields, events=events)
    state.validate()
    return state


def test_source_manifest_pins_all_fourteen_exact_engine_blobs():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "configs/plan6/evolution_engine_release_v1.json").read_text(
            encoding="utf-8",
        ),
    )
    assert manifest["schema"] == "12-6.plan6.evolution-engine-release.v1"
    assert manifest["execution_profile"] == "LOCAL_FREE"
    assert manifest["limits"]["paid_model_or_teacher_compute"] is False
    blobs = manifest["engine_git_blobs"]
    assert set(blobs) == {f"src/twelve_six/{name}.py" for name in ENGINES}
    for relative, expected in blobs.items():
        content = (root / relative).read_bytes()
        observed = hashlib.sha1(
            ("blob " + str(len(content))).encode("ascii") + bytes([0]) + content,
        ).hexdigest()
        assert observed == expected, relative


def test_fourteen_incumbent_engines_keep_importable_entrypoints():
    assert len(ENGINES) == 14
    for module, entrypoint in ENGINES.items():
        assert callable(getattr(importlib.import_module(f"twelve_six.{module}"), entrypoint))


def test_research_pyramid_promotion_restart_and_signed_rollback(tmp_path):
    research_trial = trial()
    proof = verified_receipt(research_trial, stages(research_trial))
    subject = candidate()
    assert research_trial.artifact_sha256 == subject.descendant_sha256
    original = LifecycleState.genesis(BASE, CHAMPION)
    evaluation = evaluated(subject, proof)

    with pytest.raises(ValueError, match="authorization"):
        decide_candidate(original, subject, evaluation, **trusted(original, proof))
    promotion = authorized(original, subject, evaluation)
    promoted = decide_candidate(
        original, subject, evaluation, promotion, **trusted(original, proof),
    )
    assert promoted.champion_sha256 == CHILD
    assert original.champion_sha256 == CHAMPION and original.events == ()
    assert promoted.events[0].action == "PROMOTED"

    # Durable interruption/restart boundary, with independent fresh objects.
    path = tmp_path / "lifecycle.json"
    path.write_text(json.dumps(asdict(promoted), sort_keys=True), encoding="utf-8")
    recovered = restore(path.read_text(encoding="utf-8"))
    assert recovered == promoted
    rollback = authorized(recovered, subject, evaluation, "ROLLBACK")
    rolled = rollback_promotion(
        recovered, subject, evaluation, rollback,
        trusted_state_sha256=recovered.head_sha256,
        authorizer_id="release-authorizer", authorizer_version_sha256=DATA,
        authorizer_key=AUTHOR_KEY,
    )
    assert rolled.champion_sha256 == CHAMPION
    assert tuple(event.action for event in rolled.events) == ("PROMOTED", "ROLLED_BACK")
    assert rolled.events[0] == promoted.events[0]
    assert restore(json.dumps(asdict(rolled))) == rolled


def test_missing_forged_foreign_or_self_issued_pyramid_fails_closed():
    research_trial = trial()
    evidence = stages(research_trial)
    with pytest.raises(ValueError):
        verified_receipt(research_trial, evidence[:-1])
    with pytest.raises(ValueError):
        verified_receipt(
            research_trial, (replace(evidence[0], signature="0" * 64),) + evidence[1:],
        )
    with pytest.raises(ValueError):
        verified_receipt(replace(research_trial, artifact_sha256=DATA), evidence)
    with pytest.raises(ValueError, match="self-issued"):
        issue_verified_stage(
            research_trial, stage="holdout", verifier_id=research_trial.producer_id,
            verifier_key=STAGE_KEY, sample={"x": 0}, expected_output={"score": 17},
            evaluator=lambda *_: {"score": 17},
        )


def test_failure_forgery_stale_grant_and_wrong_rollback_preserve_champion():
    proof = verified_receipt(trial(), stages(trial()))
    subject = candidate()
    original = LifecycleState.genesis(BASE, CHAMPION)
    failure = evaluated(subject, proof, qualified=False)
    rejected = decide_candidate(original, subject, failure, **trusted(original, proof))
    assert rejected.champion_sha256 == CHAMPION
    assert rejected.events[0].action == "REJECTED"
    assert original.events == ()

    evaluation = evaluated(subject, proof)
    grant = authorized(original, subject, evaluation)
    for forged in (
        replace(grant, signature="0" * 64),
        replace(grant, expected_state_sha256=BASE),
    ):
        with pytest.raises(ValueError):
            decide_candidate(original, subject, evaluation, forged,
                             **trusted(original, proof))
    with pytest.raises(ValueError):
        decide_candidate(original, subject, evaluated(subject, DATA), grant,
                         **trusted(original, proof))
    promoted = decide_candidate(original, subject, evaluation, grant,
                                **trusted(original, proof))
    with pytest.raises(ValueError):
        decide_candidate(promoted, subject, evaluation, grant,
                         **trusted(promoted, proof))
    with pytest.raises(ValueError):
        rollback_promotion(
            promoted, subject, evaluation, grant,
            trusted_state_sha256=promoted.head_sha256,
            authorizer_id="release-authorizer", authorizer_version_sha256=DATA,
            authorizer_key=AUTHOR_KEY,
        )
    assert original.champion_sha256 == CHAMPION
