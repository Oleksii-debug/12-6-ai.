"""Plan 6 Section 13: deterministic LOCAL_FREE research-loop qualification."""
from dataclasses import replace

import pytest

from twelve_six.autonomous_research_loop import (
    GENESIS, ExperimentProposal, ForecastEvidence, OutcomeEvidence,
    ResearchMap, ResearchPolicy, record_experiment, select_experiment,
)
from twelve_six.experience_replay import _mac
from twelve_six.research_engine import (
    STAGES, ResearchTrial, issue_verified_stage, verify_research_bundle,
)

H = "a" * 64
FORECAST_KEY = b"f" * 32
STAGE_KEY = b"s" * 32
OUTCOME_KEY = b"o" * 32
FORECAST_ROOT = "b" * 64
OUTCOME_ROOT = "c" * 64
FORECAST_VERSION = "d" * 64
OUTCOME_VERSION = "e" * 64

POLICY = ResearchPolicy(
    budget_units=6, max_rounds=4, min_expected_gain=10,
    min_actual_gain=2, max_low_gain_rounds=2,
)


def make_proposal(state=ResearchMap(), name="trial-01", cost=2):
    config = {
        "experiment_id": name,
        "hypothesis": "A verifiable improvement changes known scores.",
        "expected_falsifier": "Independent holdout score does not improve.",
        "parent_map_sha256": state.head_sha256,
        "max_cost_units": cost,
    }
    trial = ResearchTrial.create(
        protocol_sha256=H, model_sha256=H, data_sha256=H,
        artifact_sha256=H, seed=17, config=config, producer_id="generator",
    )
    proposal = ExperimentProposal(
        name, config["hypothesis"], config["expected_falsifier"],
        trial, state.head_sha256, cost,
    )
    return proposal


def make_forecast(proposal, gain=80):
    item = ForecastEvidence(
        proposal.identity(), "independent-forecast", FORECAST_VERSION,
        FORECAST_ROOT, gain, H,
    )
    return replace(item, signature=_mac(FORECAST_KEY, item.payload()))


def trust(state):
    return {
        "trusted_map_sha256": state.head_sha256,
        "trusted_forecast_id": "independent-forecast",
        "trusted_forecast_version_sha256": FORECAST_VERSION,
        "trusted_forecast_roots": frozenset({FORECAST_ROOT}),
        "forecast_key": FORECAST_KEY,
    }


def choose(state=ResearchMap(), name="trial-01", gain=80, cost=2):
    p = make_proposal(state, name, cost)
    f = make_forecast(p, gain)
    return p, f, select_experiment(state, POLICY, (p,), (f,), **trust(state))


def stages(proposal):
    return tuple(
        issue_verified_stage(
            proposal.trial, stage=stage, verifier_id="stage-verifier",
            verifier_key=STAGE_KEY, sample={"index": i},
            expected_output={"value": proposal.trial.seed + i},
            evaluator=lambda seed, sample: {"value": seed + sample["index"]},
        )
        for i, stage in enumerate(STAGES)
    )


def make_outcome(proposal, proofs, gain=5, spent=2, verdict="SUPPORTED"):
    receipt = verify_research_bundle(
        proposal.trial, proofs, verifier_id="stage-verifier", verifier_key=STAGE_KEY,
    )
    item = OutcomeEvidence(
        proposal.identity(), receipt, "independent-outcome", OUTCOME_VERSION,
        OUTCOME_ROOT, verdict, verdict == "FALSIFIED", gain, spent, H,
    )
    return replace(item, signature=_mac(OUTCOME_KEY, item.payload()))


def commit(state, proposal, forecast, decision, observation=None, proofs=None):
    proofs = stages(proposal) if proofs is None else proofs
    observation = (
        make_outcome(proposal, proofs) if observation is None else observation
    )
    return record_experiment(
        state, POLICY, (proposal,), (forecast,), decision, observation, proofs,
        **trust(state),
        research_verifier_id="stage-verifier", research_verifier_key=STAGE_KEY,
        outcome_verifier_id="independent-outcome",
        outcome_verifier_version_sha256=OUTCOME_VERSION,
        trusted_outcome_roots=frozenset({OUTCOME_ROOT}), outcome_key=OUTCOME_KEY,
    )


def test_verified_evidence_updates_replayable_chained_research_map():
    initial = ResearchMap()
    p, f, decision = choose(initial)
    assert initial.head_sha256 == GENESIS
    assert decision.state == "RUN_FIXTURE_CANDIDATE"
    assert not decision.training_authorized and not decision.paid_compute_authorized
    assert not decision.promotion_authorized
    assert decision == choose(initial)[2]
    first = commit(initial, p, f, decision)
    assert first != initial
    assert first.validate(POLICY) == (2, 0)
    assert first == commit(initial, p, f, decision)
    nxt, nxtforecast, nextdecision = choose(first, name="trial-02")
    second = commit(first, nxt, nxtforecast, nextdecision)
    assert second.validate(POLICY) == (4, 0)
    assert second.head_sha256 != first.head_sha256
    assert [r.experiment_id for r in second.records] == ["trial-01", "trial-02"]


def test_falsifier_and_independently_verified_negative_research_outcome():
    state = ResearchMap()
    p, f, decision = choose(state)
    proofs = stages(p)
    observed = make_outcome(p, proofs, verdict="FALSIFIED", gain=0)
    result = commit(state, p, f, decision, observed, proofs)
    assert result.records[0].verdict == "FALSIFIED"
    assert result.validate(POLICY) == (2, 1)


def test_stop_after_repeat_low_gain_without_experiment_churn():
    state = ResearchMap()
    for index in range(2):
        p, f, decision = choose(state, name=f"trial-{index}")
        proofs = stages(p)
        state = commit(state, p, f, decision, make_outcome(p, proofs, gain=0), proofs)
    assert state.validate(POLICY) == (4, 2)
    p, f, _ = choose(state, name="trial-next")
    stop = select_experiment(state, POLICY, (p,), (f,), **trust(state))
    assert stop.state == "STOP_LOW_INFORMATION_GAIN"
    assert stop.experiment_id is None
    assert stop == select_experiment(state, POLICY, (p,), (f,), **trust(state))


def test_no_expected_information_gain_and_budget_cap_stop():
    state = ResearchMap()
    p = make_proposal(state)
    f = make_forecast(p, gain=9)
    assert select_experiment(state, POLICY, (p,), (f,), **trust(state)).state == (
        "STOP_NO_INFORMATION_GAIN"
    )
    over = make_proposal(state, "expensive", 7)
    assert select_experiment(
        state, POLICY, (over,), (make_forecast(over),), **trust(state)
    ).state == "STOP_NO_INFORMATION_GAIN"
    assert select_experiment(state, POLICY, (), (), **trust(state)).experiment_id is None


def test_forecast_signatures_self_assessment_and_wrong_context_denied():
    state = ResearchMap()
    p, f, _ = choose(state)
    for bad in (
        replace(f, signature=H),
        replace(f, expected_gain=True),
        replace(f, verifier_version_sha256=H),
        replace(f, evidence_sha256=H),
        replace(f, verifier_id=p.trial.producer_id),
    ):
        with pytest.raises(ValueError):
            select_experiment(state, POLICY, (p,), (bad,), **trust(state))
    with pytest.raises(ValueError):
        select_experiment(state, POLICY, (p,), (f,), trusted_map_sha256=H,
                          **{k: v for k, v in trust(state).items()
                             if k != "trusted_map_sha256"})
    with pytest.raises(ValueError):
        select_experiment(state, POLICY, (p, p), (f, f), **trust(state))


def test_unbound_protocol_falsifier_and_explicit_budgets_denied():
    state = ResearchMap()
    p = make_proposal(state)
    with pytest.raises(ValueError):
        replace(p, expected_falsifier="changed").identity()
    with pytest.raises(ValueError):
        replace(p, parent_map_sha256=H).identity()
    with pytest.raises(ValueError):
        replace(p, hypothesis="x" * 513).identity()
    with pytest.raises(ValueError):
        replace(p, max_cost_units=0).identity()
    with pytest.raises(ValueError):
        ResearchPolicy(budget_units=True).validate()


def test_stale_or_forged_decision_cannot_enter_research_map():
    state = ResearchMap()
    p, f, decision = choose(state)
    for bad in (
        replace(decision, promotion_authorized=True),
        replace(decision, experiment_id="wrong"),
        replace(decision, receipt_sha256=H),
        replace(decision, state="STOP_NO_INFORMATION_GAIN"),
    ):
        with pytest.raises(ValueError):
            commit(state, p, f, bad)
    next_state = commit(state, p, f, decision)
    with pytest.raises(ValueError):
        commit(next_state, p, f, decision)
    with pytest.raises(ValueError):
        replace(next_state, head_sha256=H).validate(POLICY)
    with pytest.raises(ValueError):
        ResearchMap(next_state.head_sha256, next_state.records * 2).validate(POLICY)


def test_failed_stages_and_forged_outcome_are_not_research_evidence():
    state = ResearchMap()
    p, f, decision = choose(state)
    proofs = stages(p)
    valid = make_outcome(p, proofs)
    for bad in (
        replace(valid, signature=H),
        replace(valid, experiment_sha256=H),
        replace(valid, research_receipt_sha256=H),
        replace(valid, evidence_sha256=H),
        replace(valid, actual_gain=True),
        replace(valid, spent_units=3),
        replace(valid, verdict="FALSIFIED", falsifier_triggered=False),
    ):
        with pytest.raises(ValueError):
            commit(state, p, f, decision, bad, proofs)
    with pytest.raises(ValueError):
        commit(state, p, f, decision, valid, proofs[:-1])
    with pytest.raises(ValueError):
        commit(state, p, f, decision, valid, proofs[::-1])


def test_deterministic_ranking_cost_gain_and_ties():
    state = ResearchMap()
    a = make_proposal(state, name="a", cost=1)
    b = make_proposal(state, name="b", cost=2)
    chosen = select_experiment(
        state, POLICY, (b, a), (make_forecast(b, 95), make_forecast(a, 75)),
        **trust(state),
    )
    assert chosen.experiment_id == "a"
    identical = make_proposal(state, name="c", cost=1)
    tie = select_experiment(
        state, POLICY, (identical, a), (make_forecast(identical, 75),
                                      make_forecast(a, 75)), **trust(state),
    )
    assert tie.experiment_id == "a"


def test_round_limit_stops_even_with_positive_forecast():
    state = ResearchMap()
    p, f, decision = choose(state)
    full = commit(state, p, f, decision)
    limit = replace(POLICY, max_rounds=1)
    proposal = make_proposal(full, name="second")
    forecast = make_forecast(proposal)
    result = select_experiment(
        full, limit, (proposal,), (forecast,), **trust(full),
    )
    assert result.state == "STOP_ROUND_LIMIT"
    assert result.experiment_id is None


def test_spent_budget_stop_and_overspend_fail_closed():
    state = ResearchMap()
    p, f, decision = choose(state)
    full = commit(state, p, f, decision)
    bounded = replace(POLICY, budget_units=2)
    proposal = make_proposal(full, name="second", cost=1)
    forecast = make_forecast(proposal)
    result = select_experiment(
        full, bounded, (proposal,), (forecast,), **trust(full),
    )
    assert result.state == "STOP_BUDGET"
    assert result.experiment_id is None
