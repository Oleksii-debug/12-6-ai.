"""Plan 6 Section 13: bounded, evidence-led autonomous research decisions.

The planner never executes training, invokes a teacher, spends money, or promotes
a model. A trusted harness supplies signed forecasts and independently verified
ResearchTrial observations; this module selects, records and stops.
"""
from __future__ import annotations

import hmac
import json
from dataclasses import asdict, dataclass

from .experience_replay import _mac
from .post_base_instruction import bounded_id, canonical_digest, sha_field
from .research_engine import ResearchTrial, StageEvidence, verify_research_bundle

GENESIS = canonical_digest({"research_map": "plan6-v1"})
_VERDICTS = frozenset({"SUPPORTED", "FALSIFIED", "INCONCLUSIVE"})


def _text(value: object) -> None:
    if (
        type(value) is not str or not 1 <= len(value) <= 512
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError("experiment hypothesis/falsifier must be bounded text")


@dataclass(frozen=True, slots=True)
class ResearchPolicy:
    budget_units: int = 100
    max_rounds: int = 8
    min_expected_gain: int = 10
    min_actual_gain: int = 2
    max_low_gain_rounds: int = 2
    schema_version: int = 1

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported research-loop policy")
        for value, lower, upper in (
            (self.budget_units, 1, 10000), (self.max_rounds, 1, 32),
            (self.min_expected_gain, 1, 100), (self.min_actual_gain, 1, 100),
            (self.max_low_gain_rounds, 1, 8),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("unbounded research-loop policy")


@dataclass(frozen=True, slots=True)
class ExperimentProposal:
    experiment_id: str
    hypothesis: str
    expected_falsifier: str
    trial: ResearchTrial
    parent_map_sha256: str
    max_cost_units: int
    schema_version: int = 1

    def identity(self) -> str:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported experiment protocol")
        bounded_id(self.experiment_id)
        _text(self.hypothesis)
        _text(self.expected_falsifier)
        sha_field(self.parent_map_sha256)
        if (
            type(self.max_cost_units) is not int
            or not 1 <= self.max_cost_units <= 10000
            or type(self.trial) is not ResearchTrial
        ):
            raise ValueError("invalid bounded trial")
        self.trial.__post_init__()
        expected = {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "expected_falsifier": self.expected_falsifier,
            "parent_map_sha256": self.parent_map_sha256,
            "max_cost_units": self.max_cost_units,
        }
        if json.loads(self.trial.config_json) != expected:
            raise ValueError("trial protocol not bound to hypothesis/falsifier/map")
        return canonical_digest({
            "experiment_id": self.experiment_id, "trial": self.trial.identity_sha256,
            "parent": self.parent_map_sha256, "max_cost": self.max_cost_units,
        })


@dataclass(frozen=True, slots=True)
class ForecastEvidence:
    experiment_sha256: str
    verifier_id: str
    verifier_version_sha256: str
    evidence_sha256: str
    expected_gain: int
    signature: str

    def payload(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "signature"}


@dataclass(frozen=True, slots=True)
class ResearchRecord:
    experiment_id: str
    experiment_sha256: str
    research_receipt_sha256: str
    outcome_receipt_sha256: str
    verdict: str
    actual_gain: int
    spent_units: int
    parent_sha256: str
    record_sha256: str

    def digest(self) -> str:
        return canonical_digest({
            key: value for key, value in asdict(self).items() if key != "record_sha256"
        })


@dataclass(frozen=True, slots=True)
class ResearchMap:
    head_sha256: str = GENESIS
    records: tuple[ResearchRecord, ...] = ()

    def validate(self, policy: ResearchPolicy) -> tuple[int, int]:
        policy.validate()
        if type(self.records) is not tuple or len(self.records) > policy.max_rounds:
            raise ValueError("unbounded research history")
        head = GENESIS
        total = 0
        low_gain_streak = 0
        seen: set[str] = set()
        for record in self.records:
            if (
                type(record) is not ResearchRecord
                or record.experiment_id in seen or record.parent_sha256 != head
                or record.verdict not in _VERDICTS
                or type(record.actual_gain) is not int
                or not 0 <= record.actual_gain <= 100
                or type(record.spent_units) is not int
                or not 1 <= record.spent_units <= policy.budget_units
            ):
                raise ValueError("forged or duplicate research-map record")
            bounded_id(record.experiment_id)
            for root in (record.experiment_sha256, record.research_receipt_sha256,
                         record.outcome_receipt_sha256, record.record_sha256):
                sha_field(root)
            if record.digest() != record.record_sha256:
                raise ValueError("research-map hash chain changed")
            seen.add(record.experiment_id)
            head = record.record_sha256
            total += record.spent_units
            if total > policy.budget_units:
                raise ValueError("research-map budget overspent")
            low_gain_streak = (
                low_gain_streak + 1 if record.actual_gain < policy.min_actual_gain else 0
            )
        if self.head_sha256 != head:
            raise ValueError("stale research-map head")
        return total, low_gain_streak


@dataclass(frozen=True, slots=True)
class ResearchChoice:
    map_sha256: str
    experiment_id: str | None
    experiment_sha256: str | None
    state: str
    receipt_sha256: str
    training_authorized: bool = False
    paid_compute_authorized: bool = False
    promotion_authorized: bool = False


def _choice(state: ResearchMap, proposal: ExperimentProposal | None, reason: str) -> ResearchChoice:
    experiment_id = None if proposal is None else proposal.experiment_id
    experiment_sha = None if proposal is None else proposal.identity()
    receipt = canonical_digest({
        "map": state.head_sha256, "experiment_id": experiment_id,
        "experiment": experiment_sha, "state": reason,
    })
    return ResearchChoice(state.head_sha256, experiment_id, experiment_sha, reason, receipt)


def select_experiment(
    state: ResearchMap, policy: ResearchPolicy,
    proposals: tuple[ExperimentProposal, ...], forecasts: tuple[ForecastEvidence, ...],
    *, trusted_map_sha256: str, trusted_forecast_id: str,
    trusted_forecast_version_sha256: str, trusted_forecast_roots: frozenset[str],
    forecast_key: bytes,
) -> ResearchChoice:
    spent, low_streak = state.validate(policy)
    if state.head_sha256 != trusted_map_sha256:
        raise ValueError("research-map root not externally pinned")
    if type(proposals) is not tuple or type(forecasts) is not tuple:
        raise ValueError("untyped proposal/forecast batch")
    if len(proposals) != len(forecasts) or len(proposals) > 64:
        raise ValueError("unbounded or missing forecast batch")
    bounded_id(trusted_forecast_id)
    sha_field(trusted_forecast_version_sha256)
    if type(trusted_forecast_roots) is not frozenset or not trusted_forecast_roots:
        raise ValueError("missing independent forecasting roots")
    for root in trusted_forecast_roots:
        sha_field(root)
    seen: set[str] = set()
    ranked: list[tuple[float, str, ExperimentProposal]] = []
    used = {record.experiment_id for record in state.records}
    for proposal, forecast in zip(proposals, forecasts, strict=True):
        if type(proposal) is not ExperimentProposal or type(forecast) is not ForecastEvidence:
            raise ValueError("untyped experiment evidence")
        candidate_sha = proposal.identity()
        if (
            proposal.parent_map_sha256 != state.head_sha256
            or proposal.experiment_id in seen or proposal.experiment_id in used
        ):
            raise ValueError("stale/duplicate research experiment")
        seen.add(proposal.experiment_id)
        if (
            forecast.experiment_sha256 != candidate_sha
            or forecast.verifier_id != trusted_forecast_id
            or forecast.verifier_id == proposal.trial.producer_id
            or forecast.verifier_version_sha256 != trusted_forecast_version_sha256
            or forecast.evidence_sha256 not in trusted_forecast_roots
            or type(forecast.expected_gain) is not int
            or not 0 <= forecast.expected_gain <= 100
        ):
            raise ValueError("untrusted or self-issued information-gain forecast")
        sha_field(forecast.signature)
        if not hmac.compare_digest(
            forecast.signature, _mac(forecast_key, forecast.payload())
        ):
            raise ValueError("forged information-gain forecast")
        if (
            forecast.expected_gain >= policy.min_expected_gain
            and proposal.max_cost_units <= policy.budget_units - spent
        ):
            ranked.append((
                -forecast.expected_gain / proposal.max_cost_units,
                proposal.experiment_id, proposal,
            ))
    if len(state.records) >= policy.max_rounds:
        return _choice(state, None, "STOP_ROUND_LIMIT")
    if low_streak >= policy.max_low_gain_rounds:
        return _choice(state, None, "STOP_LOW_INFORMATION_GAIN")
    if spent >= policy.budget_units:
        return _choice(state, None, "STOP_BUDGET")
    if not ranked:
        return _choice(state, None, "STOP_NO_INFORMATION_GAIN")
    winner = sorted(ranked)[0][2]
    return _choice(state, winner, "RUN_FIXTURE_CANDIDATE")


@dataclass(frozen=True, slots=True)
class OutcomeEvidence:
    experiment_sha256: str
    research_receipt_sha256: str
    verifier_id: str
    verifier_version_sha256: str
    evidence_sha256: str
    verdict: str
    falsifier_triggered: bool
    actual_gain: int
    spent_units: int
    signature: str

    def payload(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if key != "signature"}


def record_experiment(
    state: ResearchMap, policy: ResearchPolicy,
    proposals: tuple[ExperimentProposal, ...], forecasts: tuple[ForecastEvidence, ...],
    choice: ResearchChoice, observation: OutcomeEvidence,
    stages: tuple[StageEvidence, ...], *, trusted_map_sha256: str,
    trusted_forecast_id: str, trusted_forecast_version_sha256: str,
    trusted_forecast_roots: frozenset[str], forecast_key: bytes,
    research_verifier_id: str, research_verifier_key: bytes,
    outcome_verifier_id: str, outcome_verifier_version_sha256: str,
    trusted_outcome_roots: frozenset[str], outcome_key: bytes,
) -> ResearchMap:
    independently_selected = select_experiment(
        state, policy, proposals, forecasts, trusted_map_sha256=trusted_map_sha256,
        trusted_forecast_id=trusted_forecast_id,
        trusted_forecast_version_sha256=trusted_forecast_version_sha256,
        trusted_forecast_roots=trusted_forecast_roots, forecast_key=forecast_key,
    )
    if (
        choice != independently_selected or choice.state != "RUN_FIXTURE_CANDIDATE"
        or choice.training_authorized or choice.paid_compute_authorized
        or choice.promotion_authorized
    ):
        raise ValueError("unselected or self-authorized research experiment")
    proposal = next(p for p in proposals if p.experiment_id == choice.experiment_id)
    verified = verify_research_bundle(
        proposal.trial, stages, verifier_id=research_verifier_id,
        verifier_key=research_verifier_key,
    )
    spent, _ = state.validate(policy)
    bounded_id(outcome_verifier_id)
    sha_field(outcome_verifier_version_sha256)
    if (
        type(observation) is not OutcomeEvidence
        or outcome_verifier_id in (
            proposal.trial.producer_id, trusted_forecast_id, research_verifier_id
        )
        or observation.verifier_id != outcome_verifier_id
        or observation.verifier_version_sha256 != outcome_verifier_version_sha256
        or observation.experiment_sha256 != proposal.identity()
        or observation.research_receipt_sha256 != verified
        or type(trusted_outcome_roots) is not frozenset
        or observation.evidence_sha256 not in trusted_outcome_roots
        or observation.verdict not in _VERDICTS
        or type(observation.falsifier_triggered) is not bool
        or (observation.verdict == "FALSIFIED") != observation.falsifier_triggered
        or type(observation.actual_gain) is not int
        or not 0 <= observation.actual_gain <= 100
        or type(observation.spent_units) is not int
        or not 1 <= observation.spent_units <= proposal.max_cost_units
        or spent + observation.spent_units > policy.budget_units
    ):
        raise ValueError("untrusted/falsified outcome or budget violation")
    for root in trusted_outcome_roots:
        sha_field(root)
    sha_field(observation.signature)
    if not hmac.compare_digest(observation.signature, _mac(outcome_key, observation.payload())):
        raise ValueError("forged research outcome")
    record = ResearchRecord(
        proposal.experiment_id, proposal.identity(), verified,
        canonical_digest(asdict(observation)), observation.verdict,
        observation.actual_gain, observation.spent_units, state.head_sha256, "",
    )
    record = ResearchRecord(
        record.experiment_id, record.experiment_sha256, record.research_receipt_sha256,
        record.outcome_receipt_sha256, record.verdict, record.actual_gain,
        record.spent_units, record.parent_sha256, record.digest(),
    )
    result = ResearchMap(record.record_sha256, state.records + (record,))
    result.validate(policy)
    return result
