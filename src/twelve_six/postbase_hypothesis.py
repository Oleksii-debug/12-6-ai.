from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

HypothesisStatus = Literal["active", "rejected", "revised"]
EvidenceKind = Literal["support", "contradiction"]


@dataclass(frozen=True)
class ScorePoint:
    sequence: int
    score: float
    reason: str
    evidence_id: str | None = None


@dataclass(frozen=True)
class Evidence:
    id: str
    hypothesis_id: str
    kind: EvidenceKind
    description: str
    weight: float
    hard: bool
    source: str


@dataclass(frozen=True)
class Contradiction:
    id: str
    hypothesis_id: str
    evidence_id: str
    description: str


@dataclass(frozen=True)
class Critique:
    id: str
    hypothesis_id: str
    text: str
    score_delta: float


@dataclass(frozen=True)
class TestRecord:
    id: str
    hypothesis_id: str
    name: str
    prediction: Any
    observed: Any
    passed: bool
    evidence_id: str


@dataclass(frozen=True)
class Hypothesis:
    id: str
    statement: str
    assumptions: tuple[str, ...]
    parent_id: str | None
    status: HypothesisStatus
    score_history: tuple[ScorePoint, ...]
    evidence_ids: tuple[str, ...] = ()
    contradiction_ids: tuple[str, ...] = ()
    critique_ids: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        return self.score_history[-1].score


class HypothesisSearch:
    """Deterministic, model-agnostic hypothesis graph for post-Base reasoning."""

    worker_id = "POSTBASE-256-HYPOTHESIS-SEARCH-V1"
    schema = "12-6.postbase-hypothesis-search.v1"

    def __init__(self) -> None:
        self._hypotheses: dict[str, Hypothesis] = {}
        self._evidence: dict[str, Evidence] = {}
        self._contradictions: dict[str, Contradiction] = {}
        self._critiques: dict[str, Critique] = {}
        self._tests: dict[str, TestRecord] = {}
        self._counters = {
            "hypothesis": 0,
            "evidence": 0,
            "contradiction": 0,
            "critique": 0,
            "test": 0,
            "score": 0,
        }

    def propose(
        self,
        statement: str,
        *,
        assumptions: tuple[str, ...] = (),
        initial_score: float = 0.5,
    ) -> Hypothesis:
        statement = self._text(statement, "statement")
        assumptions = self._assumptions(assumptions)
        score = self._score(initial_score)
        hypothesis_id = self._next("hypothesis", "H")
        point = self._score_point(score, "proposed")
        hypothesis = Hypothesis(
            id=hypothesis_id,
            statement=statement,
            assumptions=assumptions,
            parent_id=None,
            status="active",
            score_history=(point,),
        )
        self._hypotheses[hypothesis_id] = hypothesis
        return hypothesis

    def branch(
        self,
        parent_id: str,
        statement: str,
        *,
        assumptions: tuple[str, ...] | None = None,
        initial_score: float | None = None,
    ) -> Hypothesis:
        parent = self._open(parent_id)
        statement = self._text(statement, "statement")
        inherited = parent.assumptions if assumptions is None else self._assumptions(assumptions)
        score = parent.score if initial_score is None else self._score(initial_score)
        hypothesis_id = self._next("hypothesis", "H")
        point = self._score_point(score, f"branched from {parent.id}")
        hypothesis = Hypothesis(
            id=hypothesis_id,
            statement=statement,
            assumptions=inherited,
            parent_id=parent.id,
            status="active",
            score_history=(point,),
        )
        self._hypotheses[hypothesis_id] = hypothesis
        return hypothesis

    def critique(
        self,
        hypothesis_id: str,
        text: str,
        *,
        score_delta: float = 0.0,
    ) -> Critique:
        hypothesis = self._open(hypothesis_id)
        text = self._text(text, "critique")
        if not math.isfinite(score_delta) or not -1 <= score_delta <= 1:
            raise ValueError("score_delta must be finite and in [-1,1]")
        critique_id = self._next("critique", "Q")
        critique = Critique(critique_id, hypothesis.id, text, score_delta)
        self._critiques[critique_id] = critique
        hypothesis = replace(
            hypothesis,
            critique_ids=(*hypothesis.critique_ids, critique_id),
        )
        if score_delta:
            hypothesis = self._with_score(
                hypothesis,
                hypothesis.score + score_delta,
                f"critique {critique_id}",
            )
        self._hypotheses[hypothesis.id] = hypothesis
        return critique

    def test(
        self,
        hypothesis_id: str,
        *,
        name: str,
        prediction: Any,
        observed: Any,
        weight: float = 0.25,
        hard: bool = False,
        source: str = "deterministic_fixture",
    ) -> TestRecord:
        hypothesis = self._open(hypothesis_id)
        name = self._text(name, "test name")
        source = self._text(source, "source")
        weight = self._weight(weight)
        passed = prediction == observed
        kind: EvidenceKind = "support" if passed else "contradiction"
        description = (
            f"{name}: prediction matched observation"
            if passed
            else f"{name}: prediction contradicted by observation"
        )
        evidence_id = self._next("evidence", "E")
        evidence = Evidence(
            id=evidence_id,
            hypothesis_id=hypothesis.id,
            kind=kind,
            description=description,
            weight=weight,
            hard=hard,
            source=source,
        )
        self._evidence[evidence_id] = evidence

        contradiction_ids = hypothesis.contradiction_ids
        if not passed:
            contradiction_id = self._next("contradiction", "C")
            contradiction = Contradiction(
                id=contradiction_id,
                hypothesis_id=hypothesis.id,
                evidence_id=evidence_id,
                description=description,
            )
            self._contradictions[contradiction_id] = contradiction
            contradiction_ids = (*contradiction_ids, contradiction_id)

        test_id = self._next("test", "T")
        record = TestRecord(
            id=test_id,
            hypothesis_id=hypothesis.id,
            name=name,
            prediction=prediction,
            observed=observed,
            passed=passed,
            evidence_id=evidence_id,
        )
        self._tests[test_id] = record
        delta = weight if passed else -weight
        hypothesis = replace(
            hypothesis,
            evidence_ids=(*hypothesis.evidence_ids, evidence_id),
            contradiction_ids=contradiction_ids,
            test_ids=(*hypothesis.test_ids, test_id),
        )
        hypothesis = self._with_score(
            hypothesis,
            hypothesis.score + delta,
            f"test {test_id} {'passed' if passed else 'failed'}",
            evidence_id=evidence_id,
        )
        self._hypotheses[hypothesis.id] = hypothesis
        return record

    def reject(
        self,
        hypothesis_id: str,
        reason: str,
        *,
        evidence_ids: tuple[str, ...] = (),
    ) -> Hypothesis:
        hypothesis = self._open(hypothesis_id)
        reason = self._text(reason, "rejection reason")
        for evidence_id in evidence_ids:
            evidence = self._evidence.get(evidence_id)
            if evidence is None or evidence.hypothesis_id != hypothesis.id:
                raise ValueError(f"evidence {evidence_id!r} does not belong to {hypothesis.id}")
        hypothesis = self._with_score(
            hypothesis,
            0.0,
            f"rejected: {reason}",
            evidence_id=evidence_ids[-1] if evidence_ids else None,
        )
        hypothesis = replace(hypothesis, status="rejected")
        self._hypotheses[hypothesis.id] = hypothesis
        return hypothesis

    def revise(
        self,
        hypothesis_id: str,
        statement: str,
        *,
        assumptions: tuple[str, ...] | None = None,
        initial_score: float | None = None,
    ) -> Hypothesis:
        parent = self._get(hypothesis_id)
        if parent.status == "revised":
            raise ValueError("cannot revise an already revised hypothesis")
        statement = self._text(statement, "statement")
        inherited = parent.assumptions if assumptions is None else self._assumptions(assumptions)
        score = parent.score if initial_score is None else self._score(initial_score)
        hypothesis_id = self._next("hypothesis", "H")
        child = Hypothesis(
            id=hypothesis_id,
            statement=statement,
            assumptions=inherited,
            parent_id=parent.id,
            status="active",
            score_history=(self._score_point(score, f"revision of {parent.id}"),),
        )
        if parent.status == "active":
            self._hypotheses[parent.id] = replace(parent, status="revised")
        self._hypotheses[child.id] = child
        return child

    def best(self) -> Hypothesis | None:
        active = [hypothesis for hypothesis in self._hypotheses.values() if hypothesis.status == "active"]
        if not active:
            return None
        return max(active, key=lambda item: (item.score, item.id))

    def hypothesis(self, hypothesis_id: str) -> Hypothesis:
        return self._get(hypothesis_id)

    def evidence(self, evidence_id: str) -> Evidence:
        try:
            return self._evidence[evidence_id]
        except KeyError as exc:
            raise KeyError(f"unknown evidence {evidence_id!r}") from exc

    def contradiction(self, contradiction_id: str) -> Contradiction:
        try:
            return self._contradictions[contradiction_id]
        except KeyError as exc:
            raise KeyError(f"unknown contradiction {contradiction_id!r}") from exc

    def export(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "worker_id": self.worker_id,
            "hypotheses": [asdict(self._hypotheses[key]) for key in sorted(self._hypotheses)],
            "evidence": [asdict(self._evidence[key]) for key in sorted(self._evidence)],
            "contradictions": [
                asdict(self._contradictions[key]) for key in sorted(self._contradictions)
            ],
            "critiques": [asdict(self._critiques[key]) for key in sorted(self._critiques)],
            "tests": [asdict(self._tests[key]) for key in sorted(self._tests)],
            "selected_hypothesis_id": self.best().id if self.best() is not None else None,
        }

    def _with_score(
        self,
        hypothesis: Hypothesis,
        score: float,
        reason: str,
        *,
        evidence_id: str | None = None,
    ) -> Hypothesis:
        bounded = min(1.0, max(0.0, score))
        point = self._score_point(bounded, reason, evidence_id=evidence_id)
        return replace(hypothesis, score_history=(*hypothesis.score_history, point))

    def _score_point(
        self,
        score: float,
        reason: str,
        *,
        evidence_id: str | None = None,
    ) -> ScorePoint:
        self._counters["score"] += 1
        return ScorePoint(self._counters["score"], score, reason, evidence_id)

    def _get(self, hypothesis_id: str) -> Hypothesis:
        try:
            return self._hypotheses[hypothesis_id]
        except KeyError as exc:
            raise KeyError(f"unknown hypothesis {hypothesis_id!r}") from exc

    def _open(self, hypothesis_id: str) -> Hypothesis:
        hypothesis = self._get(hypothesis_id)
        if hypothesis.status != "active":
            raise ValueError(f"hypothesis {hypothesis_id} is not active")
        return hypothesis

    def _next(self, counter: str, prefix: str) -> str:
        self._counters[counter] += 1
        return f"{prefix}{self._counters[counter]:03d}"

    @staticmethod
    def _text(value: str, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be non-empty")
        return value.strip()

    @classmethod
    def _assumptions(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(cls._text(value, "assumption") for value in values)

    @staticmethod
    def _score(value: float) -> float:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("score must be finite and in [0,1]")
        return value

    @staticmethod
    def _weight(value: float) -> float:
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("weight must be finite and in (0,1]")
        return value
