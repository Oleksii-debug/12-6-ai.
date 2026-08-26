from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .postbase_deliberation import (
    AdapterContractError,
    Budget,
    Config,
    DeliberationController,
    _Stop,
)
from .postbase_hypothesis import HypothesisSearch


@dataclass(frozen=True)
class EvidenceCheck:
    name: str
    prediction: Any
    observed: Any
    weight: float = 0.25
    hard: bool = False
    source: str = "deterministic_local"


class EvidenceAdapter(Protocol):
    def checks(
        self,
        task: str,
        hypothesis_id: str,
        statement: str,
        iteration: int,
    ) -> tuple[EvidenceCheck, ...]:
        ...


@dataclass(frozen=True)
class HypothesisIntegrationConfig:
    max_evidence_checks_per_hypothesis: int = 8
    revise_first_rejected: bool = True

    def __post_init__(self) -> None:
        if self.max_evidence_checks_per_hypothesis <= 0:
            raise ValueError("max_evidence_checks_per_hypothesis must be positive")


class HypothesisDeliberationController(DeliberationController):
    """POSTBASE-255 controller integrated with POSTBASE-256 hypothesis state."""

    worker_id = "NEXT100-081-HYPOTHESIS-DELIBERATION-INTEGRATION"
    schema = "12-6.hypothesis-deliberation-integration.v1"

    def __init__(
        self,
        model,
        verifier,
        evidence: EvidenceAdapter,
        *,
        tools=None,
        config: Config | None = None,
        integration_config: HypothesisIntegrationConfig | None = None,
    ) -> None:
        super().__init__(model, verifier, tools=tools, config=config)
        if self.config.initial_branches < 2:
            raise ValueError("hypothesis integration requires at least two initial branches")
        self.evidence_adapter = evidence
        self.integration_config = integration_config or HypothesisIntegrationConfig()

    def run(self, task: str, budget: Budget) -> dict[str, Any]:
        if not task.strip():
            raise ValueError("task must be non-empty")
        started = time.monotonic()
        deadline = None if budget.wall_seconds is None else started + budget.wall_seconds
        used = {"model_calls": 0, "generated_tokens": 0, "tool_calls": 0, "candidate_branches": 0}
        local = {
            "critiques": 0,
            "evidence_tests": 0,
            "contradictions": 0,
            "rejections": 0,
            "revisions": 0,
        }
        trace: dict[str, Any] = {
            "schema": self.schema,
            "worker_id": self.worker_id,
            "consumed_authorities": [
                "POSTBASE-255-DELIBERATION-CONTROLLER-V1",
                "POSTBASE-256-HYPOTHESIS-SEARCH-V1",
            ],
            "budget": asdict(budget),
            "config": {
                "deliberation": asdict(self.config),
                "hypothesis_integration": asdict(self.integration_config),
            },
            "model_calls": [],
            "tool_calls": [],
            "branches_attempted": [],
            "comparisons": [],
            "hypothesis_events": [],
            "retained_best_history": [],
        }
        search = HypothesisSearch()
        candidates: dict[str, Any] = {}
        hypothesis_to_candidate: dict[str, str] = {}
        next_candidate = 1
        initial_hypotheses: list[str] = []
        revised_once = False
        reason = ""

        def reserve() -> str:
            nonlocal next_candidate
            self._branch_ok(budget, used)
            cid = f"candidate-{next_candidate}"
            next_candidate += 1
            used["candidate_branches"] += 1
            return cid

        def bind_proposal(response, cid: str, branch: str, iteration: int = 0):
            candidate = self._candidate(
                task, response, cid, branch, None, iteration, "propose", trace
            )
            candidates[cid] = candidate
            hypothesis = search.propose(response.text, initial_score=candidate.verification.score)
            hypothesis_to_candidate[hypothesis.id] = cid
            trace["hypothesis_events"].append(
                {
                    "event": "proposed",
                    "hypothesis_id": hypothesis.id,
                    "candidate_id": cid,
                    "branch_id": branch,
                }
            )
            return hypothesis

        def bind_revision(response, cid: str, branch: str, parent_cid: str, parent_hid: str, iteration: int):
            candidate = self._candidate(
                task, response, cid, branch, parent_cid, iteration, "revise", trace
            )
            candidates[cid] = candidate
            hypothesis = search.revise(
                parent_hid,
                response.text,
                initial_score=candidate.verification.score,
            )
            hypothesis_to_candidate[hypothesis.id] = cid
            local["revisions"] += 1
            trace["hypothesis_events"].append(
                {
                    "event": "revised",
                    "hypothesis_id": hypothesis.id,
                    "parent_hypothesis_id": parent_hid,
                    "candidate_id": cid,
                    "parent_candidate_id": parent_cid,
                }
            )
            return hypothesis

        def record_best(phase: str) -> None:
            best = search.best()
            if best is None:
                return
            selected_cid = hypothesis_to_candidate[best.id]
            active_ids = [
                item["id"]
                for item in self._public_hypothesis_state(search)["hypotheses"]
                if item["status"] == "active"
            ]
            trace["comparisons"].append(
                {
                    "round": len(trace["comparisons"]),
                    "phase": phase,
                    "selected_hypothesis_id": best.id,
                    "selected_candidate_id": selected_cid,
                    "selected_score": best.score,
                    "other_active_hypothesis_ids": [hid for hid in active_ids if hid != best.id],
                }
            )
            trace["retained_best_history"].append(
                {
                    "phase": phase,
                    "hypothesis_id": best.id,
                    "candidate_id": selected_cid,
                    "score": best.score,
                }
            )

        def apply_evidence(hid: str, iteration: int) -> tuple[str, ...]:
            hypothesis = search.hypothesis(hid)
            checks = tuple(
                self.evidence_adapter.checks(task, hid, hypothesis.statement, iteration)
            )
            if len(checks) > self.integration_config.max_evidence_checks_per_hypothesis:
                raise AdapterContractError("evidence adapter exceeded per-hypothesis check allowance")
            hard_failures: list[str] = []
            for check in checks:
                if not isinstance(check, EvidenceCheck):
                    raise AdapterContractError("evidence adapter must return EvidenceCheck values")
                record = search.test(
                    hid,
                    name=check.name,
                    prediction=check.prediction,
                    observed=check.observed,
                    weight=check.weight,
                    hard=check.hard,
                    source=check.source,
                )
                local["evidence_tests"] += 1
                if not record.passed:
                    local["contradictions"] += 1
                    if check.hard:
                        hard_failures.append(record.evidence_id)
                trace["hypothesis_events"].append(
                    {
                        "event": "evidence",
                        "hypothesis_id": hid,
                        "evidence_id": record.evidence_id,
                        "passed": record.passed,
                        "hard": check.hard,
                    }
                )
            return tuple(hard_failures)

        try:
            for n in range(self.config.initial_branches):
                self._model_ok(budget, used, deadline)
                cid = reserve()
                branch = f"branch-{n + 1}"
                response = self._stage(
                    task, "propose", branch, cid, 0,
                    None, None, budget, used, trace, deadline,
                )
                initial_hypotheses.append(bind_proposal(response, cid, branch).id)

            if len(initial_hypotheses) < 2:
                raise RuntimeError("budget exhausted before multiple hypotheses completed")

            record_best("initial")
            queue = list(initial_hypotheses)
            queue.sort(key=lambda hid: (search.hypothesis(hid).score, hid), reverse=True)
            iteration = 1
            index = 0
            while index < len(queue):
                hid = queue[index]
                index += 1
                hypothesis = search.hypothesis(hid)
                if hypothesis.status != "active":
                    continue
                cid = hypothesis_to_candidate[hid]
                candidate = candidates[cid]
                self._model_ok(budget, used, deadline)
                critique = self._stage(
                    task, "critique", candidate.branch, cid, iteration,
                    candidate.text, None, budget, used, trace, deadline,
                )
                search.critique(hid, critique.text)
                local["critiques"] += 1
                trace["hypothesis_events"].append(
                    {
                        "event": "critique",
                        "hypothesis_id": hid,
                        "critique_count": local["critiques"],
                    }
                )

                hard_failures = apply_evidence(hid, iteration)
                if hard_failures:
                    search.reject(
                        hid,
                        "hard deterministic evidence contradicted the hypothesis",
                        evidence_ids=hard_failures,
                    )
                    local["rejections"] += 1
                    trace["hypothesis_events"].append(
                        {
                            "event": "rejected",
                            "hypothesis_id": hid,
                            "evidence_ids": list(hard_failures),
                        }
                    )
                    if self.integration_config.revise_first_rejected and not revised_once:
                        self._outer_ok(budget, used, deadline)
                        revision_cid = reserve()
                        revision = self._stage(
                            task,
                            "revise",
                            candidate.branch,
                            revision_cid,
                            iteration,
                            candidate.text,
                            critique.text,
                            budget,
                            used,
                            trace,
                            deadline,
                        )
                        revised = bind_revision(
                            revision,
                            revision_cid,
                            candidate.branch,
                            cid,
                            hid,
                            iteration,
                        )
                        queue.append(revised.id)
                        revised_once = True
                record_best(f"evidence-{hid}")
                iteration += 1
            reason = "evidence_complete"
        except _Stop as stop:
            if len(initial_hypotheses) < 2:
                raise RuntimeError(f"{stop.reason} before multiple hypotheses completed") from None
            reason = stop.reason

        best = search.best()
        if best is None:
            raise RuntimeError("no active hypothesis retained")
        selected_cid = hypothesis_to_candidate[best.id]
        selected_candidate = candidates[selected_cid]
        trace["budget_consumed"] = {
            **used,
            **local,
            "wall_seconds": time.monotonic() - started,
        }
        trace["stop_reason"] = reason
        trace["selected_final_candidate"] = selected_cid
        trace["selected_hypothesis_id"] = best.id
        trace["hypothesis_state"] = self._public_hypothesis_state(search)
        return {
            "final_text": selected_candidate.text,
            "score": best.score,
            "confidence": selected_candidate.verification.confidence,
            "hypothesis_id": best.id,
            "trace": trace,
        }

    @staticmethod
    def _public_hypothesis_state(search: HypothesisSearch) -> dict[str, Any]:
        exported = search.export()
        return {
            "schema": exported["schema"],
            "worker_id": exported["worker_id"],
            "selected_hypothesis_id": exported["selected_hypothesis_id"],
            "hypotheses": exported["hypotheses"],
            "evidence": exported["evidence"],
            "contradictions": exported["contradictions"],
            "critiques": [
                {
                    "id": item["id"],
                    "hypothesis_id": item["hypothesis_id"],
                    "score_delta": item["score_delta"],
                }
                for item in exported["critiques"]
            ],
            "tests": [
                {
                    "id": item["id"],
                    "hypothesis_id": item["hypothesis_id"],
                    "name": item["name"],
                    "passed": item["passed"],
                    "evidence_id": item["evidence_id"],
                }
                for item in exported["tests"]
            ],
        }
