from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from twelve_six.memory_rag import (
    LexicalRetriever,
    MemoryDatabase,
    MemoryStoreKind,
    Provenance,
    VerificationState,
)
from twelve_six.postbase.verification import (
    CandidateFact,
    Claim,
    ConsistencyChecker,
    CrossCandidateContradictionChecker,
    ExactAnswerFixture,
    ExactAnswerFixtureVerifier,
    FinalAnswerController,
    NumericCalculatorVerifier,
    NumericCheck,
    ProvenanceRecord,
    SourceProvenanceChecker,
    StructuredFact,
    UnitTestCodeVerifier,
    UnitTestEvidence,
    VerificationRequest,
    VerificationStatus,
    VerifierEnsemble,
)
from twelve_six.postbase_deliberation import (
    Budget,
    Config,
    DeliberationController,
    Request,
    Response,
    ToolCall,
    Verification,
)
from twelve_six.postbase_hypothesis import HypothesisSearch

WORKER_ID = "NEXT100-094-ENDTOEND-RESEARCH-FIXTURE"
SCHEMA = "12-6.next100-094.endtoend-research-fixture.v1"
EXPECTED_ANSWER = 37
NOW = datetime(2026, 8, 26, 18, 20, tzinfo=timezone.utc)


def _parse_fact(content: str) -> tuple[str, int]:
    key, raw = content.split("=", 1)
    return key.strip(), int(raw.strip())


class LedgerTool:
    def execute(self, name: str, arguments: Mapping[str, Any]) -> str:
        if name != "usable_units":
            raise ValueError("unknown deterministic tool")
        expected = {"crate_count", "units_per_crate", "damaged_units"}
        if set(arguments) != expected or any(type(arguments[key]) is not int for key in expected):
            raise ValueError("invalid deterministic tool arguments")
        return str(
            arguments["crate_count"] * arguments["units_per_crate"]
            - arguments["damaged_units"]
        )


class LedgerAdapter:
    def __init__(self, arguments: Mapping[str, int]) -> None:
        self.arguments = dict(arguments)

    def generate(self, request: Request) -> Response:
        if request.stage != "propose":
            raise ValueError("fixture adapter only supports one proposal stage")
        if not request.tool_results:
            return Response(
                text="ledger calculation requested",
                generated_tokens=3,
                private_scratch="PRIVATE_NEXT100_094_SENTINEL",
                tool_calls=(ToolCall("usable_units", self.arguments),),
            )
        return Response(
            text=request.tool_results[-1],
            generated_tokens=1,
            private_scratch="PRIVATE_NEXT100_094_AFTER_TOOL",
        )


class EnsembleRankingAdapter:
    """Explicit fail-closed categorical-to-ranking seam for POSTBASE-255."""

    def __init__(self, expected: int) -> None:
        self.expected = expected
        self.ensemble = VerifierEnsemble((ExactAnswerFixtureVerifier(),))

    def evaluate(self, task: str, text: str, branch_id: str, iteration: int) -> Verification:
        del task, branch_id, iteration
        try:
            actual = int(text)
        except ValueError:
            return Verification(0.0, 1.0, "FAIL: non-integer candidate")
        result = self.ensemble.verify(
            VerificationRequest(
                claims=(Claim("candidate", "candidate equals objective ledger result"),),
                exact_fixtures=(ExactAnswerFixture("candidate", self.expected, actual),),
            )
        )
        if result.status is VerificationStatus.PASS:
            return Verification(1.0, 1.0, "PASS: deterministic exact fixture")
        return Verification(0.0, 1.0, f"{result.status}: fail-closed")


def _sanitize_deliberation_trace(trace: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(trace))
    for item in cleaned.get("model_calls", []):
        item.pop("wall_seconds", None)
    for item in cleaned.get("tool_calls", []):
        item.pop("wall_seconds", None)
    if "budget_consumed" in cleaned:
        cleaned["budget_consumed"].pop("wall_seconds", None)
    return cleaned


def _run_base_adapter(checkpoint: Path | None) -> dict[str, Any]:
    if checkpoint is None:
        return {"executed": False, "reason": "checkpoint not supplied to local mechanics call"}
    from twelve_six.inference.contracts import GenerationConfig
    from twelve_six.postbase import ControllerGenerationRequest, PostBaseModelAdapter

    expected_spec = "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
    adapter = PostBaseModelAdapter.from_checkpoint(
        checkpoint, expected_model_spec_sha256=expected_spec
    )
    response = adapter.generate(
        ControllerGenerationRequest(
            controller="deliberation",
            prompt="NEXT100-094 plumbing-only Base adapter probe",
            config=GenerationConfig(max_new_tokens=1, sample=False),
        )
    )
    base = response.base_evidence
    post = response.post_base_evidence
    return {
        "executed": True,
        "base_namespace": base.evidence_namespace,
        "post_base_namespace": post.evidence_namespace,
        "runtime_policy": post.runtime_policy,
        "model_spec_sha256": base.model_spec_sha256,
        "step": base.step,
        "tokens_seen": base.tokens_seen,
        "generated_token_count": post.generated_token_count,
        "generated_token_ids_sha256": post.generated_token_ids_sha256,
        "answer_authority": False,
    }


def run_fixture(*, checkpoint: Path | None = None) -> dict[str, Any]:
    db = MemoryDatabase()
    provenance = {
        "crate": Provenance("synthetic_ledger", "crate-count", "v1", "synthetic://next100-094/crate"),
        "unit": Provenance("synthetic_ledger", "units-per-crate", "v1", "synthetic://next100-094/unit"),
        "damage_old": Provenance("synthetic_ledger", "damaged-units", "v1", "synthetic://next100-094/damage/v1"),
        "damage_new": Provenance("synthetic_ledger", "damaged-units", "v2", "synthetic://next100-094/damage/v2"),
    }
    common = dict(timestamp=NOW, confidence=1.0, verification=VerificationState.VERIFIED)
    db.add(memory_id="crate-count", store=MemoryStoreKind.VERIFIED_FACTS, content="crate_count=7", provenance=provenance["crate"], version=1, claim_key="crate_count", claim_value="7", **common)
    db.add(memory_id="units-per-crate", store=MemoryStoreKind.VERIFIED_FACTS, content="units_per_crate=6", provenance=provenance["unit"], version=1, claim_key="units_per_crate", claim_value="6", **common)
    db.add(memory_id="damaged-v1", store=MemoryStoreKind.RESEARCH_DOCUMENTS, content="damaged_units=3", provenance=provenance["damage_old"], version=1, claim_key="damaged_units", claim_value="3", **common)
    db.add(memory_id="damaged-v2", store=MemoryStoreKind.RESEARCH_DOCUMENTS, content="damaged_units=5", provenance=provenance["damage_new"], version=2, claim_key="damaged_units", claim_value="5", supersedes=("damaged-v1",), **common)

    retrieval = LexicalRetriever(db).retrieve("crate_count units_per_crate damaged_units", limit=10)
    evidence_ids = [item.memory_id for item in retrieval.evidence]
    assert "damaged-v1" not in evidence_ids
    assert {"crate-count", "units-per-crate", "damaged-v2"}.issubset(evidence_ids)
    facts = dict(_parse_fact(item.content) for item in retrieval.evidence)
    assert facts == {"crate_count": 7, "units_per_crate": 6, "damaged_units": 5}

    search = HypothesisSearch()
    wrong = search.propose("usable units are 39", initial_score=0.85)
    correct = search.propose("usable units are 37", initial_score=0.60)
    assert search.best() is not None and search.best().id == wrong.id

    controller = DeliberationController(
        LedgerAdapter(facts),
        EnsembleRankingAdapter(EXPECTED_ANSWER),
        tools=LedgerTool(),
        config=Config(initial_branches=1, target_score=1.0, min_confidence=1.0),
    )
    deliberation = controller.run(
        "Compute usable units from current retrieved ledger evidence.",
        Budget(model_calls=2, generated_tokens=8, tool_calls=1, candidate_branches=1),
    )
    observed = int(deliberation["final_text"])
    assert observed == EXPECTED_ANSWER

    wrong_test = search.test(
        wrong.id,
        name="retrieved-ledger deterministic tool result",
        prediction=39,
        observed=observed,
        weight=0.55,
        hard=True,
        source="POSTBASE-255 deterministic tool observation",
    )
    rejected = search.reject(
        wrong.id,
        "hard deterministic evidence contradicted the stale candidate",
        evidence_ids=(wrong_test.evidence_id,),
    )
    assert rejected.status == "rejected"
    correct_test = search.test(
        correct.id,
        name="retrieved-ledger deterministic tool result",
        prediction=37,
        observed=observed,
        weight=0.30,
        hard=True,
        source="POSTBASE-255 deterministic tool observation",
    )
    assert correct_test.passed
    revised = search.revise(
        wrong.id,
        "usable units are 37 after replacing superseded damaged_units=3 with current damaged_units=5",
        initial_score=0.75,
    )
    revised_test = search.test(
        revised.id,
        name="revision agrees with current ledger",
        prediction=37,
        observed=observed,
        weight=0.20,
        hard=True,
        source="POSTBASE-358 retrieved evidence + deterministic tool",
    )
    assert revised_test.passed

    wrong_result = VerifierEnsemble((ExactAnswerFixtureVerifier(),)).verify(
        VerificationRequest(
            claims=(Claim("answer", "wrong candidate claims usable units are 39"),),
            exact_fixtures=(ExactAnswerFixture("answer", EXPECTED_ANSWER, 39),),
        )
    )
    assert wrong_result.status is VerificationStatus.FAIL

    final_ensemble = VerifierEnsemble(
        (
            ExactAnswerFixtureVerifier(),
            UnitTestCodeVerifier(),
            NumericCalculatorVerifier(),
            SourceProvenanceChecker(require_content_hash=True),
            ConsistencyChecker(),
            CrossCandidateContradictionChecker(),
        )
    )
    provenance_records = tuple(
        ProvenanceRecord(
            "answer",
            item.provenance.source_id,
            item.provenance.locator or f"synthetic://next100-094/{item.memory_id}",
            item.content_hash,
        )
        for item in retrieval.evidence
    )
    final_request = VerificationRequest(
        claims=(Claim("answer", "usable units equal 37"),),
        exact_fixtures=(ExactAnswerFixture("answer", EXPECTED_ANSWER, observed),),
        unit_tests=(UnitTestEvidence("answer", "synthetic-ledger-self-check", 0, 1),),
        numeric_checks=(NumericCheck("answer", "7 * 6 - 5", "37"),),
        structured_facts=(
            StructuredFact("answer", "usable_units", 37),
            StructuredFact("answer", "usable_units", observed),
        ),
        provenance=provenance_records,
        candidate_facts=(
            CandidateFact(correct.id, "answer", "usable_units", 37),
            CandidateFact(revised.id, "answer", "usable_units", observed),
        ),
    )
    final_result = final_ensemble.verify(final_request)
    final_plan = FinalAnswerController().build_plan(final_request, final_result)
    assert final_result.status is VerificationStatus.PASS
    assert final_plan.verified_claim_ids == ("answer",)

    public_controller_trace = _sanitize_deliberation_trace(deliberation["trace"])
    rendered = json.dumps(public_controller_trace, sort_keys=True)
    assert "PRIVATE_NEXT100_094" not in rendered

    trace = {
        "schema": SCHEMA,
        "worker_id": WORKER_ID,
        "execution_policy": {
            "runtime_policy": "LOCAL_FREE",
            "external_llm_calls": 0,
            "network_retrieval_calls": 0,
            "base_weight_changes": 0,
            "private_reasoning_exposed": False,
        },
        "objective_task": {
            "fixture": "sealed synthetic inventory ledger",
            "expression": "7 * 6 - 5",
            "expected_answer": EXPECTED_ANSWER,
        },
        "base_adapter": _run_base_adapter(checkpoint),
        "memory_retrieval": {
            "query": retrieval.query,
            "evidence": [
                {
                    "memory_id": item.memory_id,
                    "version": item.version,
                    "source_id": item.provenance.source_id,
                    "source_version": item.provenance.source_version,
                    "content_sha256": item.content_hash,
                    "supersedes": list(item.supersedes),
                }
                for item in retrieval.evidence
            ],
            "superseded_excluded": ["damaged-v1"],
            "conflicts": [asdict(item) for item in retrieval.conflicts],
        },
        "deliberation": {
            "final_observation": observed,
            "trace": public_controller_trace,
        },
        "hypothesis_search": {
            "initial_preferred": wrong.id,
            "wrong_candidate": {
                "hypothesis_id": wrong.id,
                "prediction": 39,
                "observed": observed,
                "evidence_id": wrong_test.evidence_id,
                "status": search.hypothesis(wrong.id).status,
            },
            "correct_candidate": {
                "hypothesis_id": correct.id,
                "prediction": 37,
                "status": search.hypothesis(correct.id).status,
            },
            "revision": {
                "parent_id": wrong.id,
                "hypothesis_id": revised.id,
                "prediction": 37,
                "status": search.hypothesis(revised.id).status,
            },
            "selected_hypothesis_id": search.best().id if search.best() else None,
        },
        "verifier_ensemble": {
            "wrong_candidate_status": wrong_result.status.value,
            "final_status": final_result.status.value,
            "verifiers": [item.verifier_id for item in final_ensemble.verifiers],
            "verified_claim_ids": list(final_plan.verified_claim_ids),
        },
        "final": {"answer": observed, "verified": True},
    }
    db.close()
    return trace


def canonical_json(trace: dict[str, Any]) -> str:
    return json.dumps(trace, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
