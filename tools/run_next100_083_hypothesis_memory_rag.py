from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from twelve_six.memory_rag import (
    MemoryDatabase,
    MemoryStoreKind,
    Provenance,
    VerificationState,
)
from twelve_six.postbase_hypothesis import Hypothesis, HypothesisSearch
from twelve_six.postbase_hypothesis_memory import HypothesisMemoryRAG


def relation(hypothesis: Hypothesis, _evidence):
    statement = hypothesis.statement.casefold()
    if "multiplication before addition" in statement:
        return "support"
    if "addition before multiplication" in statement:
        return "contradiction"
    return "irrelevant"


def run_probe() -> dict[str, object]:
    db = MemoryDatabase()
    provenance = Provenance(
        "synthetic_fixture",
        "python-precedence-spec",
        "v2",
        "synthetic://next100-083/operator-precedence",
    )
    item = db.add(
        memory_id="precedence-v2",
        store=MemoryStoreKind.VERIFIED_FACTS,
        content="operator precedence multiplication before addition is the active rule",
        provenance=provenance,
        timestamp=datetime(2026, 8, 26, 18, 0, tzinfo=timezone.utc),
        version=2,
        confidence=1.0,
        verification=VerificationState.VERIFIED,
        claim_key="operator_precedence",
        claim_value="multiplication_before_addition",
    )

    search = HypothesisSearch()
    wrong = search.propose("addition before multiplication", initial_score=0.85)
    correct = search.propose("multiplication before addition", initial_score=0.60)
    initial = search.best()
    if initial is None:
        raise RuntimeError("fixture requires an initial preferred hypothesis")

    integration = HypothesisMemoryRAG(search, db)
    wrong_bound = integration.retrieve_and_bind(
        wrong.id,
        "operator precedence multiplication addition",
        relation,
        max_score_delta=0.55,
    )
    correct_bound = integration.retrieve_and_bind(
        correct.id,
        "operator precedence multiplication addition",
        relation,
        max_score_delta=0.55,
    )
    final = integration.preferred()
    if final is None:
        raise RuntimeError("fixture requires a final preferred hypothesis")

    correct_binding = correct_bound.bindings[0]
    proof_passed = (
        initial.id == wrong.id
        and wrong_bound.bindings[0].relation == "contradiction"
        and correct_binding.relation == "support"
        and final.id == correct.id
        and correct_binding.memory_id == item.memory_id
        and correct_binding.version == item.version
        and correct_binding.content_hash == item.content_hash
        and correct_binding.provenance == provenance
        and correct_binding.memory_state.value == "active"
    )
    return {
        "worker_id": integration.worker_id,
        "execution_profile": "LOCAL_FREE",
        "retrieval_mode": "sqlite_bm25_local_only",
        "external_retrieval_used": False,
        "embeddings_used": False,
        "external_model_judge_used": False,
        "initial_preferred_hypothesis_id": initial.id,
        "initial_preferred_was_wrong": initial.id == wrong.id,
        "final_preferred_hypothesis_id": final.id,
        "final_preferred_is_correct": final.id == correct.id,
        "retrieved_memory_id": correct_binding.memory_id,
        "retrieved_source": correct_binding.source,
        "retrieved_provenance": {
            "source_type": correct_binding.provenance.source_type,
            "source_id": correct_binding.provenance.source_id,
            "source_version": correct_binding.provenance.source_version,
            "locator": correct_binding.provenance.locator,
        },
        "retrieved_memory_version": correct_binding.version,
        "retrieved_integrity_hash": correct_binding.content_hash,
        "retrieved_memory_state": correct_binding.memory_state.value,
        "proof_passed": proof_passed,
        "integration": integration.export(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe()
    if not report["proof_passed"]:
        raise SystemExit("NEXT100-083 hypothesis-memory-RAG proof failed")
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
