from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.postbase_hypothesis import HypothesisSearch


def run_probe() -> dict[str, object]:
    search = HypothesisSearch()
    wrong = search.propose(
        "addition is evaluated before multiplication",
        assumptions=("ordinary Python integer arithmetic is the oracle",),
        initial_score=0.85,
    )
    correct = search.propose(
        "multiplication is evaluated before addition",
        assumptions=("ordinary Python integer arithmetic is the oracle",),
        initial_score=0.60,
    )
    initial_preferred = search.best()
    if initial_preferred is None:
        raise RuntimeError("probe requires an initial hypothesis")

    search.critique(
        wrong.id,
        "the hypothesis must predict an unparenthesized mixed-precedence expression",
    )
    parenthesized = search.branch(
        wrong.id,
        "addition is evaluated first when parentheses force that subexpression",
        initial_score=0.50,
    )
    search.test(
        parenthesized.id,
        name="parenthesized precedence fixture",
        prediction=20,
        observed=(2 + 3) * 4,
        weight=0.20,
        hard=True,
        source="python_language_semantics_fixture",
    )

    wrong_test = search.test(
        wrong.id,
        name="unparenthesized precedence fixture",
        prediction=20,
        observed=2 + 3 * 4,
        weight=0.55,
        hard=True,
        source="python_language_semantics_fixture",
    )
    search.reject(
        wrong.id,
        "hard fixture returned 14 rather than the predicted 20",
        evidence_ids=(wrong_test.evidence_id,),
    )
    correct_test = search.test(
        correct.id,
        name="unparenthesized precedence fixture",
        prediction=14,
        observed=2 + 3 * 4,
        weight=0.30,
        hard=True,
        source="python_language_semantics_fixture",
    )
    revised = search.revise(
        wrong.id,
        "addition is evaluated before multiplication only when grouping changes the expression",
        initial_score=0.55,
    )
    search.test(
        revised.id,
        name="revised grouping fixture",
        prediction=20,
        observed=(2 + 3) * 4,
        weight=0.25,
        hard=True,
        source="python_language_semantics_fixture",
    )

    final_preferred = search.best()
    if final_preferred is None:
        raise RuntimeError("probe requires a final hypothesis")
    wrong_state = search.hypothesis(wrong.id)
    proof_passed = (
        initial_preferred.id == wrong.id
        and wrong_test.passed is False
        and wrong_state.status == "rejected"
        and correct_test.passed is True
        and final_preferred.id == correct.id
    )
    return {
        "worker_id": search.worker_id,
        "execution_profile": "LOCAL_FREE",
        "external_llm_used": False,
        "unrestricted_shell_used": False,
        "fixture": "objective Python operator-precedence arithmetic",
        "initial_preferred_hypothesis_id": initial_preferred.id,
        "initial_preferred_was_wrong": initial_preferred.id == wrong.id,
        "wrong_hypothesis_rejected_after_evidence": wrong_state.status == "rejected",
        "final_preferred_hypothesis_id": final_preferred.id,
        "final_preferred_is_correct": final_preferred.id == correct.id,
        "proof_passed": proof_passed,
        "search": search.export(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_probe()
    if not report["proof_passed"]:
        raise SystemExit("POSTBASE-256 objective hypothesis-rejection proof failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
