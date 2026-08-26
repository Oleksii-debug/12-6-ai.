"""Validate the immutable EVAL-354 communication-suite fixture and scorer mechanics."""

from __future__ import annotations

import json
from pathlib import Path

from twelve_six.post_base.communication_eval import evaluate_suite, load_suite, reference_responses

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "tests"
    / "fixtures"
    / "post_base"
    / "communication_eval"
    / "v1"
    / "manifest.json"
)


def main() -> int:
    suite = load_suite(MANIFEST)
    responses = reference_responses(suite)
    first = evaluate_suite(suite, responses)
    second = evaluate_suite(suite, responses)
    deterministic = first.as_dict() == second.as_dict()

    payload = first.as_dict()
    payload["suite_version"] = suite.version
    payload["cases_sha256"] = suite.cases_sha256
    payload["fixture_case_count"] = len(suite.cases)
    payload["category_counts"] = dict(suite.category_counts)
    payload["reference_fixture_mechanics_only"] = True
    payload["scorer_deterministic"] = deterministic
    payload["model_executed"] = False
    payload["optimizer_updates"] = 0
    payload["training_eligible"] = False
    payload["foreign_model_output"] = False
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0 if first.passed and deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
