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
    result = evaluate_suite(suite, reference_responses(suite))
    payload = result.as_dict()
    payload["reference_fixture_mechanics_only"] = True
    payload["model_executed"] = False
    payload["optimizer_updates"] = 0
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
