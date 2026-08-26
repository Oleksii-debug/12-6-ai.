from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.postbase_deliberation import (
    Budget, Config, DeliberationController, DeterministicMockAdapter, Verification,
)


class ConstantVerifier:
    def evaluate(self, task, text, branch_id, iteration):
        del task, text, branch_id, iteration
        return Verification(0.5, 1.0, "constant mechanics verifier")


def run(calls: int, branches: int):
    controller = DeliberationController(
        DeterministicMockAdapter(),
        ConstantVerifier(),
        config=Config(
            initial_branches=1,
            target_score=None,
            convergence_delta=0.0,
            convergence_rounds=100,
        ),
    )
    return controller.run(
        "Produce a precise bounded answer.",
        Budget(
            model_calls=calls,
            generated_tokens=512,
            tool_calls=0,
            candidate_branches=branches,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    small, large = run(3, 2), run(7, 4)
    assert large["trace"]["budget_consumed"]["model_calls"] > small["trace"]["budget_consumed"]["model_calls"]
    assert large["trace"]["budget_consumed"]["candidate_branches"] > small["trace"]["budget_consumed"]["candidate_branches"]
    assert large["score"] == small["score"]
    report = {
        "schema": "12-6.postbase255-local-free-probe.v1",
        "worker_id": "POSTBASE-255-DELIBERATION-CONTROLLER-V1",
        "execution_profile": "LOCAL_FREE",
        "external_teacher_api": False,
        "simulated_waiting": False,
        "small": small,
        "large": large,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
