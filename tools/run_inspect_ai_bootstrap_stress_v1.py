#!/usr/bin/env python3
"""Real Inspect AI runtime qualification runner."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import resource
import time
from pathlib import Path


def oracle(value: str, target: str) -> bool:
    return value == target


def run_once() -> dict:
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput
    from inspect_ai.scorer import match
    from inspect_ai.solver import solver
    from inspect_ai.solver._task_state import TaskState

    sample = Sample(
        input="Return the exact token 12-6",
        target="12-6",
        id="synthetic-001",
    )

    @solver
    def deterministic_solver():
        async def solve(state: TaskState, generate):
            state.output = ModelOutput.from_content(
                "project-deterministic-local", sample.target
            )
            return state

        return solve

    _ = deterministic_solver
    state = TaskState(
        model="project-deterministic-local",
        sample_id=sample.id or "synthetic-001",
        epoch=1,
        input=sample.input,
        messages=[],
        target=sample.target,
        output=ModelOutput.from_content(
            "project-deterministic-local", sample.target
        ),
    )

    scorer = match(location="exact", ignore_case=False)
    started = time.perf_counter()
    result = asyncio.run(scorer(state, state.target))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    value = state.output.completion
    oracle_value = oracle(value, sample.target)
    score_value = result.value if result is not None else None
    assert bool(score_value) is oracle_value
    return {
        "runtime": "EXECUTED",
        "inspect_ai_version": __import__("inspect_ai").__version__,
        "sample_id": sample.id,
        "score_value": score_value,
        "oracle_value": oracle_value,
        "parity": "PASS",
        "wall_time_ms": elapsed_ms,
        "rss_max_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run_once()
    except Exception as exc:
        result = {
            "runtime": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    result["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "network_blocked": os.environ.get(
            "INSPECT_AI_NETWORK_BLOCKED", "false"
        )
        == "true",
        "external_model_provider_used": False,
    }
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return (
        0
        if result["runtime"] == "EXECUTED"
        and result.get("parity") == "PASS"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
