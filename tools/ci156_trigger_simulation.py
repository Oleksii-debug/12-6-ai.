"""Deterministic CI-156 before/after fan-out simulation for representative PR changes."""

from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BASELINE_GLOBAL = (
    "CI",
    "D08 Purpose Environments",
    "D02 Real S0 Training",
    "D02 S0 Determinism Repeatability",
    "D02 S1 Numerical Preflight",
    "SCALE-02 S2 1M Executable Preflight",
    "TRAIN-29 S1 Training Observability",
)
M150 = "MILESTONE-150 Learned Base Ladder V1"
M150_BASE = "milestone100/first-learned-base-20260826"

SCOPES: dict[str, tuple[str, ...]] = {
    "D08 Purpose Environments": (
        ".github/workflows/d08-purpose-environments.yml",
        "requirements/locks/**",
        "requirements/profiles/**",
        "tools/verify_purpose_environment.py",
        "src/twelve_six/integration/dependency_lock.py",
        "pyproject.toml",
    ),
    "D02 Real S0 Training": (
        ".github/workflows/d02-s0-real-training.yml",
        "tools/run_s0_real_training.py",
        "tools/validate_s0_training_evidence.py",
        "configs/runs/s0_10k.d02_real_training.json",
        "tests/test_s0_real_training*.py",
    ),
    "D02 S0 Determinism Repeatability": (
        ".github/workflows/d02-s0-repeatability.yml",
        "tools/run_s0_determinism_probe.py",
        "tools/collect_s0_repeatability_evidence.py",
        "tools/validate_s0_repeatability_evidence.py",
        "configs/runs/s0_10k.d02_repeatability.json",
        "tests/test_s0_repeatability*.py",
    ),
    "D02 S1 Numerical Preflight": (
        ".github/workflows/d02-s1-numerical-preflight.yml",
        "tools/run_s1_numerical_preflight.py",
        "tools/validate_s1_numerical_preflight.py",
        "configs/runs/s1_100k.d02_numerical_preflight.json",
        "tests/test_s1_numerical_preflight*.py",
    ),
    "SCALE-02 S2 1M Executable Preflight": (
        ".github/workflows/scale02-s2-1m-executable.yml",
        "src/twelve_six/training/s2_preflight.py",
        "tests/test_s2_preflight*.py",
    ),
    "TRAIN-29 S1 Training Observability": (
        ".github/workflows/train29-s1-observability.yml",
        "src/twelve_six/training/observability.py",
        "tools/run_s1_observability_probe.py",
        "tools/validate_s1_observability_probe.py",
        "tests/test_training_observability*.py",
    ),
}

PORTABILITY_PATTERNS = (
    ".github/workflows/ci.yml",
    "requirements/**",
    "pyproject.toml",
    "src/twelve_six/integration/dependency_lock.py",
    "tools/verify_locked_environment.py",
)

CI_IGNORE = ("docs/**", "reports/**", "**/*.md")


@dataclass(frozen=True)
class Scenario:
    name: str
    base_branch: str
    paths: tuple[str, ...]


def _matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(path, pattern)


def any_match(paths: Iterable[str], patterns: Iterable[str]) -> bool:
    return any(_matches(path, pattern) for path in paths for pattern in patterns)


def ci_runs(paths: tuple[str, ...]) -> bool:
    return not paths or not all(any_match((path,), CI_IGNORE) for path in paths)


def portability_required(paths: tuple[str, ...]) -> bool:
    return any_match(paths, PORTABILITY_PATTERNS)


def before(scenario: Scenario) -> list[str]:
    workflows = list(BASELINE_GLOBAL)
    if scenario.base_branch == M150_BASE:
        workflows.append(M150)
    return workflows


def after(scenario: Scenario) -> list[str]:
    workflows: list[str] = []
    if ci_runs(scenario.paths):
        workflows.append("CI")
    for workflow, patterns in SCOPES.items():
        if any_match(scenario.paths, patterns):
            workflows.append(workflow)
    if scenario.base_branch == M150_BASE and any_match(
        scenario.paths,
        (
            ".github/workflows/milestone150-learned-base-ladder-v1.yml",
            "src/twelve_six/milestone150*.py",
            "tests/test_milestone150*.py",
            "src/twelve_six/milestone100_first_learned.py",
            "tests/test_milestone100*.py",
            "data/corpus_v01/**",
            "configs/data/corpus_v01.json",
        ),
    ):
        workflows.append(M150)
    return workflows


def representative_scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario(
            "milestone150_research_sha",
            M150_BASE,
            (
                "src/twelve_six/milestone150_resume_bridge.py",
                "tests/test_milestone150_resume_bridge.py",
                ".github/workflows/milestone150-learned-base-ladder-v1.yml",
            ),
        ),
        Scenario(
            "d02_real_training_change",
            "d02/s0-training-engine",
            ("tools/run_s0_real_training.py", ".github/workflows/d02-s0-real-training.yml"),
        ),
        Scenario(
            "dependency_lock_change",
            "main",
            ("requirements/locks/linux-x86_64/runtime.lock.txt",),
        ),
        Scenario("docs_only", "main", ("docs/ci156.md",)),
    )


def report() -> dict[str, object]:
    rows = []
    for scenario in representative_scenarios():
        old = before(scenario)
        new = after(scenario)
        rows.append(
            {
                "scenario": scenario.name,
                "base_branch": scenario.base_branch,
                "paths": list(scenario.paths),
                "before_workflows": old,
                "before_count": len(old),
                "after_workflows": new,
                "after_count": len(new),
                "workflow_runs_avoided": len(old) - len(new),
                "arm_portability_job_after": portability_required(scenario.paths),
            }
        )
    return {
        "schema": "12-6.ci156-trigger-simulation.v1",
        "baseline_semantics": "seven repository-global pull_request workflows plus MILESTONE-150 on its target branch",
        "after_semantics": "integration CI plus path-scoped experiment workflows; expensive campaigns retain workflow_dispatch",
        "scenarios": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = report()
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
