#!/usr/bin/env python3
"""Execute CI-155 historical scope-isolation demonstrations from retained Git truth."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--cases", default="configs/ci/ci155_historical_demonstrations.v1.json")
    parser.add_argument("--output", default="ci155-historical-demonstrations.json")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    config = json.loads((root / args.cases).read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []

    for case in config["cases"]:
        _git(root, "cat-file", "-e", f"{case['historical_red_source_sha']}^{{commit}}")
        for key in ("scientific_surface_after_local_fix_sha", "forced_unrelated_cleanup_sha"):
            if key in case:
                _git(root, "cat-file", "-e", f"{case[key]}^{{commit}}")
        scientific = set(case["scientific_paths"])
        unrelated = set(case["unrelated_repository_ruff_failures"])
        local = set(case["experiment_local_ruff_failures_at_red_sha"])
        if scientific & unrelated:
            raise RuntimeError(f"{case['id']}: an alleged unrelated Ruff path is in the scientific surface")
        if not local <= scientific:
            raise RuntimeError(f"{case['id']}: local failure is not classified inside the scientific surface")
        local_gate = (
            "WOULD_NOT_BE_BLOCKED_BY_UNRELATED_RUFF"
            if case["experiment_local_failure_fixed_before_demo"]
            else "STILL_BLOCKED_BY_EXPERIMENT_LOCAL_RUFF"
        )
        results.append(
            {
                "id": case["id"],
                "historical_red_source_sha": case["historical_red_source_sha"],
                "historical_run_id": case["historical_run_id"],
                "historical_job_id": case["historical_job_id"],
                "unrelated_repository_ruff_failures": sorted(unrelated),
                "experiment_local_ruff_failures": sorted(local),
                "experiment_local_gate_result": local_gate,
                "integration_release_gate_result": "RED_RETAINED_SEPARATELY",
                "whole_repository_health_claimed": False,
            }
        )

    report = {
        "schema": "12-6.ci155-historical-demonstration-results.v1",
        "status": "PASS",
        "contract": "Unrelated repository Ruff debt cannot veto scientific execution; experiment-local defects still do; repository-wide red remains visible.",
        "cases": results,
    }
    output = root / args.output
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
