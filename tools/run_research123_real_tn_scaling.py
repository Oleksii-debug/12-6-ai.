#!/usr/bin/env python3
"""Exact launcher for the RESEARCH-123 orchestration harness.

The report plan is JSON, so a tuple-valued TrainerConfig.betas field round-trips as
a JSON array. Normalize that experiment-plan value back to the incumbent D02 tuple
contract in the fresh child only. The Trainer implementation is not modified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import research123_real_tn_scaling as experiment

# Make the incumbent harness's fresh-process self-spawn return through this exact
# launcher, so its child receives the same experiment-only JSON normalization.
experiment.__file__ = __file__

if "--resume-child" in sys.argv:
    incumbent_trainer_config = experiment.TrainerConfig

    def normalized_trainer_config(**kwargs):
        if isinstance(kwargs.get("betas"), list):
            kwargs["betas"] = tuple(kwargs["betas"])
        return incumbent_trainer_config(**kwargs)

    experiment.TrainerConfig = normalized_trainer_config

_original_run_experiment = experiment.run_experiment


def _run_experiment_with_exact_repro(*, source_sha: str, output_dir: Path, torch_threads: int):
    report = _original_run_experiment(
        source_sha=source_sha,
        output_dir=output_dir,
        torch_threads=torch_threads,
    )
    report.pop("report_sha256", None)
    report["reproduction_command"] = (
        f"python tools/run_research123_real_tn_scaling.py --source-sha {source_sha} "
        f"--output-dir {output_dir} --torch-threads {torch_threads}"
    )
    report["report_sha256"] = experiment.hash_json(report)
    return report


experiment.run_experiment = _run_experiment_with_exact_repro

if __name__ == "__main__":
    raise SystemExit(experiment.main())
