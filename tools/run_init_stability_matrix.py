#!/usr/bin/env python3
"""Run the controlled 12-6 initialization-stability matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six.training.init_stability import (
    ProbeSpec,
    run_stage_matrix,
    validate_report,
    write_report,
)

_ALLOWED_CANDIDATES = (
    "stage_default",
    "unscaled_residual_control",
    "s1_width_reference_control",
)


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "12-6.init-stability-experiment.v1":
        raise ValueError("unexpected experiment config schema")
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("experiment config requires non-empty stages")
    return payload


def _probe_spec(entry: dict[str, Any]) -> ProbeSpec:
    return ProbeSpec(
        batch_size=int(entry["batch_size"]),
        sequence_length=int(entry["sequence_length"]),
        steps=int(entry["steps"]),
        seeds=tuple(int(value) for value in entry["seeds"]),
        data_seed=int(entry.get("data_seed", 424242)),
        width_reference=int(entry.get("width_reference", 48)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/init_stability_matrix_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-sha",
        help="Exact repository commit SHA bound into every report.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        choices=_ALLOWED_CANDIDATES,
        help="Run only selected candidate(s); repeat flag for several.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        help="Run only selected stage labels from the matrix config.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    requested_candidates = tuple(args.candidate or config["candidates"])
    for candidate in requested_candidates:
        if candidate not in _ALLOWED_CANDIDATES:
            raise ValueError(f"unsupported candidate in config: {candidate!r}")

    requested_stages = set(args.stage or [])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, Any]] = []

    for entry in config["stages"]:
        label = str(entry["label"])
        if requested_stages and label not in requested_stages:
            continue
        stage_path = Path(entry["stage_config"])
        probe = _probe_spec(entry)
        for candidate in requested_candidates:
            report = run_stage_matrix(
                stage_config_path=stage_path,
                candidate=candidate,
                probe=probe,
                source_sha=args.source_sha,
            )
            validate_report(report)
            output = args.output_dir / f"{label.lower()}__{candidate}.json"
            write_report(output, report)
            index.append(
                {
                    "label": label,
                    "stage": report["stage"],
                    "candidate": candidate,
                    "model_identity_sha256": report["model_identity_sha256"],
                    "init_identity_sha256": report["candidate_init_identity_sha256"],
                    "report_sha256": report["report_sha256"],
                    "output": output.name,
                    "aggregate": report["aggregate"],
                }
            )

    if not index:
        raise ValueError("matrix selection produced no runs")
    (args.output_dir / "index.json").write_text(
        json.dumps(
            {
                "schema": "12-6.init-stability-index.v1",
                "config": args.config.as_posix(),
                "runs": index,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
