#!/usr/bin/env python3
"""Execute S4 meta/resource/analogue preflights without authorizing paid compute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.s4_readiness import (
    S4RunProfile,
    accelerator_preflight,
    estimate_s4_resources,
    meta_parameter_probe,
    run_scaled_analogue,
)


def _load_run_profile(path: Path) -> tuple[Path, S4RunProfile, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("compute_authorized") is not False:
        raise ValueError("readiness probe requires compute_authorized=false")
    root = Path(__file__).resolve().parents[1]
    stage_config = root / str(payload["stage_config"])
    profile = S4RunProfile(
        name=str(payload["name"]),
        sequence_length=int(payload["sequence_length"]),
        micro_batch_size=int(payload["micro_batch_size"]),
        gradient_accumulation_steps=int(payload["gradient_accumulation_steps"]),
        max_steps=int(payload["max_steps"]),
        precision=str(payload["precision"]),
    )
    return stage_config, profile, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("configs/runs/s4_100m_pilot.json"),
    )
    parser.add_argument("--analogue", action="store_true")
    args = parser.parse_args()

    run_path = args.run_config.resolve()
    stage_path, profile, run_payload = _load_run_profile(run_path)
    result: dict[str, object] = {
        "schema_version": 1,
        "run_config": str(run_path),
        "run_name": profile.name,
        "compute_authorized": run_payload["compute_authorized"],
        "launch_state": run_payload.get("launch_state"),
        "data_mode": run_payload.get("data_mode"),
        "launch_blockers": run_payload.get("launch_blockers", []),
        "meta_parameters": meta_parameter_probe(stage_path),
        "accelerator": accelerator_preflight(),
        "resources": estimate_s4_resources(stage_path, profile).to_dict(),
    }
    if args.analogue:
        result["scaled_analogue"] = run_scaled_analogue().to_dict()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
