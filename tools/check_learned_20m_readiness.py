#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.learned_20m_readiness import (
    evaluate_learned_20m_readiness,
    verify_r01_campaign_path,
)

PHASE_FIELDS = {
    "local-pilot": "local_pilot_ready",
    "authorization-request": "authorization_request_ready",
    "material-training": "material_training_authorized",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate learned ~20M launch readiness.")
    parser.add_argument("packet", type=Path)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("configs/research/r01_20m_to_100m_scaling_campaign_v1.json"),
    )
    parser.add_argument("--require-phase", choices=sorted(PHASE_FIELDS))
    args = parser.parse_args()

    campaign_errors = verify_r01_campaign_path(args.campaign)
    if campaign_errors:
        print(json.dumps({"campaign_errors": campaign_errors}, sort_keys=True))
        return 2

    try:
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(json.dumps({"packet_error": str(exc)}, sort_keys=True))
        return 2
    if not isinstance(packet, dict):
        print(json.dumps({"packet_error": "packet root must be an object"}, sort_keys=True))
        return 2

    result = evaluate_learned_20m_readiness(packet)
    output = result.to_dict()
    print(json.dumps(output, indent=2, sort_keys=True))
    if args.require_phase and not output[PHASE_FIELDS[args.require_phase]]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
