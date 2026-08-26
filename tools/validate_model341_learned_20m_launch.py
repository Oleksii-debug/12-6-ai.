#!/usr/bin/env python3
"""Inspect MODEL-341 learned-20M launch readiness without performing training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.model341_launch_readiness import assess_model341_launch, load_packet

DEFAULT_PACKET = Path("configs/launch/model341_learned_20m_launch_packet_v1.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument(
        "--require",
        choices=("well-formed", "authorization-request", "bounded-smoke", "long-training"),
        default="well-formed",
    )
    args = parser.parse_args()

    result = assess_model341_launch(load_packet(args.packet))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))

    requirements = {
        "well-formed": True,
        "authorization-request": result.ready_for_authorization_request,
        "bounded-smoke": result.bounded_smoke_authorized,
        "long-training": result.long_training_authorized,
    }
    return 0 if requirements[args.require] else 2


if __name__ == "__main__":
    raise SystemExit(main())
