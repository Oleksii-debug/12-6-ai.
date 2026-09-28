"""Validate the Common Pile v0.1 source-rights registry."""

from __future__ import annotations

import argparse
from pathlib import Path

from twelve_six.common_pile_rights import CommonPileRightsError, load_and_validate

DEFAULT_REGISTRY = Path("configs/data/common_pile_source_rights_v1.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", nargs="?", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args()
    try:
        payload = load_and_validate(args.registry)
    except (OSError, ValueError, CommonPileRightsError) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        "PASS:",
        payload["registry_id"],
        payload["registry_identity_sha256"],
        f"sources={len(payload['sources'])}",
        "training_authorized=false",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
