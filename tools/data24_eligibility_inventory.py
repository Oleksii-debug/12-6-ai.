"""Render model-training eligibility from the canonical D03 source registry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.data.external_sources import build_eligibility_inventory

DEFAULT_REGISTRY = Path("data/external/external_sources.json")


def run(registry_path: Path) -> dict[str, object]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    return build_eligibility_inventory(registry)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    inventory = run(args.registry)
    rendered = json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
