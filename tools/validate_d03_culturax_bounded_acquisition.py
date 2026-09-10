from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.culturax_bounded_acquisition import build_bounded_plan, validate_contract


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate/build bounded CulturaX UA/EN plan")
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/data/d03_culturax_bounded_acquisition_v1.json"),
    )
    parser.add_argument("--uk-checksum", type=Path)
    parser.add_argument("--en-checksum", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    validate_contract(contract)
    if (args.uk_checksum is None) != (args.en_checksum is None):
        parser.error("--uk-checksum and --en-checksum must be supplied together")
    if args.uk_checksum is None:
        print("VALID BLOCKED_GATED_ACCESS_ACCEPTANCE_REQUIRED")
        return 0

    plan = build_bounded_plan(
        contract,
        {
            "uk": args.uk_checksum.read_text(encoding="utf-8"),
            "en": args.en_checksum.read_text(encoding="utf-8"),
        },
    )
    rendered = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
        print(plan["plan_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
