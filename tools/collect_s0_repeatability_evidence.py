"""Collect three standalone S0 probes into one repeatability evidence artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.s0_repeatability import build_s0_repeatability_evidence


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--same-seed-a", type=Path, required=True)
    parser.add_argument("--same-seed-b", type=Path, required=True)
    parser.add_argument("--different-seed", type=Path, required=True)
    parser.add_argument("--locked-environment-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    evidence = build_s0_repeatability_evidence(
        _read_json(args.same_seed_a),
        _read_json(args.same_seed_b),
        _read_json(args.different_seed),
        _read_json(args.locked_environment_evidence),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_sha": evidence["identity"]["source_sha"],
                "same_seed": evidence["identity"]["same_seed"],
                "different_seed": evidence["identity"]["different_seed"],
                "same_seed_stable_result_sha256": evidence["proof"]["same_seed_stable_result_sha256"],
                "evidence_sha256": evidence["evidence_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
