"""Validate retained TRAIN-49 AdamW epsilon evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.training.adam_epsilon_experiment import validate_adam_epsilon_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence")
    args = parser.parse_args()
    payload = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    validate_adam_epsilon_evidence(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
