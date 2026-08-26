#!/usr/bin/env python3
"""Validate and summarize the Research Corpus V1 terminal-source intake."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twelve_six.data.research_corpus_intake import (  # noqa: E402
    build_research_corpus_intake_report,
    load_research_corpus_intake,
)


AUTHORITY = (
    ROOT / "configs" / "data" / "research_corpus_v1_intake_convergence_v1.json"
)


def main() -> int:
    authority = load_research_corpus_intake(AUTHORITY)
    report = build_research_corpus_intake_report(authority)

    print("RESEARCH_CORPUS_V1_INTAKE=PASS")
    print("TRAINING_AUTHORIZED=false")
    print("AUTHORIZED_UNIQUE_OPTIMIZED_TARGETS=0")
    print(
        "KNOWN_TERMINAL_INTAKE_BYTES="
        + str(report["known_terminal_intake_total_bytes"])
    )
    print(
        "CAPACITY_PROXY_TOTAL_GAP_BYTES="
        + str(report["capacity_proxy_total_gap_bytes"])
    )
    print(
        "REPORT_JSON="
        + json.dumps(report, sort_keys=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
