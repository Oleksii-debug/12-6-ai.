#!/usr/bin/env python3
"""Fail-closed validator for SWARM-779 DataTrove bootstrap evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys

EXPECTED = {
    "project": "Oleksii-debug/12-6-ai.",
    "upstream_repository": "huggingface/datatrove",
    "release": "v0.10.0",
    "tag_commit": "7024aecca2f9ffb7b7cf0d02c0c823b8b24cf664",
    "license": "Apache-2.0",
    "python": "3.13.5",
    "requested_distribution": "datatrove==0.10.0",
    "install_exit_code": 2,
    "install_result": "BLOCKED_ENVIRONMENT",
}


def fail(message: str) -> int:
    print(f"FAIL: {message}")
    return 1


def main() -> int:
    path = Path(__file__).resolve().parents[1] / "evidence" / "audit" / "datatrove_bootstrap_stress_v1.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return fail(f"evidence unreadable or invalid JSON: {exc}")

    upstream = data.get("upstream", {})
    env = data.get("environment", {})
    install = data.get("install_attempt", {})
    runtime = data.get("runtime", {})
    base = data.get("canonical_base_boundary", {})
    promotion = data.get("promotion", {})

    checks = {
        "project": data.get("project") == EXPECTED["project"],
        "upstream_repository": upstream.get("repository") == EXPECTED["upstream_repository"],
        "release": upstream.get("release") == EXPECTED["release"],
        "tag_commit": upstream.get("tag_commit") == EXPECTED["tag_commit"],
        "license": upstream.get("license") == EXPECTED["license"],
        "python": env.get("python") == EXPECTED["python"],
        "exact_requested_distribution": install.get("requested_distribution") == EXPECTED["requested_distribution"],
        "install_attempted": install.get("attempted") is True,
        "install_exit_code": install.get("exit_code") == EXPECTED["install_exit_code"],
        "install_blocked": install.get("result") == EXPECTED["install_result"],
        "runtime_not_executed": runtime.get("real_datatrove_import_executed") is False,
        "benchmark_not_executed": runtime.get("benchmark_executed") is False,
        "parity_false": runtime.get("parity_proven") is False,
        "no_mock_runtime": runtime.get("mock_runtime_used_as_evidence") is False,
        "base_clean": all(
            base.get(key) is False
            for key in (
                "canonical_base_changed",
                "foreign_pretrained_weights_used",
                "foreign_instruction_or_alignment_behavior_used",
                "tokenizer_replaced",
                "corpus_mutated",
                "checkpoint_mutated",
                "training_executed",
                "paid_compute",
                "final_test_payload_accessed",
            )
        ),
        "promotion_not_adoptable": promotion.get("adoptable") is False,
        "promotion_candidate": promotion.get("state") == "CANDIDATE",
    }

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        return fail("; ".join(failed))

    print("PASS: DataTrove bootstrap evidence is internally consistent and fail-closed")
    print("NOTE: PASS here validates evidence mechanics only; DataTrove runtime itself was NOT_EXECUTED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
