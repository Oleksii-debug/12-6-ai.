#!/usr/bin/env python3
"""Stdlib-only validator for immutable Lingua bootstrap evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED = {
    "component_id": "LINGUA",
    "worker_id": "SWARM-780",
    "upstream_tag": "v2.2.0",
    "upstream_commit": "754ce21122c083a7200763015fdaf7cda8d85453",
    "package": "lingua-language-detector==2.2.0",
    "license": "Apache-2.0",
}


def fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def main() -> int:
    manifest_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "configs/research/lingua_bootstrap_stress_v1.json"
    )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("component_id") != EXPECTED["component_id"]:
        return fail("component identity drift")
    if data.get("worker_id") != EXPECTED["worker_id"]:
        return fail("worker identity drift")
    upstream = data.get("upstream", {})
    if upstream.get("tag") != EXPECTED["upstream_tag"]:
        return fail("upstream tag drift")
    if upstream.get("commit_sha") != EXPECTED["upstream_commit"]:
        return fail("upstream commit drift")
    if upstream.get("package") != EXPECTED["package"]:
        return fail("package version drift")
    if data.get("rights", {}).get("software_license") != EXPECTED["license"]:
        return fail("license drift")

    guard = data.get("canonical_base_impact", {})
    for field in ("canonical_base_modified", "foreign_weights_used", "foreign_alignment_used", "tokenizer_replaced", "training_executed", "paid_compute", "final_test_access"):
        if guard.get(field):
            return fail(f"forbidden canonical-boundary state: {field}")

    runtime = data.get("runtime", {})
    status = data.get("status")
    executed = all(
        runtime.get(field, False)
        for field in ("real_import_executed", "fixture_probe_executed")
    )
    if status == "ADOPTABLE_COMPONENT" and not all(
        runtime.get(field, False)
        for field in ("real_import_executed", "fixture_probe_executed", "benchmark_executed", "parity_executed")
    ):
        return fail("adoptable status without complete runtime evidence")
    if executed and status == "RETEST_RUNTIME_REQUIRED":
        return fail("executed runtime evidence conflicts with retest status")

    print("PASS: immutable Lingua bootstrap manifest contract is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
