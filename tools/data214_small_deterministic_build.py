#!/usr/bin/env python3
"""Build the DATA-214 non-authoritative deterministic convergence fixture twice."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from twelve_six import data183_corpus_v02_real as d183

AUTHORITY = "TEST_FIXTURE_ONLY_NOT_SOURCE_SNAPSHOT_OR_CORPUS_PROMOTION"

ROWS = (
    {
        "record_id": "fixture-ua-train",
        "text": "Український тестовий документ перевіряє детермінований корпусний шлях. " * 24,
        "split": "train",
        "stratum": "uk",
        "origin": "external_real",
        "source_id": "fixture-only:external-ua-contract",
    },
    {
        "record_id": "fixture-en-validation",
        "text": "English fixture document verifies the deterministic validation boundary. " * 24,
        "split": "validation",
        "stratum": "en",
        "origin": "external_real",
        "source_id": "fixture-only:external-en-contract",
    },
    {
        "record_id": "fixture-code-train",
        "text": "def deterministic_fixture(value):\n    return value + 1\n" * 32,
        "split": "train",
        "stratum": "code",
        "origin": "project_authored",
        "source_id": "fixture-only:project-authored-code",
    },
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _build(root: Path) -> dict[str, Any]:
    shard = root / "shards" / "part-00000.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in ROWS)
    shard.write_bytes(payload)
    shard_sha = _sha256_bytes(payload)
    manifest = {
        "physical": {
            "shards": [
                {
                    "path": "shards/part-00000.jsonl",
                    "sha256": shard_sha,
                }
            ]
        }
    }
    audit = d183.audit_release(root, manifest)
    return {
        "shard_sha256": shard_sha,
        "manifest_sha256": _sha256_bytes(_canonical_bytes(manifest)),
        "audit_sha256": _sha256_bytes(_canonical_bytes(audit)),
        "audit": audit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args(argv)

    out = args.output_dir.resolve()
    if out.exists():
        shutil.rmtree(out)
    build_a = _build(out / "build-a")
    build_b = _build(out / "build-b")
    for key in ("shard_sha256", "manifest_sha256", "audit_sha256"):
        if build_a[key] != build_b[key]:
            raise RuntimeError(f"deterministic fixture mismatch: {key}")

    report = {
        "schema": "12-6.data214-small-deterministic-build.v1",
        "authority": AUTHORITY,
        "source_sha": args.source_sha,
        "promoted_source_identities_consumed": False,
        "external_rows_are_contract_fixtures_only": True,
        "project_authored_code_remains_separate": True,
        "full_scientific_corpus_campaign_executed": False,
        "network_reacquisition_executed": False,
        "two_build_identity": {
            "identical": True,
            "shard_sha256": build_a["shard_sha256"],
            "manifest_sha256": build_a["manifest_sha256"],
            "audit_sha256": build_a["audit_sha256"],
        },
        "data183_audit": build_a["audit"],
        "scope": {
            "proves": [
                "deterministic shard materialization",
                "DATA-183 canonical origin separation",
                "normalized train-validation non-overlap",
                "optimized-token supply accounting",
                "project-authored code separation",
            ],
            "does_not_claim": [
                "real external source authority",
                "rights approval for fixture rows",
                "full DATA-110 policy-chain execution",
                "corpus V0.2 representativeness",
            ],
        },
    }
    report_path = out / "small-build-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", **report["two_build_identity"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
