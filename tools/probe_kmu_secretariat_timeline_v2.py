#!/usr/bin/env python3
"""Compose bounded KMu timeline discovery with the existing PR #815 page inspector.

Evidence-only LOCAL_FREE probe. It never admits corpus bytes or family credit.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


discovery = _load("kmu_timeline_discovery_v1", ROOT / "discover_kmu_news_timeline_v1.py")
probe = _load("kmu_bulk_probe_v1", ROOT / "probe_kmu_secretariat_bulk_v1.py")

SCHEMA = "12-6.kmu-secretariat-timeline-probe.v2"


def build_report(max_pages: int, max_fetches: int, delay_seconds: float) -> dict[str, Any]:
    if max_fetches < 1 or max_fetches > 500:
        raise ValueError("max_fetches must be in 1..500")
    if delay_seconds < 1.0:
        raise ValueError("delay_seconds must be >=1.0")

    inventory = discovery.discover(max_pages, delay_seconds)
    urls = list(inventory["news_urls"])[:max_fetches]
    rows: list[dict[str, Any]] = []
    for index, url in enumerate(urls):
        if index:
            time.sleep(delay_seconds)
        try:
            rows.append(probe.inspect_page(probe.SitemapEntry(url, "")))
        except Exception as exc:
            rows.append({
                "url": url,
                "lastmod": "",
                "eligible_probe_record": False,
                "rejection_reasons": [f"fetch_or_parse_error:{type(exc).__name__}"],
            })

    eligible = [row for row in rows if row.get("eligible_probe_record") is True]
    unique: dict[str, int] = {}
    for row in eligible:
        digest = str(row["normalized_probe_sha256"])
        unique.setdefault(digest, int(row["normalized_probe_bytes"]))
    observed_unique_bytes = sum(unique.values())

    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "parent_authority": {
            "pr": probe.PARENT_PR,
            "head_sha": probe.PARENT_HEAD,
            "manifest_identity_sha256": probe.PARENT_MANIFEST,
            "family_id": probe.FAMILY_ID,
        },
        "discovery": inventory,
        "sample": {
            "requested_fetch_limit": max_fetches,
            "attempted_pages": len(rows),
            "eligible_pages": len(eligible),
            "duplicate_eligible_payloads": len(eligible) - len(unique),
            "observed_unique_probe_bytes": observed_unique_bytes,
            "family_cap_bytes": probe.FAMILY_CAP_BYTES,
            "rows": rows,
        },
        "probe_boundary": {
            "training_authorized_bytes": 0,
            "family_count_credit_added": 0,
            "tokenizer_fit_executed": False,
            "optimizer_updates": 0,
            "final_test_accessed": False,
            "paid_compute_used": False,
            "canonical_registry_mutated": False,
            "probe_is_not_bulk_admission": True,
        },
    }
    core["verdict"] = "PROBE_USEFUL_OBSERVED_YIELD" if observed_unique_bytes > 0 else "PROBE_INSUFFICIENT_OBSERVED_YIELD"
    core["probe_identity_sha256"] = probe.canonical_sha256(core)
    return core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=5)
    parser.add_argument("--max-fetches", type=int, default=120)
    parser.add_argument("--delay-seconds", type=float, default=1.05)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = build_report(args.max_pages, args.max_fetches, args.delay_seconds)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
