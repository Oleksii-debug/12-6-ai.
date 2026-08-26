#!/usr/bin/env python3
"""Hash-safe privacy V3 scan of the frozen terminal source vector.

The scanner downloads exact immutable objects, verifies byte length and SHA-256,
and emits only source ids, whole-object hashes, byte counts, actions, and detector
counts. It never emits source text, previews, matched values, or per-match hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twelve_six.data.privacy_filter_v3 import (  # noqa: E402
    assert_hash_safe_evidence,
    build_manifest,
    hash_safe_scan,
)


def canonical_sha(value: object) -> str:
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", default="configs/data/next100_068_terminal_source_inventory.json")
    p.add_argument("--output", default="next100_068_privacy_real_scan.json")
    args = p.parse_args()

    inventory_path = ROOT / args.inventory
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    identity_rows = []
    results = []
    source_rows = []

    for source in inventory["sources"]:
        with urllib.request.urlopen(source["url"], timeout=60) as response:
            raw = response.read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != source["expected_raw_sha256"]:
            raise SystemExit(f"FAIL hash drift: {source['source_id']}")
        if len(raw) != source["expected_raw_bytes"]:
            raise SystemExit(f"FAIL byte drift: {source['source_id']}")
        result = hash_safe_scan(raw)
        results.append(result)
        source_rows.append({"source_id": source["source_id"], **result.evidence()})
        identity_rows.append({
            "source_id": source["source_id"],
            "raw_sha256": digest,
            "raw_bytes": len(raw),
        })

    inventory_sha = canonical_sha(identity_rows)
    manifest = build_manifest(results, inventory_sha256=inventory_sha)
    manifest["sources"] = source_rows
    manifest["source_count"] = len(source_rows)
    manifest["all_source_identities_verified"] = True
    manifest["all_matched_private_values_retained"] = False
    manifest["all_matched_private_value_hashes_retained"] = False
    assert_hash_safe_evidence(manifest)

    output = ROOT / args.output
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_count": manifest["source_count"],
        "total_input_bytes": manifest["total_input_bytes"],
        "action_counts": manifest["action_counts"],
        "detector_counts": manifest["detector_counts"],
        "inventory_sha256": inventory_sha,
        "manifest_sha256": manifest["manifest_sha256"],
        "all_source_identities_verified": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
