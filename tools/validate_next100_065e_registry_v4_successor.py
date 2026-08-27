#!/usr/bin/env python3
"""Validate immutable NEXT100-063 V4 across append-only successor PR tips.

The inherited NEXT100-065D guard binds a content-addressed V4 registry, but its
live check historically also required PR #538 to remain on the exact commit that
first published V4. PR #538 later advanced by adding V5 as a separate successor
without changing the V4 file. Treating that PR-tip movement as V4 drift makes an
immutable authority depend on unrelated append-only successor commits.

This wrapper keeps the frozen V4 head/blob/identity in the committed guard and
permits a later PR tip only when:
- the frozen V4 head is an ancestor of the live PR tip;
- the live tip still contains the exact frozen V4 Git blob;
- the incumbent V4 static contract passes unchanged.

It then reuses the incumbent live validator against that live descendant. This
does not promote V5, change the 31-object V6 vector, or grant corpus/training
capacity. V5/V7 reconciliation remains a separate authority surface.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
from types import ModuleType

LEGACY_VALIDATOR = Path(__file__).with_name("validate_next100_065d_registry_v4_guard.py")


def _load_legacy() -> ModuleType:
    spec = importlib.util.spec_from_file_location("next100_065d_registry_v4_guard", LEGACY_VALIDATOR)
    if spec is None or spec.loader is None:
        raise SystemExit("NEXT100-065E REGISTRY V4 SUCCESSOR FAIL: cannot import V4 guard")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NEXT100-065E REGISTRY V4 SUCCESSOR FAIL: {message}")


def validate_live_descendant(legacy: ModuleType) -> None:
    data = legacy.load_config()
    legacy.validate_static(data)
    registry = data["canonical_registry"]
    frozen_head = registry["head_sha"]
    frozen_blob = registry["git_blob_sha1"]

    pr = legacy.github_get(f"pulls/{registry['pr']}")
    live_head = pr.get("head", {}).get("sha")
    _require(isinstance(live_head, str) and len(live_head) == 40, "live registry head is invalid")

    if live_head != frozen_head:
        comparison = legacy.github_get(f"compare/{frozen_head}...{live_head}")
        _require(comparison.get("status") == "ahead", "live registry tip is not ahead of frozen V4")
        _require(comparison.get("behind_by") == 0, "live registry tip is not a descendant of frozen V4")
        _require(
            comparison.get("merge_base_commit", {}).get("sha") == frozen_head,
            "frozen V4 head is not the live tip merge base",
        )

    payload = legacy.github_get(f"contents/{registry['path']}?ref={live_head}")
    _require(payload.get("sha") == frozen_blob, "live descendant changed immutable V4 registry bytes")

    rebound = copy.deepcopy(data)
    rebound["canonical_registry"]["head_sha"] = live_head
    legacy.validate_live(rebound)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-live", action="store_true")
    args = parser.parse_args()
    legacy = _load_legacy()
    data = legacy.load_config()
    legacy.validate_static(data)
    if args.github_live:
        validate_live_descendant(legacy)
    print("NEXT100-065E immutable registry V4 descendant: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
