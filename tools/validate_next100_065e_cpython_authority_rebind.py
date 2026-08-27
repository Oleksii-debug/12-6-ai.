#!/usr/bin/env python3
"""Validate append-only evolution of the bound CPython source authority.

NEXT100-065C/V6 intentionally binds an immutable, terminal CPython source
admission head. The owning PR later advanced by adding an accepted-chunk
materializer and workflow hardening while leaving the source-authority config
byte-identical. A mutable PR tip must not invalidate immutable source evidence,
but arbitrary tip movement must still fail closed.

This verifier therefore permits exactly one reviewed successor tip only when:
- the frozen authority head is its ancestor;
- the exact source-authority config Git blob is unchanged;
- the commit delta touches only the reviewed hardening files;
- both the frozen and successor dedicated runs are terminal-success.

After proving those conditions it reuses the incumbent live validator for every
other authority, temporarily substituting only the CPython PR-tip/run pair.
No source bytes, accepted hashes, rights, capacity, or dedup policy are changed.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

LEGACY_VALIDATOR = Path(__file__).with_name("validate_next100_065c_authority_bindings.py")
CPYTHON_PR = 467
FROZEN_HEAD = "5a6a495a24bce449334cbc5126d0114f61a9f57c"
FROZEN_RUN = 32998356906
SUCCESSOR_HEAD = "df2d750f7e262759f3cc54c04662ffe208286dc2"
SUCCESSOR_RUN = 33006886741
AUTHORITY_CONFIG = "configs/data/next100_037_python_docs_source_authority_v1.json"
AUTHORITY_CONFIG_BLOB = "b15abac8744ccda9fe58d1351f7925b6ab328034"
AUTHORITY_IDENTITY = "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d"
ALLOWED_SUCCESSOR_DELTA = {
    ".github/workflows/next100-037-python-docs-source-authority.yml",
    "tools/materialize_next100_037_python_docs_accepted_chunks.py",
}


def _load_legacy() -> ModuleType:
    spec = importlib.util.spec_from_file_location("next100_065c_authority", LEGACY_VALIDATOR)
    if spec is None or spec.loader is None:
        raise SystemExit("NEXT100-065E CPYTHON REBIND FAIL: cannot import incumbent validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NEXT100-065E CPYTHON REBIND FAIL: {message}")


def _validate_frozen_static(legacy: ModuleType) -> None:
    v5 = legacy.load_json(legacy.V5_CONFIG)
    v6 = legacy.load_json(legacy.V6_CONFIG)
    legacy.validate_v5_static(v5)
    legacy.validate_v6_static(v6)

    cp = v5.get("cpython", {})
    _require(cp.get("head_sha") == FROZEN_HEAD, "frozen V5 CPython head drift")
    _require(cp.get("dedicated_workflow_run") == FROZEN_RUN, "frozen V5 CPython run drift")
    _require(
        cp.get("authority_identity_sha256") == AUTHORITY_IDENTITY,
        "frozen V5 CPython authority identity drift",
    )


def _validate_run(legacy: ModuleType, run_id: int, expected_head: str, label: str) -> None:
    run = legacy.github_get(f"actions/runs/{run_id}")
    _require(run.get("head_sha") == expected_head, f"{label} run head mismatch")
    _require(run.get("status") == "completed", f"{label} run nonterminal")
    _require(run.get("conclusion") == "success", f"{label} run not success")


def _validate_append_only_successor(legacy: ModuleType) -> None:
    pr = legacy.github_get(f"pulls/{CPYTHON_PR}")
    current_head = pr.get("head", {}).get("sha")
    _require(current_head == SUCCESSOR_HEAD, "CPython owner PR moved beyond reviewed successor")

    compare = legacy.github_get(f"compare/{FROZEN_HEAD}...{SUCCESSOR_HEAD}")
    _require(compare.get("status") == "ahead", "successor is not strictly ahead of frozen authority")
    _require(compare.get("behind_by") == 0, "successor is not a descendant of frozen authority")
    _require(compare.get("ahead_by") == 2, "reviewed successor commit count drift")
    _require(
        compare.get("merge_base_commit", {}).get("sha") == FROZEN_HEAD,
        "frozen authority is not the successor merge base",
    )
    files = compare.get("files")
    _require(isinstance(files, list), "compare response has no file list")
    changed = {str(item.get("filename")) for item in files if isinstance(item, dict)}
    _require(changed == ALLOWED_SUCCESSOR_DELTA, "successor changed files outside reviewed hardening delta")

    frozen_contents = legacy.github_get(f"contents/{AUTHORITY_CONFIG}?ref={FROZEN_HEAD}")
    successor_contents = legacy.github_get(f"contents/{AUTHORITY_CONFIG}?ref={SUCCESSOR_HEAD}")
    _require(
        frozen_contents.get("sha") == AUTHORITY_CONFIG_BLOB,
        "frozen CPython authority config blob drift",
    )
    _require(
        successor_contents.get("sha") == AUTHORITY_CONFIG_BLOB,
        "successor changed CPython authority config bytes",
    )

    _validate_run(legacy, FROZEN_RUN, FROZEN_HEAD, "frozen CPython authority")
    _validate_run(legacy, SUCCESSOR_RUN, SUCCESSOR_HEAD, "successor CPython authority")


def validate_live() -> None:
    legacy = _load_legacy()
    _validate_frozen_static(legacy)
    _validate_append_only_successor(legacy)

    # Reuse every incumbent live check; only the mutable CPython owner tip/run
    # changes after the immutable/append-only proof above.
    legacy.LIVE_AUTHORITIES[CPYTHON_PR] = (SUCCESSOR_HEAD, SUCCESSOR_RUN)
    legacy.validate_live()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-live", action="store_true")
    args = parser.parse_args()
    legacy = _load_legacy()
    _validate_frozen_static(legacy)
    if args.github_live:
        _validate_append_only_successor(legacy)
        legacy.LIVE_AUTHORITIES[CPYTHON_PR] = (SUCCESSOR_HEAD, SUCCESSOR_RUN)
        legacy.validate_live()
    print("NEXT100-065E CPython immutable-authority successor: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
