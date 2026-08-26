#!/usr/bin/env python3
"""Fail-closed static/live authority validator for NEXT100-065D V6."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from twelve_six.data.cross_source_capacity_audit_v6 import (
    GUTENBERG_HEAD,
    GUTENBERG_PARENT_HEAD,
    GUTENBERG_RUN,
    NUMPY_HEAD,
    NUMPY_RUN,
    _validate_config,
)

CONFIG = Path("configs/data/next100_065d_cross_source_dedup_v6.json")
REPO = "Oleksii-debug/12-6-ai."


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"NEXT100-065D FAIL: {message}")


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "config root must be object")
    return value


def github_get(path: str) -> dict[str, Any]:
    token = os.environ.get("GITHUB_TOKEN")
    require(bool(token), "GITHUB_TOKEN is required for --github-live")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "12-6-ai-next100-065d-authority-validator",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    require(isinstance(value, dict), f"GitHub response not object: {path}")
    return value


def validate_live() -> None:
    numpy_pr = github_get("pulls/468")
    require(numpy_pr.get("head", {}).get("sha") == NUMPY_HEAD, "NumPy PR #468 head moved")
    numpy_run = github_get(f"actions/runs/{NUMPY_RUN}")
    require(numpy_run.get("head_sha") == NUMPY_HEAD, "NumPy run head mismatch")
    require(numpy_run.get("status") == "completed", "NumPy run nonterminal")
    require(numpy_run.get("conclusion") == "success", "NumPy run not success")

    seal_pr = github_get("pulls/627")
    require(
        seal_pr.get("head", {}).get("sha") == GUTENBERG_HEAD,
        "Gutenberg terminal seal PR #627 head moved",
    )
    source_pr = github_get("pulls/470")
    require(
        source_pr.get("head", {}).get("sha") == GUTENBERG_PARENT_HEAD,
        "Gutenberg source PR #470 head moved",
    )
    gutenberg_run = github_get(f"actions/runs/{GUTENBERG_RUN}")
    require(
        gutenberg_run.get("head_sha") == GUTENBERG_PARENT_HEAD,
        "Gutenberg run head mismatch",
    )
    require(gutenberg_run.get("status") == "completed", "Gutenberg run nonterminal")
    require(
        gutenberg_run.get("conclusion") == "success",
        "Gutenberg run not success",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-live", action="store_true")
    args = parser.parse_args()
    _validate_config(load_config())
    if args.github_live:
        validate_live()
    print("NEXT100-065D AUTHORITY PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
