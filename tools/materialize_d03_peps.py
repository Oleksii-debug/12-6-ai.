#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from twelve_six.data.pep_intake import (
    PepIntakeError,
    materialize,
    validate_config,
    write_materialization,
)

CONFIG = Path("configs/data/d03_pep_public_domain_intake_v1.json")


def load_config(repo_root: Path) -> dict[str, Any]:
    path = repo_root / CONFIG
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=_headers(url))
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise PepIntakeError(f"HTTP {response.status} for tree request")
        if response.geturl() != url:
            raise PepIntakeError("tree request redirected")
        return json.loads(response.read().decode("utf-8"))


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers=_headers(url))
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise PepIntakeError(f"HTTP {response.status} for raw PEP request")
        if response.geturl() != url:
            raise PepIntakeError("raw PEP request redirected")
        return response.read()


def _headers(url: str) -> dict[str, str]:
    headers = {"User-Agent": "12-6-ai-d03-pep-intake-v1"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    config = load_config(root)
    validate_config(config)
    tree = fetch_json(config["upstream"]["tree_url"])
    candidates, report = materialize(config, tree, fetch_bytes)
    write_materialization(args.output_dir.resolve(), candidates, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
