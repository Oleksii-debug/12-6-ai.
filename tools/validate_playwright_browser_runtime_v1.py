#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/playwright_browser_runtime_v1.json"

REQUIRED = {
    "component_id": "PLAYWRIGHT_BROWSER_RUNTIME_V1",
    "tag": "v1.62.0",
    "commit": "e3950d9c140d007bd52853b45813c6274b24e36f",
    "license": "Apache-2.0",
    "package_version": "1.62.0",
    "promotion_requires_real_runtime": True,
    "parity_proven": False,
    "adopted": False,
}


def canonical_bytes(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def validate(cfg: dict) -> None:
    assert cfg["component_id"] == REQUIRED["component_id"]
    upstream = cfg["upstream"]
    package = cfg["package"]
    gate = cfg["runtime_gate"]
    assert upstream["tag"] == REQUIRED["tag"]
    assert upstream["commit"] == REQUIRED["commit"]
    assert upstream["license"] == REQUIRED["license"]
    assert package["version"] == REQUIRED["package_version"]
    assert gate["promotion_requires_real_runtime"] is True
    assert gate["parity_proven"] is False
    assert gate["adopted"] is False
    actions = set(cfg["allowed_actions"])
    assert "click_element" in actions and "open_page" in actions
    forbidden = set(cfg["forbidden"])
    assert {
        "arbitrary_shell",
        "wildcard_network",
        "wildcard_filesystem",
        "implicit_credentials",
    } <= forbidden
    truth = cfg["truth_boundary"]
    assert all(value is False for value in truth.values())


def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate(cfg)
    digest = hashlib.sha256(canonical_bytes(cfg)).hexdigest()
    print(json.dumps({"status": "PASS", "config_sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
