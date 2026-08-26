from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "validate_next100_047_jinja_code_source.py"
spec = importlib.util.spec_from_file_location("next100_047_jinja", SCRIPT)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_config_is_bounded_and_capacity_exact() -> None:
    cfg = mod.load_config(ROOT)
    assert cfg["source_family"] == "github:pallets/jinja"
    assert cfg["upstream_commit"] == "5ef70112a1ff19c05324ff889dd30405b1002044"
    assert len(cfg["files"]) == 5
    assert sum(item["size_bytes"] for item in cfg["files"]) == 238695
    assert all(item["path"].startswith("src/jinja2/") for item in cfg["files"])
    assert not any("test" in item["path"].casefold() for item in cfg["files"])


def test_git_blob_identity_is_content_bound() -> None:
    data = b"print('jinja')\n"
    digest = mod.git_blob_sha1(data)
    assert len(digest) == 40
    assert digest != mod.git_blob_sha1(data + b"# drift\n")


def test_secret_scan_rejects_private_key() -> None:
    raw = b"-----BEGIN PRIVATE KEY-----\nnot-real\n"
    with pytest.raises(mod.AdmissionError, match="secret-like"):
        mod.scan_secret_privacy(raw, raw.decode(), "src/jinja2/x.py")


def test_privacy_scan_allows_example_email_but_rejects_private_endpoint() -> None:
    raw = b"# support@example.com\n"
    result = mod.scan_secret_privacy(raw, raw.decode(), "src/jinja2/x.py")
    assert result["email_like_count"] == 1
    private = b"endpoint = '10.20.30.40'\n"
    with pytest.raises(mod.AdmissionError, match="private-network"):
        mod.scan_secret_privacy(private, private.decode(), "src/jinja2/x.py")


def test_near_dedup_detects_copy_and_separates_unrelated_code() -> None:
    copied = "def alpha(x):\n    return x + 1\n" * 20
    unrelated = "class Beta:\n    def run(self, value):\n        while value:\n            value -= 1\n" * 20
    assert mod.near_jaccard(copied, copied) == 1.0
    assert mod.near_jaccard(copied, unrelated) < 0.85


def test_path_gate_excludes_generated_or_vendor_surfaces() -> None:
    mod.assert_path_allowed("src/jinja2/parser.py")
    with pytest.raises(mod.AdmissionError):
        mod.assert_path_allowed("src/jinja2/generated/parser.py")
    with pytest.raises(mod.AdmissionError):
        mod.assert_path_allowed("tests/test_parser.py")
