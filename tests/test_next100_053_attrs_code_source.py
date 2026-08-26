from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "validate_next100_053_attrs_code_source.py"
spec = importlib.util.spec_from_file_location("next100_053_attrs", SCRIPT)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_config_is_bounded_and_capacity_exact() -> None:
    cfg = mod.load_config(ROOT)
    assert cfg["source_family"] == "github:python-attrs/attrs"
    assert cfg["release_tag"] == "26.1.0"
    assert cfg["upstream_commit"] == "7bfc49e9b22d5ba25b6e429524c3d49fee27cb36"
    assert len(cfg["files"]) == 4
    assert sum(item["size_bytes"] for item in cfg["files"]) == 170435
    assert all(item["path"].startswith("src/attr/") for item in cfg["files"])
    assert not any(item["path"].startswith("src/attrs/") for item in cfg["files"])
    assert not any("test" in item["path"].casefold() for item in cfg["files"])


def test_git_blob_identity_is_content_bound() -> None:
    data = b"print('attrs')\n"
    digest = mod.git_blob_sha1(data)
    assert len(digest) == 40
    assert digest != mod.git_blob_sha1(data + b"# drift\n")


def test_mit_grant_validation_allows_line_wrapped_canonical_text() -> None:
    text = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""
    mod.validate_mit_grant_text(text)


def test_mit_grant_validation_rejects_missing_core_grant() -> None:
    text = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software under restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""
    with pytest.raises(mod.AdmissionError, match="MIT grant text drift"):
        mod.validate_mit_grant_text(text)


def test_secret_scan_rejects_private_key() -> None:
    raw = b"-----BEGIN PRIVATE KEY-----\nnot-real\n"
    with pytest.raises(mod.AdmissionError, match="secret-like"):
        mod.scan_secret_privacy(raw, raw.decode(), "src/attr/x.py")


def test_privacy_scan_allows_example_email_but_rejects_private_endpoint() -> None:
    raw = b"# support@example.com\n"
    result = mod.scan_secret_privacy(raw, raw.decode(), "src/attr/x.py")
    assert result["email_like_count"] == 1
    private = b"endpoint = '10.20.30.40'\n"
    with pytest.raises(mod.AdmissionError, match="private-network"):
        mod.scan_secret_privacy(private, private.decode(), "src/attr/x.py")


def test_near_dedup_detects_copy_and_separates_unrelated_code() -> None:
    copied = "def alpha(x):\n    return x + 1\n" * 20
    unrelated = "class Beta:\n    def run(self, value):\n        while value:\n            value -= 1\n" * 20
    assert mod.near_jaccard(copied, copied) == 1.0
    assert mod.near_jaccard(copied, unrelated) < 0.85


def test_path_gate_excludes_forwarding_generated_and_vendor_surfaces() -> None:
    mod.assert_path_allowed("src/attr/_make.py")
    with pytest.raises(mod.AdmissionError):
        mod.assert_path_allowed("src/attrs/validators.py")
    with pytest.raises(mod.AdmissionError):
        mod.assert_path_allowed("src/attr/generated/parser.py")
    with pytest.raises(mod.AdmissionError):
        mod.assert_path_allowed("tests/test_make.py")
