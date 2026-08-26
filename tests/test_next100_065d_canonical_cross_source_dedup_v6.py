from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data import cross_source_capacity_audit_v6 as v6

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/next100_065d_canonical_cross_source_dedup_v6.json"


def test_static_v6_contract_accepts_canonical_vector() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    v6._validate_config(config)


def test_static_v6_contract_rejects_stale_v5_parent() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["v5_parent"]["head_sha"] = "0" * 40
    with pytest.raises(v6.CrossSourceV6Error, match="V5 parent exact head drift"):
        v6._validate_config(config)


def test_static_v6_contract_rejects_numpy_capacity_inflation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["numpy"]["selected_files"][0]["raw_bytes"] += 1
    with pytest.raises(v6.CrossSourceV6Error, match="NumPy byte-capacity arithmetic drift"):
        v6._validate_config(config)


def test_static_v6_contract_rejects_gutenberg_family_multiplication() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["expected_pre_global_dedup_vector"]["source_family_counts"]["en"] = 7
    with pytest.raises(v6.CrossSourceV6Error, match="expected family vector drift"):
        v6._validate_config(config)


def test_gutenberg_body_normalization_is_marker_bounded_and_nfc() -> None:
    raw = (
        "Project Gutenberg wrapper\r\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK DEMO ***\r\n"
        "\r\n"
        "Cafe\u0301 line\r\n"
        "Interior  spacing stays\r\n"
        "\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK DEMO ***\r\n"
        "license wrapper\r\n"
    ).encode("utf-8")
    normalized = v6._normalize_gutenberg_body(raw, "utf-8")
    assert normalized == "Café line\nInterior  spacing stays\n".encode("utf-8")
    assert b"wrapper" not in normalized


def test_gutenberg_materialization_rejects_transport_blob_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK DEMO ***\n"
        "Exact admitted body.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK DEMO ***\n"
    ).encode("utf-8")
    normalized = v6._normalize_gutenberg_body(raw, "utf-8")
    spec = {
        "head_sha": "a" * 40,
        "source_family": "en.project-gutenberg.public-domain-books",
        "stable_origin_id": "project-gutenberg:test",
        "expected_total_capacity_bytes": len(normalized),
        "records": [
            {
                "source_id": "en.project-gutenberg.test",
                "encoding": "utf-8",
                "transport_repo": "example/example",
                "transport_commit": "b" * 40,
                "transport_path": "book.txt",
                "transport_git_blob_sha1": "0" * 40,
                "raw_bytes": len(raw),
                "raw_sha256": v6._sha256(raw),
                "normalized_utf8_bytes": len(normalized),
                "normalized_sha256": v6._sha256(normalized),
            }
        ],
    }
    monkeypatch.setattr(v6.v1, "fetch_exact_source", lambda _: raw)
    with pytest.raises(v6.CrossSourceV6Error, match="Gutenberg Git blob drift"):
        v6._materialize_gutenberg(spec)


def test_numpy_materialization_binds_blob_and_preserves_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"def add_one(value):\n    return value + 1\n"
    spec = {
        "head_sha": "a" * 40,
        "upstream_commit": "b" * 40,
        "source_family": "github:numpy/numpy",
        "stable_origin_id": "github:numpy/numpy@test",
        "expected_total_capacity_bytes": len(raw),
        "selected_files": [
            {
                "path": "numpy/_core/example.py",
                "git_blob_sha1": v6._git_blob_sha1(raw),
                "raw_bytes": len(raw),
            }
        ],
    }
    monkeypatch.setattr(v6.v1, "fetch_exact_source", lambda _: raw)
    rows, payloads, evidence = v6._materialize_numpy(spec)
    assert len(rows) == 1
    assert rows[0]["declared_capacity_bytes"] == len(raw)
    assert rows[0]["source_family"] == "github:numpy/numpy"
    assert payloads[rows[0]["source_id"]] == raw
    assert evidence[0]["git_blob_sha1"] == v6._git_blob_sha1(raw)


def test_v6_expected_pre_dedup_arithmetic_is_exact() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    capacity = config["expected_pre_global_dedup_vector"]["capacity_bytes"]
    assert capacity["en"] == 150643 + 15540 + 1672110
    assert capacity["code"] == 69133 + 36898
    assert capacity["total"] == capacity["uk"] + capacity["en"] + capacity["code"]
    assert capacity["total"] == 2045180
