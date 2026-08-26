from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data import cross_source_capacity_audit_v6 as v6


CONFIG_PATH = Path("configs/data/next100_109_cross_source_dedup_v6.json")


def _config() -> dict[str, object]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v6_machine_contract_is_internally_consistent() -> None:
    config = _config()
    v6._validate_config(config)
    expected = config["expected_vector"]
    assert expected["source_capacity_bytes_before_global_dedup"] == 2_045_180
    assert expected["source_capacity_by_modality_before_global_dedup"] == {
        "uk": 100_856,
        "en": 1_838_293,
        "code": 106_031,
    }
    assert sum(expected["source_capacity_by_modality_before_global_dedup"].values()) == 2_045_180


def test_v6_rejects_training_authorization_drift() -> None:
    config = copy.deepcopy(_config())
    config["claim_boundary"]["training_authorized"] = True
    with pytest.raises(v6.CrossSourceV6Error, match="claim boundary weakened"):
        v6._validate_config(config)


def test_v6_rejects_parent_or_terminal_capacity_drift() -> None:
    config = copy.deepcopy(_config())
    config["parent_v5"]["expected_cpython_accepted_capacity_bytes"] = 17_901
    with pytest.raises(v6.CrossSourceV6Error, match="CPython terminal capacity drift"):
        v6._validate_config(config)

    config = copy.deepcopy(_config())
    config["gutenberg"]["expected_capacity_bytes"] += 1
    with pytest.raises(v6.CrossSourceV6Error, match="Gutenberg capacity drift"):
        v6._validate_config(config)


def test_gutenberg_normalizer_reproduces_marker_body_contract() -> None:
    raw = (
        "transport header\r\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n"
        "\r\n"
        "Cafe\u0301\r\n"
        "second line\r\n"
        "\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\r\n"
        "transport footer\r\n"
    ).encode("utf-8")
    assert v6._normalize_gutenberg_body(raw, "utf-8") == "Café\nsecond line\n".encode()


def test_gutenberg_normalizer_fails_closed_on_ambiguous_markers() -> None:
    raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK A ***\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK B ***\n"
        "body\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK A ***\n"
    ).encode()
    with pytest.raises(v6.CrossSourceV6Error, match="expected one Gutenberg START marker"):
        v6._normalize_gutenberg_body(raw, "utf-8")


def test_numpy_materialization_is_identity_preserving(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = b"def add_one(value):\n    return value + 1\n"
    blob = v6._git_blob_sha1(raw)
    spec = {
        "upstream_commit": "a" * 40,
        "repository_family": "github:numpy/numpy",
        "head_sha": "b" * 40,
        "dedicated_workflow_run": 1,
        "expected_capacity_bytes": len(raw),
        "selected_files": [
            {"path": "numpy/_core/example.py", "git_blob_sha1": blob, "raw_bytes": len(raw)}
        ],
    }
    monkeypatch.setattr(v6.v1, "fetch_exact_source", lambda _url: raw)
    rows, payloads, evidence = v6._materialize_numpy(spec)
    assert len(rows) == 1
    assert rows[0]["declared_capacity_bytes"] == len(raw)
    assert payloads[rows[0]["source_id"]] == raw
    assert evidence[0]["git_blob_sha1"] == blob


def test_numpy_materialization_rejects_non_python_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = b"def broken(:\n"
    spec = {
        "upstream_commit": "a" * 40,
        "repository_family": "github:numpy/numpy",
        "head_sha": "b" * 40,
        "dedicated_workflow_run": 1,
        "expected_capacity_bytes": len(raw),
        "selected_files": [
            {
                "path": "numpy/_core/broken.py",
                "git_blob_sha1": v6._git_blob_sha1(raw),
                "raw_bytes": len(raw),
            }
        ],
    }
    monkeypatch.setattr(v6.v1, "fetch_exact_source", lambda _url: raw)
    with pytest.raises(v6.CrossSourceV6Error, match="AST parse drift"):
        v6._materialize_numpy(spec)
