from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data import cross_source_capacity_audit_v5 as v5
from twelve_six.data import cross_source_capacity_audit_v6 as v6

ROOT = Path(__file__).resolve().parents[1]
V5_CONFIG = ROOT / "configs/data/next100_065c_cross_source_dedup_v5.json"
V6_CONFIG = ROOT / "configs/data/next100_065d_cross_source_dedup_v6.json"


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _synthetic_gutenberg() -> tuple[bytes, bytes, dict[str, object]]:
    raw = (
        "transport header\r\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK DEMO ***\r\n"
        "\r\n"
        "Cafe\u0301\r\n"
        "Second line\r\n"
        "\r\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK DEMO ***\r\n"
        "transport footer\r\n"
    ).encode("utf-8")
    normalized = v6._normalize_gutenberg_body(raw, "utf-8")
    record = {
        "source_id": "en.project-gutenberg.synthetic",
        "ebook_id": 999999,
        "encoding": "utf-8",
        "transport_repo": "example/synthetic",
        "transport_commit": "a" * 40,
        "transport_path": "synthetic.txt",
        "raw_bytes": len(raw),
        "git_blob_sha1": v5._git_blob_sha1(raw),
        "normalized_bytes": len(normalized),
        "normalized_sha256": v5._sha256(normalized),
    }
    spec: dict[str, object] = {
        "head_sha": "b" * 40,
        "dedicated_workflow_run": 1,
        "source_family": "en.project-gutenberg.public-domain-books",
        "normalization_policy": v6.GUTENBERG_POLICY,
        "records": [record],
        "exact_capacity_bytes": len(normalized),
    }
    return raw, normalized, spec


def test_static_v6_contract_accepts_committed_exact_vector() -> None:
    v5_config = _load(V5_CONFIG)
    v6_config = _load(V6_CONFIG)
    v6._validate_config(v6_config, v5_config)


def test_static_v6_contract_rejects_gutenberg_evaluation_authority_inflation() -> None:
    v5_config = _load(V5_CONFIG)
    v6_config = copy.deepcopy(_load(V6_CONFIG))
    gutenberg = v6_config["gutenberg"]
    assert isinstance(gutenberg, dict)
    gutenberg["evaluation"] = "ALLOWED"

    with pytest.raises(v6.CrossSourceV6Error, match="Gutenberg evaluation boundary drift"):
        v6._validate_config(v6_config, v5_config)


def test_static_v6_contract_rejects_planning_capacity_rebinding() -> None:
    v5_config = _load(V5_CONFIG)
    v6_config = copy.deepcopy(_load(V6_CONFIG))
    expected = v6_config["expected_vector"]
    assert isinstance(expected, dict)
    expected["expected_total_if_cpython_accepted_capacity_is_15540"] = 20_000_000

    with pytest.raises(v6.CrossSourceV6Error, match="V6 planning vector drift"):
        v6._validate_config(v6_config, v5_config)


def test_gutenberg_normalization_is_body_only_lf_nfc() -> None:
    raw, normalized, _ = _synthetic_gutenberg()

    assert b"transport header" not in normalized
    assert b"transport footer" not in normalized
    assert b"PROJECT GUTENBERG" not in normalized
    assert normalized == "Caf\u00e9\nSecond line\n".encode("utf-8")
    assert b"\r" not in normalized
    assert len(raw) > len(normalized)


def test_gutenberg_normalization_rejects_ambiguous_markers() -> None:
    raw = (
        "*** START OF THE PROJECT GUTENBERG EBOOK DEMO ***\n"
        "body\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK DUPLICATE ***\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK DEMO ***\n"
    ).encode("utf-8")

    with pytest.raises(v6.CrossSourceV6Error, match="START marker count drift: 2"):
        v6._normalize_gutenberg_body(raw, "utf-8")


def test_gutenberg_materialization_binds_transport_and_normalized_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, normalized, spec = _synthetic_gutenberg()
    monkeypatch.setattr(v5.v1, "fetch_exact_source", lambda _url: raw)

    rows, payloads, evidence = v6._materialize_gutenberg(spec)

    assert len(rows) == 1
    assert rows[0]["source_id"] == "en.project-gutenberg.synthetic"
    assert rows[0]["declared_capacity_bytes"] == len(normalized)
    assert payloads["en.project-gutenberg.synthetic"] == normalized
    assert evidence[0]["raw_sha256"] == v5._sha256(raw)
    assert evidence[0]["git_blob_sha1"] == v5._git_blob_sha1(raw)
    assert evidence[0]["normalized_sha256"] == v5._sha256(normalized)
    assert evidence[0]["normalization_policy"] == v6.GUTENBERG_POLICY


def test_gutenberg_materialization_rejects_same_size_transport_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, _normalized, spec = _synthetic_gutenberg()
    rebound = raw.replace(b"Second line", b"Second LINE")
    assert len(rebound) == len(raw)
    monkeypatch.setattr(v5.v1, "fetch_exact_source", lambda _url: rebound)

    with pytest.raises(v6.CrossSourceV6Error, match="Gutenberg Git blob drift"):
        v6._materialize_gutenberg(spec)
