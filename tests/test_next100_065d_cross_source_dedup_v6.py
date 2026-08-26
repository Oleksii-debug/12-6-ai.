from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data import cross_source_capacity_audit_v6 as v6

ROOT = Path(__file__).resolve().parents[1]
V5_CONFIG = ROOT / "configs/data/next100_065c_cross_source_dedup_v5.json"
V6_CONFIG = ROOT / "configs/data/next100_065d_cross_source_dedup_v6.json"


def _configs() -> tuple[dict[str, object], dict[str, object]]:
    return (
        json.loads(V5_CONFIG.read_text(encoding="utf-8")),
        json.loads(V6_CONFIG.read_text(encoding="utf-8")),
    )


def test_static_v6_contract_accepts_committed_exact_vector() -> None:
    v5_config, v6_config = _configs()
    v6._validate_config(v6_config, v5_config)


def test_static_v6_contract_rejects_gutenberg_policy_drift() -> None:
    v5_config, v6_config = _configs()
    v6_config = copy.deepcopy(v6_config)
    v6_config["gutenberg"]["normalization_policy"] = "UNREGISTERED"
    with pytest.raises(v6.CrossSourceV6Error, match="Gutenberg normalization drift"):
        v6._validate_config(v6_config, v5_config)


def test_static_v6_contract_rejects_attrs_authority_drift() -> None:
    v5_config, v6_config = _configs()
    v6_config = copy.deepcopy(v6_config)
    v6_config["attrs"]["head_sha"] = "0" * 40
    with pytest.raises(v6.CrossSourceV6Error, match="attrs head drift"):
        v6._validate_config(v6_config, v5_config)


def test_gutenberg_body_normalization_matches_preregistered_boundary() -> None:
    raw = (
        b"transport header\r\n"
        b"*** START OF THE PROJECT GUTENBERG EBOOK DEMO ***\r\n"
        b"\r\n"
        b"Cafe\xcc\x81 and body text.\r\n"
        b"\r\n"
        b"*** END OF THE PROJECT GUTENBERG EBOOK DEMO ***\r\n"
        b"transport footer\r\n"
    )
    normalized = v6._normalize_gutenberg_body(raw, "utf-8")
    assert normalized == "Caf\u00e9 and body text.\n".encode("utf-8")


def test_gutenberg_body_normalization_fails_closed_on_ambiguous_markers() -> None:
    raw = (
        b"*** START OF THE PROJECT GUTENBERG EBOOK DEMO ***\n"
        b"one\n"
        b"*** START OF THE PROJECT GUTENBERG EBOOK DEMO TWO ***\n"
        b"two\n"
        b"*** END OF THE PROJECT GUTENBERG EBOOK DEMO ***\n"
    )
    with pytest.raises(v6.CrossSourceV6Error, match="START marker count drift"):
        v6._normalize_gutenberg_body(raw, "utf-8")


def test_numpy_materialization_is_exact_utf8_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = b"def exact_identity(value):\n    return value\n"
    spec = {
        "worker": "NEXT100-049-CODE-NUMPY",
        "upstream_commit": "1" * 40,
        "source_family": "github:numpy/numpy",
        "stable_origin_id": "github:numpy/numpy",
        "head_sha": "2" * 40,
        "dedicated_workflow_run": 123,
        "normalization_policy": "STRICT_UTF8_IDENTITY_PRESERVE_V1",
        "files": [
            {
                "path": "numpy/_core/demo.py",
                "git_blob_sha1": v6.v5._git_blob_sha1(raw),
                "raw_bytes": len(raw),
            }
        ],
        "exact_capacity_bytes": len(raw),
    }
    monkeypatch.setattr(v6.v5.v1, "fetch_exact_source", lambda _: raw)

    rows, payloads, evidence = v6._materialize_numpy(spec)

    assert len(rows) == 1
    assert rows[0]["source_family"] == "github:numpy/numpy"
    source_id = rows[0]["source_id"]
    assert payloads[source_id] == raw
    assert evidence[0]["raw_sha256"] == v6.v5._sha256(raw)
    assert evidence[0]["normalization_policy"] == "STRICT_UTF8_IDENTITY_PRESERVE_V1"


def test_attrs_materialization_enters_code_family_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    first = b"def one():\n    return 1\n"
    second = b"def two():\n    return 2\n"
    payload_by_suffix = {"_make.py": first, "_funcs.py": second}
    spec = {
        "worker": "NEXT100-053-CODE-ATTRS",
        "upstream_commit": "3" * 40,
        "source_family": "github:python-attrs/attrs",
        "stable_origin_id": "github:python-attrs/attrs",
        "head_sha": "4" * 40,
        "dedicated_workflow_run": 456,
        "normalization_policy": "STRICT_UTF8_IDENTITY_PRESERVE_V1",
        "files": [
            {
                "path": "src/attr/_make.py",
                "git_blob_sha1": v6.v5._git_blob_sha1(first),
                "raw_bytes": len(first),
            },
            {
                "path": "src/attr/_funcs.py",
                "git_blob_sha1": v6.v5._git_blob_sha1(second),
                "raw_bytes": len(second),
            },
        ],
        "exact_capacity_bytes": len(first) + len(second),
    }

    def fetch(url: str) -> bytes:
        return next(raw for suffix, raw in payload_by_suffix.items() if url.endswith(suffix))

    monkeypatch.setattr(v6.v5.v1, "fetch_exact_source", fetch)
    rows, payloads, evidence = v6._materialize_attrs(spec)

    assert len(rows) == 2
    assert {row["source_family"] for row in rows} == {"github:python-attrs/attrs"}
    assert all(row["modality"] == "code" for row in rows)
    assert sum(len(value) for value in payloads.values()) == len(first) + len(second)
    assert {item["source_family"] for item in evidence} == {"github:python-attrs/attrs"}


def test_v6_report_verifier_refuses_full_cpython_source_credit() -> None:
    report = {
        "schema_version": v6.SCHEMA,
        "worker_id": v6.WORKER_ID,
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "raw_text_emitted": False,
        "source_vector": {
            "source_object_count": 35,
            "source_family_counts": {"code": 6, "en": 5, "uk": 4},
            "cpython_accepted_capacity_bytes": 17901,
            "source_capacity_bytes_before_global_dedup": 2217976,
            "pre_dedup_planning_gap_bytes": 17782024,
        },
        "materialization_evidence": [],
        "claim_boundary": {},
        "dedup_v3": {},
    }
    core = dict(report)
    report["report_sha256"] = v6.v5._sha256(v6.v5._canonical_bytes(core))
    with pytest.raises(v6.CrossSourceV6Error, match="accepted capacity invalid"):
        v6.verify_report(report)
