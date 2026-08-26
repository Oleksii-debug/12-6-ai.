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
            "source_object_count": 31,
            "source_family_counts": {"code": 5, "en": 5, "uk": 4},
            "cpython_accepted_capacity_bytes": 17901,
            "source_capacity_bytes_before_global_dedup": 2047541,
            "pre_dedup_planning_gap_bytes": 17952459,
        },
        "materialization_evidence": [],
        "claim_boundary": {},
        "dedup_v3": {},
    }
    core = dict(report)
    report["report_sha256"] = v6.v5._sha256(v6.v5._canonical_bytes(core))
    with pytest.raises(v6.CrossSourceV6Error, match="accepted capacity invalid"):
        v6.verify_report(report)
