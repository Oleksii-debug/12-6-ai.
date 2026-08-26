from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.data import cross_source_capacity_audit_v7 as v7

ROOT = Path(__file__).resolve().parents[1]
V5_CONFIG = ROOT / "configs/data/next100_065c_cross_source_dedup_v5.json"
V6_CONFIG = ROOT / "configs/data/next100_065d_cross_source_dedup_v6.json"
V7_CONFIG = ROOT / "configs/data/next100_065e_cross_source_dedup_v7.json"


def _configs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return (
        json.loads(V5_CONFIG.read_text(encoding="utf-8")),
        json.loads(V6_CONFIG.read_text(encoding="utf-8")),
        json.loads(V7_CONFIG.read_text(encoding="utf-8")),
    )


def test_static_v7_contract_accepts_exact_attrs_reconciliation() -> None:
    v5_config, v6_config, v7_config = _configs()
    v7._validate_config(v7_config, v5_config, v6_config)


def test_static_v7_contract_rejects_nonterminal_attrs() -> None:
    v5_config, v6_config, v7_config = _configs()
    v7_config = copy.deepcopy(v7_config)
    v7_config["attrs"]["dedicated_workflow_conclusion"] = "queued"
    with pytest.raises(v7.CrossSourceV7Error, match="attrs nonterminal"):
        v7._validate_config(v7_config, v5_config, v6_config)


def test_static_v7_contract_rejects_registry_self_promotion() -> None:
    v5_config, v6_config, v7_config = _configs()
    v7_config = copy.deepcopy(v7_config)
    v7_config["registry_v5_reconciliation"]["registry_workflow_terminal"] = True
    with pytest.raises(v7.CrossSourceV7Error, match="nonterminal registry"):
        v7._validate_config(v7_config, v5_config, v6_config)


def test_attrs_materialization_is_exact_utf8_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = b"def exact_attrs(value):\n    return value\n"
    spec = {
        "upstream_commit": v7.ATTRS_COMMIT,
        "stable_origin_id": v7.ATTRS_FAMILY,
        "normalization_policy": "STRICT_UTF8_IDENTITY_PRESERVE_V1",
        "files": [
            {
                "path": "src/attr/demo.py",
                "git_blob_sha1": v7.v6.v5._git_blob_sha1(raw),
                "raw_bytes": len(raw),
            }
        ],
        "exact_capacity_bytes": len(raw),
    }
    monkeypatch.setattr(v7.v6.v5.v1, "fetch_exact_source", lambda _: raw)

    rows, payloads, evidence = v7._materialize_attrs(spec)

    assert len(rows) == 1
    assert rows[0]["source_family"] == v7.ATTRS_FAMILY
    source_id = rows[0]["source_id"]
    assert payloads[source_id] == raw
    assert evidence[0]["raw_sha256"] == v7.v6.v5._sha256(raw)
    assert evidence[0]["authority_identity_sha256"] == v7.ATTRS_AUTHORITY


def test_v7_report_verifier_rejects_missing_attrs_family() -> None:
    report = {
        "schema_version": v7.SCHEMA,
        "worker_id": v7.WORKER_ID,
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "raw_text_emitted": False,
        "source_vector": {
            "source_object_count": 35,
            "source_family_counts": {"code": 5, "en": 5, "uk": 4},
            "cpython_accepted_capacity_bytes": 15540,
            "source_capacity_bytes_before_global_dedup": 2215615,
            "pre_dedup_planning_gap_bytes": 17784385,
        },
        "materialization_evidence": [],
        "claim_boundary": {},
        "dedup_v3": {},
    }
    core = dict(report)
    report["report_sha256"] = v7.v6.v5._sha256(v7.v6.v5._canonical_bytes(core))
    with pytest.raises(v7.CrossSourceV7Error, match="family vector drift"):
        v7.verify_report(report)
