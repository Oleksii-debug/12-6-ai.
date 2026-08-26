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


def test_static_v6_contract_accepts_live_reconciled_vector() -> None:
    v5_config, v6_config = _configs()
    v6._validate_config(v6_config, v5_config)


def test_static_v6_contract_rejects_gutenberg_capacity_drift() -> None:
    v5_config, v6_config = _configs()
    v6_config = copy.deepcopy(v6_config)
    v6_config["gutenberg"]["records"][1]["normalized_bytes"] -= 1
    with pytest.raises(v6.CrossSourceV6Error, match="Gutenberg capacity drift"):
        v6._validate_config(v6_config, v5_config)


def test_static_v6_contract_rejects_numpy_authority_drift() -> None:
    v5_config, v6_config = _configs()
    v6_config = copy.deepcopy(v6_config)
    v6_config["numpy"]["dedicated_workflow_conclusion"] = "failure"
    with pytest.raises(v6.CrossSourceV6Error, match="NumPy nonterminal"):
        v6._validate_config(v6_config, v5_config)


def test_static_v6_contract_rejects_training_boundary_weakened() -> None:
    v5_config, v6_config = _configs()
    v6_config = copy.deepcopy(v6_config)
    v6_config["claim_boundary"]["training_authorized"] = True
    with pytest.raises(v6.CrossSourceV6Error, match="claim boundary weakened"):
        v6._validate_config(v6_config, v5_config)


def test_gutenberg_normalizer_strips_envelope_and_preserves_body() -> None:
    raw = (
        b"Project Gutenberg license envelope\r\n"
        b"*** START OF THE PROJECT GUTENBERG EBOOK DEMO ***\r\n"
        b"\r\nBody line one.\r\nBody line two.\r\n\r\n"
        b"*** END OF THE PROJECT GUTENBERG EBOOK DEMO ***\r\n"
        b"trailing license envelope\r\n"
    )
    normalized = v6._normalize_gutenberg_body(raw, "ascii")
    assert normalized == b"Body line one.\nBody line two.\n"
    assert b"license envelope" not in normalized
    assert b"START OF" not in normalized
    assert b"END OF" not in normalized


def test_gutenberg_normalizer_rejects_ambiguous_markers() -> None:
    raw = (
        b"*** START OF THE PROJECT GUTENBERG EBOOK ONE ***\n"
        b"body\n"
        b"*** START OF THE PROJECT GUTENBERG EBOOK TWO ***\n"
        b"*** END OF THE PROJECT GUTENBERG EBOOK ONE ***\n"
    )
    with pytest.raises(v6.CrossSourceV6Error, match="START marker count drift"):
        v6._normalize_gutenberg_body(raw, "ascii")


def test_numpy_materializer_rejects_git_blob_rebinding(monkeypatch: pytest.MonkeyPatch) -> None:
    _, v6_config = _configs()
    spec = copy.deepcopy(v6_config["numpy"])
    spec["files"] = [
        {
            "path": "numpy/_core/demo.py",
            "git_blob_sha1": "0" * 40,
            "raw_bytes": 11,
        }
    ]
    spec["exact_capacity_bytes"] = 11
    monkeypatch.setattr(v6.v5.v1, "fetch_exact_source", lambda _: b"print('x')\n")
    with pytest.raises(v6.CrossSourceV6Error, match="NumPy Git blob drift"):
        v6._materialize_numpy(spec)


def test_report_verifier_refuses_full_cpython_source_credit() -> None:
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
    with pytest.raises(v6.CrossSourceV6Error, match="CPython accepted capacity invalid"):
        v6.verify_report(report)
