from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from twelve_six.data import cross_source_capacity_audit_v5 as v5
from twelve_six.data.pipeline import _quality_reason, normalize_text

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/next100_065c_cross_source_dedup_v5.json"


def _synthetic_cpython_spec(raw: bytes) -> dict[str, object]:
    normalized = normalize_text(raw.decode("utf-8"))
    chunks = v5._chunk_text(normalized, max_chars=1200, min_chars=80)
    policy = {
        "min_chars": 60,
        "max_chars": 1600,
        "min_alpha_ratio": 0.35,
        "reject_control_characters": True,
        "reject_email": True,
        "reject_phone": True,
    }
    quality_spec = {"quality_policy": policy}
    config = v5._cpython_quality_config(quality_spec)
    accepted: list[str] = []
    reasons: Counter[str] = Counter()
    for chunk in chunks:
        reason = _quality_reason(chunk, config)
        if reason is not None:
            reasons[reason] += 1
            continue
        accepted.append(v5._sha256(normalize_text(chunk).encode("utf-8")))
    return {
        "source_id": "synthetic-cpython",
        "normalization_policy": v5.CPYTHON_POLICY,
        "dedicated_workflow_conclusion": "success",
        "training": "ALLOWED_ACCEPTED_CHUNKS_ONLY",
        "evaluation": "NOT_SEPARATELY_ADMITTED",
        "acquisition_url": "memory://synthetic-cpython",
        "raw_bytes": len(raw),
        "raw_sha256": v5._sha256(raw),
        "git_blob_sha1": v5._git_blob_sha1(raw),
        "truncate_chars": 50000,
        "normalized_source_bytes": len(normalized.encode("utf-8")),
        "normalized_source_sha256": v5._sha256(normalized.encode("utf-8")),
        "chunking": {"max_chars": 1200, "min_chars": 80},
        "quality_policy": policy,
        "chunk_count": len(chunks),
        "accepted_chunk_count": len(accepted),
        "accepted_normalized_sha256": accepted,
        "rejected_chunk_count": sum(reasons.values()),
        "rejection_reasons": dict(sorted(reasons.items())),
    }


def test_static_v5_contract_accepts_committed_exact_vector() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    v5._validate_config(config)


def test_static_v5_contract_rejects_family_vector_drift() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["expected_vector"]["source_family_counts"]["en"] = 3
    with pytest.raises(v5.CrossSourceV5Error, match="family vector drift"):
        v5._validate_config(config)


def test_mdn_prose_normalization_removes_code_media_and_destinations() -> None:
    raw = b"""---\ntitle: Demo\n---\n# Compression Guide\n\nUse [HTTP compression](https://example.invalid) with `Content-Encoding`.\n\n```js\nsecretCall()\n```\n\n![diagram](image.png)\n\nNormal prose remains.\n"""
    normalized, stats = v5._normalize_mdn_prose(raw)
    text = normalized.decode("utf-8")
    assert "Compression Guide" in text
    assert "HTTP compression" in text
    assert "example.invalid" not in text
    assert "Content-Encoding" not in text
    assert "secretCall" not in text
    assert "diagram" not in text
    assert "Normal prose remains." in text
    assert stats["fenced_code_lines_removed"] == 3
    assert stats["image_lines_removed"] == 1
    assert stats["inline_code_spans_removed"] == 1


def test_cpython_materialization_excludes_privacy_rejected_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    clean = ("alpha language model training evidence and documentation " * 17).strip()
    rejected = (("beta public technical documentation and examples " * 17) + " call +380 50 123 4567").strip()
    raw = f"{clean}\n\n{rejected}\n".encode("utf-8")
    spec = _synthetic_cpython_spec(raw)
    assert spec["chunk_count"] == 2
    assert spec["accepted_chunk_count"] == 1
    assert spec["rejected_chunk_count"] == 1
    assert spec["rejection_reasons"] == {"pii_phone": 1}
    monkeypatch.setattr(v5.v1, "fetch_exact_source", lambda _: raw)

    payload, capacity, evidence = v5._materialize_cpython(spec)

    assert b"380 50 123 4567" not in payload
    assert capacity < spec["normalized_source_bytes"]
    assert evidence["accepted_chunk_count"] == 1
    assert evidence["rejected_chunk_count"] == 1
    assert evidence["rejection_reasons"] == {"pii_phone": 1}


def test_cpython_materialization_rejects_accepted_hash_rebinding(monkeypatch: pytest.MonkeyPatch) -> None:
    clean = ("alpha deterministic accepted technical document " * 20).strip()
    rejected = (("beta deterministic rejected technical document " * 20) + " +380 50 123 4567").strip()
    raw = f"{clean}\n\n{rejected}\n".encode("utf-8")
    spec = _synthetic_cpython_spec(raw)
    spec["accepted_normalized_sha256"] = ["0" * 64]
    monkeypatch.setattr(v5.v1, "fetch_exact_source", lambda _: raw)
    with pytest.raises(v5.CrossSourceV5Error, match="accepted identity/order drift"):
        v5._materialize_cpython(spec)


def test_report_verifier_refuses_full_cpython_source_credit() -> None:
    report = {
        "schema_version": v5.SCHEMA,
        "worker_id": v5.WORKER_ID,
        "local_free_only": True,
        "model_training_executed": False,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "raw_text_emitted": False,
        "source_vector": {
            "source_object_count": 23,
            "source_family_counts": {"code": 4, "en": 4, "uk": 4},
            "cpython_accepted_capacity_bytes": 17901,
            "source_capacity_bytes_before_global_dedup": 338533,
        },
        "materialization_evidence": [],
        "claim_boundary": {},
        "dedup_v3": {},
    }
    core = dict(report)
    report["report_sha256"] = v5._sha256(v5._canonical_bytes(core))
    with pytest.raises(v5.CrossSourceV5Error, match="accepted capacity invalid"):
        v5.verify_report(report)
