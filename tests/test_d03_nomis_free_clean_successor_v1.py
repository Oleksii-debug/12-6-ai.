from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run_d03_nomis_free_clean_successor_v1.py"
SPEC = importlib.util.spec_from_file_location("swarm2065_clean_successor", TOOL)
assert SPEC is not None and SPEC.loader is not None
clean = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clean)


class _QuarantineStub:
    def __init__(self, blocked_sha: str) -> None:
        self.blocked_sha = blocked_sha
        self.rejections = 0

    def validate_authority(self, authority):
        assert authority == {"authority": "ok"}
        return clean.EXPECTED_QUARANTINE_IDENTITY

    def reject_quarantined_inventory_rows(self, rows, authority):
        assert authority == {"authority": "ok"}
        for row in rows:
            if row["payload_sha256"] == self.blocked_sha:
                self.rejections += 1
                raise RuntimeError("blocked payload survived")
        return clean.EXPECTED_QUARANTINE_IDENTITY


def _synthetic_graph(blocked_payload: bytes):
    other_total = 2_213_956
    rows = []
    payloads = {}
    for index in range(34):
        source_id = f"clean-{index:02d}"
        payload = b"x" if index else b"x" * (other_total - 33)
        rows.append(
            {
                "source_id": source_id,
                "source_family": f"family-{index:02d}",
                "modality": "en",
                "declared_capacity_bytes": len(payload),
            }
        )
        payloads[source_id] = payload
    rows.insert(
        7,
        {
            "source_id": clean.BLOCKED_SOURCE_ID,
            "source_family": clean.BLOCKED_FAMILY,
            "modality": "uk",
            "declared_capacity_bytes": len(blocked_payload),
        },
    )
    payloads[clean.BLOCKED_SOURCE_ID] = blocked_payload
    return {"sources": rows}, payloads


def test_current_main_runtime_dependencies_are_exactly_blob_bound() -> None:
    clean.validate_runtime_bindings(ROOT)


def test_truth_boundary_cannot_claim_training_progress() -> None:
    assert clean.TRUTH_BOUNDARY["tokenizer_fit_authorized"] is False
    assert clean.TRUTH_BOUNDARY["authorized_training_exposure"] == 0
    assert clean.TRUTH_BOUNDARY["model_training_executed"] is False
    assert clean.TRUTH_BOUNDARY["optimizer_updates"] == 0
    assert clean.TRUTH_BOUNDARY["learned_weights_created"] is False
    assert clean.TRUTH_BOUNDARY["paid_compute_used"] is False


def test_exact_quarantine_is_removed_before_matcher_without_collateral_drift() -> None:
    blocked_payload = b"N" * clean.BLOCKED_BYTES
    blocked_sha = hashlib.sha256(blocked_payload).hexdigest()
    inventory, payloads = _synthetic_graph(blocked_payload)
    before_ids = [row["source_id"] for row in inventory["sources"]]
    stub = _QuarantineStub(blocked_sha)

    with mock.patch.object(clean, "BLOCKED_SHA256", blocked_sha):
        clean_inventory, clean_payloads, proof = clean.deauthorize_exact_nomis(
            inventory,
            payloads,
            {"authority": "ok"},
            stub,
        )

    after_ids = [row["source_id"] for row in clean_inventory["sources"]]
    assert after_ids == [
        source_id for source_id in before_ids if source_id != clean.BLOCKED_SOURCE_ID
    ]
    assert clean.BLOCKED_SOURCE_ID not in clean_payloads
    assert set(clean_payloads) == set(after_ids)
    assert proof["removed_before_new_global_dedup"] is True
    assert proof["pre_source_object_count"] == 35
    assert proof["post_source_object_count"] == 34
    assert proof["pre_source_capacity_bytes"] == 2_215_615
    assert proof["post_source_capacity_bytes"] == 2_213_956


def test_quarantine_source_hash_drift_fails_closed() -> None:
    blocked_payload = b"N" * clean.BLOCKED_BYTES
    inventory, payloads = _synthetic_graph(blocked_payload)
    stub = _QuarantineStub("f" * 64)
    with pytest.raises(clean.CleanSuccessorError, match="physical SHA-256 drift"):
        clean.deauthorize_exact_nomis(
            inventory,
            payloads,
            {"authority": "ok"},
            stub,
        )


def test_missing_quarantined_root_fails_closed() -> None:
    blocked_payload = b"N" * clean.BLOCKED_BYTES
    inventory, payloads = _synthetic_graph(blocked_payload)
    inventory["sources"] = [
        row for row in inventory["sources"] if row["source_id"] != clean.BLOCKED_SOURCE_ID
    ]
    payloads.pop(clean.BLOCKED_SOURCE_ID)
    stub = _QuarantineStub(hashlib.sha256(blocked_payload).hexdigest())
    with pytest.raises(
        clean.CleanSuccessorError,
        match="exactly one authenticated Nomis source is required",
    ):
        clean.deauthorize_exact_nomis(
            inventory,
            payloads,
            {"authority": "ok"},
            stub,
        )


def test_duplicate_quarantined_root_fails_closed() -> None:
    blocked_payload = b"N" * clean.BLOCKED_BYTES
    inventory, payloads = _synthetic_graph(blocked_payload)
    inventory["sources"].append(dict(inventory["sources"][7]))
    stub = _QuarantineStub(hashlib.sha256(blocked_payload).hexdigest())
    with mock.patch.object(clean, "BLOCKED_SHA256", stub.blocked_sha), pytest.raises(
        clean.CleanSuccessorError,
        match="exactly one authenticated Nomis source is required",
    ):
        clean.deauthorize_exact_nomis(
            inventory,
            payloads,
            {"authority": "ok"},
            stub,
        )


def test_clean_composed_oracle_is_exact_and_zero_credit() -> None:
    assert clean.EXPECTED_CLEAN_COMPOSED == {
        "source_object_count": 263,
        "source_capacity_bytes_before_global_dedup": 6_093_965,
        "source_family_counts": {"uk": 3, "en": 5, "code": 12},
        "conservative_unique_capacity_bytes_after_global_dedup": 6_093_662,
        "duplicate_discount_bytes": 303,
        "duplicate_cluster_count": 1,
        "post_dedup_survivor_source_object_count": 261,
    }
