from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "tools/verify_d03_nomis_free_execution_authority_v1.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("_d03_authority_test", VERIFIER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_closed_world_json_rejects_bool_for_integer_and_extra_flag() -> None:
    verifier = _load_verifier()
    with pytest.raises(verifier.AuthorityError):
        verifier._require_exact_json(
            {"authorized_training_exposure": False},
            {"authorized_training_exposure": 0},
            "typed counter",
        )
    with pytest.raises(verifier.AuthorityError):
        verifier._require_exact_json(
            {**verifier.DATA526_TRUTH_BOUNDARY, "corpus_released_by_alias": True},
            verifier.DATA526_TRUTH_BOUNDARY,
            "closed world",
        )


def test_closed_world_truth_boundary_rejects_positive_reseal() -> None:
    verifier = _load_verifier()
    mutated = dict(verifier.DATA526_TRUTH_BOUNDARY)
    mutated["corpus_released"] = True
    with pytest.raises(verifier.AuthorityError):
        verifier._require_exact_json(
            mutated,
            verifier.DATA526_TRUTH_BOUNDARY,
            "DATA526 truth boundary",
        )


def test_historical_direct_same_length_payload_reseal_is_rejected() -> None:
    verifier = _load_verifier()
    good_hash = verifier.sha(b"abc")
    source_rows = {
        "direct.source": {
            "source_family": "family",
            "modality": "en",
            "declared_capacity_bytes": 3,
            "comparison_payload_bytes": 3,
            "comparison_payload_sha256": good_hash,
        }
    }
    config = {
        "data213_normalized_artifact": {"sources": {}},
        "kmu_authority": {"sources": {}},
        "cpython_authority": {
            "source_id": "cpython.source",
            "accepted_normalized_sha256": [],
            "accepted_chunk_count": 0,
            "accepted_capacity_bytes": 0,
        },
    }
    good_inventory = {
        "records": [
            {
                "record_id": "direct.source",
                "source_id": "direct.source",
                "family": "family",
                "modality": "en",
                "payload_sha256": good_hash,
                "payload_bytes": 3,
            }
        ]
    }
    verifier._verify_historical_content_identity(good_inventory, source_rows, config)

    coherently_resized = {
        "records": [
            {
                **good_inventory["records"][0],
                "payload_sha256": verifier.sha(b"xyz"),
                "payload_bytes": 3,
            }
        ]
    }
    with pytest.raises(verifier.AuthorityError):
        verifier._verify_historical_content_identity(
            coherently_resized, source_rows, config
        )


def test_complete_survivor_projection_binds_row_hash_fields() -> None:
    verifier = _load_verifier()
    rows = {}
    for index, source_id in enumerate(verifier.CLUSTER):
        rows[source_id] = {
            "source_id": source_id,
            "source_family": "werkzeug",
            "modality": "code",
            "declared_capacity_bytes": 303 if source_id == verifier.SELECTED else 302,
            "verified_raw_sha256": f"{index + 1:064x}",
            "normalized_sha256": f"{index + 11:064x}",
            "stable_origin_id_sha256": f"{index + 21:064x}",
            "stable_object_id_sha256": f"{index + 31:064x}",
        }
    rows["other.source"] = {
        "source_id": "other.source",
        "source_family": "other",
        "modality": "en",
        "declared_capacity_bytes": 100,
        "verified_raw_sha256": "a" * 64,
        "normalized_sha256": "b" * 64,
        "stable_origin_id_sha256": "c" * 64,
        "stable_object_id_sha256": "d" * 64,
    }
    report = {
        "report_sha256": "e" * 64,
        "dedup_v3": {"report_sha256": "f" * 64},
        "source_vector": {
            "source_capacity_bytes_before_global_dedup": 1007,
            "duplicate_discount_bytes": 604,
        },
    }
    core = verifier._expected_survivor_core(report, rows)
    survivor = {**core, "survivor_authority_sha256": verifier.sha(verifier.canon(core))}
    mutated = dict(survivor)
    mutated_rows = [dict(row) for row in survivor["survivors"]]
    mutated_rows[0]["verified_raw_sha256"] = "0" * 64
    mutated["survivors"] = mutated_rows
    mutated_core = dict(mutated)
    mutated_core.pop("survivor_authority_sha256")
    mutated["survivor_authority_sha256"] = verifier.sha(verifier.canon(mutated_core))
    with pytest.raises(verifier.AuthorityError):
        verifier._require_exact_json(mutated, survivor, "complete survivor authority")
