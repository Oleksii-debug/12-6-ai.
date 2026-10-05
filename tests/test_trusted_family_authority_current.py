from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError
from twelve_six.data.postmaterialization_balance_projection_v1 import (
    build_postmaterialization_family_vector,
    verify_postmaterialization_family_vector,
)
from twelve_six.data.trusted_family_authority_current import (
    ADDED_FAMILY_SEMANTICS,
    SOURCE_AUTHORITY_FILES,
    TRUSTED_FAMILY_SEMANTICS,
    trusted_family_authority_root_sha256,
    trusted_family_projection,
)
from twelve_six.data.trusted_family_authority_v1 import (
    trusted_family_authority_root_sha256 as v1_root,
)
from twelve_six.data.trusted_family_authority_v1 import (
    trusted_family_projection as v1_projection,
)

NEW_FAMILIES = (
    "code.scipy.project",
    "github:Textualize/rich",
    "github:fastapi/fastapi",
    "github:fastapi/typer",
    "github:pandas-dev/pandas",
    "github:pydantic/pydantic",
)
OLD_FAMILIES = (
    "github:agronholm/anyio",
    "github:pallets/flask",
    "github:pytest-dev/pytest",
)
GIT_SHA = "1" * 40
MATERIALIZATION_SHA = "2" * 64
JSONL_SHA = "3" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _inventory(families: tuple[str, ...]) -> dict:
    rows = []
    for index, family in enumerate(families):
        record_id = f"record-{index:02d}"
        rows.append(
            {
                "record_id": record_id,
                "source_id": f"source-{index:02d}",
                "family": family,
                "modality": "code",
                "payload_sha256": hashlib.sha256(record_id.encode()).hexdigest(),
                "payload_bytes": 100 + index,
            }
        )
    rows.sort(key=lambda row: row["record_id"])
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in rows
    ]
    return {
        "schema_version": "12-6.data526-record-inventory.v1",
        "record_count": len(rows),
        "total_payload_bytes": sum(row["payload_bytes"] for row in rows),
        "record_inventory_digest_sha256": hashlib.sha256(
            _canonical(rows)
        ).hexdigest(),
        "payload_inventory_digest_sha256": hashlib.sha256(
            _canonical(payload_projection)
        ).hexdigest(),
        "records": rows,
    }


def _evidence(inventory: dict) -> dict:
    return {
        "schema_version": "12-6.d03-post-g05-g06-materialization.v1",
        "status": "MATERIALIZED_ZERO_CREDIT",
        "execution_profile": "LOCAL_FREE",
        "execution_head_sha": GIT_SHA,
        "materialization_identity_sha256": MATERIALIZATION_SHA,
        "materializer_implementation_git_blob_sha1": "4" * 40,
        "repeat_materialization_byte_identical": True,
        "remaining_materialization_blockers": [],
        "input": {},
        "result": {
            "record_payload_jsonl_sha256": JSONL_SHA,
            "record_count": inventory["record_count"],
            "total_payload_bytes": inventory["total_payload_bytes"],
            "source_object_count": len(
                {row["source_id"] for row in inventory["records"]}
            ),
            "record_inventory_digest_sha256": inventory[
                "record_inventory_digest_sha256"
            ],
            "payload_inventory_digest_sha256": inventory[
                "payload_inventory_digest_sha256"
            ],
        },
        "transform": {},
        "consumed_blockers": [],
        "truth_boundary": {
            "authorized_optimized_target_exposure": 0,
            "authorized_unique_loss_positions": 0,
            "current_corpus_eligible": False,
            "external_llm_or_api_used_for_data_or_intelligence": False,
            "final_test_outcomes_read": False,
            "foreign_pretrained_weights": False,
            "learned_weights_created": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "paid_compute_used": False,
            "tokenizer_fit_authorized": False,
            "training_authorized_bytes": 0,
            "training_executed": False,
        },
    }


def _build(families: tuple[str, ...]) -> dict:
    inventory = _inventory(families)
    evidence = _evidence(inventory)
    return build_postmaterialization_family_vector(
        inventory=inventory,
        materialization_evidence=evidence,
        expected_execution_head_sha=GIT_SHA,
        expected_materialization_identity_sha256=MATERIALIZATION_SHA,
        expected_result_jsonl_sha256=JSONL_SHA,
        expected_record_count=inventory["record_count"],
        expected_total_payload_bytes=inventory["total_payload_bytes"],
        expected_source_object_count=len(
            {row["source_id"] for row in inventory["records"]}
        ),
        expected_record_inventory_digest_sha256=inventory[
            "record_inventory_digest_sha256"
        ],
        expected_payload_inventory_digest_sha256=inventory[
            "payload_inventory_digest_sha256"
        ],
        source_git_sha=GIT_SHA,
    )


def test_v1_only_projection_and_root_remain_exact() -> None:
    assert trusted_family_projection(OLD_FAMILIES) == v1_projection(OLD_FAMILIES)
    assert trusted_family_authority_root_sha256(OLD_FAMILIES) == v1_root(
        OLD_FAMILIES
    )


def test_extended_authority_has_exact_six_code_families() -> None:
    assert set(ADDED_FAMILY_SEMANTICS) == set(NEW_FAMILIES)
    projection = trusted_family_projection(NEW_FAMILIES)
    assert [row["family"] for row in projection] == sorted(NEW_FAMILIES)
    assert all(row["stratum"] == "code" for row in projection)
    assert all(row["language"] == "und" for row in projection)
    assert all(row["modalities"] == ["code"] for row in projection)
    assert len(TRUSTED_FAMILY_SEMANTICS) == 27


def test_extended_family_identity_is_source_authority_blob_bound() -> None:
    for family in NEW_FAMILIES:
        meta = SOURCE_AUTHORITY_FILES[family]
        expected = hashlib.sha256(
            _canonical(
                {
                    "authority_git_blob_sha1": meta["blob_sha1"],
                    "source_family": family,
                    "canonical_stratum": "code",
                }
            )
        ).hexdigest()
        assert (
            TRUSTED_FAMILY_SEMANTICS[family]["source_family_identity_sha256"]
            == expected
        )


def test_source_authority_blob_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = "github:pydantic/pydantic"
    altered = copy.deepcopy(SOURCE_AUTHORITY_FILES[family])
    altered["blob_sha1"] = "0" * 40
    monkeypatch.setitem(SOURCE_AUTHORITY_FILES, family, altered)
    with pytest.raises(ValueError, match="Git blob drift"):
        trusted_family_projection([family])


def test_unknown_family_still_fails_closed() -> None:
    with pytest.raises(ValueError, match="absent from trusted current authority"):
        trusted_family_projection(["github:example/not-authorized"])


def test_postmaterialization_vector_accepts_current_extended_families() -> None:
    families = OLD_FAMILIES + NEW_FAMILIES
    vector = _build(families)
    assert vector["stratum_family_counts"] == {
        "code": len(families),
        "en": 0,
        "uk": 0,
    }
    assert {row["family"] for row in vector["families"]} == set(families)
    assert vector["trusted_family_authority_root_sha256"] == (
        trusted_family_authority_root_sha256(families)
    )
    assert verify_postmaterialization_family_vector(
        vector,
        expected_identity_sha256=vector["family_vector_identity_sha256"],
    ) == vector["family_vector_identity_sha256"]


def test_coherently_resealed_unknown_family_cannot_cross_projection() -> None:
    inventory = _inventory(NEW_FAMILIES)
    inventory["records"][0]["family"] = "github:example/not-authorized"
    inventory["record_inventory_digest_sha256"] = hashlib.sha256(
        _canonical(inventory["records"])
    ).hexdigest()
    evidence = _evidence(inventory)
    with pytest.raises(ProjectionError, match="trusted"):
        build_postmaterialization_family_vector(
            inventory=inventory,
            materialization_evidence=evidence,
            expected_execution_head_sha=GIT_SHA,
            expected_materialization_identity_sha256=MATERIALIZATION_SHA,
            expected_result_jsonl_sha256=JSONL_SHA,
            expected_record_count=inventory["record_count"],
            expected_total_payload_bytes=inventory["total_payload_bytes"],
            expected_source_object_count=inventory["record_count"],
            expected_record_inventory_digest_sha256=inventory[
                "record_inventory_digest_sha256"
            ],
            expected_payload_inventory_digest_sha256=inventory[
                "payload_inventory_digest_sha256"
            ],
            source_git_sha=GIT_SHA,
        )
