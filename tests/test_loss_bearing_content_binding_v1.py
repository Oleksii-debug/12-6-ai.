from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from twelve_six.data.deterministic_exposure_order import (
    build_deterministic_exposure_plan,
)
from twelve_six.data.loss_bearing_content_binding_v1 import (
    build_loss_bearing_content_manifest,
    verify_live_loss_bearing_batch,
)
from twelve_six.data.unique_loss_ledger_v2 import LedgerError, build_ledger


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical_sha(value: object) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rehash_materialization(value: dict) -> None:
    payload = deepcopy(value)
    payload.pop("materialization_identity_sha256", None)
    value["materialization_identity_sha256"] = _canonical_sha(payload)


def _materialization() -> dict:
    value = {
        "schema_version": "12-6.postpack-loss-materialization.v2",
        "terminal_corpus_authority_identity_sha256": _sha("terminal-corpus"),
        "stage_bindings": {
            "normalization": _sha("normalization"),
            "evaluation_reservations": _sha("reservations"),
            "dedup": _sha("dedup"),
            "split": _sha("split"),
            "packing": _sha("packing-stage"),
        },
        "tokenizer": {
            "name": "fixture-tokenizer",
            "identity_sha256": _sha("tokenizer"),
            "source_bytes_are_loss_positions": False,
        },
        "documents": [
            {
                "document_id": "doc-a",
                "language": "uk",
                "modality": "text",
                "family_id": "fixture-family",
                "normalized_payload_sha256": _sha("payload-a"),
                "token_count": 5,
                "split": "train",
                "dedup_cluster_id": "cluster-a",
                "retained_after_dedup": True,
                "evaluation_reserved": False,
                "reserved_target_ranges": [],
                "eligible_target_ranges": [[1, 5]],
            }
        ],
        "packing": {
            "identity_sha256": _sha("packing-materialization"),
            "complete_one_pass": True,
            "packs": [
                {
                    "pack_id": "pack-0",
                    "token_count": 5,
                    "token_ids": [10, 11, 12, 13, 14],
                    "loss_spans": [
                        {
                            "document_id": "doc-a",
                            "target_start": 1,
                            "target_end": 5,
                            "pack_target_start": 1,
                        }
                    ],
                }
            ],
        },
    }
    _rehash_materialization(value)
    return value


def _authority() -> tuple[dict, dict, dict]:
    materialization = _materialization()
    ledger = build_ledger(materialization)
    segment = ledger["segments"][0]
    plan = build_deterministic_exposure_plan(
        [
            {
                "global_batch_index": 0,
                "shard_index": 0,
                "worker_id": 0,
                "claims": [
                    {
                        "segment_identity_sha256": segment[
                            "segment_identity_sha256"
                        ],
                        "offset_start": 0,
                        "offset_end": 4,
                    }
                ],
                "actual_nonignored_targets": 4,
            }
        ],
        num_workers=1,
        batches_per_shard=1,
        shard_count=1,
    )
    manifest = build_loss_bearing_content_manifest(
        materialization,
        ledger,
        plan,
        expected_materialization_identity_sha256=materialization[
            "materialization_identity_sha256"
        ],
        expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
        expected_plan_identity_sha256=plan["plan_identity_sha256"],
    )
    return materialization, ledger, manifest


def test_exact_aligned_and_shifted_live_targets_match_authority() -> None:
    _materialization_value, _ledger, manifest = _authority()
    root = manifest["manifest_identity_sha256"]

    aligned = verify_live_loss_bearing_batch(
        manifest,
        expected_manifest_identity_sha256=root,
        batch_index=0,
        target_ids=[[11, 12, 13, 14]],
        loss_mask=[[1, 1, 1, 1]],
    )
    shifted = verify_live_loss_bearing_batch(
        manifest,
        expected_manifest_identity_sha256=root,
        batch_index=0,
        target_ids=[[10, 11, 12, 13, 14]],
        shifted=True,
    )

    assert aligned == shifted == manifest["batches"][0][
        "batch_content_identity_sha256"
    ]
    assert manifest["loss_bearing_target_count"] == 4
    assert manifest["training_authorized_by_this_manifest"] is False


def test_same_cardinality_target_substitution_fails_closed() -> None:
    _materialization_value, _ledger, manifest = _authority()

    with pytest.raises(
        LedgerError,
        match="live loss-bearing target content differs from D04 content authority",
    ):
        verify_live_loss_bearing_batch(
            manifest,
            expected_manifest_identity_sha256=manifest[
                "manifest_identity_sha256"
            ],
            batch_index=0,
            target_ids=[[11, 12, 99, 14]],
            loss_mask=[[1, 1, 1, 1]],
        )


def test_effective_loss_mask_change_fails_closed() -> None:
    _materialization_value, _ledger, manifest = _authority()

    with pytest.raises(
        LedgerError,
        match="live loss-bearing target count differs from D04 content authority",
    ):
        verify_live_loss_bearing_batch(
            manifest,
            expected_manifest_identity_sha256=manifest[
                "manifest_identity_sha256"
            ],
            batch_index=0,
            target_ids=[[11, 12, 13, 14]],
            loss_mask=[[1, 1, 0, 1]],
        )


def test_resealed_pack_content_cannot_reuse_frozen_materialization_root() -> None:
    materialization, _ledger, _manifest = _authority()
    frozen_materialization_root = materialization["materialization_identity_sha256"]
    mutated = deepcopy(materialization)
    mutated["packing"]["packs"][0]["token_ids"][2] = 99
    _rehash_materialization(mutated)
    mutated_ledger = build_ledger(mutated)
    segment = mutated_ledger["segments"][0]
    mutated_plan = build_deterministic_exposure_plan(
        [
            {
                "global_batch_index": 0,
                "shard_index": 0,
                "worker_id": 0,
                "claims": [
                    {
                        "segment_identity_sha256": segment[
                            "segment_identity_sha256"
                        ],
                        "offset_start": 0,
                        "offset_end": 4,
                    }
                ],
                "actual_nonignored_targets": 4,
            }
        ],
        num_workers=1,
        batches_per_shard=1,
        shard_count=1,
    )

    with pytest.raises(
        LedgerError,
        match="materialization identity does not match expected handoff",
    ):
        build_loss_bearing_content_manifest(
            mutated,
            mutated_ledger,
            mutated_plan,
            expected_materialization_identity_sha256=frozen_materialization_root,
            expected_ledger_identity_sha256=mutated_ledger[
                "ledger_identity_sha256"
            ],
            expected_plan_identity_sha256=mutated_plan["plan_identity_sha256"],
        )


def test_coherently_resealed_manifest_cannot_replace_external_root() -> None:
    _materialization_value, _ledger, manifest = _authority()
    frozen_root = manifest["manifest_identity_sha256"]
    substituted = deepcopy(manifest)
    substituted["batches"][0]["claims"][0][
        "target_token_ids_sha256"
    ] = _sha("substituted-targets")
    claim = substituted["batches"][0]["claims"][0]
    claim_core = deepcopy(claim)
    claim_core.pop("claim_content_identity_sha256", None)
    claim["claim_content_identity_sha256"] = _canonical_sha(claim_core)
    batch = substituted["batches"][0]
    batch_core = deepcopy(batch)
    batch_core.pop("batch_content_identity_sha256", None)
    batch["batch_content_identity_sha256"] = _canonical_sha(batch_core)
    manifest_core = deepcopy(substituted)
    manifest_core.pop("manifest_identity_sha256", None)
    substituted["manifest_identity_sha256"] = _canonical_sha(manifest_core)

    with pytest.raises(
        LedgerError,
        match="loss-bearing content manifest differs from external authority",
    ):
        verify_live_loss_bearing_batch(
            substituted,
            expected_manifest_identity_sha256=frozen_root,
            batch_index=0,
            target_ids=[[11, 12, 13, 14]],
            loss_mask=[[1, 1, 1, 1]],
        )
