from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from twelve_six.data.unique_loss_ledger_v2 import (
    LEDGER_SCHEMA,
    ExposureReplayGuard,
    LedgerError,
    build_ledger,
    count_nonignored_targets,
    verify_ledger,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _identity(value: dict, field: str) -> str:
    payload = deepcopy(value)
    payload.pop(field, None)
    return hashlib.sha256(
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
    ).hexdigest()


def _materialization() -> dict:
    value = {
        "schema_version": "12-6.postpack-loss-materialization.v2",
        "stage_bindings": {
            "normalization": _sha("normalization"),
            "evaluation_reservations": _sha("reservations"),
            "dedup": _sha("dedup"),
            "split": _sha("split"),
            "packing": _sha("packing-stage"),
        },
        "tokenizer": {
            "name": "s0-byte-v1",
            "identity_sha256": _sha("tokenizer"),
            "source_bytes_are_loss_positions": False,
        },
        "documents": [
            {
                "document_id": "en-doc",
                "language": "en",
                "modality": "text",
                "family_id": "family.en",
                "normalized_payload_sha256": _sha("en-payload"),
                "source_bytes": 999999,
                "token_count": 6,
                "split": "train",
                "dedup_cluster_id": "cluster-en",
                "retained_after_dedup": True,
                "evaluation_reserved": False,
                "reserved_target_ranges": [[3, 4]],
                "eligible_target_ranges": [[1, 3], [4, 6]],
            },
            {
                "document_id": "code-doc",
                "language": "python",
                "modality": "code",
                "family_id": "github:example/code",
                "normalized_payload_sha256": _sha("code-payload"),
                "source_bytes": 3,
                "token_count": 5,
                "split": "train",
                "dedup_cluster_id": "cluster-code",
                "retained_after_dedup": True,
                "evaluation_reserved": False,
                "reserved_target_ranges": [],
                "eligible_target_ranges": [[1, 5]],
            },
            {
                "document_id": "dedup-loser",
                "language": "en",
                "modality": "text",
                "family_id": "family.en",
                "normalized_payload_sha256": _sha("loser"),
                "source_bytes": 1234,
                "token_count": 8,
                "split": "train",
                "dedup_cluster_id": "cluster-en",
                "retained_after_dedup": False,
                "evaluation_reserved": False,
                "reserved_target_ranges": [],
                "eligible_target_ranges": [],
            },
            {
                "document_id": "selection-doc",
                "language": "uk",
                "modality": "text",
                "family_id": "family.selection",
                "normalized_payload_sha256": _sha("selection"),
                "source_bytes": 8888,
                "token_count": 9,
                "split": "selection",
                "dedup_cluster_id": "cluster-selection",
                "retained_after_dedup": True,
                "evaluation_reserved": True,
                "reserved_target_ranges": [],
                "eligible_target_ranges": [],
            },
        ],
        "packing": {
            "identity_sha256": _sha("packing"),
            "complete_one_pass": True,
            "packs": [
                {
                    "pack_id": "p0",
                    "token_count": 8,
                    "loss_spans": [
                        {
                            "document_id": "en-doc",
                            "target_start": 1,
                            "target_end": 3,
                            "pack_target_start": 1,
                        },
                        {
                            "document_id": "en-doc",
                            "target_start": 4,
                            "target_end": 6,
                            "pack_target_start": 4,
                        },
                    ],
                },
                {
                    "pack_id": "p1",
                    "token_count": 8,
                    "loss_spans": [
                        {
                            "document_id": "code-doc",
                            "target_start": 1,
                            "target_end": 5,
                            "pack_target_start": 2,
                        }
                    ],
                },
            ],
        },
    }
    value["materialization_identity_sha256"] = _identity(
        value, "materialization_identity_sha256"
    )
    return value


def _binding(target_count: int = 0, generation: str = "g000") -> dict:
    return {
        "checkpoint_generation": generation,
        "checkpoint_manifest_sha256": _sha(f"checkpoint-{generation}"),
        "optimizer_step": 0 if target_count == 0 else 1,
        "trainer_nonignored_target_count": target_count,
    }


def test_builds_exact_text_and_code_ledger_without_counting_source_bytes() -> None:
    materialization = _materialization()
    ledger = build_ledger(materialization)
    assert ledger["schema_version"] == LEDGER_SCHEMA
    assert ledger["one_pass_unique_nonignored_causal_loss_positions"] == 8
    assert ledger["eligible_causal_targets_before_packing"] == 8
    assert ledger["eligible_targets_not_packed"] == 0
    assert ledger["by_language"] == {"en": 4, "python": 4}
    assert ledger["by_modality"] == {"code": 4, "text": 4}
    assert ledger["by_family"] == {
        "family.en": 4,
        "github:example/code": 4,
    }
    assert ledger["padding_loss_positions"] == 0
    assert ledger["cross_document_loss_positions"] == 0
    assert ledger["source_bytes_relabelled_as_loss_positions"] is False

    changed = deepcopy(materialization)
    changed["documents"][0]["source_bytes"] = 1
    changed["documents"][1]["source_bytes"] = 10**12
    changed["materialization_identity_sha256"] = _identity(
        changed, "materialization_identity_sha256"
    )
    changed_ledger = build_ledger(changed)
    assert changed_ledger["one_pass_unique_nonignored_causal_loss_positions"] == 8


def test_reservation_target_cannot_be_packed() -> None:
    materialization = _materialization()
    materialization["packing"]["packs"][0]["loss_spans"][0]["target_end"] = 4
    materialization["materialization_identity_sha256"] = _identity(
        materialization, "materialization_identity_sha256"
    )
    with pytest.raises(LedgerError, match="non-eligible/reserved"):
        build_ledger(materialization)


def test_duplicate_logical_target_across_packs_fails() -> None:
    materialization = _materialization()
    materialization["packing"]["packs"][1]["loss_spans"].append(
        {
            "document_id": "en-doc",
            "target_start": 1,
            "target_end": 2,
            "pack_target_start": 6,
        }
    )
    materialization["materialization_identity_sha256"] = _identity(
        materialization, "materialization_identity_sha256"
    )
    with pytest.raises(LedgerError, match="replayed during packing"):
        build_ledger(materialization)


def test_two_retained_training_docs_in_same_dedup_cluster_fail() -> None:
    materialization = _materialization()
    loser = materialization["documents"][2]
    loser["retained_after_dedup"] = True
    loser["eligible_target_ranges"] = [[1, 8]]
    materialization["materialization_identity_sha256"] = _identity(
        materialization, "materialization_identity_sha256"
    )
    with pytest.raises(LedgerError, match="share dedup cluster"):
        build_ledger(materialization)


def test_incomplete_packing_is_reportable_but_not_terminal_by_default() -> None:
    materialization = _materialization()
    materialization["packing"]["complete_one_pass"] = False
    materialization["packing"]["packs"][1]["loss_spans"] = []
    materialization["materialization_identity_sha256"] = _identity(
        materialization, "materialization_identity_sha256"
    )
    with pytest.raises(LedgerError, match="requires packing.complete_one_pass"):
        build_ledger(materialization)
    ledger = build_ledger(materialization, require_complete_one_pass=False)
    assert ledger["one_pass_unique_nonignored_causal_loss_positions"] == 4
    assert ledger["eligible_targets_not_packed"] == 4


def test_replay_guard_binds_exact_positions_loss_mask_and_resume() -> None:
    ledger = build_ledger(_materialization())
    guard = ExposureReplayGuard(
        ledger,
        authorized_budget=8,
        trainer_state_binding=_binding(),
    )
    first = ledger["segments"][0]
    second = ledger["segments"][1]
    guard.authorize_loss_mask(
        [
            {
                "segment_identity_sha256": first["segment_identity_sha256"],
                "offset_start": 0,
                "offset_end": first["loss_position_count"],
            },
            {
                "segment_identity_sha256": second["segment_identity_sha256"],
                "offset_start": 0,
                "offset_end": second["loss_position_count"],
            },
        ],
        loss_mask=[1, 1, 1, 1],
    )
    assert guard.consumed_loss_positions == 4

    with pytest.raises(LedgerError, match="replay/overlapping"):
        guard.authorize_batch(
            [
                {
                    "segment_identity_sha256": first["segment_identity_sha256"],
                    "offset_start": 0,
                    "offset_end": 1,
                }
            ],
            actual_nonignored_targets=1,
        )

    checkpoint_binding = _binding(target_count=4, generation="g050")
    guard.bind_checkpoint_state(checkpoint_binding)
    state = guard.state_dict()

    resumed = ExposureReplayGuard(
        ledger,
        authorized_budget=8,
        trainer_state_binding=_binding(),
    )
    resumed.load_state_dict(
        state, expected_trainer_state_binding=checkpoint_binding
    )
    assert resumed.consumed_loss_positions == 4

    bad_binding = dict(checkpoint_binding)
    bad_binding["checkpoint_generation"] = "g051"
    with pytest.raises(LedgerError, match="trainer/checkpoint state binding mismatch"):
        resumed.load_state_dict(state, expected_trainer_state_binding=bad_binding)


def test_guard_rejects_claim_count_different_from_actual_loss_mask() -> None:
    ledger = build_ledger(_materialization())
    guard = ExposureReplayGuard(
        ledger,
        authorized_budget=8,
        trainer_state_binding=_binding(),
    )
    segment = ledger["segments"][0]
    with pytest.raises(LedgerError, match="does not match actual"):
        guard.authorize_loss_mask(
            [
                {
                    "segment_identity_sha256": segment["segment_identity_sha256"],
                    "offset_start": 0,
                    "offset_end": 2,
                }
            ],
            loss_mask=[1, 0, 0],
        )
    assert guard.consumed_loss_positions == 0


def test_count_nonignored_targets_is_strict() -> None:
    assert count_nonignored_targets([[1, 0], [True, False], [1.0, 0.0]]) == 3
    with pytest.raises(LedgerError):
        count_nonignored_targets([2])


def test_verify_rebuild_rejects_tampered_aggregate() -> None:
    materialization = _materialization()
    ledger = build_ledger(materialization)
    verify_ledger(materialization, ledger)
    tampered = deepcopy(ledger)
    tampered["one_pass_unique_nonignored_causal_loss_positions"] += 1
    with pytest.raises(LedgerError, match="does not match deterministic rebuild"):
        verify_ledger(materialization, tampered)
