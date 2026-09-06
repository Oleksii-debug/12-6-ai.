from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from twelve_six.data.deterministic_exposure_order import (
    authorize_ordered_batch,
    build_deterministic_exposure_plan,
    ordered_next_exposure_identity,
)
from twelve_six.data.identity_safe_exposure_guard import IdentitySafeExposureReplayGuard
from twelve_six.data.unique_loss_ledger_v2 import LedgerError, build_ledger


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


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
        ).encode("utf-8")
    ).hexdigest()


def _guard() -> tuple[dict, IdentitySafeExposureReplayGuard]:
    materialization = {
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
                "document_id": "doc",
                "language": "uk",
                "modality": "text",
                "family_id": "family.uk",
                "normalized_payload_sha256": _sha("payload"),
                "source_bytes": 5,
                "token_count": 5,
                "split": "train",
                "dedup_cluster_id": "cluster-doc",
                "retained_after_dedup": True,
                "evaluation_reserved": False,
                "reserved_target_ranges": [],
                "eligible_target_ranges": [[1, 5]],
            }
        ],
        "packing": {
            "identity_sha256": _sha("packing"),
            "complete_one_pass": True,
            "packs": [
                {
                    "pack_id": "p0",
                    "token_count": 5,
                    "loss_spans": [
                        {
                            "document_id": "doc",
                            "target_start": 1,
                            "target_end": 5,
                            "pack_target_start": 1,
                        }
                    ],
                }
            ],
        },
    }
    materialization["materialization_identity_sha256"] = _identity(
        materialization, "materialization_identity_sha256"
    )
    ledger = build_ledger(materialization)
    guard = IdentitySafeExposureReplayGuard(
        ledger,
        expected_ledger_identity_sha256=ledger["ledger_identity_sha256"],
        authorized_budget=4,
        trainer_state_binding={
            "checkpoint_generation": "g000",
            "checkpoint_manifest_sha256": _sha("checkpoint"),
            "optimizer_step": 0,
            "trainer_nonignored_target_count": 0,
        },
    )
    return ledger, guard


def _claim(ledger: dict, start: int, end: int) -> dict:
    return {
        "segment_identity_sha256": ledger["segments"][0]["segment_identity_sha256"],
        "offset_start": start,
        "offset_end": end,
    }


def main() -> None:
    ledger, guard = _guard()
    plan = build_deterministic_exposure_plan(
        [
            {
                "global_batch_index": 0,
                "shard_index": 0,
                "worker_id": 0,
                "claims": [_claim(ledger, 0, 2)],
                "actual_nonignored_targets": 2,
            },
            {
                "global_batch_index": 1,
                "shard_index": 0,
                "worker_id": 1,
                "claims": [_claim(ledger, 2, 4)],
                "actual_nonignored_targets": 2,
            },
        ],
        num_workers=2,
        batches_per_shard=2,
        shard_count=1,
    )

    first = ordered_next_exposure_identity(guard, plan, batch_index=0)
    observed = authorize_ordered_batch(
        guard,
        plan,
        batch_index=0,
        expected_ordered_next_exposure_identity_sha256=first,
    )
    if observed != first or guard.claim_sequence != 1:
        raise SystemExit("first ordered authorization did not advance exactly once")

    second = ordered_next_exposure_identity(guard, plan, batch_index=1)
    before = guard.state_dict()
    try:
        authorize_ordered_batch(
            guard,
            plan,
            batch_index=1,
            expected_ordered_next_exposure_identity_sha256=_sha("wrong-handoff"),
        )
    except LedgerError as exc:
        if "does not match expected handoff" not in str(exc):
            raise
    else:
        raise SystemExit("wrong ordered handoff identity was accepted")
    if guard.state_dict() != before:
        raise SystemExit("failed ordered handoff mutated replay state")

    observed = authorize_ordered_batch(
        guard,
        plan,
        batch_index=1,
        expected_ordered_next_exposure_identity_sha256=second,
    )
    if observed != second:
        raise SystemExit("second ordered identity changed during authorization")
    if guard.claim_sequence != 2 or guard.consumed_loss_positions != 4:
        raise SystemExit("ordered exposure accounting did not close exactly")

    print("D04 ORDERED EXPOSURE RUNTIME: PASS")
    print(f"plan_identity_sha256={plan['plan_identity_sha256']}")
    print(f"consumed_loss_positions={guard.consumed_loss_positions}")


if __name__ == "__main__":
    main()
