from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from twelve_six.data.deterministic_exposure_order import validate_exposure_plan_preflight
from twelve_six.data.unique_loss_ledger_v2 import LedgerError, verify_ledger

CONTENT_MANIFEST_SCHEMA = "12-6.d04-loss-bearing-content-manifest.v2"
BATCH_CONTENT_SCHEMA = "12-6.d04-loss-bearing-batch-content.v2"
CLAIM_CONTENT_SCHEMA = "12-6.d04-loss-bearing-claim-content.v2"


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_obj(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise LedgerError(f"{label} must be a 64-hex SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise LedgerError(f"{label} must be a 64-hex SHA-256 string") from exc
    return value.lower()


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerError(f"{label} must be a non-negative integer")
    return value


def _require_token_ids(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LedgerError(f"{label} must be a sequence of token IDs")
    result: list[int] = []
    for index, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise LedgerError(f"{label}[{index}] must be a non-negative integer")
        result.append(token_id)
    return tuple(result)


def _materialization_pack_tokens(
    materialization: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    packing = materialization.get("packing")
    if not isinstance(packing, Mapping):
        raise LedgerError("materialization packing must be an object")
    packs = packing.get("packs")
    if not isinstance(packs, Sequence) or isinstance(packs, (str, bytes)):
        raise LedgerError("materialization packing.packs must be a sequence")

    result: dict[str, tuple[int, ...]] = {}
    for pack_index, pack in enumerate(packs):
        if not isinstance(pack, Mapping):
            raise LedgerError(f"packing.packs[{pack_index}] must be an object")
        pack_id = pack.get("pack_id")
        if not isinstance(pack_id, str) or not pack_id:
            raise LedgerError(f"packing.packs[{pack_index}].pack_id must be non-empty")
        if pack_id in result:
            raise LedgerError(f"duplicate pack_id: {pack_id}")
        token_count = _require_nonnegative_int(
            pack.get("token_count"), f"packing.packs[{pack_index}].token_count"
        )
        token_ids = _require_token_ids(
            pack.get("token_ids"), f"packing.packs[{pack_index}].token_ids"
        )
        if len(token_ids) != token_count:
            raise LedgerError(
                f"packing.packs[{pack_index}].token_ids length differs from token_count"
            )
        result[pack_id] = token_ids
    return result


def _ledger_segments(ledger: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_segments = ledger.get("segments")
    if not isinstance(raw_segments, Sequence) or isinstance(raw_segments, (str, bytes)):
        raise LedgerError("ledger segments must be a sequence")
    result: dict[str, Mapping[str, Any]] = {}
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, Mapping):
            raise LedgerError(f"ledger segments[{index}] must be an object")
        segment_id = _require_sha256(
            segment.get("segment_identity_sha256"),
            f"ledger segments[{index}].segment_identity_sha256",
        )
        if segment_id in result:
            raise LedgerError("duplicate ledger segment identity")
        result[segment_id] = segment
    return result


def _validated_plan(
    plan: Mapping[str, Any], *, expected_plan_identity_sha256: str
) -> tuple[Sequence[Mapping[str, Any]], str, int]:
    expected_identity = _require_sha256(
        expected_plan_identity_sha256, "expected_plan_identity_sha256"
    )
    if not isinstance(plan, Mapping):
        raise LedgerError("exposure_plan must be an object")
    observed_identity = _require_sha256(
        plan.get("plan_identity_sha256"), "plan_identity_sha256"
    )
    if observed_identity != expected_identity:
        raise LedgerError("exposure plan identity does not match expected handoff")
    unhashed = deepcopy(dict(plan))
    unhashed.pop("plan_identity_sha256", None)
    if _sha256_obj(unhashed) != observed_identity:
        raise LedgerError("exposure plan self-hash mismatch")
    batches = plan.get("batches")
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)):
        raise LedgerError("exposure plan batches must be a sequence")
    total = 0
    for index, batch in enumerate(batches):
        if not isinstance(batch, Mapping):
            raise LedgerError(f"exposure plan batches[{index}] must be an object")
        total += _require_nonnegative_int(
            batch.get("actual_nonignored_targets"),
            f"exposure plan batches[{index}].actual_nonignored_targets",
        )
    return batches, observed_identity, total


def _claim_sequences(
    claim: Mapping[str, Any],
    *,
    segments: Mapping[str, Mapping[str, Any]],
    pack_tokens: Mapping[str, tuple[int, ...]],
    label: str,
) -> tuple[str, int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    segment_id = _require_sha256(
        claim.get("segment_identity_sha256"), f"{label}.segment_identity_sha256"
    )
    segment = segments.get(segment_id)
    if segment is None:
        raise LedgerError(f"{label} references unknown ledger segment")
    offset_start = _require_nonnegative_int(
        claim.get("offset_start"), f"{label}.offset_start"
    )
    offset_end = _require_nonnegative_int(claim.get("offset_end"), f"{label}.offset_end")
    segment_length = _require_nonnegative_int(
        segment.get("loss_position_count"), f"{label}.segment_loss_position_count"
    )
    if offset_start >= offset_end or offset_end > segment_length:
        raise LedgerError(f"{label} interval is outside ledger segment")

    pack_id = segment.get("pack_id")
    if not isinstance(pack_id, str) or pack_id not in pack_tokens:
        raise LedgerError(f"{label} ledger segment references unknown pack")
    pack_target_start = _require_nonnegative_int(
        segment.get("pack_target_start"), f"{label}.pack_target_start"
    )
    absolute_start = pack_target_start + offset_start
    absolute_end = pack_target_start + offset_end
    tokens = pack_tokens[pack_id]
    if absolute_start < 1 or absolute_end > len(tokens):
        raise LedgerError(f"{label} target slice exceeds canonical pack token IDs")

    targets = tokens[absolute_start:absolute_end]
    aligned_inputs = tokens[absolute_start - 1 : absolute_end - 1]
    shifted_sequence = tokens[absolute_start - 1 : absolute_end]
    if len(aligned_inputs) != len(targets) or len(shifted_sequence) != len(targets) + 1:
        raise LedgerError(f"{label} canonical predictor/target geometry is inconsistent")
    return (
        pack_id,
        absolute_start,
        absolute_end,
        aligned_inputs,
        targets,
        shifted_sequence,
    )


def build_loss_bearing_content_manifest(
    materialization: Mapping[str, Any],
    ledger: Mapping[str, Any],
    exposure_plan: Mapping[str, Any],
    *,
    expected_materialization_identity_sha256: str,
    expected_ledger_identity_sha256: str,
    expected_plan_identity_sha256: str,
) -> dict[str, Any]:
    """Bind every planned gradient row to canonical predictor and target token content.

    Each exposure-plan claim becomes one canonical live row. For aligned Trainer
    batches the row is ``pack[start-1:end-1] -> pack[start:end]``. For the
    Trainer's ordinary shifted causal mode the exact same authority is represented
    by the full canonical sequence ``pack[start-1:end]``. Both representations
    are derived only from token IDs covered by the externally expected D04
    materialization, ledger and exposure-plan roots.

    The manifest stores digests/counts only and grants no training authority.
    """
    expected_materialization_identity = _require_sha256(
        expected_materialization_identity_sha256,
        "expected_materialization_identity_sha256",
    )
    observed_materialization_identity = _require_sha256(
        materialization.get("materialization_identity_sha256"),
        "materialization_identity_sha256",
    )
    if observed_materialization_identity != expected_materialization_identity:
        raise LedgerError("materialization identity does not match expected handoff")

    verify_ledger(materialization, ledger)
    expected_ledger_identity = _require_sha256(
        expected_ledger_identity_sha256, "expected_ledger_identity_sha256"
    )
    observed_ledger_identity = _require_sha256(
        ledger.get("ledger_identity_sha256"), "ledger_identity_sha256"
    )
    if observed_ledger_identity != expected_ledger_identity:
        raise LedgerError("ledger identity does not match expected handoff")

    batches, plan_identity, planned_targets = _validated_plan(
        exposure_plan,
        expected_plan_identity_sha256=expected_plan_identity_sha256,
    )
    validate_exposure_plan_preflight(
        exposure_plan,
        ledger,
        expected_ledger_identity_sha256=expected_ledger_identity,
        expected_unique_budget=planned_targets,
    )

    pack_tokens = _materialization_pack_tokens(materialization)
    segments = _ledger_segments(ledger)
    manifest_batches: list[dict[str, Any]] = []
    manifest_target_count = 0

    for batch_index, batch in enumerate(batches):
        claims = batch.get("claims")
        if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
            raise LedgerError(
                f"exposure plan batches[{batch_index}].claims must be a sequence"
            )
        claim_contents: list[dict[str, Any]] = []
        batch_target_count = 0

        for claim_index, claim in enumerate(claims):
            if not isinstance(claim, Mapping):
                raise LedgerError(
                    f"exposure plan batches[{batch_index}].claims[{claim_index}] "
                    "must be an object"
                )
            label = f"batches[{batch_index}].claims[{claim_index}]"
            (
                pack_id,
                absolute_start,
                absolute_end,
                aligned_inputs,
                targets,
                shifted_sequence,
            ) = _claim_sequences(
                claim,
                segments=segments,
                pack_tokens=pack_tokens,
                label=label,
            )
            claim_content: dict[str, Any] = {
                "schema_version": CLAIM_CONTENT_SCHEMA,
                "segment_identity_sha256": _require_sha256(
                    claim.get("segment_identity_sha256"),
                    f"{label}.segment_identity_sha256",
                ),
                "offset_start": _require_nonnegative_int(
                    claim.get("offset_start"), f"{label}.offset_start"
                ),
                "offset_end": _require_nonnegative_int(
                    claim.get("offset_end"), f"{label}.offset_end"
                ),
                "pack_id": pack_id,
                "pack_target_start": absolute_start,
                "pack_target_end": absolute_end,
                "target_count": len(targets),
                "aligned_input_token_count": len(aligned_inputs),
                "aligned_input_token_ids_sha256": _sha256_obj(list(aligned_inputs)),
                "target_token_ids_sha256": _sha256_obj(list(targets)),
                "shifted_sequence_token_count": len(shifted_sequence),
                "shifted_sequence_token_ids_sha256": _sha256_obj(
                    list(shifted_sequence)
                ),
            }
            claim_content["claim_content_identity_sha256"] = _sha256_obj(claim_content)
            claim_contents.append(claim_content)
            batch_target_count += len(targets)

        expected_batch_targets = _require_nonnegative_int(
            batch.get("actual_nonignored_targets"),
            f"exposure plan batches[{batch_index}].actual_nonignored_targets",
        )
        if batch_target_count != expected_batch_targets:
            raise LedgerError(
                "canonical claim content count differs from exposure plan batch"
            )
        batch_content: dict[str, Any] = {
            "schema_version": BATCH_CONTENT_SCHEMA,
            "global_batch_index": _require_nonnegative_int(
                batch.get("global_batch_index"),
                f"exposure plan batches[{batch_index}].global_batch_index",
            ),
            "actual_nonignored_targets": batch_target_count,
            "row_count": len(claim_contents),
            "row_semantics": "one_exposure_claim_per_live_row_v2",
            "claims": claim_contents,
        }
        batch_content["batch_content_identity_sha256"] = _sha256_obj(batch_content)
        manifest_batches.append(batch_content)
        manifest_target_count += batch_target_count

    if manifest_target_count != planned_targets:
        raise LedgerError("content manifest target count differs from exposure plan")

    tokenizer = materialization.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise LedgerError("materialization tokenizer must be an object")
    packing = materialization.get("packing")
    if not isinstance(packing, Mapping):
        raise LedgerError("materialization packing must be an object")

    manifest: dict[str, Any] = {
        "schema_version": CONTENT_MANIFEST_SCHEMA,
        "materialization_identity_sha256": observed_materialization_identity,
        "ledger_identity_sha256": observed_ledger_identity,
        "plan_identity_sha256": plan_identity,
        "tokenizer_identity_sha256": _require_sha256(
            tokenizer.get("identity_sha256"), "tokenizer.identity_sha256"
        ),
        "packing_identity_sha256": _require_sha256(
            packing.get("identity_sha256"), "packing.identity_sha256"
        ),
        "loss_bearing_target_count": manifest_target_count,
        "live_input_binding": "canonical_predictor_context_and_targets_v2",
        "batches": manifest_batches,
        "training_authorized_by_this_manifest": False,
    }
    manifest["manifest_identity_sha256"] = _sha256_obj(manifest)
    return manifest


def _validate_manifest(
    manifest: Mapping[str, Any], *, expected_manifest_identity_sha256: str
) -> Sequence[Mapping[str, Any]]:
    if not isinstance(manifest, Mapping):
        raise LedgerError("loss-bearing content manifest must be an object")
    if manifest.get("schema_version") != CONTENT_MANIFEST_SCHEMA:
        raise LedgerError("unsupported loss-bearing content manifest schema")
    expected_identity = _require_sha256(
        expected_manifest_identity_sha256, "expected_manifest_identity_sha256"
    )
    observed_identity = _require_sha256(
        manifest.get("manifest_identity_sha256"), "manifest_identity_sha256"
    )
    unhashed = deepcopy(dict(manifest))
    unhashed.pop("manifest_identity_sha256", None)
    if _sha256_obj(unhashed) != observed_identity:
        raise LedgerError("loss-bearing content manifest self-hash mismatch")
    if observed_identity != expected_identity:
        raise LedgerError("loss-bearing content manifest differs from external authority")
    if manifest.get("training_authorized_by_this_manifest") is not False:
        raise LedgerError("loss-bearing content manifest may not self-authorize training")
    if manifest.get("live_input_binding") != "canonical_predictor_context_and_targets_v2":
        raise LedgerError("loss-bearing content manifest lacks predictor/context authority")
    batches = manifest.get("batches")
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)):
        raise LedgerError("loss-bearing content manifest batches must be a sequence")
    return batches


def _rows(value: Any, label: str) -> list[list[int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LedgerError(f"{label} must be a rank-2 sequence/tensor")
    result: list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise LedgerError(f"{label}[{row_index}] must be a sequence")
        normalized: list[int] = []
        for column, token_id in enumerate(row):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise LedgerError(f"{label}[{row_index}][{column}] must be an integer")
            normalized.append(token_id)
        result.append(normalized)
    return result


def _mask_rows(value: Any, label: str) -> list[list[bool]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LedgerError(f"{label} must be a rank-2 sequence/tensor")
    result: list[list[bool]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise LedgerError(f"{label}[{row_index}] must be a sequence")
        normalized: list[bool] = []
        for column, raw in enumerate(row):
            if isinstance(raw, bool):
                normalized.append(raw)
            elif isinstance(raw, int) and not isinstance(raw, bool) and raw in (0, 1):
                normalized.append(bool(raw))
            else:
                raise LedgerError(
                    f"{label}[{row_index}][{column}] must be boolean or 0/1 integer"
                )
        result.append(normalized)
    return result


def verify_live_loss_bearing_batch(
    manifest: Mapping[str, Any],
    *,
    expected_manifest_identity_sha256: str,
    batch_index: int,
    input_ids: Any,
    target_ids: Any,
    loss_mask: Any | None = None,
    shifted: bool = False,
    ignore_index: int = -100,
) -> str:
    """Authenticate the exact model input and loss-bearing targets pre-mutation.

    The verifier intentionally has no target-only compatibility path. The
    predictor/context tensor changes model logits and therefore gradients, so it
    is part of the scientific authority. ``shifted=False`` matches Trainer
    ``target_ids`` mode. ``shifted=True`` matches ordinary causal LM mode where
    Trainer uses one full sequence as both input and labels and consumes columns
    ``1:``.
    """
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise LedgerError("batch_index must be a non-negative integer")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise LedgerError("ignore_index must be an integer")

    batches = _validate_manifest(
        manifest,
        expected_manifest_identity_sha256=expected_manifest_identity_sha256,
    )
    if batch_index >= len(batches):
        raise LedgerError("batch_index is outside loss-bearing content manifest")
    batch = batches[batch_index]
    if not isinstance(batch, Mapping) or batch.get("schema_version") != BATCH_CONTENT_SCHEMA:
        raise LedgerError("loss-bearing content batch is malformed")
    if batch.get("global_batch_index") != batch_index:
        raise LedgerError("loss-bearing content batch index is non-canonical")
    if batch.get("row_semantics") != "one_exposure_claim_per_live_row_v2":
        raise LedgerError("loss-bearing content batch row semantics are unsupported")

    inputs = _rows(input_ids, "input_ids")
    targets = _rows(target_ids, "target_ids")
    if len(inputs) != len(targets):
        raise LedgerError("input_ids and target_ids row counts must match")
    masks = None if loss_mask is None else _mask_rows(loss_mask, "loss_mask")
    if shifted and masks is not None:
        raise LedgerError("shifted target semantics do not accept a separate loss_mask")
    if masks is not None and len(masks) != len(targets):
        raise LedgerError("loss_mask row count must match target_ids")

    claims = batch.get("claims")
    if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
        raise LedgerError("loss-bearing content batch claims must be a sequence")
    expected_rows = _require_nonnegative_int(batch.get("row_count"), "row_count")
    if expected_rows != len(claims) or len(inputs) != expected_rows:
        raise LedgerError(
            "live row count differs from canonical D04 claim-row authority"
        )

    observed_claims: list[dict[str, Any]] = []
    total_targets = 0
    for claim_index, claim in enumerate(claims):
        if not isinstance(claim, Mapping) or claim.get("schema_version") != CLAIM_CONTENT_SCHEMA:
            raise LedgerError(f"loss-bearing content claim[{claim_index}] is malformed")

        target_count = _require_nonnegative_int(
            claim.get("target_count"), f"claims[{claim_index}].target_count"
        )
        input_row = inputs[claim_index]
        target_row = targets[claim_index]

        if shifted:
            expected_sequence_count = _require_nonnegative_int(
                claim.get("shifted_sequence_token_count"),
                f"claims[{claim_index}].shifted_sequence_token_count",
            )
            if len(input_row) != expected_sequence_count or len(target_row) != expected_sequence_count:
                raise LedgerError(
                    "live shifted input/target shape differs from D04 context authority"
                )
            expected_sequence_digest = _require_sha256(
                claim.get("shifted_sequence_token_ids_sha256"),
                f"claims[{claim_index}].shifted_sequence_token_ids_sha256",
            )
            if _sha256_obj(input_row) != expected_sequence_digest:
                raise LedgerError(
                    "live predictor/context input differs from D04 content authority"
                )
            if _sha256_obj(target_row) != expected_sequence_digest:
                raise LedgerError(
                    "live shifted labels differ from D04 content authority"
                )
            live_targets = tuple(target_row[1:])
            if any(token_id < 0 for token_id in live_targets):
                raise LedgerError(
                    "shifted live target token IDs must be non-negative"
                )
        else:
            expected_input_count = _require_nonnegative_int(
                claim.get("aligned_input_token_count"),
                f"claims[{claim_index}].aligned_input_token_count",
            )
            if len(input_row) != expected_input_count or len(target_row) != target_count:
                raise LedgerError(
                    "live aligned input/target shape differs from D04 context authority"
                )
            expected_input_digest = _require_sha256(
                claim.get("aligned_input_token_ids_sha256"),
                f"claims[{claim_index}].aligned_input_token_ids_sha256",
            )
            if _sha256_obj(input_row) != expected_input_digest:
                raise LedgerError(
                    "live predictor/context input differs from D04 content authority"
                )
            mask_row = None if masks is None else masks[claim_index]
            if mask_row is not None and len(mask_row) != len(target_row):
                raise LedgerError("loss_mask shape must match target_ids")
            live_values: list[int] = []
            for column, token_id in enumerate(target_row):
                if token_id == ignore_index:
                    continue
                if token_id < 0:
                    raise LedgerError(
                        "live target token IDs must be non-negative or ignore_index"
                    )
                if mask_row is not None and not mask_row[column]:
                    continue
                live_values.append(token_id)
            live_targets = tuple(live_values)

        if len(live_targets) != target_count:
            raise LedgerError(
                "live loss-bearing target count differs from D04 content authority"
            )
        observed_target_digest = _sha256_obj(list(live_targets))
        expected_target_digest = _require_sha256(
            claim.get("target_token_ids_sha256"),
            f"claims[{claim_index}].target_token_ids_sha256",
        )
        if observed_target_digest != expected_target_digest:
            raise LedgerError(
                "live loss-bearing target content differs from D04 content authority"
            )

        observed_claim: dict[str, Any] = {
            "schema_version": CLAIM_CONTENT_SCHEMA,
            "segment_identity_sha256": _require_sha256(
                claim.get("segment_identity_sha256"),
                f"claims[{claim_index}].segment_identity_sha256",
            ),
            "offset_start": _require_nonnegative_int(
                claim.get("offset_start"), f"claims[{claim_index}].offset_start"
            ),
            "offset_end": _require_nonnegative_int(
                claim.get("offset_end"), f"claims[{claim_index}].offset_end"
            ),
            "pack_id": claim.get("pack_id"),
            "pack_target_start": _require_nonnegative_int(
                claim.get("pack_target_start"), f"claims[{claim_index}].pack_target_start"
            ),
            "pack_target_end": _require_nonnegative_int(
                claim.get("pack_target_end"), f"claims[{claim_index}].pack_target_end"
            ),
            "target_count": target_count,
            "aligned_input_token_count": _require_nonnegative_int(
                claim.get("aligned_input_token_count"),
                f"claims[{claim_index}].aligned_input_token_count",
            ),
            "aligned_input_token_ids_sha256": _require_sha256(
                claim.get("aligned_input_token_ids_sha256"),
                f"claims[{claim_index}].aligned_input_token_ids_sha256",
            ),
            "target_token_ids_sha256": observed_target_digest,
            "shifted_sequence_token_count": _require_nonnegative_int(
                claim.get("shifted_sequence_token_count"),
                f"claims[{claim_index}].shifted_sequence_token_count",
            ),
            "shifted_sequence_token_ids_sha256": _require_sha256(
                claim.get("shifted_sequence_token_ids_sha256"),
                f"claims[{claim_index}].shifted_sequence_token_ids_sha256",
            ),
        }
        if not isinstance(observed_claim["pack_id"], str) or not observed_claim["pack_id"]:
            raise LedgerError(f"claims[{claim_index}].pack_id must be non-empty")
        observed_claim["claim_content_identity_sha256"] = _sha256_obj(observed_claim)
        if observed_claim["claim_content_identity_sha256"] != claim.get(
            "claim_content_identity_sha256"
        ):
            raise LedgerError("loss-bearing content claim identity mismatch")
        observed_claims.append(observed_claim)
        total_targets += target_count

    expected_total = _require_nonnegative_int(
        batch.get("actual_nonignored_targets"), "actual_nonignored_targets"
    )
    if total_targets != expected_total:
        raise LedgerError("live loss-bearing target count differs from D04 content authority")

    observed_batch: dict[str, Any] = {
        "schema_version": BATCH_CONTENT_SCHEMA,
        "global_batch_index": batch_index,
        "actual_nonignored_targets": total_targets,
        "row_count": len(observed_claims),
        "row_semantics": "one_exposure_claim_per_live_row_v2",
        "claims": observed_claims,
    }
    observed_batch["batch_content_identity_sha256"] = _sha256_obj(observed_batch)
    expected_batch_identity = _require_sha256(
        batch.get("batch_content_identity_sha256"), "batch_content_identity_sha256"
    )
    if observed_batch["batch_content_identity_sha256"] != expected_batch_identity:
        raise LedgerError("live loss-bearing batch content identity mismatch")
    return expected_batch_identity
