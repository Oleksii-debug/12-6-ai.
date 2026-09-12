from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from twelve_six.data.deterministic_exposure_order import (
    validate_exposure_plan_preflight,
)
from twelve_six.data.unique_loss_ledger_v2 import LedgerError, verify_ledger

CONTENT_MANIFEST_SCHEMA = "12-6.d04-loss-bearing-content-manifest.v1"
BATCH_CONTENT_SCHEMA = "12-6.d04-loss-bearing-batch-content.v1"
CLAIM_CONTENT_SCHEMA = "12-6.d04-loss-bearing-claim-content.v1"


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


def _claim_target_token_ids(
    claim: Mapping[str, Any],
    *,
    segments: Mapping[str, Mapping[str, Any]],
    pack_tokens: Mapping[str, tuple[int, ...]],
    label: str,
) -> tuple[int, ...]:
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
    return tokens[absolute_start:absolute_end]


def build_loss_bearing_content_manifest(
    materialization: Mapping[str, Any],
    ledger: Mapping[str, Any],
    exposure_plan: Mapping[str, Any],
    *,
    expected_materialization_identity_sha256: str,
    expected_ledger_identity_sha256: str,
    expected_plan_identity_sha256: str,
) -> dict[str, Any]:
    """Bind each planned D04 exposure claim to exact canonical target token IDs.

    The materialization must carry exact ``packing.packs[*].token_ids``. Those IDs
    are already covered by the externally expected materialization identity. The
    deterministic ledger and exposure plan are independently root-bound as well.
    Runtime may therefore compare live loss-bearing targets against this manifest
    without trusting a caller-selected tensor fingerprint.
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
            raise LedgerError(f"exposure plan batches[{batch_index}].claims must be a sequence")
        claim_contents: list[dict[str, Any]] = []
        batch_target_count = 0
        for claim_index, claim in enumerate(claims):
            if not isinstance(claim, Mapping):
                raise LedgerError(
                    f"exposure plan batches[{batch_index}].claims[{claim_index}] "
                    "must be an object"
                )
            label = f"batches[{batch_index}].claims[{claim_index}]"
            target_ids = _claim_target_token_ids(
                claim,
                segments=segments,
                pack_tokens=pack_tokens,
                label=label,
            )
            claim_content = {
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
                "target_count": len(target_ids),
                "target_token_ids_sha256": _sha256_obj(list(target_ids)),
            }
            claim_content["claim_content_identity_sha256"] = _sha256_obj(claim_content)
            claim_contents.append(claim_content)
            batch_target_count += len(target_ids)

        expected_batch_targets = _require_nonnegative_int(
            batch.get("actual_nonignored_targets"),
            f"exposure plan batches[{batch_index}].actual_nonignored_targets",
        )
        if batch_target_count != expected_batch_targets:
            raise LedgerError("canonical claim content count differs from exposure plan batch")
        batch_content: dict[str, Any] = {
            "schema_version": BATCH_CONTENT_SCHEMA,
            "global_batch_index": _require_nonnegative_int(
                batch.get("global_batch_index"),
                f"exposure plan batches[{batch_index}].global_batch_index",
            ),
            "actual_nonignored_targets": batch_target_count,
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
    batches = manifest.get("batches")
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)):
        raise LedgerError("loss-bearing content manifest batches must be a sequence")
    return batches


def _tolist(value: Any, label: str) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LedgerError(f"{label} must be a rank-2 sequence/tensor")
    return value


def _flatten_live_targets(
    target_ids: Any,
    *,
    loss_mask: Any | None,
    shifted: bool,
    ignore_index: int,
) -> tuple[int, ...]:
    rows = _tolist(target_ids, "target_ids")
    mask_rows = None if loss_mask is None else _tolist(loss_mask, "loss_mask")
    if shifted and mask_rows is not None:
        raise LedgerError("shifted target semantics do not accept a separate loss_mask")
    if mask_rows is not None and len(mask_rows) != len(rows):
        raise LedgerError("loss_mask row count must match target_ids")

    flattened: list[int] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise LedgerError(f"target_ids[{row_index}] must be a sequence")
        row_values = list(row)
        start = 1 if shifted else 0
        mask_row = None
        if mask_rows is not None:
            raw_mask_row = mask_rows[row_index]
            if not isinstance(raw_mask_row, Sequence) or isinstance(
                raw_mask_row, (str, bytes)
            ):
                raise LedgerError(f"loss_mask[{row_index}] must be a sequence")
            mask_row = list(raw_mask_row)
            if len(mask_row) != len(row_values):
                raise LedgerError("loss_mask shape must match target_ids")
        for column in range(start, len(row_values)):
            token_id = row_values[column]
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise LedgerError("live target token IDs must be integers")
            if token_id == ignore_index:
                continue
            if token_id < 0:
                raise LedgerError("live target token IDs must be non-negative or ignore_index")
            if mask_row is not None:
                mask_value = mask_row[column]
                if isinstance(mask_value, bool):
                    enabled = mask_value
                elif isinstance(mask_value, int) and mask_value in (0, 1):
                    enabled = bool(mask_value)
                else:
                    raise LedgerError("loss_mask values must be boolean or 0/1 integers")
                if not enabled:
                    continue
            flattened.append(token_id)
    return tuple(flattened)


def verify_live_loss_bearing_batch(
    manifest: Mapping[str, Any],
    *,
    expected_manifest_identity_sha256: str,
    batch_index: int,
    target_ids: Any,
    loss_mask: Any | None = None,
    shifted: bool = False,
    ignore_index: int = -100,
) -> str:
    """Verify exact live loss-bearing targets before an optimizer mutation.

    ``shifted=False`` is for aligned ``target_ids``. ``shifted=True`` is for the
    full labels/input sequence where causal loss consumes columns ``1:``. The
    caller must use the same batch ordering as the externally bound D04 exposure
    plan. Any same-cardinality token substitution or effective mask change fails.
    """
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise LedgerError("batch_index must be a non-negative integer")
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

    live_targets = _flatten_live_targets(
        target_ids,
        loss_mask=loss_mask,
        shifted=shifted,
        ignore_index=ignore_index,
    )
    expected_total = _require_nonnegative_int(
        batch.get("actual_nonignored_targets"), "actual_nonignored_targets"
    )
    if len(live_targets) != expected_total:
        raise LedgerError("live loss-bearing target count differs from D04 content authority")

    claims = batch.get("claims")
    if not isinstance(claims, Sequence) or isinstance(claims, (str, bytes)):
        raise LedgerError("loss-bearing content batch claims must be a sequence")
    cursor = 0
    observed_claims: list[dict[str, Any]] = []
    for claim_index, claim in enumerate(claims):
        if not isinstance(claim, Mapping) or claim.get("schema_version") != CLAIM_CONTENT_SCHEMA:
            raise LedgerError(f"loss-bearing content claim[{claim_index}] is malformed")
        target_count = _require_nonnegative_int(
            claim.get("target_count"), f"claims[{claim_index}].target_count"
        )
        target_slice = live_targets[cursor : cursor + target_count]
        if len(target_slice) != target_count:
            raise LedgerError("live targets end before D04 claim content")
        observed_digest = _sha256_obj(list(target_slice))
        expected_digest = _require_sha256(
            claim.get("target_token_ids_sha256"),
            f"claims[{claim_index}].target_token_ids_sha256",
        )
        if observed_digest != expected_digest:
            raise LedgerError(
                "live loss-bearing target content differs from D04 content authority"
            )
        observed_claim = {
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
            "target_count": target_count,
            "target_token_ids_sha256": observed_digest,
        }
        observed_claim["claim_content_identity_sha256"] = _sha256_obj(observed_claim)
        if observed_claim["claim_content_identity_sha256"] != claim.get(
            "claim_content_identity_sha256"
        ):
            raise LedgerError("loss-bearing content claim identity mismatch")
        observed_claims.append(observed_claim)
        cursor += target_count
    if cursor != len(live_targets):
        raise LedgerError("live targets exceed D04 claim content")

    observed_batch: dict[str, Any] = {
        "schema_version": BATCH_CONTENT_SCHEMA,
        "global_batch_index": batch_index,
        "actual_nonignored_targets": len(live_targets),
        "claims": observed_claims,
    }
    observed_batch["batch_content_identity_sha256"] = _sha256_obj(observed_batch)
    expected_batch_identity = _require_sha256(
        batch.get("batch_content_identity_sha256"), "batch_content_identity_sha256"
    )
    if observed_batch["batch_content_identity_sha256"] != expected_batch_identity:
        raise LedgerError("live loss-bearing batch content identity mismatch")
    return expected_batch_identity
