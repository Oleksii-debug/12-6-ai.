"""Bind a terminal balanced split to canonical loss-materialization documents.

This module is deliberately a narrow bridge.  It does not select records, define a
split, tokenize, pack, count unique loss positions, or authorize training.  It
re-verifies the already-canonical balanced-selection/split application and projects
the exact selected physical rows into ``LossMaterializationDocument`` objects for
the existing two-clean packing path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data.balanced_split_application_v1 import (
    verify_balanced_split_application,
)
from twelve_six.packing.loss_materialization import LossMaterializationDocument

_REQUIRED_STAGE_BINDINGS = (
    "normalization",
    "evaluation_reservations",
    "dedup",
    "split",
    "packing",
)
_HEX = frozenset("0123456789abcdef")


class SplitPostpackBindingError(ValueError):
    """Raised when split-to-postpack projection cannot be proven fail-closed."""


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise SplitPostpackBindingError(f"{field} must be exact lowercase SHA-256")
    return value


def _normalize_stage_bindings(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(_REQUIRED_STAGE_BINDINGS):
        raise SplitPostpackBindingError(
            "stage_bindings must contain exactly normalization, "
            "evaluation_reservations, dedup, split and packing"
        )
    return {
        name: _require_sha256(value[name], f"stage_bindings.{name}")
        for name in _REQUIRED_STAGE_BINDINGS
    }


def _record_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_field: str,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    if isinstance(rows, (str, bytes)):
        raise SplitPostpackBindingError(f"{label} must be a record sequence")
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise SplitPostpackBindingError(f"{label}[{index}] must be an object")
        record_id = row.get(id_field)
        if not isinstance(record_id, str) or not record_id:
            raise SplitPostpackBindingError(
                f"{label}[{index}].{id_field} must be non-empty text"
            )
        if record_id in result:
            raise SplitPostpackBindingError(f"duplicate {label} record id: {record_id}")
        result[record_id] = row
    return result


def _split_membership(
    application: Mapping[str, Any],
    *,
    selected_ids: set[str],
) -> tuple[set[str], set[str]]:
    split_family = application.get("split_family")
    if not isinstance(split_family, Mapping):
        raise SplitPostpackBindingError("split application split_family must be an object")

    train_raw = split_family.get("shared_train_record_ids")
    validation_raw = split_family.get("validation_union_record_ids")
    if (
        not isinstance(train_raw, Sequence)
        or isinstance(train_raw, (str, bytes))
        or not isinstance(validation_raw, Sequence)
        or isinstance(validation_raw, (str, bytes))
    ):
        raise SplitPostpackBindingError(
            "split-family train/validation memberships must be sequences"
        )

    train = set(train_raw)
    validation = set(validation_raw)
    if any(not isinstance(item, str) or not item for item in train | validation):
        raise SplitPostpackBindingError("split-family memberships contain invalid record ids")
    if len(train) != len(train_raw) or len(validation) != len(validation_raw):
        raise SplitPostpackBindingError("split-family memberships contain duplicate record ids")
    if train & validation:
        raise SplitPostpackBindingError("split-family train/validation memberships overlap")
    if train | validation != selected_ids:
        missing = sorted(selected_ids - (train | validation))
        extra = sorted((train | validation) - selected_ids)
        raise SplitPostpackBindingError(
            f"split-family membership does not cover selected records; "
            f"missing={missing!r} extra={extra!r}"
        )
    if not train or not validation:
        raise SplitPostpackBindingError("split-family train and validation sets must be non-empty")
    return train, validation


def build_split_bound_loss_documents(
    application: Mapping[str, Any],
    selection: Mapping[str, Any],
    raw_records: Sequence[Mapping[str, Any]],
    *,
    expected_selection_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
    expected_split_git_blob_sha1: str,
    variant_seeds: Sequence[str],
    validation_fraction: float,
    stage_bindings: Mapping[str, str],
) -> tuple[LossMaterializationDocument, ...]:
    """Project an authenticated terminal split into existing post-pack input rows.

    ``verify_balanced_split_application`` independently rebuilds the terminal split
    from the fresh balanced selection and exact physical rows.  Consequently a
    caller cannot substitute train/validation membership by coherently resealing
    only the split application.

    The returned documents remain data mechanics only.  Validation-union documents
    are wholly reserved from optimization, while only the canonical shared train
    core is marked ``split="train"``.  No tokenizer, packing, unique-loss, exposure,
    or training authority is granted here.
    """

    verify_balanced_split_application(
        application,
        selection,
        raw_records,
        expected_selection_identity_sha256=expected_selection_identity_sha256,
        expected_retained_inventory_identity_sha256=(
            expected_retained_inventory_identity_sha256
        ),
        expected_decontamination_authority_sha256=(
            expected_decontamination_authority_sha256
        ),
        expected_dedup_authority_sha256=expected_dedup_authority_sha256,
        expected_balance_policy_identity_sha256=expected_balance_policy_identity_sha256,
        expected_balance_result_identity_sha256=expected_balance_result_identity_sha256,
        expected_split_git_blob_sha1=expected_split_git_blob_sha1,
        variant_seeds=variant_seeds,
        validation_fraction=validation_fraction,
    )

    application_identity = _require_sha256(
        application.get("application_identity_sha256"),
        "application.application_identity_sha256",
    )
    normalized_stage_bindings = _normalize_stage_bindings(stage_bindings)
    if normalized_stage_bindings["split"] != application_identity:
        raise SplitPostpackBindingError(
            "stage_bindings.split must equal the authenticated terminal "
            "split-application identity"
        )

    selection_rows = selection.get("records")
    if not isinstance(selection_rows, Sequence) or isinstance(
        selection_rows, (str, bytes)
    ):
        raise SplitPostpackBindingError("selection.records must be a sequence")
    selected_by_id = _record_map(
        selection_rows,
        id_field="record_id",
        label="selection.records",
    )
    raw_by_id = _record_map(
        raw_records,
        id_field="record_id",
        label="raw_records",
    )
    if set(raw_by_id) != set(selected_by_id):
        raise SplitPostpackBindingError(
            "physical raw-record membership differs from balanced selection"
        )

    train_ids, validation_ids = _split_membership(
        application,
        selected_ids=set(selected_by_id),
    )

    documents: list[LossMaterializationDocument] = []
    for record_id in sorted(selected_by_id):
        selected = selected_by_id[record_id]
        raw = raw_by_id[record_id]
        text = raw.get("normalized_payload")
        if not isinstance(text, str) or not text:
            raise SplitPostpackBindingError(
                f"{record_id}: normalized_payload must be non-empty text"
            )

        stratum = selected.get("stratum")
        if not isinstance(stratum, str) or not stratum:
            raise SplitPostpackBindingError(f"{record_id}: stratum must be non-empty text")
        cluster_id = selected.get("near_duplicate_cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            raise SplitPostpackBindingError(
                f"{record_id}: near_duplicate_cluster_id must be non-empty text"
            )

        is_train = record_id in train_ids
        if not is_train and record_id not in validation_ids:
            raise SplitPostpackBindingError(
                f"{record_id}: record is outside authenticated split membership"
            )
        documents.append(
            LossMaterializationDocument(
                document_id=record_id,
                text=text,
                source_id=selected["source_id"],
                language=stratum,
                modality=selected["modality"],
                family_id=selected["family"],
                normalized_payload_sha256=selected["payload_sha256"],
                source_bytes=selected["payload_bytes"],
                split="train" if is_train else "validation",
                dedup_cluster_id=cluster_id,
                retained_after_dedup=True,
                evaluation_reserved=not is_train,
                reserved_target_ranges=(),
            )
        )

    return tuple(documents)
