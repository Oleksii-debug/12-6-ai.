from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from twelve_six.data.balanced_split_application_v1 import (
    BalancedSplitApplicationError,
    verify_balanced_selection,
)

PARENTS = {
    "inventory": "1" * 64,
    "decontam": "2" * 64,
    "dedup": "3" * 64,
    "policy": "4" * 64,
    "balance": "5" * 64,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _rehash(selection: dict[str, Any]) -> None:
    core = dict(selection)
    core.pop("balanced_selection_identity_sha256", None)
    selection["balanced_selection_identity_sha256"] = hashlib.sha256(
        _canonical_bytes(core)
    ).hexdigest()


def _selection() -> dict[str, Any]:
    specs = (
        ("r1", "s1", "family-a", "uk", "a"),
        ("r2", "s2", "family-b", "en", "b"),
        ("r3", "s3", "family-c", "code", "c"),
        ("r4", "s4", "family-d", "ua", "d"),
    )
    records = []
    for record_id, source_id, family, stratum, payload in specs:
        records.append(
            {
                "record_id": record_id,
                "source_id": source_id,
                "family": family,
                "stratum": stratum,
                "modality": "text",
                "payload_sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "payload_bytes": 1,
                "near_duplicate_cluster_id": f"cluster-{record_id}",
                "purpose": "pretraining_eligible",
                "training_eligible": False,
                "evaluation_eligible": False,
                "evaluation_reserved": False,
            }
        )
    selection: dict[str, Any] = {
        "schema": "12-6.d03-balanced-selection-authority.v1",
        "terminal": True,
        "status": "PASS",
        "balanced_selection_identity_sha256": "0" * 64,
        "retained_inventory_identity_sha256": PARENTS["inventory"],
        "decontamination_authority_sha256": PARENTS["decontam"],
        "dedup_authority_sha256": PARENTS["dedup"],
        "balance_policy_identity_sha256": PARENTS["policy"],
        "balance_result_identity_sha256": PARENTS["balance"],
        "records": records,
        "totals": {
            "record_count": 4,
            "source_bytes": 4,
            "family_source_bytes": {
                "family-a": 1,
                "family-b": 1,
                "family-c": 1,
                "family-d": 1,
            },
            "stratum_source_bytes": {"code": 1, "en": 1, "ua": 1, "uk": 1},
        },
        "claim_boundary": {
            "training_eligible": False,
            "evaluation_eligible": False,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "paid_compute_authorized": False,
            "final_test_outcomes_read": False,
            "authorized_optimized_target_exposure": 0,
        },
    }
    _rehash(selection)
    return selection


def _verify(selection: dict[str, Any]) -> None:
    verify_balanced_selection(
        selection,
        expected_selection_identity_sha256=selection[
            "balanced_selection_identity_sha256"
        ],
        expected_retained_inventory_identity_sha256=PARENTS["inventory"],
        expected_decontamination_authority_sha256=PARENTS["decontam"],
        expected_dedup_authority_sha256=PARENTS["dedup"],
        expected_balance_policy_identity_sha256=PARENTS["policy"],
        expected_balance_result_identity_sha256=PARENTS["balance"],
    )


def test_valid_strict_integer_aggregates_pass() -> None:
    _verify(_selection())


@pytest.mark.parametrize("alias", [4.0, "4", True, None])
def test_record_count_numeric_aliases_fail_closed(alias: Any) -> None:
    selection = _selection()
    selection["totals"]["record_count"] = alias
    _rehash(selection)
    with pytest.raises(BalancedSplitApplicationError, match="non-negative integer"):
        _verify(selection)


@pytest.mark.parametrize("alias", [4.0, "4", True, None])
def test_source_bytes_numeric_aliases_fail_closed(alias: Any) -> None:
    selection = _selection()
    selection["totals"]["source_bytes"] = alias
    _rehash(selection)
    with pytest.raises(BalancedSplitApplicationError, match="non-negative integer"):
        _verify(selection)


@pytest.mark.parametrize("field", ["family_source_bytes", "stratum_source_bytes"])
@pytest.mark.parametrize("alias", [1.0, True, "1", None])
def test_nested_byte_numeric_aliases_fail_closed(field: str, alias: Any) -> None:
    selection = _selection()
    key = next(iter(selection["totals"][field]))
    selection["totals"][field][key] = alias
    _rehash(selection)
    with pytest.raises(BalancedSplitApplicationError, match="non-negative integer"):
        _verify(selection)


@pytest.mark.parametrize("field", ["family_source_bytes", "stratum_source_bytes"])
def test_nested_byte_maps_must_be_mappings(field: str) -> None:
    selection = _selection()
    selection["totals"][field] = []
    _rehash(selection)
    with pytest.raises(BalancedSplitApplicationError, match="must be a mapping"):
        _verify(selection)
