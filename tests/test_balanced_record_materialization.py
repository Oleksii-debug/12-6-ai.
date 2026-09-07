from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from twelve_six.data.balanced_record_materialization import (
    MATERIALIZATION_SCHEMA,
    SELECTION_POLICY,
    BalancedMaterializationError,
    BalancedRecord,
    materialize_balanced_record_set,
    record_graph_identity,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _record(
    record_id: str,
    text: str,
    *,
    family_id: str,
    stratum: str,
    cluster: str | None = None,
) -> BalancedRecord:
    payload = text.encode("utf-8")
    return BalancedRecord(
        record_id=record_id,
        text=text,
        source_id=f"source:{record_id}",
        family_id=family_id,
        stratum=stratum,
        language={"ua": "uk", "en": "en", "code": "code"}[stratum],
        modality="code" if stratum == "code" else "text",
        normalized_payload_sha256=hashlib.sha256(payload).hexdigest(),
        source_bytes=len(payload),
        dedup_cluster_id=cluster or f"cluster:{record_id}",
    )


def _allocation(
    family_id: str,
    stratum: str,
    allocated: int,
    available: int | None = None,
) -> dict:
    return {
        "family_id": family_id,
        "stratum": stratum,
        "allocated_bytes": allocated,
        "available_unique_bytes": allocated if available is None else available,
        "effective_family_cap_bytes": max(allocated, available or allocated),
    }


def _balance_result(
    allocations: list[dict],
    *,
    status: str = "PARTIAL_MIX_FEASIBLE_ACQUIRE_MORE_DATA",
) -> dict:
    maximum = sum(row["allocated_bytes"] for row in allocations)
    maximum_by_stratum = {
        stratum: sum(
            row["allocated_bytes"]
            for row in allocations
            if row["stratum"] == stratum
        )
        for stratum in ("ua", "en", "code")
    }
    body = {
        "schema_version": "12-6.next100-106-balance-gate-result.v1",
        "policy_identity_sha256": _sha("policy"),
        "dedup_authority": {
            "worker_id": "test-dedup",
            "head_sha": "a" * 40,
            "evidence_identity_sha256": _sha("dedup-evidence"),
            "terminal_verdict": "PASS",
        },
        "input_totals": {},
        "family_minimum": {},
        "maximum_feasible_total_source_bytes": maximum,
        "maximum_feasible_stratum_bytes": maximum_by_stratum,
        "target_total_source_bytes": 20_000_000,
        "target_stratum_bytes": {
            "ua": 9_000_000,
            "en": 7_000_000,
            "code": 4_000_000,
        },
        "raw_capacity_by_stratum": {},
        "raw_gap_to_target_by_stratum": {},
        "deterministic_maximum_allocation": allocations,
        "status": status,
        "next_step": "test",
        "claim_boundary": {
            "authorized_training_exposure_loss_positions": 0,
            "corpus_identity": None,
            "shard_identity": None,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "paid_compute_authorized": False,
            "source_bytes_are_loss_positions": False,
        },
    }
    payload = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        **body,
        "result_identity_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _bindings(balance_identity: str) -> dict[str, str]:
    return {
        "dedup": _sha("dedup"),
        "decontamination": _sha("decontamination"),
        "quality": _sha("quality"),
        "privacy": _sha("privacy"),
        "balance": balance_identity,
    }


def _materialize(records: list[BalancedRecord], balance: dict):
    identity = balance["result_identity_sha256"]
    return materialize_balanced_record_set(
        records,
        balance,
        expected_balance_result_identity_sha256=identity,
        stage_bindings=_bindings(identity),
    )


def test_exact_whole_record_allocation_is_deterministic_but_not_20m_terminal() -> None:
    records = [
        _record("ua-a", "a" * 4, family_id="family.ua", stratum="ua"),
        _record("ua-b", "b" * 6, family_id="family.ua", stratum="ua"),
        _record("en-a", "c" * 3, family_id="family.en", stratum="en"),
        _record("en-b", "d" * 7, family_id="family.en", stratum="en"),
        _record(
            "code-a",
            "e" * 5,
            family_id="family.code",
            stratum="code",
        ),
        _record(
            "code-b",
            "f" * 5,
            family_id="family.code",
            stratum="code",
        ),
    ]
    balance = _balance_result(
        [
            _allocation("family.ua", "ua", 10),
            _allocation("family.en", "en", 10),
            _allocation("family.code", "code", 10),
        ]
    )

    first, selected_first = _materialize(records, balance)
    second, selected_second = _materialize(list(reversed(records)), balance)

    assert first == second
    assert selected_first == selected_second
    assert first["schema_version"] == MATERIALIZATION_SCHEMA
    assert first["selection_policy"] == SELECTION_POLICY
    assert (
        first["post_policy_record_graph_identity_sha256"]
        == record_graph_identity(records)
    )
    assert first["selected_source_bytes"] == 30
    assert first["selected_source_bytes_by_stratum"] == {
        "ua": 10,
        "en": 10,
        "code": 10,
    }
    assert first["whole_record_granularity_gap_bytes"] == 0
    assert first["allocation_exact_at_record_boundaries"] is True
    assert first["terminal_for_learned20"] is False
    assert first["terminal_corpus_identity_sha256"] is None
    assert first["claim_boundary"]["training_authorized_loss_positions"] == 0


def test_whole_record_granularity_gap_fails_closed_without_truncation() -> None:
    records = [
        _record("ua-a", "a" * 6, family_id="family.ua", stratum="ua"),
        _record("ua-b", "b" * 6, family_id="family.ua", stratum="ua"),
        _record("en-a", "c" * 10, family_id="family.en", stratum="en"),
        _record(
            "code-a",
            "d" * 10,
            family_id="family.code",
            stratum="code",
        ),
    ]
    balance = _balance_result(
        [
            _allocation("family.ua", "ua", 10, available=12),
            _allocation("family.en", "en", 10),
            _allocation("family.code", "code", 10),
        ]
    )

    manifest, selected = _materialize(records, balance)

    assert manifest["selected_source_bytes"] == 26
    assert manifest["whole_record_granularity_gap_bytes"] == 4
    assert manifest["allocation_exact_at_record_boundaries"] is False
    assert manifest["terminal_for_learned20"] is False
    assert manifest["terminal_corpus_identity_sha256"] is None
    ua_selected = [
        record for record in selected if record.family_id == "family.ua"
    ]
    assert len(ua_selected) == 1
    assert ua_selected[0].source_bytes == 6


def test_allocated_family_capacity_must_equal_exact_record_graph() -> None:
    records = [
        _record(
            "ua-a",
            "a" * 12,
            family_id="family.ua",
            stratum="ua",
        )
    ]
    balance = _balance_result(
        [_allocation("family.ua", "ua", 10, available=11)]
    )
    with pytest.raises(
        BalancedMaterializationError,
        match="available_unique_bytes",
    ):
        _materialize(records, balance)


def test_balance_identity_and_stage_binding_are_fail_closed() -> None:
    record = _record(
        "ua-a",
        "a" * 10,
        family_id="family.ua",
        stratum="ua",
    )
    balance = _balance_result([_allocation("family.ua", "ua", 10)])
    tampered = dict(balance)
    tampered["status"] = "TARGET_20M_SOURCE_MIX_FEASIBLE"
    with pytest.raises(
        BalancedMaterializationError,
        match="self-identity mismatch",
    ):
        _materialize([record], tampered)

    identity = balance["result_identity_sha256"]
    bindings = _bindings(identity)
    bindings["balance"] = _sha("other-balance")
    with pytest.raises(
        BalancedMaterializationError,
        match="stage_bindings.balance",
    ):
        materialize_balanced_record_set(
            [record],
            balance,
            expected_balance_result_identity_sha256=identity,
            stage_bindings=bindings,
        )


def test_target_status_requires_exact_45_35_20_mix() -> None:
    record = _record(
        "ua-a",
        "a" * 10,
        family_id="family.ua",
        stratum="ua",
    )
    balance = _balance_result([_allocation("family.ua", "ua", 10)])
    body = dict(balance)
    body.pop("result_identity_sha256")
    body["status"] = "TARGET_20M_SOURCE_MIX_FEASIBLE"
    payload = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    forged = {
        **body,
        "result_identity_sha256": hashlib.sha256(payload).hexdigest(),
    }
    with pytest.raises(
        BalancedMaterializationError,
        match="TARGET_20M status",
    ):
        _materialize([record], forged)


def test_post_policy_record_flags_and_payload_identity_are_enforced() -> None:
    record = _record(
        "ua-a",
        "abcdefghij",
        family_id="family.ua",
        stratum="ua",
    )
    with pytest.raises(
        BalancedMaterializationError,
        match="evaluation-reserved",
    ):
        replace(record, evaluation_reserved=True)
    with pytest.raises(BalancedMaterializationError, match="quality gate"):
        replace(record, quality_pass=False)
    with pytest.raises(BalancedMaterializationError, match="privacy gate"):
        replace(record, privacy_pass=False)
    with pytest.raises(BalancedMaterializationError, match="payload hash"):
        replace(record, normalized_payload_sha256=_sha("wrong"))


def test_selected_dedup_cluster_cannot_replay_across_families() -> None:
    records = [
        _record(
            "ua-a",
            "a" * 10,
            family_id="family.ua",
            stratum="ua",
            cluster="cluster:shared",
        ),
        _record(
            "en-a",
            "b" * 10,
            family_id="family.en",
            stratum="en",
            cluster="cluster:shared",
        ),
    ]
    balance = _balance_result(
        [
            _allocation("family.ua", "ua", 10),
            _allocation("family.en", "en", 10),
        ]
    )
    with pytest.raises(
        BalancedMaterializationError,
        match="dedup cluster replay",
    ):
        _materialize(records, balance)
