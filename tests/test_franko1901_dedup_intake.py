from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six.data import franko1901_dedup_intake as mod


def _canonical_line(row: dict[str, object]) -> bytes:
    return mod._materializer_canonical_json_bytes(row)


def _row(raw_index: int, text: str) -> dict[str, object]:
    payload = text.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "language": mod.SOURCE_LANGUAGE,
        "modality": mod.SOURCE_CANDIDATE_MODALITY,
        "record_id": f"franko1901:{raw_index:06d}:{digest[:16]}",
        "source_family": mod.SOURCE_FAMILY,
        "source_id": mod.SOURCE_ID,
        "text": text,
        "text_sha256": digest,
        "text_utf8_bytes": len(payload),
    }


def _inventory_sha(rows: list[dict[str, object]]) -> str:
    projection = [
        {
            "record_id": row["record_id"],
            "text_sha256": row["text_sha256"],
            "text_utf8_bytes": row["text_utf8_bytes"],
        }
        for row in rows
    ]
    return mod._sha256(mod._materializer_canonical_json_bytes(projection))


def _historical(authority: mod.FrankoAuthority) -> dict[str, object]:
    return {
        "schema_version": mod.HISTORICAL_TERMINAL_SCHEMA,
        "decision": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        "execution": {
            "evidence_pr": 855,
            "execution_head_sha": authority.historical_execution_head,
            "workflow_run_id": authority.historical_workflow_run_id,
            "job_id": authority.historical_workflow_job_id,
            "execution_profile": "LOCAL_FREE",
        },
        "executed_product_identity": {
            "materializer_git_blob_sha1": "0" * 40,
        },
        "source": {
            "source_id": mod.SOURCE_ID,
            "source_family": mod.SOURCE_FAMILY,
            "upstream_repository": mod.SOURCE_REPOSITORY,
            "upstream_revision": mod.SOURCE_REVISION,
            "source_path": mod.SOURCE_PATH,
            "source_git_blob_sha1": mod.SOURCE_GIT_BLOB_SHA1,
            "source_bytes": mod.SOURCE_BYTES,
            "source_sha256": mod.SOURCE_SHA256,
            "license_id": mod.LICENSE_ID,
        },
        "materialization": {
            "rows_seen": authority.source_rows_seen,
            "accepted_rows": authority.candidate_records,
            "rejected_rows": authority.source_rows_seen - authority.candidate_records,
            "accepted_text_utf8_bytes": authority.candidate_text_utf8_bytes,
            "accepted_payload_jsonl_bytes": authority.candidate_jsonl_bytes,
            "accepted_payload_jsonl_sha256": authority.candidate_sha256,
            "record_inventory_identity_sha256": authority.record_inventory_sha256,
        },
        "truth_boundary": {
            "candidate_only": True,
            "canonical_capacity_credit_bytes": 0,
            "family_credit_authorized": False,
            "corpus_admitted": False,
            "global_dedup": "NOT_RUN",
            "evaluation_decontamination": "NOT_RUN",
            "post_composition_quality_privacy": "NOT_RUN",
            "balance_family_caps": "NOT_RUN_FOR_THIS_ADDITION",
            "tokenizer_fit_authorized": False,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "final_test_accessed": False,
            "paid_compute_used": False,
            "learned_20m_promoted": False,
        },
    }


def _fresh(authority: mod.FrankoAuthority) -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": mod.FRESH_AUTHORITY_SCHEMA,
        "status": mod.FRESH_AUTHORITY_STATUS,
        "execution_profile": "LOCAL_FREE",
        "product": {
            "pr": authority.product_pr,
            "exact_head": authority.product_head,
            "shared_exact_head_ci_run": authority.shared_ci_run,
        },
        "execution": {
            "workflow_run_id": authority.fresh_workflow_run_id,
            "workflow_job_id": authority.fresh_workflow_job_id,
            "artifact_id": authority.fresh_artifact_id,
            "receipt_member_sha256": authority.fresh_receipt_member_sha256,
            "conclusion": "success",
        },
        "independent_audit": {
            "issue": authority.audit_issue,
            "verdict": authority.audit_verdict,
        },
        "reproduced_candidate": {
            "accepted_rows": authority.candidate_records,
            "accepted_text_utf8_bytes": authority.candidate_text_utf8_bytes,
            "accepted_payload_jsonl_bytes": authority.candidate_jsonl_bytes,
            "accepted_payload_jsonl_sha256": authority.candidate_sha256,
            "record_inventory_identity_sha256": authority.record_inventory_sha256,
        },
        "historical_terminal_link": {
            "schema_version": mod.HISTORICAL_TERMINAL_SCHEMA,
            "git_blob_sha1": authority.historical_terminal_blob_sha1,
            "historical_execution_head": authority.historical_execution_head,
            "historical_workflow_run_id": authority.historical_workflow_run_id,
            "historical_workflow_job_id": authority.historical_workflow_job_id,
        },
        "truth_boundary": {
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights": False,
            "external_llm_or_api_used_for_data_or_intelligence": False,
        },
    }
    return {
        **core,
        "authority_identity_sha256": mod._sha256(mod._canonical(core)),
    }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _fixture(
    rows: list[dict[str, object]],
) -> tuple[bytes, bytes, bytes, mod.FrankoAuthority]:
    candidate_raw = b"".join(_canonical_line(row) for row in rows)
    raw_indexes = [
        int(str(row["record_id"]).split(":")[1])
        for row in rows
    ]
    authority = replace(
        mod.PRODUCTION_AUTHORITY,
        source_rows_seen=max(raw_indexes),
        candidate_records=len(rows),
        candidate_text_utf8_bytes=sum(int(row["text_utf8_bytes"]) for row in rows),
        candidate_jsonl_bytes=len(candidate_raw),
        candidate_sha256=hashlib.sha256(candidate_raw).hexdigest(),
        record_inventory_sha256=_inventory_sha(rows),
    )

    historical_raw = _json_bytes(_historical(authority))
    authority = replace(
        authority,
        historical_terminal_blob_sha1=mod._git_blob_sha1(historical_raw),
    )
    historical_raw = _json_bytes(_historical(authority))
    assert mod._git_blob_sha1(historical_raw) == authority.historical_terminal_blob_sha1

    fresh_raw = _json_bytes(_fresh(authority))
    authority = replace(
        authority,
        fresh_authority_blob_sha1=mod._git_blob_sha1(fresh_raw),
    )
    fresh_raw = _json_bytes(_fresh(authority))
    assert mod._git_blob_sha1(fresh_raw) == authority.fresh_authority_blob_sha1
    return candidate_raw, historical_raw, fresh_raw, authority


def _project(
    rows: list[dict[str, object]],
    *,
    retain_payloads: bool = True,
) -> mod.FrankoProjection:
    candidate_raw, historical_raw, fresh_raw, authority = _fixture(rows)
    return mod._project_candidate_bytes(
        candidate_raw,
        historical_raw,
        fresh_raw,
        authority=authority,
        retain_payloads=retain_payloads,
    )


def test_projects_one_for_one_into_incumbent_v3_without_credit() -> None:
    rows = [_row(1, "Перше прислів'я."), _row(3, "Друге прислів'я.")]
    projection = _project(rows)

    assert projection.sources is not None
    assert projection.payloads is not None
    assert len(projection.sources) == len(rows)
    assert len(projection.payloads) == len(rows)
    first = projection.sources[0]
    assert first["source_family"] == mod.SOURCE_FAMILY
    assert first["stable_origin_id"].startswith(
        f"{mod.SOURCE_REPOSITORY}@{mod.SOURCE_REVISION}:{mod.SOURCE_PATH}#"
    )
    assert first["stable_object_id"] == f"sha256:{rows[0]['text_sha256']}"
    assert first["modality"] == "uk"
    assert f"AUDIT{mod.INDEPENDENT_AUDIT_ISSUE}" in first["authority_ref"]
    assert projection.receipt["execution_gate"] == {
        "canonical_global_dedup_executed": False,
        "matcher_invoked_by_this_adapter": False,
        "status": "READY_FOR_TERMINAL_INCUMBENT_MATCHER_AFTER_DEPENDENCY_AUTHORITY",
    }
    truth = projection.receipt["truth_boundary"]
    assert truth["canonical_capacity_credited"] == 0
    assert truth["authorized_optimized_target_exposure"] == 0
    assert truth["training_executed"] is False


def test_receipt_is_text_free_self_hashed_and_can_drop_payloads() -> None:
    secret = "НЕ ВИВОДИТИ ЦЕЙ ТЕКСТ У RECEIPT"
    projection = _project([_row(4, secret)], retain_payloads=False)
    assert projection.sources is None
    assert projection.payloads is None
    assert secret not in json.dumps(projection.receipt, ensure_ascii=False)
    receipt = dict(projection.receipt)
    identity = receipt.pop("receipt_identity_sha256")
    assert identity == mod._sha256(mod._canonical(receipt))


def test_authority_chain_keeps_historical_and_fresh_execution_distinct() -> None:
    projection = _project([_row(7, "Гаразд.")])
    chain = projection.receipt["authority_chain"]
    historical = chain["historical_terminal"]
    fresh = chain["fresh_repaired_head_execution"]
    assert historical["execution_head"] == mod.HISTORICAL_EXECUTION_HEAD
    assert fresh["product_head"] == mod.UPSTREAM_PRODUCT_HEAD
    assert historical["execution_head"] != fresh["product_head"]
    assert historical["role"] == "HISTORICAL_EXECUTED_BLOB_AUTHORITY_ONLY"
    assert fresh["role"] == "REPAIRED_HEAD_EXECUTION_AUTHORITY"


def test_full_candidate_byte_drift_fails_before_self_consistent_row_can_pass() -> None:
    original = _row(1, "Оригінал.")
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([original])
    changed = _row(1, "Змінений рядок.")
    changed_raw = _canonical_line(changed)
    assert hashlib.sha256(changed_raw).hexdigest() != authority.candidate_sha256

    with pytest.raises(mod.FrankoDedupIntakeError, match="candidate .* drift"):
        mod._project_candidate_bytes(
            changed_raw,
            historical_raw,
            fresh_raw,
            authority=authority,
            retain_payloads=True,
        )


def test_candidate_byte_length_is_checked_before_json_parsing() -> None:
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([_row(1, "Текст.")])
    malicious = candidate_raw[:-1]
    with pytest.raises(mod.FrankoDedupIntakeError, match="byte length drift"):
        mod._project_candidate_bytes(
            malicious,
            historical_raw,
            fresh_raw,
            authority=authority,
            retain_payloads=True,
        )


def test_duplicate_json_key_fails_closed() -> None:
    row = _row(1, "Текст.")
    valid = _canonical_line(row).decode().rstrip("\n")
    malicious = (valid[:-1] + ',"source_id":"wrong"}\n').encode()
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([row])
    authority = replace(
        authority,
        candidate_jsonl_bytes=len(malicious),
        candidate_sha256=hashlib.sha256(malicious).hexdigest(),
    )
    historical_raw = _json_bytes(_historical(authority))
    authority = replace(
        authority,
        historical_terminal_blob_sha1=mod._git_blob_sha1(historical_raw),
    )
    historical_raw = _json_bytes(_historical(authority))
    fresh_raw = _json_bytes(_fresh(authority))
    authority = replace(
        authority,
        fresh_authority_blob_sha1=mod._git_blob_sha1(fresh_raw),
    )
    fresh_raw = _json_bytes(_fresh(authority))

    with pytest.raises(mod.FrankoDedupIntakeError, match="duplicate JSON key"):
        mod._project_candidate_bytes(
            malicious,
            historical_raw,
            fresh_raw,
            authority=authority,
            retain_payloads=True,
        )
    assert candidate_raw != malicious


def test_duplicate_record_id_fails_closed() -> None:
    first = _row(1, "Один.")
    second = _row(2, "Два.")
    second["record_id"] = first["record_id"]
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([first, second])
    with pytest.raises(mod.FrankoDedupIntakeError, match="duplicate/invalid record_id"):
        mod._project_candidate_bytes(
            candidate_raw,
            historical_raw,
            fresh_raw,
            authority=authority,
            retain_payloads=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("language", "en"),
        ("modality", "code"),
        ("source_id", "ua.verba.other"),
        ("source_family", "ua.other"),
    ],
)
def test_row_source_authority_relabel_fails_closed(field: str, value: object) -> None:
    row = _row(1, "Текст.")
    row[field] = value
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([row])
    with pytest.raises(mod.FrankoDedupIntakeError, match="drift"):
        mod._project_candidate_bytes(
            candidate_raw,
            historical_raw,
            fresh_raw,
            authority=authority,
            retain_payloads=True,
        )


def test_bool_text_byte_count_is_rejected() -> None:
    row = _row(1, "a")
    row["text_utf8_bytes"] = True
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([row])
    with pytest.raises(mod.FrankoDedupIntakeError, match="byte count invalid"):
        mod._project_candidate_bytes(
            candidate_raw,
            historical_raw,
            fresh_raw,
            authority=authority,
            retain_payloads=True,
        )


@pytest.mark.parametrize("field", ["text_utf8_bytes", "text_sha256"])
def test_payload_byte_or_hash_mismatch_fails_closed(field: str) -> None:
    row = _row(1, "Текст.")
    if field == "text_utf8_bytes":
        row[field] = int(row[field]) + 1
    else:
        row[field] = "0" * 64
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([row])
    with pytest.raises(mod.FrankoDedupIntakeError, match="mismatch|SHA drift"):
        mod._project_candidate_bytes(
            candidate_raw,
            historical_raw,
            fresh_raw,
            authority=authority,
            retain_payloads=True,
        )


def test_record_id_digest_suffix_is_bound_to_text_hash() -> None:
    row = _row(10, "Текст.")
    row["record_id"] = "franko1901:000010:" + "0" * 16
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([row])
    with pytest.raises(mod.FrankoDedupIntakeError, match="digest suffix drift"):
        mod._project_candidate_bytes(
            candidate_raw,
            historical_raw,
            fresh_raw,
            authority=authority,
            retain_payloads=True,
        )


def test_record_indexes_must_remain_strictly_increasing() -> None:
    rows = [_row(10, "Перший."), _row(9, "Другий.")]
    candidate_raw, historical_raw, fresh_raw, authority = _fixture(rows)
    with pytest.raises(mod.FrankoDedupIntakeError, match="strictly increasing"):
        mod._project_candidate_bytes(
            candidate_raw,
            historical_raw,
            fresh_raw,
            authority=authority,
            retain_payloads=True,
        )


def test_record_index_must_not_exceed_rows_seen_authority() -> None:
    rows = [_row(11, "Текст.")]
    candidate_raw, historical_raw, fresh_raw, authority = _fixture(rows)
    authority = replace(authority, source_rows_seen=10)
    historical_raw = _json_bytes(_historical(authority))
    authority = replace(
        authority,
        historical_terminal_blob_sha1=mod._git_blob_sha1(historical_raw),
    )
    historical_raw = _json_bytes(_historical(authority))
    fresh_raw = _json_bytes(_fresh(authority))
    authority = replace(
        authority,
        fresh_authority_blob_sha1=mod._git_blob_sha1(fresh_raw),
    )
    fresh_raw = _json_bytes(_fresh(authority))
    with pytest.raises(mod.FrankoDedupIntakeError, match="rows_seen"):
        mod._project_candidate_bytes(
            candidate_raw,
            historical_raw,
            fresh_raw,
            authority=authority,
            retain_payloads=True,
        )


def test_record_inventory_identity_is_independently_recomputed() -> None:
    rows = [_row(1, "Один."), _row(3, "Два.")]
    candidate_raw, historical_raw, fresh_raw, authority = _fixture(rows)
    authority = replace(authority, record_inventory_sha256="0" * 64)
    historical_raw = _json_bytes(_historical(authority))
    authority = replace(
        authority,
        historical_terminal_blob_sha1=mod._git_blob_sha1(historical_raw),
    )
    historical_raw = _json_bytes(_historical(authority))
    fresh_raw = _json_bytes(_fresh(authority))
    authority = replace(
        authority,
        fresh_authority_blob_sha1=mod._git_blob_sha1(fresh_raw),
    )
    fresh_raw = _json_bytes(_fresh(authority))

    with pytest.raises(mod.FrankoDedupIntakeError, match="record inventory identity drift"):
        mod._project_candidate_bytes(
            candidate_raw,
            historical_raw,
            fresh_raw,
            authority=authority,
            retain_payloads=True,
        )


def test_historical_terminal_blob_tamper_fails_closed() -> None:
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([_row(1, "Текст.")])
    tampered = historical_raw.replace(b'"LOCAL_FREE"', b'"PAID"', 1)
    with pytest.raises(mod.FrankoDedupIntakeError, match="historical terminal Git blob drift"):
        mod._project_candidate_bytes(
            candidate_raw,
            tampered,
            fresh_raw,
            authority=authority,
            retain_payloads=True,
        )


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("product", "exact_head", mod.HISTORICAL_EXECUTION_HEAD),
        ("execution", "workflow_run_id", 1),
        ("execution", "workflow_job_id", 2),
        ("independent_audit", "verdict", "UNKNOWN"),
    ],
)
def test_fresh_repaired_execution_identity_drift_fails_closed(
    section: str,
    field: str,
    replacement: object,
) -> None:
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([_row(1, "Текст.")])
    fresh = json.loads(fresh_raw)
    fresh[section][field] = replacement
    core = dict(fresh)
    core.pop("authority_identity_sha256")
    fresh["authority_identity_sha256"] = mod._sha256(mod._canonical(core))
    changed = _json_bytes(fresh)
    changed_authority = replace(
        authority,
        fresh_authority_blob_sha1=mod._git_blob_sha1(changed),
    )
    with pytest.raises(mod.FrankoDedupIntakeError, match="drift"):
        mod._project_candidate_bytes(
            candidate_raw,
            historical_raw,
            changed,
            authority=changed_authority,
            retain_payloads=True,
        )


def test_fresh_authority_identity_self_hash_is_required() -> None:
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([_row(1, "Текст.")])
    fresh = json.loads(fresh_raw)
    fresh["authority_identity_sha256"] = "0" * 64
    changed = _json_bytes(fresh)
    changed_authority = replace(
        authority,
        fresh_authority_blob_sha1=mod._git_blob_sha1(changed),
    )
    with pytest.raises(mod.FrankoDedupIntakeError, match="fresh authority identity drift"):
        mod._project_candidate_bytes(
            candidate_raw,
            historical_raw,
            changed,
            authority=changed_authority,
            retain_payloads=True,
        )


def test_historical_terminal_cannot_claim_global_dedup_passed() -> None:
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([_row(1, "Текст.")])
    historical = json.loads(historical_raw)
    historical["truth_boundary"]["global_dedup"] = "PASS"
    changed = _json_bytes(historical)
    changed_authority = replace(
        authority,
        historical_terminal_blob_sha1=mod._git_blob_sha1(changed),
    )
    fresh = _fresh(changed_authority)
    changed_fresh = _json_bytes(fresh)
    changed_authority = replace(
        changed_authority,
        fresh_authority_blob_sha1=mod._git_blob_sha1(changed_fresh),
    )
    changed_fresh = _json_bytes(_fresh(changed_authority))
    with pytest.raises(mod.FrankoDedupIntakeError, match="historical truth global_dedup drift"):
        mod._project_candidate_bytes(
            candidate_raw,
            changed,
            changed_fresh,
            authority=changed_authority,
            retain_payloads=True,
        )


def test_fresh_authority_cannot_widen_training_truth() -> None:
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([_row(1, "Текст.")])
    fresh = json.loads(fresh_raw)
    fresh["truth_boundary"]["authorized_optimized_target_exposure"] = 1
    core = dict(fresh)
    core.pop("authority_identity_sha256")
    fresh["authority_identity_sha256"] = mod._sha256(mod._canonical(core))
    changed = _json_bytes(fresh)
    changed_authority = replace(
        authority,
        fresh_authority_blob_sha1=mod._git_blob_sha1(changed),
    )
    with pytest.raises(
        mod.FrankoDedupIntakeError,
        match="fresh truth authorized_optimized_target_exposure drift",
    ):
        mod._project_candidate_bytes(
            candidate_raw,
            historical_raw,
            changed,
            authority=changed_authority,
            retain_payloads=True,
        )


def test_reads_each_input_file_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_raw, historical_raw, fresh_raw, authority = _fixture([_row(1, "Текст.")])
    candidate = tmp_path / "candidate.jsonl"
    historical = tmp_path / "historical.json"
    fresh = tmp_path / "fresh.json"
    candidate.write_bytes(candidate_raw)
    historical.write_bytes(historical_raw)
    fresh.write_bytes(fresh_raw)
    monkeypatch.setattr(mod, "PRODUCTION_AUTHORITY", authority)

    calls: dict[Path, int] = {}
    original = Path.read_bytes

    def counted(path: Path) -> bytes:
        calls[path] = calls.get(path, 0) + 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    projection = mod.validate_and_project_franko(candidate, historical, fresh)
    assert projection.receipt["projection"]["source_object_count"] == 1
    assert calls == {candidate: 1, historical: 1, fresh: 1}
