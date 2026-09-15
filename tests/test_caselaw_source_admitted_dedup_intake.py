from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six.data import caselaw_source_admitted_dedup_intake as mod


def _canonical_line(row: dict[str, object]) -> bytes:
    return (
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _row(record_id: str, text: str) -> dict[str, object]:
    payload = text.encode("utf-8")
    return {
        "record_id": record_id,
        "normalized_sha256": hashlib.sha256(payload).hexdigest(),
        "normalized_utf8_bytes": len(payload),
        "text": text,
        "source_family": mod.SOURCE_FAMILY,
        "source_kind": mod.SOURCE_KIND,
        "rights_basis": mod.RIGHTS_BASIS,
        "training_eligible": False,
        "evaluation_eligible": False,
    }


def _evidence(authority: mod.CaselawAuthority) -> dict[str, object]:
    return {
        "schema_version": mod.EVIDENCE_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "execution_claim_issue": 1474,
        "finalization_claim_issue": 1766,
        "prequalification_issue": 1481,
        "product_pr": authority.product_pr,
        "successful_execution": {
            "workflow_run_id": authority.workflow_run_id,
            "job_id": authority.workflow_job_id,
            "execution_head_sha": authority.execution_head,
            "checkout_merge_sha": "0" * 40,
            "checkout_main_parent_sha": "1" * 40,
            "github_head_ref": "d03/1342-caselaw-source-admission-real-replay",
            "github_base_ref": "main",
            "github_run_attempt": 1,
            "github_run_number": 3609,
            "github_job": "bootstrap",
            "workflow_path": ".github/workflows/ci.yml",
            "event": "pull_request",
            "conclusion": "success",
            "shared_pytest_passed": 1274,
        },
        "authority": {},
        "source": {
            "dataset": mod.SOURCE_DATASET,
            "revision": mod.SOURCE_REVISION,
            "family": mod.SOURCE_FAMILY,
            "ordering_policy": mod.SOURCE_ORDERING_POLICY,
            "objects": [
                {"file": name, "compressed_bytes": size, "sha256": digest}
                for name, size, digest in mod.SOURCE_OBJECTS
            ],
        },
        "result": {
            "historical_product_replay_reproduced": True,
            "input_records": authority.candidate_records,
            "input_normalized_utf8_bytes": authority.candidate_normalized_utf8_bytes,
            "historical_candidate_sha256": authority.candidate_sha256,
            "source_admitted_records": authority.candidate_records,
            "source_admitted_normalized_utf8_bytes": (
                authority.candidate_normalized_utf8_bytes
            ),
            "source_admitted_candidate_sha256": authority.candidate_sha256,
            "candidate_denied_reason_counts": {},
        },
        "determinism": {
            "independent_execution_passes": 2,
            "raw_source_objects_byte_identical": True,
            "historical_candidate_bytes_identical": True,
            "admitted_candidate_bytes_identical": True,
            "materializer_report_bytes_identical": True,
            "source_admission_report_bytes_identical": True,
        },
        "content_boundary": {
            "candidate_or_source_text_retained_in_git": False,
            "candidate_payload_uploaded_as_artifact": False,
            "durable_evidence_text_free": True,
            "execution_evidence_sources": [],
        },
        "claim_boundary": {
            "real_source_admission_executed": True,
            "candidate_status": "CASELAW_SOURCE_ADMITTED_CANDIDATE_ONLY_ZERO_CREDIT",
            "corpus_admitted": False,
            "evaluation_eligible": False,
            "global_dedup_on_this_candidate": "NOT_RUN",
            "reserved_evaluation_decontamination_on_this_candidate": "NOT_RUN",
            "privacy_filter_on_this_candidate": "NOT_RUN",
            "quality_filter_on_this_candidate": "NOT_RUN",
            "family_balance_on_this_candidate": "NOT_RUN",
            "cluster_safe_split_on_this_candidate": "NOT_RUN",
            "deterministic_packing_on_this_candidate": "NOT_RUN",
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "model_training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "external_llm_or_api_used_for_data_or_intelligence": False,
        },
        "next_required_before_any_training_credit": list(mod._REQUIRED_NEXT_GATES),
    }


def _fixture(
    rows: list[dict[str, object]],
) -> tuple[bytes, bytes, mod.CaselawAuthority]:
    candidate_raw = b"".join(_canonical_line(row) for row in rows)
    authority = replace(
        mod.PRODUCTION_AUTHORITY,
        candidate_records=len(rows),
        candidate_normalized_utf8_bytes=sum(
            int(row["normalized_utf8_bytes"]) for row in rows
        ),
        candidate_sha256=hashlib.sha256(candidate_raw).hexdigest(),
    )
    evidence = _evidence(authority)
    evidence_raw = json.dumps(evidence, indent=2).encode("utf-8") + b"\n"
    authority = replace(
        authority,
        evidence_blob_sha1=mod._git_blob_sha1(evidence_raw),
    )
    evidence_raw = json.dumps(_evidence(authority), indent=2).encode("utf-8") + b"\n"
    assert mod._git_blob_sha1(evidence_raw) == authority.evidence_blob_sha1
    return candidate_raw, evidence_raw, authority


def _project(
    rows: list[dict[str, object]],
    *,
    retain_payloads: bool = True,
) -> mod.CaselawProjection:
    candidate_raw, evidence_raw, authority = _fixture(rows)
    return mod._project_candidate_bytes(
        candidate_raw,
        evidence_raw,
        upstream_head=authority.final_head,
        authority=authority,
        retain_payloads=retain_payloads,
    )


def test_projects_one_for_one_without_matcher_or_credit() -> None:
    projection = _project(
        [_row("case-1", "First public domain case."), _row("case-2", "Second.")]
    )
    assert projection.sources is not None
    assert projection.payloads is not None
    assert len(projection.sources) == 2
    assert len(projection.payloads) == 2
    assert projection.receipt["execution_gate"] == {
        "canonical_global_dedup_executed": False,
        "matcher_invoked_by_this_adapter": False,
        "status": "READY_FOR_TERMINAL_INCUMBENT_MATCHER_AFTER_DEPENDENCY_AUTHORITY",
    }
    assert projection.receipt["truth_boundary"]["canonical_capacity_credited"] == 0
    assert projection.receipt["truth_boundary"]["training_authorized_bytes"] == 0
    receipt = dict(projection.receipt)
    identity = receipt.pop("receipt_identity_sha256")
    assert identity == mod._sha256(mod._canonical(receipt))


def test_duplicate_payload_aliases_are_preserved_for_matcher() -> None:
    projection = _project(
        [_row("case-a", "same payload"), _row("case-b", "same payload")]
    )
    assert projection.sources is not None
    first, second = projection.sources
    assert first["stable_object_id"] == second["stable_object_id"]
    assert first["stable_origin_id"] != second["stable_origin_id"]
    assert first["origin_key"] != second["origin_key"]
    assert projection.receipt["projection"]["exact_duplicate_payload_alias_count"] == 1


def test_receipt_is_text_free_and_projection_can_drop_payloads() -> None:
    text = "SECRET-LIKE-SAMPLE-TEXT"
    projection = _project([_row("case-1", text)], retain_payloads=False)
    assert projection.sources is None
    assert projection.payloads is None
    assert text not in json.dumps(projection.receipt, sort_keys=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_family", "wrong.family"),
        ("source_kind", "CourtListener"),
        ("rights_basis", "PUBLIC_DOMAIN"),
        ("training_eligible", True),
        ("training_eligible", 0),
        ("evaluation_eligible", True),
    ],
)
def test_row_authority_drift_fails_closed(field: str, value: object) -> None:
    row = _row("case-1", "payload")
    row[field] = value
    candidate_raw, evidence_raw, authority = _fixture([row])
    with pytest.raises(mod.CaselawDedupIntakeError):
        mod._project_candidate_bytes(
            candidate_raw,
            evidence_raw,
            upstream_head=authority.final_head,
            authority=authority,
            retain_payloads=True,
        )


def test_payload_byte_or_hash_drift_fails_closed() -> None:
    row = _row("case-1", "payload")
    row["normalized_utf8_bytes"] = 999
    candidate_raw, evidence_raw, authority = _fixture([row])
    with pytest.raises(mod.CaselawDedupIntakeError, match="byte count mismatch"):
        mod._project_candidate_bytes(
            candidate_raw,
            evidence_raw,
            upstream_head=authority.final_head,
            authority=authority,
            retain_payloads=True,
        )


def test_duplicate_record_id_fails_closed() -> None:
    rows = [_row("case-1", "one"), _row("case-1", "two")]
    candidate_raw, evidence_raw, authority = _fixture(rows)
    with pytest.raises(mod.CaselawDedupIntakeError, match="duplicate/invalid record_id"):
        mod._project_candidate_bytes(
            candidate_raw,
            evidence_raw,
            upstream_head=authority.final_head,
            authority=authority,
            retain_payloads=True,
        )


def test_duplicate_json_key_fails_closed() -> None:
    row = _row("case-1", "payload")
    valid = _canonical_line(row).decode("utf-8").rstrip("\n")
    malicious = valid[:-1] + ',"record_id":"case-2"}\n'
    candidate_raw = malicious.encode("utf-8")
    authority = replace(
        mod.PRODUCTION_AUTHORITY,
        candidate_records=1,
        candidate_normalized_utf8_bytes=len(b"payload"),
        candidate_sha256=hashlib.sha256(candidate_raw).hexdigest(),
    )
    evidence_raw = json.dumps(_evidence(authority), indent=2).encode("utf-8") + b"\n"
    authority = replace(authority, evidence_blob_sha1=mod._git_blob_sha1(evidence_raw))
    evidence_raw = json.dumps(_evidence(authority), indent=2).encode("utf-8") + b"\n"
    with pytest.raises(mod.CaselawDedupIntakeError, match="duplicate JSON key"):
        mod._project_candidate_bytes(
            candidate_raw,
            evidence_raw,
            upstream_head=authority.final_head,
            authority=authority,
            retain_payloads=True,
        )


def test_evidence_blob_tamper_fails_before_projection() -> None:
    candidate_raw, evidence_raw, authority = _fixture([_row("case-1", "payload")])
    tampered = evidence_raw.replace(b'"LOCAL_FREE"', b'"PAID"', 1)
    with pytest.raises(mod.CaselawDedupIntakeError, match="evidence Git blob drift"):
        mod._project_candidate_bytes(
            candidate_raw,
            tampered,
            upstream_head=authority.final_head,
            authority=authority,
            retain_payloads=True,
        )


def test_candidate_hash_and_final_head_are_exact() -> None:
    candidate_raw, evidence_raw, authority = _fixture([_row("case-1", "payload")])
    with pytest.raises(mod.CaselawDedupIntakeError, match="final head drift"):
        mod._project_candidate_bytes(
            candidate_raw,
            evidence_raw,
            upstream_head="0" * 40,
            authority=authority,
            retain_payloads=True,
        )
    with pytest.raises(mod.CaselawDedupIntakeError, match="candidate SHA-256 drift"):
        mod._project_candidate_bytes(
            candidate_raw + b" ",
            evidence_raw,
            upstream_head=authority.final_head,
            authority=authority,
            retain_payloads=True,
        )


def test_requires_terminal_zero_credit_evidence_semantics() -> None:
    candidate_raw, evidence_raw, authority = _fixture([_row("case-1", "payload")])
    evidence = json.loads(evidence_raw)
    evidence["claim_boundary"]["global_dedup_on_this_candidate"] = "PASS"
    changed = json.dumps(evidence, indent=2).encode("utf-8") + b"\n"
    changed_authority = replace(
        authority,
        evidence_blob_sha1=mod._git_blob_sha1(changed),
    )
    with pytest.raises(
        mod.CaselawDedupIntakeError,
        match="global_dedup_on_this_candidate",
    ):
        mod._project_candidate_bytes(
            candidate_raw,
            changed,
            upstream_head=changed_authority.final_head,
            authority=changed_authority,
            retain_payloads=True,
        )


def test_reads_each_input_file_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_raw, evidence_raw, authority = _fixture([_row("case-1", "payload")])
    candidate = tmp_path / "candidate.jsonl"
    evidence = tmp_path / "evidence.json"
    candidate.write_bytes(candidate_raw)
    evidence.write_bytes(evidence_raw)
    monkeypatch.setattr(mod, "PRODUCTION_AUTHORITY", authority)

    calls: dict[Path, int] = {}
    original = Path.read_bytes

    def counted(path: Path) -> bytes:
        calls[path] = calls.get(path, 0) + 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    projection = mod.validate_and_project_caselaw(
        candidate,
        evidence,
        upstream_head=authority.final_head,
    )
    assert projection.receipt["projection"]["source_object_count"] == 1
    assert calls == {candidate: 1, evidence: 1}
