from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from typing import Any

import pytest

from twelve_six.data import post_admission_dedup_intake_v2 as mod


def _line(row: dict[str, object]) -> bytes:
    return json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _blob_sha(raw: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(raw)}\0".encode() + raw,
        usedforsecurity=False,
    ).hexdigest()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _boundary() -> dict[str, object]:
    return {
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "family_credit_added": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "optimizer_updates": 0,
        "evaluation_authorized_bytes": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "learned_weights_created": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights_used": False,
        "external_llm_or_api_used_for_data_or_intelligence": False,
    }


def _arxiv_fixture() -> tuple[mod._SourceSpec, bytes, bytes]:
    text = "A deterministic abstract for the repaired post-admission intake."
    payload = text.encode()
    candidate = _line(
        {
            "record_id": "a-1",
            "source_key": "arxiv_abstracts",
            "source_label": "arxiv-abstracts",
            "normalized_sha256": _sha(payload),
            "normalized_bytes": len(payload),
            "text": text,
        }
    )
    authority = {
        "schema_version": mod.ARXIV.authority_schema,
        "status": mod.ARXIV.authority_status,
        "execution_profile": "LOCAL_FREE",
        "historical_execution": {"run_id": 34563405053},
        "admitted_candidate": {
            "source_sha256": "1" * 64,
            "source_bytes": 123,
            "candidate_sha256": _sha(candidate),
            "retained_records": 1,
            "retained_normalized_bytes": len(payload),
            "payload_rematerialized_in_this_package": False,
        },
        "truth_boundary": _boundary(),
    }
    raw = json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
    spec = replace(
        mod.ARXIV,
        authority_blob_sha1=_blob_sha(raw),
        candidate_sha256=_sha(candidate),
        retained_records=1,
        retained_normalized_bytes=len(payload),
        source_sha256="1" * 64,
        source_bytes=123,
    )
    return spec, raw, candidate


def _languk_fixture() -> tuple[mod._SourceSpec, bytes, bytes]:
    text = "Український синтетичний текст для перевірки intake."
    payload = text.encode()
    candidate = _line(
        {
            "record_id": "7",
            "normalized_sha256": _sha(payload),
            "normalized_bytes": len(payload),
            "text": text,
        }
    )
    authority = {
        "schema_version": mod.LANGUK.authority_schema,
        "status": mod.LANGUK.authority_status,
        "execution_profile": "LOCAL_FREE",
        "source_scope": {
            "dataset": "fixture/languk",
            "revision": "a" * 40,
            "file": "fixture.parquet",
            "sha256": "2" * 64,
            "bytes": 456,
        },
        "admitted_rights_candidate": {
            "retained_jsonl_sha256": _sha(candidate),
            "retained_records": 1,
            "retained_normalized_bytes": len(payload),
            "privacy_quality_retest_passed_for_retained_rows": True,
            "payload_rematerialized_in_this_package": False,
        },
        "truth_boundary": _boundary(),
    }
    raw = json.dumps(authority, sort_keys=True, separators=(",", ":")).encode()
    spec = replace(
        mod.LANGUK,
        authority_blob_sha1=_blob_sha(raw),
        candidate_sha256=_sha(candidate),
        retained_records=1,
        retained_normalized_bytes=len(payload),
        source_dataset="fixture/languk",
        source_revision="a" * 40,
        source_file="fixture.parquet",
        source_sha256="2" * 64,
        source_bytes=456,
    )
    return spec, raw, candidate


def test_repaired_api_has_no_caller_supplied_incumbent_graph_or_head() -> None:
    parameters = inspect.signature(mod.run_post_admission_global_dedup).parameters
    assert "incumbent_v9_head" not in parameters
    assert "incumbent_inventory" not in parameters
    assert "incumbent_payloads" not in parameters


def test_source_authority_and_candidate_substitution_fail_closed() -> None:
    spec, authority, candidate = _arxiv_fixture()

    with pytest.raises(mod.PostAdmissionDedupIntakeError, match="authority blob drift"):
        mod._prepare_source(spec, authority_raw=authority + b"\n", candidate_raw=candidate)

    with pytest.raises(mod.PostAdmissionDedupIntakeError, match="candidate SHA drift"):
        mod._prepare_source(spec, authority_raw=authority, candidate_raw=candidate + b" ")


def test_duplicate_keys_and_bool_byte_alias_fail_closed() -> None:
    spec, authority, _ = _languk_fixture()
    duplicate = (
        b'{"record_id":"7","record_id":"7","normalized_sha256":"'
        + b"0" * 64
        + b'","normalized_bytes":1,"text":"x"}\n'
    )
    mutated_spec = replace(
        spec,
        candidate_sha256=_sha(duplicate),
        retained_normalized_bytes=1,
    )
    value = json.loads(authority)
    admitted = value["admitted_rights_candidate"]
    admitted["retained_jsonl_sha256"] = _sha(duplicate)
    admitted["retained_normalized_bytes"] = 1
    mutated_authority = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    mutated_spec = replace(
        mutated_spec,
        authority_blob_sha1=_blob_sha(mutated_authority),
    )
    with pytest.raises(mod.PostAdmissionDedupIntakeError, match="duplicate JSON key"):
        mod._prepare_source(
            mutated_spec,
            authority_raw=mutated_authority,
            candidate_raw=duplicate,
        )

    text = "x"
    bool_row = _line(
        {
            "record_id": "7",
            "normalized_sha256": _sha(text.encode()),
            "normalized_bytes": True,
            "text": text,
        }
    )
    value["admitted_rights_candidate"]["retained_jsonl_sha256"] = _sha(bool_row)
    bool_authority = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    bool_spec = replace(
        mutated_spec,
        authority_blob_sha1=_blob_sha(bool_authority),
        candidate_sha256=_sha(bool_row),
    )
    with pytest.raises(
        mod.PostAdmissionDedupIntakeError,
        match="normalized byte count drift",
    ):
        mod._prepare_source(
            bool_spec,
            authority_raw=bool_authority,
            candidate_raw=bool_row,
        )


def test_v9_reconstruction_vector_is_a_required_cross_check() -> None:
    report = {
        "report_sha256": "a" * 64,
        "source_vector": {
            "pre_dedup_source_object_count": 2,
            "pre_dedup_declared_capacity_bytes": 2,
        },
    }
    assert mod._validate_v9_reconstruction(
        report,
        {"base": b"B"},
        {"rada": b"R"},
    ) == (2, 2)

    report["source_vector"]["pre_dedup_declared_capacity_bytes"] = 3
    with pytest.raises(
        mod.PostAdmissionDedupIntakeError,
        match="reconstructed byte total drift",
    ):
        mod._validate_v9_reconstruction(
            report,
            {"base": b"B"},
            {"rada": b"R"},
        )


def _projection(dedup: dict[str, Any]) -> dict[str, Any]:
    terminal = dedup["terminal_candidates"]
    source_ids = sorted(row["source_id"] for row in dedup["_inventory"]["sources"])
    return {
        "schema_version": "fixture.v9-survivors",
        "survivor_authority_sha256": "d" * 64,
        "pre_dedup_source_object_count": dedup["source_count"],
        "post_dedup_survivor_source_object_count": dedup["source_count"],
        "pre_dedup_declared_capacity_bytes": terminal["declared_capacity_bytes_before"],
        "post_dedup_declared_capacity_bytes": terminal[
            "conservative_unique_capacity_bytes_after"
        ],
        "duplicate_discount_bytes": 0,
        "duplicate_cluster_count": 0,
        "duplicate_clusters": [],
        "survivor_source_ids": source_ids,
    }


def test_v9_proof_and_use_share_snapshotted_authority_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arxiv, a_auth, a_candidate = _arxiv_fixture()
    languk, l_auth, l_candidate = _languk_fixture()
    monkeypatch.setattr(mod, "ARXIV", arxiv)
    monkeypatch.setattr(mod, "LANGUK", languk)

    original_inventory = {"sources": [{"source_id": "base"}], "lineage_edges": []}
    original_payloads = {"base": b"B"}
    observed: dict[str, Any] = {"expanded_calls": 0}

    def fake_v9_run(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        assert kwargs["reconstructed_v8_inventory"]["sources"][0]["source_id"] == "base"
        assert kwargs["reconstructed_v8_payloads"] == {"base": b"B"}
        original_inventory["sources"][0]["source_id"] = "evil"
        original_payloads["base"] = b"EVIL"
        return (
            {
                "report_sha256": "a" * 64,
                "source_vector": {
                    "pre_dedup_source_object_count": 2,
                    "pre_dedup_declared_capacity_bytes": 2,
                },
            },
            {},
        )

    monkeypatch.setattr(mod.v9, "run_expanded_dedup", fake_v9_run)
    monkeypatch.setattr(
        mod.v9,
        "_verify_matcher_semantic_closure",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        mod.v9,
        "validate_rada_rows",
        lambda rows, _raw, _report: list(rows),
    )
    monkeypatch.setattr(
        mod.v9,
        "validate_rada_quality_privacy_report",
        lambda _report, expected_report_sha256: expected_report_sha256,
    )

    def fake_restrict(
        inventory: dict[str, Any],
        _survivors: dict[str, Any],
    ) -> dict[str, Any]:
        assert inventory["sources"][0]["source_id"] == "base"
        return inventory

    monkeypatch.setattr(mod.v9, "_restrict_lineage_to_survivors", fake_restrict)

    def fake_filter(
        inventory: dict[str, Any],
        payloads: dict[str, bytes],
        _survivors: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
        assert inventory["sources"][0]["source_id"] == "base"
        assert payloads == {"base": b"B"}
        return ([{"source_id": "base"}], {"base": b"B"})

    monkeypatch.setattr(mod.v9, "filter_v8_survivor_inputs", fake_filter)
    monkeypatch.setattr(
        mod.v9,
        "build_rada_matcher_inputs",
        lambda _rows, authority_report_sha256: (
            [{"source_id": "rada", "authority_ref": authority_report_sha256}],
            {"rada": b"R"},
        ),
    )
    monkeypatch.setattr(mod.v9, "_derive_survivors", _projection)

    def matcher_audit(
        inventory: dict[str, Any],
        payloads: dict[str, bytes],
    ) -> dict[str, Any]:
        observed["expanded_calls"] += 1
        observed["source_ids"] = [row["source_id"] for row in inventory["sources"]]
        total = sum(len(raw) for raw in payloads.values())
        return {
            "report_sha256": "c" * 64,
            "source_count": len(payloads),
            "terminal_candidates": {
                "declared_capacity_bytes_before": total,
                "conservative_unique_capacity_bytes_after": total,
                "duplicate_discount_bytes": 0,
                "duplicate_cluster_count": 0,
            },
            "_inventory": inventory,
        }

    report, survivors = mod.run_post_admission_global_dedup(
        matcher_audit=matcher_audit,
        matcher_verify=lambda _report: None,
        reconstructed_v8_inventory=original_inventory,
        reconstructed_v8_payloads=original_payloads,
        v8_survivor_authority={},
        data526_evidence={},
        data526_record_inventory={},
        rada_language_report={},
        rada_quality_privacy_report={},
        expected_rada_report_sha256="b" * 64,
        rada_rows=[{"record_id": "r"}],
        rada_raw_jsonl=b'{"record_id":"r"}\n',
        arxiv_authority_raw=a_auth,
        arxiv_candidate_raw=a_candidate,
        languk_authority_raw=l_auth,
        languk_candidate_raw=l_candidate,
    )

    assert observed["expanded_calls"] == 1
    assert observed["source_ids"] == [
        "base",
        "rada",
        "arxiv-admitted:a-1",
        "languk-admitted:7",
    ]
    assert report["claim_boundary"]["incumbent_v9_authority_executed"] is True
    assert report["claim_boundary"]["canonical_capacity_credited"] == 0
    assert report["claim_boundary"]["training_authorized_bytes"] == 0
    assert report["claim_boundary"]["authorized_optimized_target_exposure"] == 0
    assert survivors["schema_version"] == mod.SURVIVOR_SCHEMA
    assert "evil" not in observed["source_ids"]


def test_production_constants_preserve_real_admitted_quantity_zero_credit() -> None:
    assert mod.ARXIV.retained_records + mod.LANGUK.retained_records == 1280
    assert (
        mod.ARXIV.retained_normalized_bytes
        + mod.LANGUK.retained_normalized_bytes
        == 3_949_184
    )
