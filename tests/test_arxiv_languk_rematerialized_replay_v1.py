from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from twelve_six.data.arxiv_languk_rematerialized_replay_v1 import (
    ARXIV,
    LANGUK,
    PARENT_INTAKE_BLOB_SHA1,
    PARENT_RUNNER_BLOB_SHA1,
    RematerializationError,
    build_receipt,
    canonical_json_bytes,
    git_blob_sha1,
    verify_candidate,
    verify_git_blob,
)


def _load_replay_runner():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools/run_d03_arxiv_languk_rematerialized_replay_v1.py"
    )
    spec = importlib.util.spec_from_file_location("_test_arxiv_languk_replay_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLAY_RUNNER = _load_replay_runner()


def _candidate_row(
    *,
    record_id: str = "7",
    text: str = "Український текст",
    arxiv: bool = False,
) -> dict[str, object]:
    raw = text.encode("utf-8")
    row: dict[str, object] = {
        "record_id": record_id,
        "normalized_sha256": hashlib.sha256(raw).hexdigest(),
        "normalized_bytes": len(raw),
        "text": text,
    }
    if arxiv:
        row["source_key"] = "arxiv_abstracts"
        row["source_label"] = "arxiv-abstracts"
    return row


def _fixture_spec(*, arxiv: bool, rows: list[dict[str, object]]):
    raw = b"".join(canonical_json_bytes(row) for row in rows)
    template = ARXIV if arxiv else LANGUK
    return (
        replace(
            template,
            candidate_sha256=hashlib.sha256(raw).hexdigest(),
            retained_records=len(rows),
            retained_normalized_bytes=sum(int(row["normalized_bytes"]) for row in rows),
        ),
        raw,
    )


def test_terminal_program_identities_are_pinned() -> None:
    assert ARXIV.execution_commit == "37289b5af224c58f0b169f55f4e0f8b466b90f6d"
    assert ARXIV.tool_blob_sha1 == "38c244a5a2da40ed4fb67f801f02e625bc040564"
    assert ARXIV.config_blob_sha1 == "4a819b3980634c8a2ba7de55cf854d44bcfd5757"
    assert ARXIV.candidate_sha256 == (
        "21304338039306b2df175bbb71a1aed9a443c2416f3858f3cc63ca208e9c7327"
    )
    assert (ARXIV.retained_records, ARXIV.retained_normalized_bytes) == (
        1024,
        1_139_552,
    )

    assert LANGUK.execution_commit == "5d38a32b37490b11f6fa77b3bcd21855f3e28605"
    assert LANGUK.tool_blob_sha1 == "5401881e170f2d1ae70e777a0cf3e2cfa152e8f6"
    assert LANGUK.config_blob_sha1 == "b4ee7b0f698660145cb3c1db0f8ccb690c255a5e"
    assert LANGUK.candidate_sha256 == (
        "03bf5089bb6ff4c304e3a299e2480b263a161aad8abe831db0b196af7c585db3"
    )
    assert (LANGUK.retained_records, LANGUK.retained_normalized_bytes) == (
        256,
        2_809_632,
    )


def test_git_blob_verification_is_exact() -> None:
    raw = b"trusted historical program\n"
    expected = git_blob_sha1(raw)
    verify_git_blob(raw, expected, label="fixture")
    with pytest.raises(RematerializationError, match="Git blob drift"):
        verify_git_blob(raw + b"x", expected, label="fixture")


def test_verify_languk_candidate_checks_exact_identity_and_rows() -> None:
    rows = [
        _candidate_row(record_id="10", text="Український судовий документ"),
        _candidate_row(record_id="11", text="Ще один український документ"),
    ]
    spec, raw = _fixture_spec(arxiv=False, rows=rows)
    result = verify_candidate(spec, raw)
    assert result == {
        "candidate_sha256": spec.candidate_sha256,
        "retained_records": 2,
        "retained_normalized_bytes": sum(int(row["normalized_bytes"]) for row in rows),
    }


def test_verify_arxiv_candidate_checks_source_binding() -> None:
    rows = [_candidate_row(record_id="arxiv-1", text="A long abstract", arxiv=True)]
    spec, raw = _fixture_spec(arxiv=True, rows=rows)
    verify_candidate(spec, raw)

    mutated = copy.deepcopy(rows)
    mutated[0]["source_label"] = "wrong"
    bad_spec, bad_raw = _fixture_spec(arxiv=True, rows=mutated)
    with pytest.raises(RematerializationError, match="source label drift"):
        verify_candidate(bad_spec, bad_raw)


def test_verify_candidate_rejects_duplicate_ids() -> None:
    rows = [
        _candidate_row(record_id="10", text="Документ один"),
        _candidate_row(record_id="10", text="Документ два"),
    ]
    spec, raw = _fixture_spec(arxiv=False, rows=rows)
    with pytest.raises(RematerializationError, match="duplicate record id"):
        verify_candidate(spec, raw)


def test_verify_candidate_rejects_payload_hash_drift_before_parsing() -> None:
    rows = [_candidate_row(record_id="10", text="Документ")]
    spec, raw = _fixture_spec(arxiv=False, rows=rows)
    with pytest.raises(RematerializationError, match="candidate SHA-256 drift"):
        verify_candidate(spec, raw + b"\n")


def _pass_result() -> dict[str, str]:
    return {
        "arxiv_candidate_sha256": ARXIV.candidate_sha256,
        "languk_candidate_sha256": LANGUK.candidate_sha256,
        "report_file_sha256": "1" * 64,
        "survivor_file_sha256": "2" * 64,
        "report_identity_sha256": "3" * 64,
        "survivor_authority_sha256": "4" * 64,
    }


def test_receipt_requires_two_identical_passes_and_preserves_zero_truth() -> None:
    one = _pass_result()
    receipt = build_receipt(
        pass_results=[one, dict(one)],
        incumbent_runner_blob_sha1=PARENT_RUNNER_BLOB_SHA1,
        incumbent_intake_blob_sha1=PARENT_INTAKE_BLOB_SHA1,
    )
    assert receipt["status"] == (
        "PHYSICAL_REMATERIALIZATION_AND_V9_REPLAY_EXECUTED_ZERO_CREDIT"
    )
    assert receipt["reproducibility"]["report_files_byte_identical"] is True
    assert receipt["reproducibility"]["raw_payloads_retained_after_success"] is False
    boundary = receipt["truth_boundary"]
    assert boundary["global_dedup_replay_executed"] is True
    assert boundary["canonical_capacity_credited"] == 0
    assert boundary["authorized_optimized_target_exposure"] == 0
    assert boundary["tokenizer_fit_authorized"] is False
    assert boundary["optimizer_updates"] == 0
    assert boundary["training_executed"] is False
    assert boundary["learned_weights_created"] is False
    assert boundary["final_test_outcomes_read"] is False
    assert boundary["paid_compute_used"] is False
    assert boundary["foreign_pretrained_weights_used"] is False
    assert boundary["external_llm_or_api_used_for_data_or_intelligence"] is False


def test_receipt_fails_on_second_pass_drift() -> None:
    one = _pass_result()
    two = dict(one)
    two["report_file_sha256"] = "9" * 64
    with pytest.raises(RematerializationError, match="two-pass replay mismatch"):
        build_receipt(
            pass_results=[one, two],
            incumbent_runner_blob_sha1=PARENT_RUNNER_BLOB_SHA1,
            incumbent_intake_blob_sha1=PARENT_INTAKE_BLOB_SHA1,
        )


def test_receipt_fails_on_incumbent_blob_drift() -> None:
    one = _pass_result()
    with pytest.raises(RematerializationError, match="incumbent runner blob drift"):
        build_receipt(
            pass_results=[one, dict(one)],
            incumbent_runner_blob_sha1="0" * 40,
            incumbent_intake_blob_sha1=PARENT_INTAKE_BLOB_SHA1,
        )


def test_receipt_serialization_contains_no_payload_text() -> None:
    one = _pass_result()
    receipt = build_receipt(
        pass_results=[one, dict(one)],
        incumbent_runner_blob_sha1=PARENT_RUNNER_BLOB_SHA1,
        incumbent_intake_blob_sha1=PARENT_INTAKE_BLOB_SHA1,
    )
    serialized = canonical_json_bytes(receipt)
    decoded = json.loads(serialized)
    assert decoded["historical_materializers"]["arxiv"]["retained_records"] == 1024
    assert "text" not in serialized.decode("utf-8").lower()


def _runner_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        v7_root=Path("v7-root"),
        bulk_workspace=Path("bulk-workspace"),
        v8_config=Path("configs/v8.json"),
        data526_config=Path("configs/data526.json"),
        v8_report=Path("authority/v8-report.json"),
        v8_survivors=Path("authority/v8-survivors.json"),
        data526_evidence=Path("authority/data526-evidence.json"),
        data526_record_inventory=Path("authority/data526-inventory.json"),
        rada_language_report=Path("authority/rada-language.json"),
        rada_quality_privacy_jsonl=Path("authority/rada-quality-privacy.jsonl"),
        rada_quality_privacy_report=Path("authority/rada-quality-privacy.json"),
        expected_rada_report_sha256="a" * 64,
        arxiv_authority=Path("authority/arxiv.json"),
        languk_authority=Path("authority/languk.json"),
        output_report=tmp_path / "outer-report.json",
        output_survivors=tmp_path / "outer-survivors.json",
        output_receipt=tmp_path / "outer-receipt.json",
    )


def test_replay_command_uses_extracted_exact_parent_tree_and_isolated_python(
    tmp_path: Path,
) -> None:
    args = _runner_args(tmp_path)
    parent_root = tmp_path / "parent-pr1800"
    command = REPLAY_RUNNER._replay_command(
        args,
        parent_root=parent_root,
        arxiv_candidate=tmp_path / "arxiv.jsonl",
        languk_candidate=tmp_path / "languk.jsonl",
        report=tmp_path / "report.json",
        survivors=tmp_path / "survivors.json",
    )

    assert command[1] == "-I"
    assert Path(command[2]) == parent_root / REPLAY_RUNNER.INCUMBENT_RUNNER_REL
    assert Path(command[2]) != REPLAY_RUNNER.ROOT / REPLAY_RUNNER.INCUMBENT_RUNNER_REL


def test_replay_command_ignores_substituted_current_checkout_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    poison_root = tmp_path / "poison-current-checkout"
    poison_runner = poison_root / REPLAY_RUNNER.INCUMBENT_RUNNER_REL
    poison_runner.parent.mkdir(parents=True)
    poison_runner.write_text("raise RuntimeError('substituted current checkout')\n")
    poison_dependency = poison_root / "tools/run_d03_expanded_global_dedup_v9.py"
    poison_dependency.write_text("raise RuntimeError('substituted transitive dependency')\n")
    monkeypatch.setattr(REPLAY_RUNNER, "ROOT", poison_root)

    args = _runner_args(tmp_path)
    parent_root = tmp_path / "exact-parent-tree"
    command = REPLAY_RUNNER._replay_command(
        args,
        parent_root=parent_root,
        arxiv_candidate=tmp_path / "arxiv.jsonl",
        languk_candidate=tmp_path / "languk.jsonl",
        report=tmp_path / "report.json",
        survivors=tmp_path / "survivors.json",
    )

    assert Path(command[2]) == parent_root / REPLAY_RUNNER.INCUMBENT_RUNNER_REL
    assert Path(command[2]) != poison_runner
    assert command[1] == "-I"


def test_outer_outputs_reject_existing_and_symlink_targets(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    args.output_report.write_bytes(b"do-not-overwrite")
    with pytest.raises(RematerializationError, match="refusing to overwrite outer report"):
        REPLAY_RUNNER._verify_outer_output_targets(args)
    assert args.output_report.read_bytes() == b"do-not-overwrite"

    args.output_report.unlink()
    dangling = tmp_path / "missing-target"
    args.output_receipt.symlink_to(dangling)
    with pytest.raises(RematerializationError, match="refusing to overwrite outer receipt"):
        REPLAY_RUNNER._verify_outer_output_targets(args)


def test_outer_outputs_must_be_distinct(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    args.output_survivors = args.output_report
    with pytest.raises(RematerializationError, match="must be distinct"):
        REPLAY_RUNNER._verify_outer_output_targets(args)


def test_exclusive_output_publication_never_clobbers(tmp_path: Path) -> None:
    output = tmp_path / "authority.json"
    REPLAY_RUNNER._write_new_bytes(output, b"first", label="authority")
    assert output.read_bytes() == b"first"
    with pytest.raises(RematerializationError, match="refusing to overwrite authority"):
        REPLAY_RUNNER._write_new_bytes(output, b"second", label="authority")
    assert output.read_bytes() == b"first"


def test_repaired_receipt_binds_wrapper_and_exact_parent_execution() -> None:
    one = _pass_result()
    receipt = build_receipt(
        pass_results=[one, dict(one)],
        incumbent_runner_blob_sha1=PARENT_RUNNER_BLOB_SHA1,
        incumbent_intake_blob_sha1=PARENT_INTAKE_BLOB_SHA1,
    )
    wrapper = {
        "source_head_sha": "a" * 40,
        "runner_path": REPLAY_RUNNER.WRAPPER_RUNNER_REL.as_posix(),
        "runner_blob_sha1": "b" * 40,
        "helper_path": REPLAY_RUNNER.WRAPPER_HELPER_REL.as_posix(),
        "helper_blob_sha1": "c" * 40,
    }
    repaired = REPLAY_RUNNER._finalize_receipt(
        receipt,
        wrapper_execution_authority=wrapper,
    )

    assert repaired["schema_version"] == REPLAY_RUNNER.REPAIRED_RECEIPT_SCHEMA
    assert repaired["parent_authority"]["execution_tree_mode"] == (
        "EXTRACTED_EXACT_GIT_TREE"
    )
    assert repaired["parent_authority"]["isolated_python_mode"] is True
    assert repaired["wrapper_execution_authority"] == wrapper
    assert repaired["truth_boundary"]["canonical_capacity_credited"] == 0

    bad = dict(wrapper)
    bad["source_head_sha"] = "not-a-git-sha"
    with pytest.raises(RematerializationError, match="wrapper source head malformed"):
        REPLAY_RUNNER._finalize_receipt(
            receipt,
            wrapper_execution_authority=bad,
        )
