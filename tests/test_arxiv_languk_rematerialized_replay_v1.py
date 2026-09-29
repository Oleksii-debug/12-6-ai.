from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from twelve_six.data.arxiv_languk_rematerialized_replay_v1 import (
    ARXIV,
    LANGUK,
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


def test_runner_uses_private_helper_exception_identity() -> None:
    assert REPLAY_RUNNER.RematerializationError is not RematerializationError
    assert REPLAY_RUNNER.RematerializationError.__name__ == RematerializationError.__name__
    assert REPLAY_RUNNER.RematerializationError.__module__ == "_pr1851_replay_helper"


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
    )
    assert receipt["status"] == (
        "PHYSICAL_REMATERIALIZATION_AND_CURRENT_CLEAN_DEDUP_REPLAY_EXECUTED_ZERO_CREDIT"
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
    assert "external_llm_or_api_used_for_data_or_intelligence" not in boundary
    assert boundary["current_corpus_external_llm_free_claimed_by_this_replay"] is False


def test_receipt_fails_on_second_pass_drift() -> None:
    one = _pass_result()
    two = dict(one)
    two["report_file_sha256"] = "9" * 64
    with pytest.raises(RematerializationError, match="two-pass replay mismatch"):
        build_receipt(
            pass_results=[one, two],
        )


def test_receipt_serialization_contains_no_payload_text() -> None:
    one = _pass_result()
    receipt = build_receipt(
        pass_results=[one, dict(one)],
    )
    serialized = canonical_json_bytes(receipt)
    decoded = json.loads(serialized)
    assert decoded["historical_materializers"]["arxiv"]["retained_records"] == 1024
    assert "text" not in serialized.decode("utf-8").lower()


def _runner_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        v7_root=Path("v7-root"),
        arxiv_authority=Path("authority/arxiv.json"),
        languk_authority=Path("authority/languk.json"),
        output_report=tmp_path / "outer-report.json",
        output_survivors=tmp_path / "outer-survivors.json",
        output_receipt=tmp_path / "outer-receipt.json",
    )

def test_outer_outputs_reject_existing_and_symlink_targets(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    args.output_report.write_bytes(b"do-not-overwrite")
    with pytest.raises(REPLAY_RUNNER.RematerializationError, match="refusing to overwrite outer report"):
        REPLAY_RUNNER._verify_outer_output_targets(args)
    assert args.output_report.read_bytes() == b"do-not-overwrite"

    args.output_report.unlink()
    dangling = tmp_path / "missing-target"
    args.output_receipt.symlink_to(dangling)
    with pytest.raises(REPLAY_RUNNER.RematerializationError, match="refusing to overwrite outer receipt"):
        REPLAY_RUNNER._verify_outer_output_targets(args)


def test_outer_outputs_must_be_distinct(tmp_path: Path) -> None:
    args = _runner_args(tmp_path)
    args.output_survivors = args.output_report
    with pytest.raises(REPLAY_RUNNER.RematerializationError, match="must be distinct"):
        REPLAY_RUNNER._verify_outer_output_targets(args)


def test_exclusive_output_publication_never_clobbers(tmp_path: Path) -> None:
    output = tmp_path / "authority.json"
    REPLAY_RUNNER._write_new_bytes(output, b"first", label="authority")
    assert output.read_bytes() == b"first"
    with pytest.raises(REPLAY_RUNNER.RematerializationError, match="refusing to overwrite authority"):
        REPLAY_RUNNER._write_new_bytes(output, b"second", label="authority")
    assert output.read_bytes() == b"first"


def test_repaired_receipt_binds_current_clean_execution_and_scopes_provenance() -> None:
    one = _pass_result()
    receipt = build_receipt(pass_results=[one, dict(one)])
    wrapper = {
        "source_head_sha": "a" * 40,
        "runner_path": REPLAY_RUNNER.WRAPPER_RUNNER_REL.as_posix(),
        "runner_blob_sha1": "b" * 40,
        "helper_path": REPLAY_RUNNER.WRAPPER_HELPER_REL.as_posix(),
        "helper_blob_sha1": "c" * 40,
    }
    current = {
        "clean_successor_tool_git_blob_sha1": REPLAY_RUNNER.EXPECTED_CLEAN_SUCCESSOR_BLOB,
        "survivor_tool_git_blob_sha1": REPLAY_RUNNER.EXPECTED_SURVIVOR_TOOL_BLOB,
        "indexed_executor_git_blob_sha1": REPLAY_RUNNER.EXPECTED_INDEXED_EXECUTOR_BLOB,
        "source_intake_git_blob_sha1": REPLAY_RUNNER.EXPECTED_CURRENT_INTAKE_BLOB,
    }

    repaired = REPLAY_RUNNER._finalize_receipt(
        receipt,
        wrapper_execution_authority=wrapper,
        current_clean_execution_authority=current,
    )

    assert repaired["schema_version"] == REPLAY_RUNNER.REPAIRED_RECEIPT_SCHEMA
    assert repaired["historical_parent_lineage"][
        "used_as_current_corpus_execution_authority"
    ] is False
    assert repaired["current_clean_execution_authority"][
        "clean_successor_product_pr"
    ] == 2107
    assert repaired["current_clean_execution_authority"][
        "indexed_executor_product_pr"
    ] == 1459
    truth = repaired["truth_boundary"]
    assert "external_llm_or_api_used_for_data_or_intelligence" not in truth
    assert truth["current_corpus_external_llm_free_claimed_by_this_replay"] is False
    assert truth["canonical_capacity_credited"] == 0
    assert truth["authorized_optimized_target_exposure"] == 0

    bad = dict(wrapper)
    bad["source_head_sha"] = "not-a-git-sha"
    with pytest.raises(REPLAY_RUNNER.RematerializationError, match="wrapper source head malformed"):
        REPLAY_RUNNER._finalize_receipt(
            receipt,
            wrapper_execution_authority=bad,
            current_clean_execution_authority=current,
        )


def test_declared_capacity_arithmetic_is_distinct_from_raw_payload_bytes() -> None:
    inventory = {
        "sources": [
            {"declared_capacity_bytes": 7, "expected_raw_bytes": 11},
            {"declared_capacity_bytes": 5, "expected_raw_bytes": 5},
        ]
    }
    assert REPLAY_RUNNER._declared_capacity_bytes(inventory) == 12
    with pytest.raises(REPLAY_RUNNER.RematerializationError, match="declared capacity invalid"):
        REPLAY_RUNNER._declared_capacity_bytes(
            {"sources": [{"declared_capacity_bytes": True}]}
        )


def test_current_clean_bindings_and_combined_capacity_are_exact() -> None:
    assert REPLAY_RUNNER.CURRENT_MAIN_AT_CONVERGENCE == (
        "7b3df41c10a826183fab0b04ae85a90cdf0ce351"
    )
    assert REPLAY_RUNNER.CURRENT_CLEAN_BASE_PR == 2107
    assert REPLAY_RUNNER.CURRENT_INDEXED_EXECUTOR_PR == 1459
    assert REPLAY_RUNNER.EXPECTED_BASE_PRE_DEDUP_SOURCE_COUNT == 263
    assert REPLAY_RUNNER.EXPECTED_BASE_PRE_DEDUP_BYTES == 6_093_965
    assert REPLAY_RUNNER.EXPECTED_EXTENSION_SOURCE_COUNT == 1_280
    assert REPLAY_RUNNER.EXPECTED_EXTENSION_BYTES == 3_949_184
    assert REPLAY_RUNNER.EXPECTED_COMBINED_PRE_DEDUP_SOURCE_COUNT == 1_543
    assert REPLAY_RUNNER.EXPECTED_COMBINED_PRE_DEDUP_BYTES == 10_043_149


def test_legacy_contaminated_execution_helpers_are_not_exposed() -> None:
    assert not hasattr(REPLAY_RUNNER, "_prepare_parent_execution_tree")
    assert not hasattr(REPLAY_RUNNER, "_replay_command")
    source = Path(REPLAY_RUNNER.__file__).read_text(encoding="utf-8")
    assert "build_post_admission_intake(" not in source
    assert "PARENT_RUNNER_BLOB_SHA1" not in source
    assert "PARENT_INTAKE_BLOB_SHA1" not in source


def test_build_retains_terminal_namespace_through_indexed_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v7_root = tmp_path / "v7"
    data_root = v7_root / "src/twelve_six/data"
    data_root.mkdir(parents=True)

    foreign = ModuleType("twelve_six.data.sentinel")
    previous = sys.modules.get("twelve_six.data.sentinel")
    sys.modules["twelve_six.data.sentinel"] = foreign
    observed: list[str] = []

    def fake_reconstruct(*, v7_root: Path, workspace: Path):
        del workspace
        assert sys.modules["twelve_six"].__path__ == [
            str((v7_root / "src/twelve_six").resolve())
        ]
        assert sys.modules["twelve_six.data"].__path__ == [
            str((v7_root / "src/twelve_six/data").resolve())
        ]
        assert "twelve_six.data.sentinel" not in sys.modules
        observed.append("reconstruct")
        return object(), {"sources": []}, {}, {}

    def fake_finish(*args, **kwargs):
        del args, kwargs
        assert "twelve_six" in sys.modules
        assert "twelve_six.data" in sys.modules
        assert "twelve_six.data.sentinel" not in sys.modules
        observed.append("indexed-replay")
        return {"report": True}, {"survivors": True}

    monkeypatch.setattr(
        REPLAY_RUNNER,
        "_verify_current_clean_dependencies",
        lambda: {"binding": "ok"},
    )
    monkeypatch.setattr(
        REPLAY_RUNNER,
        "_reconstruct_current_clean_base",
        fake_reconstruct,
    )
    monkeypatch.setattr(
        REPLAY_RUNNER,
        "_build_current_clean_replay_with_base",
        fake_finish,
    )

    args = SimpleNamespace(v7_root=v7_root)
    try:
        report, survivors = REPLAY_RUNNER._build_current_clean_replay(
            args,
            pass_root=tmp_path / "pass",
            arxiv_candidate=tmp_path / "arxiv.jsonl",
            languk_candidate=tmp_path / "languk.jsonl",
            source_intake=object(),
            indexed_executor=object(),
        )
        assert report == {"report": True}
        assert survivors == {"survivors": True}
        assert observed == ["reconstruct", "indexed-replay"]
        assert sys.modules.get("twelve_six.data.sentinel") is foreign
    finally:
        if previous is None:
            sys.modules.pop("twelve_six.data.sentinel", None)
        else:
            sys.modules["twelve_six.data.sentinel"] = previous


def test_terminal_v7_namespace_bypasses_init_and_restores_current_modules(
    tmp_path: Path,
) -> None:
    v7_root = tmp_path / "v7"
    package = v7_root / "src/twelve_six"
    data = package / "data"
    data.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "raise RuntimeError('historical root init must not run')\n",
        encoding="utf-8",
    )
    (data / "__init__.py").write_text(
        "raise RuntimeError('historical data init must not run')\n",
        encoding="utf-8",
    )
    (data / "probe.py").write_text("AUTHORITY = 'terminal-v7'\n", encoding="utf-8")

    foreign = ModuleType("twelve_six.data.probe")
    foreign.AUTHORITY = "current-main"
    previous = {
        name: sys.modules.get(name)
        for name in ("twelve_six", "twelve_six.data", "twelve_six.data.probe")
    }
    sys.modules["twelve_six.data.probe"] = foreign
    old_path = list(sys.path)
    try:
        with REPLAY_RUNNER._isolated_terminal_v7_namespace(v7_root):
            loaded = __import__(
                "twelve_six.data.probe",
                fromlist=["AUTHORITY"],
            )
            assert loaded.AUTHORITY == "terminal-v7"
            assert loaded is not foreign
            assert sys.modules["twelve_six"].__path__ == [str(package.resolve())]
            assert sys.modules["twelve_six.data"].__path__ == [str(data.resolve())]
        assert sys.modules.get("twelve_six.data.probe") is foreign
        assert sys.path == old_path
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_wrapper_keeps_current_package_out_of_terminal_v7_bootstrap_path() -> None:
    source = Path(REPLAY_RUNNER.__file__).read_text(encoding="utf-8")
    assert "from twelve_six.data.arxiv_languk_rematerialized_replay_v1 import" not in source
    assert REPLAY_RUNNER.EXPECTED_V7_HEAD == (
        "d3333ec1b4a508df232a5aefccd6686adda745fb"
    )
    assert REPLAY_RUNNER.EXPECTED_V7_TREE == "f6bb58379e9e249583480c246b844b673be38b4c"
    assert REPLAY_RUNNER.sys.dont_write_bytecode is True
