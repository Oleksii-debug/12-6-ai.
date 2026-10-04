from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
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

def _disk_pass_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    arxiv = tmp_path / "arxiv.jsonl"
    languk = tmp_path / "languk.jsonl"
    report = tmp_path / "report.json"
    survivors = tmp_path / "survivors.json"
    arxiv.write_bytes(b"candidate-arxiv")
    languk.write_bytes(b"candidate-languk")
    body = {"truth_boundary": {"canonical_capacity_credited": 0}}
    report_value = {
        **body,
        "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }
    survivor_value = {
        "v8_report_sha256": report_value["report_sha256"],
        "survivor_authority_sha256": "a" * 64,
    }
    report.write_bytes(canonical_json_bytes(report_value))
    survivors.write_bytes(canonical_json_bytes(survivor_value))
    return arxiv, languk, report, survivors


def _fake_disk_verifier(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    observed: list[str] = []

    def verify(report: dict[str, object], authority: dict[str, object]) -> None:
        observed.append("verified")
        if (
            authority["v8_report_sha256"] != report["report_sha256"]
            or authority["survivor_authority_sha256"] != "a" * 64
        ):
            raise ValueError("survivor authority mismatch")

    monkeypatch.setattr(REPLAY_RUNNER, "verify_candidate", lambda *_: None)
    monkeypatch.setattr(
        REPLAY_RUNNER,
        "_load_module",
        lambda *args: SimpleNamespace(verify_survivor_authority=verify),
    )
    return observed


def test_disk_pass_rechecks_serialized_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _disk_pass_files(tmp_path)
    observed = _fake_disk_verifier(monkeypatch)
    result = REPLAY_RUNNER._pass_result(
        arxiv_candidate=paths[0],
        languk_candidate=paths[1],
        report=paths[2],
        survivors=paths[3],
    )
    assert result["report_file_sha256"] == hashlib.sha256(
        paths[2].read_bytes()
    ).hexdigest()
    assert observed == ["verified"]


def test_disk_pass_rejects_modified_report_with_stale_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    arxiv, languk, report, survivors = _disk_pass_files(tmp_path)
    _fake_disk_verifier(monkeypatch)
    value = json.loads(report.read_text(encoding="utf-8"))
    value["truth_boundary"]["canonical_capacity_credited"] = 1
    report.write_bytes(canonical_json_bytes(value))
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError, match="replay report self-hash mismatch"
    ):
        REPLAY_RUNNER._pass_result(
            arxiv_candidate=arxiv, languk_candidate=languk,
            report=report, survivors=survivors,
        )


def test_disk_pass_rejects_modified_survivor_with_stale_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    arxiv, languk, report, survivors = _disk_pass_files(tmp_path)
    observed = _fake_disk_verifier(monkeypatch)
    value = json.loads(survivors.read_text(encoding="utf-8"))
    value["survivor_authority_sha256"] = "b" * 64
    survivors.write_bytes(canonical_json_bytes(value))
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError, match="survivor authority invalid"
    ):
        REPLAY_RUNNER._pass_result(
            arxiv_candidate=arxiv, languk_candidate=languk,
            report=report, survivors=survivors,
        )
    assert observed == ["verified"]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"a":1,"a":2}', "duplicate JSON key"),
        (b'{"a":NaN}', "nonfinite JSON"),
        (b'{"a":Infinity}', "nonfinite JSON"),
        (b'{"a":1} trailing', "cannot decode"),
    ],
)
def test_disk_json_loader_rejects_ambiguous_inputs(raw: bytes, message: str) -> None:
    with pytest.raises(REPLAY_RUNNER.RematerializationError, match=message):
        REPLAY_RUNNER._load_json_object(raw, label="test authority")


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


@pytest.mark.parametrize("changed", ["report", "survivors"])
def test_postverification_mutation_blocks_all_outer_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    pass_root = tmp_path / "pass-1"
    pass_root.mkdir()
    arxiv, languk, report, survivors = _disk_pass_files(pass_root)
    _fake_disk_verifier(monkeypatch)
    result = REPLAY_RUNNER._pass_result(
        arxiv_candidate=arxiv, languk_candidate=languk,
        report=report, survivors=survivors,
    )
    verified_report = pass_root / "current-clean-report.json"
    verified_survivors = pass_root / "current-clean-survivors.json"
    verified_report.write_bytes(report.read_bytes())
    verified_survivors.write_bytes(survivors.read_bytes())
    modified = verified_report if changed == "report" else verified_survivors
    modified.write_bytes(modified.read_bytes() + b"tampered")
    args = _runner_args(tmp_path)
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError,
        match="changed after pass verification",
    ):
        REPLAY_RUNNER._publish_verified_outputs(
            args, pass_root=pass_root, pass_result=result,
            receipt={"status": "not-published"},
        )
    for path in (args.output_report, args.output_survivors, args.output_receipt):
        assert not path.exists()


def test_publishing_uses_captured_bytes_if_workspace_changes_during_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root = tmp_path / "pass-1"
    pass_root.mkdir()
    arxiv, languk, report, survivors = _disk_pass_files(pass_root)
    _fake_disk_verifier(monkeypatch)
    result = REPLAY_RUNNER._pass_result(
        arxiv_candidate=arxiv, languk_candidate=languk,
        report=report, survivors=survivors,
    )
    saved_report, saved_survivors = report.read_bytes(), survivors.read_bytes()
    verified_report = pass_root / "current-clean-report.json"
    verified_survivors = pass_root / "current-clean-survivors.json"
    verified_report.write_bytes(saved_report)
    verified_survivors.write_bytes(saved_survivors)
    args = _runner_args(tmp_path)
    receipt = {"status": "verified"}
    actual_link = REPLAY_RUNNER._link_staged_new_bytes

    def mutate_after_first_link(staged: Path, path: Path, *, label: str) -> None:
        actual_link(staged, path, label=label)
        if label == "outer report":
            verified_report.write_bytes(b"tampered-report")
            verified_survivors.write_bytes(b"tampered-survivors")

    monkeypatch.setattr(REPLAY_RUNNER, "_link_staged_new_bytes", mutate_after_first_link)
    REPLAY_RUNNER._publish_verified_outputs(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    assert args.output_report.read_bytes() == saved_report
    assert args.output_survivors.read_bytes() == saved_survivors
    assert args.output_receipt.read_bytes() == canonical_json_bytes(receipt)
    assert REPLAY_RUNNER.sha256_bytes(args.output_report.read_bytes()) == result[
        "report_file_sha256"
    ]
    assert REPLAY_RUNNER.sha256_bytes(args.output_survivors.read_bytes()) == result[
        "survivor_file_sha256"
    ]


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


@pytest.mark.parametrize("name", ["historical", "pass-1", "pass-2"])
def test_workspace_preflight_preserves_existing_evidence(tmp_path: Path, name: str) -> None:
    workspace = tmp_path / "workspace"
    occupied = workspace / name
    occupied.mkdir(parents=True)
    evidence = occupied / "existing-evidence.json"
    evidence.write_text('{"important":true}', encoding="utf-8")
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError,
        match="refusing to reuse workspace evidence",
    ):
        REPLAY_RUNNER._verify_workspace_targets(workspace)
    assert evidence.read_text(encoding="utf-8") == '{"important":true}'


def test_workspace_preflight_allows_empty_fresh_workspace(tmp_path: Path) -> None:
    REPLAY_RUNNER._verify_workspace_targets(tmp_path / "not-created")
    workspace = tmp_path / "empty"
    workspace.mkdir()
    REPLAY_RUNNER._verify_workspace_targets(workspace)


def test_workspace_preflight_rejects_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "important"
    target.mkdir()
    evidence = target / "evidence.txt"
    evidence.write_text("retain", encoding="utf-8")
    link = tmp_path / "workspace-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation unavailable")
    with pytest.raises(REPLAY_RUNNER.RematerializationError, match="real directory"):
        REPLAY_RUNNER._verify_workspace_targets(link)
    assert evidence.read_text(encoding="utf-8") == "retain"


def test_historical_extract_never_deletes_existing_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "historical" / "arxiv"
    destination.mkdir(parents=True)
    evidence = destination / "prior-result.json"
    evidence.write_text('{"sealed":true}', encoding="utf-8")

    def must_not_invoke_git(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("historical Git execution on occupied destination")

    monkeypatch.setattr(REPLAY_RUNNER, "_git", must_not_invoke_git)
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError,
        match="refusing to overwrite historical checkout",
    ):
        REPLAY_RUNNER._extract_commit(tmp_path, "a" * 40, destination)
    assert evidence.read_text(encoding="utf-8") == '{"sealed":true}'


def test_wrapper_keeps_current_package_out_of_terminal_v7_bootstrap_path() -> None:
    source = Path(REPLAY_RUNNER.__file__).read_text(encoding="utf-8")
    assert "from twelve_six.data.arxiv_languk_rematerialized_replay_v1 import" not in source
    assert REPLAY_RUNNER.EXPECTED_V7_HEAD == (
        "d3333ec1b4a508df232a5aefccd6686adda745fb"
    )
    assert REPLAY_RUNNER.EXPECTED_V7_TREE == "f6bb58379e9e249583480c246b844b673be38b4c"
    assert REPLAY_RUNNER.sys.dont_write_bytecode is True


def _publication_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, SimpleNamespace, dict[str, str], dict[str, object]]:
    pass_root = tmp_path / "pass-1"
    pass_root.mkdir()
    arxiv, languk, report, survivors = _disk_pass_files(pass_root)
    _fake_disk_verifier(monkeypatch)
    pass_result = REPLAY_RUNNER._pass_result(
        arxiv_candidate=arxiv, languk_candidate=languk,
        report=report, survivors=survivors,
    )
    (pass_root / "current-clean-report.json").write_bytes(report.read_bytes())
    (pass_root / "current-clean-survivors.json").write_bytes(survivors.read_bytes())
    return pass_root, _runner_args(tmp_path), pass_result, {"status": "zero-credit"}


def test_staging_failure_does_not_publish_any_outer_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    original_stage = REPLAY_RUNNER._stage_new_bytes

    def stage(path: Path, raw: bytes, *, label: str) -> Path:
        if label == "outer survivors":
            raise OSError("injected staging ENOSPC")
        return original_stage(path, raw, label=label)

    monkeypatch.setattr(REPLAY_RUNNER, "_stage_new_bytes", stage)
    with pytest.raises(OSError, match="injected staging ENOSPC"):
        REPLAY_RUNNER._publish_verified_outputs(
            args, pass_root=pass_root, pass_result=result, receipt=receipt,
        )
    assert not (pass_root / "outer-publication-intent.json").exists()
    for path in (args.output_report, args.output_survivors, args.output_receipt):
        assert not path.exists()
    assert not list(tmp_path.glob(".outer-*.tmp"))


def test_partial_outer_publication_keeps_authenticated_recovery_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    real_link = REPLAY_RUNNER._link_staged_new_bytes

    def fail_second(staged: Path, path: Path, *, label: str) -> None:
        if label == "outer survivors":
            raise REPLAY_RUNNER.RematerializationError("injected publish failure")
        real_link(staged, path, label=label)

    monkeypatch.setattr(REPLAY_RUNNER, "_link_staged_new_bytes", fail_second)
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError, match="manual reconciliation required",
    ):
        REPLAY_RUNNER._publish_verified_outputs(
            args, pass_root=pass_root, pass_result=result, receipt=receipt,
        )
    intent_path = pass_root / "outer-publication-intent.json"
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    assert intent["status"] == "PREPARED_UNCOMMITTED"
    assert intent["canonical_capacity_credited"] == 0
    assert intent["training_authorized"] is False
    assert intent["outputs"][0]["sha256"] == result["report_file_sha256"]
    assert args.output_report.read_bytes() == (
        pass_root / "current-clean-report.json"
    ).read_bytes()
    assert not args.output_survivors.exists()
    assert not args.output_receipt.exists()
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError, match="refusing to overwrite outer report",
    ):
        REPLAY_RUNNER._verify_outer_output_targets(args)
    assert not list(tmp_path.glob(".outer-*.tmp"))


def test_late_receipt_race_never_clobbers_untrusted_competing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    real_link = REPLAY_RUNNER._link_staged_new_bytes

    def compete(staged: Path, path: Path, *, label: str) -> None:
        if label == "outer receipt":
            path.write_bytes(b"competing publisher")
        real_link(staged, path, label=label)

    monkeypatch.setattr(REPLAY_RUNNER, "_link_staged_new_bytes", compete)
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError, match="commit receipt not verified",
    ):
        REPLAY_RUNNER._publish_verified_outputs(
            args, pass_root=pass_root, pass_result=result, receipt=receipt,
        )
    assert args.output_receipt.read_bytes() == b"competing publisher"
    assert args.output_report.exists() and args.output_survivors.exists()
    assert (pass_root / "outer-publication-intent.json").exists()
    assert not list(tmp_path.glob(".outer-*.tmp"))


def test_short_staged_write_never_creates_final_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fdopen = REPLAY_RUNNER.os.fdopen

    class ShortWriter:
        def __init__(self, original):
            self.original = original

        def __enter__(self):
            self.original.__enter__()
            return self

        def __exit__(self, *args):
            return self.original.__exit__(*args)

        def write(self, raw: bytes) -> int:
            self.original.write(raw[:1])
            return 1

    monkeypatch.setattr(
        REPLAY_RUNNER.os, "fdopen",
        lambda descriptor, mode: ShortWriter(real_fdopen(descriptor, mode)),
    )
    path = tmp_path / "report.json"
    with pytest.raises(OSError, match="incomplete staged write"):
        REPLAY_RUNNER._write_new_bytes(path, b"complete-evidence", label="report")
    assert not path.exists()
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_fsync_failure_never_creates_final_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_sync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(REPLAY_RUNNER.os, "fsync", fail_sync)
    path = tmp_path / "report.json"
    with pytest.raises(OSError, match="injected fsync failure"):
        REPLAY_RUNNER._write_new_bytes(path, b"complete-evidence", label="report")
    assert not path.exists()
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_complete_outer_publication_receipt_commits_prepared_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    REPLAY_RUNNER._publish_verified_outputs(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    intent = json.loads(
        (pass_root / "outer-publication-intent.json").read_text(encoding="utf-8")
    )
    assert intent["status"] == "PREPARED_UNCOMMITTED"
    assert intent["outputs"][-1]["label"] == "outer receipt"
    for entry in intent["outputs"]:
        path = Path(entry["path"])
        assert REPLAY_RUNNER.sha256_bytes(path.read_bytes()) == entry["sha256"]
    assert args.output_receipt.read_bytes() == canonical_json_bytes(receipt)
    assert not list(tmp_path.glob(".outer-*.tmp"))


@pytest.mark.parametrize(
    "failed_label", ["outer report", "outer survivors", "outer receipt"],
)
def test_each_staging_failure_keeps_entire_final_batch_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_label: str,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    original_stage = REPLAY_RUNNER._stage_new_bytes

    def fail_staging(path: Path, raw: bytes, *, label: str) -> Path:
        if label == failed_label:
            raise OSError(f"staging failed: {label}")
        return original_stage(path, raw, label=label)

    monkeypatch.setattr(REPLAY_RUNNER, "_stage_new_bytes", fail_staging)
    with pytest.raises(OSError, match="staging failed"):
        REPLAY_RUNNER._publish_verified_outputs(
            args, pass_root=pass_root, pass_result=result, receipt=receipt,
        )
    assert not (pass_root / "outer-publication-intent.json").exists()
    assert not args.output_report.exists()
    assert not args.output_survivors.exists()
    assert not args.output_receipt.exists()
    assert not list(tmp_path.glob(".outer-*.tmp"))


@pytest.mark.parametrize(
    ("failed_label", "published"),
    [
        ("outer report", ()),
        ("outer survivors", ("outer report",)),
        ("outer receipt", ("outer report", "outer survivors")),
    ],
)
def test_each_link_failure_is_recoverable_without_partial_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    failed_label: str, published: tuple[str, ...],
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    original_link = REPLAY_RUNNER._link_staged_new_bytes
    targets = {
        "outer report": args.output_report,
        "outer survivors": args.output_survivors,
        "outer receipt": args.output_receipt,
    }

    def fail_publication(staged: Path, path: Path, *, label: str) -> None:
        if label == failed_label:
            raise REPLAY_RUNNER.RematerializationError(f"link failed: {label}")
        original_link(staged, path, label=label)

    monkeypatch.setattr(REPLAY_RUNNER, "_link_staged_new_bytes", fail_publication)
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError, match="manual reconciliation required",
    ):
        REPLAY_RUNNER._publish_verified_outputs(
            args, pass_root=pass_root, pass_result=result, receipt=receipt,
        )
    intent = json.loads(
        (pass_root / "outer-publication-intent.json").read_text(encoding="utf-8")
    )
    assert intent["training_authorized"] is False
    for item in intent["outputs"]:
        path = targets[item["label"]]
        if item["label"] in published:
            assert REPLAY_RUNNER.sha256_bytes(path.read_bytes()) == item["sha256"]
        else:
            assert not path.exists()
    assert not args.output_receipt.exists()
    assert not list(tmp_path.glob(".outer-*.tmp"))


def test_outer_output_cannot_replace_or_preempt_publication_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    args.output_survivors = pass_root / "outer-publication-intent.json"
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError, match="cannot alias publication intent",
    ):
        REPLAY_RUNNER._publish_verified_outputs(
            args, pass_root=pass_root, pass_result=result, receipt=receipt,
        )
    assert not args.output_report.exists()
    assert not args.output_survivors.exists()
    assert not args.output_receipt.exists()
    assert not list(tmp_path.glob(".outer-*.tmp"))


def _inspect_in_fresh_process(
    pass_root: Path, args: SimpleNamespace,
    pass_result: dict[str, str], receipt: dict[str, object],
) -> dict[str, object]:
    """A new interpreter imports the runner and validates persisted evidence."""
    input_value = {
        "pass_root": str(pass_root),
        "output_report": str(args.output_report),
        "output_survivors": str(args.output_survivors),
        "output_receipt": str(args.output_receipt),
        "pass_result": pass_result,
        "receipt": receipt,
    }
    script = """
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "_pr1851_fresh_recovery_consumer",
    Path("tools/run_d03_arxiv_languk_rematerialized_replay_v1.py"),
)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
data = json.loads(sys.argv[1])
args = SimpleNamespace(
    output_report=Path(data["output_report"]),
    output_survivors=Path(data["output_survivors"]),
    output_receipt=Path(data["output_receipt"]),
)
result = runner.inspect_outer_publication_recovery(
    args, pass_root=Path(data["pass_root"]),
    pass_result=data["pass_result"], receipt=data["receipt"],
)
print(json.dumps(result, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, json.dumps(input_value)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, encoding="utf-8",
        timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout)


def test_fresh_runner_inspects_complete_receipt_without_republishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    REPLAY_RUNNER._publish_verified_outputs(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    # Import a fresh consumer module: this is a read-only recovery decision.
    recovered = _inspect_in_fresh_process(
        pass_root, args, result, receipt,
    )
    assert recovered["status"] == "COMMITTED_ZERO_CREDIT"
    assert recovered["missing"] == []
    assert recovered["canonical_capacity_credited"] == 0
    assert recovered["training_authorized"] is False
    assert len(recovered["published"]) == 3


@pytest.mark.parametrize("stop_after", [0, 1, 2])
def test_fresh_runner_classifies_interrupted_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_after: int,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    original_link = REPLAY_RUNNER._link_staged_new_bytes
    seen: list[str] = []

    def fail_after_prefix(staged: Path, path: Path, *, label: str) -> None:
        if label != "publication intent":
            if len(seen) == stop_after:
                raise REPLAY_RUNNER.RematerializationError("injected crash")
            seen.append(label)
        original_link(staged, path, label=label)

    monkeypatch.setattr(REPLAY_RUNNER, "_link_staged_new_bytes", fail_after_prefix)
    with pytest.raises(REPLAY_RUNNER.RematerializationError, match="manual reconciliation"):
        REPLAY_RUNNER._publish_verified_outputs(
            args, pass_root=pass_root, pass_result=result, receipt=receipt,
        )
    recovered = _inspect_in_fresh_process(
        pass_root, args, result, receipt,
    )
    assert recovered["status"] == (
        "PREPARED_UNCOMMITTED" if stop_after == 0 else "PARTIAL_UNCOMMITTED"
    )
    assert len(recovered["published"]) == stop_after
    assert len(recovered["missing"]) == 3 - stop_after
    assert not args.output_receipt.exists()
    assert recovered["training_authorized"] is False


@pytest.mark.parametrize("tamper", ["journal", "report", "survivors", "receipt"])
def test_recovery_never_accepts_mutated_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    REPLAY_RUNNER._publish_verified_outputs(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    targets = {
        "journal": pass_root / "outer-publication-intent.json",
        "report": args.output_report,
        "survivors": args.output_survivors,
        "receipt": args.output_receipt,
    }
    target = targets[tamper]
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(REPLAY_RUNNER.RematerializationError, match="recovery"):
        REPLAY_RUNNER.inspect_outer_publication_recovery(
            args, pass_root=pass_root, pass_result=result, receipt=receipt,
        )


def test_recovery_refuses_receipt_without_both_verified_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    REPLAY_RUNNER._publish_verified_outputs(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    args.output_survivors.unlink()
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError,
        match="receipt exists without both authenticated source authorities",
    ):
        REPLAY_RUNNER.inspect_outer_publication_recovery(
            args, pass_root=pass_root, pass_result=result, receipt=receipt,
        )


@pytest.mark.parametrize("phase", ["before-link", "after-link"])
@pytest.mark.parametrize(
    ("bad_label", "expected_published"),
    [
        ("outer report", ()),
        ("outer survivors", ("outer report",)),
        ("outer receipt", ("outer report", "outer survivors")),
    ],
)
def test_mutated_staged_outer_output_cannot_commit_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    phase: str, bad_label: str, expected_published: tuple[str, ...],
) -> None:
    """A corrupt stage cannot get a successful receipt at the link boundary."""
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    original_link = REPLAY_RUNNER._link_staged_new_bytes

    def corrupt_stage(staged: Path, final: Path, *, label: str) -> None:
        if label == bad_label and phase == "before-link":
            staged.write_bytes(b"modified after staging")
        original_link(staged, final, label=label)
        if label == bad_label and phase == "after-link":
            staged.write_bytes(b"modified after linking")

    monkeypatch.setattr(REPLAY_RUNNER, "_link_staged_new_bytes", corrupt_stage)
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError, match="manual reconciliation required",
    ):
        REPLAY_RUNNER._publish_verified_outputs(
            args, pass_root=pass_root, pass_result=result, receipt=receipt,
        )

    targets = {
        "outer report": args.output_report,
        "outer survivors": args.output_survivors,
        "outer receipt": args.output_receipt,
    }
    raw = {
        "outer report": (pass_root / "current-clean-report.json").read_bytes(),
        "outer survivors": (pass_root / "current-clean-survivors.json").read_bytes(),
        "outer receipt": REPLAY_RUNNER.canonical_json_bytes(receipt),
    }
    for label, path in targets.items():
        if label in expected_published:
            assert path.read_bytes() == raw[label]
        else:
            assert not path.exists()
    assert not args.output_receipt.exists()
    assert not list(tmp_path.glob(".outer-*.tmp"))
    recovery = REPLAY_RUNNER.inspect_outer_publication_recovery(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    assert recovery["published"] == list(expected_published)
    assert recovery["status"] == (
        "PREPARED_UNCOMMITTED" if not expected_published else "PARTIAL_UNCOMMITTED"
    )
    assert recovery["canonical_capacity_credited"] == 0
    assert recovery["training_authorized"] is False


@pytest.mark.parametrize("phase", ["before-link", "after-link"])
def test_mutated_intent_stage_cannot_publish_outer_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    original_link = REPLAY_RUNNER._link_staged_new_bytes

    def corrupt_intent(staged: Path, final: Path, *, label: str) -> None:
        if label == "publication intent" and phase == "before-link":
            staged.write_bytes(b"forged intent")
        original_link(staged, final, label=label)
        if label == "publication intent" and phase == "after-link":
            staged.write_bytes(b"forged intent")

    monkeypatch.setattr(REPLAY_RUNNER, "_link_staged_new_bytes", corrupt_intent)
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError,
        match="changed before publication|failed exact byte/inode verification",
    ):
        REPLAY_RUNNER._publish_verified_outputs(
            args, pass_root=pass_root, pass_result=result, receipt=receipt,
        )
    assert not (pass_root / "outer-publication-intent.json").exists()
    assert not args.output_report.exists()
    assert not args.output_survivors.exists()
    assert not args.output_receipt.exists()
    assert not list(tmp_path.glob(".outer-*.tmp"))


def test_verified_one_output_preserves_utf8_and_no_replace(tmp_path: Path) -> None:
    target = tmp_path / "Український результат.json"
    payload = '{"status":"verified","language":"Українська"}\n'.encode()
    REPLAY_RUNNER._write_new_bytes(target, payload, label="test report")
    assert target.read_bytes() == payload
    with pytest.raises(REPLAY_RUNNER.RematerializationError, match="refusing to overwrite"):
        REPLAY_RUNNER._write_new_bytes(target, b"replacement", label="test report")
    assert target.read_bytes() == payload
    assert not list(tmp_path.glob(".*.tmp"))


def test_replaced_stage_inode_does_not_delete_foreign_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An open original descriptor prevents inode reuse after a stage-path swap."""
    final = tmp_path / "outer-report.json"
    original_link = REPLAY_RUNNER._link_staged_new_bytes

    def replace_stage(staged: Path, path: Path, *, label: str) -> None:
        try:
            staged.unlink()
        except OSError:
            pytest.skip("unlink of an open staged file is unsupported")
        staged.write_bytes(b"foreign replacement")
        original_link(staged, path, label=label)

    monkeypatch.setattr(REPLAY_RUNNER, "_link_staged_new_bytes", replace_stage)
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError, match="manual reconciliation required",
    ):
        REPLAY_RUNNER._write_new_bytes(final, b"original", label="outer report")
    assert final.read_bytes() == b"foreign replacement"
    assert not list(tmp_path.glob(".outer-report.json.*.tmp"))


def test_failed_owned_rollback_is_reported_for_manual_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "outer-report.json"
    original_link = REPLAY_RUNNER._link_staged_new_bytes
    original_unlink = Path.unlink

    def corrupt_after_link(staged: Path, path: Path, *, label: str) -> None:
        original_link(staged, path, label=label)
        staged.write_bytes(b"corrupt-but-owned")

    def fail_final_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == final:
            raise OSError("injected rollback failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(REPLAY_RUNNER, "_link_staged_new_bytes", corrupt_after_link)
    monkeypatch.setattr(Path, "unlink", fail_final_unlink)
    with pytest.raises(
        REPLAY_RUNNER.PublicationIndeterminate, match="ROLLBACK_INCOMPLETE",
    ):
        REPLAY_RUNNER._write_new_bytes(final, b"expected", label="outer report")
    assert final.read_bytes() == b"corrupt-but-owned"
    staged = list(tmp_path.glob(".outer-report.json.*.tmp"))
    assert len(staged) == 1 and staged[0].read_bytes() == b"corrupt-but-owned"
    monkeypatch.setattr(Path, "unlink", original_unlink)
    staged[0].unlink()
    final.unlink()


def test_verified_single_output_cleanup_error_is_not_a_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "single-output.json"
    payload = b'{"verified":true}\n'
    original_unlink = Path.unlink

    def deny_stage_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(".single-output.json.") and path.suffix == ".tmp":
            raise OSError("injected cleanup sharing violation")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_stage_cleanup)
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError,
        match="was published and byte-verified, but staged cleanup is pending",
    ):
        REPLAY_RUNNER._write_new_bytes(final, payload, label="single output")
    assert final.read_bytes() == payload
    staged = list(tmp_path.glob(".single-output.json.*.tmp"))
    assert len(staged) == 1 and staged[0].read_bytes() == payload
    monkeypatch.setattr(Path, "unlink", original_unlink)
    staged[0].unlink()
    assert final.read_bytes() == payload


def test_prepublication_cleanup_error_preserves_failed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "failed-output.json"
    original_unlink = Path.unlink

    def refuse_link(staged: Path, path: Path, *, label: str) -> None:
        raise REPLAY_RUNNER.RematerializationError("injected link failure")

    def deny_stage_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(".failed-output.json.") and path.suffix == ".tmp":
            raise OSError("injected cleanup error")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(REPLAY_RUNNER, "_link_staged_new_bytes", refuse_link)
    monkeypatch.setattr(Path, "unlink", deny_stage_cleanup)
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError,
        match="publication failed and staged cleanup is pending",
    ) as caught:
        REPLAY_RUNNER._write_new_bytes(final, b"expected", label="failed output")
    assert "injected link failure" in str(caught.value)
    assert "injected cleanup error" in str(caught.value)
    assert isinstance(caught.value.__cause__, REPLAY_RUNNER.RematerializationError)
    assert not final.exists()
    staged = list(tmp_path.glob(".failed-output.json.*.tmp"))
    assert len(staged) == 1
    monkeypatch.setattr(Path, "unlink", original_unlink)
    staged[0].unlink()


def test_receipt_commit_with_cleanup_error_is_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    original_unlink = Path.unlink

    def deny_report_stage(path: Path, *args: object, **kwargs: object) -> None:
        if (
            path.parent == args.output_report.parent
            and path.name.startswith(f".{args.output_report.name}.")
            and path.suffix == ".tmp"
        ):
            raise OSError("injected outer cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_report_stage)
    with pytest.raises(
        REPLAY_RUNNER.RematerializationError,
        match="outer receipt published and byte-verified; staged cleanup pending",
    ):
        REPLAY_RUNNER._publish_verified_outputs(
            args, pass_root=pass_root, pass_result=result, receipt=receipt,
        )
    assert args.output_receipt.read_bytes() == REPLAY_RUNNER.canonical_json_bytes(
        receipt
    )
    recovered = REPLAY_RUNNER.inspect_outer_publication_recovery(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    assert recovered["status"] == "COMMITTED_ZERO_CREDIT"
    assert recovered["training_authorized"] is False
    staged = list(args.output_report.parent.glob(f".{args.output_report.name}.*.tmp"))
    assert len(staged) == 1
    monkeypatch.setattr(Path, "unlink", original_unlink)
    staged[0].unlink()


def test_staging_dual_fault_reports_orphan_and_preserves_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed fsync plus locked cleanup must expose the exact orphan path."""
    target = tmp_path / "Український результат.json"
    payload = b"verified bytes"
    blocked: list[Path] = []

    def fail_fsync(descriptor: int) -> None:
        del descriptor
        raise OSError("injected ENOSPC during fsync")

    original_unlink = Path.unlink

    def lock_stage(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(f".{target.name}.") and path.suffix == ".tmp":
            blocked.append(path)
            raise PermissionError("injected sharing violation")
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(REPLAY_RUNNER.os, "fsync", fail_fsync)
        fault.setattr(Path, "unlink", lock_stage)
        with pytest.raises(
            REPLAY_RUNNER.RematerializationError, match="STAGING_CLEANUP_INCOMPLETE",
        ) as caught:
            REPLAY_RUNNER._write_new_bytes(target, payload, label="test report")

    assert not target.exists()
    assert len(blocked) == 1
    assert str(blocked[0]) in str(caught.value)
    assert "manual reconciliation required" in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)
    assert "ENOSPC" in str(caught.value.__cause__)
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == blocked
    assert blocked[0].read_bytes() == payload

    blocked[0].unlink()
    REPLAY_RUNNER._write_new_bytes(target, payload, label="test report")
    assert target.read_bytes() == payload
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))


def test_outer_staging_dual_fault_cleans_prior_stages_without_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failed current stage is reported even if not in the batch registry."""
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    original_fsync = REPLAY_RUNNER.os.fsync
    original_unlink = Path.unlink
    calls = 0
    blocked: list[Path] = []

    def fail_second_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected ENOSPC on survivors stage")
        original_fsync(descriptor)

    def lock_survivors_stage(path: Path, *args: object, **kwargs: object) -> None:
        if (
            path.name.startswith(f".{args.output_survivors.name}.")
            and path.suffix == ".tmp"
        ):
            blocked.append(path)
            raise PermissionError("injected survivors sharing violation")
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(REPLAY_RUNNER.os, "fsync", fail_second_fsync)
        fault.setattr(Path, "unlink", lock_survivors_stage)
        with pytest.raises(
            REPLAY_RUNNER.RematerializationError, match="STAGING_CLEANUP_INCOMPLETE",
        ) as caught:
            REPLAY_RUNNER._publish_verified_outputs(
                args, pass_root=pass_root, pass_result=result, receipt=receipt,
            )

    assert calls == 2
    assert not (pass_root / "outer-publication-intent.json").exists()
    assert not any(
        path.exists()
        for path in (args.output_report, args.output_survivors, args.output_receipt)
    )
    assert not list(tmp_path.glob(f".{args.output_report.name}.*.tmp"))
    assert len(blocked) == 1
    assert blocked[0].exists()
    assert str(blocked[0]) in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)
    assert "ENOSPC" in str(caught.value.__cause__)

    blocked[0].unlink()
    REPLAY_RUNNER._publish_verified_outputs(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    recovered = REPLAY_RUNNER.inspect_outer_publication_recovery(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    assert recovered["status"] == "COMMITTED_ZERO_CREDIT"
    assert recovered["canonical_capacity_credited"] == 0
    assert recovered["training_authorized"] is False


def test_outer_staging_and_prior_cleanup_dual_faults_report_both_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prior cleanup fault must not hide the unregistered current orphan."""
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    original_fsync = REPLAY_RUNNER.os.fsync
    original_unlink = Path.unlink
    calls = 0
    blocked: list[Path] = []

    def fail_second_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected ENOSPC during survivors staging")
        original_fsync(descriptor)

    def lock_both_stages(path: Path, *args_: object, **kwargs: object) -> None:
        if (
            path.suffix == ".tmp"
            and (
                path.name.startswith(f".{args.output_report.name}.")
                or path.name.startswith(f".{args.output_survivors.name}.")
            )
        ):
            blocked.append(path)
            raise PermissionError("injected stage sharing violation")
        original_unlink(path, *args_, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(REPLAY_RUNNER.os, "fsync", fail_second_fsync)
        fault.setattr(Path, "unlink", lock_both_stages)
        with pytest.raises(
            REPLAY_RUNNER.RematerializationError,
            match="outer publication incomplete and staged cleanup pending",
        ) as caught:
            REPLAY_RUNNER._publish_verified_outputs(
                args, pass_root=pass_root, pass_result=result, receipt=receipt,
            )

    assert calls == 2
    assert len(blocked) == 2
    assert len(set(blocked)) == 2
    assert all(path.exists() for path in blocked)
    assert all(str(path) in str(caught.value) for path in blocked)
    assert "STAGING_CLEANUP_INCOMPLETE" in str(caught.value)
    assert "ENOSPC" in str(caught.value)
    assert isinstance(caught.value.__cause__, REPLAY_RUNNER.RematerializationError)
    assert isinstance(caught.value.__cause__.__cause__, OSError)
    assert not (pass_root / "outer-publication-intent.json").exists()
    assert not any(
        path.exists()
        for path in (args.output_report, args.output_survivors, args.output_receipt)
    )

    for path in blocked:
        path.unlink()
    REPLAY_RUNNER._publish_verified_outputs(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    recovery = REPLAY_RUNNER.inspect_outer_publication_recovery(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    assert recovery["status"] == "COMMITTED_ZERO_CREDIT"
    assert recovery["training_authorized"] is False


def test_create_then_raise_rolls_back_only_owned_single_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "Український результат.json"
    payload = b"authenticated result"
    original_link = REPLAY_RUNNER.os.link

    def create_then_raise(stage: Path, final: Path) -> None:
        original_link(stage, final)
        raise OSError("injected post-create EIO")

    with monkeypatch.context() as fault:
        fault.setattr(REPLAY_RUNNER.os, "link", create_then_raise)
        with pytest.raises(
            REPLAY_RUNNER.RematerializationError, match="cannot publish",
        ):
            REPLAY_RUNNER._write_new_bytes(target, payload, label="test output")

    assert not target.exists()
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))
    REPLAY_RUNNER._write_new_bytes(target, payload, label="test output")
    assert target.read_bytes() == payload


@pytest.mark.parametrize(
    ("label", "prior"),
    [
        ("outer report", ()),
        ("outer survivors", ("outer report",)),
        ("outer receipt", ("outer report", "outer survivors")),
    ],
)
def test_create_then_raise_cannot_commit_outer_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    label: str, prior: tuple[str, ...],
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    targets = {
        "outer report": args.output_report,
        "outer survivors": args.output_survivors,
        "outer receipt": args.output_receipt,
    }
    original_link = REPLAY_RUNNER.os.link

    def create_then_raise(stage: Path, final: Path) -> None:
        original_link(stage, final)
        if final == targets[label]:
            raise OSError("injected post-create EIO")

    with monkeypatch.context() as fault:
        fault.setattr(REPLAY_RUNNER.os, "link", create_then_raise)
        with pytest.raises(
            REPLAY_RUNNER.RematerializationError, match="partial outer publication",
        ):
            REPLAY_RUNNER._publish_verified_outputs(
                args, pass_root=pass_root, pass_result=result, receipt=receipt,
            )

    for name, target in targets.items():
        assert target.exists() is (name in prior)
    assert not args.output_receipt.exists()
    assert not list(tmp_path.glob(".outer-*.tmp"))
    for target in targets.values():
        target.unlink(missing_ok=True)
    (pass_root / "outer-publication-intent.json").unlink()
    REPLAY_RUNNER._publish_verified_outputs(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    recovery = REPLAY_RUNNER.inspect_outer_publication_recovery(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    assert recovery["status"] == "COMMITTED_ZERO_CREDIT"
    assert recovery["training_authorized"] is False


@pytest.mark.parametrize("link_raises", [False, True])
def test_uninspectable_final_keeps_original_stage_for_manual_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, link_raises: bool,
) -> None:
    target = tmp_path / "uninspectable-result.json"
    payload = b"exact original bytes"
    original_link = REPLAY_RUNNER.os.link
    original_stat = Path.stat

    def maybe_raise_after_link(stage: Path, final: Path) -> None:
        original_link(stage, final)
        if link_raises and final == target:
            raise OSError("injected post-create EIO")

    def deny_final_stat(path: Path, *args: object, **kwargs: object):
        if path == target and kwargs.get("follow_symlinks") is False:
            raise PermissionError("injected NTFS sharing denial")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(REPLAY_RUNNER.os, "link", maybe_raise_after_link)
        fault.setattr(Path, "stat", deny_final_stat)
        with pytest.raises(
            REPLAY_RUNNER.PublicationIndeterminate,
            match="PUBLICATION_INDETERMINATE",
        ) as caught:
            REPLAY_RUNNER._write_new_bytes(target, payload, label="test output")

    staged = list(tmp_path.glob(f".{target.name}.*.tmp"))
    assert len(staged) == 1
    assert staged[0] == caught.value.staged
    assert str(staged[0]) in str(caught.value)
    assert str(target) in str(caught.value)
    assert target.read_bytes() == staged[0].read_bytes() == payload
    target.unlink()
    staged[0].unlink()
    REPLAY_RUNNER._write_new_bytes(target, payload, label="test output")
    assert target.read_bytes() == payload


def test_uninspectable_outer_report_retains_stage_and_never_commits_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    original_stat = Path.stat

    def deny_report_stat(path: Path, *args: object, **kwargs: object):
        if path == args.output_report and kwargs.get("follow_symlinks") is False:
            raise PermissionError("injected final inode inspection denial")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "stat", deny_report_stat)
        with pytest.raises(
            REPLAY_RUNNER.RematerializationError,
            match="PUBLICATION_INDETERMINATE",
        ):
            REPLAY_RUNNER._publish_verified_outputs(
                args, pass_root=pass_root, pass_result=result, receipt=receipt,
            )

    retained = list(
        args.output_report.parent.glob(f".{args.output_report.name}.*.tmp")
    )
    assert len(retained) == 1
    assert retained[0].read_bytes() == args.output_report.read_bytes()
    assert not args.output_survivors.exists()
    assert not args.output_receipt.exists()
    retained[0].unlink()
    args.output_report.unlink()
    (pass_root / "outer-publication-intent.json").unlink()
    REPLAY_RUNNER._publish_verified_outputs(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    recovery = REPLAY_RUNNER.inspect_outer_publication_recovery(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    assert recovery["status"] == "COMMITTED_ZERO_CREDIT"
    assert recovery["training_authorized"] is False


def test_eexist_same_inode_cannot_authorize_removal_of_preexisting_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "existing-output.json"
    original_link = REPLAY_RUNNER.os.link

    def preexisting_alias_then_eexist(
        stage: Path, final: Path, *, label: str,
    ) -> None:
        assert label == "test output"
        original_link(stage, final)
        raise REPLAY_RUNNER.RematerializationError(
            "refusing to overwrite test output"
        ) from FileExistsError("injected EEXIST")

    with monkeypatch.context() as fault:
        fault.setattr(
            REPLAY_RUNNER, "_link_staged_new_bytes", preexisting_alias_then_eexist,
        )
        with pytest.raises(
            REPLAY_RUNNER.PublicationIndeterminate,
            match="PUBLICATION_INDETERMINATE",
        ):
            REPLAY_RUNNER._write_new_bytes(
                target, b"original", label="test output",
            )

    retained = list(tmp_path.glob(".existing-output.json.*.tmp"))
    assert len(retained) == 1
    assert target.read_bytes() == retained[0].read_bytes() == b"original"
    retained[0].unlink()
    target.unlink()
    REPLAY_RUNNER._write_new_bytes(target, b"original", label="test output")
    assert target.read_bytes() == b"original"


def test_link_and_cleanup_dual_fault_preserves_original_io_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "дві-помилки.json"
    original_unlink = Path.unlink

    def fail_link(stage: Path, final: Path) -> None:
        raise OSError(f"injected link ENOSPC: {stage} -> {final}")

    def deny_stage_cleanup(
        path: Path, *args: object, **kwargs: object,
    ) -> None:
        if path.name.startswith(f".{target.name}.") and path.suffix == ".tmp":
            raise PermissionError("injected stage sharing violation")
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(REPLAY_RUNNER.os, "link", fail_link)
        fault.setattr(Path, "unlink", deny_stage_cleanup)
        with pytest.raises(
            REPLAY_RUNNER.RematerializationError,
            match="publication failed and staged cleanup is pending",
        ) as caught:
            REPLAY_RUNNER._write_new_bytes(target, b"trusted", label="test output")

    orphaned = list(tmp_path.glob(f".{target.name}.*.tmp"))
    assert len(orphaned) == 1
    assert str(orphaned[0]) in str(caught.value)
    assert "injected link ENOSPC" in str(caught.value)
    assert "injected stage sharing violation" in str(caught.value)
    assert isinstance(caught.value.__cause__, REPLAY_RUNNER.RematerializationError)
    assert isinstance(caught.value.__cause__.__cause__, OSError)
    assert not target.exists()
    orphaned[0].unlink()
    REPLAY_RUNNER._write_new_bytes(target, b"trusted", label="test output")
    assert target.read_bytes() == b"trusted"


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_link_then_process_interrupt_rolls_back_owned_single_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: type[BaseException],
) -> None:
    target = tmp_path / "interrupt-after-link.json"
    raw = b"exact source"
    original_link = REPLAY_RUNNER.os.link

    def link_then_interrupt(stage: Path, final: Path) -> None:
        original_link(stage, final)
        raise interruption("injected interruption after link")

    with monkeypatch.context() as fault:
        fault.setattr(REPLAY_RUNNER.os, "link", link_then_interrupt)
        with pytest.raises(interruption, match="interruption after link"):
            REPLAY_RUNNER._write_new_bytes(target, raw, label="interrupted output")

    assert not target.exists()
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))
    REPLAY_RUNNER._write_new_bytes(target, raw, label="interrupted output")
    assert target.read_bytes() == raw


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interrupt_during_postlink_stat_retains_original_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interruption: type[BaseException],
) -> None:
    target = tmp_path / "interrupted-inspection.json"
    raw = b"verified source"
    original_stat = Path.stat

    def interrupt_final_stat(path: Path, *args: object, **kwargs: object):
        if path == target and kwargs.get("follow_symlinks") is False:
            raise interruption("injected interruption during inode inspection")
        return original_stat(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "stat", interrupt_final_stat)
        with pytest.raises(
            REPLAY_RUNNER.PublicationIndeterminate,
            match="PUBLICATION_INDETERMINATE",
        ) as caught:
            REPLAY_RUNNER._write_new_bytes(target, raw, label="interrupted output")

    retained = list(tmp_path.glob(f".{target.name}.*.tmp"))
    assert len(retained) == 1
    assert caught.value.staged == retained[0]
    assert target.read_bytes() == retained[0].read_bytes() == raw
    assert isinstance(caught.value.__cause__, interruption)
    target.unlink()
    retained[0].unlink()
    REPLAY_RUNNER._write_new_bytes(target, raw, label="interrupted output")
    assert target.read_bytes() == raw


def test_interrupt_during_own_rollback_retains_stage_for_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "interrupted-rollback.json"
    raw = b"exact source"
    original_match = REPLAY_RUNNER._matches_staged_identity
    original_unlink = Path.unlink

    def deny_final_validation(
        path: Path, expected: bytes, identity: tuple[int, int],
    ) -> bool:
        return path != target and original_match(path, expected, identity)

    def interrupt_rollback(path: Path, *args: object, **kwargs: object) -> None:
        if path == target:
            raise KeyboardInterrupt("injected interrupt during rollback")
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(REPLAY_RUNNER, "_matches_staged_identity", deny_final_validation)
        fault.setattr(Path, "unlink", interrupt_rollback)
        with pytest.raises(
            REPLAY_RUNNER.PublicationIndeterminate, match="ROLLBACK_INCOMPLETE",
        ) as caught:
            REPLAY_RUNNER._write_new_bytes(target, raw, label="interrupted output")

    retained = list(tmp_path.glob(f".{target.name}.*.tmp"))
    assert len(retained) == 1 and caught.value.staged == retained[0]
    assert target.read_bytes() == retained[0].read_bytes() == raw
    assert isinstance(caught.value.__cause__, KeyboardInterrupt)
    target.unlink()
    retained[0].unlink()
    REPLAY_RUNNER._write_new_bytes(target, raw, label="interrupted output")
    assert target.read_bytes() == raw


def test_outer_link_interrupt_preserves_receipt_last_and_clean_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    original_link = REPLAY_RUNNER.os.link

    def interrupt_outer_report(stage: Path, final: Path) -> None:
        original_link(stage, final)
        if final == args.output_report:
            raise KeyboardInterrupt("injected post-link interruption")

    with monkeypatch.context() as fault:
        fault.setattr(REPLAY_RUNNER.os, "link", interrupt_outer_report)
        with pytest.raises(KeyboardInterrupt, match="post-link interruption"):
            REPLAY_RUNNER._publish_verified_outputs(
                args, pass_root=pass_root, pass_result=result, receipt=receipt,
            )

    assert not args.output_report.exists()
    assert not args.output_survivors.exists()
    assert not args.output_receipt.exists()
    assert not list(tmp_path.glob(".outer-*.tmp"))
    intent = pass_root / "outer-publication-intent.json"
    assert intent.exists()
    recovery = REPLAY_RUNNER.inspect_outer_publication_recovery(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    assert recovery["status"] == "PREPARED_UNCOMMITTED"
    assert recovery["training_authorized"] is False
    intent.unlink()
    REPLAY_RUNNER._publish_verified_outputs(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    recovery = REPLAY_RUNNER.inspect_outer_publication_recovery(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    assert recovery["status"] == "COMMITTED_ZERO_CREDIT"
    assert recovery["training_authorized"] is False


def test_failed_stage_fsync_and_interrupting_cleanup_preserve_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "stage-interruption.json"
    original_unlink = Path.unlink

    def fail_fsync(_fd: int) -> None:
        raise OSError("injected primary fsync failure")

    def interrupt_stage_unlink(
        path: Path, *args: object, **kwargs: object,
    ) -> None:
        if path.name.startswith(f".{target.name}.") and path.suffix == ".tmp":
            raise KeyboardInterrupt("injected stage cleanup interruption")
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(REPLAY_RUNNER.os, "fsync", fail_fsync)
        fault.setattr(Path, "unlink", interrupt_stage_unlink)
        with pytest.raises(
            REPLAY_RUNNER.RematerializationError, match="STAGING_CLEANUP_INCOMPLETE",
        ) as caught:
            REPLAY_RUNNER._write_new_bytes(target, b"trusted", label="test output")

    staged = list(tmp_path.glob(f".{target.name}.*.tmp"))
    assert len(staged) == 1
    assert str(staged[0]) in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)
    assert "primary fsync failure" in str(caught.value.__cause__)
    assert not target.exists()
    staged[0].unlink()
    REPLAY_RUNNER._write_new_bytes(target, b"trusted", label="test output")
    assert target.read_bytes() == b"trusted"


def test_verified_output_cleanup_interrupt_reports_committed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "committed-interruption.json"
    original_unlink = Path.unlink

    def interrupt_stage_unlink(
        path: Path, *args: object, **kwargs: object,
    ) -> None:
        if path.name.startswith(f".{target.name}.") and path.suffix == ".tmp":
            raise KeyboardInterrupt("injected cleanup interruption")
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "unlink", interrupt_stage_unlink)
        with pytest.raises(
            REPLAY_RUNNER.RematerializationError, match="published and byte-verified",
        ) as caught:
            REPLAY_RUNNER._write_new_bytes(target, b"trusted", label="test output")

    staged = list(tmp_path.glob(f".{target.name}.*.tmp"))
    assert len(staged) == 1
    assert target.read_bytes() == staged[0].read_bytes() == b"trusted"
    assert str(staged[0]) in str(caught.value)
    assert isinstance(caught.value.__cause__, KeyboardInterrupt)
    staged[0].unlink()


def test_outer_stage_cleanup_interrupt_keeps_committed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    original_unlink = Path.unlink

    def interrupt_report_stage_unlink(
        path: Path, *args_: object, **kwargs: object,
    ) -> None:
        if (
            path.name.startswith(f".{args.output_report.name}.")
            and path.suffix == ".tmp"
        ):
            raise KeyboardInterrupt("injected outer cleanup interruption")
        original_unlink(path, *args_, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "unlink", interrupt_report_stage_unlink)
        with pytest.raises(
            REPLAY_RUNNER.RematerializationError,
            match="outer receipt published and byte-verified",
        ) as caught:
            REPLAY_RUNNER._publish_verified_outputs(
                args, pass_root=pass_root, pass_result=result, receipt=receipt,
            )

    staged = list(
        args.output_report.parent.glob(f".{args.output_report.name}.*.tmp")
    )
    assert len(staged) == 1
    assert str(staged[0]) in str(caught.value)
    assert isinstance(caught.value.__cause__, KeyboardInterrupt)
    recovery = REPLAY_RUNNER.inspect_outer_publication_recovery(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    assert recovery["status"] == "COMMITTED_ZERO_CREDIT"
    assert recovery["training_authorized"] is False
    staged[0].unlink()


@pytest.mark.parametrize(
    "denied_label", ["outer report", "outer survivors", "outer receipt"],
)
def test_recovery_refuses_uninspectable_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, denied_label: str,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    REPLAY_RUNNER._publish_verified_outputs(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    targets = {
        "outer report": args.output_report,
        "outer survivors": args.output_survivors,
        "outer receipt": args.output_receipt,
    }
    before = {label: path.read_bytes() for label, path in targets.items()}
    original_lstat = Path.lstat

    def deny_final_lstat(path: Path, *a: object, **kw: object):
        if path == targets[denied_label]:
            raise PermissionError("injected NTFS metadata sharing denial")
        return original_lstat(path, *a, **kw)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "lstat", deny_final_lstat)
        with pytest.raises(
            REPLAY_RUNNER.RematerializationError,
            match=f"cannot inspect recovery {denied_label}",
        ) as caught:
            REPLAY_RUNNER.inspect_outer_publication_recovery(
                args, pass_root=pass_root, pass_result=result, receipt=receipt,
            )
    assert isinstance(caught.value.__cause__, PermissionError)
    assert {label: path.read_bytes() for label, path in targets.items()} == before
    recovered = REPLAY_RUNNER.inspect_outer_publication_recovery(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    assert recovered["status"] == "COMMITTED_ZERO_CREDIT"
    assert recovered["training_authorized"] is False


def test_recovery_does_not_trust_hidden_exists_for_real_committed_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    REPLAY_RUNNER._publish_verified_outputs(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    original_exists = Path.exists

    def hide_existing_report(path: Path, *a: object, **kw: object) -> bool:
        if path == args.output_report:
            return False
        return original_exists(path, *a, **kw)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "exists", hide_existing_report)
        recovered = REPLAY_RUNNER.inspect_outer_publication_recovery(
            args, pass_root=pass_root, pass_result=result, receipt=receipt,
        )
    assert recovered["status"] == "COMMITTED_ZERO_CREDIT"
    assert recovered["missing"] == []
    assert len(recovered["published"]) == 3
    assert recovered["training_authorized"] is False


@pytest.mark.parametrize("interruption", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("deny_final_inspection", [False, True])
def test_held_source_close_interruption_keeps_original_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException], deny_final_inspection: bool,
) -> None:
    final = tmp_path / "held-source-close.json"
    raw = b"authenticated result"
    stage = REPLAY_RUNNER._stage_new_bytes(final, raw, label="test output")
    original_open = Path.open
    original_stat = Path.stat
    wrapped = False

    class InterruptedClose:
        def __init__(self, held: object) -> None:
            self.held = held

        def __getattr__(self, name: str):
            return getattr(self.held, name)

        def close(self) -> None:
            self.held.close()
            raise interruption("injected held source close interruption")

    def open_held_once(path: Path, *args: object, **kwargs: object):
        nonlocal wrapped
        handle = original_open(path, *args, **kwargs)
        if path == stage and args and args[0] == "rb" and not wrapped:
            wrapped = True
            return InterruptedClose(handle)
        return handle

    def deny_final_stat(path: Path, *a: object, **kw: object):
        if (
            deny_final_inspection and path == final
            and kw.get("follow_symlinks") is False
        ):
            raise PermissionError("injected final metadata denial")
        return original_stat(path, *a, **kw)

    with monkeypatch.context() as fault:
        fault.setattr(REPLAY_RUNNER, "_stage_new_bytes", lambda *a, **kw: stage)
        fault.setattr(Path, "open", open_held_once)
        fault.setattr(Path, "stat", deny_final_stat)
        with pytest.raises(
            REPLAY_RUNNER.PublicationIndeterminate,
            match="SOURCE_CLOSE_INDETERMINATE",
        ) as caught:
            REPLAY_RUNNER._write_new_bytes(final, raw, label="test output")

    assert wrapped
    assert caught.value.staged == stage
    assert stage.read_bytes() == final.read_bytes() == raw
    assert str(stage) in str(caught.value) and str(final) in str(caught.value)
    if deny_final_inspection:
        assert isinstance(
            caught.value.__cause__, REPLAY_RUNNER.PublicationIndeterminate,
        )
    else:
        assert isinstance(caught.value.__cause__, interruption)
    final.unlink()
    stage.unlink()
    REPLAY_RUNNER._write_new_bytes(final, raw, label="test output")
    assert final.read_bytes() == raw


def test_close_fault_ignores_unrelated_active_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "unrelated-close.json"
    raw = b"authenticated output"
    stage = REPLAY_RUNNER._stage_new_bytes(final, raw, label="test output")
    original_open = Path.open
    wrapped = False

    class CloseFailure:
        def __init__(self, held: object) -> None:
            self.held = held

        def __getattr__(self, name: str):
            return getattr(self.held, name)

        def close(self) -> None:
            self.held.close()
            raise OSError("injected close EIO")

    def open_held_once(path: Path, *a: object, **kw: object):
        nonlocal wrapped
        held = original_open(path, *a, **kw)
        if path == stage and a and a[0] == "rb" and not wrapped:
            wrapped = True
            return CloseFailure(held)
        return held

    with monkeypatch.context() as fault:
        fault.setattr(REPLAY_RUNNER, "_stage_new_bytes", lambda *a, **kw: stage)
        fault.setattr(Path, "open", open_held_once)
        try:
            raise ValueError("unrelated ambient failure")
        except ValueError:
            with pytest.raises(
                REPLAY_RUNNER.PublicationIndeterminate,
                match="SOURCE_CLOSE_INDETERMINATE",
            ) as caught:
                REPLAY_RUNNER._write_new_bytes(final, raw, label="test output")

    assert wrapped
    assert isinstance(caught.value.__cause__, OSError)
    assert "injected close EIO" in str(caught.value.__cause__)
    assert "unrelated ambient" not in str(caught.value)
    assert caught.value.staged == stage
    assert final.read_bytes() == stage.read_bytes() == raw
    stage.unlink()
    final.unlink()
    REPLAY_RUNNER._write_new_bytes(final, raw, label="test output")
    assert final.read_bytes() == raw


def test_single_output_cleanup_with_ambient_exception_reports_committed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    final = tmp_path / "ambient-committed.json"
    original_unlink = Path.unlink

    def deny_stage_unlink(path: Path, *a: object, **kw: object) -> None:
        if path.name.startswith(f".{final.name}.") and path.suffix == ".tmp":
            raise PermissionError("injected stage sharing denial")
        original_unlink(path, *a, **kw)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "unlink", deny_stage_unlink)
        try:
            raise ValueError("unrelated ambient failure")
        except ValueError:
            with pytest.raises(
                REPLAY_RUNNER.RematerializationError,
                match="published and byte-verified",
            ) as caught:
                REPLAY_RUNNER._write_new_bytes(
                    final, b"trusted", label="test output",
                )

    staged = list(tmp_path.glob(f".{final.name}.*.tmp"))
    assert len(staged) == 1
    assert final.read_bytes() == staged[0].read_bytes() == b"trusted"
    assert isinstance(caught.value.__cause__, PermissionError)
    assert "unrelated ambient" not in str(caught.value)
    staged[0].unlink()


def test_outer_cleanup_with_ambient_exception_preserves_receipt_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pass_root, args, result, receipt = _publication_case(tmp_path, monkeypatch)
    original_unlink = Path.unlink

    def deny_report_stage_unlink(
        path: Path, *a: object, **kw: object,
    ) -> None:
        if (
            path.name.startswith(f".{args.output_report.name}.")
            and path.suffix == ".tmp"
        ):
            raise PermissionError("injected outer sharing denial")
        original_unlink(path, *a, **kw)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "unlink", deny_report_stage_unlink)
        try:
            raise ValueError("unrelated ambient failure")
        except ValueError:
            with pytest.raises(
                REPLAY_RUNNER.RematerializationError,
                match="outer receipt published and byte-verified",
            ) as caught:
                REPLAY_RUNNER._publish_verified_outputs(
                    args, pass_root=pass_root, pass_result=result, receipt=receipt,
                )

    assert isinstance(caught.value.__cause__, PermissionError)
    assert "unrelated ambient" not in str(caught.value)
    assert args.output_receipt.read_bytes() == canonical_json_bytes(receipt)
    staged = list(
        tmp_path.glob(f".{args.output_report.name}.*.tmp")
    )
    assert len(staged) == 1
    staged[0].unlink()
    recovery = REPLAY_RUNNER.inspect_outer_publication_recovery(
        args, pass_root=pass_root, pass_result=result, receipt=receipt,
    )
    assert recovery["status"] == "COMMITTED_ZERO_CREDIT"
    assert recovery["training_authorized"] is False
