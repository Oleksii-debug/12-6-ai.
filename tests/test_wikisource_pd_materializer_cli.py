from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import pytest

from twelve_six.data.wikisource_pd_contract import (
    WikisourceIntakeError,
    validate_control_contract,
)

ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/materialize_d03_wikisource_lesia1892.py"
CANONICAL_CONTRACT = ROOT / "configs/data/d03_wikisource_lesia1892_current_main_v1.json"


def _load_tool() -> dict[str, object]:
    return runpy.run_path(str(TOOL_PATH))


def test_strict_control_loader_accepts_canonical_contract() -> None:
    loader = _load_tool()["load_strict_json_object"]
    assert callable(loader)
    contract = loader(CANONICAL_CONTRACT.read_text(encoding="utf-8"))
    validate_control_contract(contract)


@pytest.mark.parametrize(
    "raw",
    [
        '{"execution_class":"PAID","execution_class":"LOCAL_FREE"}',
        '{"truth_boundary":{"training_authorized_bytes":1,"training_authorized_bytes":0}}',
        '{"\\u0065xecution_class":"PAID","execution_class":"LOCAL_FREE"}',
    ],
)
def test_strict_control_loader_rejects_duplicate_members(raw: str) -> None:
    loader = _load_tool()["load_strict_json_object"]
    assert callable(loader)
    with pytest.raises(WikisourceIntakeError, match="duplicate JSON object member"):
        loader(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
        '{"value":-1e400}',
    ],
)
def test_strict_control_loader_rejects_non_finite_numbers(raw: str) -> None:
    loader = _load_tool()["load_strict_json_object"]
    assert callable(loader)
    with pytest.raises(WikisourceIntakeError, match="non-finite JSON"):
        loader(raw)


@pytest.mark.parametrize("raw", ["[]", "null", '"text"', "0"])
def test_strict_control_loader_requires_object_root(raw: str) -> None:
    loader = _load_tool()["load_strict_json_object"]
    assert callable(loader)
    with pytest.raises(WikisourceIntakeError, match="must be a JSON object"):
        loader(raw)



@pytest.mark.parametrize("literal", ["1e-9999", "-1e-9999", "4.2e-9999"])
def test_strict_control_loader_rejects_nonzero_float_underflow(literal: str) -> None:
    loader = _load_tool()["load_strict_json_object"]
    assert callable(loader)
    with pytest.raises(WikisourceIntakeError, match="underflowed to zero"):
        loader('{"value":' + literal + "}")


def test_strict_control_loader_bounds_integer_before_python_conversion() -> None:
    loader = _load_tool()["load_strict_json_object"]
    assert callable(loader)
    before = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        with pytest.raises(WikisourceIntakeError, match="integer exceeds 64 digits"):
            loader('{"value":' + "9" * 100_000 + "}")
    finally:
        sys.set_int_max_str_digits(before)


def test_control_file_reader_detects_descriptor_stamp_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_tool()
    reader = module["_read_control_contract"]
    assert callable(reader)
    path = tmp_path / "contract.json"
    path.write_text("{}", encoding="utf-8")
    tool_os = reader.__globals__["os"]
    actual_fstat = tool_os.fstat
    calls = 0

    def drifting_fstat(fd: int):
        nonlocal calls
        info = actual_fstat(fd)
        calls += 1
        if calls >= 2:
            values = list(info)
            values[6] = info.st_size + 1
            return tool_os.stat_result(values)
        return info

    monkeypatch.setattr(tool_os, "fstat", drifting_fstat)
    with pytest.raises(WikisourceIntakeError, match="changed during read"):
        reader(path)


def test_control_file_reader_rejects_symlink(tmp_path: Path) -> None:
    module = _load_tool()
    reader = module["_read_control_contract"]
    assert callable(reader)
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    linked = tmp_path / "linked.json"
    try:
        linked.symlink_to(source)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    with pytest.raises(WikisourceIntakeError, match="regular non-symlink"):
        reader(linked)


def test_control_file_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
        pytest.skip("FIFO support unavailable")
    module = _load_tool()
    reader = module["_read_control_contract"]
    assert callable(reader)
    path = tmp_path / "control FIFO.pipe"
    try:
        os.mkfifo(path)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"FIFO unavailable: {exc}")
    with pytest.raises(WikisourceIntakeError, match="regular non-symlink"):
        reader(path)


@pytest.mark.parametrize(
    ("abbreviated", "canonical"),
    [
        ("--cand", "--candidate-out"),
        ("--rep", "--report-out"),
        ("--max-p", "--max-pages"),
        ("--cont", "--contract"),
    ],
)
def test_cli_rejects_abbreviated_authority_options(
    tmp_path: Path,
    abbreviated: str,
    canonical: str,
) -> None:
    module = _load_tool()
    main = module["main"]
    assert callable(main)
    candidate = tmp_path / "candidate.jsonl"
    report = tmp_path / "report.json"
    argv = [
        "--contract",
        str(CANONICAL_CONTRACT),
        "--candidate-out",
        str(candidate),
        "--report-out",
        str(report),
        "--max-pages",
        "1",
    ]
    argv[argv.index(canonical)] = abbreviated
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2
    assert not candidate.exists()
    assert not report.exists()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--candidate-out", "second-candidate.jsonl"),
        ("--report-out", "second-report.json"),
        ("--max-pages", "2"),
        ("--contract", "second-contract.json"),
    ],
)
def test_cli_rejects_duplicate_authority_options(
    tmp_path: Path,
    option: str,
    value: str,
) -> None:
    module = _load_tool()
    main = module["main"]
    assert callable(main)
    candidate = tmp_path / "candidate.jsonl"
    report = tmp_path / "report.json"
    argv = [
        "--contract",
        str(CANONICAL_CONTRACT),
        "--candidate-out",
        str(candidate),
        "--report-out",
        str(report),
        "--max-pages",
        "1",
        option,
        str(tmp_path / value) if option != "--max-pages" else value,
    ]
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2
    assert not candidate.exists()
    assert not report.exists()


def test_output_pair_refuses_preexisting_file_before_materialization(
    tmp_path: Path,
) -> None:
    module = _load_tool()
    prepare = module["_prepare_output_pair"]
    assert callable(prepare)
    candidate = tmp_path / "candidate.jsonl"
    report = tmp_path / "report.json"
    candidate.write_bytes(b"DO_NOT_OVERWRITE")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare(candidate, report)
    assert candidate.read_bytes() == b"DO_NOT_OVERWRITE"
    assert not report.exists()


def test_output_pair_rolls_back_first_file_if_second_create_fails(
    tmp_path: Path,
) -> None:
    module = _load_tool()
    publish = module["_publish_output_pair"]
    assert callable(publish)
    candidate = tmp_path / "candidate.jsonl"
    report = tmp_path / "report.json"
    report.write_bytes(b"FOREIGN_REPORT")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish(candidate, b'{"page":1}\n', report, b'{"report":true}\n')
    assert not candidate.exists()
    assert report.read_bytes() == b"FOREIGN_REPORT"


def test_output_pair_rollback_preserves_substituted_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_tool()
    publish = module["_publish_output_pair"]
    assert callable(publish)
    candidate = tmp_path / "candidate.jsonl"
    report = tmp_path / "report.json"
    moved = tmp_path / "owned-candidate-moved-aside"
    unrelated = b"UNRELATED_REPLACEMENT"
    actual_write = publish.__globals__["_write_create_only_durable"]
    calls = 0

    def fail_second(path: Path, payload: bytes):
        nonlocal calls
        calls += 1
        if calls == 1:
            return actual_write(path, payload)
        candidate.rename(moved)
        candidate.write_bytes(unrelated)
        raise OSError("injected report create failure")

    monkeypatch.setitem(publish.__globals__, "_write_create_only_durable", fail_second)
    with pytest.raises(
        WikisourceIntakeError,
        match="partial materialization publication rollback failed",
    ):
        publish(candidate, b'{"page":1}\n', report, b'{"report":true}\n')
    assert candidate.read_bytes() == unrelated
    assert moved.read_bytes() == b'{"page":1}\n'
    assert not report.exists()


def test_cli_rejects_ambiguous_contract_before_materialization_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_tool()
    main = module["main"]
    assert callable(main)

    def fail_if_materialized(*args: object, **kwargs: object) -> None:
        pytest.fail("materialize_live must not run for an ambiguous control contract")

    main.__globals__["materialize_live"] = fail_if_materialized

    contract_path = tmp_path / "ambiguous.json"
    candidate_path = tmp_path / "candidate.jsonl"
    report_path = tmp_path / "report.json"
    contract_path.write_text(
        '{"execution_class":"PAID","execution_class":"LOCAL_FREE"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL_PATH),
            "--contract",
            str(contract_path),
            "--candidate-out",
            str(candidate_path),
            "--report-out",
            str(report_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert not candidate_path.exists()
    assert not report_path.exists()
