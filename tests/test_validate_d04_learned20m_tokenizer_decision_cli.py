from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from twelve_six.tokenization.decision_authority import TokenizerDecisionError

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "validate_d04_learned20m_tokenizer_decision.py"
HASH_ARGS = (
    "--expected-selection-identity-sha256",
    "1" * 64,
    "--expected-application-identity-sha256",
    "2" * 64,
    "--expected-retained-inventory-identity-sha256",
    "3" * 64,
    "--expected-decontamination-authority-sha256",
    "4" * 64,
    "--expected-dedup-authority-sha256",
    "5" * 64,
    "--expected-balance-policy-identity-sha256",
    "6" * 64,
    "--expected-balance-result-identity-sha256",
    "7" * 64,
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_d04_learned20m_tokenizer_decision_cli",
        TOOL,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_accepts_finite_json_object(tmp_path: Path) -> None:
    path = tmp_path / "finite.json"
    path.write_text(
        '{"count":2,"ratio":1.25,"nested":{"value":-3.5}}',
        encoding="utf-8",
    )

    assert _module()._load(path) == {
        "count": 2,
        "ratio": 1.25,
        "nested": {"value": -3.5},
    }


def test_load_rejects_recursive_duplicate_members(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"outer":{"same":1,"same":2}}', encoding="utf-8")

    with pytest.raises(ValueError, match=r"^duplicate_json_key$"):
        _module()._load(path)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_load_rejects_nonfinite_constants(tmp_path: Path, value: str) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text(f'{{"value":{value}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="non_finite_json_constant"):
        _module()._load(path)


@pytest.mark.parametrize("value", ["1e400", "-1e400"])
def test_load_rejects_float_overflow(tmp_path: Path, value: str) -> None:
    path = tmp_path / "overflow.json"
    path.write_text(f'{{"value":{value}}}', encoding="utf-8")

    with pytest.raises(ValueError, match="non_finite_json_number"):
        _module()._load(path)


def test_load_redacts_secret_duplicate_member(tmp_path: Path) -> None:
    cli = _module()
    secret = "PRIVATE_TOKENIZER_SOURCE_KEY_998877"
    path = tmp_path / "duplicate-secret.json"
    path.write_text(
        '{"' + secret + '":1,"' + secret + '":2}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"^duplicate_json_key$") as caught:
        cli._load(path)
    assert secret not in str(caught.value)


@pytest.mark.parametrize("negative", [False, True])
def test_load_bounds_integer_before_python_conversion(
    tmp_path: Path,
    negative: bool,
) -> None:
    cli = _module()
    path = tmp_path / "huge-int.json"
    digits = ("-" if negative else "") + "9" * 100_000
    path.write_text('{"value":' + digits + "}", encoding="utf-8")
    before = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(0)
        with pytest.raises(ValueError, match="JSON integer exceeds 64 digits"):
            cli._load(path)
    finally:
        sys.set_int_max_str_digits(before)


def test_load_missing_secret_path_is_redacted(tmp_path: Path) -> None:
    cli = _module()
    secret = "PRIVATE-TOKENIZER-PATH-998877"
    with pytest.raises(
        ValueError, match=r"^cannot read tokenizer authority input$"
    ) as caught:
        cli._load(tmp_path / f"{secret}.json")
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_load_fifo_never_blocks(tmp_path: Path) -> None:
    if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
        pytest.skip("POSIX nonblocking FIFO support required")
    fifo = tmp_path / "tokenizer FIFO із пробілами.pipe"
    os.mkfifo(fifo)
    program = (
        "import importlib.util,sys\n"
        f"p={str(TOOL)!r}\n"
        "s=importlib.util.spec_from_file_location('tok_cli',p)\n"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n"
        "try:\n"
        " m._load(m.Path(sys.argv[1]))\n"
        "except ValueError as e:\n"
        " assert str(e) == 'tokenizer input must be a regular file'\n"
        "else:\n"
        " raise AssertionError('FIFO accepted')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(fifo)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_load_rejects_regular_file_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _module()
    requested = tmp_path / "requested.json"
    substitute = tmp_path / "substitute.json"
    requested.write_text("{}", encoding="utf-8")
    substitute.write_text("{}", encoding="utf-8")
    actual_open = cli.os.open

    def open_substitute(_path: Path, flags: int) -> int:
        return actual_open(substitute, flags)

    monkeypatch.setattr(cli.os, "open", open_substitute)
    with pytest.raises(ValueError, match="changed between check and open"):
        cli._load(requested)


def test_load_valid_regular_symlink_is_supported(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows symlink creation may require additional privileges")
    source = tmp_path / "source.json"
    source.write_text('{"safe":1}', encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.symlink_to(source.resolve())
    assert _module()._load(linked) == {"safe": 1}


def test_load_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[1,2,3]", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain one JSON object"):
        _module()._load(path)


@pytest.mark.parametrize("bad_target", ["selection", "application", "report"])
def test_main_malformed_authority_fails_machine_readably_without_output(
    tmp_path: Path,
    bad_target: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _module()
    selection = tmp_path / "selection.json"
    application = tmp_path / "application.json"
    report = tmp_path / "report.json"
    output = tmp_path / "out.json"
    selection.write_text("{}", encoding="utf-8")
    application.write_text("{}", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")

    malformed = '{"outer":{"same":1,"same":2}}'
    bad_path = {
        "selection": selection,
        "application": application,
        "report": report,
    }[bad_target]
    bad_path.write_text(malformed, encoding="utf-8")

    argv = [
        str(TOOL),
        "--balanced-selection",
        str(selection),
        "--split-application",
        str(application),
        *HASH_ARGS,
        "--output",
        str(output),
    ]
    if bad_target == "report":
        argv.extend(["--verify-report", str(report)])
    monkeypatch.setattr(sys, "argv", argv)

    assert cli.main() == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["contract_valid"] is False
    assert "duplicate_json_key:same" in payload["error"]
    assert not output.exists()


@pytest.mark.parametrize("target", ["selection", "application", "report"])
@pytest.mark.parametrize("preexisting_output", [False, True])
def test_main_semantic_authority_rejection_is_one_line_and_nonpublishing(
    tmp_path: Path,
    target: str,
    preexisting_output: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _module()
    selection = tmp_path / "selection.json"
    application = tmp_path / "application.json"
    report = tmp_path / "report.json"
    output = tmp_path / "out.json"
    selection.write_text("{}", encoding="utf-8")
    application.write_text("{}", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    old_output = '{"do_not_overwrite":true}\n'
    if preexisting_output:
        output.write_text(old_output, encoding="utf-8")

    if target == "application":
        # The real application validator requires a qualified selection first.
        # Exercise its domain-error return path without fabricating authority.
        def reject_application(*_args, **_kwargs):
            raise TokenizerDecisionError("split application fields are not closed-world")

        monkeypatch.setattr(cli, "bind_byte_baseline_decision", reject_application)

    argv = [
        str(TOOL),
        "--balanced-selection",
        str(selection),
        "--split-application",
        str(application),
        *HASH_ARGS,
        "--output",
        str(output),
    ]
    if target == "report":
        argv.extend(["--verify-report", str(report)])
    monkeypatch.setattr(sys, "argv", argv)

    assert cli.main() == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["contract_valid"] is False
    expected_error = {
        "selection": "unsupported balanced-selection authority",
        "application": "split application fields are not closed-world",
        "report": "report fields are not closed-world",
    }[target]
    assert payload["error"] == expected_error
    if preexisting_output:
        assert output.read_text(encoding="utf-8") == old_output
    else:
        assert not output.exists()


@pytest.mark.parametrize(
    ("abbreviated", "canonical"),
    [
        ("--balanced-sel", "--balanced-selection"),
        ("--split-app", "--split-application"),
        ("--expected-select", "--expected-selection-identity-sha256"),
        ("--verif", "--verify-report"),
    ],
)
def test_parse_args_rejects_abbreviated_authority_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    abbreviated: str,
    canonical: str,
) -> None:
    cli = _module()
    selection = tmp_path / "selection.json"
    application = tmp_path / "application.json"
    report = tmp_path / "report.json"
    for path in (selection, application, report):
        path.write_text("{}", encoding="utf-8")
    argv = [
        str(TOOL),
        "--balanced-selection",
        str(selection),
        "--split-application",
        str(application),
        *HASH_ARGS,
        "--verify-report",
        str(report),
    ]
    option_index = argv.index(canonical)
    argv[option_index] = abbreviated
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        cli.parse_args()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    ("option", "extra_value", "use_equals"),
    [
        ("--balanced-selection", "second-selection.json", False),
        ("--expected-selection-identity-sha256", "8" * 64, False),
        ("--output", "second-output.json", False),
        ("--verify-report", "second-report.json", True),
    ],
)
def test_parse_args_rejects_duplicate_authority_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    option: str,
    extra_value: str,
    use_equals: bool,
) -> None:
    cli = _module()
    selection = tmp_path / "selection.json"
    application = tmp_path / "application.json"
    report = tmp_path / "report.json"
    output = tmp_path / "output.json"
    for path in (selection, application, report):
        path.write_text("{}", encoding="utf-8")
    argv = [
        str(TOOL),
        "--balanced-selection",
        str(selection),
        "--split-application",
        str(application),
        *HASH_ARGS,
        "--output",
        str(output),
        "--verify-report",
        str(report),
    ]
    duplicate = (
        f"{option}={tmp_path / extra_value}"
        if use_equals and option in {"--output", "--verify-report", "--balanced-selection"}
        else f"{option}={extra_value}"
        if use_equals
        else option
    )
    if use_equals:
        argv.append(duplicate)
    else:
        argv.extend([option, str(tmp_path / extra_value) if option in {
            "--balanced-selection", "--output", "--verify-report"
        } else extra_value])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as caught:
        cli.parse_args()
    assert caught.value.code == 2


def test_main_valid_dispatch_preserves_report_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _module()
    selection = tmp_path / "selection.json"
    application = tmp_path / "application.json"
    output = tmp_path / "out.json"
    selection.write_text("{}", encoding="utf-8")
    application.write_text("{}", encoding="utf-8")

    # Test only the CLI's successful dispatch; this stub grants no authority.
    fake_report = {"schema": "test-only", "training_authorized_by_this_report": False}
    monkeypatch.setattr(
        cli, "bind_byte_baseline_decision", lambda *_a, **_k: fake_report
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL),
            "--balanced-selection",
            str(selection),
            "--split-application",
            str(application),
            *HASH_ARGS,
            "--output",
            str(output),
        ],
    )

    assert cli.main() == 0
    assert capsys.readouterr().err == ""
    assert json.loads(output.read_text(encoding="utf-8")) == fake_report


def test_main_unexpected_programming_failure_is_not_misreported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _module()
    selection = tmp_path / "selection.json"
    application = tmp_path / "application.json"
    output = tmp_path / "out.json"
    selection.write_text("{}", encoding="utf-8")
    application.write_text("{}", encoding="utf-8")

    def unexpected(*_args, **_kwargs):
        raise RuntimeError("unexpected programming failure")

    monkeypatch.setattr(cli, "bind_byte_baseline_decision", unexpected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL),
            "--balanced-selection",
            str(selection),
            "--split-application",
            str(application),
            *HASH_ARGS,
            "--output",
            str(output),
        ],
    )
    with pytest.raises(RuntimeError, match="unexpected programming failure"):
        cli.main()
    assert not output.exists()


@pytest.mark.parametrize("kind", ["arrays", "objects"])
def test_load_excessive_json_depth_is_controlled(
    tmp_path: Path, kind: str,
) -> None:
    raw = (
        '{"root":' + "[" * 10000 + "0" + "]" * 10000 + "}"
        if kind == "arrays"
        else '{"root":' + '{"k":' * 10000 + "0" + "}" * 10000 + "}"
    )
    path = tmp_path / "deep.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="JSON nesting limit exceeded"):
        _module()._load(path)


@pytest.mark.parametrize("kind", ["arrays", "objects"])
@pytest.mark.parametrize("target", ["selection", "application", "report"])
def test_deep_external_authority_does_not_publish(
    tmp_path: Path, kind: str, target: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _module()
    paths = {k: tmp_path / f"{k}.json" for k in (
        "selection", "application", "report"
    )}
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")
    raw = (
        '{"root":' + "[" * 10000 + "0" + "]" * 10000 + "}"
        if kind == "arrays"
        else '{"root":' + '{"k":' * 10000 + "0" + "}" * 10000 + "}"
    )
    paths[target].write_text(raw, encoding="utf-8")
    before = {p: p.read_bytes() for p in paths.values()}
    output = tmp_path / "out.json"
    argv = [
        str(TOOL), "--balanced-selection", str(paths["selection"]),
        "--split-application", str(paths["application"]),
        *HASH_ARGS, "--output", str(output),
    ]
    if target == "report":
        argv.extend(["--verify-report", str(paths["report"])])
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == 2
    out = capsys.readouterr()
    assert out.err == ""
    lines = out.out.splitlines()
    assert len(lines) == 1
    error = json.loads(lines[0])
    assert error["contract_valid"] is False
    assert "JSON nesting limit exceeded" in error["error"]
    assert {p: p.read_bytes() for p in paths.values()} == before
    assert not output.exists()


@pytest.mark.parametrize("target", ["selection", "application", "report", "other"])
def test_successful_dispatch_never_overwrites_any_existing_file(
    tmp_path: Path, target: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _module()
    paths = {k: tmp_path / f"{k}.json" for k in (
        "selection", "application", "report", "other"
    )}
    for path in paths.values():
        path.write_text('{"original":true}\n', encoding="utf-8")
    before = {p: p.read_bytes() for p in paths.values()}
    monkeypatch.setattr(
        cli, "verify_byte_baseline_decision", lambda *_a, **_k: None
    )
    monkeypatch.setattr(sys, "argv", [
        str(TOOL), "--balanced-selection", str(paths["selection"]),
        "--split-application", str(paths["application"]),
        *HASH_ARGS, "--verify-report", str(paths["report"]),
        "--output", str(paths[target]),
    ])
    assert cli.main() == 2
    out = capsys.readouterr()
    assert out.err == ""
    lines = out.out.splitlines()
    assert len(lines) == 1
    assert "refusing to overwrite existing output" in (
        json.loads(lines[0])["error"]
    )
    assert {p: p.read_bytes() for p in paths.values()} == before
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("alias", ["hardlink", "symlink"])
def test_output_alias_of_input_is_rejected(tmp_path: Path, alias: str) -> None:
    cli = _module()
    authority = tmp_path / "source.json"
    authority.write_text('{"original":true}\n', encoding="utf-8")
    output = tmp_path / "alias.json"
    try:
        if alias == "hardlink":
            cli.os.link(authority, output)
        else:
            output.symlink_to(authority)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"filesystem cannot create {alias}: {exc}")
    before = authority.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli._write(output, {"schema": "test-only"})
    assert authority.read_bytes() == before
    assert output.read_bytes() == before


def test_failed_atomic_link_removes_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _module()
    output = tmp_path / "out.json"

    def broken_link(*_a: object, **_k: object) -> None:
        raise OSError("link failure")

    monkeypatch.setattr(cli.os, "link", broken_link)
    with pytest.raises(OSError, match="link failure"):
        cli._write(output, {"schema": "test-only"})
    assert not output.exists()
    assert not list(tmp_path.glob(".out.json.*.tmp"))


def test_postlink_cleanup_denial_reports_committed_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _module()
    selection = tmp_path / "selection.json"
    application = tmp_path / "application.json"
    selection.write_text("{}", encoding="utf-8")
    application.write_text("{}", encoding="utf-8")
    output = tmp_path / "decision.json"
    report = {"schema": "test-only", "status": "PASS_ZERO_CREDIT"}
    monkeypatch.setattr(
        cli, "bind_byte_baseline_decision", lambda *_a, **_k: report
    )
    real_unlink = Path.unlink

    def deny_stage_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(f".{output.name}.") and path.suffix == ".tmp":
            raise PermissionError("injected Windows-style sharing denial")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_stage_cleanup)
    monkeypatch.setattr(sys, "argv", [
        str(TOOL), "--balanced-selection", str(selection),
        "--split-application", str(application), *HASH_ARGS,
        "--output", str(output),
    ])
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    status = json.loads(captured.out)
    assert status["contract_valid"] is True
    assert status["output_committed"] is True
    assert status["cleanup_pending"] is True
    assert "COMMITTED_AND_VERIFIED" in status["recovery"]
    expected = cli._serialize_report(report).encode("utf-8")
    assert output.read_bytes() == expected
    staged = list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert len(staged) == 1 and staged[0].read_bytes() == expected
    real_unlink(staged[0])


def test_link_create_then_raise_is_reconciled_as_committed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _module()
    output = tmp_path / "decision.json"
    actual_link = cli.os.link

    def link_then_raise(stage: Path, final: Path) -> None:
        actual_link(stage, final)
        raise PermissionError("injected post-create link error")

    monkeypatch.setattr(cli.os, "link", link_then_raise)
    cli._write(output, {"schema": "test-only", "status": "zero-credit"})
    assert output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_postcreate_process_interrupt_rolls_back_owned_final_and_rethrows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption: type[BaseException],
) -> None:
    cli = _module()
    output = tmp_path / "interrupted.json"
    actual_link = cli.os.link

    def link_then_interrupt(stage: Path, final: Path) -> None:
        actual_link(stage, final)
        raise interruption("injected post-create interruption")

    monkeypatch.setattr(cli.os, "link", link_then_interrupt)
    with pytest.raises(interruption, match="post-create interruption"):
        cli._write(output, {"schema": "test-only", "status": "zero-credit"})
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_postcreate_interrupt_with_rollback_denial_retains_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _module()
    output = tmp_path / "rollback-denied.json"
    actual_link = cli.os.link
    actual_unlink = Path.unlink

    def link_then_interrupt(stage: Path, final: Path) -> None:
        actual_link(stage, final)
        raise KeyboardInterrupt("injected post-create interruption")

    def deny_final_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == output:
            raise PermissionError("injected rollback denial")
        actual_unlink(path, *args, **kwargs)

    monkeypatch.setattr(cli.os, "link", link_then_interrupt)
    monkeypatch.setattr(Path, "unlink", deny_final_unlink)
    with pytest.raises(
        cli.PublicationIndeterminate, match="ROLLBACK_INDETERMINATE"
    ) as caught:
        cli._write(output, {"schema": "test-only", "status": "zero-credit"})
    staged = list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert len(staged) == 1 and caught.value.staged == staged[0]
    assert output.exists()
    actual_unlink(output)
    actual_unlink(staged[0])


def test_foreign_final_after_link_error_is_indeterminate_and_never_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _module()
    output = tmp_path / "decision.json"
    foreign = b'{"foreign":true}\n'

    def create_foreign_then_raise(_stage: Path, final: Path) -> None:
        final.write_bytes(foreign)
        raise PermissionError("injected foreign create race")

    monkeypatch.setattr(cli.os, "link", create_foreign_then_raise)
    with pytest.raises(cli.PublicationIndeterminate, match="not the staged") as caught:
        cli._write(output, {"schema": "test-only", "status": "zero-credit"})
    assert output.read_bytes() == foreign
    staged = list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert len(staged) == 1 and caught.value.staged == staged[0]
    output.unlink()
    staged[0].unlink()


def test_staging_primary_failure_plus_cleanup_denial_preserves_primary_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _module()
    output = tmp_path / "decision.json"
    actual_unlink = Path.unlink

    def fail_fsync(_fd: int) -> None:
        raise OSError("primary staging fsync failure")

    def deny_stage_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if path.name.startswith(f".{output.name}.") and path.suffix == ".tmp":
            raise PermissionError("secondary cleanup denial")
        actual_unlink(path, *args, **kwargs)

    monkeypatch.setattr(cli.os, "fsync", fail_fsync)
    monkeypatch.setattr(Path, "unlink", deny_stage_cleanup)
    with pytest.raises(
        cli.PublicationIndeterminate, match="STAGING_CLEANUP_INDETERMINATE"
    ) as caught:
        cli._write(output, {"schema": "test-only"})
    assert isinstance(caught.value.__cause__, OSError)
    assert "primary staging fsync failure" in str(caught.value.__cause__)
    assert not output.exists()
    staged = list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert len(staged) == 1
    actual_unlink(staged[0])


def test_nonfinite_and_recursive_reports_are_not_published(
    tmp_path: Path,
) -> None:
    cli = _module()
    output = tmp_path / "out.json"
    recursive: dict[str, object] = {}
    recursive["self"] = recursive
    for report in ({"loss": float("nan")}, recursive, {"surrogate": "\ud800"}):
        with pytest.raises(ValueError, match="not strict finite JSON"):
            cli._write(output, report)
        assert not output.exists()



@pytest.mark.parametrize("kind", ["nan", "infinity", "circular", "unsupported", "surrogate"])
def test_main_stdout_rejects_invalid_report_with_one_json_error(
    tmp_path: Path,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _module()
    selection = tmp_path / "selection.json"
    application = tmp_path / "application.json"
    selection.write_text("{}", encoding="utf-8")
    application.write_text("{}", encoding="utf-8")
    recursive: dict[str, object] = {}
    recursive["self"] = recursive
    reports = {
        "nan": {"loss": float("nan")},
        "infinity": {"loss": float("inf")},
        "circular": recursive,
        "unsupported": {"unserializable": object()},
        "surrogate": {"invalid_unicode": "\ud800"},
    }
    monkeypatch.setattr(
        cli, "bind_byte_baseline_decision",
        lambda *_a, **_k: reports[kind],
    )
    monkeypatch.setattr(sys, "argv", [
        str(TOOL), "--balanced-selection", str(selection),
        "--split-application", str(application), *HASH_ARGS,
    ])
    assert cli.main() == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    error = json.loads(lines[0])
    assert error["contract_valid"] is False
    assert error["error"] == "tokenizer report is not strict finite JSON"


def test_main_stdout_finite_report_is_canonical_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _module()
    selection = tmp_path / "selection.json"
    application = tmp_path / "application.json"
    selection.write_text("{}", encoding="utf-8")
    application.write_text("{}", encoding="utf-8")
    fake_report = {"schema": "test-only", "ratio": 1.25, "authorized": False}
    monkeypatch.setattr(
        cli, "bind_byte_baseline_decision", lambda *_a, **_k: fake_report
    )
    monkeypatch.setattr(sys, "argv", [
        str(TOOL), "--balanced-selection", str(selection),
        "--split-application", str(application), *HASH_ARGS,
    ])
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        json.dumps(
            fake_report, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ) + "\n"
    )

def test_unexpected_product_recursion_remains_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _module()
    selection = tmp_path / "selection.json"
    application = tmp_path / "application.json"
    for path in (selection, application):
        path.write_text("{}", encoding="utf-8")

    def unexpected(*_a: object, **_k: object) -> None:
        raise RecursionError("unexpected Product recursion")

    monkeypatch.setattr(cli, "bind_byte_baseline_decision", unexpected)
    monkeypatch.setattr(sys, "argv", [
        str(TOOL), "--balanced-selection", str(selection),
        "--split-application", str(application), *HASH_ARGS,
    ])
    with pytest.raises(RecursionError, match="unexpected Product recursion"):
        cli.main()


@pytest.mark.parametrize("number", ["1e-4000", "-1e-4000", "0.0001e-9999"])
def test_load_rejects_nonzero_float_underflow(tmp_path: Path, number: str) -> None:
    path = tmp_path / "underflow.json"
    path.write_text(f'{{"number":{number}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="nonzero_json_number_underflowed_to_zero"):
        _module()._load(path)


@pytest.mark.parametrize("number", ["0e-9999", "-0.000e-9999", "0.0", "1.25e-3"])
def test_load_preserves_genuine_zero_and_finite_float(
    tmp_path: Path, number: str,
) -> None:
    path = tmp_path / "finite.json"
    path.write_text(f'{{"number":{number}}}', encoding="utf-8")
    assert _module()._load(path) == {"number": float(number)}


@pytest.mark.parametrize("target", ["selection", "application", "report"])
def test_external_float_underflow_fails_without_publication(
    tmp_path: Path,
    target: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _module()
    paths = {key: tmp_path / f"{key}.json" for key in (
        "selection", "application", "report"
    )}
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")
    paths[target].write_text('{"ignored":1e-4000}', encoding="utf-8")
    original = {path: path.read_bytes() for path in paths.values()}
    output = tmp_path / "decision.json"
    argv = [
        str(TOOL), "--balanced-selection", str(paths["selection"]),
        "--split-application", str(paths["application"]),
        *HASH_ARGS, "--output", str(output),
    ]
    if target == "report":
        argv.extend(["--verify-report", str(paths["report"])])
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    result = json.loads(lines[0])
    assert result["contract_valid"] is False
    assert "nonzero_json_number_underflowed_to_zero" in result["error"]
    assert {path: path.read_bytes() for path in paths.values()} == original
    assert not output.exists()


def test_short_staged_report_write_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _module()
    output = tmp_path / "report.json"
    actual_fdopen = cli.os.fdopen

    class ShortWriter:
        def __init__(self, real):
            self.real = real

        def __enter__(self):
            self.real.__enter__()
            return self

        def __exit__(self, *args):
            return self.real.__exit__(*args)

        def write(self, payload: bytes) -> int:
            self.real.write(payload[:1])
            return 1

    monkeypatch.setattr(
        cli.os, "fdopen", lambda descriptor, mode: ShortWriter(
            actual_fdopen(descriptor, mode)
        ),
    )
    with pytest.raises(OSError, match="incomplete tokenizer report staging write"):
        cli._write(output, {"schema": "test-only", "status": "zero-credit"})
    assert not output.exists()
    assert not list(tmp_path.glob(".report.json.*.tmp"))


@pytest.mark.parametrize("bad_target", ["selection", "application", "report"])
@pytest.mark.parametrize("failure", ["oversize", "invalid_utf8", "surrogate"])
def test_bounded_external_authorities_never_publish(
    tmp_path: Path,
    bad_target: str,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _module()
    paths = {key: tmp_path / f"{key} with кирилиця.json" for key in (
        "selection", "application", "report"
    )}
    for path in paths.values():
        path.write_text("{}", encoding="utf-8")
    invalid_bytes = {
        "oversize": b"{}" + b" " * cli.MAX_INPUT_BYTES,
        "invalid_utf8": b'{"value":"\xff"}',
        "surrogate": br'{"value":"\ud800"}',
    }[failure]
    paths[bad_target].write_bytes(invalid_bytes)
    output = tmp_path / "never publish.json"
    argv = [
        str(TOOL), "--balanced-selection", str(paths["selection"]),
        "--split-application", str(paths["application"]),
        *HASH_ARGS, "--output", str(output),
    ]
    if bad_target == "report":
        argv.extend(["--verify-report", str(paths["report"])])
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    error = json.loads(lines[0])
    assert error["contract_valid"] is False
    assert not output.exists()
    assert paths[bad_target].read_bytes() == invalid_bytes


def test_load_accepts_exact_input_byte_limit(tmp_path: Path) -> None:
    cli = _module()
    path = tmp_path / "limit.json"
    prefix = b'{"ignored":0}'
    path.write_bytes(prefix + b" " * (cli.MAX_INPUT_BYTES - len(prefix)))
    assert cli._load(path) == {"ignored": 0}


@pytest.mark.parametrize("kind", ["depth", "nodes"])
def test_load_rejects_bounded_structure(tmp_path: Path, kind: str) -> None:
    cli = _module()
    path = tmp_path / "structure.json"
    raw = (
        '{"root":' + "[" * (cli.MAX_JSON_DEPTH + 1) +
        "0" + "]" * (cli.MAX_JSON_DEPTH + 1) + "}"
        if kind == "depth"
        else '{"root":[' + ",".join(["0"] * (cli.MAX_JSON_NODES + 1)) + "]}"
    )
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds JSON structure limit"):
        cli._load(path)


@pytest.mark.parametrize("location", ["key", "value"])
def test_load_rejects_decoded_unpaired_surrogates(
    tmp_path: Path, location: str,
) -> None:
    cli = _module()
    path = tmp_path / "surrogate.json"
    path.write_bytes(
        br'{"\ud800":"safe"}' if location == "key" else br'{"safe":"\ud800"}'
    )
    with pytest.raises(UnicodeError):
        cli._load(path)


def test_load_rejects_input_one_byte_over_limit(tmp_path: Path) -> None:
    cli = _module()
    path = tmp_path / "oversize.json"
    path.write_bytes(b"{}" + b" " * (cli.MAX_INPUT_BYTES - 1))
    with pytest.raises(ValueError, match="exceeds byte limit"):
        cli._load(path)
