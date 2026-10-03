from __future__ import annotations

import importlib.util
import json
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

    with pytest.raises(ValueError, match="duplicate_json_key:same"):
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
