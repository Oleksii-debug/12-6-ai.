"""Source and evidence preservation for the D03 -> NEXT100 adapter CLI."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.adapt_d03_postdecontam_family_vector_to_next100_106_v1 as cli
from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError


def _configure(
    monkeypatch: pytest.MonkeyPatch, vector: Path, authority: Path, output: Path,
) -> None:
    args = SimpleNamespace(
        family_vector=vector,
        expected_family_vector_identity_sha256="a" * 64,
        dedup_authority=authority,
        expected_dedup_worker_id="dedup",
        expected_dedup_head_sha="b" * 40,
        expected_dedup_evidence_identity_sha256="c" * 64,
        output=output,
    )
    monkeypatch.setattr(cli, "_parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(cli, "_load_authority_json", lambda path: {"path": str(path)})
    monkeypatch.setattr(
        cli, "adapt_family_vector_to_next100_106",
        lambda *args, **kwargs: {
            "schema_version": "test-only",
            "claim_boundary": {"model_training_authorized": False},
            "value": "Український текст",
        },
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (b'{"id":1,"id":2}', "duplicate JSON key"),
        (b'{"outer":{"id":1,"id":2}}', "duplicate JSON key"),
        (b'{"id":NaN}', "nonstandard JSON constant"),
        (b'{"id":Infinity}', "nonstandard JSON constant"),
        (b'{"id":1e400}', "nonfinite JSON number"),
        (b'{"id":1e-9999}', "underflowed to zero"),
        (b'{"id":' + b"9" * 65 + b"}", "digit limit"),
        (b'{"id":"\\ud800"}', "invalid Unicode"),
        (b'{"\\ud800":"id"}', "invalid Unicode"),
        (b'{"id":' + b"[" * 10000 + b"0" + b"]" * 10000 + b"}", "strict JSON"),
        (b'{"id":' + b"[" * 64 + b"0" + b"]" * 64 + b"}", "structure limit"),
        (b"x" * (cli.MAX_AUTHORITY_JSON_BYTES + 1), "byte limit"),
        (b"[]", "top-level JSON object"),
        (b"\xff", "strict UTF-8"),
    ],
)
def test_authority_loader_rejects_ambiguous_oversized_and_invalid_json(
    tmp_path: Path, raw: bytes, expected: str,
) -> None:
    source = tmp_path / "неоднозначний документ з пробілами.json"
    source.write_bytes(raw)
    with pytest.raises(ProjectionError, match=expected):
        cli._load_authority_json(source)


def test_authority_loader_accepts_valid_utf8_and_finite_zero(tmp_path: Path) -> None:
    source = tmp_path / "правильний документ.json"
    source.write_bytes('{"word":"Україна","zero":0e-9999}'.encode("utf-8"))
    assert cli._load_authority_json(source) == {"word": "Україна", "zero": 0.0}


def test_authority_loader_accepts_exact_byte_and_node_limits(tmp_path: Path) -> None:
    source = tmp_path / "точна межа байтів.json"
    raw = b'{"pad":"' + b"x" * (cli.MAX_AUTHORITY_JSON_BYTES - 10) + b'"}'
    assert len(raw) == cli.MAX_AUTHORITY_JSON_BYTES
    source.write_bytes(raw)
    assert len(cli._load_authority_json(source)["pad"]) == cli.MAX_AUTHORITY_JSON_BYTES - 10

    source = tmp_path / "точна межа вузлів.json"
    source.write_bytes(b'{"rows":[' + b",".join([b"0"] * 9998) + b"]}")
    assert len(cli._load_authority_json(source)["rows"]) == 9998


def test_authority_loader_rejects_missing_file_and_node_budget(tmp_path: Path) -> None:
    with pytest.raises(ProjectionError, match="cannot read adapter authority"):
        cli._load_authority_json(tmp_path / "missing.json")
    source = tmp_path / "wide.json"
    source.write_bytes(b'{"rows":[' + b",".join([b"0"] * 10001) + b"]}")
    with pytest.raises(ProjectionError, match="structure limit"):
        cli._load_authority_json(source)


def test_real_cli_rejects_duplicate_source_keys_before_publication(
    tmp_path: Path,
) -> None:
    from test_postdecontam_balance_projection_v1 import (
        DEDUP_EVIDENCE_SHA,
        DEDUP_HEAD_SHA,
        DEDUP_WORKER_ID,
        _build,
        _dedup_authority,
    )

    vector = _build(tmp_path)
    original = json.dumps(vector, ensure_ascii=False, separators=(",", ":"))
    vector_path = tmp_path / "family vector.json"
    vector_path.write_text(
        '{"schema":"forged",' + original[1:],
        encoding="utf-8",
    )
    authority_path = tmp_path / "dedup authority.json"
    authority_path.write_text(
        json.dumps(_dedup_authority(), ensure_ascii=False), encoding="utf-8"
    )
    before_vector = vector_path.read_bytes()
    before_authority = authority_path.read_bytes()
    output = tmp_path / "result.json"
    run = subprocess.run(
        [
            sys.executable, str(Path(cli.__file__).resolve()),
            "--family-vector", str(vector_path),
            "--expected-family-vector-identity-sha256",
            vector["family_vector_identity_sha256"],
            "--dedup-authority", str(authority_path),
            "--expected-dedup-worker-id", DEDUP_WORKER_ID,
            "--expected-dedup-head-sha", DEDUP_HEAD_SHA,
            "--expected-dedup-evidence-identity-sha256", DEDUP_EVIDENCE_SHA,
            "--output", str(output),
        ],
        text=True, capture_output=True, check=False,
    )
    assert run.returncode != 0
    assert "FAIL_CLOSED" in run.stderr
    assert "duplicate JSON key" in run.stderr
    assert not output.exists()
    assert vector_path.read_bytes() == before_vector
    assert authority_path.read_bytes() == before_authority


@pytest.mark.parametrize("destination", ["family-vector", "dedup-authority", "existing"])
def test_main_never_truncates_inputs_or_existing_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination: str,
) -> None:
    vector = tmp_path / "сімейний вектор.json"
    authority = tmp_path / "dedup authority.json"
    existing = tmp_path / "old verified result.json"
    vector.write_bytes(b"original vector")
    authority.write_bytes(b"original dedup authority")
    existing.write_bytes(b"original result")
    target = {
        "family-vector": vector,
        "dedup-authority": authority,
        "existing": existing,
    }[destination]
    _configure(monkeypatch, vector, authority, target)
    with pytest.raises(SystemExit, match="FAIL_CLOSED: refusing to overwrite"):
        cli.main()
    assert vector.read_bytes() == b"original vector"
    assert authority.read_bytes() == b"original dedup authority"
    assert existing.read_bytes() == b"original result"


def test_main_refuses_symlink_without_changing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = tmp_path / "vector.json"
    authority = tmp_path / "dedup.json"
    vector.write_bytes(b"original vector")
    authority.write_bytes(b"original authority")
    output = tmp_path / "output.json"
    try:
        output.symlink_to(vector)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks unavailable on this host")
    _configure(monkeypatch, vector, authority, output)
    with pytest.raises(SystemExit, match="FAIL_CLOSED: refusing to overwrite"):
        cli.main()
    assert output.is_symlink()
    assert vector.read_bytes() == b"original vector"


def test_main_writes_fresh_utf8_lf_output_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = tmp_path / "vector.json"
    authority = tmp_path / "dedup.json"
    vector.write_bytes(b"vector")
    authority.write_bytes(b"authority")
    output = tmp_path / "новий результат із пробілами.json"
    _configure(monkeypatch, vector, authority, output)
    assert cli.main() == 0
    raw = output.read_bytes()
    assert raw.endswith(b"\n")
    assert b"\r\n" not in raw
    result = json.loads(raw)
    assert result["value"] == "Український текст"
    assert result["claim_boundary"]["model_training_authorized"] is False
    with pytest.raises(SystemExit, match="FAIL_CLOSED: refusing to overwrite"):
        cli.main()
    assert output.read_bytes() == raw


def test_exclusive_create_refuses_destination_created_after_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = tmp_path / "vector.json"
    authority = tmp_path / "dedup.json"
    vector.write_bytes(b"vector")
    authority.write_bytes(b"authority")
    output = tmp_path / "racing result.json"
    original_link = cli.os.link

    def racing_link(source: Path, target: Path) -> None:
        if target == output:
            output.write_bytes(b"concurrent verified result")
        original_link(source, target)

    monkeypatch.setattr(cli.os, "link", racing_link)
    with pytest.raises(ProjectionError, match="refusing to overwrite"):
        cli._write_new_output(
            output, b"candidate", family_vector=vector, dedup_authority=authority,
        )
    assert output.read_bytes() == b"concurrent verified result"



def test_short_staging_write_is_rejected() -> None:
    class ShortWriter:
        def write(self, payload: bytes) -> int:
            return len(payload) - 1

        def flush(self) -> None:
            raise AssertionError("incomplete output must not be flushed")

        def fileno(self) -> int:
            raise AssertionError("incomplete output must not be synced")

    with pytest.raises(OSError, match="incomplete staged adapter output"):
        cli._write_staged_bytes(ShortWriter(), b"candidate")


def test_mid_staging_write_failure_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = tmp_path / "vector.json"
    authority = tmp_path / "authority.json"
    vector.write_bytes(b"vector")
    authority.write_bytes(b"authority")
    output = tmp_path / "result.json"

    def partial_write(destination: object, payload: bytes) -> None:
        destination.write(payload[:4])
        raise OSError("simulated disk full during write")

    monkeypatch.setattr(cli, "_write_staged_bytes", partial_write)
    with pytest.raises(ProjectionError, match="simulated disk full"):
        cli._write_new_output(
            output, b"candidate", family_vector=vector, dedup_authority=authority,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert vector.read_bytes() == b"vector"
    assert authority.read_bytes() == b"authority"


def test_staging_sync_failure_never_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = tmp_path / "vector.json"
    authority = tmp_path / "authority.json"
    vector.write_bytes(b"vector")
    authority.write_bytes(b"authority")
    output = tmp_path / "result.json"

    def fail_sync(_descriptor: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(cli.os, "fsync", fail_sync)
    with pytest.raises(ProjectionError, match="simulated fsync failure"):
        cli._write_new_output(
            output, b"candidate", family_vector=vector, dedup_authority=authority,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert vector.read_bytes() == b"vector"
    assert authority.read_bytes() == b"authority"


def test_main_does_not_publish_if_authority_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = tmp_path / "vector.json"
    authority = tmp_path / "dedup.json"
    vector.write_bytes(b"vector")
    authority.write_bytes(b"authority")
    output = tmp_path / "result.json"
    _configure(monkeypatch, vector, authority, output)

    def reject(*args: object, **kwargs: object) -> None:
        raise ProjectionError("source lineage mismatch")

    monkeypatch.setattr(cli, "adapt_family_vector_to_next100_106", reject)
    with pytest.raises(SystemExit, match="FAIL_CLOSED: source lineage mismatch"):
        cli.main()
    assert not output.exists()
    assert vector.read_bytes() == b"vector"
    assert authority.read_bytes() == b"authority"


@pytest.mark.parametrize("output_is_input", [False, True])
def test_real_cli_binds_valid_vector_and_preserves_original_bytes(
    tmp_path: Path, output_is_input: bool,
) -> None:
    """Execute the actual adapter with checked-in real validation code and synthetic inputs."""
    from test_postdecontam_balance_projection_v1 import (
        DEDUP_EVIDENCE_SHA,
        DEDUP_HEAD_SHA,
        DEDUP_WORKER_ID,
        _build,
        _dedup_authority,
    )

    vector = _build(tmp_path)
    vector_path = tmp_path / "source family vector.json"
    authority_path = tmp_path / "source dedup authority.json"
    vector_path.write_text(json.dumps(vector, ensure_ascii=False), encoding="utf-8")
    authority_path.write_text(
        json.dumps(_dedup_authority(), ensure_ascii=False), encoding="utf-8",
    )
    vector_original = vector_path.read_bytes()
    authority_original = authority_path.read_bytes()
    output = vector_path if output_is_input else tmp_path / "new vector.json"
    args = [
        sys.executable,
        str(Path(cli.__file__).resolve()),
        "--family-vector", str(vector_path),
        "--expected-family-vector-identity-sha256",
        vector["family_vector_identity_sha256"],
        "--dedup-authority", str(authority_path),
        "--expected-dedup-worker-id", DEDUP_WORKER_ID,
        "--expected-dedup-head-sha", DEDUP_HEAD_SHA,
        "--expected-dedup-evidence-identity-sha256", DEDUP_EVIDENCE_SHA,
        "--output", str(output),
    ]
    run = subprocess.run(args, capture_output=True, text=True, check=False)
    assert vector_path.read_bytes() == vector_original
    assert authority_path.read_bytes() == authority_original
    if output_is_input:
        assert run.returncode != 0
        assert "FAIL_CLOSED" in run.stderr
        return

    assert run.returncode == 0, run.stderr
    result = json.loads(output.read_bytes())
    assert result["schema_version"] == cli.adapt_family_vector_to_next100_106(
        vector,
        expected_family_vector_identity_sha256=vector["family_vector_identity_sha256"],
        dedup_authority=_dedup_authority(),
        expected_dedup_worker_id=DEDUP_WORKER_ID,
        expected_dedup_head_sha=DEDUP_HEAD_SHA,
        expected_dedup_evidence_identity_sha256=DEDUP_EVIDENCE_SHA,
    )["schema_version"]
    assert result["terminal"] is True
    assert output.read_bytes().endswith(b"\n")
    assert b"\r\n" not in output.read_bytes()
    second = subprocess.run(args, capture_output=True, text=True, check=False)
    assert second.returncode != 0
    assert "FAIL_CLOSED" in second.stderr
    assert json.loads(output.read_bytes()) == result


def test_staged_byte_mutation_after_sync_fails_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = tmp_path / "vector.json"
    authority = tmp_path / "dedup.json"
    output = tmp_path / "result.json"
    vector.write_bytes(b"unchanged vector")
    authority.write_bytes(b"unchanged authority")
    original_link = cli.os.link

    def tamper_then_link(stage: Path, result: Path) -> None:
        stage.write_bytes(b"foreign tampered result")
        original_link(stage, result)

    monkeypatch.setattr(cli.os, "link", tamper_then_link)
    with pytest.raises(ProjectionError, match="failed byte/path verification"):
        cli._write_new_output(
            output, b"validated expected output",
            family_vector=vector, dedup_authority=authority,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert vector.read_bytes() == b"unchanged vector"
    assert authority.read_bytes() == b"unchanged authority"


def test_output_parent_symlink_refused_before_staging(
    tmp_path: Path,
) -> None:
    vector = tmp_path / "vector.json"
    authority = tmp_path / "dedup.json"
    canonical = tmp_path / "canonical"
    alias = tmp_path / "alias"
    canonical.mkdir()
    vector.write_bytes(b"unchanged vector")
    authority.write_bytes(b"unchanged authority")
    try:
        alias.symlink_to(canonical, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable on this host")
    output = alias / "result.json"
    with pytest.raises(ProjectionError, match="parent must have no symlink"):
        cli._write_new_output(
            output, b"candidate", family_vector=vector, dedup_authority=authority,
        )
    assert not output.exists()
    assert not list(canonical.iterdir())
    assert vector.read_bytes() == b"unchanged vector"
    assert authority.read_bytes() == b"unchanged authority"

@pytest.mark.parametrize("bad_value", [
    float("nan"), float("inf"), float("-inf"), "\ud800", object(),
])
def test_main_rejects_invalid_result_json_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_value: object,
) -> None:
    vector = tmp_path / "unchanged family vector.json"
    authority = tmp_path / "unchanged dedup authority.json"
    vector.write_bytes(b"original vector")
    authority.write_bytes(b"original dedup")
    output = tmp_path / "must not publish.json"
    _configure(monkeypatch, vector, authority, output)
    monkeypatch.setattr(
        cli, "adapt_family_vector_to_next100_106",
        lambda *args, **kwargs: {"claim_boundary": {"model_training_authorized": False},
                                 "nested": {"bad": bad_value}},
    )
    with pytest.raises(
        SystemExit, match="FAIL_CLOSED: adapter result cannot be encoded as strict UTF-8 JSON",
    ):
        cli.main()
    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert vector.read_bytes() == b"original vector"
    assert authority.read_bytes() == b"original dedup"


def test_verified_publication_with_stage_cleanup_failure_is_truthful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    vector = tmp_path / "vector.json"
    authority = tmp_path / "authority.json"
    output = tmp_path / "опублікований результат.json"
    vector.write_bytes(b"original vector")
    authority.write_bytes(b"original authority")
    _configure(monkeypatch, vector, authority, output)
    original_unlink = Path.unlink

    def refuse_stage_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.startswith(f".{output.name}.") and self.suffix == ".tmp":
            raise PermissionError("injected stage sharing violation")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_stage_unlink)
    assert cli.main() == 0
    emitted = capsys.readouterr()
    assert emitted.out.strip() == str(output)
    assert emitted.err.startswith("OUTPUT_COMMITTED_CLEANUP_PENDING: ")
    warning = json.loads(emitted.err.split(": ", 1)[1])
    assert warning["output"] == str(output)
    assert len(list(tmp_path.glob(f".{output.name}.*.tmp"))) == 1
    assert json.loads(output.read_bytes())["value"] == "Український текст"
    assert vector.read_bytes() == b"original vector"
    assert authority.read_bytes() == b"original authority"
    with pytest.raises(SystemExit, match="FAIL_CLOSED: refusing to overwrite"):
        cli.main()
    assert json.loads(output.read_bytes())["value"] == "Український текст"


def test_failed_rollback_explicitly_reports_unverified_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = tmp_path / "vector.json"
    authority = tmp_path / "authority.json"
    output = tmp_path / "result.json"
    vector.write_bytes(b"original vector")
    authority.write_bytes(b"original authority")
    original_link = cli.os.link
    original_unlink = Path.unlink

    def tamper_then_link(stage: Path, final: Path) -> None:
        stage.write_bytes(b"unverified tampered bytes")
        original_link(stage, final)

    def refuse_rollback(self: Path, *args: object, **kwargs: object) -> None:
        if self == output:
            raise PermissionError("injected final sharing violation")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(cli.os, "link", tamper_then_link)
    monkeypatch.setattr(Path, "unlink", refuse_rollback)
    with pytest.raises(ProjectionError, match="ROLLBACK_INCOMPLETE"):
        cli._write_new_output(
            output, b"verified expected bytes",
            family_vector=vector, dedup_authority=authority,
        )
    assert output.read_bytes() == b"unverified tampered bytes"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))
    assert vector.read_bytes() == b"original vector"
    assert authority.read_bytes() == b"original authority"


def test_unpublished_stage_cleanup_failure_is_not_reported_as_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = tmp_path / "vector.json"
    authority = tmp_path / "authority.json"
    output = tmp_path / "result.json"
    vector.write_bytes(b"original vector")
    authority.write_bytes(b"original authority")
    original_unlink = Path.unlink

    def refuse_stage_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.startswith(f".{output.name}.") and self.suffix == ".tmp":
            raise PermissionError("injected stage sharing violation")
        original_unlink(self, *args, **kwargs)

    def unsupported_link(_stage: Path, _final: Path) -> None:
        raise OSError("injected link failure")

    monkeypatch.setattr(Path, "unlink", refuse_stage_unlink)
    monkeypatch.setattr(cli.os, "link", unsupported_link)
    with pytest.raises(ProjectionError, match="STAGING_CLEANUP_INCOMPLETE"):
        cli._write_new_output(
            output, b"candidate",
            family_vector=vector, dedup_authority=authority,
        )
    assert not output.exists()
    assert len(list(tmp_path.glob(f".{output.name}.*.tmp"))) == 1
    assert vector.read_bytes() == b"original vector"
    assert authority.read_bytes() == b"original authority"


def test_committed_unicode_output_survives_narrow_console_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = tmp_path / "vector.json"
    authority = tmp_path / "dedup authority.json"
    output = tmp_path / "підтверджений результат.json"
    vector.write_bytes(b"original vector")
    authority.write_bytes(b"original authority")
    _configure(monkeypatch, vector, authority, output)

    class NarrowConsole:
        encoding = "cp1252"

        def __init__(self) -> None:
            self.value = ""

        def write(self, text: str) -> int:
            text.encode(self.encoding, errors="strict")
            self.value += text
            return len(text)

        def flush(self) -> None:
            pass

    console = NarrowConsole()
    monkeypatch.setattr(cli.sys, "stdout", console)
    assert cli.main() == 0
    assert output.is_file()
    assert json.loads(output.read_bytes())["value"] == "Український текст"
    assert "\\u" in console.value
    assert "json" in console.value
    assert vector.read_bytes() == b"original vector"
    assert authority.read_bytes() == b"original authority"


def test_real_text_stream_stdout_fallback_preserves_exact_committed_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = tmp_path / "vector.json"
    authority = tmp_path / "dedup.json"
    output = tmp_path / "кириличний шлях із пробілами.json"
    vector.write_bytes(b"original vector")
    authority.write_bytes(b"original authority")
    _configure(monkeypatch, vector, authority, output)
    narrow_stdout = io.TextIOWrapper(
        io.BytesIO(), encoding="cp1252", errors="strict",
    )
    warning_stderr = io.StringIO()
    monkeypatch.setattr(cli.sys, "stdout", narrow_stdout)
    monkeypatch.setattr(cli.sys, "stderr", warning_stderr)
    assert cli.main() == 0
    narrow_stdout.flush()
    emitted = narrow_stdout.buffer.getvalue()
    assert b"\\u" in emitted and b".json" in emitted
    warning = warning_stderr.getvalue()
    assert warning.startswith("OUTPUT_COMMITTED_STDOUT_ENCODING_UNAVAILABLE: ")
    assert warning.isascii()
    assert json.loads(warning.split(": ", 1)[1])["output"] == str(output)
    assert json.loads(output.read_bytes())["value"] == "Український текст"
    assert vector.read_bytes() == b"original vector"
    assert authority.read_bytes() == b"original authority"
    with pytest.raises(SystemExit, match="FAIL_CLOSED: refusing to overwrite"):
        cli.main()


@pytest.mark.parametrize("stream_encoding", ["cp1252", "ascii"])
def test_committed_warning_survives_restricted_stdout_and_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream_encoding: str,
) -> None:
    vector = tmp_path / "family vector.json"
    authority = tmp_path / "dedup authority.json"
    output = tmp_path / "український шлях із пробілами.json"
    vector.write_bytes(b"original vector")
    authority.write_bytes(b"original authority")
    _configure(monkeypatch, vector, authority, output)
    narrow_stdout = io.TextIOWrapper(
        io.BytesIO(), encoding=stream_encoding, errors="strict",
    )
    narrow_stderr = io.TextIOWrapper(
        io.BytesIO(), encoding=stream_encoding, errors="strict",
    )
    monkeypatch.setattr(cli.sys, "stdout", narrow_stdout)
    monkeypatch.setattr(cli.sys, "stderr", narrow_stderr)
    assert cli.main() == 0
    narrow_stdout.flush()
    narrow_stderr.flush()
    assert b"\\u" in narrow_stdout.buffer.getvalue()
    warning = narrow_stderr.buffer.getvalue().decode("ascii")
    assert warning.startswith("OUTPUT_COMMITTED_STDOUT_ENCODING_UNAVAILABLE: ")
    assert json.loads(warning.split(": ", 1)[1])["output"] == str(output)
    assert json.loads(output.read_bytes())["value"] == "Український текст"
    assert vector.read_bytes() == b"original vector"
    assert authority.read_bytes() == b"original authority"
