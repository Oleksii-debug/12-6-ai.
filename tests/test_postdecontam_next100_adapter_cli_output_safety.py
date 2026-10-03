"""Source and evidence preservation for the D03 -> NEXT100 adapter CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.adapt_d03_postdecontam_family_vector_to_next100_106_v1 as cli
from twelve_six.data.postdecontam_balance_projection_v1 import ProjectionError


def _configure(monkeypatch: pytest.MonkeyPatch, vector: Path, authority: Path, output: Path) -> None:
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
    monkeypatch.setattr(cli, "load_json", lambda path: {"path": str(path)})
    monkeypatch.setattr(
        cli, "adapt_family_vector_to_next100_106",
        lambda *args, **kwargs: {
            "schema_version": "test-only",
            "claim_boundary": {"model_training_authorized": False},
            "value": "Український текст",
        },
    )


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
    original_open = Path.open

    def racing_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        if path == output and mode == "xb":
            with open(output, "wb") as incumbent:
                incumbent.write(b"concurrent verified result")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    with pytest.raises(ProjectionError, match="refusing to overwrite"):
        cli._write_new_output(
            output, b"candidate", family_vector=vector, dedup_authority=authority,
        )
    assert output.read_bytes() == b"concurrent verified result"


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
