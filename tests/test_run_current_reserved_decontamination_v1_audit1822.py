from __future__ import annotations

from pathlib import Path

import pytest

import tools.run_current_reserved_decontamination_v1 as runner


def test_publish_race_preserves_concurrent_empty_destination(
    tmp_path: Path, monkeypatch
):
    output_dir = tmp_path / "bundle"
    original = runner._rename_directory_no_replace

    def race(source: Path, destination: Path) -> None:
        destination.mkdir()
        original(source, destination)

    monkeypatch.setattr(runner, "_rename_directory_no_replace", race)
    files = {
        runner.REPORT_NAME: b"{}\n",
        runner.EVIDENCE_NAME: b"{}\n",
        runner.RECEIPT_NAME: b"{}\n",
    }

    with pytest.raises(FileExistsError):
        runner._publish_bundle(output_dir, files)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []
    assert not list(tmp_path.glob(".bundle.tmp-*"))
