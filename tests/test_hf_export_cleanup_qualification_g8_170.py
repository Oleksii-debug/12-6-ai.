"""Qualification-only adversarial checks for D05 HF temporary cleanup.

This file is carried only by DO-NOT-MERGE qualification PR #2957. It grants
zero training/scientific authority and must not be treated as Product source.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from twelve_six.checkpoint import hf_export


class _HostileNotePrimary(RuntimeError):
    def add_note(self, _note: str) -> None:
        raise AssertionError("virtual add_note dispatch is forbidden")


def _fake_verified() -> SimpleNamespace:
    return SimpleNamespace(_manifest_bytes=b"{}", _artifacts={})


@pytest.mark.parametrize(
    "interrupt_type",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
)
def test_verified_reference_materialization_interrupt_cleans_private_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
) -> None:
    primary = interrupt_type("verified reference materialization interrupted")
    real_write_bytes = Path.write_bytes

    def interrupt_reference_write(path: Path, data: bytes) -> int:
        if ".reference-" in path.parent.name:
            raise primary
        return real_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", interrupt_reference_write)

    with pytest.raises(interrupt_type) as caught:
        hf_export._materialize_verified_reference(
            _fake_verified(),
            tmp_path,
            "hf",
        )

    assert caught.value is primary
    assert not list(tmp_path.glob(".hf.reference-*"))


def test_verified_reference_cleanup_double_fault_preserves_hostile_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = _HostileNotePrimary("verified reference primary")
    real_write_bytes = Path.write_bytes
    real_rmtree = hf_export.shutil.rmtree

    def fail_reference_write(path: Path, data: bytes) -> int:
        if ".reference-" in path.parent.name:
            raise primary
        return real_write_bytes(path, data)

    def fail_reference_cleanup(path, *args, **kwargs):
        if ".reference-" in Path(path).name:
            raise OSError("verified reference cleanup double-fault")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_bytes", fail_reference_write)
    monkeypatch.setattr(hf_export.shutil, "rmtree", fail_reference_cleanup)
    try:
        with pytest.raises(_HostileNotePrimary) as caught:
            hf_export._materialize_verified_reference(
                _fake_verified(),
                tmp_path,
                "hf",
            )

        assert caught.value is primary
        assert any(
            "verified checkpoint reference cleanup also failed" in note
            for note in getattr(primary, "__notes__", ())
        )
        assert list(tmp_path.glob(".hf.reference-*"))
    finally:
        monkeypatch.setattr(hf_export.shutil, "rmtree", real_rmtree)
        for path in tmp_path.glob(".hf.reference-*"):
            real_rmtree(path)


def test_malformed_primary_notes_do_not_mask_cleanup_double_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / ".hf.staging-malformed-notes"
    root.mkdir()
    identity = hf_export._temporary_directory_identity(root)
    primary = RuntimeError("publish primary with malformed notes")
    primary.__notes__ = "malformed-note-storage"

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("staging cleanup double-fault")

    monkeypatch.setattr(hf_export, "_remove_temp_path_strict", fail_cleanup)

    hf_export._cleanup_temp_paths_strict(
        ((root, "HF export staging", identity),),
        primary_exc=primary,
    )

    assert primary.__notes__ == "malformed-note-storage"
