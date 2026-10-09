"""S15 clean-checkout proof refuses unsafe archives and forged release flags."""
from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from tools import plan2_s15_git_clean_checkout_v1 as clean


def _tar(name: str, raw: bytes = b"safe", kind: bytes | None = None) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        item = tarfile.TarInfo(name)
        if kind is not None:
            item.type = kind
        else:
            item.size = len(raw)
        archive.addfile(item, io.BytesIO(raw) if kind is None else None)
    return stream.getvalue()


@pytest.mark.parametrize("path", ["/etc/passwd", "../escape", "a/../escape",
                                  "a\\malformed", "./ambiguous"])
def test_unsafe_member_is_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(clean.GitCheckoutDenied, match="unsafe"):
        clean._extract(_tar(path), tmp_path)


def test_symlink_member_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(clean.GitCheckoutDenied, match="link or special"):
        clean._extract(_tar("point", kind=tarfile.SYMTYPE), tmp_path)


def test_normal_member_is_exact_byte_copy(tmp_path: Path) -> None:
    assert clean._extract(_tar("a/b.txt", b"immutable\n"), tmp_path) == 1
    assert (tmp_path / "a/b.txt").read_bytes() == b"immutable\n"


def test_existing_source_file_cannot_be_overwritten(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("previous")
    with pytest.raises(clean.GitCheckoutDenied, match="overwrite"):
        clean._extract(_tar("a.txt"), tmp_path)


def test_stage_refuses_symlink_before_running_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "directory"
    target.mkdir()
    link = tmp_path / "symlink"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(clean, "verify", lambda _: pytest.fail("unexpected audit"))
    with pytest.raises(clean.GitCheckoutDenied, match="symlink"):
        clean.stage(tmp_path, link)


def test_immutable_receipt_refuses_changed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = {"schema_version": clean.SCHEMA,
               "decision": "CLEAN_GIT_CHECKOUT_REPRODUCED_NOT_PRODUCTION_RELEASE",
               "terminal_done": False, "production_release_authorized": False}
    monkeypatch.setattr(clean, "verify", lambda _: fixture)
    out = tmp_path / "out"
    clean.stage(tmp_path, out)
    (out / clean.OUTPUT).write_bytes(b'{"terminal_done":true}\n')
    with pytest.raises(clean.GitCheckoutDenied, match="immutable"):
        clean.stage(tmp_path, out)


def test_git_head_dirt_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".git").mkdir()

    def fake_git(_root: Path, *args: str) -> bytes:
        if args[0] == "rev-parse" and "--show-toplevel" in args:
            return str(tmp_path).encode()
        if args[0] == "status":
            return b" M configs/data/plan2_source_inventory_v1.json\n"
        return b""

    monkeypatch.setattr(clean, "_git", fake_git)
    with pytest.raises(clean.GitCheckoutDenied, match="dirty"):
        clean._source(tmp_path)
