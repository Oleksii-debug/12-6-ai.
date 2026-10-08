"""Plan 1 Section 5 deterministic release fixture, recovery and adversarial tests."""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from twelve_six.integration.release_fixture import (
    FixtureIntegrityError, admit_rollback, admit_update, inspect_fixture,
    pack_fixture, sign_fixture, verify_fixture,
)

_KEY = bytes(range(32))


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path, name: str, text: str) -> bytes:
    (tmp_path / name).write_text(text, encoding="utf-8")
    return pack_fixture(tmp_path, [name])


def test_reproducible_archive_and_exact_inventory(tmp_path: Path) -> None:
    (tmp_path / "a").write_text("a")
    (tmp_path / "b").write_text("b")
    first = pack_fixture(tmp_path, ["b", "a"])
    assert first == pack_fixture(tmp_path, ["a", "b"])
    assert inspect_fixture(first, _sha(first)) == {"a": _sha(b"a"), "b": _sha(b"b")}


@pytest.mark.parametrize("name", ["../outside", "/tmp/absolute", "a/../b", "a//b", "./a", "a\\b"])
def test_unsafe_paths_fail_closed(tmp_path: Path, name: str) -> None:
    with pytest.raises(FixtureIntegrityError):
        pack_fixture(tmp_path, [name])


def test_symlink_and_duplicate_fail_closed(tmp_path: Path) -> None:
    _fixture(tmp_path, "a", "a")
    with pytest.raises(FixtureIntegrityError, match="duplicate"):
        pack_fixture(tmp_path, ["a", "a"])
    link = tmp_path / "link"
    try:
        link.symlink_to(tmp_path / "a")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    with pytest.raises(FixtureIntegrityError, match="symlink"):
        pack_fixture(tmp_path, ["link"])


def test_fixture_sign_verify_update_rollback_recovery(tmp_path: Path) -> None:
    old_bytes = _fixture(tmp_path, "v1", "release 1")
    new_bytes = _fixture(tmp_path, "v2", "release 2")
    old = sign_fixture(key=_KEY, archive_sha256=_sha(old_bytes), release_id="unit.test",
                       generation=1, previous_sha256=None)
    new = sign_fixture(key=_KEY, archive_sha256=_sha(new_bytes), release_id="unit.test",
                       generation=2, previous_sha256=_sha(old_bytes))
    verify_fixture(key=_KEY, receipt=old, archive_bytes=old_bytes)
    admit_update(current=old, proposed=new, key=_KEY, current_bytes=old_bytes, proposed_bytes=new_bytes)
    admit_rollback(current=new, previous=old, key=_KEY, current_bytes=new_bytes, previous_bytes=old_bytes)
    with pytest.raises(FixtureIntegrityError):
        admit_update(current=new, proposed=old, key=_KEY, current_bytes=new_bytes, proposed_bytes=old_bytes)
    with pytest.raises(FixtureIntegrityError):
        admit_rollback(current=old, previous=new, key=_KEY, current_bytes=old_bytes, previous_bytes=new_bytes)


def test_corruption_key_rotation_and_receipt_forgery_fail_closed(tmp_path: Path) -> None:
    payload = _fixture(tmp_path, "payload", "bytes")
    receipt = sign_fixture(key=_KEY, archive_sha256=_sha(payload), release_id="unit.test",
                           generation=1, previous_sha256=None)
    with pytest.raises(FixtureIntegrityError, match="signature"):
        verify_fixture(key=bytes(range(1, 33)), receipt=receipt, archive_bytes=payload)
    with pytest.raises(FixtureIntegrityError, match="artifact hash"):
        verify_fixture(key=_KEY, receipt=receipt, archive_bytes=payload + b"bad")
    with pytest.raises(FixtureIntegrityError, match="signature"):
        verify_fixture(key=_KEY, receipt=dict(receipt, release_id="different"), archive_bytes=payload)
    with pytest.raises(FixtureIntegrityError, match="schema"):
        verify_fixture(key=_KEY, receipt=dict(receipt, admin_override=True), archive_bytes=payload)


def test_reject_unsafe_archive_members(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("../escape", "unsafe")
    with pytest.raises(FixtureIntegrityError, match="unsafe"):
        inspect_fixture(buf.getvalue(), _sha(buf.getvalue()))
    with pytest.raises(FixtureIntegrityError, match="artifact hash"):
        inspect_fixture(buf.getvalue(), "0" * 64)


def test_reject_invalid_signature_inputs(tmp_path: Path) -> None:
    payload = _fixture(tmp_path, "p", "a")
    for gen in (True, 0, 1.5, "1"):
        with pytest.raises(FixtureIntegrityError):
            sign_fixture(key=_KEY, archive_sha256=_sha(payload), release_id="unit.test",
                         generation=gen, previous_sha256=None)
    with pytest.raises(FixtureIntegrityError):
        sign_fixture(key=b"weak", archive_sha256=_sha(payload), release_id="unit.test",
                     generation=1, previous_sha256=None)


def test_forged_current_receipt_blocks_update_and_rollback(tmp_path: Path) -> None:
    old_bytes = _fixture(tmp_path, "a", "old")
    new_bytes = _fixture(tmp_path, "b", "new")
    old = sign_fixture(key=_KEY, archive_sha256=_sha(old_bytes), release_id="fixture",
                       generation=1, previous_sha256=None)
    new = sign_fixture(key=_KEY, archive_sha256=_sha(new_bytes), release_id="fixture",
                       generation=2, previous_sha256=_sha(old_bytes))
    with pytest.raises(FixtureIntegrityError, match="signature"):
        admit_update(current=dict(old, generation=99), proposed=new, key=_KEY,
                     current_bytes=old_bytes, proposed_bytes=new_bytes)
    with pytest.raises(FixtureIntegrityError, match="signature"):
        admit_rollback(current=dict(new, generation=99), previous=old, key=_KEY,
                       current_bytes=new_bytes, previous_bytes=old_bytes)
