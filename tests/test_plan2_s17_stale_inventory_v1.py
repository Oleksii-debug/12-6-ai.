"""S17 nonproduction: pinned inventories reject stale or changed physical lineage.

These fixtures exercise only the byte-witness guard. They do not establish
independent checkout, lawful corpus admission, tokenizer identity or release.
"""

import os
from pathlib import Path

import pytest

from tools import plan2_s17_clean_rebuild_witness_v1 as witness


def _pair(tmp_path: Path) -> tuple[Path, Path, str]:
    a, b = tmp_path / "first", tmp_path / "second"
    for root in (a, b):
        root.mkdir()
        (root / "shard-0001.bin").write_bytes(b"nonproduction-shard")
    expected = witness.digest(witness.canonical(witness.inventory(a)[0]))
    return a, b, expected


def test_stale_pin_denies_identically_changed_bytes(tmp_path: Path) -> None:
    a, b, expected = _pair(tmp_path)
    for root in (a, b):
        (root / "shard-0001.bin").write_bytes(b"changed-shard")
    with pytest.raises(witness.RebuildDenied, match="independent expected hash"):
        witness.compare_rebuilds(a, b, expected_inventory_sha256=expected)


def test_stale_pin_denies_identically_changed_membership(tmp_path: Path) -> None:
    a, b, expected = _pair(tmp_path)
    for root in (a, b):
        (root / "shard-0001.bin").rename(root / "shard-0002.bin")
    with pytest.raises(witness.RebuildDenied, match="independent expected hash"):
        witness.compare_rebuilds(a, b, expected_inventory_sha256=expected)


def test_different_membership_denied_even_when_original_pin_matches(tmp_path: Path) -> None:
    a, b, expected = _pair(tmp_path)
    (b / "unexpected.bin").write_bytes(b"nonproduction-extra")
    with pytest.raises(witness.RebuildDenied, match="member bytes differ"):
        witness.compare_rebuilds(a, b, expected_inventory_sha256=expected)


def test_reused_root_via_symlink_denied(tmp_path: Path) -> None:
    a, b, expected = _pair(tmp_path)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(a, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")
    with pytest.raises(witness.RebuildDenied, match="linked rebuild root"):
        witness.compare_rebuilds(alias, b, expected_inventory_sha256=expected)


def test_hardlinked_member_denied(tmp_path: Path) -> None:
    a, b, expected = _pair(tmp_path)
    try:
        os.link(a / "shard-0001.bin", tmp_path / "external-link")
    except OSError:
        pytest.skip("hardlinks unavailable")
    with pytest.raises(witness.RebuildDenied, match="hardlinked"):
        witness.compare_rebuilds(a, b, expected_inventory_sha256=expected)


def test_invalid_expected_pin_denied(tmp_path: Path) -> None:
    a, b, _ = _pair(tmp_path)
    with pytest.raises(witness.RebuildDenied, match="independently pinned lowercase"):
        witness.compare_rebuilds(a, b, expected_inventory_sha256="A" * 64)
