"""W4 S17: same-byte physical replacements must invalidate NONRELEASE receipts."""
from pathlib import Path
import os
import pytest
from tools import plan2_s17_clean_rebuild_witness_v1 as w

def trees(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    for root in (a, b):
        root.mkdir()
        (root / "tokenizer.bin").write_bytes(b"synthetic-fixture")
    return a, b, w.digest(w.canonical(w.inventory(a)[0]))

def test_stable_nonrelease(tmp_path):
    a, b, h = trees(tmp_path)
    result = w.compare_rebuilds(a, b, expected_inventory_sha256=h)
    assert result["byte_identical"] and not result["terminal_done"]
    assert not result["production_release_authorized"]

@pytest.mark.parametrize("which", (0, 1))
def test_same_bytes_new_inode_denied(tmp_path, monkeypatch, which):
    a, b, h = trees(tmp_path)
    original, calls = w.inventory, [0]
    target = (a, b)[which] / "tokenizer.bin"
    def changed(root):
        result = original(root)
        calls[0] += 1
        if calls[0] == 1:
            old = target.with_name("old")
            inode = target.stat().st_ino
            target.rename(old)
            target.write_bytes(b"synthetic-fixture")
            if target.stat().st_ino == inode:
                pytest.skip("inode reused")
            old.unlink()
        return result
    monkeypatch.setattr(w, "inventory", changed)
    with pytest.raises(w.RebuildDenied, match="physical member identity"):
        w.compare_rebuilds(a, b, expected_inventory_sha256=h)

def test_hardlink_during_scan_denied(tmp_path, monkeypatch):
    a, b, h = trees(tmp_path)
    original, calls = w.inventory, [0]
    def changed(root):
        result = original(root)
        calls[0] += 1
        if calls[0] == 1:
            try:
                os.link(a / "tokenizer.bin", tmp_path / "extra-link")
            except OSError:
                pytest.skip("hardlinks unavailable")
        return result
    monkeypatch.setattr(w, "inventory", changed)
    with pytest.raises(w.RebuildDenied):
        w.compare_rebuilds(a, b, expected_inventory_sha256=h)
