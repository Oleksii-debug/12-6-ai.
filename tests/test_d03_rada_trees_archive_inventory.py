from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/inventory_d03_rada_trees_archive.py"
spec = importlib.util.spec_from_file_location("rada_inventory", MODULE_PATH)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_config_is_fail_closed() -> None:
    value = module.load_config()
    assert value["claim_boundary"]["training_authorized_bytes"] == 0
    assert value["claim_boundary"]["archive_sha256_pinned"] is False
    assert value["primary_archive"]["exact_content_sha256"] is None
    assert value["parent"]["probe_head_sha"] == "92c1fd05d4399b0f0c4a35f0689160383f963c9c"


def test_listing_parser_is_sorted_and_rejects_duplicate_nfkc_paths() -> None:
    listing = """Path = b.txt\nSize = 2\nFolder = -\n\nPath = a.txt\nSize = 1\nFolder = -\n"""
    parsed = module.parse_7z_slt_listing(listing)
    assert parsed == [{"path": "a.txt", "size_bytes": 1}, {"path": "b.txt", "size_bytes": 2}]

    collision = """Path = café.txt\nSize = 1\nFolder = -\n\nPath = café.txt\nSize = 1\nFolder = -\n"""
    with pytest.raises(ValueError, match="duplicate normalized"):
        module.parse_7z_slt_listing(collision)


def test_member_path_rejects_traversal_and_absolute_paths() -> None:
    for raw in ("../escape.txt", "/abs.txt", "C:/abs.txt", "a/../b.txt", "a//b.txt"):
        with pytest.raises(ValueError):
            module.canonical_member_path(raw)
    assert module.canonical_member_path("folder\\file.txt") == "folder/file.txt"


def test_inventory_tree_is_deterministic_and_hash_bound(tmp_path: Path) -> None:
    (tmp_path / "z").mkdir()
    (tmp_path / "z/b.txt").write_bytes(b"beta")
    (tmp_path / "a.txt").write_bytes(b"alpha")
    expected = [{"path": "a.txt", "size_bytes": 5}, {"path": "z/b.txt", "size_bytes": 4}]
    first = module.inventory_extracted_tree(tmp_path, expected, 100, 1000)
    second = module.inventory_extracted_tree(tmp_path, expected, 100, 1000)
    assert first == second
    assert [item["path"] for item in first] == ["a.txt", "z/b.txt"]
    old_hash = first[0]["sha256"]
    (tmp_path / "a.txt").write_bytes(b"alphA")
    third = module.inventory_extracted_tree(tmp_path, expected, 100, 1000)
    assert third[0]["sha256"] != old_hash


def test_inventory_rejects_listing_mismatch(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"a")
    with pytest.raises(ValueError, match="listing/extraction mismatch"):
        module.inventory_extracted_tree(tmp_path, [{"path": "missing.txt", "size_bytes": 1}], 100, 1000)


def test_inventory_rejects_symlink_and_size_bounds(tmp_path: Path) -> None:
    (tmp_path / "real.txt").write_bytes(b"12345")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(tmp_path / "real.txt")
    except OSError:
        pytest.skip("symlinks unavailable")
    expected = [{"path": "link.txt", "size_bytes": 5}, {"path": "real.txt", "size_bytes": 5}]
    with pytest.raises(ValueError, match="symlink forbidden"):
        module.inventory_extracted_tree(tmp_path, expected, 100, 1000)

    link.unlink()
    with pytest.raises(ValueError, match="max_single_member_bytes"):
        module.inventory_extracted_tree(tmp_path, [{"path": "real.txt", "size_bytes": 5}], 4, 1000)


def test_validate_bounds_rejects_total_overflow() -> None:
    entries = [{"path": "a", "size_bytes": 6}, {"path": "b", "size_bytes": 6}]
    with pytest.raises(ValueError, match="max_total_uncompressed_bytes"):
        module.validate_bounds(entries, 10, 10)
