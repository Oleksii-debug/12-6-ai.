"""S17 nonproduction: directory-only changes must not get byte-identity credit."""
from pathlib import Path

import pytest

from tools import plan2_s17_clean_rebuild_witness_v1 as witness


def _roots(tmp_path: Path) -> tuple[Path, Path, str]:
    first, second = tmp_path / "first", tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        (root / "tokenizer.bin").write_bytes(b"synthetic-nonrelease-fixture")
    expected = witness.digest(witness.canonical(witness.inventory(first)[0]))
    return first, second, expected


@pytest.mark.parametrize("which", (0, 1))
def test_one_sided_unrepresented_empty_directory_denied(tmp_path: Path, which: int) -> None:
    first, second, expected = _roots(tmp_path)
    ((first, second)[which] / "unexpected-empty").mkdir()
    with pytest.raises(witness.RebuildDenied, match="empty artifact directory"):
        witness.compare_rebuilds(first, second, expected_inventory_sha256=expected)


def test_identical_but_unpinned_empty_directories_denied(tmp_path: Path) -> None:
    first, second, expected = _roots(tmp_path)
    for root in (first, second):
        (root / "unrepresented-empty").mkdir()
    with pytest.raises(witness.RebuildDenied, match="empty artifact directory"):
        witness.compare_rebuilds(first, second, expected_inventory_sha256=expected)


def test_populated_directory_is_part_of_pinned_file_inventory(tmp_path: Path) -> None:
    first, second, _ = _roots(tmp_path)
    for root in (first, second):
        (root / "shards").mkdir()
        (root / "shards" / "0001.bin").write_bytes(b"fixture-shard")
    expected = witness.digest(witness.canonical(witness.inventory(first)[0]))
    result = witness.compare_rebuilds(first, second, expected_inventory_sha256=expected)
    assert result["byte_identical"] is True
    assert result["production_release_authorized"] is False
