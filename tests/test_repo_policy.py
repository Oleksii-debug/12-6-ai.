from __future__ import annotations

from pathlib import Path

import pytest

from twelve_six.integration.repo_policy import (
    MAX_TRACKED_BYTES,
    RepositoryPolicyError,
    validate_tracked_paths,
)


def _write(root: Path, relative: str, payload: bytes = b"ok") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_regular_small_source_file_is_allowed(tmp_path: Path) -> None:
    _write(tmp_path, "src/example.py")
    validate_tracked_paths(tmp_path, ("src/example.py",))


@pytest.mark.parametrize(
    "relative",
    [
        "weights.safetensors",
        "model.ckpt",
        "model.pth",
        "model.pt",
        "model.gguf",
        "bundle.zip",
        "bundle.7z",
        "bundle.rar",
        "bundle.tar",
        "bundle.tgz",
        "bundle.tar.gz",
        "artifacts/run/metrics.json",
        "checkpoints/s0/manifest.json",
        "private-data/customer.jsonl",
        "private_data/customer.jsonl",
        "secrets/token.txt",
        ".env",
        "credentials.json",
        "service-account.json",
    ],
)
def test_model_archive_runtime_and_private_paths_are_rejected(tmp_path: Path, relative: str) -> None:
    _write(tmp_path, relative)
    with pytest.raises(RepositoryPolicyError):
        validate_tracked_paths(tmp_path, (relative,))


def test_large_tracked_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "data" / "large.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("wb") as handle:
        handle.truncate(MAX_TRACKED_BYTES + 1)
    with pytest.raises(RepositoryPolicyError, match="exceeds"):
        validate_tracked_paths(tmp_path, ("data/large.jsonl",))


def test_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable in this test environment")
    with pytest.raises(RepositoryPolicyError, match="symlink"):
        validate_tracked_paths(tmp_path, ("link.txt",))


def test_unsafe_relative_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(RepositoryPolicyError, match="unsafe tracked path"):
        validate_tracked_paths(tmp_path, ("../outside",))
