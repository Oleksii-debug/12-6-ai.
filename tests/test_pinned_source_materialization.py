from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from twelve_six.data.pinned_source_materialization import (
    PinnedSourceMaterializationError,
    materialize_pinned_sources,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _make_git_source(tmp_path: Path) -> tuple[Path, str, bytes]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@example.invalid")
    payload = "детермінований тестовий payload\n".encode("utf-8")
    path = repo / "data" / "sample.txt"
    path.parent.mkdir()
    path.write_bytes(payload)
    _git(repo, "add", "data/sample.txt")
    _git(repo, "commit", "-m", "fixture")
    return repo, _git(repo, "rev-parse", "HEAD"), payload


def _base_config(source: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "12-6.pinned-source-payload-materialization.v1",
        "local_free_only": True,
        "model_training_executed": False,
        "sources": [source],
    }


def _common_source(payload: bytes) -> dict[str, object]:
    return {
        "source_id": "fixture-source",
        "source_family": "ua.fixture.family",
        "stratum": "ua",
        "authority_id": "FIXTURE-AUTHORITY",
        "authority_identity_sha256": "a" * 64,
        "expected_raw_bytes": len(payload),
        "expected_raw_sha256": _sha256(payload),
    }


def test_materializes_exact_git_object_and_is_deterministic(tmp_path: Path) -> None:
    repo, commit, payload = _make_git_source(tmp_path)
    source = {
        **_common_source(payload),
        "provider": "git_object",
        "git_commit": commit,
        "git_path": "data/sample.txt",
    }
    config = _base_config(source)

    first = materialize_pinned_sources(
        config,
        repo_root=repo,
        output_dir=tmp_path / "out-a",
    )
    second = materialize_pinned_sources(
        config,
        repo_root=repo,
        output_dir=tmp_path / "out-b",
    )

    assert first == second
    assert first["source_count"] == 1
    assert first["total_raw_bytes"] == len(payload)
    assert first["model_training_executed"] is False
    assert (tmp_path / "out-a" / "fixture-source.raw").read_bytes() == payload
    assert (tmp_path / "out-a" / "manifest.json").read_bytes() == (
        tmp_path / "out-b" / "manifest.json"
    ).read_bytes()


def test_rejects_hash_drift_before_writing_payload(tmp_path: Path) -> None:
    repo, commit, payload = _make_git_source(tmp_path)
    source = {
        **_common_source(payload),
        "provider": "git_object",
        "git_commit": commit,
        "git_path": "data/sample.txt",
        "expected_raw_sha256": "0" * 64,
    }
    out = tmp_path / "out"

    with pytest.raises(PinnedSourceMaterializationError, match="raw SHA-256 drift"):
        materialize_pinned_sources(_base_config(source), repo_root=repo, output_dir=out)

    assert not (out / "fixture-source.raw").exists()
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize("git_path", ["../escape.txt", "/absolute.txt", "a\\b.txt", "HEAD:data.txt"])
def test_rejects_unsafe_git_paths(tmp_path: Path, git_path: str) -> None:
    repo, commit, payload = _make_git_source(tmp_path)
    source = {
        **_common_source(payload),
        "provider": "git_object",
        "git_commit": commit,
        "git_path": git_path,
    }
    with pytest.raises(PinnedSourceMaterializationError, match="unsafe git_path"):
        materialize_pinned_sources(
            _base_config(source),
            repo_root=repo,
            output_dir=tmp_path / "out",
        )


def test_rejects_moving_git_ref_instead_of_exact_commit(tmp_path: Path) -> None:
    repo, _, payload = _make_git_source(tmp_path)
    source = {
        **_common_source(payload),
        "provider": "git_object",
        "git_commit": "main",
        "git_path": "data/sample.txt",
    }
    with pytest.raises(PinnedSourceMaterializationError, match="exact lowercase 40-hex"):
        materialize_pinned_sources(
            _base_config(source),
            repo_root=repo,
            output_dir=tmp_path / "out",
        )


def test_https_payload_is_hash_pinned_and_records_final_locator(tmp_path: Path) -> None:
    payload = b"%PDF-pinned-fixture\n"
    source = {
        **_common_source(payload),
        "source_id": "nist-fixture",
        "source_family": "en.usgov.nist.fixture",
        "stratum": "en",
        "provider": "https_exact",
        "url": "https://example.invalid/pinned.pdf",
        "authority_normalized_bytes": 123,
        "authority_normalized_sha256": "b" * 64,
    }

    def fake_download(url: str, expected_bytes: int) -> tuple[bytes, str]:
        assert expected_bytes == len(payload)
        return payload, url

    manifest = materialize_pinned_sources(
        _base_config(source),
        repo_root=tmp_path,
        output_dir=tmp_path / "out",
        https_downloader=fake_download,
    )
    row = manifest["sources"][0]
    assert row["provider"] == "https_exact"
    assert row["resolved_locator"] == "https://example.invalid/pinned.pdf"
    assert row["normalization_verification_status"] == "NOT_EXECUTED_BY_THIS_TOOL"
    assert row["authority_normalized_bytes"] == 123
    assert row["authority_normalized_sha256"] == "b" * 64


def test_rejects_non_https_final_locator_from_injected_downloader(tmp_path: Path) -> None:
    payload = b"payload"
    source = {
        **_common_source(payload),
        "provider": "https_exact",
        "url": "https://example.invalid/source",
    }

    def bad_redirect(url: str, expected_bytes: int) -> tuple[bytes, str]:
        return payload, "http://example.invalid/source"

    with pytest.raises(PinnedSourceMaterializationError, match="non-HTTPS"):
        materialize_pinned_sources(
            _base_config(source),
            repo_root=tmp_path,
            output_dir=tmp_path / "out",
            https_downloader=bad_redirect,
        )
