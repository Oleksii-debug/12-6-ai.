from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/download_d03_rada_trees_from_hf_snapshot.py"
SPEC = importlib.util.spec_from_file_location("rada_exact_acquisition", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def handoff() -> dict[str, object]:
    return {
        "snapshot_identity_sha256": "1" * 64,
        "size_bytes": 6,
        "git_blob_oid": "2" * 40,
        "xet_hash": "3" * 64,
    }


def test_exact_resolve_url_is_immutable_revision_bound() -> None:
    assert module.RESOLVE_URL == (
        "https://huggingface.co/datasets/uacorpus/Rada_Trees/resolve/"
        "1b994a5804dcda122721e8d33a03fd172cf8d867/Rada_Trees.7z"
    )


def test_provider_url_policy_is_https_and_origin_scoped() -> None:
    assert module._provider_https_url("https://cas-bridge.xethub.hf.co/object")
    assert module._provider_https_url("https://cdn-lfs-us-1.hf.co/object")
    assert module._provider_https_url("https://huggingface.co/object")
    assert not module._provider_https_url("http://huggingface.co/object")
    assert not module._provider_https_url("https://example.com/object")
    assert not module._provider_https_url("https://user@huggingface.co/object")


def test_matching_first_redirect_binds_xet_commit_and_size() -> None:
    evidence = module.validate_first_redirect(
        handoff(),
        module.RESOLVE_URL,
        302,
        {
            "X-Xet-Hash": "3" * 64,
            "X-Repo-Commit": module.bridge.REVISION,
            "X-Linked-Size": "6",
        },
        "https://cas-bridge.xethub.hf.co/object",
    )
    assert evidence["status"] == 302
    assert evidence["xet_hash"] == "3" * 64
    assert evidence["linked_size_bytes"] == 6


@pytest.mark.parametrize(
    ("headers", "target", "code", "message"),
    [
        ({"X-Xet-Hash": "4" * 64}, "https://cas-bridge.xethub.hf.co/x", 302, "X-Xet-Hash"),
        (
            {"X-Xet-Hash": "3" * 64, "X-Repo-Commit": "0" * 40},
            "https://cas-bridge.xethub.hf.co/x",
            302,
            "X-Repo-Commit",
        ),
        (
            {"X-Xet-Hash": "3" * 64, "X-Linked-Size": "7"},
            "https://cas-bridge.xethub.hf.co/x",
            302,
            "X-Linked-Size",
        ),
        ({"X-Xet-Hash": "3" * 64}, "https://example.com/x", 302, "approved"),
        ({"X-Xet-Hash": "3" * 64}, "https://cas-bridge.xethub.hf.co/x", 307, "302"),
    ],
)
def test_redirect_drift_fails_closed(
    headers: dict[str, str],
    target: str,
    code: int,
    message: str,
) -> None:
    with pytest.raises(module.AcquisitionError, match=message):
        module.validate_first_redirect(
            handoff(),
            module.RESOLVE_URL,
            code,
            headers,
            target,
        )


def test_redirect_source_url_drift_fails_closed() -> None:
    with pytest.raises(module.AcquisitionError, match="source URL drift"):
        module.validate_first_redirect(
            handoff(),
            module.RESOLVE_URL + "?download=true",
            302,
            {"X-Xet-Hash": "3" * 64},
            "https://cas-bridge.xethub.hf.co/x",
        )


def test_stream_exact_bytes_hashes_complete_payload() -> None:
    source = io.BytesIO(b"abcdef")
    destination = io.BytesIO()
    size, digest = module._stream_exact_bytes(source, destination, 6)
    assert size == 6
    assert destination.getvalue() == b"abcdef"
    assert digest == hashlib.sha256(b"abcdef").hexdigest()


def test_stream_short_payload_fails_closed() -> None:
    with pytest.raises(module.AcquisitionError, match="byte-size mismatch"):
        module._stream_exact_bytes(io.BytesIO(b"abc"), io.BytesIO(), 6)


def test_stream_oversized_payload_fails_closed() -> None:
    with pytest.raises(module.AcquisitionError, match="exceeded"):
        module._stream_exact_bytes(io.BytesIO(b"abcdefg"), io.BytesIO(), 6)
