from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest

from twelve_six.data.eval647_code_selection_reserve_v1 import (
    Eval647ReserveError,
    git_blob_sha1,
    load_reservation,
    materialize,
    sha256_bytes,
    validate_reservation,
)

ROOT = Path(__file__).resolve().parents[1]


def test_reservation_is_exact_closed_world_and_nontraining():
    reservation = load_reservation(ROOT)
    validate_reservation(reservation, require_sealed=False)
    truth = reservation["truth_boundary"]
    assert truth == {
        "final_test_eligible": False,
        "future_training_prohibited": True,
        "model_training_eligible": False,
        "tokenizer_fit_eligible": False,
    }
    assert {obj["source_family"] for obj in reservation["objects"]} == {
        "github:jd/tenacity",
        "github:more-itertools/more-itertools",
    }


def test_unsealed_reservation_cannot_be_terminally_materialized(tmp_path: Path):
    with pytest.raises(Eval647ReserveError, match="raw SHA-256 is not sealed"):
        materialize(ROOT, tmp_path, require_sealed=True, fetch=lambda _: b"never-used")


def test_wrong_source_bytes_fail_before_any_credit(tmp_path: Path):
    with pytest.raises(Eval647ReserveError, match="byte-count mismatch"):
        materialize(ROOT, tmp_path, require_sealed=False, fetch=lambda _: b"wrong")


def test_git_blob_hash_is_content_addressed():
    raw = b"hello\n"
    assert git_blob_sha1(raw) != git_blob_sha1(raw + b"x")
    assert sha256_bytes(raw) != sha256_bytes(raw + b"x")


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") != "true", reason="needs GitHub network")
def test_live_pinned_sources_materialize_twice_identically(tmp_path: Path):
    first = materialize(ROOT, tmp_path / "first", require_sealed=False)
    second = materialize(ROOT, tmp_path / "second", require_sealed=False)

    assert first == second
    for relative in ("membership.jsonl", "materialization-evidence.json"):
        assert (tmp_path / "first" / relative).read_bytes() == (
            tmp_path / "second" / relative
        ).read_bytes()
    first_payload = sorted((tmp_path / "first" / "payload").iterdir())
    second_payload = sorted((tmp_path / "second" / "payload").iterdir())
    assert [path.name for path in first_payload] == [path.name for path in second_payload]
    assert [path.read_bytes() for path in first_payload] == [
        path.read_bytes() for path in second_payload
    ]

    discovered = {
        item["record_id"]: {
            "raw_bytes": item["raw_bytes"],
            "raw_sha256": item["raw_sha256"],
            "git_blob_sha1": item["git_blob_sha1"],
            "license_raw_sha256": item["license"]["raw_sha256"],
            "license_git_blob_sha1": item["license"]["git_blob_sha1"],
        }
        for item in first["objects"]
    }
    warnings.warn(
        "EVAL647_LIVE_SOURCE_DISCOVERY=" + json.dumps(discovered, sort_keys=True),
        stacklevel=1,
    )
