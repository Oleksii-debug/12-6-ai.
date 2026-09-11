from __future__ import annotations

import gzip
import io
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

import twelve_six.data.common_pile_loc_intake as loc_intake
from twelve_six.data.common_pile_loc_intake import (
    LocIntakeError,
    iter_gzip_jsonl_bytes,
    materialize,
    self_identity,
    validate_config,
)

CONFIG_PATH = "configs/data/d03_common_pile_loc_intake_v1.json"


def load_config() -> dict:
    return json.loads(open(CONFIG_PATH, encoding="utf-8").read())


def reseal(config: dict) -> dict:
    config = deepcopy(config)
    config["contract_identity_sha256"] = self_identity(config, "contract_identity_sha256")
    return config


def valid_record() -> dict:
    paragraph = (
        "This public volume records historical institutions, communities, and events. "
        "The scanned pages preserve ordinary prose for research and reading. "
    )
    return {
        "id": "abc123",
        "text": paragraph * 18,
        "source": "loc_books",
        "added": "2024-05-13T00:00:00",
        "metadata": {
            "license": "Public Domain",
            "title": "Metadata only",
            "author": "Metadata only",
            "year": 1901,
            "language": "english",
            "item_url": "https://www.loc.gov/item/abc123",
            "text_file_url": "https://tile.loc.gov/storage-services/abc123_djvu.txt",
        },
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("training_authorized_bytes", False),
        ("authorized_unique_loss_positions", False),
        ("canonical_capacity_credit_bytes", False),
        ("corpus_admitted", 0),
        ("evaluation_eligible", 0),
        ("tokenizer_fit_permitted", 0),
        ("model_training_permitted", 0),
        ("paid_compute_authorized", 0),
        ("final_test_payload_accessed", 0),
    ],
)
def test_zero_credit_and_final_test_types_are_exact(field: str, bad_value: object) -> None:
    config = load_config()
    config[field] = bad_value
    with pytest.raises(LocIntakeError):
        validate_config(reseal(config))


def test_final_test_boundary_is_required_false_and_closed_world() -> None:
    config = load_config()
    assert config["final_test_payload_accessed"] is False

    widened = load_config()
    widened["final_test_payload_accessed"] = True
    with pytest.raises(LocIntakeError, match="final_test_payload_accessed drift"):
        validate_config(reseal(widened))

    missing = load_config()
    del missing["final_test_payload_accessed"]
    with pytest.raises(LocIntakeError, match="config schema drift"):
        validate_config(reseal(missing))

    extra = load_config()
    extra["final_test_outcomes_read"] = False
    with pytest.raises(LocIntakeError, match="config schema drift"):
        validate_config(reseal(extra))


def test_authority_subobjects_are_closed_world() -> None:
    for key in ("common_pile_audit", "upstream", "rights_policy", "selection_policy"):
        config = load_config()
        config[key]["unexpected_authority"] = False
        with pytest.raises(LocIntakeError, match="schema drift"):
            validate_config(reseal(config))


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("max_total_normalized_utf8_bytes", 5_000_000),
        ("max_remote_compressed_prefix_bytes", 67_108_864),
        ("max_jsonl_line_bytes", 8_388_608),
        ("max_documents", 129),
        ("max_examined_documents", 257),
        ("max_single_normalized_utf8_bytes", 1_100_000),
        ("min_single_normalized_utf8_bytes", 1_023),
    ],
)
def test_selection_policy_drift_is_rejected_even_when_resealed(
    field: str,
    bad_value: int,
) -> None:
    config = load_config()
    config["selection_policy"][field] = bad_value
    with pytest.raises(LocIntakeError, match=rf"selection_policy {field} drift"):
        validate_config(reseal(config))


def test_materialization_report_carries_final_test_false() -> None:
    candidates, report = materialize(load_config(), [valid_record()])
    assert len(candidates) == 1
    assert report["final_test_payload_accessed"] is False


def test_direct_caller_cannot_forge_full_shard_verification() -> None:
    config = load_config()
    forged_receipt = {
        "source_revision": config["upstream"]["revision"],
        "source_shard_path": config["upstream"]["shard_path"],
        "observed_bytes": config["upstream"]["shard_compressed_bytes"],
        "observed_sha256": config["upstream"]["shard_lfs_sha256"],
    }
    with pytest.raises(LocIntakeError, match="verifier-produced receipt"):
        materialize(
            config,
            [valid_record()],
            full_shard_verification=forged_receipt,
        )
    with pytest.raises(LocIntakeError, match="verifier-produced receipt"):
        materialize(
            config,
            [valid_record()],
            full_shard_verification=True,
        )


def test_successful_verifier_receipt_is_bound_to_exact_shard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    expected_sha = config["upstream"]["shard_lfs_sha256"]
    real_sha256 = loc_intake.hashlib.sha256
    missing = object()

    class VerifiedDigest:
        def update(self, data: bytes) -> None:
            assert isinstance(data, bytes)

        def hexdigest(self) -> str:
            return expected_sha

    def controlled_sha256(data: object = missing) -> object:
        if data is missing:
            return VerifiedDigest()
        assert isinstance(data, bytes)
        return real_sha256(data)

    class FakePath:
        def stat(self) -> SimpleNamespace:
            return SimpleNamespace(st_size=config["upstream"]["shard_compressed_bytes"])

        def open(self, mode: str) -> io.BytesIO:
            assert mode == "rb"
            return io.BytesIO(b"independently observed shard bytes")

    monkeypatch.setattr(loc_intake.hashlib, "sha256", controlled_sha256)
    receipt = loc_intake.verify_full_shard(FakePath(), config)
    candidates, report = materialize(
        config,
        [valid_record()],
        full_shard_verification=receipt,
    )
    assert len(candidates) == 1
    assert report["full_shard_hash_verified"] is True


def test_canonical_parser_skips_oversize_row_without_truncation() -> None:
    oversized = b'{"text":"' + b"a" * 200 + b'"}\n'
    good = b'{"id":"ok"}\n'
    raw = gzip.compress(oversized + good, mtime=0)

    parsed = list(
        iter_gzip_jsonl_bytes(
            raw,
            max_jsonl_line_bytes=50,
            skip_oversize_lines=True,
        )
    )
    assert parsed == [{"id": "ok"}]

    with pytest.raises(LocIntakeError, match="line exceeds"):
        list(iter_gzip_jsonl_bytes(raw, max_jsonl_line_bytes=50))


def test_oversize_skip_mode_is_exact_bool_and_bounded_at_eof() -> None:
    raw = gzip.compress(b'{"text":"' + b"a" * 200, mtime=0)
    assert list(
        iter_gzip_jsonl_bytes(
            raw,
            max_jsonl_line_bytes=50,
            skip_oversize_lines=True,
        )
    ) == []

    with pytest.raises(LocIntakeError, match="exact bool"):
        list(
            iter_gzip_jsonl_bytes(
                raw,
                max_jsonl_line_bytes=50,
                skip_oversize_lines=1,
            )
        )
