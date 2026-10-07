from __future__ import annotations

import gzip
import json
from copy import deepcopy
from pathlib import Path

import pytest

import twelve_six.data.common_pile_loc_intake as loc_intake
from twelve_six.data.common_pile_loc_intake import (
    LocIntakeError,
    iter_gzip_jsonl_bytes,
    materialize,
    materialize_verified_shard,
    self_identity,
    validate_config,
    verify_full_shard,
)

CONFIG_PATH = "configs/data/d03_common_pile_loc_intake_v1.json"
TOOL_PATH = Path("tools/materialize_d03_common_pile_loc.py")


def load_config() -> dict:
    return json.loads(open(CONFIG_PATH, encoding="utf-8").read())


def reseal(config: dict) -> dict:
    config = deepcopy(config)
    config["contract_identity_sha256"] = self_identity(config, "contract_identity_sha256")
    return config


def valid_record(record_id: str = "abc123") -> dict:
    paragraph = (
        "This public volume records historical institutions, communities, and events. "
        "The scanned pages preserve ordinary prose for research and reading. "
    )
    return {
        "id": record_id,
        "text": paragraph * 18,
        "source": "loc_books",
        "added": "2024-05-13T00:00:00",
        "metadata": {
            "license": "Public Domain",
            "title": "Metadata only",
            "author": "Metadata only",
            "year": 1901,
            "language": "english",
            "item_url": f"https://www.loc.gov/item/{record_id}",
            "text_file_url": (
                f"https://tile.loc.gov/storage-services/{record_id}_djvu.txt"
            ),
        },
    }


def tiny_full_shard_config(
    monkeypatch: pytest.MonkeyPatch,
    raw_gzip: bytes,
) -> dict:
    config = load_config()
    digest = loc_intake.sha256_bytes(raw_gzip)
    expected_upstream = deepcopy(loc_intake._EXPECTED_UPSTREAM)
    expected_upstream["shard_compressed_bytes"] = len(raw_gzip)
    expected_upstream["shard_lfs_sha256"] = digest
    monkeypatch.setattr(loc_intake, "_EXPECTED_UPSTREAM", expected_upstream)
    config["upstream"]["shard_compressed_bytes"] = len(raw_gzip)
    config["upstream"]["shard_lfs_sha256"] = digest
    return reseal(config)


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
    assert report["full_shard_hash_verified"] is False


def test_old_module_receipt_capability_is_structurally_gone() -> None:
    assert not hasattr(loc_intake, "_FullShardVerificationReceipt")
    assert not hasattr(loc_intake, "_FULL_SHARD_RECEIPT_SEAL")

    with pytest.raises(TypeError, match="unexpected keyword"):
        materialize(
            load_config(),
            [valid_record()],
            full_shard_verification=True,  # type: ignore[call-arg]
        )


def test_verified_shard_positive_fact_is_derived_from_same_exact_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (json.dumps(valid_record()) + "\n").encode("utf-8")
    raw_gzip = gzip.compress(payload, mtime=0)
    config = tiny_full_shard_config(monkeypatch, raw_gzip)

    candidates, report = materialize_verified_shard(config, raw_gzip)
    assert [row["source_record_id"] for row in candidates] == ["abc123"]
    assert report["full_shard_hash_verified"] is True

    unrelated_candidates, unrelated_report = materialize(
        config,
        [valid_record("unrelated")],
    )
    assert [row["source_record_id"] for row in unrelated_candidates] == ["unrelated"]
    assert unrelated_report["full_shard_hash_verified"] is False

    with pytest.raises(TypeError, match="unexpected keyword"):
        materialize_verified_shard(
            config,
            raw_gzip,
            records=[valid_record("unrelated")],  # type: ignore[call-arg]
        )


def test_verified_shard_rejects_same_length_byte_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (json.dumps(valid_record()) + "\n").encode("utf-8")
    raw_gzip = gzip.compress(payload, mtime=0)
    config = tiny_full_shard_config(monkeypatch, raw_gzip)
    mutated = raw_gzip[:-1] + bytes([raw_gzip[-1] ^ 1])

    with pytest.raises(LocIntakeError, match="SHA-256 mismatch"):
        materialize_verified_shard(config, mutated)


def test_verify_full_shard_reads_once_and_returns_verified_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (json.dumps(valid_record()) + "\n").encode("utf-8")
    raw_gzip = gzip.compress(payload, mtime=0)
    config = tiny_full_shard_config(monkeypatch, raw_gzip)

    class FakePath:
        reads = 0

        def read_bytes(self) -> bytes:
            self.reads += 1
            return raw_gzip

    path = FakePath()
    snapshot = verify_full_shard(path, config)  # type: ignore[arg-type]
    assert snapshot == raw_gzip
    assert path.reads == 1


def test_canonical_cli_has_one_shard_read_and_no_receipt_handoff() -> None:
    source = TOOL_PATH.read_text(encoding="utf-8")
    assert source.count(".read_bytes()") == 1
    assert "verify_full_shard" not in source
    assert "full_shard_verification" not in source
    assert "materialize_verified_shard(config, raw)" in source


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
