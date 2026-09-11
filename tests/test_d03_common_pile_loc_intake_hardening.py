from __future__ import annotations

import gzip
import json
from copy import deepcopy

import pytest

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


def test_materialization_report_carries_final_test_false() -> None:
    config = load_config()
    paragraph = (
        "This public volume records historical institutions, communities, and events. "
        "The scanned pages preserve ordinary prose for research and reading. "
    )
    record = {
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
    candidates, report = materialize(config, [record])
    assert len(candidates) == 1
    assert report["final_test_payload_accessed"] is False


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
