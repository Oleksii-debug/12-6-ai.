from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.filter_d03_rada_bulk_quality_privacy import (
    QualityPrivacyError,
    _chunk_text,
    _validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (ROOT / "configs/data/d03_rada_bulk_quality_privacy_v1.json").read_text(
        encoding="utf-8"
    )
)


def test_short_chunking_tail_is_accounted_without_emitting_it() -> None:
    retained = "А" * 1190
    dropped = "Б" * 20

    chunks, stats = _chunk_text(
        retained + "\n" + dropped,
        max_chars=1200,
        min_chars=80,
    )

    assert chunks == (retained,)
    assert dropped not in chunks
    assert stats == {
        "short_fragment_count": 1,
        "short_fragment_chars": len(dropped),
        "short_fragment_bytes": len(dropped.encode("utf-8")),
    }


def test_chunk_accounting_contract_cannot_be_disabled() -> None:
    weakened = copy.deepcopy(CONFIG)
    weakened["output_contract"]["chunking_short_fragments_accounted"] = False

    with pytest.raises(QualityPrivacyError, match="short-fragment accounting disabled"):
        _validate_config(weakened)


def test_chunking_drop_privacy_boundary_cannot_emit_text_or_hashes() -> None:
    for field in ("chunking_dropped_text_emitted", "chunking_dropped_hashes_emitted"):
        weakened = copy.deepcopy(CONFIG)
        weakened["output_contract"][field] = True
        with pytest.raises(QualityPrivacyError, match="emission enabled"):
            _validate_config(weakened)
