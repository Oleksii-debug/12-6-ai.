from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.validate_next100_104_source_capacity_hardening import (
    CapacityHardeningError,
    validate,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "configs/data/next100_104_source_capacity_hardening_v1.json"


def _load() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_capacity_hardening_authority_passes() -> None:
    validate(_load())


def test_cannot_restore_parent_exact_total() -> None:
    document = copy.deepcopy(_load())
    document["corrected_fail_closed_accounting"][
        "candidate_exact_training_eligible_bytes"
    ] = 565_743
    with pytest.raises(CapacityHardeningError, match="must remain null"):
        validate(document)


def test_cannot_authorize_loss_positions() -> None:
    document = copy.deepcopy(_load())
    document["corrected_fail_closed_accounting"][
        "authorized_balanced_no_replay_loss_positions"
    ] = 1
    with pytest.raises(CapacityHardeningError, match="cannot authorize loss positions"):
        validate(document)


def test_cpython_rejected_chunks_cannot_disappear() -> None:
    document = copy.deepcopy(_load())
    document["finding"]["rejected_chunk_count"] = 0
    with pytest.raises(CapacityHardeningError, match="rejected chunk count drift"):
        validate(document)
