from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any

LEDGER_SCHEMA = "12-6.unique-loss-position-ledger.v2"
MATERIALIZATION_SCHEMA = "12-6.postpack-loss-materialization.v2"
EXPOSURE_STATE_SCHEMA = "12-6.unique-loss-exposure-state.v2"
POSITION_POLICY = "logical-causal-token-target-postpack-v2"
REQUIRED_STAGE_BINDINGS = (
    "normalization",
    "evaluation_reservations",
    "dedup",
    "split",
    "packing",
)


class LedgerError(ValueError):
    pass


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

