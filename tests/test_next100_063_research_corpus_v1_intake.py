from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.validate_next100_063_research_corpus_v1_intake import (
    IntakeValidationError,
    validate_intake,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/data/next100_063_research_corpus_v1_intake.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_frozen_intake_passes() -> None:
    validate_intake(_manifest())


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("authorize_training", "claim firewall"),
        ("inflate_loss_positions", "cannot authorize loss positions"),
        ("drop_record", "record count mismatch"),
        ("duplicate_hash", "duplicate normalized content hash"),
        ("move_source_head", "source authority exact-head vector mismatch"),
        ("rewrite_parent_green", "must not be rewritten as terminal success"),
    ],
)
def test_adversarial_mutations_fail_closed(mutation: str, match: str) -> None:
    document = copy.deepcopy(_manifest())
    if mutation == "authorize_training":
        document["claim_boundary"]["training_authorized"] = True
    elif mutation == "inflate_loss_positions":
        document["claim_boundary"]["authorized_unique_loss_positions"] = 1
    elif mutation == "drop_record":
        document["records"].pop()
    elif mutation == "duplicate_hash":
        document["records"][1]["normalized_sha256"] = document["records"][0][
            "normalized_sha256"
        ]
    elif mutation == "move_source_head":
        document["source_authorities"][0]["head_sha"] = "0" * 40
    elif mutation == "rewrite_parent_green":
        document["parent_corpus_authority"]["terminal_state"] = "TERMINAL_SUCCESS"
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(IntakeValidationError, match=match):
        validate_intake(document)
