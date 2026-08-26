from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.validate_research_corpus_v1_successor_intake import (
    IntakeValidationError,
    validate_payload,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "data" / "research_corpus_v1_successor_intake_v1.json"


def load_payload() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def rebind(payload: dict) -> dict:
    value = copy.deepcopy(payload)
    value.pop("intake_identity_sha256", None)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    payload["intake_identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def test_frozen_intake_validates() -> None:
    validate_payload(load_payload())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda p: p["terminal_additions"][0].update(terminal_verdict="RETEST"), "not terminal"),
        (
            lambda p: p["terminal_additions"][1].update(
                family_id=p["base_registry"]["families"][0]
            ),
            "drift: family_id",
        ),
        (
            lambda p: p["terminal_additions"][3].update(normalized_bytes=59359),
            "drift: normalized_bytes",
        ),
        (
            lambda p: p["gates"].update(training_authorized_loss_positions=1),
            "training exposure",
        ),
        (
            lambda p: p["truth_boundary"].update(corpus_admission_claimed=True),
            "truth boundary",
        ),
    ],
)
def test_adversarial_rebound_mutations_fail_closed(mutation, message: str) -> None:
    payload = load_payload()
    mutation(payload)
    rebind(payload)
    with pytest.raises(IntakeValidationError, match=message):
        validate_payload(payload)


def test_unrebound_mutation_fails_identity_first() -> None:
    payload = load_payload()
    payload["pre_decontamination_projection"]["normalized_bytes"] += 1
    with pytest.raises(IntakeValidationError, match="intake identity mismatch"):
        validate_payload(payload)
