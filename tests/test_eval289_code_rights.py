from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from twelve_six.eval289_code_rights import (
    BLOCKER,
    Eval289Error,
    assess_candidate,
    validate_authority,
)

AUTHORITY = Path("evidence/eval289/code-evaluation-rights-reservation.json")


def _authority() -> dict:
    return json.loads(AUTHORITY.read_text(encoding="utf-8"))


def test_terminal_wave1_objects_fail_closed() -> None:
    value = _authority()
    validate_authority(value)
    assert value["status"] == BLOCKER
    assert value["observed_source_family_count"] == 2
    assert value["eligible_object_count"] == 0
    assert value["reservation"]["active"] is False
    assert value["reservation"]["objects"] == []


def test_training_permission_does_not_imply_evaluation_permission() -> None:
    candidate = deepcopy(_authority()["candidates"][0])
    candidate["training_exposure"] = {"exposed": False}
    assert assess_candidate(candidate) == ["NO_EXPLICIT_EVALUATION_USE_AUTHORITY"]


def test_evaluation_permission_does_not_erase_prior_training_exposure() -> None:
    candidate = deepcopy(_authority()["candidates"][0])
    candidate["evaluation_use_explicitly_authorized"] = True
    assert assess_candidate(candidate) == ["ALREADY_EXPOSED_TO_MODEL_TRAINING"]


def test_tampered_reservation_fails() -> None:
    value = _authority()
    value["reservation"]["active"] = True
    with pytest.raises(Eval289Error):
        validate_authority(value)
