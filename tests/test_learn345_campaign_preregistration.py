from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.validate_learn345_campaign_preregistration import (
    canonical_without_identity,
    validate,
)


EVIDENCE = Path("evidence/learn345/20m_campaign_preregistration_v1.json")


def _write(tmp_path: Path, data: dict) -> Path:
    data = copy.deepcopy(data)
    data["evidence_identity_sha256"] = hashlib.sha256(
        canonical_without_identity(data)
    ).hexdigest()
    p = tmp_path / "evidence.json"
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return p


def test_committed_evidence_validates() -> None:
    validate(EVIDENCE)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("long_campaign_executed",), True),
        (("optimizer_updates_executed",), 1),
        (("campaign", "replay_allowed"), True),
        (("campaign", "replacement_sampling_allowed"), True),
        (("campaign", "padding_counts_as_data"), True),
        (("selection_rule", "final_test_may_influence_selection"), True),
        (("final_test_firewall", "final_test_outcomes_read_before_selection_lock"), True),
        (("truth_boundary", "campaign_runnable_now"), True),
    ],
)
def test_fail_closed_invariants(tmp_path: Path, path: tuple[str, ...], value) -> None:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    p = _write(tmp_path, data)
    with pytest.raises(SystemExit):
        validate(p)


def test_cannot_invent_missing_model_identity(tmp_path: Path) -> None:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    data["observed_authorities"]["primary_20m_architecture"][
        "modelspec_identity_sha256"
    ] = "0" * 64
    p = _write(tmp_path, data)
    with pytest.raises(SystemExit):
        validate(p)
