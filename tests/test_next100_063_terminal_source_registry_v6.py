from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tools.validate_next100_063_terminal_source_registry_v6 import (
    BUNDLE_PATH,
    EVIDENCE_PATH,
    EXPECTED_CONTRACT_BLOB,
    EXPECTED_EVIDENCE_BLOB,
    RegistryV6Error,
    V6_PATH,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    return _load(V6_PATH), _load(BUNDLE_PATH), _load(EVIDENCE_PATH)


def _validate(
    v6: dict[str, object],
    bundle: dict[str, object],
    evidence: dict[str, object],
) -> None:
    validate(
        v6,
        bundle,
        evidence,
        bundle_blob_sha1=EXPECTED_CONTRACT_BLOB,
        evidence_blob_sha1=EXPECTED_EVIDENCE_BLOB,
    )


def test_v6_accepts_exact_terminal_bundle_composition() -> None:
    _validate(*_valid())


def test_v6_arithmetic_reduces_planning_gap_without_granting_training() -> None:
    v6, _, _ = _valid()
    inv = v6["derived_pre_successor_global_dedup_inventory"]
    assert inv["candidate_numeric_training_capacity_bytes"] == 6_095_624
    assert inv["target_gap_numeric_training_capacity_bytes"] == 13_904_376
    assert inv["by_stratum"]["code"]["numeric_training_capacity_bytes"] == 4_156_475
    gates = v6["downstream_gate_vector"]
    assert gates["authorized_balanced_no_replay_loss_positions"] == 0
    assert gates["tokenizer_fit"] == "BLOCKED"
    assert gates["long_training"] == "BLOCKED"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("supersedes", "v5_git_blob_sha1"), "0" * 40),
        (("terminal_addition", "workflow_conclusion"), "failure"),
        (("terminal_addition", "numeric_training_capacity_bytes"), 3_880_010),
        (("downstream_gate_vector", "authorized_balanced_no_replay_loss_positions"), 1),
        (("downstream_gate_vector", "tokenizer_fit"), "AUTHORIZED"),
        (("downstream_gate_vector", "paid_compute"), "AUTHORIZED"),
    ],
)
def test_v6_rejects_authority_or_promotion_drift(
    path: tuple[str, str],
    value: object,
) -> None:
    v6, bundle, evidence = _valid()
    mutated = deepcopy(v6)
    mutated[path[0]][path[1]] = value
    with pytest.raises(RegistryV6Error):
        _validate(mutated, bundle, evidence)


def test_v6_rejects_nonterminal_bundle_evidence() -> None:
    v6, bundle, evidence = _valid()
    mutated = deepcopy(evidence)
    mutated["workflow_conclusion"] = "failure"
    with pytest.raises(RegistryV6Error):
        _validate(v6, bundle, mutated)


def test_v6_rejects_family_overlap_with_historical_v5() -> None:
    v6, bundle, evidence = _valid()
    mutated = deepcopy(evidence)
    mutated["families"][0]["family_id"] = "github:numpy/numpy"
    with pytest.raises(RegistryV6Error, match="overlaps V5"):
        _validate(v6, bundle, mutated)


def test_v6_rejects_source_byte_inflation() -> None:
    v6, bundle, evidence = _valid()
    mutated = deepcopy(evidence)
    mutated["families"][0]["eligible_utf8_bytes"] += 1
    with pytest.raises(RegistryV6Error, match="byte arithmetic"):
        _validate(v6, bundle, mutated)


def test_v6_rejects_bundle_self_promotion() -> None:
    v6, bundle, evidence = _valid()
    mutated = deepcopy(evidence)
    mutated["claim_boundary"]["automatic_canonical_capacity_credit"] = True
    with pytest.raises(RegistryV6Error, match="self-promoted"):
        _validate(v6, bundle, mutated)
