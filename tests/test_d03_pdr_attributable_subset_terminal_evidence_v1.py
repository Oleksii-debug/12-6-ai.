from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EVIDENCE = ROOT / "evidence" / "d03_pdr_attributable_subset_terminal_execution_v1.json"

SPEC = importlib.util.spec_from_file_location(
    "pdr_subset_authority_terminal",
    TOOLS / "derive_d03_pdr_attributable_subset_authority_v1.py",
)
assert SPEC and SPEC.loader
pdr = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pdr
SPEC.loader.exec_module(pdr)


def _mutated(mutator) -> bytes:
    evidence = pdr.validate_terminal_evidence(EVIDENCE.read_bytes())
    mutator(evidence)
    return json.dumps(evidence, ensure_ascii=False).encode("utf-8")


def test_terminal_pdr_subset_evidence_binds_physical_replay_without_credit() -> None:
    evidence = pdr.validate_terminal_evidence(EVIDENCE.read_bytes())

    assert evidence["execution"]["run_id"] == 36521055931
    assert evidence["attributable_subset"]["attributable_record_count"] == 498
    assert evidence["attributable_subset"]["excluded_record_count"] == 668
    assert evidence["attributable_subset"]["attributable_normalized_utf8_bytes"] == 3_727_864
    assert evidence["truth_boundary"]["source_rights_review_status"] == "REVIEW_REQUIRED"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.__setitem__("unexpected_root", "forbidden"),
        lambda value: value["execution"].__setitem__("unexpected_nested", "forbidden"),
        lambda value: value["truth_boundary"].__setitem__("training_authorized_bytes", False),
        lambda value: value["execution"].__setitem__("run_id", True),
    ],
)
def test_terminal_evidence_unknown_members_and_bool_int_aliases_fail_closed(mutator) -> None:
    with pytest.raises(pdr.SubsetAccountingError):
        pdr.validate_terminal_evidence(_mutated(mutator))


def test_terminal_evidence_duplicate_member_fails_closed() -> None:
    raw = EVIDENCE.read_bytes()
    duplicate = raw.replace(
        b'"schema_version": "12-6.d03-pdr-attributable-subset-terminal-evidence.v1",',
        b'"schema_version": "12-6.d03-pdr-attributable-subset-terminal-evidence.v1",\n'
        b'  "schema_version": "12-6.d03-pdr-attributable-subset-terminal-evidence.v1",',
        1,
    )
    with pytest.raises(pdr.SubsetAccountingError, match="duplicate JSON member"):
        pdr.validate_terminal_evidence(duplicate)


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_terminal_evidence_nonfinite_numbers_fail_closed(number: str) -> None:
    raw = EVIDENCE.read_text(encoding="utf-8")
    mutated = raw.replace('"run_attempt": 1', f'"run_attempt": {number}', 1).encode("utf-8")
    with pytest.raises(pdr.SubsetAccountingError, match="non-finite JSON number"):
        pdr.validate_terminal_evidence(mutated)
