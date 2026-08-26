from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validate_tok239_authority_gate import Tok239GateError, load_and_validate


GATE = Path("evidence/tok239/authority-gate.json")


def _write(tmp_path: Path, report: dict) -> Path:
    path = tmp_path / "gate.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_committed_gate_is_valid_and_fail_closed() -> None:
    report = load_and_validate(GATE)
    assert report["status"] == "BLOCKED_NO_TERMINAL_EXTERNAL_REAL_RESEARCH_CORPUS"
    assert report["numerical_execution_permitted"] is False
    assert report["v1_evidence_permitted"] is False
    assert report["training_started"] is False
    assert report["optimizer_updates"] == 0


def test_tampering_is_rejected_before_scientific_claim(tmp_path: Path) -> None:
    report = json.loads(GATE.read_text(encoding="utf-8"))
    report["numerical_execution_permitted"] = True
    with pytest.raises(Tok239GateError, match="self-hash mismatch"):
        load_and_validate(_write(tmp_path, report))


def test_no_silent_data25_or_data183_v1_fallback() -> None:
    report = load_and_validate(GATE)
    assert report["non_substitutions"] == {
        "data183_as_research_corpus_v1": False,
        "data25_as_research_corpus_v1": False,
        "final_test_as_selection_validation": False,
    }


def test_preregistered_tok187_family_is_preserved() -> None:
    report = load_and_validate(GATE)
    protocol = report["preregistered_protocol"]
    assert protocol["requested_vocab_grid"] == [320, 384, 437, 512]
    assert protocol["independent_tokenizer_trainings_per_candidate"] == 2
    assert protocol["paired_model_seeds"] == [1337, 7331, 18701]
    assert protocol["target_total_model_parameters"] == 467_808
    assert protocol["evaluation_split"] == "selection-validation"
    assert protocol["final_test_exposure_prohibited"] is True
    assert report["incumbent_bpe"]["new_bpe_library_implemented"] is False
