from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.validate_tok315_tokenizer_fit_eligibility import (
    Tok315EligibilityError,
    canonical_sha,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/tok315/tokenizer-fit-eligibility-v1.json"
CONTRACT = ROOT / "configs/data/data300_corpus_v03_frozen_build_contract_v2.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(evidence: dict) -> None:
    core = dict(evidence)
    core.pop("evidence_sha256", None)
    evidence["evidence_sha256"] = canonical_sha(core)


def test_committed_tok315_evidence_validates() -> None:
    validate(_load(EVIDENCE), _load(CONTRACT))


def test_selection_validation_cannot_become_tokenizer_fit_input() -> None:
    contract = _load(CONTRACT)
    contract["split_contract"]["selection_validation"]["may_fit_tokenizer"] = True
    with pytest.raises(Tok315EligibilityError, match="selection-validation"):
        validate(_load(EVIDENCE), contract)


def test_final_test_cannot_become_tokenizer_fit_input() -> None:
    contract = _load(CONTRACT)
    contract["split_contract"]["final_test"]["may_fit_tokenizer"] = True
    with pytest.raises(Tok315EligibilityError, match="final-test"):
        validate(_load(EVIDENCE), contract)


def test_rehashed_unlisted_source_still_fails_exact_inventory_binding() -> None:
    evidence = copy.deepcopy(_load(EVIDENCE))
    evidence["tokenizer_training_inventory"]["sources"].append(
        {
            "source_id": "selection-validation:forbidden",
            "family": "forbidden",
            "language": "en",
            "modality": "text",
            "normalized_bytes": 1,
            "raw_sha256": "0" * 64,
        }
    )
    evidence["tokenizer_training_inventory"]["source_count"] = 6
    _rehash(evidence)
    with pytest.raises(Tok315EligibilityError, match="source count|allowlist"):
        validate(evidence, _load(CONTRACT))


def test_tok315_cannot_overclaim_reserved_byte_overlap_proof() -> None:
    evidence = copy.deepcopy(_load(EVIDENCE))
    evidence["proof"]["reserved_byte_overlap"] = "PASS"
    _rehash(evidence)
    with pytest.raises(Tok315EligibilityError, match="overclaimed"):
        validate(evidence, _load(CONTRACT))


def test_future_bpe_cannot_start_before_g08() -> None:
    evidence = copy.deepcopy(_load(EVIDENCE))
    evidence["future_bpe"]["fit_may_start_now"] = True
    evidence["proof"]["bpe_fit_execution_permitted"] = True
    _rehash(evidence)
    with pytest.raises(Tok315EligibilityError, match="BPE fit"):
        validate(evidence, _load(CONTRACT))


def test_tok315_cannot_choose_tokenizer_winner() -> None:
    evidence = copy.deepcopy(_load(EVIDENCE))
    evidence["tokenizer_winner"] = "BPE"
    evidence["future_bpe"]["winner"] = True
    _rehash(evidence)
    with pytest.raises(Tok315EligibilityError, match="winner"):
        validate(evidence, _load(CONTRACT))
