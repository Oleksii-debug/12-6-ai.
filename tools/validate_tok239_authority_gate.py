#!/usr/bin/env python3
"""Validate the TOK-239 fail-closed authority record.

This validator does not train a tokenizer or a model. It protects the scientific
boundary that forbids TOK-239 numerical V1 evidence until a terminal
external-real research corpus and non-final selection-validation authority exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "12-6.tok239-bpe-research-corpus-v1-gate.v1"
EXPECTED_WORKER = "TOK-239-BPE-RESEARCH-CORPUS-V1"
EXPECTED_STATUS = "BLOCKED_NO_TERMINAL_EXTERNAL_REAL_RESEARCH_CORPUS"
EXPECTED_GRID = [320, 384, 437, 512]
EXPECTED_SEEDS = [1337, 7331, 18701]
EXPECTED_TARGET_PARAMETERS = 467_808
EXPECTED_BUDGET = 16_384
EXPECTED_RUNNER = "src/twelve_six/tok187_bpe_real.py"
EXPECTED_IMPLEMENTATION = "src/twelve_six/tokenization/experiments.py::train_hf_tokenizer"


class Tok239GateError(RuntimeError):
    """Raised when the committed TOK-239 authority gate is weakened or corrupt."""


def canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_and_validate(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise Tok239GateError("authority gate must be a JSON object")

    expected_hash = str(report.get("gate_sha256", ""))
    core = dict(report)
    core.pop("gate_sha256", None)
    if canonical_sha(core) != expected_hash:
        raise Tok239GateError("authority gate self-hash mismatch")

    if report.get("schema") != EXPECTED_SCHEMA:
        raise Tok239GateError("TOK-239 gate schema drift")
    if report.get("worker_id") != EXPECTED_WORKER:
        raise Tok239GateError("TOK-239 worker identity drift")
    if report.get("status") != EXPECTED_STATUS:
        raise Tok239GateError("blocked status was weakened")
    if report.get("local_free_only") is not True:
        raise Tok239GateError("LOCAL_FREE boundary was weakened")
    if report.get("numerical_execution_permitted") is not False:
        raise Tok239GateError("numerical execution cannot be permitted while blocked")
    if report.get("v1_evidence_permitted") is not False:
        raise Tok239GateError("V1 evidence cannot be claimed while blocked")
    if report.get("training_started") is not False or report.get("optimizer_updates") != 0:
        raise Tok239GateError("blocked gate cannot contain model training")

    scan = report.get("authority_scan")
    if not isinstance(scan, dict):
        raise Tok239GateError("authority scan missing")
    if scan.get("milestone_238_terminal_authority_found") is not False:
        raise Tok239GateError("committed blocker must not claim terminal MILESTONE-238")
    if scan.get("data230_terminal_corpus_identity_found") is not False:
        raise Tok239GateError("committed blocker must not claim terminal DATA-230")
    if scan.get("eval233_selection_validation_records") != 0:
        raise Tok239GateError("committed EVAL-233 selection-validation must remain empty")

    bpe = report.get("incumbent_bpe")
    if not isinstance(bpe, dict):
        raise Tok239GateError("incumbent BPE binding missing")
    if bpe.get("runner") != EXPECTED_RUNNER:
        raise Tok239GateError("TOK-187 runner binding drift")
    if bpe.get("implementation") != EXPECTED_IMPLEMENTATION:
        raise Tok239GateError("HF Tokenizers implementation binding drift")
    if bpe.get("new_bpe_library_implemented") is not False:
        raise Tok239GateError("TOK-239 must not introduce another BPE library")

    protocol = report.get("preregistered_protocol")
    if not isinstance(protocol, dict):
        raise Tok239GateError("preregistered protocol missing")
    if protocol.get("requested_vocab_grid") != EXPECTED_GRID:
        raise Tok239GateError("vocabulary grid drift")
    if protocol.get("independent_tokenizer_trainings_per_candidate") != 2:
        raise Tok239GateError("each tokenizer candidate requires two trainings")
    if protocol.get("byte_identical_artifact_identity_required") is not True:
        raise Tok239GateError("byte-identical tokenizer identity requirement weakened")
    if protocol.get("fertility_strata") != ["ua", "en", "code"]:
        raise Tok239GateError("UA/EN/code fertility contract drift")
    if protocol.get("worst_modality_fertility_required") is not True:
        raise Tok239GateError("worst-modality fertility requirement weakened")
    if protocol.get("target_total_model_parameters") != EXPECTED_TARGET_PARAMETERS:
        raise Tok239GateError("~500K matched-capacity target drift")
    if protocol.get("optimized_token_budget") != EXPECTED_BUDGET:
        raise Tok239GateError("bounded probe optimized-token budget drift")
    if protocol.get("paired_model_seeds") != EXPECTED_SEEDS:
        raise Tok239GateError("paired model-seed set drift")
    if protocol.get("evaluation_split") != "selection-validation":
        raise Tok239GateError("TOK-239 may evaluate promotion only on selection-validation")
    if protocol.get("final_test_exposure_prohibited") is not True:
        raise Tok239GateError("final-test isolation weakened")
    if protocol.get("primary_rank_metric") != "paired held-out selection-validation aggregate BPB":
        raise Tok239GateError("primary ranking metric drift")
    if protocol.get("promotion_requires_multiple_paired_model_seeds") is not True:
        raise Tok239GateError("promotion seed requirement weakened")

    non_sub = report.get("non_substitutions")
    if non_sub != {
        "data183_as_research_corpus_v1": False,
        "data25_as_research_corpus_v1": False,
        "final_test_as_selection_validation": False,
    }:
        raise Tok239GateError("non-substitution boundary drift")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=Path("evidence/tok239/authority-gate.json"),
    )
    args = parser.parse_args(argv)
    report = load_and_validate(args.path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "gate_sha256": report["gate_sha256"],
                "numerical_execution_permitted": report["numerical_execution_permitted"],
                "v1_evidence_permitted": report["v1_evidence_permitted"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
