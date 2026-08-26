#!/usr/bin/env python3
"""Validate the fail-closed TOK-316 BPE reproducibility authority gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "12-6.tok316-bpe-reproducibility-v03-gate.v1"
EXPECTED_WORKER = "TOK-316-BPE-REPRODUCIBILITY-V03"
EXPECTED_STATUS = "BLOCKED_NO_TERMINAL_TOKENIZER_FIT_CORPUS"
EXPECTED_BASE_SHA = "8ea7f830e50a23754d189dd4134f4afad76a7ee9"
EXPECTED_CONTRACT_ID = "07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5"
EXPECTED_GRID = [320, 384, 437, 512]
EXPECTED_LIBRARY_VERSION = "0.23.1"


class Tok316GateError(RuntimeError):
    """Authority evidence is missing, drifted, or semantically weakened."""


def _canonical_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_gate(value: dict[str, Any]) -> None:
    if value.get("schema") != EXPECTED_SCHEMA:
        raise Tok316GateError("schema drift")
    if value.get("worker_id") != EXPECTED_WORKER:
        raise Tok316GateError("worker id drift")
    if value.get("status") != EXPECTED_STATUS:
        raise Tok316GateError("blocked status weakened")
    if value.get("local_free_only") is not True:
        raise Tok316GateError("LOCAL_FREE boundary weakened")

    recorded_hash = value.get("gate_sha256")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise Tok316GateError("missing gate SHA-256")
    core = dict(value)
    core.pop("gate_sha256", None)
    if _canonical_sha(core) != recorded_hash:
        raise Tok316GateError("gate self-hash mismatch")

    base = value.get("base")
    if not isinstance(base, dict):
        raise Tok316GateError("base binding missing")
    if base.get("head_sha") != EXPECTED_BASE_SHA:
        raise Tok316GateError("DATA-300 exact head drift")
    if base.get("contract_identity_sha256") != EXPECTED_CONTRACT_ID:
        raise Tok316GateError("DATA-300 contract identity drift")
    if base.get("corpus_state") != "NOT_BUILT_NOT_FROZEN_NOT_TERMINAL":
        raise Tok316GateError("corpus truth boundary weakened")

    prereq = value.get("prerequisite_scan")
    if not isinstance(prereq, dict):
        raise Tok316GateError("prerequisite scan missing")
    if prereq.get("data301_branch_found") is not True:
        raise Tok316GateError("DATA-301 observed-state drift")
    if prereq.get("data301_head_sha") != EXPECTED_BASE_SHA:
        raise Tok316GateError("DATA-301 cutoff head drift")
    if prereq.get("data301_commits_ahead_of_data300") != 0:
        raise Tok316GateError("evidence cutoff no longer describes zero-ahead DATA-301")
    if prereq.get("data301_terminal_corpus_identity_found") is not False:
        raise Tok316GateError("terminal DATA-301 identity may not be invented")
    if prereq.get("tok315_branch_found") is not False:
        raise Tok316GateError("TOK-315 branch state may not be invented")
    if prereq.get("tok315_tokenizer_fit_inventory_identity_found") is not False:
        raise Tok316GateError("TOK-315 inventory identity may not be invented")
    if prereq.get("eligible_tokenizer_fit_manifest_found") is not False:
        raise Tok316GateError("eligible tokenizer-fit manifest may not be invented")

    maintained = value.get("maintained_bpe")
    if not isinstance(maintained, dict):
        raise Tok316GateError("maintained BPE binding missing")
    if maintained.get("implementation") != (
        "src/twelve_six/tokenization/experiments.py::train_hf_tokenizer"
    ):
        raise Tok316GateError("maintained HF BPE implementation drift")
    if maintained.get("library") != "tokenizers":
        raise Tok316GateError("maintained library drift")
    if maintained.get("library_version") != EXPECTED_LIBRARY_VERSION:
        raise Tok316GateError("tokenizers version drift")
    if maintained.get("algorithm") != "bpe":
        raise Tok316GateError("algorithm drift")
    if maintained.get("runtime_substitution_allowed") is not False:
        raise Tok316GateError("runtime substitution may not be enabled")
    if maintained.get("self_written_bpe_substitution_allowed") is not False:
        raise Tok316GateError("self-written BPE substitution may not be enabled")

    protocol = value.get("protocol")
    if not isinstance(protocol, dict):
        raise Tok316GateError("protocol missing")
    if protocol.get("requested_vocab_grid") != EXPECTED_GRID:
        raise Tok316GateError("vocabulary grid drift")
    if protocol.get("independent_trainings_per_candidate") != 2:
        raise Tok316GateError("repeated-training count drift")
    if protocol.get("strict_roundtrip_required") is not True:
        raise Tok316GateError("strict roundtrip gate weakened")
    if protocol.get("unintended_unknown_tokens_required") != 0:
        raise Tok316GateError("unknown-token gate weakened")
    if protocol.get("fertility_strata") != ["ua", "en", "code"]:
        raise Tok316GateError("fertility strata drift")
    if protocol.get("metric_surface") != (
        "eligible tokenizer-fit train corpus only; no selection-validation or final-test bytes"
    ):
        raise Tok316GateError("metric surface drift")
    if protocol.get("winner_selection_permitted") is not False:
        raise Tok316GateError("tokenizer winner selection must remain prohibited")
    if protocol.get("model_family_selection_permitted") is not False:
        raise Tok316GateError("model-family selection must remain prohibited")
    if protocol.get("embedding_tax_baseline_vocab") != 256:
        raise Tok316GateError("byte baseline vocabulary drift")

    execution = value.get("execution")
    if not isinstance(execution, dict):
        raise Tok316GateError("execution truth missing")
    if execution.get("tokenizer_training_started") is not False:
        raise Tok316GateError("blocked evidence cannot claim tokenizer training started")
    if execution.get("independent_training_runs_completed") != 0:
        raise Tok316GateError("blocked evidence cannot claim completed trainings")
    if execution.get("expected_independent_training_runs") != 8:
        raise Tok316GateError("expected training count drift")
    if execution.get("final_test_bytes_read") is not False:
        raise Tok316GateError("final-test exposure is prohibited")
    if execution.get("selection_validation_bytes_read") is not False:
        raise Tok316GateError("selection-validation exposure is prohibited for TOK-316")
    if execution.get("numerical_winner_claimed") is not False:
        raise Tok316GateError("numerical winner may not be claimed")

    results = value.get("candidate_results")
    if not isinstance(results, list) or len(results) != 4:
        raise Tok316GateError("candidate result vector must contain four rows")
    for expected_vocab, row in zip(EXPECTED_GRID, results, strict=True):
        if row.get("requested_vocab_size") != expected_vocab:
            raise Tok316GateError("candidate ordering or vocabulary size drift")
        if row.get("training_run_1") != "NOT_RUN" or row.get("training_run_2") != "NOT_RUN":
            raise Tok316GateError("blocked evidence cannot claim a candidate training run")
        for key in (
            "byte_identical_artifacts",
            "strict_roundtrip",
            "unintended_unknown_tokens",
            "ua_fertility",
            "en_fertility",
            "code_fertility",
            "throughput",
        ):
            if row.get(key) is not None:
                raise Tok316GateError(f"blocked evidence cannot publish numerical {key}")
        delta = expected_vocab - 256
        if row.get("embedding_tax_over_byte_baseline_tied_per_d_model") != delta:
            raise Tok316GateError("tied embedding-tax coefficient drift")
        if row.get("embedding_tax_over_byte_baseline_untied_per_d_model") != 2 * delta:
            raise Tok316GateError("untied embedding-tax coefficient drift")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("evidence/tok316/authority-gate.json"),
    )
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Tok316GateError("gate must be a JSON object")
    validate_gate(value)
    print("TOK316_AUTHORITY_GATE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
