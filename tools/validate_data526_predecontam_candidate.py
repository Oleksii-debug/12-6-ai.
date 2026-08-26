#!/usr/bin/env python3
"""Validate the exact DATA-526 pre-decontamination candidate record inventory."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from build_predecontam_candidate_identity import build_candidate

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "configs/data/data526_predecontam_source_records_v1.json"
CANDIDATE_PATH = ROOT / "evidence/data526/predecontam_candidate_v1.json"

EXPECTED_CANDIDATE = "749d1449182abb4d71f90eb3510fb212c5ac8f90d15d8ff60a407b0cebd1baaa"
EXPECTED_INVENTORY = "3b3f6cda92b248d327861e335ec9ccc4ad6fb6a250ac020c9618bf6f14310f21"
EXPECTED_AUTHORITY_BUNDLE = "4d338d4fb79c37afa501cb64663e6a7ca329b1f643f9ad104d489d90aac01b85"
EXPECTED_BYTES = {"ua": 90044, "en": 144151, "code": 9703}
EXPECTED_COUNTS = {"ua": 2, "en": 5, "code": 2}
EXPECTED_FAMILY_COUNTS = {"ua": 2, "en": 2, "code": 2}
EXPECTED_AUTHORITY_RUNS = {
    "data287_incumbent_registry": 32968622282,
    "next100_022_ua_wikisource": 32998002424,
    "next100_034_nist_terminal": 32998703545,
}
EXPECTED_SOURCE_IDS = {
    "external-real:en.standardebooks.manual.8-typography",
    "external-real:en.standardebooks.manual.9-metadata",
    "external-real:ua.rada.open-data.laws-texts.d23314",
    "external-real:code.encode.httpx._content",
    "external-real:code.psf.requests._internal_utils",
    "ua.wikisource.lesia-ukrainka.na-krylah-pisen.1892.page13",
    "en.nist.technical-series.NIST.SP.800-204",
    "en.nist.technical-series.NIST.SP.800-204C",
    "en.nist.technical-series.NIST.SP.800-215",
}


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate() -> dict[str, Any]:
    inventory = _load(INVENTORY_PATH)
    frozen = _load(CANDIDATE_PATH)

    _require(inventory["worker_id"] == "DATA-526-PREDECONTAM-CANDIDATE-RECORDS", "worker drift")
    _require(inventory["repository"] == "Oleksii-debug/12-6-ai.", "repository identity drift")
    _require(inventory["execution_profile"] == "LOCAL_FREE", "execution profile is not LOCAL_FREE")

    truth = inventory["truth_boundary"]
    _require(truth["state"] == "PRE_DECONTAMINATION_INPUT_ONLY", "input truth state drift")
    for key in (
        "decontamination_executed",
        "final_training_authorized",
        "replay_authorized",
        "tokenizer_fit_authorized",
        "long_training_authorized",
        "paid_compute_used",
    ):
        _require(truth[key] is False, f"truth boundary weakened: {key}")
    _require(truth["final_corpus_identity"] is None, "final corpus identity fabricated")

    authorities = inventory["authority_evidence"]
    _require(set(authorities) == set(EXPECTED_AUTHORITY_RUNS), "authority set drift")
    for name, run in EXPECTED_AUTHORITY_RUNS.items():
        authority = authorities[name]
        _require(authority["dedicated_workflow_run"] == run, f"workflow run drift: {name}")
        _require(authority["dedicated_workflow_conclusion"] == "success", f"authority is not terminal-success: {name}")

    held = inventory["held_out_authorities"]
    cpython = held["next100_037_cpython_docs"]
    _require(cpython["dedicated_workflow_conclusion"] == "success", "CPython authority unexpectedly nonterminal")
    _require(
        cpython["status"] == "HOLD_UNTIL_ACCEPTED_CHUNK_RECORD_MATERIALIZATION",
        "CPython fail-closed hold removed",
    )
    _require("14 of 16 chunks" in cpython["reason"], "CPython accepted/rejected chunk boundary lost")
    _require("17901-byte" in cpython["reason"], "CPython full-source exclusion rationale lost")

    records = inventory["records"]
    _require(len(records) == 9, "record count drift")
    _require({record["source_id"] for record in records} == EXPECTED_SOURCE_IDS, "source inventory drift")
    _require("en.python.docs.tutorial-introduction" not in EXPECTED_SOURCE_IDS, "CPython full source accidentally admitted")

    rebuilt = build_candidate(inventory)
    _require(rebuilt == frozen, "frozen candidate does not exactly match deterministic rebuild")
    _require(frozen["candidate_identity_sha256"] == EXPECTED_CANDIDATE, "candidate identity drift")
    _require(frozen["candidate_record_inventory_identity_sha256"] == EXPECTED_INVENTORY, "record inventory identity drift")
    _require(frozen["source_authority_bundle_identity_sha256"] == EXPECTED_AUTHORITY_BUNDLE, "authority bundle identity drift")
    _require(frozen["record_count"] == 9, "frozen record count drift")
    _require(frozen["counts_by_stratum"] == EXPECTED_COUNTS, "stratum record-count drift")
    _require(frozen["normalized_utf8_bytes_by_stratum"] == EXPECTED_BYTES, "stratum byte-count drift")
    _require(frozen["total_normalized_utf8_bytes"] == sum(EXPECTED_BYTES.values()) == 243898, "total byte-count drift")

    family_counts = {
        stratum: len(families)
        for stratum, families in frozen["independent_families_by_stratum"].items()
    }
    _require(family_counts == EXPECTED_FAMILY_COUNTS, "family-diversity vector drift")
    _require(all(count >= 2 for count in family_counts.values()), "two-family gate regressed")

    _require(frozen["state"] == "PRE_DECONTAMINATION_CANDIDATE_ONLY", "candidate state promoted")
    _require(frozen["decontamination_required"] is True, "decontamination requirement removed")
    _require(frozen["decontamination_executed"] is False, "decontamination falsely claimed")
    _require(frozen["final_corpus_identity"] is None, "final corpus identity fabricated")
    _require(frozen["final_training_authorized"] is False, "training prematurely authorized")
    _require(frozen["replay_authorized"] is False, "replay prematurely authorized")

    return {
        "status": "PASS_PRE_DECONTAMINATION_IDENTITY_ONLY",
        "candidate_identity_sha256": EXPECTED_CANDIDATE,
        "record_inventory_identity_sha256": EXPECTED_INVENTORY,
        "record_count": 9,
        "normalized_utf8_bytes": 243898,
        "family_counts": EXPECTED_FAMILY_COUNTS,
        "decontamination_executed": False,
        "final_training_authorized": False,
        "next_gate": "EXACT_AND_NEAR_MATCH_EVALUATION_DECONTAMINATION",
    }


def main() -> int:
    print(json.dumps(validate(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
