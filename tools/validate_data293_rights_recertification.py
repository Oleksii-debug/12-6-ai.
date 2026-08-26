#!/usr/bin/env python3
"""Validate DATA-293 independent rights recertification without network access."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECERT = ROOT / "configs/data/data293_independent_rights_recertification_v1.json"
DATA229 = ROOT / "data/registry/real_snapshots.v1.json"

EXPECTED_ADMITTED = {
    "ua.rada.open-data.laws-texts.d23314",
    "en.standardebooks.manual.8-typography",
    "en.standardebooks.manual.9-metadata",
    "code.encode.httpx._content",
    "code.psf.requests._internal_utils",
}
EXPECTED_NOT_ADMITTED = {
    "uk.kubernetes.docs.concepts-index",
    "en.python.docs.tutorial-introduction",
}
EXPECTED_FAMILIES = {
    "ua.rada.open-data.laws-texts",
    "en.standardebooks.manual",
    "github:encode/httpx",
    "github:psf/requests",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"DATA-293 FAIL: {message}")


def main() -> None:
    recert = json.loads(RECERT.read_text(encoding="utf-8"))
    require(
        recert.get("schema_version") == "12-6.data293-independent-rights-recertification.v1",
        "schema mismatch",
    )
    require(recert.get("local_free_only") is True, "LOCAL_FREE must be true")

    authorities = recert["baseline_authorities"]
    require(
        authorities["DATA-229"]["source_sha"]
        == "90bc0b7f8b696ec35202532b13edf6ab29a662fe",
        "DATA-229 authority drift",
    )
    require(
        authorities["DATA-227"]["source_sha"]
        == "8ebdb2e132ed7bae5245e9d4c140752640ab9885",
        "DATA-227 authority drift",
    )
    require(
        authorities["DATA-227"]["workflow_run"] == 32956209865,
        "DATA-227 terminal run drift",
    )
    require(
        authorities["DATA-228"]["source_sha"]
        == "46a70c990dab6ff72bb84ddb54cff1156b491b40",
        "DATA-228 candidate authority drift",
    )
    require(
        authorities["DATA-228"]["status"]
        == "TERMINAL_FAILURE_BEFORE_SOURCE_MATERIALIZATION",
        "DATA-228 must remain fail-closed",
    )
    require(
        authorities["DATA-278"]["status"] == "NO_DURABLE_TERMINAL_AUTHORITY_DISCOVERED",
        "DATA-278 must not be invented",
    )

    admitted = recert["admitted"]
    not_admitted = recert["not_admitted"]
    require({x["source_id"] for x in admitted} == EXPECTED_ADMITTED, "admitted set drift")
    require(
        {x["source_id"] for x in not_admitted} == EXPECTED_NOT_ADMITTED,
        "not-admitted set drift",
    )
    require(len(admitted) == 5, "exactly five training-admitted objects required")
    require(len(not_admitted) == 2, "exactly two fail-closed DATA-228 candidates required")

    families = {x["source_family"] for x in admitted}
    require(families == EXPECTED_FAMILIES, "independent source-family set drift")
    se = [x for x in admitted if x["source_id"].startswith("en.standardebooks.manual.")]
    require(len(se) == 2, "both Standard Ebooks objects required")
    require(
        len({x["source_family"] for x in se}) == 1,
        "Standard Ebooks documents must remain one family",
    )

    for row in admitted:
        rights = row["rights"]
        require(rights["evaluation"] == "NOT_SEPARATELY_ADMITTED", f"{row['source_id']}: evaluation weakened")
        require(rights["model_training"].startswith("ALLOWED"), f"{row['source_id']}: training not admitted")

    require(
        next(x for x in admitted if x["source_id"].startswith("ua.rada"))["rights"]["redistribution"]
        == "ALLOWED_WITH_SOURCE_ATTRIBUTION",
        "Rada attribution condition lost",
    )
    require(
        next(x for x in admitted if x["source_id"] == "code.encode.httpx._content")["rights"]["redistribution"]
        == "ALLOWED_WITH_BSD_COPYRIGHT_CONDITIONS_DISCLAIMER",
        "HTTPX BSD redistribution conditions lost",
    )
    requests = next(x for x in admitted if x["source_id"] == "code.psf.requests._internal_utils")
    require(
        requests["rights"]["redistribution"] == "ALLOWED_WITH_APACHE_LICENSE_AND_APPLICABLE_NOTICE",
        "Requests Apache redistribution conditions lost",
    )
    require(
        requests["rights_evidence"]["notice_git_blob_sha1"]
        == "1ff62db688277b77c83c1766dac7f165364d3528",
        "Requests NOTICE binding drift",
    )

    for row in not_admitted:
        require(
            row["corpus_status"] == "NOT_ADMITTED_EVIDENCE_NOT_MATERIALIZED",
            f"{row['source_id']}: DATA-228 candidate improperly admitted",
        )
        require(
            row["evaluation"] == "NOT_SEPARATELY_ADMITTED",
            f"{row['source_id']}: evaluation improperly admitted",
        )

    data229 = json.loads(DATA229.read_text(encoding="utf-8"))
    require(
        data229["registry_identity_sha256"]
        == "1357a343eb4ea973950d8991913109cbea53fe4fa891f0be9745ab497eb59486",
        "DATA-229 registry identity drift",
    )
    text_by_id = {x["raw_identity"]["source_id"]: x for x in data229["sources"]}
    expected_text = {
        "ua.rada.open-data.laws-texts.d23314": "36eae31c3b0676ea7c02236fa05bd695c240c9a8eade5febc00457b8103ee1a4",
        "en.standardebooks.manual.8-typography": "21582c7f0e4ad39f2b0ed97bbc2c082d275e898b7a63c28e6d9badb8ee0f7860",
        "en.standardebooks.manual.9-metadata": "7ac53dfb4bf6f73f178560e09f33160d0250c69fb679802f3254dc0eb4c9f509",
    }
    require(set(text_by_id) == set(expected_text), "DATA-229 text source set drift")
    for source_id, raw_sha in expected_text.items():
        row = text_by_id[source_id]
        require(row["raw_identity"]["raw_sha256"] == raw_sha, f"{source_id}: raw identity drift")
        require(
            row["rights"]["evaluation"]["status"] == "NOT_SEPARATELY_ADMITTED",
            f"{source_id}: DATA-229 evaluation authority drift",
        )

    summary = recert["inventory_summary"]
    require(summary["training_admitted_objects"] == 5, "summary admitted-object count drift")
    require(summary["training_admitted_families"] == 4, "summary family count drift")
    require(summary["evaluation_admitted_objects"] == 0, "evaluation-admitted count must remain zero")
    require(summary["family_counts"] == {"uk": 1, "en": 1, "code": 2}, "stratum family counts drift")

    print("DATA-293 PASS: 5 training objects / 4 families / 0 evaluation objects / 2 fail-closed candidates")


if __name__ == "__main__":
    main()
