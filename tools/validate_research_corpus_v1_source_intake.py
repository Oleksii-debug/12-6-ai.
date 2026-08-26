#!/usr/bin/env python3
"""Fail-closed validator for the Research Corpus V1 source-intake convergence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTAKE = ROOT / "data/registry/research_corpus_v1_source_intake.v1.json"
DEFAULT_PARENT = ROOT / "data/registry/external_snapshots.v2.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_PARENT_HEAD = "b0523ccbc4b957615aac849d476cfa851be87578"
EXPECTED_PARENT_ID = "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"
EXPECTED_ADDITIVE = {
    "NEXT100-026-DATA-UA-CABINET-MINISTRY": ("40950a950b60921fd856af2719e1ae2486d9e892", 32997970539),
    "NEXT100-027-DATA-UA-PUBLIC-DOMAIN-LIT": ("d75edd497c7fb1054e86d892c9462f059c1f4aa9", 32998503672),
    "NEXT100-022-DATA-UA-WIKISOURCE": ("84c51e42b6daa51796fd20d793b5ef1ff01cc9d2", 32998002424),
    "NEXT100-037-DATA-EN-PYTHON-DOCS": ("5a6a495a24bce449334cbc5126d0114f61a9f57c", 32998356906),
    "NEXT100-038-DATA-EN-MDN": ("902eccc0b3efff09a38dc89cda789180b6c6e754", 32998544359),
    "NEXT100-045-CODE-STARLETTE": ("c6756b5ebb6eb1d3bf3de2499167833d99d99a72", 32998101312),
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def self_identity(payload: dict[str, Any]) -> str:
    copy = dict(payload)
    claimed = copy.pop("intake_identity_sha256", None)
    if claimed is not None and not isinstance(claimed, str):
        raise AssertionError("intake identity must be a string")
    return hashlib.sha256(canonical_json_bytes(copy)).hexdigest()


def _fraction(value: Any) -> Fraction:
    return Fraction(str(value))


def _source_level_feasible(total: int, families: dict[str, list[int]], shares: dict[str, Fraction]) -> bool:
    max_within = Fraction(3, 5)
    max_total = Fraction(1, 4)
    for stratum, available in families.items():
        needed = shares[stratum] * total
        capacity = sum(
            min(Fraction(byte_count), max_within * needed, max_total * total)
            for byte_count in available
        )
        if capacity < needed:
            return False
    return True


def balanced_source_byte_upper_bound(
    families: dict[str, list[int]], shares: dict[str, Fraction]
) -> int:
    upper = sum(sum(values) for values in families.values())
    lo, hi = 0, upper + 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if _source_level_feasible(mid, families, shares):
            lo = mid
        else:
            hi = mid
    return lo


def validate(intake_path: Path = DEFAULT_INTAKE, parent_path: Path = DEFAULT_PARENT) -> dict[str, Any]:
    intake = json.loads(intake_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))

    assert intake["schema_version"] == "12-6.research-corpus-v1-source-intake.v1"
    assert intake["execution_profile"] == "LOCAL_FREE"
    assert intake["local_free_only"] is True
    assert HEX64.fullmatch(intake["intake_identity_sha256"])
    assert self_identity(intake) == intake["intake_identity_sha256"]

    parent_binding = intake["parent_registry"]
    assert parent_binding["head_sha"] == EXPECTED_PARENT_HEAD
    assert parent_binding["registry_identity_sha256"] == EXPECTED_PARENT_ID
    assert parent_binding["dedicated_workflow_conclusion"] == "success"
    assert parent["registry_identity_sha256"] == EXPECTED_PARENT_ID
    assert parent["source_count"] == parent_binding["source_count"] == 5
    assert parent["independent_source_family_count"] == parent_binding["independent_family_count"] == 4
    assert parent["byte_report"]["unique_normalized_bytes"] == parent_binding["normalized_source_bytes"] == 183061
    assert parent["claim_boundary"]["training_authorized_source_count"] == 5
    assert parent["claim_boundary"]["evaluation_authorized_source_count"] == 0

    additions = intake["additive_terminal_authorities"]
    assert len(additions) == len(EXPECTED_ADDITIVE)
    by_worker = {row["worker"]: row for row in additions}
    assert len(by_worker) == len(additions)
    assert set(by_worker) == set(EXPECTED_ADDITIVE)

    for worker, (expected_head, expected_run) in EXPECTED_ADDITIVE.items():
        row = by_worker[worker]
        assert HEX40.fullmatch(row["head_sha"])
        assert row["head_sha"] == expected_head
        assert row["dedicated_workflow_run_id"] == expected_run
        assert row["dedicated_workflow_conclusion"] == "success"
        assert row["bounded_normalized_source_bytes"] > 0
        assert row["bounded_record_count"] > 0
        assert row["requires_corpus_decontamination"] is True
        assert "NOT" in row["evaluation_use"]
        payload_identity = row["payload_identity"]
        if isinstance(payload_identity, list):
            assert payload_identity and len(set(payload_identity)) == len(payload_identity)
            assert all(HEX64.fullmatch(value) for value in payload_identity)
        else:
            assert HEX64.fullmatch(payload_identity)

    family_rows = list(intake["parent_family_inventory"]) + [
        {
            "stratum": row["stratum"],
            "family_id": row["family_id"],
            "normalized_source_bytes": row["bounded_normalized_source_bytes"],
        }
        for row in additions
    ]
    family_keys = [(row["stratum"], row["family_id"]) for row in family_rows]
    assert len(family_keys) == len(set(family_keys)), "duplicate family credit in intake"

    families: dict[str, list[int]] = {"uk": [], "en": [], "code": []}
    for row in family_rows:
        assert row["stratum"] in families
        assert row["normalized_source_bytes"] > 0
        families[row["stratum"]].append(row["normalized_source_bytes"])

    diagnostics = intake["source_level_diagnostics"]
    assert diagnostics["diagnostic_only"] is True
    assert diagnostics["bounded_normalized_source_bytes_including_parent"] == sum(
        row["normalized_source_bytes"] for row in family_rows
    )
    expected_counts = {key: len(values) for key, values in families.items()}
    assert diagnostics["family_counts_after_exact_head_success_additions"] == expected_counts
    minimum = intake["composition_contract"]["minimum_independent_families_per_stratum"]
    assert all(count >= minimum for count in expected_counts.values())

    shares = {
        key: _fraction(value)
        for key, value in intake["composition_contract"]["stratum_target_share"].items()
    }
    assert set(shares) == set(families)
    assert sum(shares.values()) == 1
    assert intake["composition_contract"]["maximum_single_family_share_within_stratum"] == 0.60
    assert intake["composition_contract"]["maximum_single_family_share_total"] == 0.25
    computed_upper = balanced_source_byte_upper_bound(families, shares)
    assert computed_upper == diagnostics["balanced_source_byte_upper_bound"] == 61455
    expected_strata = diagnostics["balanced_source_byte_upper_bound_strata"]
    for key, share in shares.items():
        assert Fraction(str(expected_strata[key])) == share * computed_upper

    readiness = intake["readiness"]
    assert readiness == {
        "authorized_unique_loss_positions": 0,
        "candidate_record_inventory_identity_sha256": None,
        "evaluation_decontamination_passed": False,
        "exact_candidate_record_inventory_materialized": False,
        "final_corpus_identity_sha256": None,
        "learned_20m_claim_authorized": False,
        "long_training_authorized": False,
        "quality_privacy_dedup_split_terminal": False,
        "source_authority_intake_frozen": True,
        "tokenizer_fit_authorized": False,
        "two_clean_build_reproducibility_passed": False,
    }

    observed = {row["worker"]: row for row in intake["observed_not_counted"]}
    assert observed["NEXT100-021-DATA-UA-WIKIPEDIA"]["candidate_bytes"] == 0
    assert "REJECT" in observed["NEXT100-021-DATA-UA-WIKIPEDIA"]["reason"]
    assert observed["NEXT100-028-DATA-UA-TECH-GITHUB"]["candidate_bytes"] == 59986
    assert observed["NEXT100-030-DATA-UA-OER"]["candidate_bytes"] is None
    assert "queued" in observed["NEXT100-051-CODE-RICH"]["reason"]

    assert intake["next_required_gate"]["name"] == "RESEARCH_CORPUS_V1_EXACT_RECORD_MATERIALIZATION_AND_IDENTITY"
    return intake


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intake", type=Path, default=DEFAULT_INTAKE)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args()
    intake = validate(args.intake, args.parent)
    print(json.dumps({
        "status": "PASS",
        "intake_identity_sha256": intake["intake_identity_sha256"],
        "authorized_unique_loss_positions": intake["readiness"]["authorized_unique_loss_positions"],
        "long_training_authorized": intake["readiness"]["long_training_authorized"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
