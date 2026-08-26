#!/usr/bin/env python3
"""Fail-closed validator for NEXT100-063 source authority convergence."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("configs/data/next100_063_source_registry_convergence_v1.json")
SCHEMA = "12-6.next100-063-source-registry-convergence.v1"
WORKER = "NEXT100-063-SOURCE-REGISTRY-CONVERGENCE"
STRATA = ("uk", "en", "code")

EXPECTED_BASE = {
    "worker_id": "NEXT100-065-CROSSSOURCE-DEDUP-V3",
    "head_sha": "efc278cec0e4773eb4ff405bf4b4d24ee63b5d13",
    "config_path": "configs/data/next100_065_cross_source_dedup_v3.json",
    "config_blob_sha1": "c1e05f09490e25f6fed765dfb70d900717528f4d",
    "workflow_run": 32999969398,
    "workflow_name": "NEXT100-065 Cross-Source Dedup V3",
    "workflow_conclusion": "success",
    "terminal_refresh_cutoff_utc": "2026-08-26T18:20:54Z",
}
EXPECTED_LATE = {
    "NEXT100-026-DATA-UA-CABINET-MINISTRY": {
        "pr": 449,
        "head_sha": "40950a950b60921fd856af2719e1ae2486d9e892",
        "workflow_run": 32997970539,
        "workflow_name": "NEXT100-026 KMu Source Rights Audit",
        "authority_path": "configs/data/next100_026_kmu_source_audit_v1.json",
        "authority_blob_sha1": "6f0a60dc161c0bb2d7600c3c062ae78e624b240e",
        "authority_identity": "1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9",
        "family_id": "ua.kmu.portal.secretariat-news",
        "stratum": "uk",
        "numeric_capacity_bytes": 9153,
        "capacity_object_count": 6,
        "independent_family_credit": 1,
    },
    "NEXT100-034-DATA-EN-NIST": {
        "pr": 472,
        "head_sha": "b7491745b34ac8679baaf69cb96cd609dcbe0a16",
        "workflow_run": 32998703545,
        "workflow_name": "NEXT100-034 NIST authority",
        "authority_path": "configs/data/next100_034_nist_terminal_authority_v2.json",
        "authority_blob_sha1": "8cef5ae316a45ad6265b732be86bd54a977405a1",
        "authority_identity": "3ffba0fcd08ab42e940b2db12ffafb6f7234ad0bae6f7fe523071497485b9d1c",
        "family_id": "en.usgov.nist.technical-series",
        "stratum": "en",
        "numeric_capacity_bytes": 59358,
        "capacity_object_count": 3,
        "independent_family_credit": 1,
    },
    "NEXT100-027-DATA-UA-PUBLIC-DOMAIN-LIT": {
        "pr": 462,
        "head_sha": "d75edd497c7fb1054e86d892c9462f059c1f4aa9",
        "workflow_run": 32998503672,
        "workflow_name": "NEXT100-027 Ukrainian public-domain literature",
        "authority_path": "configs/data/next100_027_ua_public_domain_lit_v1.json",
        "authority_blob_sha1": "c09b8951aeaaa9d42da40ffcc180750fba4258c3",
        "authority_identity": "85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7",
        "family_id": "ua.verba.public-domain.nomis1864",
        "stratum": "uk",
        "numeric_capacity_bytes": 1659,
        "capacity_object_count": 1,
        "independent_family_credit": 1,
    },
    "NEXT100-037-DATA-EN-PYTHON-DOCS": {
        "pr": 467,
        "head_sha": "5a6a495a24bce449334cbc5126d0114f61a9f57c",
        "workflow_run": 32998356906,
        "workflow_name": "NEXT100-037 Python Docs Source Authority",
        "authority_path": "configs/data/next100_037_python_docs_source_authority_v1.json",
        "authority_blob_sha1": "b15abac8744ccda9fe58d1351f7925b6ab328034",
        "authority_identity": "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d",
        "family_id": "python.cpython.documentation",
        "stratum": "en",
        "numeric_capacity_bytes": 0,
        "capacity_object_count": 0,
        "independent_family_credit": 0,
    },
}


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sum_total(mapping: dict[str, Any]) -> int:
    return sum(int(mapping[key]) for key in STRATA)


def validate(data: dict[str, Any]) -> dict[str, Any]:
    _require(data.get("schema_version") == SCHEMA, "schema mismatch")
    _require(data.get("worker_id") == WORKER, "worker mismatch")
    for key, expected in (
        ("local_free_only", True),
        ("model_training_executed", False),
        ("tokenizer_fit_executed", False),
        ("paid_compute_used", False),
        ("final_test_payload_read", False),
    ):
        _require(data.get(key) is expected, f"unsafe boundary: {key}")

    base = data.get("base_authority")
    _require(isinstance(base, dict), "base_authority missing")
    for key, expected in EXPECTED_BASE.items():
        _require(base.get(key) == expected, f"base binding changed: {key}")
    base_capacity = base.get("numeric_capacity_bytes")
    base_families = base.get("independent_family_counts")
    _require(
        isinstance(base_capacity, dict) and isinstance(base_families, dict),
        "base vectors missing",
    )
    _require(
        _sum_total(base_capacity) == base_capacity.get("total") == 243970,
        "base capacity arithmetic mismatch",
    )
    _require(
        _sum_total(base_families) == base_families.get("total") == 7,
        "base family arithmetic mismatch",
    )
    _require(base.get("source_object_count") == 11, "base source-object count changed")

    policy = data.get("credit_policy")
    _require(isinstance(policy, dict), "credit_policy missing")
    _require(
        "dedicated source workflow run id and exact workflow name" in policy.get(
            "workflow_binding_rule", ""
        ),
        "dedicated workflow binding rule missing",
    )

    late = data.get("late_authorities")
    _require(
        isinstance(late, list) and len(late) == len(EXPECTED_LATE),
        "late authority set changed",
    )
    by_worker: dict[str, dict[str, Any]] = {}
    for row in late:
        _require(isinstance(row, dict), "late authority row must be object")
        worker = row.get("worker_id")
        _require(
            isinstance(worker, str) and worker not in by_worker,
            "late worker ids must be unique",
        )
        by_worker[worker] = row
    _require(set(by_worker) == set(EXPECTED_LATE), "unexpected or missing late authority")

    add_capacity = {key: 0 for key in STRATA}
    add_families = {key: 0 for key in STRATA}
    add_objects = 0
    for worker, expected in EXPECTED_LATE.items():
        row = by_worker[worker]
        for key, value in expected.items():
            _require(row.get(key) == value, f"{worker}: binding changed: {key}")
        _require(
            row.get("workflow_conclusion") == "success",
            f"{worker}: workflow is not terminal-success",
        )
        _require(row.get("terminal_status") == "ADMIT", f"{worker}: source is not ADMIT")
        _require(
            row.get("training_authorized") is True,
            f"{worker}: training permission missing",
        )
        _require(
            row.get("evaluation_authorized") is False,
            f"{worker}: evaluation permission must not be inferred",
        )
        capacity = int(row["numeric_capacity_bytes"])
        family_credit = int(row["independent_family_credit"])
        object_count = int(row["capacity_object_count"])
        _require(
            capacity >= 0 and family_credit in (0, 1) and object_count >= 0,
            f"{worker}: invalid credit",
        )
        if worker == "NEXT100-037-DATA-EN-PYTHON-DOCS":
            _require(
                row.get("capacity_credit_status")
                == "BLOCKED_ACCEPTED_CHUNK_BYTE_LEDGER_NOT_MATERIALIZED",
                "CPython docs must fail closed on exact accepted-byte capacity",
            )
            _require(
                row.get("source_normalized_bytes_not_capacity_credit") == 17901,
                "CPython source normalized bytes changed",
            )
            _require(
                row.get("accepted_chunk_count") == 14
                and row.get("rejected_chunk_count") == 2,
                "CPython quality partition changed",
            )
            _require(
                capacity == family_credit == object_count == 0,
                "CPython docs received premature numeric credit",
            )
        add_capacity[row["stratum"]] += capacity
        add_families[row["stratum"]] += family_credit
        add_objects += object_count

    expected_capacity = {
        key: int(base_capacity[key]) + add_capacity[key] for key in STRATA
    }
    expected_capacity["total"] = _sum_total(expected_capacity)
    expected_families = {
        key: int(base_families[key]) + add_families[key] for key in STRATA
    }
    expected_families["total"] = _sum_total(expected_families)
    expected_objects = int(base["source_object_count"]) + add_objects

    vector = data.get("converged_pre_successor_dedup_vector")
    _require(isinstance(vector, dict), "converged vector missing")
    _require(
        vector.get("numeric_capacity_bytes") == expected_capacity,
        "converged capacity arithmetic mismatch",
    )
    _require(
        vector.get("independent_family_counts") == expected_families,
        "converged family arithmetic mismatch",
    )
    _require(
        vector.get("numeric_source_object_count") == expected_objects,
        "converged object count mismatch",
    )
    minimum = int(vector.get("family_minimum_required_per_stratum", 0))
    _require(minimum == 2, "family minimum policy changed")
    _require(
        all(expected_families[key] >= minimum for key in STRATA),
        "pre-dedup family minimum does not pass",
    )
    _require(
        vector.get("family_minimum_candidate_status")
        == "PASS_PRE_SUCCESSOR_GLOBAL_DEDUP",
        "family candidate status mismatch",
    )
    _require(
        vector.get("canonical_balance_diversity_status")
        == "RETEST_REQUIRED_AFTER_SUCCESSOR_GLOBAL_DEDUP",
        "G09 was promoted prematurely",
    )

    acquisition = data.get("acquisition_plan_bytes")
    _require(isinstance(acquisition, dict), "acquisition plan missing")
    targets = acquisition.get("frozen_targets")
    gaps = acquisition.get("remaining_gap")
    _require(
        isinstance(targets, dict) and isinstance(gaps, dict),
        "acquisition vectors missing",
    )
    _require(
        _sum_total(targets) == targets.get("total") == 20000000,
        "frozen acquisition target changed",
    )
    expected_gaps = {
        key: int(targets[key]) - expected_capacity[key] for key in STRATA
    }
    expected_gaps["total"] = _sum_total(expected_gaps)
    _require(gaps == expected_gaps, "remaining acquisition gap mismatch")
    _require(
        "not optimized causal-target" in acquisition.get("note", ""),
        "byte/token truth boundary missing",
    )

    gates = data.get("downstream_handoff")
    _require(isinstance(gates, list), "downstream handoff missing")
    statuses = {
        row.get("gate"): row.get("status")
        for row in gates
        if isinstance(row, dict)
    }
    required = {
        "GLOBAL_CROSS_SOURCE_DEDUP": "REQUIRED",
        "BALANCE_DIVERSITY": "RETEST_REQUIRED",
        "CORPUS_MATERIALIZATION": "BLOCKED",
        "DECONTAMINATION": "BLOCKED",
        "UNIQUE_LOSS_LEDGER": "BLOCKED",
        "TOKENIZER_FIT": "BLOCKED",
        "LEARNED_20M_CAMPAIGN": "BLOCKED",
    }
    _require(statuses == required, "downstream gate truth boundary changed")

    boundary = data.get("claim_boundary")
    _require(isinstance(boundary, dict), "claim boundary missing")
    for key in (
        "canonical_registry_rewritten",
        "post_dedup_capacity_claimed",
        "research_corpus_v1_released",
        "learned_20m_checkpoint_claimed",
        "learned_100m_checkpoint_claimed",
    ):
        _require(boundary.get(key) is False, f"premature claim: {key}")

    return {
        "status": "PASS",
        "capacity_bytes": expected_capacity,
        "family_counts": expected_families,
        "numeric_source_object_count": expected_objects,
        "next_gate": "GLOBAL_CROSS_SOURCE_DEDUP",
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        report = validate(data)
    except (
        OSError,
        json.JSONDecodeError,
        ValidationError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"NEXT100-063 FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
