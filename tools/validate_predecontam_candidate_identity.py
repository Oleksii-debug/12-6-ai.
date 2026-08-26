#!/usr/bin/env python3
"""Independently verify a ~20M pre-decontamination candidate identity.

This verifier is read-only. It cannot mark a corpus final, authorize training,
or perform decontamination. It recomputes every derived field published by
build_predecontam_candidate_identity.py and fails closed on any mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
STRATA = ("ua", "en", "code")
OUTPUT_SCHEMA = "12-6.predecontam-candidate-identity.v1"
EXPECTED_SCOPE = "sha256(canonical JSON without candidate_identity_sha256)"

RECORD_FIELDS = {
    "authority_head_sha",
    "authority_identity_sha256",
    "evaluation_reserved",
    "final_training_eligible",
    "normalized_sha256",
    "normalized_utf8_bytes",
    "pre_decontamination_candidate",
    "raw_sha256",
    "source_family",
    "source_id",
    "source_rights_training_allowed",
    "stratum",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_hex(value: Any, regex: re.Pattern[str], field: str) -> str:
    if not isinstance(value, str) or regex.fullmatch(value) is None:
        raise ValueError(f"{field} must be lowercase hexadecimal with exact width")
    return value


def _validate_record(record: Any, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"records[{index}] must be an object")
    if set(record) != RECORD_FIELDS:
        missing = sorted(RECORD_FIELDS - set(record))
        extra = sorted(set(record) - RECORD_FIELDS)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(extra))
        raise ValueError(f"records[{index}] field mismatch: " + "; ".join(detail))

    for field in ("source_id", "source_family"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"records[{index}].{field} must be non-empty")

    if record["stratum"] not in STRATA:
        raise ValueError(f"records[{index}].stratum must be one of {STRATA}")

    _require_hex(record["raw_sha256"], HEX64, f"records[{index}].raw_sha256")
    _require_hex(
        record["normalized_sha256"], HEX64, f"records[{index}].normalized_sha256"
    )
    _require_hex(
        record["authority_head_sha"], HEX40, f"records[{index}].authority_head_sha"
    )
    _require_hex(
        record["authority_identity_sha256"],
        HEX64,
        f"records[{index}].authority_identity_sha256",
    )

    size = record["normalized_utf8_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(
            f"records[{index}].normalized_utf8_bytes must be a positive integer"
        )

    required_flags = {
        "source_rights_training_allowed": True,
        "pre_decontamination_candidate": True,
        "final_training_eligible": False,
        "evaluation_reserved": False,
    }
    for field, expected in required_flags.items():
        if record[field] is not expected:
            raise ValueError(f"records[{index}].{field} must be {expected}")

    return record


def validate_candidate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("candidate must be a JSON object")
    if value.get("schema") != OUTPUT_SCHEMA:
        raise ValueError(f"schema must be {OUTPUT_SCHEMA}")
    if value.get("state") != "PRE_DECONTAMINATION_CANDIDATE_ONLY":
        raise ValueError("state must remain PRE_DECONTAMINATION_CANDIDATE_ONLY")
    if value.get("decontamination_required") is not True:
        raise ValueError("decontamination_required must be true")
    if value.get("decontamination_executed") is not False:
        raise ValueError("decontamination_executed must be false")
    if value.get("final_corpus_identity") is not None:
        raise ValueError("pre-decontamination candidate cannot carry final_corpus_identity")
    if value.get("final_training_authorized") is not False:
        raise ValueError("pre-decontamination candidate cannot authorize training")
    if value.get("replay_authorized") is not False:
        raise ValueError("pre-decontamination candidate cannot authorize replay")
    if value.get("candidate_identity_scope") != EXPECTED_SCOPE:
        raise ValueError("candidate_identity_scope mismatch")

    records_raw = value.get("records")
    if not isinstance(records_raw, list) or not records_raw:
        raise ValueError("records must be a non-empty list")
    records = [_validate_record(record, i) for i, record in enumerate(records_raw)]

    expected_order = sorted(
        records,
        key=lambda r: (
            r["stratum"],
            r["source_family"],
            r["source_id"],
            r["normalized_sha256"],
            r["raw_sha256"],
        ),
    )
    if records != expected_order:
        raise ValueError("records are not in canonical builder order")

    source_ids = [record["source_id"] for record in records]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate source_id in candidate inventory")
    normalized = [record["normalized_sha256"] for record in records]
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate normalized content identity in candidate inventory")

    counts = {
        stratum: sum(record["stratum"] == stratum for record in records)
        for stratum in STRATA
    }
    bytes_by_stratum = {
        stratum: sum(
            record["normalized_utf8_bytes"]
            for record in records
            if record["stratum"] == stratum
        )
        for stratum in STRATA
    }
    families = {
        stratum: sorted(
            {
                record["source_family"]
                for record in records
                if record["stratum"] == stratum
            }
        )
        for stratum in STRATA
    }
    if any(len(families[stratum]) < 2 for stratum in STRATA):
        raise ValueError("candidate must preserve >=2 independent families per stratum")

    if value.get("record_count") != len(records):
        raise ValueError("record_count mismatch")
    if value.get("counts_by_stratum") != counts:
        raise ValueError("counts_by_stratum mismatch")
    if value.get("normalized_utf8_bytes_by_stratum") != bytes_by_stratum:
        raise ValueError("normalized_utf8_bytes_by_stratum mismatch")
    if value.get("total_normalized_utf8_bytes") != sum(bytes_by_stratum.values()):
        raise ValueError("total_normalized_utf8_bytes mismatch")
    if value.get("independent_families_by_stratum") != families:
        raise ValueError("independent_families_by_stratum mismatch")

    authority_pairs = sorted(
        {
            (record["authority_head_sha"], record["authority_identity_sha256"])
            for record in records
        }
    )
    authority_bundle = [
        {"head_sha": head, "authority_identity_sha256": identity}
        for head, identity in authority_pairs
    ]
    if value.get("source_authority_bundle") != authority_bundle:
        raise ValueError("source_authority_bundle mismatch")
    if value.get("source_authority_bundle_identity_sha256") != _sha256(authority_bundle):
        raise ValueError("source_authority_bundle_identity_sha256 mismatch")
    if value.get("candidate_record_inventory_identity_sha256") != _sha256(records):
        raise ValueError("candidate_record_inventory_identity_sha256 mismatch")

    candidate_identity = _require_hex(
        value.get("candidate_identity_sha256"), HEX64, "candidate_identity_sha256"
    )
    unsigned = dict(value)
    del unsigned["candidate_identity_sha256"]
    if candidate_identity != _sha256(unsigned):
        raise ValueError("candidate_identity_sha256 mismatch")

    return {
        "schema": OUTPUT_SCHEMA,
        "candidate_identity_sha256": candidate_identity,
        "candidate_record_inventory_identity_sha256": value[
            "candidate_record_inventory_identity_sha256"
        ],
        "source_authority_bundle_identity_sha256": value[
            "source_authority_bundle_identity_sha256"
        ],
        "record_count": len(records),
        "total_normalized_utf8_bytes": sum(bytes_by_stratum.values()),
        "counts_by_stratum": counts,
        "independent_families_by_stratum": families,
        "final_training_authorized": False,
        "decontamination_executed": False,
        "verdict": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    value = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = validate_candidate(value)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print("PREDECONTAM_IDENTITY_VALIDATION=PASS")
    print("CANDIDATE_IDENTITY_SHA256=" + report["candidate_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
