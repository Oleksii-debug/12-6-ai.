#!/usr/bin/env python3
"""Build an immutable pre-decontamination candidate-record identity.

This tool deliberately does not create a final corpus identity and cannot
authorize training. It exists to give exact/near-match decontamination a
stable, cryptographically bound input inventory.
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
INPUT_SCHEMA = "12-6.predecontam-source-records.v1"
OUTPUT_SCHEMA = "12-6.predecontam-candidate-identity.v1"


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


def _validate_record(record: dict[str, Any]) -> dict[str, Any]:
    required = {
        "source_id",
        "source_family",
        "stratum",
        "raw_sha256",
        "normalized_sha256",
        "normalized_utf8_bytes",
        "authority_head_sha",
        "authority_identity_sha256",
        "source_rights_training_allowed",
        "pre_decontamination_candidate",
        "final_training_eligible",
        "evaluation_reserved",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError("missing record fields: " + ", ".join(missing))

    source_id = record["source_id"]
    source_family = record["source_family"]
    stratum = record["stratum"]
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must be non-empty")
    if not isinstance(source_family, str) or not source_family.strip():
        raise ValueError("source_family must be non-empty")
    if stratum not in STRATA:
        raise ValueError(f"stratum must be one of {STRATA}")

    raw_sha256 = _require_hex(record["raw_sha256"], HEX64, "raw_sha256")
    normalized_sha256 = _require_hex(
        record["normalized_sha256"], HEX64, "normalized_sha256"
    )
    authority_head_sha = _require_hex(
        record["authority_head_sha"], HEX40, "authority_head_sha"
    )
    authority_identity_sha256 = _require_hex(
        record["authority_identity_sha256"], HEX64, "authority_identity_sha256"
    )
    size = record["normalized_utf8_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("normalized_utf8_bytes must be a positive integer")

    if record["source_rights_training_allowed"] is not True:
        raise ValueError("pre-decontamination candidates require training-use rights")
    if record["pre_decontamination_candidate"] is not True:
        raise ValueError("record is not admitted to the pre-decontamination candidate")
    if record["final_training_eligible"] is not False:
        raise ValueError("pre-decontamination builder cannot mark final training eligibility")
    if record["evaluation_reserved"] is not False:
        raise ValueError("evaluation-reserved material cannot enter a training candidate")

    return {
        "authority_head_sha": authority_head_sha,
        "authority_identity_sha256": authority_identity_sha256,
        "evaluation_reserved": False,
        "final_training_eligible": False,
        "normalized_sha256": normalized_sha256,
        "normalized_utf8_bytes": size,
        "pre_decontamination_candidate": True,
        "raw_sha256": raw_sha256,
        "source_family": source_family,
        "source_id": source_id,
        "source_rights_training_allowed": True,
        "stratum": stratum,
    }


def build_candidate(input_value: dict[str, Any]) -> dict[str, Any]:
    if input_value.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"input schema must be {INPUT_SCHEMA}")
    raw_records = input_value.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("records must be a non-empty list")

    records = [_validate_record(dict(item)) for item in raw_records]
    records.sort(
        key=lambda r: (
            r["stratum"],
            r["source_family"],
            r["source_id"],
            r["normalized_sha256"],
            r["raw_sha256"],
        )
    )

    source_ids = [r["source_id"] for r in records]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate source_id in candidate inventory")
    normalized = [r["normalized_sha256"] for r in records]
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate normalized content identity in candidate inventory")

    families = {
        stratum: sorted({r["source_family"] for r in records if r["stratum"] == stratum})
        for stratum in STRATA
    }
    missing_diversity = {
        stratum: family_list
        for stratum, family_list in families.items()
        if len(family_list) < 2
    }
    if missing_diversity:
        detail = ", ".join(
            f"{stratum}={len(family_list)}" for stratum, family_list in missing_diversity.items()
        )
        raise ValueError("candidate requires >=2 independent families per stratum: " + detail)

    counts = {
        stratum: sum(1 for r in records if r["stratum"] == stratum)
        for stratum in STRATA
    }
    bytes_by_stratum = {
        stratum: sum(
            r["normalized_utf8_bytes"] for r in records if r["stratum"] == stratum
        )
        for stratum in STRATA
    }

    authority_bundle = sorted(
        {
            (r["authority_head_sha"], r["authority_identity_sha256"])
            for r in records
        }
    )
    authority_bundle_payload = [
        {"head_sha": head, "authority_identity_sha256": authority}
        for head, authority in authority_bundle
    ]

    record_inventory_identity = _sha256(records)
    authority_bundle_identity = _sha256(authority_bundle_payload)

    output: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "state": "PRE_DECONTAMINATION_CANDIDATE_ONLY",
        "decontamination_required": True,
        "decontamination_executed": False,
        "final_corpus_identity": None,
        "final_training_authorized": False,
        "replay_authorized": False,
        "records": records,
        "record_count": len(records),
        "counts_by_stratum": counts,
        "normalized_utf8_bytes_by_stratum": bytes_by_stratum,
        "total_normalized_utf8_bytes": sum(bytes_by_stratum.values()),
        "independent_families_by_stratum": families,
        "source_authority_bundle": authority_bundle_payload,
        "source_authority_bundle_identity_sha256": authority_bundle_identity,
        "candidate_record_inventory_identity_sha256": record_inventory_identity,
        "candidate_identity_scope": (
            "sha256(canonical JSON without candidate_identity_sha256)"
        ),
    }
    output["candidate_identity_sha256"] = _sha256(output)
    return output


def _synthetic_record(index: int, stratum: str, family: str) -> dict[str, Any]:
    seed = f"{stratum}:{family}:{index}"
    return {
        "source_id": seed,
        "source_family": family,
        "stratum": stratum,
        "raw_sha256": hashlib.sha256(("raw:" + seed).encode()).hexdigest(),
        "normalized_sha256": hashlib.sha256(("norm:" + seed).encode()).hexdigest(),
        "normalized_utf8_bytes": 1000 + index,
        "authority_head_sha": hashlib.sha1(("head:" + family).encode()).hexdigest(),
        "authority_identity_sha256": hashlib.sha256(
            ("authority:" + family).encode()
        ).hexdigest(),
        "source_rights_training_allowed": True,
        "pre_decontamination_candidate": True,
        "final_training_eligible": False,
        "evaluation_reserved": False,
    }


def self_test() -> None:
    rows: list[dict[str, Any]] = []
    index = 0
    for stratum in STRATA:
        for suffix in ("a", "b"):
            index += 1
            rows.append(_synthetic_record(index, stratum, f"{stratum}.family.{suffix}"))

    first = build_candidate({"schema": INPUT_SCHEMA, "records": rows})
    second = build_candidate({"schema": INPUT_SCHEMA, "records": list(reversed(rows))})
    assert first == second
    assert first["record_count"] == 6
    assert first["final_training_authorized"] is False
    assert first["decontamination_required"] is True

    duplicate = json.loads(json.dumps(rows))
    duplicate[-1]["normalized_sha256"] = duplicate[0]["normalized_sha256"]
    try:
        build_candidate({"schema": INPUT_SCHEMA, "records": duplicate})
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate normalized identity did not fail closed")

    promoted = json.loads(json.dumps(rows))
    promoted[0]["final_training_eligible"] = True
    try:
        build_candidate({"schema": INPUT_SCHEMA, "records": promoted})
    except ValueError:
        pass
    else:
        raise AssertionError("pre-decontamination builder promoted final eligibility")

    print("PREDECONTAM_CANDIDATE_IDENTITY_SELF_TEST=PASS")
    print("SYNTHETIC_CANDIDATE_SHA256=" + first["candidate_identity_sha256"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    if args.inventory is None or args.output is None:
        parser.error("--inventory and --output are required unless --self-test is used")

    input_value = json.loads(args.inventory.read_text(encoding="utf-8"))
    output = build_candidate(input_value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PREDECONTAM_CANDIDATE_IDENTITY=" + output["candidate_identity_sha256"])
    print(
        "PREDECONTAM_RECORD_INVENTORY="
        + output["candidate_record_inventory_identity_sha256"]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
