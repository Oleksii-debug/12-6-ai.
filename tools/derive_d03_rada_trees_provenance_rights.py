#!/usr/bin/env python3
"""Derive fail-closed Rada_Trees period/session provenance and rights-scope evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-rada-trees-provenance-rights-report.v1"


class ProvenanceRightsError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProvenanceRightsError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def verify_full_scan(report: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    require(
        report.get("schema_version")
        == "12-6.d03-rada-trees-secondary-plaintext-full-scan.v1",
        "full-scan schema drift",
    )
    expected = config["parent_full_scan"]
    claimed_report_sha = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    require(canonical_sha256(core) == claimed_report_sha, "full-scan report hash invalid")
    require(claimed_report_sha == expected["report_sha256"], "full-scan report drift")
    classification = report.get("classification")
    require(isinstance(classification, dict), "full-scan classification missing")
    metadata = classification.get("member_metadata")
    require(isinstance(metadata, list), "full-scan member metadata missing")
    require(
        canonical_sha256(sorted(metadata, key=lambda row: str(row["path"])))
        == expected["member_metadata_identity_sha256"],
        "member metadata identity drift",
    )
    require(
        classification.get("raw_member_text_emitted") is False,
        "full-scan raw-text boundary weakened",
    )
    return metadata


def exact_unique_survivors(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row for row in metadata if row.get("classification") == "PLAIN_TEXT_CANDIDATE"
    ]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_hash[str(row["sha256"])].append(row)
    survivors = [
        min(group, key=lambda row: str(row["path"])) for group in by_hash.values()
    ]
    return sorted(survivors, key=lambda row: str(row["path"]))


def inventory_identity(rows: list[dict[str, Any]]) -> str:
    return canonical_sha256(rows)


def derive(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    metadata = verify_full_scan(report, config)
    survivors = exact_unique_survivors(metadata)
    parent = config["parent_full_scan"]
    require(len(survivors) == parent["exact_unique_payload_count"], "survivor count drift")
    require(
        sum(int(row["size_bytes"]) for row in survivors) == parent["exact_unique_bytes"],
        "survivor byte total drift",
    )

    policy = config["path_provenance_policy"]
    pattern = re.compile(policy["dated_plenary_path_regex"])
    holds_by_path = {row["path"]: row for row in policy["explicit_holds"]}
    accepted: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    year_counts: Counter[int] = Counter()
    year_bytes: Counter[int] = Counter()
    accepted_dates: set[str] = set()

    for row in survivors:
        path = str(row["path"])
        compact = {
            "path": path,
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"]),
        }
        match = pattern.fullmatch(path)
        if match:
            parsed = date.fromisoformat(match.group("date"))
            require(
                policy["minimum_year"] <= parsed.year <= policy["maximum_year"],
                f"dated path outside approved period: {path}",
            )
            compact["date"] = parsed.isoformat()
            accepted.append(compact)
            year_counts[parsed.year] += 1
            year_bytes[parsed.year] += int(row["size_bytes"])
            accepted_dates.add(parsed.isoformat())
            continue

        hold = holds_by_path.get(path)
        require(hold is not None, f"unrecognized path is not explicitly held: {path}")
        require(str(row["sha256"]) == hold["sha256"], f"held SHA drift: {path}")
        require(int(row["size_bytes"]) == hold["size_bytes"], f"held size drift: {path}")
        held.append(compact)

    expected = config["expected_result"]
    accepted_bytes = sum(row["size_bytes"] for row in accepted)
    held_bytes = sum(row["size_bytes"] for row in held)
    require(len(accepted) == expected["accepted_exact_unique_members"], "accepted count drift")
    require(accepted_bytes == expected["accepted_exact_unique_bytes"], "accepted bytes drift")
    require(len(held) == expected["held_exact_unique_members"], "held count drift")
    require(held_bytes == expected["held_exact_unique_bytes"], "held bytes drift")
    require(
        inventory_identity(accepted) == expected["accepted_path_inventory_sha256"],
        "accepted inventory drift",
    )
    require(
        inventory_identity(held) == expected["held_path_inventory_sha256"],
        "held inventory drift",
    )
    dates = sorted(accepted_dates)
    require(dates[0] == expected["minimum_date"], "minimum session date drift")
    require(dates[-1] == expected["maximum_date"], "maximum session date drift")
    require(len(dates) == expected["unique_session_dates"], "unique session-date count drift")
    years = sorted(year_counts)
    require(
        years == list(range(policy["minimum_year"], policy["maximum_year"] + 1)),
        "year coverage gap",
    )

    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution_profile": config["execution_profile"],
        "config_sha256": canonical_sha256(config),
        "parent_full_scan": config["parent_full_scan"],
        "source": config["source"],
        "rights_authorities": config["rights_authorities"],
        "provenance": {
            "accepted_status": policy["accepted_status"],
            "accepted_exact_unique_members": len(accepted),
            "accepted_exact_unique_bytes": accepted_bytes,
            "accepted_path_inventory_sha256": inventory_identity(accepted),
            "held_status": policy["held_status"],
            "held_exact_unique_members": len(held),
            "held_exact_unique_bytes": held_bytes,
            "held_path_inventory_sha256": inventory_identity(held),
            "explicit_holds": [
                {**row, "reason": holds_by_path[row["path"]]["reason"]} for row in held
            ],
            "minimum_date": dates[0],
            "maximum_date": dates[-1],
            "unique_session_dates": len(dates),
            "year_member_counts": {str(year): year_counts[year] for year in years},
            "year_bytes": {str(year): year_bytes[year] for year in years},
        },
        "decision": {
            "status": "DATED_PARLIAMENT_TRANSCRIPT_RIGHTS_SCOPE_SUPPORTED_WITH_ATTRIBUTION_ONE_SOURCE_SCOPE_HOLD",
            "attribution_required": True,
            "rights_scope_supported_bytes_are_training_credit": False,
            "next_required": config["next_required_after_rights_scope"],
        },
        "claim_boundary": config["claim_boundary"],
    }
    return {**core, "report_sha256": canonical_sha256(core)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-scan-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.full_scan_report.read_text(encoding="utf-8"))
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = derive(report, config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print("D03_RADA_TREES_PROVENANCE_RIGHTS=PASS_ZERO_CREDIT")
        print(
            "RIGHTS_SCOPE_CANDIDATE_BYTES="
            + str(result["provenance"]["accepted_exact_unique_bytes"])
        )
        print("HELD_BYTES=" + str(result["provenance"]["held_exact_unique_bytes"]))
        print("REPORT_SHA256=" + result["report_sha256"])
        return 0
    except (OSError, ValueError, KeyError, TypeError, ProvenanceRightsError) as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
