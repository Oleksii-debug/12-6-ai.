#!/usr/bin/env python3
"""Derive a fail-closed Ukrainian orthographic-signal gate from terminal Rada_Trees metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-rada-trees-language-evidence-report.v1"


class LanguageEvidenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LanguageEvidenceError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def verify_parent(report: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    require(
        report.get("schema_version")
        == "12-6.d03-rada-trees-secondary-plaintext-full-scan.v1",
        "full-scan schema drift",
    )
    core = dict(report)
    claimed = core.pop("report_sha256", None)
    require(canonical_sha256(core) == claimed, "full-scan report hash invalid")
    parent = config["parent_full_scan"]
    require(claimed == parent["report_sha256"], "full-scan report drift")
    classification = report.get("classification")
    require(isinstance(classification, dict), "classification missing")
    metadata = classification.get("member_metadata")
    require(isinstance(metadata, list), "member metadata missing")
    require(
        canonical_sha256(sorted(metadata, key=lambda row: str(row["path"])))
        == parent["member_metadata_identity_sha256"],
        "member metadata identity drift",
    )
    require(
        classification.get("raw_member_text_emitted") is False,
        "raw-text boundary weakened",
    )
    return metadata


def exact_unique_survivors(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        if row.get("classification") == "PLAIN_TEXT_CANDIDATE":
            by_hash[str(row["sha256"])].append(row)
    survivors = [min(group, key=lambda row: str(row["path"])) for group in by_hash.values()]
    return sorted(survivors, key=lambda row: str(row["path"]))


def _rights_inventory(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(row["path"]),
            "size_bytes": int(row["size_bytes"]),
            "sha256": str(row["sha256"]),
            "date": str(row["path"])[6:16],
        }
        for row in rows
    ]


def derive(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    metadata = verify_parent(report, config)
    survivors = exact_unique_survivors(metadata)
    parent = config["parent_full_scan"]
    require(len(survivors) == parent["exact_unique_payload_count"], "survivor count drift")
    require(
        sum(int(row["size_bytes"]) for row in survivors) == parent["exact_unique_bytes"],
        "survivor byte drift",
    )

    scope = config["rights_scope"]
    pattern = re.compile(scope["accepted_path_regex"])
    accepted = [row for row in survivors if pattern.fullmatch(str(row["path"]))]
    inventory = _rights_inventory(accepted)
    require(
        len(inventory) == scope["accepted_exact_unique_members"],
        "rights-scope count drift",
    )
    require(
        sum(row["size_bytes"] for row in inventory) == scope["accepted_exact_unique_bytes"],
        "rights-scope byte drift",
    )
    require(
        canonical_sha256(inventory) == scope["accepted_path_inventory_sha256"],
        "rights-scope inventory drift",
    )

    policy = config["language_signal_policy"]
    passed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    minimum_cyrillic: tuple[float, str] | None = None
    minimum_ukrainian: tuple[float, str] | None = None
    minimum_letters: tuple[int, str] | None = None
    total_letters = 0
    total_ukrainian_specific = 0
    weighted_cyrillic_numerator = 0.0

    for row in accepted:
        metrics = row.get("text_metrics")
        require(isinstance(metrics, dict), f"text metrics missing: {row['path']}")
        letters = int(metrics["letters"])
        ukrainian_specific = int(metrics["ukrainian_specific_letter_count"])
        cyrillic_fraction = float(metrics["cyrillic_letter_fraction"])
        require(letters > 0, f"letters non-positive: {row['path']}")
        ukrainian_fraction = ukrainian_specific / letters
        compact = {
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
            "size_bytes": int(row["size_bytes"]),
        }
        reasons: list[str] = []
        if letters < policy["minimum_letters"]:
            reasons.append("TOO_FEW_LETTERS")
        if cyrillic_fraction < policy["minimum_cyrillic_letter_fraction"]:
            reasons.append("CYRILLIC_FRACTION_BELOW_FLOOR")
        if ukrainian_fraction < policy["minimum_ukrainian_specific_letter_fraction"]:
            reasons.append("UKRAINIAN_SPECIFIC_FRACTION_BELOW_FLOOR")
        if reasons:
            failed.append({**compact, "reasons": reasons})
        else:
            passed.append(compact)
        candidate_cyrillic = (cyrillic_fraction, str(row["path"]))
        candidate_ukrainian = (ukrainian_fraction, str(row["path"]))
        candidate_letters = (letters, str(row["path"]))
        if minimum_cyrillic is None or candidate_cyrillic < minimum_cyrillic:
            minimum_cyrillic = candidate_cyrillic
        if minimum_ukrainian is None or candidate_ukrainian < minimum_ukrainian:
            minimum_ukrainian = candidate_ukrainian
        if minimum_letters is None or candidate_letters < minimum_letters:
            minimum_letters = candidate_letters
        total_letters += letters
        total_ukrainian_specific += ukrainian_specific
        weighted_cyrillic_numerator += cyrillic_fraction * letters

    require(minimum_cyrillic is not None, "no rights-scope language rows")
    require(minimum_ukrainian is not None, "no Ukrainian-specific metrics")
    require(minimum_letters is not None, "no letter metrics")
    expected = config["expected_result"]
    require(len(passed) == expected["passed_members"], "pass count drift")
    require(sum(row["size_bytes"] for row in passed) == expected["passed_bytes"], "pass byte drift")
    require(len(failed) == expected["failed_members"], "failed count drift")
    require(
        abs(minimum_cyrillic[0] - expected["observed_minimum_cyrillic_letter_fraction"])
        < 1e-15,
        "minimum Cyrillic fraction drift",
    )
    require(
        abs(
            minimum_ukrainian[0]
            - expected["observed_minimum_ukrainian_specific_letter_fraction"]
        )
        < 1e-15,
        "minimum Ukrainian-specific fraction drift",
    )
    require(minimum_letters[0] == expected["observed_minimum_letters"], "minimum letters drift")

    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution_profile": config["execution_profile"],
        "config_sha256": canonical_sha256(config),
        "parent_full_scan": parent,
        "rights_scope": scope,
        "language_signal_policy": policy,
        "language_evidence": {
            "evaluated_exact_unique_members": len(accepted),
            "evaluated_bytes": sum(int(row["size_bytes"]) for row in accepted),
            "passed_members": len(passed),
            "passed_bytes": sum(row["size_bytes"] for row in passed),
            "failed_members": len(failed),
            "failed_bytes": sum(row["size_bytes"] for row in failed),
            "observed_minimum_cyrillic_letter_fraction": minimum_cyrillic[0],
            "minimum_cyrillic_fraction_path": minimum_cyrillic[1],
            "observed_minimum_ukrainian_specific_letter_fraction": minimum_ukrainian[0],
            "minimum_ukrainian_specific_fraction_path": minimum_ukrainian[1],
            "observed_minimum_letters": minimum_letters[0],
            "minimum_letters_path": minimum_letters[1],
            "total_letters": total_letters,
            "total_ukrainian_specific_letters": total_ukrainian_specific,
            "weighted_ukrainian_specific_letter_fraction": (
                total_ukrainian_specific / total_letters
            ),
            "weighted_cyrillic_letter_fraction": (
                weighted_cyrillic_numerator / total_letters
            ),
            "passed_inventory_sha256": canonical_sha256(passed),
            "failed_inventory": failed,
            "raw_text_accessed_by_this_derivation": False,
        },
        "decision": {
            "status": "RETAINED_METADATA_UKRAINIAN_LANGUAGE_SIGNAL_PASS_ALL_RIGHTS_SCOPE_SURVIVORS",
            "language_gate_scope": "ORTHOGRAPHIC_SIGNAL_FROM_TERMINAL_RETAINED_TEXT_METRICS",
            "quality_gate_complete": False,
            "privacy_gate_complete": False,
            "training_admission_claimed": False,
            "next_required": config["next_required"],
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
        print("D03_RADA_TREES_LANGUAGE_SIGNAL=PASS_ZERO_CREDIT")
        print("PASSED_BYTES=" + str(result["language_evidence"]["passed_bytes"]))
        print("REPORT_SHA256=" + result["report_sha256"])
        return 0
    except (OSError, ValueError, KeyError, TypeError, LanguageEvidenceError) as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
