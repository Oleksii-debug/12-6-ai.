#!/usr/bin/env python3
"""Derive a text-free Ukrainian-language gate from terminal Rada_Trees metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "12-6.d03-rada-trees-language-gate.v1"
REPORT_SCHEMA = "12-6.d03-rada-trees-language-gate-report.v1"
FULL_SCAN_SCHEMA = "12-6.d03-rada-trees-secondary-plaintext-full-scan.v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class LanguageGateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LanguageGateError(message)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_config(config: dict[str, Any]) -> None:
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema drift")
    require(config.get("execution_profile") == "LOCAL_FREE", "execution profile weakened")
    policy = config.get("language_policy")
    require(isinstance(policy, dict), "language policy missing")
    require(policy.get("mode") == "uk", "language mode drift")
    require(
        policy.get("metric_authority")
        == "FULL_SCAN_TEXT_DERIVED_METRICS_TEXT_NOT_RETAINED",
        "metric authority drift",
    )
    require(
        policy.get("missing_metric_action") == "FAIL_CLOSED",
        "missing metrics must fail closed",
    )
    boundary = config.get("claim_boundary")
    require(isinstance(boundary, dict), "claim boundary missing")
    required_false = (
        "language_gate_pass_is_training_credit",
        "quality_complete",
        "privacy_complete",
        "global_dedup_complete",
        "evaluation_decontamination_complete",
        "family_cap_mix_complete",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_accessed",
        "paid_compute_used",
    )
    for key in required_false:
        require(boundary.get(key) is False, f"claim boundary weakened: {key}")
    require(boundary.get("training_authorized_bytes") == 0, "training bytes must remain zero")
    require(
        boundary.get("unique_causal_loss_positions_authorized") == 0,
        "loss positions must remain zero",
    )
    require(boundary.get("optimizer_updates") == 0, "optimizer updates must remain zero")


def verify_full_scan(
    report: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    parent = config["parent_full_scan"]
    require(report.get("schema_version") == FULL_SCAN_SCHEMA, "full-scan schema drift")
    claimed = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    require(canonical_sha256(core) == claimed, "full-scan report self-hash invalid")
    require(claimed == parent["report_sha256"], "full-scan report identity drift")
    classification = report.get("classification")
    require(isinstance(classification, dict), "classification missing")
    require(
        classification.get("raw_member_text_emitted") is False,
        "raw-text boundary weakened",
    )
    metadata = classification.get("member_metadata")
    require(isinstance(metadata, list) and metadata, "member metadata missing")
    identity = canonical_sha256(sorted(metadata, key=lambda row: str(row["path"])))
    require(
        identity == parent["member_metadata_identity_sha256"],
        "member metadata identity drift",
    )
    return metadata


def exact_unique_survivors(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        if row.get("classification") != "PLAIN_TEXT_CANDIDATE":
            continue
        digest = row.get("sha256")
        require(
            isinstance(digest, str) and HEX64.fullmatch(digest) is not None,
            "candidate SHA-256 invalid",
        )
        by_hash[digest].append(row)
    survivors = [
        min(group, key=lambda row: str(row["path"])) for group in by_hash.values()
    ]
    return sorted(survivors, key=lambda row: str(row["path"]))


def rights_supported(
    survivors: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    parent = config["parent_provenance_rights"]
    pattern = re.compile(parent["dated_plenary_path_regex"])
    accepted: list[dict[str, Any]] = []
    held_seen = False
    for row in survivors:
        path = str(row["path"])
        if pattern.fullmatch(path):
            accepted.append(row)
            continue
        require(path == parent["held_path"], f"unexpected non-plenary survivor: {path}")
        require(row.get("sha256") == parent["held_sha256"], "held SHA drift")
        require(row.get("size_bytes") == parent["held_bytes"], "held byte drift")
        held_seen = True
    require(held_seen, "expected source-scope hold absent")

    compact: list[dict[str, Any]] = []
    for row in accepted:
        match = pattern.fullmatch(str(row["path"]))
        require(match is not None, "accepted path no longer matches plenary policy")
        compact.append(
            {
                "path": str(row["path"]),
                "size_bytes": int(row["size_bytes"]),
                "sha256": str(row["sha256"]),
                "date": match.group("date"),
            }
        )
    require(
        len(accepted) == parent["accepted_exact_unique_members"],
        "rights-supported count drift",
    )
    require(
        sum(int(row["size_bytes"]) for row in accepted)
        == parent["accepted_exact_unique_bytes"],
        "rights-supported byte drift",
    )
    require(
        canonical_sha256(compact) == parent["accepted_path_inventory_sha256"],
        "rights-supported inventory drift",
    )
    return accepted


def assess_metrics(
    metrics: dict[str, Any], policy: dict[str, Any]
) -> tuple[bool, tuple[str, ...]]:
    required = (
        "letters",
        "cyrillic_letter_fraction",
        "ukrainian_specific_letter_count",
        "tab_fraction",
    )
    for field in required:
        value = metrics.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise LanguageGateError(f"missing/invalid language metric: {field}")
    for field in ("cyrillic_letter_fraction", "tab_fraction"):
        value = float(metrics[field])
        if not 0.0 <= value <= 1.0:
            raise LanguageGateError(f"out-of-range language metric: {field}")
    reasons: list[str] = []
    if float(metrics["cyrillic_letter_fraction"]) < float(
        policy["minimum_cyrillic_letter_fraction"]
    ):
        reasons.append("cyrillic_fraction")
    if int(metrics["ukrainian_specific_letter_count"]) < int(
        policy["minimum_ukrainian_specific_letter_count"]
    ):
        reasons.append("ukrainian_specific_letters")
    if int(metrics["letters"]) < int(policy["minimum_letters"]):
        reasons.append("letter_count")
    if float(metrics["tab_fraction"]) > float(policy["maximum_tab_fraction"]):
        reasons.append("tab_fraction")
    return not reasons, tuple(reasons)


def derive(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    verify_config(config)
    metadata = verify_full_scan(report, config)
    survivors = exact_unique_survivors(metadata)
    scan = config["parent_full_scan"]
    require(
        len(survivors) == scan["exact_unique_payload_count"],
        "exact-unique count drift",
    )
    require(
        sum(int(row["size_bytes"]) for row in survivors) == scan["exact_unique_bytes"],
        "exact-unique byte drift",
    )
    accepted = rights_supported(survivors, config)
    policy = config["language_policy"]

    decisions: list[dict[str, Any]] = []
    passed = rejected = passed_bytes = rejected_bytes = 0
    cyrillic: list[float] = []
    ua_letters: list[int] = []
    letters: list[int] = []
    tabs: list[float] = []
    for row in accepted:
        metrics = row.get("text_metrics")
        require(isinstance(metrics, dict), f"text metrics missing: {row.get('path')}")
        ok, reasons = assess_metrics(metrics, policy)
        size = int(row["size_bytes"])
        decisions.append(
            {
                "path": str(row["path"]),
                "sha256": str(row["sha256"]),
                "size_bytes": size,
                "accepted": ok,
                "reasons": list(reasons),
            }
        )
        if ok:
            passed += 1
            passed_bytes += size
        else:
            rejected += 1
            rejected_bytes += size
        cyrillic.append(float(metrics["cyrillic_letter_fraction"]))
        ua_letters.append(int(metrics["ukrainian_specific_letter_count"]))
        letters.append(int(metrics["letters"]))
        tabs.append(float(metrics["tab_fraction"]))

    result = {
        "language_pass_members": passed,
        "language_pass_bytes": passed_bytes,
        "language_reject_members": rejected,
        "language_reject_bytes": rejected_bytes,
        "minimum_observed_cyrillic_letter_fraction": min(cyrillic),
        "minimum_observed_ukrainian_specific_letter_count": min(ua_letters),
        "minimum_observed_letters": min(letters),
        "maximum_observed_tab_fraction": max(tabs),
        "language_decision_inventory_sha256": canonical_sha256(decisions),
    }
    for key, value in result.items():
        require(
            value == config["expected_result"].get(key),
            f"expected language result drift: {key}",
        )
    status = (
        "UKRAINIAN_LANGUAGE_METRICS_PASS_ZERO_CREDIT"
        if rejected == 0
        else "LANGUAGE_REJECTIONS_REQUIRE_DOWNSTREAM_SURVIVOR_UPDATE"
    )
    core = {
        "schema_version": REPORT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "config_sha256": canonical_sha256(config),
        "parent_full_scan": config["parent_full_scan"],
        "parent_provenance_rights": config["parent_provenance_rights"],
        "source": config["source"],
        "language_policy": policy,
        "language_result": result,
        "decision": {
            "status": status,
            "raw_text_emitted": False,
            "quality_complete": False,
            "privacy_complete": False,
            "training_credit_granted": False,
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
        print("D03_RADA_TREES_LANGUAGE_GATE=" + result["decision"]["status"])
        print(
            "LANGUAGE_PASS_BYTES="
            + str(result["language_result"]["language_pass_bytes"])
        )
        print("REPORT_SHA256=" + result["report_sha256"])
        return 0
    except (OSError, ValueError, KeyError, TypeError, LanguageGateError) as exc:
        print(f"BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
