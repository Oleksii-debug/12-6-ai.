#!/usr/bin/env python3
"""DATA-296 deterministic external-real document-quality stress audit.

This is statistics-only. It does not read model results or final-test outcomes and
does not select a policy. Exact upstream identities and incumbent code identities
are preregistered in the committed plan.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.machinery
import json
import re
import sys
import types
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# This audit is deliberately stdlib-only. The project package root imports the
# model runtime (and therefore torch), but the bound DATA-296 data modules do not
# require it. Install namespace packages without executing twelve_six/__init__.py
# so the audit measures data filters rather than runtime packaging side effects.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"


def _install_namespace(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    module.__spec__ = spec
    sys.modules[name] = module


if "twelve_six" not in sys.modules:
    _install_namespace("twelve_six", _SRC_ROOT / "twelve_six")
if "twelve_six.data" not in sys.modules:
    _install_namespace("twelve_six.data", _SRC_ROOT / "twelve_six" / "data")
if "twelve_six.packing" not in sys.modules:
    _install_namespace("twelve_six.packing", _SRC_ROOT / "twelve_six" / "packing")

from twelve_six.data.document_quality import ModeThresholds, QualityPolicy, assess_document
from twelve_six.data.multilingual_pretraining import detect_language, strict_normalize_utf8
from twelve_six.data.source_intake import DownloadedBytes, extract_text

SCHEMA = "12-6.data296-quality-filter-external-audit-report.v1"
_UK_SPECIFIC = frozenset("іїєґІЇЄҐ")
_RST_PATTERNS = (
    re.compile(r"(?m)^\s*\.\.\s+\S+"),
    re.compile(r"(?m)^\s*:[A-Za-z][A-Za-z0-9_-]*:"),
    re.compile(r":[A-Za-z][A-Za-z0-9_-]*:`"),
    re.compile(r"``[^`\n]+``"),
    re.compile(r"(?m)^[=\-~^\"'#*+]{3,}\s*$"),
)


class Data296Error(RuntimeError):
    pass


def cjson(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def verify_preregistration(plan: dict[str, Any], repo: Path) -> None:
    if plan.get("schema_version") != "12-6.data296-quality-filter-external-audit.v1":
        raise Data296Error("unsupported preregistration schema")
    if plan.get("local_free_only") is not True:
        raise Data296Error("LOCAL_FREE contract changed")
    if plan["comparison_contract"]["no_policy_winner_selected"] is not True:
        raise Data296Error("policy selection is forbidden in DATA-296")
    if plan["comparison_contract"]["no_final_test_or_model_metric_inputs"] is not True:
        raise Data296Error("model/final-test input boundary changed")
    claimed = plan.get("preregistration_sha256")
    core = dict(plan)
    core.pop("preregistration_sha256", None)
    if sha256(cjson(core)) != claimed:
        raise Data296Error("preregistration identity mismatch")

    authority = plan["authority_boundary"]
    for path_key, blob_key in (
        ("quality_module_path", "quality_module_git_blob_sha1"),
        ("source_intake_module_path", "source_intake_module_git_blob_sha1"),
    ):
        path = repo / authority[path_key]
        payload = path.read_bytes()
        if git_blob_sha1(payload) != authority[blob_key]:
            raise Data296Error(f"incumbent implementation drift: {authority[path_key]}")


def policy_from_plan(raw: dict[str, Any]) -> QualityPolicy:
    kwargs = {key: value for key, value in raw.items() if key not in {"role", "expected_policy_sha256"}}
    kwargs["uk"] = ModeThresholds(**kwargs["uk"])
    kwargs["en"] = ModeThresholds(**kwargs["en"])
    kwargs["code"] = ModeThresholds(**kwargs["code"])
    policy = QualityPolicy(**kwargs)
    if policy.manifest()["policy_sha256"] != raw["expected_policy_sha256"]:
        raise Data296Error(f"policy identity mismatch: {policy.policy_id}")
    return policy


def download(url: str, max_bytes: int = 2_000_000) -> DownloadedBytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "12-6-DATA-296-quality-audit/1.0",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read(max_bytes + 1)
        content_type = response.headers.get("Content-Type")
    if len(payload) > max_bytes:
        raise Data296Error(f"bounded acquisition exceeded {max_bytes} bytes: {url}")
    return DownloadedBytes(payload=payload, content_type=content_type)


def line_packs(text: str, target_chars: int) -> list[str]:
    if target_chars <= 0:
        raise Data296Error("target_chars must be positive")
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    if not lines:
        return [text]
    packs: list[str] = []
    current: list[str] = []
    current_chars = 0
    for line in lines:
        if current and current_chars >= target_chars:
            packs.append("".join(current))
            current = []
            current_chars = 0
        current.append(line)
        current_chars += len(line)
    if current:
        packs.append("".join(current))
    if "".join(packs) != text:
        raise Data296Error("audit partition did not preserve source text exactly")
    return packs


def is_ukrainian_dominant(text: str) -> bool:
    latin = cyrillic = specific = 0
    for char in text:
        if not char.isalpha():
            continue
        name = __import__("unicodedata").name(char, "")
        if "LATIN" in name:
            latin += 1
        elif "CYRILLIC" in name:
            cyrillic += 1
            if char in _UK_SPECIFIC:
                specific += 1
    alpha = latin + cyrillic
    return alpha > 0 and cyrillic / alpha >= 0.85 and specific >= 2


def is_rst_syntax_bearing(text: str) -> bool:
    return any(pattern.search(text) for pattern in _RST_PATTERNS)


def is_parse_valid_python(text: str) -> bool:
    try:
        ast.parse(text)
    except (SyntaxError, ValueError):
        return False
    return True


def current_reviewed_source_lid(text: str, expected: str) -> dict[str, Any]:
    evidence = detect_language(text, modality="natural", language_hint=None)
    alpha = max(evidence.script.alphabetic_letters, 1)
    cyrillic_ratio = evidence.script.cyrillic_letters / alpha
    admitted = evidence.label == expected
    rule = "DIRECT_DETECTOR_MATCH" if admitted else "REJECT"
    if (
        not admitted
        and expected == "uk"
        and evidence.label == "mixed"
        and cyrillic_ratio >= 0.85
        and evidence.ukrainian_lexical_hits >= 2
        and evidence.script.ukrainian_specific_letters >= 2
    ):
        admitted = True
        rule = "CURRENT_REVIEWED_UK_DOMINANT_CYRILLIC_OVERRIDE"
    return {
        "raw_detector_label": evidence.label,
        "raw_detector_reason": evidence.reason,
        "expected": expected,
        "current_reviewed_source_admitted": admitted,
        "current_reviewed_source_rule": rule,
        "cyrillic_ratio": round(cyrillic_ratio, 6),
        "ukrainian_lexical_hits": evidence.ukrainian_lexical_hits,
        "ukrainian_specific_letters": evidence.script.ukrainian_specific_letters,
        "latin_letters": evidence.script.latin_letters,
        "cyrillic_letters": evidence.script.cyrillic_letters,
    }


def acquire_text_objects(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    source_checks: list[dict[str, Any]] = []
    target = plan["audit_partition"]["natural_target_chars"]
    for item in plan["text_objects"]:
        downloaded = download(item["url"])
        raw = downloaded.payload
        if len(raw) != item["raw_bytes"] or sha256(raw) != item["raw_sha256"]:
            raise Data296Error(f"{item['source_id']}: immutable raw identity drift")
        extracted, encoding = extract_text(downloaded, item["adapter"])
        bounded = extracted[: item["max_extracted_characters"]]
        normalized, _ = strict_normalize_utf8(bounded)
        normalized_bytes = normalized.encode("utf-8")
        if (
            len(normalized_bytes) != item["normalized_utf8_bytes"]
            or sha256(normalized_bytes) != item["normalized_sha256"]
        ):
            raise Data296Error(f"{item['source_id']}: normalized identity drift")
        lid = current_reviewed_source_lid(normalized, item["mode"])
        if not lid["current_reviewed_source_admitted"]:
            raise Data296Error(f"{item['source_id']}: current reviewed-source LID no longer admits source")
        packs = line_packs(normalized, target)
        source_checks.append(
            {
                "source_id": item["source_id"],
                "family_id": item["family_id"],
                "mode": item["mode"],
                "raw_bytes": len(raw),
                "normalized_utf8_bytes": len(normalized_bytes),
                "normalized_sha256": sha256(normalized_bytes),
                "decoded_encoding": encoding,
                "pack_count": len(packs),
                "lid": lid,
            }
        )
        for idx, text in enumerate(packs):
            records.append(
                {
                    "record_id": f"{item['source_id']}#pack-{idx:04d}",
                    "source_id": item["source_id"],
                    "family_id": item["family_id"],
                    "mode": item["mode"],
                    "text": text,
                    "categories": {
                        "ukrainian_dominant": is_ukrainian_dominant(text),
                        "rst_syntax_bearing": is_rst_syntax_bearing(text),
                        "parse_valid_python_pack": False,
                    },
                }
            )
    return records, source_checks


def acquire_code_objects(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    source_checks: list[dict[str, Any]] = []
    target = plan["audit_partition"]["code_target_chars"]
    for item in plan["code_objects"]:
        downloaded = download(item["url"], max_bytes=250_000)
        raw = downloaded.payload
        if len(raw) != item["raw_bytes"] or git_blob_sha1(raw) != item["git_blob_sha1"]:
            raise Data296Error(f"{item['source_id']}: immutable Git object identity drift")
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise Data296Error(f"{item['source_id']}: code is not strict UTF-8") from exc
        if not is_parse_valid_python(text):
            raise Data296Error(f"{item['source_id']}: admitted Python source no longer parses")
        packs = line_packs(text, target)
        source_checks.append(
            {
                "source_id": item["source_id"],
                "family_id": item["family_id"],
                "mode": "code",
                "raw_bytes": len(raw),
                "raw_sha256": sha256(raw),
                "git_blob_sha1": git_blob_sha1(raw),
                "full_source_parse_valid_python": True,
                "pack_count": len(packs),
            }
        )
        for idx, pack in enumerate(packs):
            records.append(
                {
                    "record_id": f"{item['source_id']}#pack-{idx:04d}",
                    "source_id": item["source_id"],
                    "family_id": item["family_id"],
                    "mode": "code",
                    "text": pack,
                    "categories": {
                        "ukrainian_dominant": False,
                        "rst_syntax_bearing": False,
                        "parse_valid_python_pack": is_parse_valid_python(pack),
                    },
                }
            )
    return records, source_checks


def _unique_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    unique = []
    for record in records:
        digest = sha256(record["text"].encode("utf-8"))
        key = (record["family_id"], digest)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def _fraction(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def audit_policy(
    policy: QualityPolicy,
    role: str,
    records: list[dict[str, Any]],
    source_checks: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    unique_records = _unique_records(records)
    decisions: dict[str, Any] = {}
    for record in unique_records:
        decisions[record["record_id"]] = assess_document(
            record["record_id"], record["text"], record["mode"], policy=policy
        )

    family_rows = []
    for family in sorted({record["family_id"] for record in unique_records}):
        subset = [record for record in unique_records if record["family_id"] == family]
        total_bytes = sum(len(record["text"].encode("utf-8")) for record in subset)
        rejected = [record for record in subset if not decisions[record["record_id"]].accepted]
        rejected_bytes = sum(len(record["text"].encode("utf-8")) for record in rejected)
        reason_counts: Counter[str] = Counter()
        for record in rejected:
            reason_counts.update(decisions[record["record_id"]].reasons)
        family_rows.append(
            {
                "family_id": family,
                "mode": subset[0]["mode"],
                "unique_record_count": len(subset),
                "rejected_unique_records": len(rejected),
                "record_rejection_rate": _fraction(len(rejected), len(subset)),
                "audit_unique_bytes": total_bytes,
                "rejected_unique_bytes": rejected_bytes,
                "byte_rejection_rate": _fraction(rejected_bytes, total_bytes),
                "retained_unique_bytes": total_bytes - rejected_bytes,
                "reason_counts": dict(sorted(reason_counts.items())),
            }
        )

    category_rows = []
    for category in ("ukrainian_dominant", "rst_syntax_bearing", "parse_valid_python_pack"):
        subset = [record for record in unique_records if record["categories"][category]]
        total_bytes = sum(len(record["text"].encode("utf-8")) for record in subset)
        rejected = [record for record in subset if not decisions[record["record_id"]].accepted]
        rejected_bytes = sum(len(record["text"].encode("utf-8")) for record in rejected)
        systematic = (
            len(subset) >= gate["min_category_records"]
            and total_bytes >= gate["min_category_unique_bytes"]
            and _fraction(rejected_bytes, total_bytes) >= gate["rejected_unique_byte_fraction_gte"]
        )
        category_rows.append(
            {
                "category": category,
                "unique_record_count": len(subset),
                "unique_bytes": total_bytes,
                "rejected_unique_records": len(rejected),
                "rejected_unique_bytes": rejected_bytes,
                "rejected_unique_byte_fraction": _fraction(rejected_bytes, total_bytes),
                "systematic_deletion_detected": systematic,
            }
        )

    full_source_sanity = []
    source_map = {item["source_id"]: item for item in source_checks}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[record["source_id"]].append(record)
    for source_id in sorted(by_source):
        text = "".join(record["text"] for record in by_source[source_id])
        check = source_map[source_id]
        decision = assess_document(f"{source_id}#full-source", text, check["mode"], policy=policy)
        full_source_sanity.append(
            {
                "source_id": source_id,
                "family_id": check["family_id"],
                "accepted": decision.accepted,
                "reasons": list(decision.reasons),
                "utf8_bytes": len(text.encode("utf-8")),
                "full_source_parse_valid_python": check.get("full_source_parse_valid_python"),
            }
        )

    return {
        "role": role,
        "policy_id": policy.policy_id,
        "policy_sha256": policy.manifest()["policy_sha256"],
        "families": family_rows,
        "diagnostic_categories": category_rows,
        "full_source_sanity": full_source_sanity,
    }


def comparison_rows(policy_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    incumbent = next(item for item in policy_reports if item["role"] == "INCUMBENT")
    inc = {row["family_id"]: row for row in incumbent["families"]}
    rows = []
    for report in policy_reports:
        if report["role"] == "INCUMBENT":
            continue
        for row in report["families"]:
            base = inc[row["family_id"]]
            rows.append(
                {
                    "role": report["role"],
                    "policy_id": report["policy_id"],
                    "family_id": row["family_id"],
                    "retained_unique_bytes_delta_vs_incumbent": row["retained_unique_bytes"]
                    - base["retained_unique_bytes"],
                    "record_rejection_rate_delta_vs_incumbent": round(
                        row["record_rejection_rate"] - base["record_rejection_rate"], 6
                    ),
                    "byte_rejection_rate_delta_vs_incumbent": round(
                        row["byte_rejection_rate"] - base["byte_rejection_rate"], 6
                    ),
                }
            )
    return rows


def run(repo: Path, plan_path: Path, output_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    verify_preregistration(plan, repo)
    policies = [(raw["role"], policy_from_plan(raw)) for raw in plan["policies"]]
    text_records, text_checks = acquire_text_objects(plan)
    code_records, code_checks = acquire_code_objects(plan)
    records = text_records + code_records
    source_checks = text_checks + code_checks

    for source in source_checks:
        source_records = [record for record in records if record["source_id"] == source["source_id"]]
        partition_bytes = len("".join(record["text"] for record in source_records).encode("utf-8"))
        expected = source.get("normalized_utf8_bytes", source["raw_bytes"])
        if partition_bytes != expected:
            raise Data296Error(f"{source['source_id']}: audit partition byte mismatch")

    gate = plan["audit_partition"]["systematic_deletion_gate"]
    policy_reports = [
        audit_policy(policy, role, records, source_checks, gate)
        for role, policy in policies
    ]
    report_core = {
        "schema_version": SCHEMA,
        "worker_id": plan["worker_id"],
        "local_free_only": True,
        "preregistration_sha256": plan["preregistration_sha256"],
        "authority_boundary": plan["authority_boundary"],
        "source_checks": source_checks,
        "policy_reports": policy_reports,
        "comparisons_vs_incumbent": comparison_rows(policy_reports),
        "policy_selection": "FORBIDDEN_NOT_PERFORMED",
        "model_results_read": False,
        "final_test_outcomes_read": False,
        "scope_note": (
            "Statistics are computed on deterministic no-overlap line packs that preserve each "
            "admitted source byte-for-byte after the exact admitted normalization. Full-source "
            "sanity decisions are also reported. This audit does not promote a policy."
        ),
    }
    report = {**report_core, "report_sha256": sha256(cjson(report_core))}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("DATA296_REPORT_SHA256=" + report["report_sha256"])
    print("DATA296_PREREGISTRATION_SHA256=" + report["preregistration_sha256"])
    for policy_report in policy_reports:
        print(
            "DATA296_POLICY="
            + json.dumps(
                {
                    "role": policy_report["role"],
                    "policy_id": policy_report["policy_id"],
                    "policy_sha256": policy_report["policy_sha256"],
                    "families": policy_report["families"],
                    "diagnostic_categories": policy_report["diagnostic_categories"],
                    "full_source_sanity": policy_report["full_source_sanity"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    for source in source_checks:
        print(
            "DATA296_SOURCE="
            + json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--plan", default="configs/data/data296_quality_filter_external_audit_v1.json"
    )
    parser.add_argument(
        "--output", default="data296-evidence/data296-quality-filter-external-audit.json"
    )
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    run(repo, repo / args.plan, repo / args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
