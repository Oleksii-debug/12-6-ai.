#!/usr/bin/env python3
"""Fail-closed, zero-credit Derzhgeocadastre candidate materializer.

Selective current-main successor to NEXT100-025 / PR #471. Source locking proves
identity only; it can never grant corpus/training authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

CONFIG = Path("configs/data/d03_derzhgeocadastre_zero_credit_v1.json")
SCHEMA = "12-6.d03-derzhgeocadastre-zero-credit.v1"
REPORT_SCHEMA = "12-6.d03-derzhgeocadastre-zero-credit-report.v1"
FAMILY = "ua.data-gov.derzhgeocadastre.dataset-register"
DATASET_ID = "644b424e-9884-43cb-8204-0214657f2248"
PUBLISHER = "Державна служба України з питань геодезії, картографії та кадастру"
TITLE = "Реєстр наборів даних, що перебувають у володінні розпорядника інформації"
INCUMBENT_HEAD = "73fc25ea9d30756af7410ba52462d0b4faa9e11b"
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-яІіЇїЄєҐґ]{2,}")
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?38[\s().-]*)?0\d{2}[\s().-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)"
)
LONG_ID_RE = re.compile(r"(?<!\d)\d{10,}(?!\d)")
URL_RE = re.compile(r"https?://\S+", re.I)
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ'’\-]+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
UA_LEXEMES = (
    "держав", "дан", "інформац", "набір", "реєстр", "україн",
    "оновлен", "розпоряд", "публіч", "норматив", "послуг",
)
ZERO_KEYS = (
    "canonical_capacity_credited",
    "family_credit_added",
    "training_authorized_bytes",
    "optimizer_updates",
)
FALSE_KEYS = (
    "training_eligible",
    "evaluation_eligible",
    "tokenizer_fit_authorized",
    "model_training_executed",
    "final_test_accessed",
    "paid_compute_used",
)
SAFE_ALLOW = (
    "title", "name", "description", "desc", "назв", "опис", "keyword", "ключ",
    "frequency", "period", "частот", "purpose", "basis", "підстав", "признач",
    "format", "theme", "category", "категор",
)
SAFE_DENY = (
    "email", "mail", "phone", "tel", "contact", "person", "responsible",
    "address", "відповід", "телефон", "пошта", "адрес",
)
QUALITY_POLICY = {
    "min_accepted_records": 12,
    "min_record_utf8_bytes": 80,
    "min_total_normalized_utf8_bytes": 12000,
    "min_cyrillic_alpha_ratio": 0.78,
    "min_uk_specific_chars": 20,
    "min_uk_lexical_hits": 3,
    "near_duplicate_5token_jaccard": 0.92,
}
REQUIRED_GATES = (
    "CURRENT_GLOBAL_EXACT_NEAR_LINEAGE_DEDUP",
    "FRESH_RESERVED_EVALUATION_DECONTAMINATION",
    "POST_COMPOSITION_QUALITY_PRIVACY",
    "BALANCE_AND_FAMILY_CAPS",
    "CLUSTER_SAFE_SPLIT",
    "DETERMINISTIC_PACKING_TWO_CLEAN_BUILDS",
    "POSITIVE_UNIQUE_LOSS_LEDGER",
)


class CandidateError(RuntimeError):
    """Fail-closed candidate materialization error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateError(f"cannot load config: {path}") from exc
    require(isinstance(cfg, dict), "config root must be object")
    require(cfg.get("schema_version") == SCHEMA, "schema drift")
    require(cfg.get("execution_profile") == "LOCAL_FREE", "LOCAL_FREE boundary drift")
    require(cfg["incumbent_authority"]["pr"] == 471, "incumbent PR drift")
    require(cfg["incumbent_authority"]["head_sha"] == INCUMBENT_HEAD, "incumbent head drift")
    require(cfg["dataset"]["dataset_id"] == DATASET_ID, "dataset id drift")
    require(cfg["dataset"]["publisher"] == PUBLISHER, "publisher drift")
    require(cfg["dataset"]["title"] == TITLE, "dataset title drift")
    require(cfg["family"]["family_id"] == FAMILY, "family drift")
    require(
        cfg["dataset"]["dataset_page"]
        == f"https://data.gov.ua/dataset/{DATASET_ID}",
        "dataset page drift",
    )
    require(
        cfg["dataset"]["package_api"]
        == f"https://data.gov.ua/api/3/action/package_show?id={DATASET_ID}",
        "package API drift",
    )
    selection = cfg["resource_selection"]
    require(selection["allowed_formats"] == ["json"], "resource format policy drift")
    require(
        selection["exclude_name_fragments"]
        == ["structure", "структур", "працівник", "employee"],
        "resource exclusion policy drift",
    )
    require(
        selection["prefer_name_fragments"] == ["register", "реєстр"],
        "resource preference policy drift",
    )
    require(selection["max_download_bytes"] == 10_000_000, "download envelope drift")
    require(
        set(selection["lock"])
        == {
            "resource_id", "resource_url", "resource_last_modified", "raw_sha256",
            "raw_bytes", "normalized_sha256", "normalized_utf8_bytes",
        },
        "source lock schema drift",
    )
    require(
        tuple(cfg["safe_text"]["allow_key_fragments"]) == SAFE_ALLOW,
        "safe-text allowlist drift",
    )
    require(
        tuple(cfg["safe_text"]["deny_key_fragments"]) == SAFE_DENY,
        "safe-text denylist drift",
    )
    require(cfg["quality"] == QUALITY_POLICY, "quality policy drift")
    rights = cfg["rights"]
    require(
        rights["model_training_purpose_rights"] == "ALLOWED_WITH_ATTRIBUTION",
        "rights purpose drift",
    )
    require(rights["evaluation"] == "NOT_ADMITTED", "evaluation rights drift")
    evidence = Path(rights["rights_evidence_path"])
    try:
        evidence_hash = sha256(evidence.read_bytes())
    except OSError as exc:
        raise CandidateError("rights evidence unavailable") from exc
    require(evidence_hash == rights["rights_evidence_sha256"], "rights evidence drift")

    boundary = cfg["claim_boundary"]
    require(boundary.get("candidate_only") is True, "candidate-only boundary disabled")
    for key in ZERO_KEYS:
        require(boundary.get(key) == 0, f"zero-credit boundary drift: {key}")
    for key in FALSE_KEYS:
        require(boundary.get(key) is False, f"false boundary drift: {key}")
    require(
        tuple(cfg["required_downstream_gates"]) == REQUIRED_GATES,
        "required downstream gates drift",
    )
    return cfg


def decode_json(payload: bytes) -> Any:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return json.loads(payload.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise CandidateError("resource is not decodable JSON")


def validate_package(package: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    require(package.get("name") == DATASET_ID, "dataset name drift")
    require(package.get("title") == TITLE, "dataset title drift")
    organization = package.get("organization")
    require(isinstance(organization, Mapping), "publisher metadata missing")
    require(organization.get("title") == PUBLISHER, "publisher mismatch")
    license_title = str(package.get("license_title") or "")
    require(
        "creative commons attribution" in license_title.casefold(),
        "dataset license mismatch",
    )


def pick_resource(package: Mapping[str, Any], cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    resources = package.get("resources")
    require(isinstance(resources, list), "resource list missing")
    lock = cfg["resource_selection"]["lock"]
    if lock["resource_id"] is not None:
        matches = [
            item for item in resources
            if isinstance(item, Mapping) and item.get("id") == lock["resource_id"]
        ]
        require(len(matches) == 1, "locked resource missing or ambiguous")
        return matches[0]

    allowed = {
        str(value).casefold().lstrip(".")
        for value in cfg["resource_selection"]["allowed_formats"]
    }
    excluded = [
        str(value).casefold()
        for value in cfg["resource_selection"]["exclude_name_fragments"]
    ]
    preferred = [
        str(value).casefold()
        for value in cfg["resource_selection"]["prefer_name_fragments"]
    ]
    candidates: list[tuple[int, str, str, Mapping[str, Any]]] = []
    for resource in resources:
        if not isinstance(resource, Mapping):
            continue
        fmt = str(resource.get("format") or "").casefold().lstrip(".")
        name = str(resource.get("name") or "")
        url = str(resource.get("url") or "")
        if fmt not in allowed or not url:
            continue
        folded = name.casefold()
        if any(fragment in folded for fragment in excluded):
            continue
        preference = int(any(fragment in folded for fragment in preferred))
        stamp = str(resource.get("last_modified") or resource.get("created") or "")
        candidates.append((preference, stamp, name, resource))
    require(bool(candidates), "no admissible JSON resource candidate")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def validate_resource_url(url: str) -> None:
    parsed = urlparse(url)
    require(parsed.scheme == "https", "resource URL must be HTTPS")
    require(parsed.hostname in {"data.gov.ua", "www.data.gov.ua"}, "resource host drift")
    require(not parsed.username and not parsed.password, "credential-bearing URL rejected")


def candidate_record_lists(value: Any) -> list[list[Mapping[str, Any]]]:
    output: list[list[Mapping[str, Any]]] = []
    if isinstance(value, list):
        rows = [item for item in value if isinstance(item, Mapping)]
        if rows:
            output.append(rows)
        for item in value:
            output.extend(candidate_record_lists(item))
    elif isinstance(value, Mapping):
        for item in value.values():
            output.extend(candidate_record_lists(item))
    return output


def flatten_scalars(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.extend(flatten_scalars(item, path))
    elif isinstance(value, list):
        scalars = [
            str(item).strip()
            for item in value
            if isinstance(item, (str, int, float, bool)) and str(item).strip()
        ]
        if scalars:
            output.append((prefix, "; ".join(scalars)))
        for index, item in enumerate(value):
            if isinstance(item, (Mapping, list)):
                output.extend(flatten_scalars(item, f"{prefix}[{index}]"))
    elif isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        if text:
            output.append((prefix, text))
    return output


def normalize_scalar(value: str) -> str:
    value = unicodedata.normalize("NFC", value.replace("\xa0", " "))
    value = " ".join(value.split())
    return URL_RE.sub("", value).strip(" ;,")


def safe_record_text(
    record: Mapping[str, Any], cfg: Mapping[str, Any]
) -> tuple[str, dict[str, int]]:
    allow = [value.casefold() for value in cfg["safe_text"]["allow_key_fragments"]]
    deny = [value.casefold() for value in cfg["safe_text"]["deny_key_fragments"]]
    stats = {"excluded_key": 0, "email": 0, "phone": 0, "long_id": 0, "control": 0}
    kept: list[str] = []
    for key, raw in flatten_scalars(record):
        folded_key = key.casefold()
        if any(fragment in folded_key for fragment in deny):
            stats["excluded_key"] += 1
            continue
        if not any(fragment in folded_key for fragment in allow):
            continue
        value = normalize_scalar(raw)
        if not value:
            continue
        if CONTROL_RE.search(value):
            stats["control"] += 1
            continue
        if EMAIL_RE.search(value):
            stats["email"] += 1
            continue
        if PHONE_RE.search(value):
            stats["phone"] += 1
            continue
        if LONG_ID_RE.search(value):
            stats["long_id"] += 1
            continue
        kept.append(f"{key}: {value}")
    text = "\n".join(kept).strip()
    return (text + "\n" if text else ""), stats


def language_evidence(text: str) -> dict[str, int | float]:
    letters = [char for char in text if char.isalpha()]
    cyr = [char for char in letters if "\u0400" <= char <= "\u052f"]
    lowered = text.casefold()
    return {
        "alpha_chars": len(letters),
        "cyrillic_alpha_ratio": round(len(cyr) / len(letters), 6) if letters else 0.0,
        "uk_specific_chars": sum(lowered.count(char) for char in "іїєґ"),
        "uk_lexical_hits": sum(1 for stem in UA_LEXEMES if stem in lowered),
    }


def shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    words = [match.group(0).casefold() for match in WORD_RE.finditer(text)]
    return {
        tuple(words[index:index + size])
        for index in range(max(0, len(words) - size + 1))
    }


def jaccard(left: set[Any], right: set[Any]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _locked(lock: Mapping[str, Any]) -> bool:
    values = list(lock.values())
    require(
        all(value is None for value in values) or all(value is not None for value in values),
        "partial source lock rejected",
    )
    return all(value is not None for value in values)


def materialize_from_bytes(
    package_response_bytes: bytes,
    resource_bytes: bytes,
    cfg: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    package_response = decode_json(package_response_bytes)
    require(
        isinstance(package_response, Mapping) and package_response.get("success") is True,
        "CKAN package_show did not succeed",
    )
    package = package_response.get("result")
    require(isinstance(package, Mapping), "CKAN package result missing")
    validate_package(package, cfg)
    resource = pick_resource(package, cfg)
    resource_url = str(resource.get("url") or "")
    validate_resource_url(resource_url)
    require(
        len(resource_bytes) <= cfg["resource_selection"]["max_download_bytes"],
        "resource exceeds byte envelope",
    )
    root = decode_json(resource_bytes)
    lists = candidate_record_lists(root)
    require(bool(lists), "no record list found")
    records = max(lists, key=len)

    accepted: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_shingles: list[set[tuple[str, ...]]] = []
    rejected: dict[str, int] = {
        "no_safe_text": 0, "too_short": 0, "language": 0,
        "exact_duplicate": 0, "near_duplicate": 0,
    }
    scalar_rejections = {
        "excluded_key": 0, "email": 0, "phone": 0, "long_id": 0, "control": 0
    }
    quality = cfg["quality"]
    for ordinal, record in enumerate(records):
        text, stats = safe_record_text(record, cfg)
        for key, count in stats.items():
            scalar_rejections[key] += count
        if not text:
            rejected["no_safe_text"] += 1
            continue
        payload = text.encode("utf-8")
        if len(payload) < quality["min_record_utf8_bytes"]:
            rejected["too_short"] += 1
            continue
        lang = language_evidence(text)
        if lang["cyrillic_alpha_ratio"] < 0.68 or lang["uk_specific_chars"] < 1:
            rejected["language"] += 1
            continue
        digest = sha256(payload)
        if digest in seen_hashes:
            rejected["exact_duplicate"] += 1
            continue
        current = shingles(text)
        if any(
            jaccard(current, prior) > quality["near_duplicate_5token_jaccard"]
            for prior in seen_shingles
        ):
            rejected["near_duplicate"] += 1
            continue
        seen_hashes.add(digest)
        seen_shingles.append(current)
        accepted.append({
            "record_id": f"{FAMILY}:{ordinal}",
            "normalized_sha256": digest,
            "normalized_utf8_bytes": len(payload),
            "text": text,
            "training_eligible": False,
            "evaluation_eligible": False,
        })

    require(
        len(accepted) >= quality["min_accepted_records"],
        "substantiality record gate failed",
    )
    aggregate = "\n---\n".join(row["text"].rstrip() for row in accepted).strip() + "\n"
    aggregate_bytes = aggregate.encode("utf-8")
    require(
        len(aggregate_bytes) >= quality["min_total_normalized_utf8_bytes"],
        "substantiality byte gate failed",
    )
    aggregate_lang = language_evidence(aggregate)
    require(
        aggregate_lang["cyrillic_alpha_ratio"] >= quality["min_cyrillic_alpha_ratio"]
        and aggregate_lang["uk_specific_chars"] >= quality["min_uk_specific_chars"]
        and aggregate_lang["uk_lexical_hits"] >= quality["min_uk_lexical_hits"],
        "aggregate Ukrainian language gate failed",
    )

    lock_values = {
        "resource_id": resource.get("id"),
        "resource_url": resource_url,
        "resource_last_modified": resource.get("last_modified"),
        "raw_sha256": sha256(resource_bytes),
        "raw_bytes": len(resource_bytes),
        "normalized_sha256": sha256(aggregate_bytes),
        "normalized_utf8_bytes": len(aggregate_bytes),
    }
    expected_lock = cfg["resource_selection"]["lock"]
    locked = _locked(expected_lock)
    if locked:
        require(dict(expected_lock) == lock_values, "locked source identity mismatch")

    candidate_payload = b"".join(canonical(row) for row in accepted)
    boundary = dict(cfg["claim_boundary"])
    report = {
        "schema_version": REPORT_SCHEMA,
        "source_lock_state": "LOCKED_ZERO_CREDIT" if locked else "PROBE_LOCK_REQUIRED",
        "source_family": FAMILY,
        "source_records": len(records),
        "retained_records": len(accepted),
        "retained_normalized_utf8_bytes": len(aggregate_bytes),
        "candidate_jsonl_sha256": sha256(candidate_payload),
        "resource_identity": lock_values,
        "aggregate_language": aggregate_lang,
        "record_dispositions": rejected,
        "privacy_scalar_rejections": scalar_rejections,
        "rights_evidence_sha256": cfg["rights"]["rights_evidence_sha256"],
        "model_training_purpose_rights": cfg["rights"]["model_training_purpose_rights"],
        "evaluation_authority": "NOT_ADMITTED",
        "claim_boundary": boundary,
        "required_downstream_gates": list(REQUIRED_GATES),
        "global_dedup_completed": False,
        "reserved_evaluation_decontamination_completed": False,
        "training_authorized": False,
    }
    report["report_identity_sha256"] = sha256(canonical(report))
    return candidate_payload, report


def fetch_bounded(url: str, max_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "12-6-ai-D03-Derzhgeocadastre/1 (bounded LOCAL_FREE qualification)",
            "Accept": "application/json,text/plain;q=0.9,*/*;q=0.1",
            "Cache-Control": "no-cache",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read(max_bytes + 1)
    except OSError as exc:
        raise CandidateError(f"network acquisition failed: {url}") from exc
    require(len(payload) <= max_bytes, f"response exceeds max bytes: {url}")
    return payload


def execute_network(cfg: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    package_a = fetch_bounded(cfg["dataset"]["package_api"], 2_000_000)
    package_b = fetch_bounded(cfg["dataset"]["package_api"], 2_000_000)
    require(package_a == package_b, "repeat package metadata acquisition differs")
    package = decode_json(package_a)
    require(isinstance(package, Mapping) and package.get("success") is True, "bad package")
    result = package.get("result")
    require(isinstance(result, Mapping), "package result missing")
    resource = pick_resource(result, cfg)
    url = str(resource.get("url") or "")
    validate_resource_url(url)
    limit = cfg["resource_selection"]["max_download_bytes"]
    raw_a = fetch_bounded(url, limit)
    time.sleep(0.4)
    raw_b = fetch_bounded(url, limit)
    require(raw_a == raw_b, "repeat resource acquisition differs")
    return materialize_from_bytes(package_a, raw_a, cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    candidate, report = execute_network(cfg)
    args.candidate_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_jsonl.write_bytes(candidate)
    args.report.write_bytes(canonical(report))
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
