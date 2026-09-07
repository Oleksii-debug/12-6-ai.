#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-яІіЇїЄєҐґ]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?38[\s().-]*)?0\d{2}[\s().-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)")
LONG_ID_RE = re.compile(r"(?<!\d)\d{10,}(?!\d)")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ'’\-]+", re.UNICODE)
UA_LEXEMES = ("держав", "дан", "інформац", "набір", "реєстр", "україн", "оновлен", "розпоряд", "публіч", "норматив", "послуг")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: object) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def fetch(url: str, max_bytes: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "12-6-ai-NEXT100-025/1.0 (bounded open-data snapshot)",
            "Accept": "application/json,text/plain;q=0.9,*/*;q=0.1",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise RuntimeError(f"response exceeds max bytes: {url}")
    return payload


def load_json_bytes(payload: bytes) -> object:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return json.loads(payload.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise RuntimeError("resource is not decodable JSON")


def pick_resource(package: dict, cfg: dict) -> dict:
    resources = package.get("resources") or []
    expected_id = cfg["resource_selection"].get("expected_resource_id")
    if expected_id:
        matches = [resource for resource in resources if resource.get("id") == expected_id]
        if len(matches) != 1:
            raise RuntimeError(f"locked resource id missing or ambiguous: {expected_id}")
        return matches[0]

    allowed = {value.casefold().lstrip(".") for value in cfg["resource_selection"]["allowed_formats"]}
    excluded = [value.casefold() for value in cfg["resource_selection"]["exclude_name_fragments"]]
    preferred = [value.casefold() for value in cfg["resource_selection"]["prefer_name_fragments"]]
    candidates = []
    for resource in resources:
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
    if not candidates:
        raise RuntimeError("no admissible JSON resource candidate")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates[0][3]


def candidate_record_lists(value: object) -> list[list[dict]]:
    out: list[list[dict]] = []
    if isinstance(value, list):
        dicts = [item for item in value if isinstance(item, dict)]
        if dicts:
            out.append(dicts)
        for item in value:
            out.extend(candidate_record_lists(item))
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(candidate_record_lists(item))
    return out


def flatten_scalars(value: object, prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.extend(flatten_scalars(item, path))
    elif isinstance(value, list):
        scalar_items = [str(item) for item in value if isinstance(item, (str, int, float, bool)) and str(item).strip()]
        if scalar_items:
            out.append((prefix, "; ".join(scalar_items)))
        for index, item in enumerate(value):
            if isinstance(item, (dict, list)):
                out.extend(flatten_scalars(item, f"{prefix}[{index}]"))
    elif isinstance(value, (str, int, float, bool)):
        text = str(value).strip()
        if text:
            out.append((prefix, text))
    return out


def language_evidence(text: str, cfg: dict) -> dict:
    letters = [char for char in text if char.isalpha()]
    cyr = [char for char in letters if "\u0400" <= char <= "\u052f"]
    ratio = len(cyr) / len(letters) if letters else 0.0
    lowered = text.casefold()
    return {
        "alpha_chars": len(letters),
        "cyrillic_alpha_ratio": round(ratio, 6),
        "uk_specific_chars": sum(lowered.count(char) for char in "іїєґ"),
        "uk_lexical_hits": sum(1 for stem in UA_LEXEMES if stem in lowered),
    }


def shingle_set(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = [match.group(0).casefold() for match in WORD_RE.finditer(text)]
    if len(words) < n:
        return set()
    return {tuple(words[index:index+n]) for index in range(len(words) - n + 1)}


def jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def safe_record_text(record: dict, cfg: dict) -> tuple[str, dict]:
    safe_fragments = [value.casefold() for value in cfg["safe_text_key_fragments"]]
    excluded = [value.casefold() for value in cfg["privacy"]["exclude_key_fragments"]]
    kept: list[str] = []
    rejected_scalars = {"email": 0, "phone": 0, "long_numeric_identifier": 0, "excluded_key": 0}
    for key, raw_value in flatten_scalars(record):
        folded_key = key.casefold()
        if any(fragment in folded_key for fragment in excluded):
            rejected_scalars["excluded_key"] += 1
            continue
        if not any(fragment in folded_key for fragment in safe_fragments):
            continue
        value = unicodedata.normalize("NFC", " ".join(raw_value.replace("\xa0", " ").split()))
        value = URL_RE.sub("", value).strip(" ;,")
        if not value:
            continue
        if EMAIL_RE.search(value):
            rejected_scalars["email"] += 1
            continue
        if PHONE_RE.search(value):
            rejected_scalars["phone"] += 1
            continue
        if LONG_ID_RE.search(value):
            rejected_scalars["long_numeric_identifier"] += 1
            continue
        kept.append(f"{key}: {value}")
    text = "\n".join(kept).strip()
    return (text + "\n" if text else ""), rejected_scalars


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if cfg["local_free_only"] is not True:
        raise RuntimeError("LOCAL_FREE gate is not true")
    rights_path = Path(cfg["rights_evidence"]["path"])
    actual_rights_sha = sha256(rights_path.read_bytes())
    if actual_rights_sha != cfg["rights_evidence"]["sha256"]:
        raise RuntimeError("rights evidence SHA mismatch")
    required = ("acquisition", "storage", "analysis", "model_training")
    if any(cfg["rights"]["uses"].get(use) != "ALLOWED" for use in required):
        raise RuntimeError("required rights are not all ALLOWED")
    if cfg["rights"]["uses"].get("redistribution") != "ALLOWED_WITH_ATTRIBUTION":
        raise RuntimeError("redistribution attribution boundary changed")
    if cfg["rights"]["uses"].get("evaluation") != "NOT_ADMITTED":
        raise RuntimeError("evaluation boundary weakened")

    package_payload = fetch(cfg["dataset"]["package_api"], 2_000_000)
    package_response = load_json_bytes(package_payload)
    if not isinstance(package_response, dict) or package_response.get("success") is not True:
        raise RuntimeError("CKAN package_show did not succeed")
    package = package_response.get("result")
    if not isinstance(package, dict):
        raise RuntimeError("CKAN package result missing")
    if package.get("id") != cfg["dataset"]["dataset_id"]:
        raise RuntimeError("dataset identity mismatch")
    organization = package.get("organization") or {}
    org_title = organization.get("title") if isinstance(organization, dict) else None
    if org_title != cfg["dataset"]["publisher"]:
        raise RuntimeError(f"publisher mismatch: {org_title!r}")
    license_title = str(package.get("license_title") or "")
    if "creative commons attribution" not in license_title.casefold():
        raise RuntimeError(f"dataset license mismatch: {license_title!r}")

    resource = pick_resource(package, cfg)
    resource_url = str(resource.get("url") or "")
    parsed = urlparse(resource_url)
    if parsed.scheme != "https" or parsed.hostname not in {"data.gov.ua", "www.data.gov.ua"}:
        raise RuntimeError(f"resource escaped data.gov.ua boundary: {resource_url}")

    max_bytes = cfg["resource_selection"]["max_download_bytes"]
    raw_a = fetch(resource_url, max_bytes)
    time.sleep(0.4)
    raw_b = fetch(resource_url, max_bytes)
    if raw_a != raw_b:
        raise RuntimeError("repeat acquisition raw bytes differ")
    raw_hash = sha256(raw_a)
    root = load_json_bytes(raw_a)

    lists = candidate_record_lists(root)
    if not lists:
        raise RuntimeError("no record list found in JSON resource")
    records = max(lists, key=len)

    accepted: list[dict] = []
    normalized_seen: set[str] = set()
    shingles: list[set] = []
    rejected = {"too_short": 0, "language": 0, "exact_duplicate": 0, "near_duplicate": 0, "no_safe_text": 0}
    scalar_rejections = {"email": 0, "phone": 0, "long_numeric_identifier": 0, "excluded_key": 0}

    for ordinal, record in enumerate(records):
        text, scalar_stats = safe_record_text(record, cfg)
        for key, value in scalar_stats.items():
            scalar_rejections[key] += value
        if not text:
            rejected["no_safe_text"] += 1
            continue
        encoded = text.encode("utf-8")
        if len(encoded) < cfg["quality"]["min_record_utf8_bytes"]:
            rejected["too_short"] += 1
            continue
        lang = language_evidence(text, cfg["language"])
        if lang["cyrillic_alpha_ratio"] < 0.68 or lang["uk_specific_chars"] < 1:
            rejected["language"] += 1
            continue
        record_hash = sha256(encoded)
        if record_hash in normalized_seen:
            rejected["exact_duplicate"] += 1
            continue
        current_shingles = shingle_set(text)
        if any(jaccard(current_shingles, prior) > cfg["dedup"]["intra_family_near_duplicate_5token_jaccard"] for prior in shingles):
            rejected["near_duplicate"] += 1
            continue
        normalized_seen.add(record_hash)
        shingles.append(current_shingles)
        accepted.append({"ordinal": ordinal, "normalized_sha256": record_hash, "normalized_utf8_bytes": len(encoded), "text": text, "language": lang})

    aggregate_text = "\n---\n".join(item["text"].rstrip() for item in accepted).strip() + "\n"
    aggregate_bytes = aggregate_text.encode("utf-8")
    aggregate_hash = sha256(aggregate_bytes)
    aggregate_lang = language_evidence(aggregate_text, cfg["language"])
    lang_pass = (
        aggregate_lang["cyrillic_alpha_ratio"] >= cfg["language"]["min_cyrillic_alpha_ratio"]
        and aggregate_lang["uk_specific_chars"] >= cfg["language"]["min_uk_specific_chars"]
        and aggregate_lang["uk_lexical_hits"] >= cfg["language"]["min_uk_lexical_hits"]
    )

    if len(accepted) < cfg["quality"]["min_accepted_records"]:
        raise RuntimeError(f"substantiality record gate failed: accepted={len(accepted)} rejected={rejected}")
    if len(aggregate_bytes) < cfg["quality"]["min_total_normalized_utf8_bytes"]:
        raise RuntimeError(f"substantiality byte gate failed: bytes={len(aggregate_bytes)}")
    if not lang_pass:
        raise RuntimeError(f"aggregate Ukrainian language gate failed: {aggregate_lang}")
    if aggregate_hash in cfg["dedup"]["cross_family_normalized_hashes"]:
        raise RuntimeError("cross-family normalized exact duplicate")

    lock_values = {
        "expected_resource_id": resource.get("id"),
        "expected_resource_url": resource_url,
        "expected_resource_last_modified": resource.get("last_modified"),
        "expected_raw_sha256": raw_hash,
        "expected_raw_bytes": len(raw_a),
        "expected_normalized_sha256": aggregate_hash,
        "expected_normalized_utf8_bytes": len(aggregate_bytes),
    }
    if cfg["mode"] == "LOCKED":
        for field, actual in lock_values.items():
            if cfg["resource_selection"].get(field) != actual:
                raise RuntimeError(f"locked identity mismatch {field}: expected={cfg['resource_selection'].get(field)!r} actual={actual!r}")

    raw_dir = output / "snapshots" / "sha256" / raw_hash
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "payload").write_bytes(raw_a)
    (output / "normalized.txt").write_bytes(aggregate_bytes)

    train_rows = []
    for item in accepted:
        train_rows.append({
            "source_id": cfg["family"]["family_id"],
            "source_version": resource.get("id"),
            "source_url": resource_url,
            "dataset_id": cfg["dataset"]["dataset_id"],
            "language": "uk",
            "license": cfg["rights"]["dataset_license_label"],
            "attribution_required": True,
            "raw_sha256": raw_hash,
            "normalized_sha256": item["normalized_sha256"],
            "text": item["text"],
            "training_eligible": cfg["mode"] == "LOCKED",
            "evaluation_eligible": False,
        })
    with (output / "train.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "schema_version": "12-6.next100-025-data-gov-snapshot-report.v1",
        "worker": cfg["worker"],
        "status": "PASS" if cfg["mode"] == "LOCKED" else "PROBE_LOCK_REQUIRED",
        "mode": cfg["mode"],
        "local_free_only": True,
        "family": cfg["family"],
        "dataset": {
            "id": package.get("id"),
            "name": package.get("name"),
            "title": package.get("title"),
            "metadata_modified": package.get("metadata_modified"),
            "publisher": org_title,
            "license_title": license_title,
        },
        "resource": {
            "id": resource.get("id"),
            "name": resource.get("name"),
            "format": resource.get("format"),
            "url": resource_url,
            "created": resource.get("created"),
            "last_modified": resource.get("last_modified"),
            "raw_sha256": raw_hash,
            "raw_bytes": len(raw_a),
        },
        "normalized": {
            "sha256": aggregate_hash,
            "utf8_bytes": len(aggregate_bytes),
            "accepted_records": len(accepted),
            "source_record_count": len(records),
            "rejected_records": rejected,
        },
        "rights": cfg["rights"],
        "rights_evidence_sha256": actual_rights_sha,
        "language": {**aggregate_lang, "passed": lang_pass},
        "privacy": {
            "policy": "safe-key allowlist plus scalar exclusion for contacts/PII-like values",
            "excluded_scalars": scalar_rejections,
            "passed": True,
        },
        "dedup": {
            "accepted_unique_normalized_records": len(normalized_seen),
            "near_duplicate_threshold": cfg["dedup"]["intra_family_near_duplicate_5token_jaccard"],
            "cross_family_reference": cfg["dedup"]["cross_family_reference"],
            "cross_family_exact_normalized_exclusions": cfg["dedup"]["cross_family_normalized_hashes"],
        },
        "attribution": {
            "required": True,
            "template": cfg["rights"]["attribution_template"],
            "changes": "Selected safe administrative text fields only; contacts and PII-like scalars excluded; URLs removed from text; Unicode NFC; whitespace normalized; records deduplicated.",
        },
        "lock_values": lock_values,
        "evaluation_authority": "NOT_ADMITTED",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")

    files = []
    for path in sorted(path for path in output.rglob("*") if path.is_file() and path.name != "artifact-manifest.json"):
        data = path.read_bytes()
        files.append({"path": path.relative_to(output).as_posix(), "sha256": sha256(data), "size_bytes": len(data)})
    manifest_core = {
        "schema_version": "12-6.next100-025-artifact-manifest.v1",
        "dataset_id": cfg["dataset"]["dataset_id"],
        "resource_id": resource.get("id"),
        "raw_sha256": raw_hash,
        "normalized_sha256": aggregate_hash,
        "files": files,
    }
    manifest = {**manifest_core, "manifest_sha256": sha256(canonical_json(manifest_core))}
    (output / "artifact-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
