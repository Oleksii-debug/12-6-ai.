#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import unicodedata
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

AUTHOR_MARKER = "Департамент інформації та комунікацій з громадськістю Секретаріату Кабінету Міністрів України"
RIGHTS_MARKERS = (
    "Весь контент доступний за ліцензією",
    "Creative Commons Attribution 4.0 International license",
    "якщо не зазначено інше",
)
STOP_MARKERS = (
    "Попередня", "Наступна", "За темою", "За темами", "Мапа порталу",
    "Власність Секретаріату Кабінету Міністрів України",
)
CONTRARY_MARKERS = (
    "all rights reserved", "усі права захищено", "всі права захищено",
    "не дозволяється копіювання",
)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-zА-Яа-яІіЇїЄєҐґ]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?38[\s().-]*)?0\d{2}[\s().-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)")
LONG_ID_RE = re.compile(r"(?<!\d)\d{10,}(?!\d)")
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яІіЇїЄєҐґ'’\-]+", re.UNICODE)
UA_LEXEMES = ("україн", "уряд", "держав", "громад", "кабінет", "міністр", "політик", "послуг", "бюджет", "рішен")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: object) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.title_depth = 0
        self.title_parts: list[str] = []
        self.nodes: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "svg", "noscript", "template"}:
            self.skip_depth += 1
        if tag == "title":
            self.title_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "svg", "noscript", "template"} and self.skip_depth:
            self.skip_depth -= 1
        if tag == "title" and self.title_depth:
            self.title_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        value = " ".join(html.unescape(data).replace("\xa0", " ").split())
        if not value:
            return
        if self.title_depth:
            self.title_parts.append(value)
        self.nodes.append(value)


def fetch(url: str, max_bytes: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "12-6-ai-NEXT100-025/1.0 (bounded research snapshot)",
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
            "Accept-Language": "uk,en;q=0.5",
            "Cache-Control": "no-cache",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type.lower():
            raise RuntimeError(f"non-HTML content type for {url}: {content_type!r}")
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise RuntimeError(f"document exceeds max bytes: {url}")
    return payload


def parse_visible(raw: bytes) -> tuple[str, list[str]]:
    text = raw.decode("utf-8", errors="strict")
    parser = VisibleTextParser()
    parser.feed(text)
    return " ".join(parser.title_parts).strip(), parser.nodes


def normalize_article(raw: bytes, author_marker: str) -> tuple[str, dict]:
    title, nodes = parse_visible(raw)
    joined = "\n".join(nodes)
    for marker in RIGHTS_MARKERS:
        if marker.casefold() not in joined.casefold():
            raise RuntimeError(f"missing rights marker: {marker}")
    for marker in CONTRARY_MARKERS:
        if marker.casefold() in joined.casefold():
            raise RuntimeError(f"contrary rights marker present: {marker}")
    author_index = next(
        (i for i, value in enumerate(nodes) if author_marker in value and "опубліковано" in value),
        None,
    )
    if author_index is None:
        raise RuntimeError("required Secretariat author marker not found")
    body: list[str] = []
    for value in nodes[author_index + 1:]:
        if any(value.startswith(marker) for marker in STOP_MARKERS):
            if body:
                break
            continue
        if value in {"Image", "iframe", "Facebook", "Twitter", "Telegram", "Viber"}:
            continue
        body.append(value)
    if not body:
        raise RuntimeError("empty article body")
    clean_title = title.split("| Кабінет Міністрів України", 1)[0].strip()
    text = "\n".join([clean_title] + body)
    text = unicodedata.normalize("NFC", text)
    text = "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip()).strip() + "\n"
    return text, {"title": clean_title, "author_marker": author_marker}


def language_evidence(text: str, cfg: dict) -> dict:
    letters = [char for char in text if char.isalpha()]
    cyrillic = [char for char in letters if "\u0400" <= char <= "\u052f"]
    ratio = (len(cyrillic) / len(letters)) if letters else 0.0
    uk_specific = sum(text.casefold().count(char) for char in "іїєґ")
    lowered = text.casefold()
    lexical_hits = sum(1 for stem in UA_LEXEMES if stem in lowered)
    return {
        "alpha_chars": len(letters),
        "cyrillic_alpha_ratio": round(ratio, 6),
        "uk_specific_chars": uk_specific,
        "uk_lexical_hits": lexical_hits,
        "passed": ratio >= cfg["min_cyrillic_alpha_ratio"] and uk_specific >= cfg["min_uk_specific_chars"] and lexical_hits >= cfg["min_uk_lexical_hits"],
    }


def privacy_evidence(text: str) -> dict:
    emails = sorted(set(EMAIL_RE.findall(text)))
    phones = sorted(set(PHONE_RE.findall(text)))
    long_ids = sorted(set(LONG_ID_RE.findall(text)))
    return {
        "emails": emails,
        "phone_numbers": phones,
        "long_numeric_identifiers": long_ids,
        "passed": not emails and not phones and not long_ids,
    }


def shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = [match.group(0).casefold() for match in WORD_RE.finditer(text)]
    return {tuple(words[index:index+n]) for index in range(len(words)-n+1)} if len(words) >= n else set()


def jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    raw_root = output / "snapshots" / "sha256"
    normalized_root = output / "normalized"
    raw_root.mkdir(parents=True, exist_ok=True)
    normalized_root.mkdir(parents=True, exist_ok=True)

    rights_path = Path(cfg["rights_evidence"]["path"])
    actual_rights_sha = sha256(rights_path.read_bytes())
    if actual_rights_sha != cfg["rights_evidence"]["sha256"]:
        raise RuntimeError("rights evidence SHA mismatch")
    if cfg["local_free_only"] is not True:
        raise RuntimeError("LOCAL_FREE gate not true")
    if cfg["rights"]["license_id"] != "CC-BY-4.0":
        raise RuntimeError("unexpected license")
    required_uses = ("acquisition", "storage", "analysis", "model_training", "redistribution")
    if any(cfg["rights"]["uses"].get(use) != "ALLOWED" for use in required_uses):
        raise RuntimeError("required training rights not all ALLOWED")
    if cfg["rights"]["uses"].get("evaluation") != "NOT_ADMITTED":
        raise RuntimeError("evaluation boundary weakened")

    urls = cfg["snapshot"]["urls"]
    if not (1 <= len(urls) <= cfg["snapshot"]["max_documents"]):
        raise RuntimeError("bounded URL count violated")
    if len({item["url"] for item in urls}) != len(urls):
        raise RuntimeError("duplicate acquisition URL")
    for item in urls:
        parsed = urlparse(item["url"])
        if parsed.scheme != "https" or parsed.hostname != cfg["family"]["host"]:
            raise RuntimeError("host/scheme boundary violated")

    records: list[dict] = []
    raw_seen: dict[str, str] = {}
    normalized_seen: dict[str, str] = {}
    shingle_records: list[tuple[str, set]] = []
    rejected: list[dict] = []
    max_bytes = cfg["snapshot"]["max_download_bytes_per_document"]

    for index, item in enumerate(urls, 1):
        url = item["url"]
        try:
            raw_a = fetch(url, max_bytes)
            time.sleep(0.15)
            raw_b = fetch(url, max_bytes)
            if raw_a != raw_b:
                raise RuntimeError("repeat acquisition raw bytes differ")
            raw_hash = sha256(raw_a)
            text, meta = normalize_article(raw_a, cfg["family"]["author_prefix"])
            normalized = text.encode("utf-8")
            normalized_hash = sha256(normalized)
            language = language_evidence(text, cfg["language"])
            privacy = privacy_evidence(text)
            if not language["passed"]:
                raise RuntimeError(f"UA language gate failed: {language}")
            if not privacy["passed"]:
                raise RuntimeError(f"privacy gate failed: {privacy}")
            if len(normalized) < cfg["quality"]["min_document_utf8_bytes"]:
                raise RuntimeError(f"quality size gate failed: {len(normalized)}")
            if raw_hash in raw_seen:
                raise RuntimeError(f"intra-family raw duplicate of {raw_seen[raw_hash]}")
            if normalized_hash in normalized_seen:
                raise RuntimeError(f"intra-family normalized duplicate of {normalized_seen[normalized_hash]}")
            if normalized_hash in cfg["dedup"]["cross_family_normalized_hashes"]:
                raise RuntimeError("cross-family normalized exact duplicate")
            current_shingles = shingles(text)
            near = []
            for prior_url, prior_shingles in shingle_records:
                score = jaccard(current_shingles, prior_shingles)
                if score > cfg["quality"]["max_near_duplicate_jaccard"]:
                    near.append((prior_url, score))
            if near:
                raise RuntimeError(f"intra-family near duplicate: {near}")

            actuals = {
                "expected_raw_sha256": raw_hash,
                "expected_raw_bytes": len(raw_a),
                "expected_normalized_sha256": normalized_hash,
                "expected_normalized_utf8_bytes": len(normalized),
            }
            if cfg["mode"] == "LOCKED":
                for field, actual in actuals.items():
                    if item.get(field) != actual:
                        raise RuntimeError(f"locked identity mismatch {field}: expected={item.get(field)} actual={actual}")

            payload_path = raw_root / raw_hash / "payload"
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_bytes(raw_a)
            normalized_path = normalized_root / f"{index:03d}-{normalized_hash}.txt"
            normalized_path.write_bytes(normalized)
            raw_seen[raw_hash] = url
            normalized_seen[normalized_hash] = url
            shingle_records.append((url, current_shingles))
            records.append({
                "url": url,
                "raw_sha256": raw_hash,
                "raw_bytes": len(raw_a),
                "normalized_sha256": normalized_hash,
                "normalized_utf8_bytes": len(normalized),
                "language_evidence": language,
                "privacy_evidence": privacy,
                "meta": meta,
                "snapshot_uri": f"artifact:{payload_path.as_posix()}",
                "normalized_uri": f"artifact:{normalized_path.as_posix()}",
                "lock_values": actuals,
            })
        except Exception as exc:
            rejected.append({"url": url, "reason": str(exc)})

    accepted = len(records)
    total_normalized = sum(record["normalized_utf8_bytes"] for record in records)
    if accepted < cfg["quality"]["min_accepted_documents"]:
        raise RuntimeError(f"substantiality doc gate failed: accepted={accepted}, rejected={rejected}")
    if total_normalized < cfg["quality"]["min_total_normalized_utf8_bytes"]:
        raise RuntimeError(f"substantiality byte gate failed: total={total_normalized}")

    identity_payload = canonical_json([
        {"url": record["url"], "raw_sha256": record["raw_sha256"], "normalized_sha256": record["normalized_sha256"]}
        for record in records
    ])
    family_snapshot_sha = sha256(identity_payload)
    report = {
        "schema_version": "12-6.next100-025-kmu-snapshot-report.v1",
        "worker": cfg["worker"],
        "status": "PASS" if cfg["mode"] == "LOCKED" else "PROBE_LOCK_REQUIRED",
        "mode": cfg["mode"],
        "local_free_only": True,
        "family": cfg["family"],
        "rights": cfg["rights"],
        "rights_evidence_sha256": actual_rights_sha,
        "snapshot_id": cfg["snapshot"]["snapshot_id"],
        "family_snapshot_identity_sha256": family_snapshot_sha,
        "accepted_documents": accepted,
        "rejected_documents": rejected,
        "total_normalized_utf8_bytes": total_normalized,
        "raw_total_bytes": sum(record["raw_bytes"] for record in records),
        "language": {"expected": "uk", "all_passed": all(record["language_evidence"]["passed"] for record in records)},
        "privacy": {"all_passed": all(record["privacy_evidence"]["passed"] for record in records)},
        "dedup": {
            "intra_family_raw_unique": len(raw_seen) == accepted,
            "intra_family_normalized_unique": len(normalized_seen) == accepted,
            "intra_family_near_duplicate_threshold": cfg["quality"]["max_near_duplicate_jaccard"],
            "cross_family_exact_normalized_exclusions": cfg["dedup"]["cross_family_normalized_hashes"],
            "cross_family_reference": cfg["dedup"]["cross_family_reference"],
        },
        "attribution": {
            "required": cfg["rights"]["attribution_required"],
            "template": cfg["rights"]["attribution_template"],
            "changes": "HTML boilerplate removed; Unicode NFC; whitespace normalized; title retained; images/embeds/attachments excluded.",
        },
        "evaluation_authority": "NOT_ADMITTED",
        "records": records,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")

    with (output / "train.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            text = Path(record["normalized_uri"].removeprefix("artifact:")).read_text(encoding="utf-8")
            handle.write(json.dumps({
                "source_id": cfg["family"]["family_id"],
                "source_url": record["url"].split("?=", 1)[0],
                "source_version": cfg["snapshot"]["snapshot_id"],
                "language": "uk",
                "license_id": cfg["rights"]["license_id"],
                "raw_sha256": record["raw_sha256"],
                "normalized_sha256": record["normalized_sha256"],
                "text": text,
                "training_eligible": cfg["mode"] == "LOCKED",
                "evaluation_eligible": False,
            }, ensure_ascii=False, sort_keys=True) + "\n")

    artifact_files = []
    for path in sorted(path for path in output.rglob("*") if path.is_file() and path.name != "artifact-manifest.json"):
        payload = path.read_bytes()
        artifact_files.append({"path": path.as_posix(), "sha256": sha256(payload), "size_bytes": len(payload)})
    manifest_core = {
        "schema_version": "12-6.next100-025-artifact-manifest.v1",
        "family_snapshot_identity_sha256": family_snapshot_sha,
        "files": artifact_files,
    }
    manifest = {**manifest_core, "manifest_sha256": sha256(canonical_json(manifest_core))}
    (output / "artifact-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
