#!/usr/bin/env python3
"""Build and verify the bounded NEXT100-028 php/doc-uk D03-style snapshot."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEADER_RE = re.compile(
    r"EN-Revision:\s*([0-9a-f]{40})\s+Maintainer:\s*([^\s]+)\s+Status:\s*([^\s-]+)",
    re.IGNORECASE,
)
_XML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_XML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
_ENTITY_RE = re.compile(r"&[A-Za-z_][A-Za-z0-9_.:-]*;")
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b")
_SECRET_RES = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*[\"'][^\"'\n]{8,}[\"']"
    ),
)
_UK_LEXEMES = frozenset(
    {
        "і",
        "або",
        "але",
        "буде",
        "для",
        "значення",
        "можна",
        "не",
        "після",
        "потрібно",
        "тип",
        "функція",
        "що",
        "як",
        "якщо",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def fetch(url: str, max_bytes: int) -> bytes:
    request = Request(
        url,
        headers={
            "User-Agent": "12-6-next100-028/1.0 (+bounded-source-audit)",
            "Accept-Encoding": "identity",
        },
    )
    with urlopen(request, timeout=30) as response:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > max_bytes:
            raise RuntimeError(f"declared object too large: {url}: {declared} > {max_bytes}")
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise RuntimeError(f"download exceeded bound: {url}: {len(payload)} > {max_bytes}")
    return payload


def normalize_docbook_xml(raw: bytes) -> bytes:
    text = raw.decode("utf-8", errors="strict")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = _XML_COMMENT_RE.sub("\n", text)
    text = text.replace("<![CDATA[", "\n").replace("]]>", "\n")
    text = _XML_TAG_RE.sub("\n", text)
    text = html.unescape(text)
    text = _ENTITY_RE.sub(" ", text)
    lines: list[str] = []
    for line in text.splitlines():
        collapsed = " ".join(line.split())
        if collapsed:
            lines.append(collapsed)
    if not lines:
        raise RuntimeError("normalization produced empty text")
    return ("\n".join(lines) + "\n").encode("utf-8")


def frame_bundle(rows: list[tuple[str, bytes]]) -> bytes:
    out = bytearray()
    for path, payload in sorted(rows):
        path_bytes = path.encode("utf-8")
        out.extend(len(path_bytes).to_bytes(4, "big"))
        out.extend(path_bytes)
        out.extend(len(payload).to_bytes(8, "big"))
        out.extend(payload)
    return bytes(out)


def language_metrics(payload: bytes) -> dict[str, Any]:
    text = payload.decode("utf-8")
    letters = [char for char in text if char.isalpha()]
    cyrillic = [char for char in letters if "\u0400" <= char <= "\u04ff"]
    latin = [char for char in letters if ("A" <= char <= "Z") or ("a" <= char <= "z")]
    lowered = text.casefold()
    uk_specific = sum(lowered.count(char) for char in "іїєґ")
    ru_specific = sum(lowered.count(char) for char in "ыэёъ")
    tokens = {token.casefold() for token in _TOKEN_RE.findall(text)}
    lexical_hits = sorted(tokens & _UK_LEXEMES)
    alphabetic = max(len(letters), 1)
    return {
        "alphabetic_letters": len(letters),
        "cyrillic_letters": len(cyrillic),
        "latin_letters": len(latin),
        "cyrillic_ratio": round(len(cyrillic) / alphabetic, 6),
        "uk_specific_letters": uk_specific,
        "ru_specific_letters": ru_specific,
        "uk_lexical_hits": lexical_hits,
    }


def shingles(payload: bytes, size: int) -> set[tuple[str, ...]]:
    tokens = [token.casefold() for token in _TOKEN_RE.findall(payload.decode("utf-8"))]
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def scan_privacy(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8", errors="strict")
    secret_findings = [pattern.pattern for pattern in _SECRET_RES if pattern.search(text)]
    emails = [match.group(0) for match in _EMAIL_RE.finditer(text)]
    nonexample_emails = [
        address
        for address in emails
        if not address.casefold().endswith(("@example.com", "@example.org", "@example.net"))
    ]
    return {
        "secret_pattern_count": len(secret_findings),
        "secret_patterns": secret_findings,
        "email_count": len(emails),
        "nonexample_email_count": len(nonexample_emails),
        "nonexample_emails": nonexample_emails,
        "pass": not secret_findings and not nonexample_emails,
    }


def all_sha256_strings(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for item in value.values():
            found |= all_sha256_strings(item)
    elif isinstance(value, list):
        for item in value:
            found |= all_sha256_strings(item)
    elif isinstance(value, str) and _SHA256_RE.fullmatch(value):
        found.add(value)
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["worker_id"] != "NEXT100-028-DATA-UA-TECH-GITHUB":
        raise RuntimeError("worker identity mismatch")
    if config["local_free_only"] is not True:
        raise RuntimeError("LOCAL_FREE boundary changed")
    mode = config["mode"]
    if mode not in {"PROBE", "LOCKED"}:
        raise RuntimeError(f"unsupported mode: {mode}")

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "raw").mkdir(parents=True)

    source = config["source"]
    commit = source["commit"]
    raw_rows: list[tuple[str, bytes]] = []
    normalized_rows: list[tuple[str, bytes]] = []
    file_reports: list[dict[str, Any]] = []
    quality_pass = True
    privacy_pass = True

    for item in config["selection"]["files"]:
        path = item["path"]
        url = f"https://raw.githubusercontent.com/php/doc-uk/{commit}/{path}"
        raw = fetch(url, max(item["expected_size"] + 1, 1024 * 64))
        actual_blob = git_blob_sha1(raw)
        if actual_blob != item["git_blob_sha1"]:
            raise RuntimeError(f"git blob mismatch for {path}: {actual_blob}")
        if len(raw) != item["expected_size"]:
            raise RuntimeError(f"size mismatch for {path}: {len(raw)}")
        header = _HEADER_RE.search(raw[:512].decode("utf-8", errors="strict"))
        ready = bool(header and header.group(3).casefold() == "ready")
        no_conflict = not any(marker in raw for marker in (b"<<<<<<<", b"=======", b">>>>>>>"))
        no_todo = re.search(rb"(?i)\b(?:TODO|FIXME)\b", raw) is None
        quality_ok = bool(header) and ready and no_conflict and no_todo
        quality_pass = quality_pass and quality_ok

        normalized = normalize_docbook_xml(raw)
        privacy = scan_privacy(raw)
        privacy_pass = privacy_pass and privacy["pass"]
        metrics = language_metrics(normalized)

        raw_path = out_dir / "raw" / path
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)
        raw_rows.append((path, raw))
        normalized_rows.append((path, normalized))
        file_reports.append(
            {
                "path": path,
                "raw_bytes": len(raw),
                "raw_sha256": sha256_bytes(raw),
                "git_blob_sha1": actual_blob,
                "normalized_bytes": len(normalized),
                "normalized_sha256": sha256_bytes(normalized),
                "header": {
                    "en_revision": header.group(1) if header else None,
                    "maintainer": header.group(2) if header else None,
                    "status": header.group(3).casefold() if header else None,
                },
                "quality_pass": quality_ok,
                "language": metrics,
                "privacy": privacy,
            }
        )

    total_raw = sum(len(payload) for _, payload in raw_rows)
    if total_raw > config["selection"]["max_total_raw_bytes"]:
        raise RuntimeError("bounded selection exceeded max_total_raw_bytes")

    raw_bundle = frame_bundle(raw_rows)
    normalized_bundle = frame_bundle(normalized_rows)
    raw_bundle_sha = sha256_bytes(raw_bundle)
    normalized_bundle_sha = sha256_bytes(normalized_bundle)
    normalized_total = sum(len(payload) for _, payload in normalized_rows)
    corpus_language = language_metrics(b"\n".join(payload for _, payload in normalized_rows))
    language_pass = (
        corpus_language["cyrillic_ratio"] >= 0.55
        and corpus_language["uk_specific_letters"] >= 25
        and len(corpus_language["uk_lexical_hits"]) >= 4
        and corpus_language["uk_specific_letters"] > corpus_language["ru_specific_letters"]
    )

    shingle_size = config["dedup"]["intra_snapshot_near_duplicate"]["token_shingle_size"]
    threshold = config["dedup"]["intra_snapshot_near_duplicate"]["jaccard_threshold"]
    shingle_sets = {path: shingles(payload, shingle_size) for path, payload in normalized_rows}
    near_pairs: list[dict[str, Any]] = []
    paths = sorted(shingle_sets)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            score = jaccard(shingle_sets[left], shingle_sets[right])
            if score >= threshold:
                near_pairs.append({"left": left, "right": right, "jaccard": round(score, 6)})

    raw_hashes = [report["raw_sha256"] for report in file_reports]
    normalized_hashes = [report["normalized_sha256"] for report in file_reports]
    intra_exact_pass = len(raw_hashes) == len(set(raw_hashes)) and len(normalized_hashes) == len(
        set(normalized_hashes)
    )

    registry_path = ROOT / config["dedup"]["terminal_registry"]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    incumbent_hashes: set[str] = set()
    incumbent_families: set[str] = set()
    for entry in registry["sources"]:
        incumbent_hashes.add(entry["snapshot"]["raw_sha256"])
        incumbent_hashes.add(entry["snapshot"]["normalized_sha256"])
        incumbent_families.add(entry["independent_source_family"]["family_id"])
    candidate_hashes = set(raw_hashes + normalized_hashes + [raw_bundle_sha, normalized_bundle_sha])
    terminal_exact_collisions = sorted(candidate_hashes & incumbent_hashes)
    family_new = source["source_family"] not in incumbent_families

    reserved_path = ROOT / config["dedup"]["reserved_fingerprints"]
    reserved = json.loads(reserved_path.read_text(encoding="utf-8"))
    reserved_collisions = sorted(candidate_hashes & all_sha256_strings(reserved))

    license_cfg = config["license"]
    license_url = (
        "https://raw.githubusercontent.com/php/doc-en/"
        f"{license_cfg['license_commit']}/{license_cfg['license_path']}"
    )
    license_raw = fetch(license_url, 128 * 1024)
    actual_license_blob = git_blob_sha1(license_raw)
    if actual_license_blob != license_cfg["license_blob_sha1"]:
        raise RuntimeError(f"license blob mismatch: {actual_license_blob}")
    license_sha = sha256_bytes(license_raw)
    license_text = license_raw.decode("utf-8", errors="strict")
    license_content_pass = (
        "Creative Commons" in license_text
        and "Attribution" in license_text
        and "Reproduce" in license_text
        and "Distribute" in license_text
        and "Adaptation" in license_text
    )

    gates = {
        "exact_commit_and_git_blobs": True,
        "bounded_selection": total_raw <= config["selection"]["max_total_raw_bytes"],
        "license_blob_exact": actual_license_blob == license_cfg["license_blob_sha1"],
        "license_content": license_content_pass,
        "quality": quality_pass,
        "uk_language": language_pass,
        "privacy": privacy_pass,
        "intra_exact_dedup": intra_exact_pass,
        "intra_near_dedup": not near_pairs,
        "terminal_registry_exact_dedup": not terminal_exact_collisions,
        "reserved_fingerprint_dedup": not reserved_collisions,
        "independent_family": family_new,
        "training_rights": config["purpose_decisions"]["model_training"] == "ALLOWED",
        "redistribution_rights": config["purpose_decisions"]["redistribution"] == "ALLOWED",
        "evaluation_firewall": config["purpose_decisions"]["evaluation"] == "NOT_SEPARATELY_ADMITTED",
    }
    all_gates_pass = all(gates.values())

    lock_observed = {
        "file_count": len(file_reports),
        "raw_bytes": total_raw,
        "raw_bundle_sha256": raw_bundle_sha,
        "normalized_bytes": normalized_total,
        "normalized_bundle_sha256": normalized_bundle_sha,
        "license_sha256": license_sha,
    }
    lock_expected = config["expected_lock"]
    lock_matches = all(
        lock_expected.get(key) == value
        for key, value in lock_observed.items()
        if key != "file_count" or mode == "LOCKED"
    )
    if mode == "PROBE":
        lock_matches = False

    authority: dict[str, Any] = {
        "schema_version": "12-6.next100-028-php-doc-uk-evidence.v1",
        "worker_id": config["worker_id"],
        "mode": mode,
        "local_free_only": True,
        "source": source,
        "license": {
            **license_cfg,
            "license_sha256": license_sha,
            "verified_git_blob_sha1": actual_license_blob,
        },
        "purpose_decisions": config["purpose_decisions"],
        "selection_rule": config["selection"]["rule"],
        "normalization": config["normalization"],
        "files": file_reports,
        "snapshot": lock_observed,
        "language": corpus_language,
        "quality": {
            "all_selected_status_ready": quality_pass,
            "explicit_wip_exclusions": config["selection"]["explicit_exclusions"],
            "human_translation_provenance": config["quality"]["human_translation_provenance"],
        },
        "privacy": {
            "pass": privacy_pass,
            "nonexample_email_total": sum(
                report["privacy"]["nonexample_email_count"] for report in file_reports
            ),
            "secret_pattern_total": sum(
                report["privacy"]["secret_pattern_count"] for report in file_reports
            ),
        },
        "dedup": {
            "intra_near_pairs_at_or_above_threshold": near_pairs,
            "terminal_registry_exact_collisions": terminal_exact_collisions,
            "reserved_fingerprint_collisions": reserved_collisions,
            "incumbent_families": sorted(incumbent_families),
            "candidate_family": source["source_family"],
            "family_new": family_new,
        },
        "gates": gates,
        "all_content_gates_pass": all_gates_pass,
        "lock_matches": lock_matches,
        "terminal": bool(mode == "LOCKED" and all_gates_pass and lock_matches),
        "terminal_state": (
            "ADMIT"
            if mode == "LOCKED" and all_gates_pass and lock_matches
            else "RETEST_PROBE_LOCK_REQUIRED"
            if mode == "PROBE" and all_gates_pass
            else "REJECT"
        ),
        "evaluation_reserved": False,
        "evaluation_use": "NOT_SEPARATELY_ADMITTED",
    }
    authority["evidence_identity_sha256"] = sha256_bytes(canonical_json_bytes(authority))

    raw_bundle_path = out_dir / "raw.bundle.bin"
    normalized_bundle_path = out_dir / "normalized.bundle.bin"
    raw_bundle_path.write_bytes(raw_bundle)
    normalized_bundle_path.write_bytes(normalized_bundle)
    (out_dir / "license.xml").write_bytes(license_raw)
    attribution = (
        "PHP Manual Ukrainian documentation snapshot\n"
        f"Source: https://github.com/php/doc-uk @ {commit}\n"
        "Copyright: PHP Documentation Group\n"
        "License: Creative Commons Attribution 3.0 or later\n"
        "License scope: https://www.php.net/license/\n"
        "Ukrainian manual copyright: https://www.php.net/manual/uk/copyright.php\n"
        "Selection and normalization performed by NEXT100-028; modifications are recorded in authority.json.\n"
    )
    (out_dir / "ATTRIBUTION.txt").write_text(attribution, encoding="utf-8")
    (out_dir / "authority.json").write_bytes(canonical_json_bytes(authority))

    print("NEXT100_028_PROBE_LOCK=" + json.dumps(lock_observed, sort_keys=True))
    print("NEXT100_028_GATES=" + json.dumps(gates, sort_keys=True))
    print(f"NEXT100_028_EVIDENCE_IDENTITY={authority['evidence_identity_sha256']}")
    print(f"NEXT100_028_TERMINAL_STATE={authority['terminal_state']}")
    if mode == "LOCKED" and not authority["terminal"]:
        raise SystemExit("LOCKED run failed terminal admission gates")


if __name__ == "__main__":
    main()
