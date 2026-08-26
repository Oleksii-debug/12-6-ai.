#!/usr/bin/env python3
"""Deterministic D03-style verifier for the bounded php/doc-uk snapshot."""
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
HEADER = re.compile(r"EN-Revision:\s*([0-9a-f]{40})\s+Maintainer:\s*([^\s]+)\s+Status:\s*([^\s-]+)", re.I)
TOKEN = re.compile(r"[\w]+", re.UNICODE)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b")
COMMENT = re.compile(r"<!--.*?-->", re.S)
TAG = re.compile(r"<[^>]+>", re.S)
ENTITY = re.compile(r"&[A-Za-z_][A-Za-z0-9_.:-]*;")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*[\"'][^\"'\n]{8,}[\"']"),
)
UK_LEXEMES = {"і", "або", "але", "буде", "для", "значення", "можна", "не", "після", "потрібно", "тип", "функція", "що", "як", "якщо"}


def canon(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def fetch(url: str, limit: int) -> bytes:
    req = Request(url, headers={"User-Agent": "12-6-next100-028/2.0", "Accept-Encoding": "identity"})
    with urlopen(req, timeout=30) as response:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > limit:
            raise RuntimeError(f"declared object too large: {url}")
        data = response.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError(f"download exceeded bound: {url}")
    return data


def normalize(raw: bytes) -> bytes:
    text = raw.decode("utf-8", "strict").replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = COMMENT.sub("\n", text).replace("<![CDATA[", "\n").replace("]]>", "\n")
    text = TAG.sub("\n", text)
    text = html.unescape(text)
    text = ENTITY.sub(" ", text)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        raise RuntimeError("normalization produced empty text")
    return ("\n".join(lines) + "\n").encode("utf-8")


def frame(rows: list[tuple[str, bytes]]) -> bytes:
    out = bytearray()
    for path, payload in sorted(rows):
        p = path.encode("utf-8")
        out += len(p).to_bytes(4, "big") + p + len(payload).to_bytes(8, "big") + payload
    return bytes(out)


def lang(data: bytes) -> dict[str, Any]:
    text = data.decode("utf-8")
    letters = [c for c in text if c.isalpha()]
    cyr = [c for c in letters if "\u0400" <= c <= "\u04ff"]
    lat = [c for c in letters if ("A" <= c <= "Z") or ("a" <= c <= "z")]
    lower = text.casefold()
    uk = sum(lower.count(c) for c in "іїєґ")
    ru = sum(lower.count(c) for c in "ыэёъ")
    hits = sorted({t.casefold() for t in TOKEN.findall(text)} & UK_LEXEMES)
    return {
        "alphabetic_letters": len(letters),
        "cyrillic_letters": len(cyr),
        "latin_letters": len(lat),
        "cyrillic_ratio": round(len(cyr) / max(len(letters), 1), 6),
        "uk_specific_letters": uk,
        "ru_specific_letters": ru,
        "uk_lexical_hits": hits,
    }


def privacy(raw: bytes) -> dict[str, Any]:
    text = raw.decode("utf-8", "strict")
    secrets = [p.pattern for p in SECRET_PATTERNS if p.search(text)]
    emails = [m.group(0) for m in EMAIL.finditer(text)]
    nonexamples = [e for e in emails if not e.casefold().endswith(("@example.com", "@example.org", "@example.net"))]
    return {
        "secret_pattern_count": len(secrets),
        "secret_patterns": secrets,
        "email_count": len(emails),
        "nonexample_email_count": len(nonexamples),
        "nonexample_emails": nonexamples,
        "pass": not secrets and not nonexamples,
    }


def shingles(data: bytes, n: int) -> set[tuple[str, ...]]:
    words = [t.casefold() for t in TOKEN.findall(data.decode("utf-8"))]
    if not words:
        return set()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i:i+n]) for i in range(len(words)-n+1)}


def jac(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def collect_sha256(obj: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for value in obj.values():
            out |= collect_sha256(value)
    elif isinstance(obj, list):
        for value in obj:
            out |= collect_sha256(value)
    elif isinstance(obj, str) and SHA256.fullmatch(obj):
        out.add(obj)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if cfg["worker_id"] != "NEXT100-028-DATA-UA-TECH-GITHUB" or cfg["local_free_only"] is not True:
        raise RuntimeError("worker/LOCAL_FREE boundary mismatch")
    if cfg["mode"] not in {"PROBE", "LOCKED"}:
        raise RuntimeError("invalid mode")

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "raw").mkdir(parents=True)

    source = cfg["source"]
    raw_rows: list[tuple[str, bytes]] = []
    norm_rows: list[tuple[str, bytes]] = []
    reports: list[dict[str, Any]] = []
    quality_ok = True
    privacy_ok = True

    for item in cfg["selection"]["files"]:
        path = item["path"]
        raw = fetch(f"https://raw.githubusercontent.com/php/doc-uk/{source['commit']}/{path}", max(65536, item["expected_size"] + 1))
        if len(raw) != item["expected_size"]:
            raise RuntimeError(f"size mismatch: {path}")
        blob = git_sha1(raw)
        if blob != item["git_blob_sha1"]:
            raise RuntimeError(f"Git blob mismatch: {path}: {blob}")
        decoded = raw.decode("utf-8", "strict")
        header = HEADER.search(decoded[:2048])
        ready = bool(header and header.group(3).casefold() == "ready")
        clean = not any(marker in decoded for marker in ("<<<<<<<", "=======", ">>>>>>>"))
        no_todo = re.search(r"(?i)\b(?:TODO|FIXME)\b", decoded) is None
        file_quality = bool(header) and ready and clean and no_todo
        quality_ok &= file_quality
        norm = normalize(raw)
        p = privacy(raw)
        privacy_ok &= p["pass"]
        raw_path = out / "raw" / path
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw)
        raw_rows.append((path, raw))
        norm_rows.append((path, norm))
        reports.append({
            "path": path,
            "git_blob_sha1": blob,
            "raw_bytes": len(raw),
            "raw_sha256": sha256(raw),
            "normalized_bytes": len(norm),
            "normalized_sha256": sha256(norm),
            "header": {
                "en_revision": header.group(1) if header else None,
                "maintainer": header.group(2) if header else None,
                "status": header.group(3).casefold() if header else None,
            },
            "quality_pass": file_quality,
            "language": lang(norm),
            "privacy": p,
        })

    raw_bytes = sum(len(x) for _, x in raw_rows)
    if raw_bytes > cfg["selection"]["max_total_raw_bytes"]:
        raise RuntimeError("selection byte bound exceeded")
    raw_bundle = frame(raw_rows)
    norm_bundle = frame(norm_rows)
    norm_bytes = sum(len(x) for _, x in norm_rows)
    raw_bundle_sha = sha256(raw_bundle)
    norm_bundle_sha = sha256(norm_bundle)
    corpus_lang = lang(b"\n".join(x for _, x in norm_rows))
    lang_ok = (
        corpus_lang["cyrillic_ratio"] >= 0.55
        and corpus_lang["uk_specific_letters"] >= 25
        and len(corpus_lang["uk_lexical_hits"]) >= 4
        and corpus_lang["uk_specific_letters"] > corpus_lang["ru_specific_letters"]
    )

    exact_ok = len({r["raw_sha256"] for r in reports}) == len(reports) and len({r["normalized_sha256"] for r in reports}) == len(reports)
    n = cfg["dedup"]["intra_snapshot_near_duplicate"]["token_shingle_size"]
    threshold = cfg["dedup"]["intra_snapshot_near_duplicate"]["jaccard_threshold"]
    sets = {p: shingles(data, n) for p, data in norm_rows}
    near: list[dict[str, Any]] = []
    paths = sorted(sets)
    for i, left in enumerate(paths):
        for right in paths[i+1:]:
            score = jac(sets[left], sets[right])
            if score >= threshold:
                near.append({"left": left, "right": right, "jaccard": round(score, 6)})

    registry = json.loads((ROOT / cfg["dedup"]["terminal_registry"]).read_text(encoding="utf-8"))
    incumbent_hashes: set[str] = set()
    incumbent_families: set[str] = set()
    for entry in registry["sources"]:
        incumbent_hashes |= {entry["snapshot"]["raw_sha256"], entry["snapshot"]["normalized_sha256"]}
        incumbent_families.add(entry["independent_source_family"]["family_id"])
    candidate_hashes = {raw_bundle_sha, norm_bundle_sha} | {r["raw_sha256"] for r in reports} | {r["normalized_sha256"] for r in reports}
    terminal_collisions = sorted(candidate_hashes & incumbent_hashes)
    reserved = json.loads((ROOT / cfg["dedup"]["reserved_fingerprints"]).read_text(encoding="utf-8"))
    reserved_collisions = sorted(candidate_hashes & collect_sha256(reserved))

    lic = cfg["license"]
    lic_raw = fetch(f"https://raw.githubusercontent.com/php/doc-en/{lic['license_commit']}/{lic['license_path']}", 131072)
    lic_blob = git_sha1(lic_raw)
    if lic_blob != lic["license_blob_sha1"]:
        raise RuntimeError(f"license blob mismatch: {lic_blob}")
    lic_sha = sha256(lic_raw)
    lic_text = lic_raw.decode("utf-8", "strict").casefold()
    lic_content_ok = all(term in lic_text for term in ("creative commons", "attribution", "reproduce", "distribute", "adaptation"))

    gates = {
        "exact_commit_and_git_blobs": True,
        "bounded_selection": raw_bytes <= cfg["selection"]["max_total_raw_bytes"],
        "license_blob_exact": True,
        "license_content": lic_content_ok,
        "quality": quality_ok,
        "uk_language": lang_ok,
        "privacy": privacy_ok,
        "intra_exact_dedup": exact_ok,
        "intra_near_dedup": not near,
        "terminal_registry_exact_dedup": not terminal_collisions,
        "reserved_fingerprint_dedup": not reserved_collisions,
        "independent_family": source["source_family"] not in incumbent_families,
        "training_rights": cfg["purpose_decisions"]["model_training"] == "ALLOWED",
        "redistribution_rights": cfg["purpose_decisions"]["redistribution"] == "ALLOWED",
        "evaluation_firewall": cfg["purpose_decisions"]["evaluation"] == "NOT_SEPARATELY_ADMITTED",
    }
    observed = {
        "file_count": len(reports),
        "raw_bytes": raw_bytes,
        "raw_bundle_sha256": raw_bundle_sha,
        "normalized_bytes": norm_bytes,
        "normalized_bundle_sha256": norm_bundle_sha,
        "license_sha256": lic_sha,
    }
    all_gates = all(gates.values())
    lock_matches = cfg["mode"] == "LOCKED" and all(cfg["expected_lock"].get(k) == v for k, v in observed.items())
    terminal = bool(all_gates and lock_matches)
    state = "ADMIT" if terminal else "RETEST_PROBE_LOCK_REQUIRED" if cfg["mode"] == "PROBE" and all_gates else "REJECT"

    authority: dict[str, Any] = {
        "schema_version": "12-6.next100-028-php-doc-uk-evidence.v2",
        "worker_id": cfg["worker_id"],
        "mode": cfg["mode"],
        "local_free_only": True,
        "source": source,
        "license": {**lic, "verified_git_blob_sha1": lic_blob, "license_sha256": lic_sha},
        "purpose_decisions": cfg["purpose_decisions"],
        "selection_rule": cfg["selection"]["rule"],
        "normalization": cfg["normalization"],
        "files": reports,
        "snapshot": observed,
        "language": corpus_lang,
        "quality": {
            "all_selected_status_ready": quality_ok,
            "explicit_wip_exclusions": cfg["selection"]["explicit_exclusions"],
            "human_translation_provenance": cfg["quality"]["human_translation_provenance"],
        },
        "privacy": {
            "pass": privacy_ok,
            "nonexample_email_total": sum(r["privacy"]["nonexample_email_count"] for r in reports),
            "secret_pattern_total": sum(r["privacy"]["secret_pattern_count"] for r in reports),
        },
        "dedup": {
            "intra_near_pairs_at_or_above_threshold": near,
            "terminal_registry_exact_collisions": terminal_collisions,
            "reserved_fingerprint_collisions": reserved_collisions,
            "incumbent_families": sorted(incumbent_families),
            "candidate_family": source["source_family"],
            "family_new": source["source_family"] not in incumbent_families,
        },
        "gates": gates,
        "all_content_gates_pass": all_gates,
        "lock_matches": lock_matches,
        "terminal": terminal,
        "terminal_state": state,
        "evaluation_reserved": False,
        "evaluation_use": "NOT_SEPARATELY_ADMITTED",
    }
    authority["evidence_identity_sha256"] = sha256(canon(authority))
    (out / "raw.bundle.bin").write_bytes(raw_bundle)
    (out / "normalized.bundle.bin").write_bytes(norm_bundle)
    (out / "license.xml").write_bytes(lic_raw)
    (out / "ATTRIBUTION.txt").write_text(
        "PHP Manual Ukrainian documentation snapshot\n"
        f"Source: https://github.com/php/doc-uk @ {source['commit']}\n"
        "Copyright: PHP Documentation Group\n"
        "License: Creative Commons Attribution 3.0 or later\n"
        "Modifications: bounded selection and deterministic normalization by NEXT100-028.\n",
        encoding="utf-8",
    )
    (out / "authority.json").write_bytes(canon(authority))
    print("NEXT100_028_PROBE_LOCK=" + json.dumps(observed, sort_keys=True))
    print("NEXT100_028_LANGUAGE=" + json.dumps(corpus_lang, ensure_ascii=False, sort_keys=True))
    print("NEXT100_028_PRIVACY=" + json.dumps(authority["privacy"], sort_keys=True))
    print("NEXT100_028_DEDUP=" + json.dumps(authority["dedup"], sort_keys=True))
    print("NEXT100_028_GATES=" + json.dumps(gates, sort_keys=True))
    print("NEXT100_028_EVIDENCE_IDENTITY=" + authority["evidence_identity_sha256"])
    print("NEXT100_028_TERMINAL_STATE=" + state)
    if cfg["mode"] == "LOCKED" and not terminal:
        raise SystemExit("LOCKED run failed terminal admission gates")


if __name__ == "__main__":
    main()
