#!/usr/bin/env python3
"""Build and verify the bounded NEXT100-030 Ukrainian Rust Book OER snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOKEN_RE = re.compile(r"[\w’'\-]+", re.UNICODE)
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()\-]{7,}\d)(?!\d)")
SECRET_RES = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*['\"][^'\"\n]{8,}['\"]"),
)
UK_LEXEMES = frozenset({"і", "або", "але", "для", "коли", "можна", "не", "після", "програма", "розділ", "треба", "що", "як", "якщо"})


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def fetch(url: str, max_bytes: int) -> bytes:
    req = Request(url, headers={"User-Agent": "12-6-next100-030/1.0", "Accept-Encoding": "identity"})
    with urlopen(req, timeout=30) as response:
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > max_bytes:
            raise RuntimeError(f"declared object too large: {declared} > {max_bytes}: {url}")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise RuntimeError(f"download exceeded bound: {len(data)} > {max_bytes}: {url}")
    return data


def normalize_markdown_prose(raw: bytes) -> bytes:
    text = raw.decode("utf-8", errors="strict").replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"<!--.*?-->", "\n", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "\n", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^\s*\[[^\]]+\]:\s+\S+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"`[^`]*`", " ", text)
    lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}(?:#{1,6}|>|[-*+]\s+|\d+\.\s+)", "", line)
        line = line.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
        collapsed = " ".join(line.split())
        if collapsed:
            lines.append(collapsed)
    if not lines:
        raise RuntimeError("normalization produced empty prose")
    return ("\n".join(lines) + "\n").encode("utf-8")


def frame_bundle(rows: list[tuple[str, bytes]]) -> bytes:
    out = bytearray()
    for path, payload in sorted(rows):
        p = path.encode("utf-8")
        out.extend(len(p).to_bytes(4, "big"))
        out.extend(p)
        out.extend(len(payload).to_bytes(8, "big"))
        out.extend(payload)
    return bytes(out)


def language_metrics(payload: bytes) -> dict[str, Any]:
    text = payload.decode("utf-8")
    letters = [c for c in text if c.isalpha()]
    cyr = [c for c in letters if "\u0400" <= c <= "\u04ff"]
    lat = [c for c in letters if ("A" <= c <= "Z") or ("a" <= c <= "z")]
    low = text.casefold()
    uk_specific = sum(low.count(c) for c in "іїєґ")
    ru_specific = sum(low.count(c) for c in "ыэёъ")
    tokens = [t.casefold() for t in TOKEN_RE.findall(text)]
    return {
        "alphabetic_letters": len(letters),
        "cyrillic_letters": len(cyr),
        "latin_letters": len(lat),
        "cyrillic_ratio": round(len(cyr) / max(len(letters), 1), 6),
        "uk_specific_letters": uk_specific,
        "ru_specific_letters": ru_specific,
        "uk_lexical_hits": sorted(set(tokens) & UK_LEXEMES),
        "word_count": len(tokens),
    }


def scan_privacy(payload: bytes) -> dict[str, Any]:
    text = payload.decode("utf-8", errors="strict")
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    secrets = [p.pattern for p in SECRET_RES if p.search(text)]
    return {"email_count": len(emails), "phone_like_count": len(phones), "secret_pattern_count": len(secrets), "pass": not emails and not phones and not secrets}


def shingles(payload: bytes, size: int) -> set[tuple[str, ...]]:
    tokens = [t.casefold() for t in TOKEN_RE.findall(payload.decode("utf-8"))]
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i+size]) for i in range(len(tokens)-size+1)}


def jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def all_sha256_strings(value: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(value, dict):
        for v in value.values():
            out |= all_sha256_strings(v)
    elif isinstance(value, list):
        for v in value:
            out |= all_sha256_strings(v)
    elif isinstance(value, str) and SHA256_RE.fullmatch(value):
        out.add(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if cfg["worker_id"] != "NEXT100-030-DATA-UA-OER":
        raise RuntimeError("worker mismatch")
    if cfg["local_free_only"] is not True or cfg["training_executed"] is not False:
        raise RuntimeError("execution boundary changed")
    if cfg["mode"] not in {"PROBE", "LOCKED"}:
        raise RuntimeError("invalid mode")

    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "raw").mkdir(parents=True)
    (out / "normalized").mkdir(parents=True)

    src = cfg["source"]
    commit = src["commit"]
    raw_rows: list[tuple[str, bytes]] = []
    norm_rows: list[tuple[str, bytes]] = []
    reports: list[dict[str, Any]] = []
    quality_pass = True
    privacy_pass = True

    for item in cfg["selection"]["files"]:
        path = item["path"]
        url = f"https://raw.githubusercontent.com/rust-lang-ua/rustbook_ukrainian/{commit}/{path}"
        raw = fetch(url, cfg["selection"]["max_file_raw_bytes"])
        blob = git_blob_sha1(raw)
        if blob != item["git_blob_sha1"]:
            raise RuntimeError(f"blob mismatch {path}: {blob}")
        text = raw.decode("utf-8", errors="strict")
        quality_ok = not any(mark in text for mark in ("<<<<<<<", "=======", ">>>>>>>", "TODO", "FIXME"))
        norm = normalize_markdown_prose(raw)
        lm = language_metrics(norm)
        priv = scan_privacy(raw)
        quality_ok = quality_ok and lm["word_count"] >= cfg["quality"]["min_words_per_file"]
        quality_pass = quality_pass and quality_ok
        privacy_pass = privacy_pass and priv["pass"]
        (out / "raw" / Path(path).name).write_bytes(raw)
        (out / "normalized" / (Path(path).stem + ".txt")).write_bytes(norm)
        raw_rows.append((path, raw))
        norm_rows.append((path, norm))
        reports.append({"path": path, "git_blob_sha1": blob, "raw_bytes": len(raw), "raw_sha256": sha256_bytes(raw), "normalized_bytes": len(norm), "normalized_sha256": sha256_bytes(norm), "language": lm, "privacy": priv, "quality_pass": quality_ok})

    raw_total = sum(len(p) for _, p in raw_rows)
    norm_total = sum(len(p) for _, p in norm_rows)
    if raw_total > cfg["selection"]["max_total_raw_bytes"]:
        raise RuntimeError("raw bundle bound exceeded")
    raw_bundle_sha = sha256_bytes(frame_bundle(raw_rows))
    norm_bundle_sha = sha256_bytes(frame_bundle(norm_rows))
    aggregate_language = language_metrics(b"\n".join(p for _, p in norm_rows))
    language_pass = (
        aggregate_language["cyrillic_ratio"] >= cfg["quality"]["min_cyrillic_ratio"]
        and aggregate_language["uk_specific_letters"] >= cfg["quality"]["min_uk_specific_letters"]
        and aggregate_language["ru_specific_letters"] <= cfg["quality"]["max_ru_specific_letters"]
        and len(aggregate_language["uk_lexical_hits"]) >= cfg["quality"]["min_uk_lexical_hits"]
    )

    raw_hashes = [r["raw_sha256"] for r in reports]
    norm_hashes = [r["normalized_sha256"] for r in reports]
    intra_exact = len(raw_hashes) == len(set(raw_hashes)) and len(norm_hashes) == len(set(norm_hashes))
    shingle_size = cfg["dedup"]["token_shingle_size"]
    near_threshold = cfg["dedup"]["near_jaccard_threshold"]
    candidate_shingles = {path: shingles(payload, shingle_size) for path, payload in norm_rows}
    intra_near: list[dict[str, Any]] = []
    paths = sorted(candidate_shingles)
    for i, left in enumerate(paths):
        for right in paths[i + 1:]:
            score = jaccard(candidate_shingles[left], candidate_shingles[right])
            if score >= near_threshold:
                intra_near.append({"left": left, "right": right, "jaccard": round(score, 6)})

    registry = json.loads((ROOT / cfg["dedup"]["terminal_registry"]).read_text(encoding="utf-8"))
    incumbent_hashes: set[str] = set()
    incumbent_families: set[str] = set()
    cross_near: list[dict[str, Any]] = []
    candidate_agg = shingles(b"\n".join(p for _, p in norm_rows), shingle_size)
    for entry in registry["sources"]:
        incumbent_hashes |= {entry["snapshot"]["raw_sha256"], entry["snapshot"]["normalized_sha256"]}
        fam = entry["independent_source_family"]["family_id"]
        incumbent_families.add(fam)
        uri = entry["snapshot"].get("snapshot_uri", "")
        if uri.startswith("file:"):
            p = ROOT / uri[5:]
            if p.is_file():
                try:
                    inc = p.read_bytes().decode("utf-8", errors="strict").encode("utf-8")
                except UnicodeDecodeError:
                    continue
                score = jaccard(candidate_agg, shingles(inc, shingle_size))
                if score >= near_threshold:
                    cross_near.append({"family": fam, "source_id": entry["source_id"], "jaccard": round(score, 6)})
    candidate_hashes = set(raw_hashes + norm_hashes + [raw_bundle_sha, norm_bundle_sha])
    exact_registry_collisions = sorted(candidate_hashes & incumbent_hashes)
    reserved = json.loads((ROOT / cfg["dedup"]["reserved_fingerprints"]).read_text(encoding="utf-8"))
    reserved_collisions = sorted(candidate_hashes & all_sha256_strings(reserved))

    license_reports = []
    for lic in cfg["license"]["files"]:
        url = f"https://raw.githubusercontent.com/rust-lang-ua/rustbook_ukrainian/{commit}/{lic['path']}"
        raw = fetch(url, 64 * 1024)
        blob = git_blob_sha1(raw)
        if blob != lic["git_blob_sha1"]:
            raise RuntimeError(f"license blob mismatch: {lic['path']}")
        license_reports.append({"path": lic["path"], "git_blob_sha1": blob, "sha256": sha256_bytes(raw), "bytes": len(raw)})

    gates = {
        "exact_revision_and_git_blobs": True,
        "bounded_acquisition": raw_total <= cfg["selection"]["max_total_raw_bytes"],
        "license_exact": len(license_reports) == 2,
        "training_rights": cfg["purpose_decisions"]["model_training"] == "ALLOWED",
        "redistribution_rights": cfg["purpose_decisions"]["redistribution"] == "ALLOWED_WITH_NOTICES",
        "evaluation_firewall": cfg["purpose_decisions"]["evaluation"] == "NOT_SEPARATELY_ADMITTED",
        "language_uk": language_pass,
        "quality": quality_pass,
        "privacy": privacy_pass,
        "intra_exact_dedup": intra_exact,
        "intra_near_dedup": not intra_near,
        "registry_exact_dedup": not exact_registry_collisions,
        "registry_near_dedup": not cross_near,
        "reserved_exact_dedup": not reserved_collisions,
        "family_independent": src["source_family"] not in incumbent_families,
        "eval_family_not_reserved": src["source_family"] not in set(cfg["dedup"]["known_evaluation_families"]),
    }
    all_pass = all(gates.values())
    core = {
        "schema_version": "12-6.next100-030-oer-source-report.v1",
        "worker_id": cfg["worker_id"],
        "source": src,
        "license": {"declared_surface": cfg["license"]["declared_surface"], "operational_compliance": cfg["license"]["operational_compliance"], "files": license_reports},
        "purpose_decisions": cfg["purpose_decisions"],
        "selection": {"file_count": len(reports), "raw_bytes": raw_total, "raw_bundle_sha256": raw_bundle_sha, "normalized_bytes": norm_total, "normalized_bundle_sha256": norm_bundle_sha, "normalization": cfg["normalization"], "files": reports},
        "aggregate_language": aggregate_language,
        "dedup": {"intra_near_pairs": intra_near, "registry_exact_collisions": exact_registry_collisions, "registry_near_collisions": cross_near, "reserved_exact_collisions": reserved_collisions},
        "gates": gates,
        "training_executed": False,
        "local_free_only": True,
    }
    manifest_sha = sha256_bytes(canonical_json_bytes(core))
    core["manifest_sha256"] = manifest_sha
    expected = cfg["expected_lock"]
    lock_matches = True
    if cfg["mode"] == "LOCKED":
        actual = {
            "file_count": len(reports),
            "raw_bytes": raw_total,
            "raw_bundle_sha256": raw_bundle_sha,
            "normalized_bytes": norm_total,
            "normalized_bundle_sha256": norm_bundle_sha,
            "manifest_sha256": manifest_sha,
            "license_sha256": {r["path"]: r["sha256"] for r in license_reports},
        }
        lock_matches = actual == expected
        if not lock_matches:
            raise RuntimeError(f"lock mismatch: actual={actual!r}")
    verdict = "ADMIT" if cfg["mode"] == "LOCKED" and all_pass and lock_matches else ("PROBE_PASS" if all_pass else "REJECT")
    core["mode"] = cfg["mode"]
    core["lock_matches"] = lock_matches
    core["terminal"] = cfg["mode"] == "LOCKED"
    core["verdict"] = verdict
    (out / "source_report.json").write_bytes(canonical_json_bytes(core))
    (out / "manifest.json").write_bytes(canonical_json_bytes({"manifest_sha256": manifest_sha, "raw_bundle_sha256": raw_bundle_sha, "normalized_bundle_sha256": norm_bundle_sha, "verdict": verdict}))
    if cfg["mode"] == "LOCKED" and verdict != "ADMIT":
        raise RuntimeError(f"terminal verdict is not ADMIT: {verdict}")


if __name__ == "__main__":
    main()
