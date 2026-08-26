#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "in",
    "is", "it", "its", "not", "of", "on", "or", "that", "the", "their",
    "there", "they", "this", "to", "was", "were", "which", "with", "you",
}
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
START_RE = re.compile(r"^\*\*\* START OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")
END_RE = re.compile(r"^\*\*\* END OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")


class GateFailure(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha256(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(raw)


def git_blob_sha1(data: bytes) -> str:
    framed = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    return hashlib.sha1(framed).hexdigest()


def fetch_bytes(url: str, attempts: int = 3) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "12-6-ai-NEXT100-033-bounded-source-qualification/1.0"},
    )
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(2)
    raise GateFailure(f"acquisition failed for {url}: {last}")


def normalize_pg_body(raw: bytes, encoding: str) -> tuple[bytes, dict[str, Any]]:
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise GateFailure(f"decode failure under preregistered encoding {encoding}: {exc}") from exc

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if START_RE.match(line.strip())]
    ends = [i for i, line in enumerate(lines) if END_RE.match(line.strip())]
    if len(starts) != 1:
        raise GateFailure(f"expected exactly one Gutenberg START marker, got {len(starts)}")
    if len(ends) != 1:
        raise GateFailure(f"expected exactly one Gutenberg END marker, got {len(ends)}")
    if ends[0] <= starts[0]:
        raise GateFailure("Gutenberg END marker does not follow START marker")

    body_lines = lines[starts[0] + 1 : ends[0]]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    body = "\n".join(body_lines)
    if body.startswith("\ufeff"):
        body = body[1:]
    body = unicodedata.normalize("NFC", body)
    normalized = (body + "\n").encode("utf-8")

    return normalized, {
        "start_marker_line_1based": starts[0] + 1,
        "end_marker_line_1based": ends[0] + 1,
        "removed_prefix_lines_including_start": starts[0] + 1,
        "removed_suffix_lines_including_end": len(lines) - ends[0],
        "normalizer_id": "NEXT100_033_PG_BODY_NFC_LF_V1",
    }


def quality_metrics(text: str) -> dict[str, Any]:
    letters = [ch for ch in text if ch.isalpha()]
    ascii_letters = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in letters)
    words = [w.lower() for w in WORD_RE.findall(text)]
    stop_hits = sum(w in STOPWORDS for w in words)
    return {
        "characters": len(text),
        "letters": len(letters),
        "words_ascii_tokenized": len(words),
        "ascii_letter_fraction_of_letters": (ascii_letters / len(letters)) if letters else 0.0,
        "english_stopword_fraction_of_words": (stop_hits / len(words)) if words else 0.0,
        "replacement_chars": text.count("\ufffd"),
        "nul_chars": text.count("\x00"),
        "line_count": text.count("\n"),
    }


def shingle_set(text: str, width: int) -> set[bytes]:
    words = [w.lower() for w in WORD_RE.findall(text)]
    if len(words) < width:
        return set()
    out: set[bytes] = set()
    for i in range(len(words) - width + 1):
        piece = "\x1f".join(words[i : i + width]).encode("ascii")
        out.add(hashlib.blake2b(piece, digest_size=8).digest())
    return out


def jaccard(a: set[bytes], b: set[bytes]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return (len(a & b) / union) if union else 0.0


def collect_named_values(obj: Any, key_name: str) -> list[str]:
    values: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == key_name and isinstance(value, str):
                values.append(value)
            values.extend(collect_named_values(value, key_name))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(collect_named_values(value, key_name))
    return values


def scan_eval_reserved_paths(repo_root: Path, ebook_ids: list[int]) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    path_terms = ("eval", "reserved", "selection", "validation")
    for top in ("data", "configs", "reports", "docs"):
        root = repo_root / top
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(repo_root).as_posix()
            low = rel.lower()
            if "next100_033" in low:
                continue
            if not any(term in low for term in path_terms):
                continue
            if path.suffix.lower() not in {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}:
                continue
            try:
                text = path.read_text("utf-8", errors="ignore")
            except OSError:
                continue
            for ebook_id in ebook_ids:
                if re.search(rf"(?<!\d){ebook_id}(?!\d)", text):
                    matches.append({"ebook_id": ebook_id, "path": rel})
    return {"matches": matches, "status": "PASS" if not matches else "FAIL"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    repo_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out)
    raw_dir = out_root / "raw"
    norm_dir = out_root / "normalized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    norm_dir.mkdir(parents=True, exist_ok=True)

    config_raw = config_path.read_bytes()
    config = json.loads(config_raw)
    config_sha = sha256_bytes(config_raw)

    rights_rel = config["rights_evidence"]["path"]
    rights_path = repo_root / rights_rel
    rights_raw = rights_path.read_bytes()
    rights_sha = sha256_bytes(rights_raw)
    if rights_sha != config["rights_evidence"]["sha256"]:
        raise GateFailure(
            f"rights evidence drift: expected {config['rights_evidence']['sha256']} got {rights_sha}"
        )

    registry_path = repo_root / config["base_registry"]["registry_path"]
    live_registry = json.loads(registry_path.read_text("utf-8"))
    incumbent_normalized_hashes = set(collect_named_values(live_registry, "normalized_sha256"))

    eval_gate = scan_eval_reserved_paths(
        repo_root, [int(x) for x in config["evaluation_reservation_gate"]["candidate_ebook_ids"]]
    )
    if eval_gate["status"] != "PASS":
        raise GateFailure(f"evaluation-reserved exact ebook id match: {eval_gate['matches']}")

    works: list[dict[str, Any]] = []
    shingle_sets: dict[str, set[bytes]] = {}
    normalized_hashes: set[str] = set()

    for source in config["sources"]:
        repo = source["transport_repo"]
        commit = source["transport_commit"]
        path = source["transport_path"]
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        raw = fetch_bytes(url)

        if len(raw) != int(source["transport_bytes"]):
            raise GateFailure(
                f"{source['source_id']} raw size mismatch: expected {source['transport_bytes']} got {len(raw)}"
            )
        blob_sha = git_blob_sha1(raw)
        if blob_sha != source["transport_git_blob_sha1"]:
            raise GateFailure(
                f"{source['source_id']} Git blob mismatch: expected {source['transport_git_blob_sha1']} got {blob_sha}"
            )

        normalized, extraction = normalize_pg_body(raw, source["encoding"])
        text = normalized.decode("utf-8")
        q = quality_metrics(text)
        qg = config["quality_gate"]
        if len(normalized) < int(qg["min_normalized_utf8_bytes_per_work"]):
            raise GateFailure(f"{source['source_id']} normalized text too small: {len(normalized)}")
        if q["ascii_letter_fraction_of_letters"] < float(qg["min_ascii_letter_fraction_of_letters"]):
            raise GateFailure(f"{source['source_id']} failed ASCII-letter language screen: {q}")
        if q["english_stopword_fraction_of_words"] < float(qg["min_english_stopword_fraction_of_words"]):
            raise GateFailure(f"{source['source_id']} failed English stopword screen: {q}")
        if q["replacement_chars"] > int(qg["max_replacement_chars"]):
            raise GateFailure(f"{source['source_id']} contains replacement characters")
        if q["nul_chars"] > int(qg["max_nul_chars"]):
            raise GateFailure(f"{source['source_id']} contains NUL characters")

        normalized_sha = sha256_bytes(normalized)
        if normalized_sha in normalized_hashes:
            raise GateFailure(f"exact normalized duplicate inside candidate: {source['source_id']}")
        if normalized_sha in incumbent_normalized_hashes:
            raise GateFailure(f"exact normalized duplicate against live registry: {source['source_id']}")
        normalized_hashes.add(normalized_sha)

        safe_id = source["source_id"].replace("/", "_")
        raw_path = raw_dir / f"{safe_id}.txt"
        norm_path = norm_dir / f"{safe_id}.txt"
        raw_path.write_bytes(raw)
        norm_path.write_bytes(normalized)

        width = int(config["dedup_gate"]["token_shingle_width"])
        shingle_sets[source["source_id"]] = shingle_set(text, width)

        runtime_source = {
            "source_id": source["source_id"],
            "ebook_id": source["ebook_id"],
            "title": source["title"],
            "author": source["author"],
            "family_id": config["source_family"]["family_id"],
            "project_gutenberg_landing_url": source["project_gutenberg_landing_url"],
            "transport_repo": repo,
            "transport_commit": commit,
            "transport_path": path,
            "transport_git_blob_sha1": blob_sha,
            "raw_bytes": len(raw),
            "raw_sha256": sha256_bytes(raw),
            "normalized_utf8_bytes": len(normalized),
            "normalized_sha256": normalized_sha,
            "quality": q,
            "extraction": extraction,
        }
        runtime_source["source_manifest_sha256"] = canonical_json_sha256(runtime_source)
        works.append(runtime_source)

    pairwise: list[dict[str, Any]] = []
    threshold = float(config["dedup_gate"]["max_pairwise_jaccard"])
    for i in range(len(works)):
        for j in range(i + 1, len(works)):
            a = works[i]["source_id"]
            b = works[j]["source_id"]
            score = jaccard(shingle_sets[a], shingle_sets[b])
            pairwise.append({"source_a": a, "source_b": b, "jaccard": score})
            if score > threshold:
                raise GateFailure(f"near-duplicate gate failed for {a} vs {b}: {score} > {threshold}")

    corpus_identity_material = {
        "normalizer_id": config["normalization"]["normalizer_id"],
        "family_identity_sha256": config["source_family"]["family_identity_sha256"],
        "sources": [
            {
                "source_id": work["source_id"],
                "raw_sha256": work["raw_sha256"],
                "normalized_sha256": work["normalized_sha256"],
            }
            for work in works
        ],
    }
    total_norm = sum(int(w["normalized_utf8_bytes"]) for w in works)
    report = {
        "schema_version": "12-6.next100-033.gutenberg-en-terminal-evidence.v1",
        "worker_id": config["worker_id"],
        "terminal_decision": "ADMIT",
        "local_free_only": True,
        "config_sha256": config_sha,
        "rights_evidence_sha256": rights_sha,
        "family": {
            "family_id": config["source_family"]["family_id"],
            "family_identity_sha256": config["source_family"]["family_identity_sha256"],
            "independent_family_credit": 1,
            "transport_is_not_independent_family": True,
        },
        "purpose": {
            "model_training": "ALLOWED_FOR_EXACT_ADMITTED_NORMALIZED_BODIES",
            "redistribution": config["authority_boundary"]["redistribution"],
            "evaluation": "NOT_AUTHORIZED",
        },
        "evaluation_reservation_gate": eval_gate,
        "dedup": {
            "pairwise_8token_jaccard": pairwise,
            "exact_normalized_unique_within_candidate": True,
            "exact_normalized_no_match_live_registry": True,
        },
        "quality_gate": "PASS",
        "normalization_gate": "PASS",
        "exact_transport_identity_gate": "PASS",
        "works": works,
        "normalized_utf8_bytes_total": total_norm,
        "corpus_identity_sha256": canonical_json_sha256(corpus_identity_material),
        "limitations": [
            "One Project Gutenberg family only; three titles are not three independent families.",
            "No claim that Project Gutenberg is broadly representative of English.",
            "No universal worldwide public-domain claim.",
            "No evaluation authorization and no universal semantic benchmark-clean claim.",
            "GITenberg is an immutable transport mirror of Project Gutenberg lineage, not a separate provenance family.",
        ],
    }

    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", "utf-8")
    print("NEXT100_033_REPORT=" + json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as exc:
        print(f"NEXT100_033_GATE_FAILURE={exc}", file=sys.stderr)
        raise SystemExit(2)
